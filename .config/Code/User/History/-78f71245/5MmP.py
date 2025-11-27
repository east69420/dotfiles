"""Tiered feature engineering pipeline with heavy cache and live state support.

This module reorganizes the legacy feature engineering code into three distinct
feature categories (stateless, rolling, and complex/grouped) and exposes a
three-tier caching strategy:

* Tier 1 (Heavy cache): expensive grouped aggregations persisted to disk.
* Tier 2 (Live state cache): in-memory rolling window state for incremental updates.
* Tier 3 (On-demand row projection): combines cached state with the latest bar.

Quick start (sklearn-compatible fit/transform pattern):

1. **Training Phase**: Fit on historical data to build heavy cache and learn patterns::

         from featureEngineer import FeatureEngineer

         # Initialize with production settings
         fe = FeatureEngineer()
         
         # Fit on training data (builds heavy cache, ~1-2s for 1000 rows)
         fe.fit(historical_ohlcv_data)
         
         # Transform training data (reuses cache, fast)
         training_features = fe.transform(historical_ohlcv_data)

2. **Real-time Inference**: Transform new data using fitted pipeline::

         # Transform single new row (target: <100ms latency)
         new_features = fe.transform(latest_market_data)
         
         # Transform updated/corrected data (same performance)
         corrected_features = fe.transform(corrected_market_data)

3. **Performance**: Typical latency ~50-100ms per row for real-time inference.

Advanced usage for streaming:

4. For high-frequency streaming updates, use the live state system::

       live_state = fe.initialize_live_state(price_df)
       # Append the next bar
       next_features = fe.ingest_live_row(next_bar)
       # Replace the latest bar with revised data
       revised_features = fe.ingest_live_row(revised_bar)

The design targets low-fragmentation DataFrame operations while remaining fully
compatible with the existing offline batch workflow.
"""

from __future__ import annotations

import pickle
import time
import warnings
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta
from sklearn.base import BaseEstimator, TransformerMixin

from ML_general_tools import stat_tools
from data_pipeline import is_europe_dst


HEAVY_CACHE_VERSION = 1
DEFAULT_PREV_DAYTYPE_WINDOW = 90
DEFAULT_EMPIRICAL_THRESHOLDS = (0.0001, 0.0005, 0.001)
DEFAULT_EMPIRICAL_WINDOW = 90
DEFAULT_EMPIRICAL_MIN_COUNT = 20
HOURS_PER_YEAR = 24 * 365
# Standard normal quantiles for 90%, 50%, and 10% (fixed z-scores)
Z_SCORE_90 = 1.2815515655446004
Z_SCORE_75 = 0.6744897501960817
Z_SCORE_25 = -Z_SCORE_75
Z_SCORE_10 = -Z_SCORE_90

@dataclass
class HeavyFeaturePayload:
    """Serialized representation of complex/grouped features."""

    prev_cycle_lookup: pd.DataFrame
    prev_cycle_stats_lookup: pd.DataFrame
    empirical_lookup: pd.DataFrame
    metadata: Dict[str, Any]


class HeavyFeatureCache:
    """Persist heavy feature payloads to disk with version awareness."""

    def __init__(self, cache_dir: Optional[Path]) -> None:
        self.cache_dir = Path(cache_dir or Path("cache") / "heavy_features")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.payload: Optional[HeavyFeaturePayload] = None

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / f"heavy_features_v{HEAVY_CACHE_VERSION}.pkl"

    def load(self) -> bool:
        cache_path = self.cache_path
        if not cache_path.exists():
            return False
        try:
            with cache_path.open("rb") as fh:
                payload = pickle.load(fh)
        except Exception:
            return False
        if not isinstance(payload, HeavyFeaturePayload):
            return False
        metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
        if metadata.get("version") != HEAVY_CACHE_VERSION:
            return False
        self.payload = payload
        return True

    def save(self, payload: HeavyFeaturePayload) -> None:
        cache_path = self.cache_path
        with cache_path.open("wb") as fh:
            pickle.dump(payload, fh)
        self.payload = payload


@dataclass
class LiveStateCache:
    """Hold incremental feature state for real-time updates."""

    features_df: pd.DataFrame
    price_history: pd.DataFrame
    rolling_states: Dict[str, Any]
    ema_states: Dict[str, Any]
    metadata: Dict[str, Any]

    def append(self, features_row: pd.Series, price_row: pd.Series) -> None:
        self.features_df = pd.concat([self.features_df, features_row.to_frame().T])
        self.price_history = pd.concat([self.price_history, price_row.to_frame().T])


@dataclass
class RollingWindowState:
    """Maintain running statistics for a fixed-length window."""

    window: int
    values: deque = field(default_factory=deque)
    sum_: float = 0.0
    sum_sq: float = 0.0

    def append(self, value: float) -> None:
        self.values.append(value)
        self.sum_ += value
        self.sum_sq += value * value
        if len(self.values) > self.window:
            old = self.values.popleft()
            self.sum_ -= old
            self.sum_sq -= old * old

    def mean(self, min_periods: int = 1) -> float:
        n = len(self.values)
        if n < max(1, min_periods):
            return float("nan")
        return self.sum_ / n

    def std(self, min_periods: int = 2) -> float:
        n = len(self.values)
        if n < max(2, min_periods):
            return float("nan")
        mean = self.sum_ / n
        variance = (self.sum_sq - n * mean * mean) / max(1, n - 1)
        variance = max(variance, 0.0)
        return float(np.sqrt(variance))


class StatelessFeatureBlock:
    """Generate row-wise features that only require the most recent bar."""

    def __init__(self, expiration_hour: int, include_dst: bool) -> None:
        self.expiration_hour = expiration_hour
        self.include_dst = include_dst

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of strictly causal, per-row features."""

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Stateless features require a DatetimeIndex")
        if df.empty:
            return pd.DataFrame(index=df.index)

        idx = df.index
        current_time = idx + pd.Timedelta(hours=1)
        expiry_today = current_time.normalize() + pd.Timedelta(hours=self.expiration_hour)
        expiry_tomorrow = expiry_today + pd.Timedelta(days=1)
        time_diff = np.where(current_time < expiry_today, expiry_today - current_time, expiry_tomorrow - current_time)
        time_to_exp = pd.Series(
            [float(delta / pd.Timedelta(hours=1)) for delta in time_diff],
            index=idx,
            dtype="float64",
        )
        time_to_exp[np.isclose(time_to_exp, 0.0, atol=1e-9)] = 24.0

        stateless = pd.DataFrame(index=idx)
        stateless["time_to_exp1_hr"] = time_to_exp
        stateless["time_elapsed"] = 24.0 - time_to_exp
        stateless["hour"] = current_time.hour
        stateless["day_of_week"] = current_time.dayofweek
        stateless["is_weekend"] = stateless["day_of_week"].isin([5, 6]).astype(int)

        hour_of_week = stateless["day_of_week"] * 24 + stateless["hour"]
        stateless["hour_of_week"] = hour_of_week
        stateless["hour_of_week_sin"] = np.sin(2 * np.pi * hour_of_week / 168)
        stateless["hour_of_week_cos"] = np.cos(2 * np.pi * hour_of_week / 168)
        stateless["hours_since_week_start"] = hour_of_week.astype(float)

        stateless["day_type_num"] = np.select(
            [stateless["day_of_week"] < 5, stateless["day_of_week"] == 5, stateless["day_of_week"] == 6],
            [0, 1, 2],
            default=-1,
        )

        stateless["prev_close"] = df["c"].shift(1)
        if "volCcy" in df.columns:
            stateless["volCcy_prev"] = df["volCcy"].shift(1).round()
        else:
            stateless["volCcy_prev"] = np.nan

        cycle_start_mask = ((idx + pd.Timedelta(hours=1)).hour == self.expiration_hour)
        cycle_start_flags = cycle_start_mask.astype(int)
        cycle_ids = cycle_start_flags.cumsum()
        stateless["cycle_id"] = cycle_ids

        cycle_anchor_close = df["c"].where(cycle_start_mask)
        stateless["window_prev_close"] = cycle_anchor_close.ffill()
        stateless.loc[stateless["window_prev_close"].isna(), "window_prev_close"] = stateless.loc[
            stateless["window_prev_close"].isna(), "prev_close"
        ]
        stateless["_cycle_start_ts"] = (idx + pd.Timedelta(hours=1)).to_series().where(cycle_start_mask).ffill().values

        close_shift_1 = df["c"].shift(1)
        close_shift_2 = df["c"].shift(2)
        safe_close_shift_2 = close_shift_2.replace(0, np.nan)

        stateless["returns_1h"] = ((close_shift_1 / safe_close_shift_2) - 1).fillna(0)

        horizons = [2, 3, 4, 5, 6, 12, 24, 72]
        for horizon in horizons:
            denominator = df["c"].shift(horizon + 1).replace(0, np.nan)
            ratio = close_shift_1 / denominator
            stateless[f"returns_{horizon}h"] = (ratio - 1).fillna(0)
            with np.errstate(divide="ignore", invalid="ignore"):
                stateless[f"logret_{horizon}h"] = np.log(ratio.replace(0, np.nan))

        for window_hours, label in [(24 * 7 + 1, "returns_1wk"), (24 * 30 + 1, "returns_1M")]:
            denominator = df["c"].shift(window_hours).replace(0, np.nan)
            ratio = close_shift_1 / denominator
            stateless[label] = (ratio - 1).fillna(0)

        if {"h", "l"}.issubset(df.columns):
            high_shift_1 = df["h"].shift(1)
            low_shift_1 = df["l"].shift(1)
            safe_low = low_shift_1.replace(0, np.nan)

            stateless["ret_h_pc"] = (high_shift_1 / safe_close_shift_2) - 1
            stateless["ret_l_pc"] = (low_shift_1 / safe_close_shift_2) - 1
            stateless["ret_c_pc"] = (close_shift_1 / safe_close_shift_2) - 1
            stateless["ret_h_l"] = (high_shift_1 / safe_low) - 1

            for window in [2, 3, 6, 12, 24]:
                rolling_high = high_shift_1.rolling(window=window, min_periods=1).max()
                rolling_low = low_shift_1.rolling(window=window, min_periods=1).min()
                stateless[f"range_{window}h"] = rolling_high - rolling_low
                stateless[f"range_pc_{window}h"] = (
                    (rolling_high - rolling_low) / stateless["prev_close"].shift(1).replace(0, np.nan)
                )

            trading_range = high_shift_1 - low_shift_1
            stateless["range"] = trading_range
            stateless["range_pc"] = trading_range / stateless["prev_close"].shift(1).replace(0, np.nan)
            stateless["close_to_high"] = (high_shift_1 - close_shift_1) / (high_shift_1 - low_shift_1).replace(0, np.nan)
            stateless["close_to_low"] = (close_shift_1 - low_shift_1) / (high_shift_1 - low_shift_1).replace(0, np.nan)

            same_high_low = high_shift_1 == low_shift_1
            stateless.loc[same_high_low, ["close_to_high", "close_to_low"]] = 0.5

            stateless["close_pos_in_bar"] = (
                (close_shift_1 - low_shift_1) / (high_shift_1 - low_shift_1).replace(0, np.nan)
            ).clip(0, 1)
            stateless.loc[same_high_low, "close_pos_in_bar"] = 0.5

            prev_close_shift_1 = stateless["prev_close"].shift(1).replace(0, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                stateless["logret_h_pc"] = np.log(high_shift_1 / prev_close_shift_1)
                stateless["logret_l_pc"] = np.log(low_shift_1 / prev_close_shift_1)
                stateless["logret_c_pc"] = np.log(close_shift_1 / prev_close_shift_1)

        if self.include_dst:
            stateless["is_dst"] = is_europe_dst(idx).astype(int)

        return stateless.replace([np.inf, -np.inf], np.nan)


class RollingFeatureBlock:
    """Produce fixed-window rolling statistics without duplicating logic elsewhere."""

    def __init__(self,
                 vol_window_sizes: Iterable[int],
                 vlm_window_sizes: Iterable[int],
                 vol_types_to_calc: Iterable[str],
                 vol_trading_periods: int) -> None:
        self.vol_window_sizes = sorted(set(vol_window_sizes))
        self.vlm_window_sizes = sorted(set(vlm_window_sizes))
        self.vol_types_to_calc = [v.lower() for v in vol_types_to_calc]
        self.vol_trading_periods = vol_trading_periods
        self.stats = stat_tools()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return combined trend/volatility/liquidity/relative-position features."""

        features = []

        if {"c", "h", "l"}.issubset(df.columns):
            features.append(self._trend_features(df))
            features.append(self._relative_position_features(df))
        if "volCcy_prev" in df.columns or "volCcy" in df.columns:
            features.append(self._liquidity_features(df))
        features.append(self._volatility_features(df))

        valid = [f for f in features if f is not None and not f.empty]
        if not valid:
            return pd.DataFrame(index=df.index)
        return pd.concat(valid, axis=1)

    # --- Private helpers -------------------------------------------------

    def _trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        trend_df = pd.DataFrame(index=df.index)
        closes = df["c"].shift(1)
        
        # Calculate trend strength features (backward z-scores)
        # How big was the move over the last N hours relative to volatility?
        safe_close = closes.replace(0, np.nan)
        
        for w in self.vol_window_sizes:
            if w <= 1:
                continue
            min_p = max(1, w // 2)

            close_w_ago = df["c"].shift(w + 1).replace(0, np.nan)
            
            # Standard momentum (returns)
            momentum = ((closes / close_w_ago) - 1).fillna(0)
            trend_df[f"momentum_{w}h"] = momentum
            trend_df[f"momentum_signed_sqrt_{w}h"] = np.sign(momentum) * np.sqrt(np.abs(momentum))
            
            # Trend strength: backward return / volatility (z-score of move)
            # This normalizes the move by historical volatility
            backward_return = np.log(safe_close / close_w_ago)
            
            # Use realized volatility as the denominator
            # Get log returns for volatility calculation
            prev_close_lag = safe_close.shift(1).replace(0, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_ret = np.log(safe_close / prev_close_lag)
            log_ret = log_ret.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Rolling std (annualized volatility)
            box_std = log_ret.rolling(w, min_periods=min_p).std()
            
            # Per-period std for the window length (de-annualize for the period)
            box_std_period = box_std * np.sqrt(w)  # Adjust for window length
            
            # Trend strength = realized move / expected move
            # This gives you "how many sigmas was this move?"
            trend_strength = backward_return / box_std_period.replace(0, np.nan)
            trend_df[f"trend_strength_{w}h"] = trend_strength.fillna(0)

        try:
            macd = ta.trend.MACD(df["c"].shift(1), window_slow=26, window_fast=12, window_sign=9, fillna=True)
            trend_df["macd"] = macd.macd()
            trend_df["macd_signal"] = macd.macd_signal()
            trend_df["macd_hist"] = macd.macd_diff()

            adx = ta.trend.ADXIndicator(df["h"].shift(1), df["l"].shift(1), df["c"].shift(1), window=14, fillna=True)
            trend_df["adx"] = adx.adx()
            trend_df["adx_pos"] = adx.adx_pos()
            trend_df["adx_neg"] = adx.adx_neg()
        except Exception:
            for col in ["macd", "macd_signal", "macd_hist", "adx", "adx_pos", "adx_neg"]:
                trend_df[col] = np.nan

        return trend_df

    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        vol_df = pd.DataFrame(index=df.index)
        prev_close = df["prev_close"].replace(0, np.nan)
        prev_close_lag = prev_close.shift(1).replace(0, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret = np.log(prev_close / prev_close_lag)
        log_ret = log_ret.replace([np.inf, -np.inf], np.nan).fillna(0)

        for w in self.vol_window_sizes:
            if "raw" in self.vol_types_to_calc:
                min_p = max(3, w // 4)
                vol = log_ret.rolling(w, min_periods=min_p).std() * np.sqrt(self.vol_trading_periods)
                vol_df[f"vol_raw_{w}h"] = vol.fillna(0)
            if "gkyz" in self.vol_types_to_calc:
                try:
                    gkyz = self.stats.get_GKYZ(df, w, self.vol_trading_periods)
                    vol_df[f"vol_gkyz_{w}h"] = gkyz.reindex(df.index).ffill()
                except Exception:
                    vol_df[f"vol_gkyz_{w}h"] = pd.Series(0, index=df.index)
            if "atr" in self.vol_types_to_calc:
                vol_df[f"atr_{w}h"] = self._calculate_atr(df, w)
            if "parkinson" in self.vol_types_to_calc:
                # Parkinson volatility: range-based estimator using only H/L
                # Formula: σ = √[(1/(4*ln(2))) * mean((ln(H/L))²)] * √(trading_periods)
                # More efficient than close-to-close, captures intraday volatility
                high_shifted = df["h"].shift(1)
                low_shifted = df["l"].shift(1)
                safe_low = low_shifted.replace(0, np.nan).clip(lower=1e-9)
                
                with np.errstate(divide="ignore", invalid="ignore"):
                    hl_ratio = high_shifted / safe_low
                    log_hl = np.log(hl_ratio.replace([0, np.inf, -np.inf], np.nan))
                    log_hl_sq = log_hl ** 2
                
                min_p = max(3, w // 4)
                # Parkinson coefficient: 1/(4*ln(2)) ≈ 0.3607
                parkinson_coef = 1.0 / (4.0 * np.log(2.0))
                mean_log_hl_sq = log_hl_sq.rolling(w, min_periods=min_p).mean()
                vol_parkinson = np.sqrt(parkinson_coef * mean_log_hl_sq) * np.sqrt(self.vol_trading_periods)
                vol_df[f"vol_parkinson_{w}h"] = vol_parkinson.fillna(0)
            if "rogers_satchell" in self.vol_types_to_calc:
                # Rogers-Satchell volatility: drift-independent OHLC estimator
                # Formula: σ = √[mean(ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O))] * √(trading_periods)
                # Superior to Parkinson as it incorporates open/close (drift info)
                high_shifted = df["h"].shift(1)
                low_shifted = df["l"].shift(1)
                close_shifted = df["c"].shift(1)
                open_shifted = df["o"].shift(1) if "o" in df.columns else close_shifted
                
                safe_close = close_shifted.replace(0, np.nan).clip(lower=1e-9)
                safe_open = open_shifted.replace(0, np.nan).clip(lower=1e-9)
                
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_hc = np.log(high_shifted / safe_close)
                    log_ho = np.log(high_shifted / safe_open)
                    log_lc = np.log(low_shifted / safe_close)
                    log_lo = np.log(low_shifted / safe_open)
                    
                    # Rogers-Satchell formula
                    rs_component = (log_hc * log_ho) + (log_lc * log_lo)
                    rs_component = rs_component.replace([np.inf, -np.inf], np.nan)
                
                min_p = max(3, w // 4)
                mean_rs = rs_component.rolling(w, min_periods=min_p).mean()
                # Ensure non-negative before sqrt (numerical stability)
                vol_rs = np.sqrt(mean_rs.clip(lower=0)) * np.sqrt(self.vol_trading_periods)
                vol_df[f"vol_rs_{w}h"] = vol_rs.fillna(0)
            if "vol_zscore" in self.vol_types_to_calc:
                # Z-score of volatility: (current_vol - rolling_mean) / rolling_std
                # Use raw volatility as base, with longer lookback for mean/std calculation
                base_vol_col = f"vol_raw_{w}h"
                if base_vol_col in vol_df.columns:
                    base_vol = vol_df[base_vol_col]
                else:
                    # Calculate raw vol if not already computed
                    min_p = max(3, w // 4)
                    base_vol = log_ret.rolling(w, min_periods=min_p).std() * np.sqrt(self.vol_trading_periods)
                
                # Use 2x window for mean/std to capture volatility regime
                zscore_window = min(w * 2, max(self.vol_window_sizes))
                min_p_zscore = max(w, zscore_window // 2)
                vol_ma = base_vol.rolling(zscore_window, min_periods=min_p_zscore).mean()
                vol_std = base_vol.rolling(zscore_window, min_periods=min_p_zscore).std().replace(0, 1e-9)
                vol_df[f"vol_zscore_{w}h"] = ((base_vol - vol_ma) / vol_std).fillna(0)
            if "log_vol" in self.vol_types_to_calc:
                # Log-volatility: useful for multiplicative volatility models
                # Use raw volatility as base
                base_vol_col = f"vol_raw_{w}h"
                if base_vol_col in vol_df.columns:
                    base_vol = vol_df[base_vol_col]
                else:
                    # Calculate raw vol if not already computed
                    min_p = max(3, w // 4)
                    base_vol = log_ret.rolling(w, min_periods=min_p).std() * np.sqrt(self.vol_trading_periods)
                
                # Clip to avoid log(0) and handle very small volatilities
                vol_clipped = base_vol.clip(lower=1e-6)
                vol_df[f"log_vol_{w}h"] = np.log(vol_clipped).fillna(0)

        if "skew" in self.vol_types_to_calc:
            ewma = log_ret.ewm(span=24, min_periods=3).mean()
            ewmstd = log_ret.ewm(span=24, min_periods=3).std()
            centered = log_ret - ewma
            skew_num = (centered ** 3).ewm(span=24, min_periods=3).mean()
            vol_df["returns_skew_24h"] = (skew_num / (ewmstd ** 3 + 1e-9)).fillna(0)

        if "kurtosis" in self.vol_types_to_calc:
            ewma = log_ret.ewm(span=24, min_periods=4).mean()
            ewmstd = log_ret.ewm(span=24, min_periods=4).std()
            centered = log_ret - ewma
            kurt_num = (centered ** 4).ewm(span=24, min_periods=4).mean()
            vol_df["returns_kurtosis_24h"] = (kurt_num / (ewmstd ** 4 + 1e-9) - 3).fillna(0)

        def _lookup(col: str) -> Optional[pd.Series]:
            if col in vol_df.columns:
                return vol_df[col]
            if col in df.columns:
                return df[col]
            return None

        ## lets use raw vol as thats what blackscholes does and they actually got a noble prize somehow
        vol_24 = _lookup("vol_raw_24h")
        vol_144 = _lookup("vol_raw_144h")
        vol_288 = _lookup("vol_raw_288h")

        epsilon = 1e-10
        vol_df["vol_ratio_24h_144h"] = vol_24 / (vol_144 + epsilon)
        vol_df["vol_ratio_24h_288h"] = vol_24 / (vol_288 + epsilon)


        ## essentially using pas 288 as more stable hour to predict blackscholes.
        sigma_unit = vol_288 / np.sqrt(float(self.vol_trading_periods))
        sigma_24h = sigma_unit * np.sqrt(24.0)

        exp_ret_p90_24h = sigma_24h * Z_SCORE_90
        exp_ret_p75_24h = sigma_24h * Z_SCORE_75
        exp_ret_p25_24h = sigma_24h * Z_SCORE_25
        exp_ret_p10_24h = sigma_24h * Z_SCORE_10

        ## expected 24 hour moves based on blackschoels and 
        vol_df["exp_ret_p90_24h"] = exp_ret_p90_24h
        vol_df["exp_ret_p75_24h"] = exp_ret_p75_24h
        vol_df["exp_ret_p25_24h"] = exp_ret_p25_24h
        vol_df["exp_ret_p10_24h"] = exp_ret_p10_24h

        vol_df["exp_band_width_24h"] = exp_ret_p90_24h - exp_ret_p10_24h
        vol_df["exp_logmove_p90_24h"] = np.log1p(exp_ret_p90_24h.clip(lower=-0.999999))
        vol_df["exp_logmove_p10_24h"] = np.log1p(exp_ret_p10_24h.clip(lower=-0.999999))



        tte_series = df.get("time_to_exp1_hr")
        if vol_24 is not None and tte_series is not None:
            tte_clipped = tte_series.clip(lower=1e-6)
            with np.errstate(invalid="ignore"):
                sigma_tte = sigma_unit * np.sqrt(tte_clipped)
                exp_ret_p90_tte = sigma_tte * Z_SCORE_90
                exp_ret_p10_tte = sigma_tte * Z_SCORE_10

            vol_df["exp_ret_p90_tte"] = exp_ret_p90_tte
            vol_df["exp_ret_p10_tte"] = exp_ret_p10_tte
            vol_df["exp_band_width_tte"] = exp_ret_p90_tte - exp_ret_p10_tte
            vol_df["exp_logmove_p90_tte"] = np.log1p(exp_ret_p90_tte.clip(lower=-0.999999))
            vol_df["exp_logmove_p10_tte"] = np.log1p(exp_ret_p10_tte.clip(lower=-0.999999))


        if "returns_24h" in df.columns and "exp_band_width_24h" in vol_df.columns:
            band_24 = vol_df["exp_band_width_24h"].replace(0, np.nan)
            vol_df["realized_to_expected_24h"] = df["returns_24h"].abs() / band_24
        if tte_series is not None and "returns_1h" in df.columns and "exp_band_width_tte" in vol_df.columns:
            band_tte = vol_df["exp_band_width_tte"].replace(0, np.nan)
            vol_df["realized_to_expected_tte"] = df["returns_1h"].abs() / band_tte

        return vol_df

    def _liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        vlm_df = pd.DataFrame(index=df.index)
        vol_series = df.get("volCcy_prev")
        if vol_series is None:
            vol_series = df["volCcy"].shift(1)
        vol_adj = vol_series.replace(0, 1e-6)
        for w in self.vlm_window_sizes:
            min_p = max(1, w // 2)
            ma = vol_adj.rolling(w, min_periods=min_p).mean()
            std = vol_adj.rolling(w, min_periods=min_p).std().replace(0, 1e-6)
            vlm_df[f"vlm_ma_{w}h"] = ma
            vlm_df[f"vlm_zscore_{w}h"] = (vol_adj - ma) / std

        ratio_frames = []
        if "vlm_ma_24h" in vlm_df.columns and "vlm_ma_168h" in vlm_df.columns:
            if "vlm_ma_3h" in vlm_df.columns:
                ratio_frames.append(
                    pd.Series(
                        vlm_df["vlm_ma_3h"] / vlm_df["vlm_ma_168h"].replace(0, np.nan),
                        name="vlm_ratio_3h_168h",
                    )
                )
            if "vlm_ma_6h" in vlm_df.columns:
                ratio_frames.append(
                    pd.Series(
                        vlm_df["vlm_ma_6h"] / vlm_df["vlm_ma_168h"].replace(0, np.nan),
                        name="vlm_ratio_6h_168h",
                    )
                )
            ratio_frames.append(
                pd.Series(
                    vlm_df["vlm_ma_24h"] / vlm_df["vlm_ma_168h"].replace(0, np.nan),
                    name="vlm_ratio_24h_168h",
                )
            )
        if "vlm_ma_24h" in vlm_df.columns and "vlm_ma_720h" in vlm_df.columns:
            ratio_frames.append(
                pd.Series(
                    vlm_df["vlm_ma_24h"] / vlm_df["vlm_ma_720h"].replace(0, np.nan),
                    name="vlm_ratio_24h_720h",
                )
            )
        if "vlm_ma_24h" in vlm_df.columns and "vlm_ma_2160h" in vlm_df.columns:
            ratio_frames.append(
                pd.Series(
                    vlm_df["vlm_ma_24h"] / vlm_df["vlm_ma_2160h"].replace(0, np.nan),
                    name="vlm_ratio_24h_2160h",
                )
            )

        if ratio_frames:
            vlm_df = pd.concat([vlm_df] + [s.to_frame() for s in ratio_frames], axis=1)

        return vlm_df

    def _relative_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        rel_df = pd.DataFrame(index=df.index)
        high = df["h"].shift(1)
        low = df["l"].shift(1)
        close = df["c"].shift(1)

        for w in self.vol_window_sizes:
            if w <= 1:
                continue
            min_p = max(1, w // 2)
            rolling_high = high.rolling(window=w, min_periods=min_p).max()
            rolling_low = low.rolling(window=w, min_periods=min_p).min()
            range_w = rolling_high - rolling_low
            # Add epsilon to avoid division by zero in flat periods; flat periods get 0.5 (neutral)
            epsilon = 1e-10
            rel_df[f"stoch_pos_{w}h"] = ((close - rolling_low) / (range_w + epsilon)).clip(0, 1)
            rel_df[f"dist_from_high_{w}h"] = (rolling_high - close) / close.replace(0, np.nan)
            rel_df[f"dist_from_low_{w}h"] = (close - rolling_low) / close.replace(0, np.nan)
            rel_df[f"price_rank_{w}h"] = close.rolling(w).rank(pct=True)

        if 24 in self.vol_window_sizes:
            rel_df["new_24h_high"] = (high > df["h"].shift(2).rolling(23).max()).astype(int)
            rel_df["new_24h_low"] = (low < df["l"].shift(2).rolling(23).min()).astype(int)

        return rel_df

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, window: int) -> pd.Series:
        prev_close = df["prev_close"]
        tr_components = pd.concat(
            [
                df["h"] - df["l"],
                (df["h"] - prev_close).abs(),
                (df["l"] - prev_close).abs(),
            ],
            axis=1,
        )
        tr = tr_components.max(axis=1)
        min_p = max(1, window // 2)
        return tr.rolling(window=window, min_periods=min_p).mean().fillna(0)


class ComplexFeatureBlock:
    def __init__(self,
                 prev_daytype_window: int,
                 empirical_window: int,
                 empirical_min_count: int,
                 empirical_thresholds: Iterable[float]) -> None:
        self.prev_daytype_window = prev_daytype_window
        self.empirical_window = empirical_window
        self.empirical_min_count = empirical_min_count
        self.empirical_thresholds = list(empirical_thresholds)
        self.expiration_hour: Optional[int] = None

    def build_payload(self, df: pd.DataFrame) -> HeavyFeaturePayload:
        if self.expiration_hour is None:
            raise RuntimeError("ComplexFeatureBlock requires expiration_hour to be set")
        if "time_to_exp1_hr" not in df.columns or "day_type_num" not in df.columns:
            raise ValueError("Complex features require time_to_exp1_hr and day_type_num columns")

        working = df.copy()
        if "_cycle_start_ts" not in working.columns:
            working["_cycle_start_ts"] = df.index.to_series().where((df.index.hour == self.expiration_hour))

        working = self._add_current_cycle_progress_features(working)
        prev_cycle = self._add_prev_daytype_cycle_slice_features(working)
        prev_cycle_stats = self._add_rolling_stats_for_prev_daytype_features(prev_cycle)
        empirical = self._add_empirical_probability_features(prev_cycle_stats)

        time_bucket = self._bucket_time_to_exp(empirical["time_to_exp1_hr"])
        empirical = empirical.assign(_tte_bucket=time_bucket)

        prev_slice_cols = [col for col in empirical.columns 
                          if col.startswith("prev_") 
                          and not col.endswith("_tte_bucket")
                          and col != "prev_close"]  # Exclude prev_close to avoid duplication
        stats_cols = [col for col in empirical.columns if any(col.endswith(suffix) for suffix in ("_med", "_p10", "_p90"))]
        empirical_cols = [col for col in empirical.columns if col.startswith("emp_")]

        prev_cycle_lookup = (
            empirical.groupby(["day_type_num", "_tte_bucket"], dropna=False)[prev_slice_cols]
            .median()
            .sort_index()
        )

        prev_cycle_stats_lookup = (
            empirical.groupby(["day_type_num", "_tte_bucket"], dropna=False)[stats_cols]
            .last()
            .sort_index()
        )

        empirical_lookup = (
            empirical.groupby(["hour", "day_type_num"], dropna=False)[empirical_cols]
            .last()
            .sort_index()
        )

        metadata = {
            "version": HEAVY_CACHE_VERSION,
            "prev_daytype_window": self.prev_daytype_window,
            "empirical_window": self.empirical_window,
            "empirical_thresholds": self.empirical_thresholds,
            "empirical_min_count": self.empirical_min_count,
        }

        return HeavyFeaturePayload(
            prev_cycle_lookup=prev_cycle_lookup,
            prev_cycle_stats_lookup=prev_cycle_stats_lookup,
            empirical_lookup=empirical_lookup,
            metadata=metadata,
        )

    def lookup_for_row(self, row: pd.Series, payload: HeavyFeaturePayload) -> pd.Series:
        out_parts: List[pd.Series] = []
        tte_bucket = self._bucket_time_to_exp(pd.Series([row["time_to_exp1_hr"]])).iloc[0]
        day_type = row.get("day_type_num", np.nan)
        hour = row.get("hour", np.nan)

        if not np.isnan(day_type) and not np.isnan(tte_bucket):
            idx = (day_type, tte_bucket)
            if idx in payload.prev_cycle_lookup.index:
                out_parts.append(payload.prev_cycle_lookup.loc[idx])
            if idx in payload.prev_cycle_stats_lookup.index:
                out_parts.append(payload.prev_cycle_stats_lookup.loc[idx])

        if not np.isnan(day_type) and not np.isnan(hour):
            idx_emp = (hour, day_type)
            if idx_emp in payload.empirical_lookup.index:
                out_parts.append(payload.empirical_lookup.loc[idx_emp])

        if not out_parts:
            return pd.Series(dtype="float64")
        return pd.concat(out_parts)

    # --- Heavy feature helpers -------------------------------------------

    @staticmethod
    def _bucket_time_to_exp(series: pd.Series, precision: int = 2) -> pd.Series:
        return series.round(precision)

    def _add_current_cycle_progress_features(self, df: pd.DataFrame) -> pd.DataFrame:
        prog_cols = ["cCProgActP", "cCProgMinP", "cCProgMaxP", "cCProgMinT", "cCProgMaxT", "cCProgVlm"]
        df = df.copy()
        df[prog_cols] = np.nan

        expiry_times = df.index[df.index.hour == self.expiration_hour].sort_values()
        for i in range(len(expiry_times)):
            cycle_start = expiry_times[i]
            cycle_end = expiry_times[i + 1] - pd.Timedelta(hours=1) if i + 1 < len(expiry_times) else df.index[-1]
            cycle_mask = (df.index >= cycle_start) & (df.index <= cycle_end)
            cycle_df = df.loc[cycle_mask]
            if cycle_df.empty:
                continue

            prog_start_price = cycle_df.iloc[0]["prev_close"]
            cum_max = cycle_df["h"].expanding().max()
            cum_min = cycle_df["l"].expanding().min()
            idxs = cycle_df.index
            cum_max_t = cycle_df["h"].expanding().apply(
                lambda x: (idxs[x.argmax()] - cycle_start).total_seconds() // 3600
            )
            cum_min_t = cycle_df["l"].expanding().apply(
                lambda x: (idxs[x.argmin()] - cycle_start).total_seconds() // 3600
            )
            vol_series = cycle_df.get("volCcy_prev")
            if vol_series is None and "volCcy" in cycle_df.columns:
                # Fallback to shifted live volume if the pre-shifted field is missing.
                vol_series = cycle_df["volCcy"].shift(1)
            if vol_series is None:
                vol_series = pd.Series(0.0, index=cycle_df.index)
            cum_vlm = vol_series.fillna(0.0).cumsum()
            prog_actp = (cycle_df["c"] / prog_start_price) - 1
            prog_maxp = (cum_max / prog_start_price) - 1
            prog_minp = (cum_min / prog_start_price) - 1

            df.loc[cycle_mask, "cCProgActP"] = prog_actp.values
            df.loc[cycle_mask, "cCProgMaxP"] = prog_maxp.values
            df.loc[cycle_mask, "cCProgMinP"] = prog_minp.values
            df.loc[cycle_mask, "cCProgMaxT"] = cum_max_t.values
            df.loc[cycle_mask, "cCProgMinT"] = cum_min_t.values
            df.loc[cycle_mask, "cCProgVlm"] = cum_vlm.values

        return df

    def _add_prev_daytype_cycle_slice_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_types = ["weekday", "saturday", "sunday"]
        base_names = [
            "ProgActP",
            "ProgMinP",
            "ProgMaxP",
            "ProgMinT",
            "ProgMaxT",
            "ProgVlm",
            "RemActP",
            "RemMinP",
            "RemMaxP",
            "RemMinT",
            "RemMaxT",
        ]
        feature_names = [f"prev_{ftype}_{bname}" for ftype in feature_types for bname in base_names]
        results = pd.DataFrame(np.nan, index=df.index, columns=feature_names)

        for idx, row in df.iterrows():
            stats = self._get_prev_daytype_cycle_stats(row, df)
            results.loc[idx] = stats

        return pd.concat([df, results], axis=1)

    def _get_prev_daytype_cycle_stats(self, row: pd.Series, historical_df: pd.DataFrame) -> Tuple:
        t_now = row.name
        time_elapsed = 24 - row.get("time_to_exp1_hr", 0)
        nan_stats = (np.nan,) * 33

        def get_prev_cycle_start(ref_time: pd.Timestamp, target_dow: int) -> Optional[pd.Timestamp]:
            for i in range(1, 8):
                candidate = (ref_time - pd.Timedelta(days=i)).normalize() + pd.Timedelta(hours=self.expiration_hour)
                if candidate.dayofweek == target_dow:
                    return candidate
            return None

        daytype_map = {"weekday": [0, 1, 2, 3, 4], "saturday": [5], "sunday": [6]}
        stats: List[float] = []
        for ftype, dows in daytype_map.items():
            prev_cycle_start = None
            for dow in dows:
                candidate = get_prev_cycle_start(t_now, dow)
                if candidate is not None and candidate in historical_df.index:
                    prev_cycle_start = candidate
                    break
            if prev_cycle_start is None:
                stats.extend([np.nan] * 11)
                continue

            try:
                prog_end = prev_cycle_start + pd.Timedelta(hours=time_elapsed - 1)
                prog_data = historical_df.loc[prev_cycle_start:prog_end]
                rem_start = prev_cycle_start + pd.Timedelta(hours=time_elapsed)
                rem_end = prev_cycle_start + pd.Timedelta(hours=23)
                rem_data = historical_df.loc[rem_start:rem_end]

                if prog_data.empty:
                    prog_stats = [0.0] * 6
                else:
                    prog_start_price = prog_data.iloc[0]["prev_close"]
                    prog_end_price = prog_data.iloc[-1]["c"]
                    prog_max = prog_data["h"].max()
                    prog_min = prog_data["l"].min()
                    prog_max_t = int((prog_data["h"].idxmax() - prev_cycle_start).total_seconds() / 3600)
                    prog_min_t = int((prog_data["l"].idxmin() - prev_cycle_start).total_seconds() / 3600)
                    vol_series = prog_data.get("volCcy_prev")
                    if vol_series is None and "volCcy" in prog_data.columns:
                        vol_series = prog_data["volCcy"].shift(1)
                    if vol_series is None:
                        vol_series = pd.Series(0.0, index=prog_data.index)
                    prog_vlm = vol_series.fillna(0.0).sum()
                    prog_actp = (prog_end_price / prog_start_price) - 1
                    prog_maxp = (prog_max / prog_start_price) - 1
                    prog_minp = (prog_min / prog_start_price) - 1
                    prog_stats = [prog_actp, prog_minp, prog_maxp, prog_min_t, prog_max_t, prog_vlm]

                if rem_data.empty:
                    rem_stats = [np.nan] * 5
                else:
                    rem_start_price = rem_data.iloc[0]["prev_close"]
                    rem_end_price = rem_data.iloc[-1]["c"]
                    rem_max = rem_data["h"].max()
                    rem_min = rem_data["l"].min()
                    rem_max_t = int((rem_data["h"].idxmax() - prev_cycle_start).total_seconds() / 3600)
                    rem_min_t = int((rem_data["l"].idxmin() - prev_cycle_start).total_seconds() / 3600)
                    rem_actp = (rem_end_price / rem_start_price) - 1
                    rem_maxp = (rem_max / rem_start_price) - 1
                    rem_minp = (rem_min / rem_start_price) - 1
                    rem_stats = [rem_actp, rem_minp, rem_maxp, rem_min_t, rem_max_t]

                stats.extend(prog_stats + rem_stats)
            except Exception:
                stats.extend([np.nan] * 11)

        if len(stats) != 33:
            return nan_stats
        return tuple(stats)

    def _add_rolling_stats_for_prev_daytype_features(self, df: pd.DataFrame) -> pd.DataFrame:
        stat_cols = [
            col
            for col in df.columns
            if col.startswith("prev_")
            and (
                col.endswith("_ProgMaxP")
                or col.endswith("_RemMaxP")
                or col.endswith("_ProgMinP")
                or col.endswith("_RemMinP")
            )
        ]
        grouped = df.groupby(["day_type_num", "time_to_exp1_hr"], group_keys=False)
        for col in stat_cols:
            df[f"{col}_med_{self.prev_daytype_window}"] = grouped[col].transform(
                lambda s: s.rolling(self.prev_daytype_window, min_periods=max(5, self.prev_daytype_window // 5)).median()
            )
            df[f"{col}_p10_{self.prev_daytype_window}"] = grouped[col].transform(
                lambda s: s.rolling(self.prev_daytype_window, min_periods=max(5, self.prev_daytype_window // 5)).quantile(0.1)
            )
            df[f"{col}_p90_{self.prev_daytype_window}"] = grouped[col].transform(
                lambda s: s.rolling(self.prev_daytype_window, min_periods=max(5, self.prev_daytype_window // 5)).quantile(0.9)
            )
        return df

    def _add_empirical_probability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"logret_h_pc", "logret_l_pc"}
        if not required_cols.issubset(df.columns):
            raise ValueError("Empirical probability features require logret_h_pc and logret_l_pc")

        thresholds = self.empirical_thresholds
        for t in thresholds:
            df[f"emp_freq_logret_h_pc_ge_{t}"] = np.nan
            df[f"emp_freq_logret_l_pc_le_-{t}"] = np.nan

        for p in [0.05, 0.3, 0.5, 0.7, 0.95]:
            df[f"emp_pct_{int(p * 100)}_logret_h_pc"] = np.nan
            df[f"emp_pct_{int(p * 100)}_logret_l_pc"] = np.nan

        df["emp_median_logret_range"] = np.nan
        if "realized_to_expected_24h" in df.columns:
            df["emp_median_realized_to_expected_24h"] = np.nan
        if "realized_to_expected_tte" in df.columns:
            df["emp_median_realized_to_expected_tte"] = np.nan

        grouped = df.groupby(["hour", "day_type_num"])
        for (hour, day_type), group in grouped:
            idx = group.index
            for t in thresholds:
                up_mask = (group["logret_h_pc"] >= t).astype(float)
                down_mask = (group["logret_l_pc"] <= -t).astype(float)
                up_prob = up_mask.rolling(window=self.empirical_window, min_periods=self.empirical_min_count).mean().shift(1)
                down_prob = down_mask.rolling(window=self.empirical_window, min_periods=self.empirical_min_count).mean().shift(1)
                df.loc[idx, f"emp_freq_logret_h_pc_ge_{t}"] = up_prob.values
                df.loc[idx, f"emp_freq_logret_l_pc_le_-{t}"] = down_prob.values

            for p in [0.05, 0.3, 0.5, 0.7, 0.95]:
                up_pct = group["logret_h_pc"].rolling(window=self.empirical_window, min_periods=self.empirical_min_count).quantile(p).shift(1)
                down_pct = group["logret_l_pc"].rolling(window=self.empirical_window, min_periods=self.empirical_min_count).quantile(p).shift(1)
                df.loc[idx, f"emp_pct_{int(p * 100)}_logret_h_pc"] = up_pct.values
                df.loc[idx, f"emp_pct_{int(p * 100)}_logret_l_pc"] = down_pct.values

            logret_range = group["logret_h_pc"] - group["logret_l_pc"]
            median_range = logret_range.rolling(window=self.empirical_window, min_periods=self.empirical_min_count).median().shift(1)
            df.loc[idx, "emp_median_logret_range"] = median_range.values

            if "realized_to_expected_24h" in group.columns:
                rte24 = group["realized_to_expected_24h"].rolling(
                    window=self.empirical_window,
                    min_periods=self.empirical_min_count,
                ).median().shift(1)
                df.loc[idx, "emp_median_realized_to_expected_24h"] = rte24.values

            if "realized_to_expected_tte" in group.columns:
                rtette = group["realized_to_expected_tte"].rolling(
                    window=self.empirical_window,
                    min_periods=self.empirical_min_count,
                ).median().shift(1)
                df.loc[idx, "emp_median_realized_to_expected_tte"] = rtette.values

        return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """High-level orchestrator for the tiered feature pipeline."""

    def __init__(self,
                 expiration_hour: int = 8,
                 vol_window_sizes: Iterable[int] = (3, 6, 12, 24, 48, 72, 24 * 3, 24 * 7, 24 * 14, 24 * 30, 24 * 90),
                 vlm_window_sizes: Iterable[int] = (3, 6, 12, 24, 48, 72, 24 * 3, 24 * 7, 24 * 14, 24 * 30, 24 * 90),
                 vol_types_to_calc: Iterable[str] = ("raw", "gkyz", "parkinson", "rogers_satchell", "skew", "kurtosis", "vol_zscore", "log_vol"),  ## remember to manage based on rocket or catboost
                 vol_trading_periods: int = 24 * 365,
                 include_price: bool = False,
                 include_trend: bool = True,
                 include_volatility: bool = True,
                 include_relative_position: bool = True,
                 include_temporal: bool = True,
                 include_liquidity: bool = True,
                 include_non_linear: bool = True,
                 include_custom_interactions: bool = False,
                 include_prev_week_cycle: bool = True,
                 include_dst_feature: bool = True,
                 cache_dir: Optional[Path] = Path("cache") / "heavy_features",
                 verbose: bool = False) -> None:
        self.expiration_hour = expiration_hour
        self.include_price = include_price
        self.include_trend = include_trend
        self.include_volatility = include_volatility
        self.include_relative_position = include_relative_position
        self.include_temporal = include_temporal
        self.include_liquidity = include_liquidity
        self.include_non_linear = include_non_linear
        self.include_custom_interactions = include_custom_interactions
        self.include_prev_week_cycle = include_prev_week_cycle
        self.include_dst_feature = include_dst_feature
        self.verbose = verbose
        self.vol_window_sizes = list(vol_window_sizes)
        self.vlm_window_sizes = list(vlm_window_sizes)
        self.vol_types_to_calc = [v.lower() for v in vol_types_to_calc]
        self.vol_trading_periods = vol_trading_periods
        self.prev_daytype_window = DEFAULT_PREV_DAYTYPE_WINDOW
        self.empirical_window = DEFAULT_EMPIRICAL_WINDOW
        self.empirical_min_count = DEFAULT_EMPIRICAL_MIN_COUNT
        self.empirical_thresholds = list(DEFAULT_EMPIRICAL_THRESHOLDS)

        self.stateless_block = StatelessFeatureBlock(expiration_hour, include_dst_feature)
        self.rolling_block = RollingFeatureBlock(self.vol_window_sizes,
                                                 self.vlm_window_sizes,
                                                 self.vol_types_to_calc,
                                                 self.vol_trading_periods)
        self.complex_block = ComplexFeatureBlock(self.prev_daytype_window,
                                                 self.empirical_window,
                                                 self.empirical_min_count,
                                                 self.empirical_thresholds)
        self.complex_block.expiration_hour = expiration_hour

        self.heavy_cache = HeavyFeatureCache(cache_dir)
        self.live_state = None
        self.feature_names_out_ = None
        self._full_reference = None
        self._reference_features = None
        self._heavy_payload: Optional[HeavyFeaturePayload] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        reference = self._prepare_reference_frame(X)
        self._full_reference = reference

        self._log(f"fit start; rows={len(reference)}")
        start = time.perf_counter()
        features = self._compute_all_features(reference, build_heavy=True)
        self.feature_names_out_ = features.columns.tolist()
        self._reference_features = features
        duration = time.perf_counter() - start
        self._log(
            f"fit complete; rows={len(reference)}, cols={features.shape[1]}, elapsed={duration:.2f}s"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._full_reference is None:
            raise RuntimeError("fit() must be called before transform().")
        frame = self._prepare_reference_frame(X)
        self._log(f"transform start; rows={len(frame)}")
        start = time.perf_counter()
        features = self._compute_all_features(frame, build_heavy=False)
        duration = time.perf_counter() - start
        self._log(
            f"transform complete; rows={len(frame)}, cols={features.shape[1]}, elapsed={duration:.2f}s"
        )
        return features

    # Live update API -------------------------------------------------
    def initialize_live_state(self, historical_df: pd.DataFrame) -> LiveStateCache:
        features = self.transform(historical_df)
        price_cols = [col for col in ["o", "h", "l", "c", "volCcy"] if col in historical_df.columns]
        price_history = historical_df.loc[features.index, price_cols]
        metadata = {"max_history": self._live_history_window()}
        self.live_state = LiveStateCache(
            features_df=features.copy(),
            price_history=price_history.copy(),
            rolling_states={},
            ema_states={},
            metadata=metadata,
        )
        return self.live_state

    def compute_next_row(self, new_row: pd.Series, *, commit: bool = False) -> pd.Series:
        self._ensure_live_state_ready()
        if not isinstance(new_row.name, pd.Timestamp):
            raise TypeError("New row must have a DatetimeIndex timestamp as its name.")

        if not self.live_state.price_history.empty:
            last_index = self.live_state.price_history.index[-1]
            if new_row.name <= last_index:
                raise ValueError(
                    f"New row timestamp {new_row.name} must be greater than last history index {last_index}."
                )

        return self._append_live_row(new_row, commit=commit)

    def _live_history_window(self) -> int:
        return max(self.vol_window_sizes + self.vlm_window_sizes + [24 * 30]) + 5

    def update_last_row(self, updated_row: pd.Series, *, commit: bool = True) -> pd.Series:
        self._ensure_live_state_ready()
        if self.live_state.price_history.empty:
            raise RuntimeError("Live state history is empty; cannot update last row.")
        if not isinstance(updated_row.name, pd.Timestamp):
            raise TypeError("Updated row must have a DatetimeIndex timestamp as its name.")
        last_index = self.live_state.price_history.index[-1]
        if updated_row.name != last_index:
            raise ValueError(
                f"Updated row timestamp {updated_row.name} does not match last history index {last_index}."
            )
        return self._update_live_last_row(updated_row, commit=commit)

    def ingest_live_row(self, row: pd.Series, *, commit: bool = True) -> pd.Series:
        """Ingest a streaming bar, updating or appending as needed.

        If ``row.name`` (timestamp) matches the most recent entry, the cached
        features are recomputed for that bar. If the timestamp is newer than the
        cached history, the bar is appended. Rows older than the last cached bar
        raise a ``ValueError``.
        """

        self._ensure_live_state_ready()
        if not isinstance(row.name, pd.Timestamp):
            raise TypeError("Row must have a DatetimeIndex timestamp as its name.")

        if self.live_state.price_history.empty:
            return self._append_live_row(row, commit=commit)

        last_index = self.live_state.price_history.index[-1]
        if row.name > last_index:
            return self._append_live_row(row, commit=commit)
        if row.name == last_index:
            return self._update_live_last_row(row, commit=commit)

        raise ValueError(
            f"Row timestamp {row.name} precedes the last cached timestamp {last_index}; "
            "historical backfills are not supported by ingest_live_row()."
        )

    def _ensure_live_state_ready(self) -> None:
        if self.live_state is None:
            raise RuntimeError("Live state not initialized; call initialize_live_state() first.")
        if self._heavy_payload is None and self.include_prev_week_cycle:
            if not self.heavy_cache.load():
                raise RuntimeError("Heavy cache unavailable; run fit() to build heavy features.")
            self._heavy_payload = self.heavy_cache.payload

    def _append_live_row(self, new_row: pd.Series, *, commit: bool) -> pd.Series:
        price_cols = self.live_state.price_history.columns
        missing_cols = [col for col in price_cols if col not in new_row.index]
        if missing_cols:
            raise ValueError(f"New row missing required columns: {missing_cols}")

        history_frame = self.live_state.price_history
        max_history = self.live_state.metadata.get("max_history")
        if max_history is not None and len(history_frame) >= max_history:
            trimmed_history = history_frame.tail(max_history - 1)
        else:
            trimmed_history = history_frame

        new_price_frame = new_row[price_cols].to_frame().T
        candidate_history = pd.concat([trimmed_history, new_price_frame])
        features = self._compute_all_features(candidate_history, build_heavy=False)
        new_features = features.iloc[-1]

        if commit:
            self.live_state.append(new_features, new_row[price_cols])
        return new_features

    def _update_live_last_row(self, updated_row: pd.Series, *, commit: bool) -> pd.Series:
        price_history = self.live_state.price_history
        price_cols = price_history.columns
        missing_cols = [col for col in price_cols if col not in updated_row.index]
        if missing_cols:
            raise ValueError(f"Updated row missing required columns: {missing_cols}")

        last_index = price_history.index[-1]
        history_frame = price_history.copy()
        history_frame.loc[last_index, price_cols] = updated_row[price_cols].values

        max_history = self.live_state.metadata.get("max_history")
        if max_history is not None and len(history_frame) > max_history:
            trimmed_history = history_frame.tail(max_history)
        else:
            trimmed_history = history_frame

        features = self._compute_all_features(trimmed_history, build_heavy=False)
        updated_features = features.iloc[-1]

        if commit:
            self.live_state.features_df.loc[last_index] = updated_features
            self.live_state.price_history.loc[last_index, price_cols] = updated_row[price_cols].values

        return updated_features

    # --- Internal helpers --------------------------------------------

    def _prepare_reference_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Input DataFrame must use a DatetimeIndex.")
        frame = df.copy()
        if frame.index.tz is not None:
            frame = frame.tz_localize(None)
        if not frame.index.is_monotonic_increasing:
            frame = frame.sort_index()
        return frame

    def _compute_all_features(self, df: pd.DataFrame, *, build_heavy: bool) -> pd.DataFrame:
        timings: List[Tuple[str, float]] = []

        suppress_warnings = not self.verbose
        warnings_ctx = warnings.catch_warnings() if suppress_warnings else nullcontext()
        with warnings_ctx:
            if suppress_warnings:
                warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
                warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)

            t_start = time.perf_counter()
            stateless = self.stateless_block.compute(df)
            timings.append(("stateless", time.perf_counter() - t_start))

            t_start = time.perf_counter()
            working = pd.concat([df, stateless], axis=1)
            if "volCcy" in working.columns:
                working["volCcy"] = working["volCcy"].round()
            timings.append(("merge_stateless", time.perf_counter() - t_start))

            if self.include_temporal:
                t_start = time.perf_counter()
                working = self._add_temporal_features(working)
                timings.append(("temporal", time.perf_counter() - t_start))

            t_start = time.perf_counter()
            rolling = self.rolling_block.compute(working)
            working = pd.concat([working, rolling], axis=1)
            timings.append(("rolling", time.perf_counter() - t_start))

            if self.include_prev_week_cycle:
                t_start = time.perf_counter()
                if build_heavy or self._heavy_payload is None:
                    payload = self.complex_block.build_payload(working)
                    self.heavy_cache.save(payload)
                    self._heavy_payload = payload
                else:
                    payload = self._heavy_payload
                heavy_df = self._render_heavy_features(working, payload)
                working = pd.concat([working, heavy_df], axis=1)
                timings.append(("prev_week_cycle", time.perf_counter() - t_start))

            # Add current cycle features (fast computation, not cached)
            t_start = time.perf_counter()
            working = self._add_current_cycle_features(working)
            timings.append(("current_cycle", time.perf_counter() - t_start))

            if self.include_non_linear:
                t_start = time.perf_counter()
                working = self._add_non_linear_features(working)
                timings.append(("non_linear", time.perf_counter() - t_start))
            if self.include_custom_interactions:
                t_start = time.perf_counter()
                working = self._add_custom_interactions(working)
                timings.append(("custom_interactions", time.perf_counter() - t_start))

            t_start = time.perf_counter()
            working = working.replace([np.inf, -np.inf], np.nan)
            working = self._apply_feature_toggles(working)
            working = working.copy()
            timings.append(("cleanup", time.perf_counter() - t_start))

            if self.verbose:
                total = sum(duration for _, duration in timings)
                summary = ", ".join(
                    f"{name}:{duration * 1000:.1f}ms" for name, duration in timings if duration > 0.0
                )
                self._log(
                    f"feature build complete; rows={len(df)}, cols={working.shape[1]}, total={total:.2f}s [{summary}]"
                )
            return working

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[FeatureEngineer] {message}")

    def _render_heavy_features(self, df: pd.DataFrame, payload: HeavyFeaturePayload) -> pd.DataFrame:
        pieces: List[pd.DataFrame] = []
        tte_bucket = self.complex_block._bucket_time_to_exp(df["time_to_exp1_hr"])
        day_type = df["day_type_num"]
        hour = df["hour"]

        if not payload.prev_cycle_lookup.empty:
            idx = pd.MultiIndex.from_arrays(
                [day_type.values, tte_bucket.values],
                names=payload.prev_cycle_lookup.index.names,
            )
            prev_cycle = payload.prev_cycle_lookup.reindex(idx)
            prev_cycle.index = df.index
            pieces.append(prev_cycle)

        if not payload.prev_cycle_stats_lookup.empty:
            idx_stats = pd.MultiIndex.from_arrays(
                [day_type.values, tte_bucket.values],
                names=payload.prev_cycle_stats_lookup.index.names,
            )
            stats_df = payload.prev_cycle_stats_lookup.reindex(idx_stats)
            stats_df.index = df.index
            pieces.append(stats_df)

        if not payload.empirical_lookup.empty:
            idx_emp = pd.MultiIndex.from_arrays(
                [hour.values, day_type.values],
                names=payload.empirical_lookup.index.names,
            )
            emp_df = payload.empirical_lookup.reindex(idx_emp)
            emp_df.index = df.index
            pieces.append(emp_df)

        if not pieces:
            return pd.DataFrame(index=df.index)
        heavy_df = pd.concat(pieces, axis=1)
        return heavy_df

    def _apply_feature_toggles(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df

        if not self.include_price:
            drop_cols = [col for col in ["o", "h", "l", "c", "volCcy"] if col in result.columns]
            result = result.drop(columns=drop_cols)
        if not self.include_trend:
            #trend_prefixes = ("sma_", "ema_", "momentum_", "macd", "adx")  # dont want sma and ema in features as they are prices 
            trend_prefixes = ("momentum_", "macd", "adx")
            trend_cols = [col for col in result.columns if col.startswith(trend_prefixes)]
            result = result.drop(columns=trend_cols, errors="ignore")
        if not self.include_volatility:
            vol_prefixes = ("vol_", "returns_skew", "returns_kurtosis", "atr_", "log_vol_")
            vol_cols = [col for col in result.columns if col.startswith(vol_prefixes)]
            result = result.drop(columns=vol_cols, errors="ignore")
        if not self.include_relative_position:
            rel_prefixes = ("stoch_pos_", "dist_from_", "price_rank_", "new_24h")
            rel_cols = [col for col in result.columns if col.startswith(rel_prefixes)]
            result = result.drop(columns=rel_cols, errors="ignore")
        if not self.include_liquidity:
            vlm_cols = [col for col in result.columns if col.startswith("vlm_")]
            result = result.drop(columns=vlm_cols, errors="ignore")
        if not self.include_temporal:
            temporal_cols = [col for col in ["tte_phase_cos", "tte_phase_sin", "is_dst"] if col in result.columns]
            result = result.drop(columns=temporal_cols, errors="ignore")
        helper_cols = [col for col in ["cycle_id", "_cycle_start_ts", "_tte_bucket"] if col in result.columns]
        if helper_cols:
            result = result.drop(columns=helper_cols, errors="ignore")
        return result

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "time_to_exp1_hr" not in df.columns:
            return df
        df = df.copy()
        tte = df["time_to_exp1_hr"]
        df["tte_phase_cos"] = np.cos(2 * np.pi * tte / 24)
        df["tte_phase_sin"] = np.sin(2 * np.pi * tte / 24)
        return df

    def _add_non_linear_features(self, df: pd.DataFrame) -> pd.DataFrame:
        required_base = {"c", "h", "l"}
        if not required_base.issubset(df.columns):
            return df

        vol_series = df.get("volCcy_prev")
        if vol_series is None and "volCcy" in df.columns:
            vol_series = df["volCcy"].shift(1)
        if vol_series is None:
            return df

        working = df.copy()
        if "volCcy_prev" not in working.columns:
            working["volCcy_prev"] = vol_series
        if "vol_gkyz_24h" not in working.columns:
            working["vol_gkyz_24h"] = 0.0
        if "vlm_ma_24h" not in working.columns:
            working["vlm_ma_24h"] = vol_series.rolling(24, min_periods=1).mean()

        working = self._add_vol_volume_interactions(working)
        working = self._add_higher_order_momentum(working)
        working = self._add_non_linear_range_features(working)
        working = self._add_liquidity_shock_features(working, vol_series)
        working = self._add_time_decay_features(working)
        working = self._add_tail_specific_features(working)
        working = self._add_time_to_expiry_interactions(working)
        return working

    def _add_time_to_expiry_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        if "time_to_exp1_hr" not in df.columns:
            return df
        tte = df["time_to_exp1_hr"]
        tte_sqrt = np.sqrt(tte.clip(lower=1e-6))  # Black-Scholes scaling: σ√t
        
        key_features = [
            "vol_gkyz_24h",
            "vlm_ma_24h",
            "returns_kurtosis_24h",
            "returns_skew_24h",
            "extreme_prob",
            "vol_clustering",
            "exp_ret_p90_24h",
            "exp_ret_p10_24h",
            "exp_band_width_24h",
            "exp_ret_p90_tte",
            "exp_ret_p10_tte",
            "exp_band_width_tte",
        ]
        for feat in key_features:
            if feat in df.columns:
                # Black-Scholes √t scaling (most important for volatility)
                df[f"{feat}_x_tte_sqrt"] = df[feat] * tte_sqrt
                # Linear scaling (for non-vol features)
                df[f"{feat}_x_tte"] = df[feat] * tte
                # Higher order terms for capturing non-linear time decay
                df[f"{feat}_x_tte_sq"] = df[feat] * (tte ** 2)
                df[f"{feat}_x_tte_cu"] = df[feat] * (tte ** 3)
        return df

    def _add_vol_volume_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["vol_gkyz_24h", "vlm_ma_24h", "returns_1h"]
        if not all(col in df.columns for col in cols):
            return df
        safe_vlm = df["vlm_ma_24h"].clip(lower=1e-6)
        safe_vol = df["vol_gkyz_24h"].clip(lower=1e-6)
        df["vol_weighted_vol"] = safe_vol * np.log1p(safe_vlm)
        df["vol_vlm_ratio_change"] = (safe_vol / safe_vlm).pct_change().fillna(0)
        df["asym_vol_vlm_impact"] = np.sign(df["returns_1h"]) * (safe_vol ** 2) * np.sqrt(safe_vlm)
        return df

    def _add_current_cycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add current cycle progress features to transform pipeline.
        
        Fast computation of current cycle state for 'where we are now' context.
        Complements previous cycle features with current positioning.
        """
        # Only compute if we have the required OHLC data
        required_cols = ["h", "l", "c", "prev_close"]
        if not all(col in df.columns for col in required_cols):
            if self.verbose:
                missing = [col for col in required_cols if col not in df.columns]
                print(f"Skipping current cycle features - missing columns: {missing}")
            return df
        
        return self.complex_block._add_current_cycle_progress_features(df)

    def _add_higher_order_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        for w in [6, 12, 24]:
            col = f"momentum_{w}h"
            if col in df.columns:
                df[f"momentum_accel_{w}h"] = df[col] * df[col].diff()
        if "momentum_24h" in df.columns:
            momentum_24 = df["momentum_24h"]
            df["signed_momentum_power"] = np.sign(momentum_24) * np.abs(momentum_24) ** 1.5
            if "vol_gkyz_24h" in df.columns:
                df["mom_vol_interaction"] = momentum_24 * df["vol_gkyz_24h"].rolling(6, min_periods=1).std()
        return df

    def _add_non_linear_range_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "range_pc" not in df.columns:
            df["range_pc"] = (df["h"].shift(1) - df["l"].shift(1)) / df["prev_close"].replace(0, np.nan)
        if "vol_gkyz_24h" not in df.columns:
            df["vol_gkyz_24h"] = 0.0
        if "close_pos_in_bar" not in df.columns:
            range_bar = df["h"].shift(1) - df["l"].shift(1)
            df["close_pos_in_bar"] = np.where(
                range_bar.abs() > 1e-9,
                (df["c"].shift(1) - df["l"].shift(1)) / range_bar,
                0.5,
            )

        range_pc = df["range_pc"]
        vol_24 = df["vol_gkyz_24h"]
        close_pos = df["close_pos_in_bar"].clip(0, 1)
        range_ma = range_pc.rolling(24, min_periods=1).mean().clip(lower=1e-9)
        df["compressed_range_vol"] = np.sqrt(range_pc.clip(lower=0)) * vol_24
        df["range_expansion"] = (range_pc / range_ma) ** 2
        df["nl_pos_in_range"] = np.sin(np.pi * close_pos) * vol_24
        return df

    def _add_liquidity_shock_features(self, df: pd.DataFrame, vol_series: pd.Series) -> pd.DataFrame:
        required = {"vol_gkyz_24h", "vlm_ma_24h"}
        if not required.issubset(df.columns):
            return df

        safe_vol = df["vol_gkyz_24h"].clip(lower=1e-6)
        safe_vlm = df["vlm_ma_24h"].clip(lower=1e-6)
        df["volume_surprise"] = (vol_series - df["vlm_ma_24h"]) / (safe_vlm * safe_vol)
        df["liq_vol"] = vol_series.rolling(6, min_periods=2).std() / safe_vlm
        df["liq_vol_ratio_change"] = (safe_vlm / safe_vol).pct_change().rolling(6, min_periods=2).mean()
        trend_std = safe_vlm.rolling(24, min_periods=5).std().clip(lower=1e-6)
        df["volume_trend_z"] = (vol_series - safe_vlm) / trend_std
        return df

    def _add_time_decay_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "returns_1h" in df.columns:
            returns = df["returns_1h"]
            for w in [6, 12, 24]:
                weights = np.exp(-np.linspace(0, 1, w))
                weights /= weights.sum()
                df[f"exp_decay_ret_{w}h"] = returns.rolling(w).apply(
                    lambda x, wts=weights[::-1]: float(np.dot(x, wts)), raw=True
                )
        if "vol_gkyz_24h" in df.columns and "time_to_exp1_hr" in df.columns:
            df["time_adj_vol"] = df["vol_gkyz_24h"] * (1 + df["time_to_exp1_hr"] / 24)
        if {"vol_gkyz_24h", "tte_phase_sin", "tte_phase_cos"}.issubset(df.columns):
            df["cyclical_vol_compression"] = (
                df["tte_phase_sin"] * df["vol_gkyz_24h"] +
                df["tte_phase_cos"] * df["vol_gkyz_24h"].diff()
            )
        return df

    def _add_tail_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"vol_gkyz_24h", "returns_kurtosis_24h", "returns_skew_24h"}
        if not required.issubset(df.columns):
            return df
        safe_vol = df["vol_gkyz_24h"].clip(lower=1e-6)
        df["extreme_prob"] = (
            df["returns_kurtosis_24h"] * df["returns_skew_24h"] * safe_vol
        ).rolling(12, min_periods=3).mean()
        df["vol_clustering"] = (df["vol_gkyz_24h"].diff() > 0).rolling(24, min_periods=6).sum() * safe_vol
        return df

    def _add_custom_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        def safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-8) -> pd.Series:
            denom = b.replace(0, np.nan).fillna(eps)
            return a / denom

        # Original interactions
        if {"pWRemMaxP_p90_vs_median_spread", "vol_gkyz_24h_x_tte"}.issubset(df.columns):
            df["pWRem_spread_to_vol"] = safe_div(
                df["pWRemMaxP_p90_vs_median_spread"],
                df["vol_gkyz_24h_x_tte"].abs() + 1e-6,
            )
        if {"cWProgMaxP_vs_p90_upside", "vol_gkyz_24h"}.issubset(df.columns):
            df["progmax_vs_vol"] = safe_div(df["cWProgMaxP_vs_p90_upside"], df["vol_gkyz_24h"].abs() + 1e-6)
        if {"returns_1h", "vol_gkyz_24h", "vlm_ma_24h"}.issubset(df.columns):
            df["shock_absorption"] = df["returns_1h"] * safe_div(df["vol_gkyz_24h"], df["vlm_ma_24h"])
        
        # Critical new interactions for direct target optimization
        df = self._add_tte_volatility_interactions(df)
        df = self._add_weekend_regime_interactions(df)
        df = self._add_cycle_progress_interactions(df) 
        df = self._add_extreme_event_interactions(df)
        
        return df
    
    def _add_tte_volatility_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-to-expiry × volatility interactions crucial for return prediction"""
        
        if "time_to_exp1_hr" not in df.columns:
            return df
            
        tte = df["time_to_exp1_hr"]
        tte_sqrt = np.sqrt(tte.clip(lower=1e-6))  # Black-Scholes: σ√t
        tte_normalized = tte / 168  # Normalize by week
        
        # Core TTE × volatility combinations
        vol_features = ["vol_gkyz_3h", "vol_gkyz_6h", "vol_gkyz_12h", "vol_gkyz_24h", "vol_gkyz_288h",
                       "vol_raw_24h", "vol_raw_288h"]  # Include raw vol for consistency
        
        for vol_feat in vol_features:
            if vol_feat in df.columns:
                # Black-Scholes √t scaling (PRIMARY for volatility → expected move)
                df[f"{vol_feat}_x_tte_sqrt"] = df[vol_feat] * tte_sqrt
                
                # Linear TTE (for regime/level effects)
                df[f"{vol_feat}_x_tte"] = df[vol_feat] * tte
                
                # Squared TTE (for strong time decay near expiry)
                df[f"{vol_feat}_x_tte_sq"] = df[vol_feat] * (tte_normalized ** 2)
                
                # Cyclical TTE effects (captures intraday patterns)
                if "tte_phase_sin" in df.columns:
                    df[f"{vol_feat}_x_tte_sin"] = df[vol_feat] * df["tte_phase_sin"]
                if "tte_phase_cos" in df.columns:
                    df[f"{vol_feat}_x_tte_cos"] = df[vol_feat] * df["tte_phase_cos"]
        
        # Volatility term structure × TTE (all three scalings)
        if {"vol_gkyz_3h", "vol_gkyz_24h"}.issubset(df.columns):
            vol_term_slope = df["vol_gkyz_24h"] - df["vol_gkyz_3h"]
            df["vol_term_x_tte_sqrt"] = vol_term_slope * tte_sqrt  # Black-Scholes scaling
            df["vol_term_x_tte"] = vol_term_slope * tte
            df["vol_term_x_tte_sq"] = vol_term_slope * (tte_normalized ** 2)
        
        return df
    
    def _add_weekend_regime_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add weekend/weekday regime-specific interactions"""
        
        if "is_weekend" not in df.columns:
            return df
            
        # Weekend-specific volatility behavior
        vol_features = ["vol_gkyz_3h", "vol_gkyz_6h", "vol_gkyz_12h", "vol_gkyz_24h"]
        for vol_feat in vol_features:
            if vol_feat in df.columns:
                df[f"{vol_feat}_weekend"] = df[vol_feat] * df["is_weekend"]
                df[f"{vol_feat}_weekday"] = df[vol_feat] * (1 - df["is_weekend"])
        
        # Weekend × previous cycle progress (leveraging historical weekend patterns)
        prev_weekend_cycles = ['prev_saturday', 'prev_sunday'] 
        prev_cycle_metrics = ['ProgActP', 'ProgMaxP', 'ProgMinP', 'ProgVlm']
        
        for weekend_cycle in prev_weekend_cycles:
            for metric in prev_cycle_metrics:
                prev_feat = f"{weekend_cycle}_{metric}"
                if prev_feat in df.columns:
                    df[f"{prev_feat}_weekend"] = df[prev_feat] * df["is_weekend"]
        
        # Weekend × volume effects
        if "vlm_ma_24h" in df.columns:
            df["volume_weekend_effect"] = df["vlm_ma_24h"] * df["is_weekend"]
        
        return df
        
    def _add_cycle_progress_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add previous cycle progress × market condition interactions
        
        Note: Uses 'prev_' cycle features (weekday/saturday/sunday) which track 
        complete historical cycles, not current incomplete cycle progress.
        This is superior for prediction as it uses complete cycle information.
        """
        
        # Previous cycle progress features (complete historical cycles)
        prev_cycle_types = ['prev_weekday', 'prev_saturday', 'prev_sunday']
        prog_metrics = ['ProgActP', 'ProgMaxP', 'ProgMinP', 'ProgVlm']
        vol_features = ["vol_gkyz_6h", "vol_gkyz_12h", "vol_gkyz_24h"]
        
        # Previous cycle progress × volatility interactions
        for cycle_type in prev_cycle_types:
            for prog_metric in prog_metrics:
                prog_feature = f"{cycle_type}_{prog_metric}"
                if prog_feature in df.columns:
                    for vol_feat in vol_features:
                        if vol_feat in df.columns:
                            # Use simpler naming pattern to match existing interactions
                            interaction_name = f"{prog_feature}_x_{vol_feat.replace('vol_gkyz_', 'vol')}"
                            df[interaction_name] = df[prog_feature] * df[vol_feat]
        
        # Previous cycle range × volatility (more stable than current cycle)
        for cycle_type in prev_cycle_types:
            max_feat = f"{cycle_type}_ProgMaxP"
            min_feat = f"{cycle_type}_ProgMinP"
            if {max_feat, min_feat, "vol_gkyz_12h"}.issubset(df.columns):
                cycle_range = df[max_feat] - df[min_feat]
                df[f"{cycle_type}_range_x_vol"] = cycle_range * df["vol_gkyz_12h"]
        
        # Cross-regime cycle comparisons (weekday vs weekend behavior)
        if {"prev_weekday_ProgActP", "prev_saturday_ProgActP"}.issubset(df.columns):
            df["weekday_vs_saturday_prog"] = df["prev_weekday_ProgActP"] - df["prev_saturday_ProgActP"]
        
        if {"prev_weekday_ProgActP", "prev_sunday_ProgActP"}.issubset(df.columns):
            df["weekday_vs_sunday_prog"] = df["prev_weekday_ProgActP"] - df["prev_sunday_ProgActP"]
        
        # Previous cycle activity × time effects
        if {"prev_weekday_ProgActP", "hour_of_week_sin"}.issubset(df.columns):
            df["prev_cycle_progress_x_hour"] = df["prev_weekday_ProgActP"] * df["hour_of_week_sin"]
            
        return df
    
    def _add_extreme_event_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interactions specifically designed for extreme event prediction"""
        
        # Compressed range × volatility for tail events
        if {"compressed_range_vol", "vol_gkyz_3h"}.issubset(df.columns):
            df["extreme_range_vol"] = df["compressed_range_vol"] * df["vol_gkyz_3h"]
        
        # Skewness × volatility for asymmetric moves
        if {"returns_skew_24h", "vol_gkyz_6h"}.issubset(df.columns):
            df["skew_vol_extreme"] = df["returns_skew_24h"] * df["vol_gkyz_6h"]
        
        # Kurtosis × volatility for fat-tail events
        if {"returns_kurtosis_24h", "vol_gkyz_12h"}.issubset(df.columns):
            df["kurtosis_vol_extreme"] = df["returns_kurtosis_24h"] * df["vol_gkyz_12h"]
        
        # Distance from highs × volatility for reversal prediction
        if {"dist_from_high_144h", "vol_gkyz_24h"}.issubset(df.columns):
            df["distance_vol_extreme"] = df["dist_from_high_144h"] * df["vol_gkyz_24h"]
        
        # Volume surprise × volatility clustering
        if {"volume_surprise", "vol_clustering"}.issubset(df.columns):
            df["vol_surprise_clustering"] = df["volume_surprise"] * df["vol_clustering"]
            
        return df
