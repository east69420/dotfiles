# targetEngineer.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

class ExpirationTargetEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, expiration_hour: int = 8, targets_to_process=None):
        self.targets_to_process = targets_to_process
        self.expiration_hour = expiration_hour
        self._feature_names_out = None
        self._input_features = None

    def fit(self, X: pd.DataFrame, y=None):
        """Fit method. Validates input and stores column names."""
        required_cols = ['c', 'h', 'l', 'prev_close',]
        if not all(col in X.columns for col in required_cols):
            missing = [col for col in required_cols if col not in X.columns]
            raise ValueError(f"Missing required columns: {missing}")
            
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("Input DataFrame must have a DatetimeIndex.")

        self._input_features = X.columns
        return self

    def _get_expiration_timestamps(self, current_ts: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Calculate expiration timestamps."""
        if current_ts.hour < self.expiration_hour:
            exp1 = current_ts.normalize().replace(hour=self.expiration_hour)
        else:
            exp1 = (current_ts + pd.Timedelta(days=1)).normalize().replace(hour=self.expiration_hour)
        exp2 = exp1 + pd.Timedelta(days=1)
        return exp1, exp2

    def _calculate_price_targets(self, X):
        """Helper method to compute all price targets without tail indicators"""
        timestamps = X.index
        prices_c = X['c']
        prices_h = X['h']
        prices_l = X['l']

        # Pre-calculate expiration timestamps aligned to the bar close (current_ts + 1h)
        one_hour = pd.Timedelta(hours=1)
        exp_map = {ts: self._get_expiration_timestamps(ts + one_hour) for ts in timestamps}
        exp1_times = pd.Series({ts: exp[0] for ts, exp in exp_map.items()})

        targets_df = pd.DataFrame(index=timestamps, dtype=float)
        
        for current_ts, row in X.iterrows():
            try:
                reference_price = row['prev_close']
                if isinstance(reference_price, pd.Series):
                    if reference_price.empty or reference_price.isna().all() or (reference_price <= 1e-9).all():
                        continue
                    reference_price = reference_price.iloc[0]
                elif pd.isna(reference_price) or reference_price <= 1e-9:
                    continue
            except KeyError:
                continue
    
            exp1_ts = exp1_times.get(current_ts)
            results = {}

            if exp1_ts:
                current_bar_end = current_ts + one_hour
                if current_bar_end >= exp1_ts:
                    continue

                mask1 = (timestamps >= current_bar_end) & (timestamps < exp1_ts)
                prices1_h = prices_h[mask1]
                prices1_l = prices_l[mask1]
                
                if not prices1_h.empty:
                    results['max_p1'] = prices1_h.max()
                    results['min_p1'] = prices1_l.min()
                    results['exp1_max_ret'] = (prices1_h.max() / reference_price) - 1.0
                    results['exp1_min_ret'] = (prices1_l.min() / reference_price) - 1.0

                    max_idx = prices1_h.idxmax()
                    min_idx = prices1_l.idxmin()
                    window_length = (exp1_ts - current_bar_end).total_seconds() / 3600.0
                    peak_offset = (max_idx - current_bar_end).total_seconds() / 3600.0
                    trough_offset = (min_idx - current_bar_end).total_seconds() / 3600.0
                    results['exp1_peak_frac'] = peak_offset / window_length if window_length > 0 else np.nan
                    results['exp1_trough_frac'] = trough_offset / window_length if window_length > 0 else np.nan
                    results['exp1_peak_hours_to_expiry'] = (exp1_ts - max_idx).total_seconds() / 3600.0
                    results['exp1_trough_hours_to_expiry'] = (exp1_ts - min_idx).total_seconds() / 3600.0

                    try:
                        close_idx_arr = prices_c.index.get_indexer([exp1_ts], method='ffill')
                        if close_idx_arr[0] != -1:
                            close_idx = close_idx_arr[0]
                            found_ts = prices_c.index[close_idx]
                            if found_ts <= exp1_ts and found_ts >= current_ts:
                                results['exp1_close_ret'] = (prices_c.iloc[close_idx] / reference_price) - 1.0
                    except KeyError:
                        pass
                    
                    for k, v in results.items():
                        targets_df.at[current_ts, k] = v

        return targets_df

    def _calculate_absolute_expiry_targets(self, X):
        timestamps = X.index
        prices_c = X['c']
        prices_h = X['h']
        prices_l = X['l']
        
        exp_map = {ts: self._get_expiration_timestamps(ts) for ts in timestamps}
        exp1_times = pd.Series({ts: exp[0] for ts, exp in exp_map.items()})
        exp1_starts = exp1_times.shift(1, fill_value=exp1_times.iloc[0] - pd.Timedelta(days=1))
        exp1_ends = exp1_times

        abs_targets = pd.DataFrame(index=timestamps, dtype=float)

        for exp_start, exp_end in sorted(set(zip(exp1_starts, exp1_ends))):
            mask = (timestamps >= exp_start) & (timestamps < exp_end)
            window_idx = timestamps[mask]
            if len(window_idx) == 0:
                continue
            window_h = prices_h[mask]
            window_l = prices_l[mask]
            window_c = prices_c[mask]
            
            prev_idx = X.index.get_indexer([exp_start], method='ffill')[0] - 1
            if prev_idx >= 0:
                prev_close_ts = X.index[prev_idx]
                window_prev_close = X.loc[prev_close_ts, 'c']
            else:
                window_prev_close = np.nan

            if window_h.empty or pd.isna(window_prev_close) or window_prev_close <= 1e-9:
                continue

            abs_max = window_h.max()
            abs_min = window_l.min()
            abs_max_ret = (abs_max / window_prev_close) - 1.0
            abs_min_ret = (abs_min / window_prev_close) - 1.0
            max_idx = window_h.idxmax()
            min_idx = window_l.idxmin()
            window_length = (exp_end - exp_start).total_seconds() / 3600.0
            peak_offset = (max_idx - exp_start).total_seconds() / 3600.0
            trough_offset = (min_idx - exp_start).total_seconds() / 3600.0
            
            if not window_c.empty:
                close_at_expiry = window_c.iloc[-1]
                abs_peak_to_close_ret = (close_at_expiry / abs_max) - 1.0 if abs_max > 1e-9 else np.nan
                abs_trough_to_close_ret = (close_at_expiry / abs_min) - 1.0 if abs_min > 1e-9 else np.nan
            else:
                abs_peak_to_close_ret = np.nan
                abs_trough_to_close_ret = np.nan

            abs_targets.loc[window_idx, 'abs_max_p1'] = abs_max
            abs_targets.loc[window_idx, 'abs_min_p1'] = abs_min
            abs_targets.loc[window_idx, 'abs_exp1_max_ret'] = abs_max_ret
            abs_targets.loc[window_idx, 'abs_exp1_min_ret'] = abs_min_ret
            abs_targets.loc[window_idx, 'abs_exp1_peak_frac'] = peak_offset / window_length if window_length > 0 else np.nan
            abs_targets.loc[window_idx, 'abs_exp1_trough_frac'] = trough_offset / window_length if window_length > 0 else np.nan
            abs_targets.loc[window_idx, 'abs_exp1_peak_hours_to_expiry'] = (exp_end - max_idx).total_seconds() / 3600.0
            abs_targets.loc[window_idx, 'abs_exp1_trough_hours_to_expiry'] = (exp_end - min_idx).total_seconds() / 3600.0
            abs_targets.loc[window_idx, 'abs_exp1_peak_to_close_ret'] = abs_peak_to_close_ret
            abs_targets.loc[window_idx, 'abs_exp1_trough_to_close_ret'] = abs_trough_to_close_ret

            # Exp2 logic
            exp2_start = exp_end
            exp2_end = exp2_start + pd.Timedelta(days=1)
            mask2 = (timestamps >= exp2_start) & (timestamps < exp2_end)
            window2_h = prices_h[mask2]
            window2_l = prices_l[mask2]
            window2_c = prices_c[mask2]

            if not window2_h.empty:
                abs_max_p2 = window2_h.max()
                abs_min_p2 = window2_l.min()
                if not window2_c.empty:
                    close_at_expiry2 = window2_c.iloc[-1]
                    abs_peak_to_close_ret2 = (close_at_expiry2 / abs_max_p2) - 1.0 if abs_max_p2 > 1e-9 else np.nan
                    abs_trough_to_close_ret2 = (close_at_expiry2 / abs_min_p2) - 1.0 if abs_min_p2 > 1e-9 else np.nan
                else:
                    abs_peak_to_close_ret2 = np.nan
                    abs_trough_to_close_ret2 = np.nan

                abs_targets.loc[window_idx, 'abs_max_p2'] = abs_max_p2
                abs_targets.loc[window_idx, 'abs_min_p2'] = abs_min_p2
                abs_targets.loc[window_idx, 'abs_exp2_max_ret'] = (abs_max_p2 / window_prev_close) - 1.0
                abs_targets.loc[window_idx, 'abs_exp2_min_ret'] = (abs_min_p2 / window_prev_close) - 1.0
                abs_targets.loc[window_idx, 'abs_exp2_peak_to_close_ret'] = abs_peak_to_close_ret2
                abs_targets.loc[window_idx, 'abs_exp2_trough_to_close_ret'] = abs_trough_to_close_ret2

        return abs_targets

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, '_input_features')
        df = X.copy()
    
        engineered_targets = [
            'logret_up_1h', 'logret_down_1h',
            'logret_up_3h', 'logret_down_3h',
            'logret_up_6h', 'logret_down_6h',
            'logret_up_12h', 'logret_down_12h',   
            'logret_up_24h', 'logret_down_24h',
            'next_24h_vol'
        ]
        
        df['logret_up_1h'] = np.log(df['h'].shift(-1) / df['c'])
        df['logret_down_1h'] = np.log(df['l'].shift(-1) / df['c'])
        df['logret_up_3h'] = np.log(df['h'].shift(-3).rolling(window=3, min_periods=3).max() / df['c'])
        df['logret_down_3h'] = np.log(df['l'].shift(-3).rolling(window=3, min_periods=3).min() / df['c'])
        df['logret_up_6h'] = np.log(df['h'].shift(-6).rolling(window=6, min_periods=6).max() / df['c'])
        df['logret_down_6h'] = np.log(df['l'].shift(-6).rolling(window=6, min_periods=6).min() / df['c'])
        df['logret_up_12h'] = np.log(df['h'].shift(-12).rolling(window=12, min_periods=12).max() / df['c'])
        df['logret_down_12h'] = np.log(df['l'].shift(-12).rolling(window=12, min_periods=12).min() / df['c'])
        df['logret_up_24h'] = np.log(df['h'].shift(-24).rolling(window=24, min_periods=24).max() / df['c'])
        df['logret_down_24h'] = np.log(df['l'].shift(-24).rolling(window=24, min_periods=24).min() / df['c'])
        
        safe_close = df['c'].replace(0, np.nan)
        future_log_returns = np.log(safe_close.shift(-1) / safe_close)
        df['next_24h_vol'] = future_log_returns.shift(-1).rolling(window=24, min_periods=18).std() * np.sqrt(24 * 365)

        price_targets = self._calculate_price_targets(df)
        abs_targets = self._calculate_absolute_expiry_targets(df)
    
        df_targets = pd.concat([price_targets, abs_targets, df[engineered_targets]], axis=1)
    
        self._feature_names_out = list(df_targets.columns)

        if self.targets_to_process is not None:
            df_targets = df_targets[[col for col in df_targets.columns if col in self.targets_to_process]]
        return df_targets       
    
    def get_feature_names_out(self, input_features=None):
        if self._feature_names_out is None:
            raise NotFittedError("Call 'transform' first.")
        return np.array(self._feature_names_out)


class VolatilityRegimeEngineer(BaseEstimator, TransformerMixin):
    """
    Classifies market regimes using deseasonalized volatility normalization.
    
    Parameters
    ----------
    lookback_window : int, default=24
        Hours to look back for calculating current volatility condition.
        
    seasonal_window : int, default=720
        Hours of history to use for learning seasonal volatility patterns (training).
    
    forward_window : int, default=24
        Hours to look forward for regime classification.
        
    trend_std : float, default=1.2
        The Z-Score (Forward Window Units) required to classify a Trend.
        Example: 1.2 means "Did price move 1.2 Daily Sigmas?"
        
    jump_std : float, default=1.6
        The Z-Score (Forward Window Units) required to classify a Jump... BUT achieved in jump_speed_window.
        Example: 1.6 means "Did price move 1.6 Daily Sigmas... within 3 hours?"
        Note: If jump_std > trend_std, Jumps are strictly larger events than Trends.
        
    jump_speed_window : int, default=3
        Sub-window (hours) to detect fast shock moves.
        
    retracement_threshold : float, default=0.5
        Max retracement from extreme to still qualify as trending.
    """
    
    def __init__(self, 
                 lookback_window: int = 24,
                 seasonal_window: int = 720,
                 forward_window: int = 24,
                 trend_std: float = 1.2,
                 jump_std: float = 1.6,
                 jump_speed_window: int = 3,
                 retracement_threshold: float = 0.5):
        self.lookback_window = lookback_window
        self.seasonal_window = seasonal_window
        self.forward_window = forward_window
        self.trend_std = trend_std
        self.jump_std = jump_std
        self.jump_speed_window = jump_speed_window
        self.retracement_threshold = retracement_threshold
        
        self._feature_names_out = None
        self._seasonal_vol_lookup = None
        self.global_vol_median = None
        
    def fit(self, X: pd.DataFrame, y=None):
        required_cols = ['c', 'h', 'l']
        if not all(col in X.columns for col in required_cols):
            missing = [col for col in required_cols if col not in X.columns]
            raise ValueError(f"Missing required columns: {missing}")
            
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("Input DataFrame must have a DatetimeIndex.")
        
        # 1. Learn Seasonal Patterns
        self._seasonal_vol_lookup = self._build_seasonal_vol_lookup(X)
        
        # 2. Learn Global Volatility (The Scalar)
        self.global_vol_median = self._seasonal_vol_lookup.median()
        
        return self
    
    def _build_seasonal_vol_lookup(self, X: pd.DataFrame) -> pd.DataFrame:
        safe_open = X['o'].replace(0, np.nan) if 'o' in X.columns else X['c'].replace(0, np.nan)
        safe_high = X['h'].replace(0, np.nan)
        safe_low = X['l'].replace(0, np.nan)
        safe_close = X['c'].replace(0, np.nan)
        
        # Rogers-Satchell per-bar variance
        term1 = np.log(safe_high / safe_close) * np.log(safe_high / safe_open)
        term2 = np.log(safe_low / safe_close) * np.log(safe_low / safe_open)
        rs_variance = (term1 + term2).clip(lower=0)
        rs_vol = np.sqrt(rs_variance)
        
        vol_df = pd.DataFrame({
            'vol': rs_vol,
            'hour': X.index.hour,
            'day_of_week': X.index.dayofweek
        }, index=X.index)
        
        # Robust Median Seasonality (Group by H/D)
        # We need to compute rolling medians per group
        seasonal_vol = vol_df.groupby(['hour', 'day_of_week'])['vol'].rolling(
            window=max(4, self.seasonal_window // 168), 
            min_periods=4
        ).median().reset_index(level=[0, 1], drop=True)
        
        lookup = vol_df.copy()
        lookup['seasonal_vol'] = seasonal_vol
        
        # Return the last known median for each bucket
        return lookup.groupby(['hour', 'day_of_week'])['seasonal_vol'].last()
    
    def _calculate_lookback_box(self, X: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=X.index)
        
        safe_open = X['o'].replace(0, np.nan) if 'o' in X.columns else X['c'].replace(0, np.nan)
        safe_high = X['h'].replace(0, np.nan)
        safe_low = X['l'].replace(0, np.nan)
        safe_close = X['c'].replace(0, np.nan)
        
        # 1. Raw Rolling Rogers-Satchell
        term1 = np.log(safe_high / safe_close) * np.log(safe_high / safe_open)
        term2 = np.log(safe_low / safe_close) * np.log(safe_low / safe_open)
        raw_variance = (term1 + term2).clip(lower=0)
        
        raw_vol = np.sqrt(raw_variance.rolling(
            window=self.lookback_window, 
            min_periods=int(self.lookback_window * 0.5)
        ).mean())
        
        # 2. Map Seasonality to Timestamp
        # (Optimized lookup)
        time_keys = pd.Series(list(zip(X.index.hour, X.index.dayofweek)), index=X.index)
        
        if self._seasonal_vol_lookup is not None:
            seasonal_vol_series = time_keys.map(self._seasonal_vol_lookup)
            # Fill unknown times with global median
            seasonal_vol_series = seasonal_vol_series.fillna(self.global_vol_median)
        else:
            seasonal_vol_series = pd.Series(self.global_vol_median, index=X.index)

        # 3. Calculate "Adjusted" Volatility
        # Ratio = Raw / Seasonal (Unitless Strength)
        # Adjusted = Ratio * Global_Median (Back to Volatility Units)
        deseasonalized_ratio = raw_vol / seasonal_vol_series.replace(0, np.nan)
        adjusted_vol = deseasonalized_ratio * self.global_vol_median
        
        df['box_std'] = adjusted_vol
        df['raw_vol'] = raw_vol
        df['seasonal_vol'] = seasonal_vol_series
        df['reference_price'] = safe_close
        
        return df
    
    def _assign_regime_labels(self, X: pd.DataFrame, box_df: pd.DataFrame) -> pd.DataFrame:
        results = pd.DataFrame(index=X.index)
        results['regime_label'] = pd.Series(dtype='Int64')
        results['max_fwd_z_score'] = np.nan
        results['max_jump_z_score'] = np.nan
        
        # ------------------------------------------------------------
        # TIME NORMALIZATION (Crucial Step)
        # Convert Jump Threshold (Daily Units) -> Short Window Units
        # If Fwd=24, JumpWin=3 -> Scale = sqrt(8) = 2.83
        # ------------------------------------------------------------
        time_scaling_factor = np.sqrt(self.forward_window / self.jump_speed_window)
        REAL_JUMP_THRESHOLD = self.jump_std * time_scaling_factor
        
        safe_close = X['c'].replace(0, np.nan)
        high = X['h']
        low = X['l']
        
        sqrt_fwd = np.sqrt(self.forward_window)
        sqrt_jump = np.sqrt(self.jump_speed_window)
        
        for idx in X.index:
            box_std = box_df.loc[idx, 'box_std']
            ref_price = box_df.loc[idx, 'reference_price']
            
            if pd.isna(box_std) or box_std <= 1e-9 or ref_price <= 0:
                continue
                
            idx_pos = X.index.get_loc(idx)
            fwd_end_pos = min(idx_pos + self.forward_window, len(X.index))
            
            if fwd_end_pos <= idx_pos + 1:
                continue
            
            fwd_indices = X.index[idx_pos+1:fwd_end_pos]
            fwd_highs = high.loc[fwd_indices]
            fwd_lows = low.loc[fwd_indices]
            
            if fwd_highs.empty:
                continue

            # --- Forward Window Metrics ---
            max_high = fwd_highs.max()
            min_low = fwd_lows.min()
            final_close = X.iloc[fwd_end_pos-1]['c']
            
            ret_max = np.log(max_high / ref_price)
            ret_min = np.log(min_low / ref_price)
            max_abs_ret = max(abs(ret_max), abs(ret_min))
            
            # 1. Forward Z-Score (Daily Sigmas)
            fwd_z_score = max_abs_ret / (box_std * sqrt_fwd)
            results.at[idx, 'max_fwd_z_score'] = fwd_z_score
            
            # 2. Jump Window Metrics
            max_jump_z = 0.0
            for i in range(len(fwd_indices) - self.jump_speed_window + 1):
                sub_indices = fwd_indices[i : i + self.jump_speed_window]
                s_high = high.loc[sub_indices].max()
                s_low = low.loc[sub_indices].min()
                s_ret = max(abs(np.log(s_high/ref_price)), abs(np.log(s_low/ref_price)))
                
                # Short Window Z-Score
                current_z = s_ret / (box_std * sqrt_jump)
                if current_z > max_jump_z:
                    max_jump_z = current_z
            
            results.at[idx, 'max_jump_z_score'] = max_jump_z

            # --- CLASSIFICATION ---
            
            # Check JUMP (Using Scaled Threshold)
            if max_jump_z > REAL_JUMP_THRESHOLD:
                results.at[idx, 'regime_label'] = 2
                continue
            
            # Check TREND
            if fwd_z_score > self.trend_std:
                # Retracement check
                if ret_max > abs(ret_min): # Uptrend
                    dist_to_high = np.log(max_high / final_close)
                    if dist_to_high / ret_max <= (1 - self.retracement_threshold):
                        results.at[idx, 'regime_label'] = 1
                else: # Downtrend
                    dist_to_low = np.log(final_close / min_low)
                    if dist_to_low / abs(ret_min) <= (1 - self.retracement_threshold):
                        results.at[idx, 'regime_label'] = 1
                        
                if not pd.isna(results.at[idx, 'regime_label']):
                    continue
            
            # Check CHOP
            results.at[idx, 'regime_label'] = 0
        
        return results
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, '_seasonal_vol_lookup')
        
        box_df = self._calculate_lookback_box(X)
        results = self._assign_regime_labels(X, box_df)
        
        output = results.copy()
        output['box_std_deseasonalized'] = box_df['box_std']
        output['box_std_raw'] = box_df['raw_vol']
        output['seasonal_vol'] = box_df['seasonal_vol']
        
        self._feature_names_out = list(output.columns)
        return output
    
    def get_feature_names_out(self, input_features=None):
        if self._feature_names_out is None:
            raise NotFittedError("Call 'transform' first.")
        return np.array(self._feature_names_out)
    
    def get_regime_distribution(self, X: pd.DataFrame) -> pd.Series:
        transformed = self.transform(X)
        regime_counts = transformed['regime_label'].value_counts().sort_index()
        regime_names = {
            0: 'Class 0: Chop/Mean Reversion',
            1: 'Class 1: Trending',
            2: 'Class 2: Jump/Event'
        }
        regime_counts.index = regime_counts.index.map(lambda x: regime_names.get(x, f'Class {x}'))
        return regime_counts