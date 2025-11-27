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
            # SAFER REFERENCE PRICE HANDLING
            try:
                reference_price = row['prev_close']
                # Handle case where reference_price might be a Series (duplicate columns)
                if isinstance(reference_price, pd.Series):
                    if reference_price.empty or reference_price.isna().all() or (reference_price <= 1e-9).all():
                        continue
                    reference_price = reference_price.iloc[0]  # Take first value
                elif pd.isna(reference_price) or reference_price <= 1e-9:
                    continue
            except KeyError:
                continue  # Skip if reference price not available
    
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
                    # Basic price targets

                    results['max_p1'] = prices1_h.max()
                    results['min_p1'] = prices1_l.min()
                    results['exp1_max_ret'] = (prices1_h.max() / reference_price) - 1.0
                    results['exp1_min_ret'] = (prices1_l.min() / reference_price) - 1.0

                    # Peak/trough timing
                    max_idx = prices1_h.idxmax()
                    min_idx = prices1_l.idxmin()
                    window_length = (exp1_ts - current_bar_end).total_seconds() / 3600.0
                    peak_offset = (max_idx - current_bar_end).total_seconds() / 3600.0
                    trough_offset = (min_idx - current_bar_end).total_seconds() / 3600.0
                    results['exp1_peak_frac'] = peak_offset / window_length if window_length > 0 else np.nan
                    results['exp1_trough_frac'] = trough_offset / window_length if window_length > 0 else np.nan
                    results['exp1_peak_hours_to_expiry'] = (exp1_ts - max_idx).total_seconds() / 3600.0
                    results['exp1_trough_hours_to_expiry'] = (exp1_ts - min_idx).total_seconds() / 3600.0

                    # Close return
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
        """
        For each expiry window, compute the absolute peak/trough and their times.
        Assign these to all rows in the window.
        Also computes absolute min/max for the following expiry window (exp2).
        """
        timestamps = X.index
        prices_c = X['c']
        prices_h = X['h']
        prices_l = X['l']
        prev_close = X['prev_close']

        # Map each timestamp to its expiry window start and end
        exp_map = {ts: self._get_expiration_timestamps(ts) for ts in timestamps}
        exp1_times = pd.Series({ts: exp[0] for ts, exp in exp_map.items()})
        exp2_times = pd.Series({ts: exp[1] for ts, exp in exp_map.items()})
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
            # Use the close from the row immediately before the window start (even if exp_start is not in index)
            prev_idx = X.index.get_indexer([exp_start], method='ffill')[0] - 1
            if prev_idx >= 0:
                prev_close_ts = X.index[prev_idx]
                window_prev_close = X.loc[prev_close_ts, 'c']
            else:
                window_prev_close = np.nan

            if window_h.empty or pd.isna(window_prev_close) or window_prev_close <= 1e-9:
                continue

            # abs_targets.loc[window_idx, 'window_prev_close'] = window_prev_close

            # Absolute peak/trough and their times for expiry 1
            abs_max = window_h.max()
            abs_min = window_l.min()
            abs_max_ret = (abs_max / window_prev_close) - 1.0
            abs_min_ret = (abs_min / window_prev_close) - 1.0
            max_idx = window_h.idxmax()
            min_idx = window_l.idxmin()
            window_length = (exp_end - exp_start).total_seconds() / 3600.0
            peak_offset = (max_idx - exp_start).total_seconds() / 3600.0
            trough_offset = (min_idx - exp_start).total_seconds() / 3600.0
            abs_peak_frac = peak_offset / window_length if window_length > 0 else np.nan
            abs_trough_frac = trough_offset / window_length if window_length > 0 else np.nan
            abs_peak_hours_to_expiry = (exp_end - max_idx).total_seconds() / 3600.0
            abs_trough_hours_to_expiry = (exp_end - min_idx).total_seconds() / 3600.0

            # Absolute mean reversion targets for expiry 1
            if not window_c.empty:
                close_at_expiry = window_c.iloc[-1]
                abs_peak_to_close_ret = (close_at_expiry / abs_max) - 1.0 if abs_max > 1e-9 else np.nan
                abs_trough_to_close_ret = (close_at_expiry / abs_min) - 1.0 if abs_min > 1e-9 else np.nan
            else:
                abs_peak_to_close_ret = np.nan
                abs_trough_to_close_ret = np.nan

            # abs_targets.loc[window_idx, 'window_prev_close'] = window_prev_close
            abs_targets.loc[window_idx, 'abs_max_p1'] = abs_max
            abs_targets.loc[window_idx, 'abs_min_p1'] = abs_min
            abs_targets.loc[window_idx, 'abs_exp1_max_ret'] = abs_max_ret
            abs_targets.loc[window_idx, 'abs_exp1_min_ret'] = abs_min_ret
            abs_targets.loc[window_idx, 'abs_exp1_peak_frac'] = abs_peak_frac
            abs_targets.loc[window_idx, 'abs_exp1_trough_frac'] = abs_trough_frac
            abs_targets.loc[window_idx, 'abs_exp1_peak_hours_to_expiry'] = abs_peak_hours_to_expiry
            abs_targets.loc[window_idx, 'abs_exp1_trough_hours_to_expiry'] = abs_trough_hours_to_expiry
            abs_targets.loc[window_idx, 'abs_exp1_peak_to_close_ret'] = abs_peak_to_close_ret
            abs_targets.loc[window_idx, 'abs_exp1_trough_to_close_ret'] = abs_trough_to_close_ret

            # --- NEW: Absolute min/max for the following expiry window (exp2) ---
            # Find exp2 window for this expiry
            exp2_start = exp_end
            exp2_end = exp2_start + pd.Timedelta(days=1)
            mask2 = (timestamps >= exp2_start) & (timestamps < exp2_end)
            window2_idx = timestamps[mask2]
            window2_h = prices_h[mask2]
            window2_l = prices_l[mask2]
            window2_c = prices_c[mask2]

            if not window2_h.empty:
                abs_max_p2 = window2_h.max()
                abs_min_p2 = window2_l.min()
                abs_exp2_max_ret = (abs_max_p2 / window_prev_close) - 1.0
                abs_exp2_min_ret = (abs_min_p2 / window_prev_close) - 1.0
                max2_idx = window2_h.idxmax()
                min2_idx = window2_l.idxmin()
                # Mean reversion for exp2
                if not window2_c.empty:
                    close_at_expiry2 = window2_c.iloc[-1]
                    abs_peak_to_close_ret2 = (close_at_expiry2 / abs_max_p2) - 1.0 if abs_max_p2 > 1e-9 else np.nan
                    abs_trough_to_close_ret2 = (close_at_expiry2 / abs_min_p2) - 1.0 if abs_min_p2 > 1e-9 else np.nan
                else:
                    abs_peak_to_close_ret2 = np.nan
                    abs_trough_to_close_ret2 = np.nan

                abs_targets.loc[window_idx, 'abs_max_p2'] = abs_max_p2
                abs_targets.loc[window_idx, 'abs_min_p2'] = abs_min_p2
                abs_targets.loc[window_idx, 'abs_exp2_max_ret'] = abs_exp2_max_ret
                abs_targets.loc[window_idx, 'abs_exp2_min_ret'] = abs_exp2_min_ret
                abs_targets.loc[window_idx, 'abs_exp2_peak_to_close_ret'] = abs_peak_to_close_ret2
                abs_targets.loc[window_idx, 'abs_exp2_trough_to_close_ret'] = abs_trough_to_close_ret2

        return abs_targets


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, '_input_features')
        df = X.copy()
    
        # --- Engineered logrets ---
        engineered_targets = [
            'logret_up_1h', 'logret_down_1h',
            'logret_up_3h', 'logret_down_3h',
            'logret_up_6h', 'logret_down_6h',
            'logret_up_12h', 'logret_down_12h',   
            'logret_up_24h', 'logret_down_24h',
            'next_24h_vol'    # Add realized volatility target
        ]
        
        # Engineered logrets: all compare future high/low to current close
        df['logret_up_1h'] = np.log(df['h'].shift(-1) / df['c'])
        df['logret_down_1h'] = np.log(df['l'].shift(-1) / df['c'])
        
        # For 3h: max high and min low over the next 3 hours (including the next hour, not current)
        df['logret_up_3h'] = np.log(df['h'].shift(-3).rolling(window=3, min_periods=3).max() / df['c'])
        df['logret_down_3h'] = np.log(df['l'].shift(-3).rolling(window=3, min_periods=3).min() / df['c'])
        
        # For 6h: max high and min low over the next 6 hours
        df['logret_up_6h'] = np.log(df['h'].shift(-6).rolling(window=6, min_periods=6).max() / df['c'])
        df['logret_down_6h'] = np.log(df['l'].shift(-6).rolling(window=6, min_periods=6).min() / df['c'])
    
            # For 12h: max high and min low over the next 12 hours
        df['logret_up_12h'] = np.log(df['h'].shift(-12).rolling(window=12, min_periods=12).max() / df['c'])
        df['logret_down_12h'] = np.log(df['l'].shift(-12).rolling(window=12, min_periods=12).min() / df['c'])

        # For 24h: max high and min low over the next 24 hours
        df['logret_up_24h'] = np.log(df['h'].shift(-24).rolling(window=24, min_periods=24).max() / df['c'])
        df['logret_down_24h'] = np.log(df['l'].shift(-24).rolling(window=24, min_periods=24).min() / df['c'])
        
        # --- Next 24-hour realized volatility ---
        # Calculate close-to-close log returns for the next 24 hours
        # This is the realized volatility that will occur over the next 24 hours
        safe_close = df['c'].replace(0, np.nan)
        future_log_returns = np.log(safe_close.shift(-1) / safe_close)
        # Rolling std of future returns over next 24 hours, annualized
        # Use shift(-24) to look forward, then rolling to get the window
        df['next_24h_vol'] = future_log_returns.shift(-1).rolling(window=24, min_periods=18).std() * np.sqrt(24 * 365)


        # --- Calculate price targets ---
        price_targets = self._calculate_price_targets(df)
        abs_targets = self._calculate_absolute_expiry_targets(df)
    
        # Merge all targets
        df_targets = pd.concat([price_targets, abs_targets, df[engineered_targets]], axis=1)
    
        self._feature_names_out = list(df_targets.columns)

        # Always filter, even if empty
        if self.targets_to_process is not None:
            df_targets = df_targets[[col for col in df_targets.columns if col in self.targets_to_process]]
        return df_targets       
    
    def get_feature_names_out(self, input_features=None):
        if self._feature_names_out is None:
            raise NotFittedError("Call 'transform' first.")
        return np.array(self._feature_names_out)


class VolatilityRegimeEngineer(BaseEstimator, TransformerMixin):
    """
    Creates volatility regime labels for classification:
    0 = Normal/Mean Reverting
    1 = Trending
    2 = Jump Risk
    
    Parameters
    ----------
    lookback_window : int, default=24
        Hours to look back for regime calculation
    
    forward_window : int, default=12
        Hours to look forward for regime labeling
        
    vol_percentile : float, default=75
        Percentile threshold for high volatility (jump risk)
        
    trend_threshold : float, default=0.6
        Threshold for directional consistency to identify trends
        
    jump_threshold_std : float, default=2.5
        Standard deviation threshold for identifying jumps
    """
    
    def __init__(self, 
                 lookback_window: int = 24,
                 forward_window: int = 12,
                 vol_percentile: float = 75,
                 trend_threshold: float = 0.6,
                 jump_threshold_std: float = 2.5):
        self.lookback_window = lookback_window
        self.forward_window = forward_window
        self.vol_percentile = vol_percentile
        self.trend_threshold = trend_threshold
        self.jump_threshold_std = jump_threshold_std
        self._feature_names_out = None
        self._vol_threshold = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit method. Calculates volatility thresholds from training data."""
        required_cols = ['c', 'h', 'l', 'o']
        if not all(col in X.columns for col in required_cols):
            missing = [col for col in required_cols if col not in X.columns]
            raise ValueError(f"Missing required columns: {missing}")
            
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("Input DataFrame must have a DatetimeIndex.")
        
        # Calculate realized volatility to set threshold
        safe_close = X['c'].replace(0, np.nan)
        log_returns = np.log(safe_close / safe_close.shift(1))
        realized_vol = log_returns.rolling(window=self.lookback_window, min_periods=int(self.lookback_window*0.75)).std()
        realized_vol_annualized = realized_vol * np.sqrt(24 * 365)
        
        # Store the percentile threshold for jump detection
        self._vol_threshold = np.nanpercentile(realized_vol_annualized.dropna(), self.vol_percentile)
        
        return self
    
    def _calculate_regime_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate features used for regime classification."""
        df = X.copy()
        safe_close = df['c'].replace(0, np.nan)
        
        # 1. Realized volatility (annualized)
        log_returns = np.log(safe_close / safe_close.shift(1))
        realized_vol = log_returns.rolling(window=self.lookback_window, 
                                          min_periods=int(self.lookback_window*0.75)).std()
        df['realized_vol'] = realized_vol * np.sqrt(24 * 365)
        
        # 2. High-Low range as % of close (volatility proxy)
        df['hl_range_pct'] = ((df['h'] - df['l']) / df['c']) * 100
        
        # 3. Directional consistency (trend indicator)
        # Positive if upward trend, negative if downward
        returns = log_returns.rolling(window=self.lookback_window, 
                                      min_periods=int(self.lookback_window*0.75)).apply(
            lambda x: (x > 0).sum() / len(x) if len(x) > 0 else np.nan
        )
        df['directional_consistency'] = returns
        
        # 4. Gap detection (for jump risk)
        # Gap between close and next open
        df['gap_size'] = np.abs(np.log(df['o'] / safe_close.shift(1)))
        
        # 5. Intrabar range expansion (jump indicator)
        # Compare current bar range to recent average
        avg_range = df['hl_range_pct'].rolling(window=self.lookback_window, 
                                               min_periods=int(self.lookback_window*0.75)).mean()
        df['range_expansion'] = df['hl_range_pct'] / avg_range
        
        # 6. Mean reversion indicator
        # Distance from moving average as % of std dev
        ma = safe_close.rolling(window=self.lookback_window, 
                               min_periods=int(self.lookback_window*0.75)).mean()
        std = safe_close.rolling(window=self.lookback_window, 
                                min_periods=int(self.lookback_window*0.75)).std()
        df['zscore'] = (safe_close - ma) / std
        
        # 7. Return to mean measure (for forward window)
        # How much price reverts after reaching extremes
        df['max_deviation_fwd'] = df['h'].shift(-self.forward_window).rolling(
            window=self.forward_window, min_periods=int(self.forward_window*0.75)).max()
        df['min_deviation_fwd'] = df['l'].shift(-self.forward_window).rolling(
            window=self.forward_window, min_periods=int(self.forward_window*0.75)).min()
        
        return df
    
    def _assign_regime_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Assign regime labels based on forward-looking characteristics.
        
        Returns
        -------
        pd.Series with regime labels: 0=Normal/Mean Reverting, 1=Trending, 2=Jump Risk
        """
        labels = pd.Series(index=df.index, dtype='Int64')
        
        # Calculate forward-looking metrics for labeling
        safe_close = df['c'].replace(0, np.nan)
        
        # Forward returns and volatility
        fwd_returns = np.log(safe_close.shift(-self.forward_window) / safe_close)
        fwd_high = df['h'].shift(-self.forward_window).rolling(
            window=self.forward_window, min_periods=int(self.forward_window*0.5)).max()
        fwd_low = df['l'].shift(-self.forward_window).rolling(
            window=self.forward_window, min_periods=int(self.forward_window*0.5)).min()
        
        # Forward realized volatility
        fwd_vol = np.log(safe_close.shift(-1) / safe_close).shift(-self.forward_window).rolling(
            window=self.forward_window, min_periods=int(self.forward_window*0.5)).std() * np.sqrt(24 * 365)
        
        # Forward range expansion
        fwd_range_pct = ((fwd_high - fwd_low) / safe_close) * 100
        
        # Forward directional consistency
        fwd_direction = np.log(safe_close.shift(-1) / safe_close).shift(-self.forward_window).rolling(
            window=self.forward_window, min_periods=int(self.forward_window*0.5)).apply(
            lambda x: np.abs((x > 0).sum() / len(x) - 0.5) if len(x) > 0 else np.nan
        )
        
        # Mean reversion measure: does price return to starting level?
        fwd_close = safe_close.shift(-self.forward_window)
        price_reversion = np.abs(np.log(fwd_close / safe_close))
        
        # Check for large jumps in forward window
        fwd_gaps = df['gap_size'].shift(-self.forward_window).rolling(
            window=self.forward_window, min_periods=1).max()
        fwd_range_expansion = df['range_expansion'].shift(-self.forward_window).rolling(
            window=self.forward_window, min_periods=1).max()
        
        # REGIME CLASSIFICATION LOGIC
        for idx in df.index:
            if pd.isna(fwd_vol.loc[idx]) or pd.isna(fwd_direction.loc[idx]):
                labels.loc[idx] = pd.NA
                continue
            
            # JUMP RISK (Priority 1): High volatility or large gaps/range expansion
            if (fwd_vol.loc[idx] > self._vol_threshold or 
                fwd_gaps.loc[idx] > self.jump_threshold_std * df['gap_size'].std() or
                fwd_range_expansion.loc[idx] > self.jump_threshold_std):
                labels.loc[idx] = 2
                
            # TRENDING (Priority 2): Strong directional consistency
            elif fwd_direction.loc[idx] > self.trend_threshold:
                labels.loc[idx] = 1
                
            # NORMAL/MEAN REVERTING (Default): Lower volatility with reversion
            else:
                labels.loc[idx] = 0
        
        return labels
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate regime labels and supporting features.
        
        Returns
        -------
        pd.DataFrame with regime label and intermediate features
        """
        check_is_fitted(self, '_vol_threshold')
        
        # Calculate regime features
        df_features = self._calculate_regime_features(X)
        
        # Assign regime labels
        regime_labels = self._assign_regime_labels(df_features)
        
        # Create output dataframe
        output = pd.DataFrame(index=X.index)
        output['regime_label'] = regime_labels
        
        # Add useful features for model training
        output['regime_realized_vol'] = df_features['realized_vol']
        output['regime_hl_range_pct'] = df_features['hl_range_pct']
        output['regime_directional_consistency'] = df_features['directional_consistency']
        output['regime_gap_size'] = df_features['gap_size']
        output['regime_range_expansion'] = df_features['range_expansion']
        output['regime_zscore'] = df_features['zscore']
        
        self._feature_names_out = list(output.columns)
        
        return output
    
    def get_feature_names_out(self, input_features=None):
        if self._feature_names_out is None:
            raise NotFittedError("Call 'transform' first.")
        return np.array(self._feature_names_out)
    
    def get_regime_distribution(self, X: pd.DataFrame) -> pd.Series:
        """
        Get the distribution of regimes in the dataset.
        
        Returns
        -------
        pd.Series with counts for each regime
        """
        transformed = self.transform(X)
        return transformed['regime_label'].value_counts().sort_index()