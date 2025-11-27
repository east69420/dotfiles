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
        required_cols = ['c', 'h', 'l', 'prev_close',]
        if not all(col in X.columns for col in required_cols):
            missing = [col for col in required_cols if col not in X.columns]
            raise ValueError(f"Missing required columns: {missing}")
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("Input DataFrame must have a DatetimeIndex.")
        self._input_features = X.columns
        return self

    def _get_expiration_timestamps(self, current_ts: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        if current_ts.hour < self.expiration_hour:
            exp1 = current_ts.normalize().replace(hour=self.expiration_hour)
        else:
            exp1 = (current_ts + pd.Timedelta(days=1)).normalize().replace(hour=self.expiration_hour)
        exp2 = exp1 + pd.Timedelta(days=1)
        return exp1, exp2

    def _calculate_price_targets(self, X):
        timestamps = X.index
        prices_c = X['c']
        prices_h = X['h']
        prices_l = X['l']
        one_hour = pd.Timedelta(hours=1)
        exp_map = {ts: self._get_expiration_timestamps(ts + one_hour) for ts in timestamps}
        exp1_times = pd.Series({ts: exp[0] for ts, exp in exp_map.items()})
        targets_df = pd.DataFrame(index=timestamps, dtype=float)
        
        for current_ts, row in X.iterrows():
            try:
                reference_price = row['prev_close']
                if isinstance(reference_price, pd.Series):
                    reference_price = reference_price.iloc[0]
                if pd.isna(reference_price) or reference_price <= 1e-9: continue
            except KeyError: continue
    
            exp1_ts = exp1_times.get(current_ts)
            results = {}
            if exp1_ts:
                current_bar_end = current_ts + one_hour
                if current_bar_end >= exp1_ts: continue
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
                    if window_length > 0:
                        results['exp1_peak_frac'] = (max_idx - current_bar_end).total_seconds() / 3600.0 / window_length
                        results['exp1_trough_frac'] = (min_idx - current_bar_end).total_seconds() / 3600.0 / window_length
                    
                    results['exp1_peak_hours_to_expiry'] = (exp1_ts - max_idx).total_seconds() / 3600.0
                    results['exp1_trough_hours_to_expiry'] = (exp1_ts - min_idx).total_seconds() / 3600.0

                    try:
                        close_idx_arr = prices_c.index.get_indexer([exp1_ts], method='ffill')
                        if close_idx_arr[0] != -1:
                            close_idx = close_idx_arr[0]
                            found_ts = prices_c.index[close_idx]
                            if found_ts <= exp1_ts and found_ts >= current_ts:
                                results['exp1_close_ret'] = (prices_c.iloc[close_idx] / reference_price) - 1.0
                    except KeyError: pass
                    for k, v in results.items(): targets_df.at[current_ts, k] = v
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
            if len(window_idx) == 0: continue
            
            window_h = prices_h[mask]
            window_l = prices_l[mask]
            window_c = prices_c[mask]
            
            prev_idx = X.index.get_indexer([exp_start], method='ffill')[0] - 1
            if prev_idx < 0: continue
            window_prev_close = X.iloc[prev_idx]['c']
            if pd.isna(window_prev_close) or window_prev_close <= 1e-9 or window_h.empty: continue

            abs_max = window_h.max()
            abs_min = window_l.min()
            
            # Exp1 Targets
            abs_targets.loc[window_idx, 'abs_max_p1'] = abs_max
            abs_targets.loc[window_idx, 'abs_min_p1'] = abs_min
            abs_targets.loc[window_idx, 'abs_exp1_max_ret'] = (abs_max / window_prev_close) - 1.0
            abs_targets.loc[window_idx, 'abs_exp1_min_ret'] = (abs_min / window_prev_close) - 1.0
            
            # Exp2 Targets
            exp2_start = exp_end
            exp2_end = exp2_start + pd.Timedelta(days=1)
            mask2 = (timestamps >= exp2_start) & (timestamps < exp2_end)
            window2_h = prices_h[mask2]
            window2_l = prices_l[mask2]
            
            if not window2_h.empty:
                abs_max_p2 = window2_h.max()
                abs_min_p2 = window2_l.min()
                abs_targets.loc[window_idx, 'abs_max_p2'] = abs_max_p2
                abs_targets.loc[window_idx, 'abs_min_p2'] = abs_min_p2
                abs_targets.loc[window_idx, 'abs_exp2_max_ret'] = (abs_max_p2 / window_prev_close) - 1.0
                abs_targets.loc[window_idx, 'abs_exp2_min_ret'] = (abs_min_p2 / window_prev_close) - 1.0

        return abs_targets

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, '_input_features')
        df = X.copy()
        
        # Engineered logrets
        for i in [1, 3, 6, 12, 24]:
            df[f'logret_up_{i}h'] = np.log(df['h'].shift(-i).rolling(window=i, min_periods=i).max() / df['c'])
            df[f'logret_down_{i}h'] = np.log(df['l'].shift(-i).rolling(window=i, min_periods=i).min() / df['c'])
        
        safe_close = df['c'].replace(0, np.nan)
        future_log_returns = np.log(safe_close.shift(-1) / safe_close)
        df['next_24h_vol'] = future_log_returns.shift(-1).rolling(window=24, min_periods=18).std() * np.sqrt(24 * 365)

        price_targets = self._calculate_price_targets(df)
        abs_targets = self._calculate_absolute_expiry_targets(df)
        
        # Select engineered columns
        eng_cols = [c for c in df.columns if 'logret_' in c or c == 'next_24h_vol']
        df_targets = pd.concat([price_targets, abs_targets, df[eng_cols]], axis=1)
    
        self._feature_names_out = list(df_targets.columns)
        if self.targets_to_process is not None:
            df_targets = df_targets[[col for col in df_targets.columns if col in self.targets_to_process]]
        return df_targets       
    
    def get_feature_names_out(self, input_features=None):
        if self._feature_names_out is None: raise NotFittedError("Call 'transform' first.")
        return np.array(self._feature_names_out)


class VolatilityRegimeEngineer(BaseEstimator, TransformerMixin):
    """
    Classifies market regimes using deseasonalized volatility and path efficiency.
    """
    def __init__(self, 
                 lookback_window: int = 24,
                 seasonal_window: int = 720,
                 forward_window: int = 24,
                 trend_std: float = 1.2,
                 jump_std: float = 1.6,
                 jump_speed_window: int = 3,
                 retracement_threshold: float = 0.5,
                 trend_min_efficiency: float = 0.25, # NEW
                 trend_min_r2: float = 0.6           # NEW
                 ):
        self.lookback_window = lookback_window
        self.seasonal_window = seasonal_window
        self.forward_window = forward_window
        self.trend_std = trend_std
        self.jump_std = jump_std
        self.jump_speed_window = jump_speed_window
        self.retracement_threshold = retracement_threshold
        self.trend_min_efficiency = trend_min_efficiency
        self.trend_min_r2 = trend_min_r2
        
        self._feature_names_out = None
        self._seasonal_vol_lookup = None
        self.global_vol_median = None
        
    def fit(self, X: pd.DataFrame, y=None):
        required_cols = ['c', 'h', 'l']
        if not all(col in X.columns for col in required_cols):
            raise ValueError(f"Missing cols: {required_cols}")
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("Index must be DatetimeIndex")
            
        self._seasonal_vol_lookup = self._build_seasonal_vol_lookup(X)
        self.global_vol_median = self._seasonal_vol_lookup.median()
        return self
    
    def _build_seasonal_vol_lookup(self, X: pd.DataFrame) -> pd.DataFrame:
        safe_open = X['o'].replace(0, np.nan) if 'o' in X.columns else X['c'].replace(0, np.nan)
        safe_high = X['h'].replace(0, np.nan)
        safe_low = X['l'].replace(0, np.nan)
        safe_close = X['c'].replace(0, np.nan)
        
        term1 = np.log(safe_high / safe_close) * np.log(safe_high / safe_open)
        term2 = np.log(safe_low / safe_close) * np.log(safe_low / safe_open)
        rs_vol = np.sqrt((term1 + term2).clip(lower=0))
        
        vol_df = pd.DataFrame({'vol': rs_vol, 'h': X.index.hour, 'd': X.index.dayofweek}, index=X.index)
        
        seasonal_vol = vol_df.groupby(['h', 'd'])['vol'].rolling(
            window=max(4, self.seasonal_window // 168), min_periods=4
        ).median().reset_index(level=[0, 1], drop=True)
        
        lookup = vol_df.copy()
        lookup['seasonal_vol'] = seasonal_vol
        return lookup.groupby(['h', 'd'])['seasonal_vol'].last()
    
    def _calculate_lookback_box(self, X: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=X.index)
        safe_open = X['o'].replace(0, np.nan) if 'o' in X.columns else X['c'].replace(0, np.nan)
        safe_high = X['h'].replace(0, np.nan)
        safe_low = X['l'].replace(0, np.nan)
        safe_close = X['c'].replace(0, np.nan)
        
        term1 = np.log(safe_high / safe_close) * np.log(safe_high / safe_open)
        term2 = np.log(safe_low / safe_close) * np.log(safe_low / safe_open)
        raw_vol = np.sqrt(((term1 + term2).clip(lower=0)).rolling(
            window=self.lookback_window, min_periods=int(self.lookback_window/2)).mean())
        
        time_keys = pd.Series(list(zip(X.index.hour, X.index.dayofweek)), index=X.index)
        seasonal_vol_series = time_keys.map(self._seasonal_vol_lookup).fillna(self.global_vol_median)

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
        
        # Scale Jump threshold to be consistent with Daily units
        time_scaling_factor = np.sqrt(self.forward_window / self.jump_speed_window)
        REAL_JUMP_THRESHOLD = self.jump_std * time_scaling_factor
        
        safe_close = X['c'].replace(0, np.nan)
        high = X['h']
        low = X['l']
        sqrt_fwd = np.sqrt(self.forward_window)
        sqrt_jump = np.sqrt(self.jump_speed_window)
        
        # Pre-calc 1-period moves for Efficiency
        abs_diffs = np.abs(np.log(safe_close / safe_close.shift(1)))
        
        for idx in X.index:
            box_std = box_df.loc[idx, 'box_std']
            ref_price = box_df.loc[idx, 'reference_price']
            
            # If box_std is NaN or too small (flat period), assign chop (0) and continue
            if pd.isna(box_std) or box_std <= 1e-9 or ref_price <= 0:
                results.at[idx, 'regime_label'] = 0
                results.at[idx, 'max_fwd_z_score'] = 0.0
                results.at[idx, 'max_jump_z_score'] = 0.0
                continue
            
            idx_pos = X.index.get_loc(idx)
            fwd_end_pos = min(idx_pos + self.forward_window, len(X.index))
            if fwd_end_pos <= idx_pos + 1: continue
            
            fwd_indices = X.index[idx_pos+1:fwd_end_pos]
            fwd_highs = high.loc[fwd_indices]
            fwd_lows = low.loc[fwd_indices]
            if fwd_highs.empty: continue

            max_high = fwd_highs.max()
            min_low = fwd_lows.min()
            final_close = X.iloc[fwd_end_pos-1]['c']
            
            ret_max = np.log(max_high / ref_price)
            ret_min = np.log(min_low / ref_price)
            max_abs_ret = max(abs(ret_max), abs(ret_min))
            
            fwd_z_score = max_abs_ret / (box_std * sqrt_fwd)
            results.at[idx, 'max_fwd_z_score'] = fwd_z_score
            
            # Check JUMP
            max_jump_z = 0.0
            for i in range(len(fwd_indices) - self.jump_speed_window + 1):
                sub_indices = fwd_indices[i : i + self.jump_speed_window]
                s_high = high.loc[sub_indices].max()
                s_low = low.loc[sub_indices].min()
                s_ret = max(abs(np.log(s_high/ref_price)), abs(np.log(s_low/ref_price)))
                current_z = s_ret / (box_std * sqrt_jump)
                if current_z > max_jump_z: max_jump_z = current_z
            results.at[idx, 'max_jump_z_score'] = max_jump_z

            if max_jump_z > REAL_JUMP_THRESHOLD:
                results.at[idx, 'regime_label'] = 2
                continue
            
            # Check TREND with Hardening
            if fwd_z_score > self.trend_std:
                # 1. Efficiency
                total_path = abs_diffs.loc[fwd_indices].sum()
                net_move = abs(np.log(final_close / ref_price))
                efficiency = net_move / total_path if total_path > 1e-9 else 0.0
                
                # 2. Linearity (R2)
                fwd_prices = safe_close.loc[fwd_indices].values
                if len(fwd_prices) > 2:
                    corr = np.corrcoef(fwd_prices, np.arange(len(fwd_prices)))[0, 1]
                    r2 = corr**2
                else: r2 = 0.0
                
                if efficiency >= self.trend_min_efficiency and r2 >= self.trend_min_r2:
                    # Retracement (Persistence)
                    if ret_max > abs(ret_min): # Up
                        if np.log(max_high / final_close) / ret_max <= (1 - self.retracement_threshold):
                            results.at[idx, 'regime_label'] = 1
                    else: # Down
                        if np.log(final_close / min_low) / abs(ret_min) <= (1 - self.retracement_threshold):
                            results.at[idx, 'regime_label'] = 1
                
                if not pd.isna(results.at[idx, 'regime_label']): continue
            
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
        if self._feature_names_out is None: raise NotFittedError("Call 'transform' first.")
        return np.array(self._feature_names_out)
    
    def get_regime_distribution(self, X: pd.DataFrame) -> pd.Series:
        transformed = self.transform(X)
        regime_counts = transformed['regime_label'].value_counts().sort_index()
        return regime_counts