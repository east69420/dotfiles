import os
import warnings
import pandas as pd
from pathlib import Path
import glob

# --- Configuration ---
CONFIG = {
    # --- Data Configuration ---
    "data": {
        "path": ".hist_db_1h.csv",
        "slice_start": 0,
        "slice_end": -1,
        "force_refresh": True,
        "symbol": "btcusd",
        "frequency": "1h",
    },
    
    # --- Feature Engineering ---
    "features": {
        "params": {
            "include_price": True,
            "include_trend": True,
            "include_volatility": True,
            "include_temporal": True,
            "include_liquidity": True,
            "include_custom_interactions": True,
            "include_non_linear": True,
            "include_prev_week_cycle": True,
            "vol_types_to_calc": ['raw', 'gkyz', 'skew', 'parkinson', 'kurtosis', 'vol_zscore', 'log_vol'],
            "vol_window_sizes": [3, 6, 12, 24, 72, 144, 288],
            "vlm_window_sizes": [3, 6, 12, 24, 72, 144, 288],
            "verbose": False
        },
        "heavy_cache": {
            "enabled": True,
            "directory": "cache/heavy_features",
            "key_fields": ["symbol", "frequency"],
        },
        "caching": {
            "enabled": True,
            "directory": "feature_cache",
            "compression": True,
            "validation": True
        },
        "derived_probability_features": {
            "enabled": False,
            "models_directory": "bs_focus/trained_models",  # Directory to scan
            "file_pattern": "catboost_*_hpt_bundle.pkl",  # Pattern to match
            "models_to_use": {}  # This will be populated automatically
        }
    },

    # --- Target Configuration ---
    "targets": {
        "params": {},
        "to_process": {
            # absolute peakstroughs (not time depenedent)
            'window_prev_close': True, # <-- added to store previous close for price calculations
            # "max_p1": True,
            # "max_p2": True,
            # "abs_exp1_max_ret": [0.50, 0.60],
            # "abs_exp1_min_ret": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
            # "abs_exp1_peak_to_close_ret": [0.50, 0.90], # mean revesion
            # "abs_exp1_trough_to_close_ret": [0.10, 0.50], # mean reversion

            # # # --- NEW: absolute min/max for following expiry window (exp2) ---
            # "abs_exp2_max_ret": [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
            # "abs_exp2_min_ret": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
            # "abs_exp2_peak_to_close_ret": [0.50, 0.90],
            # "abs_exp2_trough_to_close_ret": [0.10, 0.50],

            ## we need to target volatility first
            "next_24h_vol": True,
            
            # LEVEL I: logrets hourly
            # "logret_up_12h": [0.80],
            # "logret_up_1h": [0.25, 0.50, 0.70, 0.80, 0.95],
            # "logret_down_1h": [0.05, 0.20, 0.30, 0.50, 0.75],
            # "logret_up_3h": [0.25, 0.50, 0.70, 0.80, 0.95],
            # "logret_down_3h": [0.05, 0.20, 0.30, 0.50, 0.75],
            # "logret_up_6h": [0.25, 0.50, 0.70, 0.80, 0.95],
            # "logret_down_6h": [0.05, 0.20, 0.30, 0.50, 0.75],
            # "logret_up_12h": [0.25, 0.50, 0.70, 0.80, 0.95],    # <-- added
            # "logret_down_12h": [0.05, 0.20, 0.30, 0.50, 0.75], # <-- added
            # "logret_up_24h": [0.25, 0.50, 0.70, 0.80, 0.95],   # <-- added
            # "logret_down_24h": [0.05, 0.20, 0.30, 0.50, 0.75], # <-- added

            # ## LEVEL II: expiry forecasts 
            # "exp1_max_ret": [0.80],
            # "exp1_min_ret": [0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97],
            # "exp1_max_ret": [0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97],
            # "exp1_close_ret": [0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97],

            ## light version
            # "exp1_min_ret": [ 0.25, 0.50, 0.75],
            # "exp1_max_ret":  [ 0.25, 0.50, 0.75],
            # "exp1_close_ret":  [ 0.25, 0.50, 0.75],
 



            # "exp1_peak_hours_to_expiry": [0.30, 0.50, 0.70],
            # "exp1_trough_hours_to_expiry": [0.30, 0.50, 0.70],


            ## we need a model for volaitlity and predicting futrue vol.


            ## we need a model for longer term price movements, e.g uses S&P500, gold, other markets, etc.
            ## we need a model that tracks onchain stuff, peraps hasrate or anything else useful.
            ## we do the standard logret models
            ## we use those as basemodels for final price models


        },
    },
    
    # --- Data Splitting ---
    "splitting": {
        "train_pct": 0.7,
        "val_pct": 0.2,
        "time_based": True
    },

    # --- Feature Selection ---
    "feature_selection": {
        "run_RFECV": True,
        "use_all_features": False,
        "rfecv_engine": "catboost",  # Use 'catboost' for RFECV
        #"target_percentile": 0.5,   # <--- ADD THIS LINE (0.5 for 50th, 0.95 for 95th, etc.)
        "handpicked_filter": {
            "enabled": False,
            "file": "handpicked_logret_filter.csv",
            "feature_column": "feature",
        },

        "rfecv_settings": {
            "n_cv_splits": 5,   # This is usually 5
            "scoring_metric": "neg_mean_squared_error",
            "step": 3,  # higher prunes faster and avoids overfitting
            "min_features": 5
        },
        "estimator_params": {  # Params for CatBoost RFECV estimator
            "loss_function": "RMSE",
            "iterations": 50,         # Number of boosting rounds for RFECV
            "learning_rate": 0.05,     # Reasonable default for RFECV
            "depth": 6,
            "l2_leaf_reg": 3,
            "random_seed": 42,
            "thread_count": -1,
            "verbose": True
        }
    },
    
    # --- Model Training ---
    "model": {
        "engine": "catboost", # 'lightgbm', 'xgboost', or 'catboost'
        "early_stopping_rounds": 20, # CatBoost also supports this
        "eval_frequency": 10,      # For CatBoost, logging_level controls this, or use custom callback
        "verbose_eval": 10,        # CatBoost: verbose (int), logging_level ('Silent', 'Verbose', 'Info', 'Debug')
        "use_pipeline": False,
        "include_scaler": False,
        "params": {
            "lightgbm": {
                'objective': 'huber',         
                'metric': 'l2',                    # (or "l1", "rmse", "mae")
                'n_estimators': 500, # May need re-tuning via HPT
                'learning_rate': 0.01, # Placeholder, HPT will find better
                'feature_fraction': 0.7, # Placeholder
                'bagging_fraction': 0.7, # Placeholder
                'bagging_freq': 5,       # Placeholder
                'lambda_l1': 0.1,        # Placeholder
                'lambda_l2': 0.1,        # Placeholder
                'num_leaves': 31,        # Placeholder
                'min_data_in_leaf': 20,  # Still relevant
                # 'max_depth': -1, # Default, let it grow unless HPT tunes it
                'verbose': -1,
                'n_jobs': -1,
                'seed': 42,
                'boosting_type': 'gbdt'
                # REMOVE scale_pos_weight, class_weight, is_unbalance from HPT results / final params
            },
            "xgboost": { # Already configured for regression, might be okay
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse', # Often paired with squarederror, or 'mae'
                'n_estimators': 1000, # HPT should tune
                'learning_rate': 0.5, # Placeholder
                'max_depth': 5,        # Placeholder
                'subsample': 0.8,      # Placeholder
                'colsample_bytree': 0.8, # Placeholder
                'gamma': 0,            # Placeholder
                'reg_alpha': 0.0,      # Placeholder
                'reg_lambda': 1.0,     # Placeholder
                'verbosity': 0,
                'n_jobs': -1, # Use -1 for all cores if desired
                'seed': 42
                # REMOVE scale_pos_weight from HPT results / final params
            },

            "catboost": { # NEW SECTION FOR CATBOOST
                #'loss_function': 'Quantile:alpha=0.5',      # Use Quantile for asymmetric loss
                #'eval_metric': 'Quantile:alpha=0.5',  # Optional: match eval_metric to loss
                'iterations': 500,
                'learning_rate': 0.02,
                'depth': 10,
                'l2_leaf_reg': 3,
                'border_count': 64,
                'random_seed': 42,
                # 'verbose': 100,
                'thread_count': -1,
                # 'early_stopping_rounds': 20,
                # CatBoost specific params to consider for HPT:
                # 'bagging_temperature': 1,
                # 'random_strength': 1,
                # 'one_hot_max_size': 2,
                # 'leaf_estimation_method': 'Newton' or 'Gradient'
            },
        },

    },
    # --- Hyperparameter Tuning ---
    "hpt": {
        "enabled": True,
        "n_trials": 2000, # Or more for regression (use 200 for production)
        "timeout_seconds": None,
        "use_weighted_loss": True, # Re-evaluate if still desired for regression
        "time_to_expiry_col": 'time_to_exp1_hr', # if use_weighted_loss is True
        "weighting": { # if use_weighted_loss is True
            "method": 'linear',
            "max_multiplier": 5.0
        },
        "storage": {
            "db": None,
            "save_studies": True
        },
    },
    
    # --- Output Configuration ---
    "output": {
        "directory": "research_vol", # NAME DIRECORY TO USE HERE
        "subdirectories": {
            "features": "feature_selection",
            "models": "trained_models",
            "hpt": "hpt_studies",
            "cache": "feature_cache"
        },
        "save_predictions": True
    },

    # --- Stages Configuration ---
    "stages": {
        "stage1": {
            "target_prefixes": ["logret_up_", "logret_down_"],
            # "feature_lists": {
            #     "logret_up": "top10.json",
            #     "logret_down": "top10.json",
            # },
        },
        "stage2": {
            "target_prefixes": ["exp1_", "exp2_"],
            "feature_file": None,
        },
        "stage3": {
            "target_prefixes": [],
            "feature_file": None,
            },
        }
    }

# --- Auto-discovery helpers ---
def _stage_to_int(stage_token):
    """Convert stage tokens like 'stage1' into deterministic integers."""
    if stage_token is None:
        return 0
    if isinstance(stage_token, (int, float)):
        return int(stage_token)
    digits = "".join(ch for ch in str(stage_token) if ch.isdigit())
    return int(digits) if digits else 0


def _target_priority(target_type):
    """Assign a priority so level-one (logret) models sort ahead of others."""
    if not target_type:
        return 99
    lowered = target_type.lower()
    if lowered.startswith("logret"):
        return 0
    if lowered.startswith("exp1"):
        return 1
    return 2


# --- Auto-discovery function for trained models ---
def auto_discover_models(models_directory, file_pattern="catboost_*_hpt_bundle.pkl"):
    """
    Automatically discover all trained model files and create the models_to_use dict.
    
    Args:
        models_directory (str): Directory containing the trained models
        file_pattern (str): Glob pattern to match model files
    
    Returns:
        dict: Dictionary mapping model names to their file paths
    """
    models_to_use = {}
    feature_name_counts = {}

    # Create full path pattern
    full_pattern = os.path.join(models_directory, file_pattern)

    # Find all matching files
    model_files = glob.glob(full_pattern)

    print(f"🔍 Scanning for models in: {models_directory}")
    print(f"📋 Pattern: {file_pattern}")
    print(f"✅ Found {len(model_files)} model files")

    discovered_entries = []

    for file_path in model_files:
        filename = os.path.basename(file_path)
        raw_model_name = filename.replace("catboost_", "").replace("_hpt_bundle.pkl", "")

        path_str = str(Path(file_path).resolve())

        tokens = raw_model_name.split('_')
        stage = next((token for token in tokens if token.lower().startswith("stage")), None)
        horizon = next((token for token in tokens if token.endswith("h")), None)
        percentile_token = next((token for token in tokens if token.startswith('p') and token[1:].isdigit()), None)
        percentile = int(percentile_token[1:]) if percentile_token else None

        target_type = "_".join(tokens[:2]) if len(tokens) >= 2 else tokens[0]
        filtered_tokens = [token for token in tokens if token.lower() != (stage.lower() if stage else "")]
        filtered_tokens = [token for token in filtered_tokens if not token.lower().startswith("stage")]
        feature_base_name = "_".join(filtered_tokens) if filtered_tokens else raw_model_name

        discovered_entries.append({
            "path": path_str,
            "raw_model_name": raw_model_name,
            "stage": stage,
            "stage_num": _stage_to_int(stage),
            "target_type": target_type,
            "target_priority": _target_priority(target_type),
            "horizon": horizon,
            "horizon_value": int(horizon[:-1]) if horizon and horizon[:-1].isdigit() else None,
            "percentile": percentile,
            "feature_base_name": feature_base_name,
        })

    discovered_entries.sort(
        key=lambda entry: (
            entry["stage_num"],
            entry["target_priority"],
            entry["horizon_value"] if entry["horizon_value"] is not None else -1,
            entry["percentile"] if entry["percentile"] is not None else -1,
            entry["feature_base_name"],
        )
    )

    for entry in discovered_entries:
        base_feature_name = f"model_{entry['feature_base_name']}"
        count = feature_name_counts.get(base_feature_name, 0)
        if count > 0:
            feature_name = f"{base_feature_name}_v{count + 1}"
        else:
            feature_name = base_feature_name
        feature_name_counts[base_feature_name] = count + 1

        models_to_use[feature_name] = {
            "model_artifact_path": entry["path"],
            "stage": entry["stage"],
            "target_type": entry["target_type"],
            "horizon": entry["horizon"],
            "percentile": entry["percentile"],
            "feature_name": feature_name,
            "feature_base_name": entry["feature_base_name"],
            "source_model_name": entry["raw_model_name"],
        }

        print(f"   📦 {feature_name} -> {entry['path']} (stage: {entry['stage']})")

    return models_to_use

# --- Auto-populate the models_to_use dictionary ---
def setup_derived_probability_features(config):
    """
    Setup derived probability features by auto-discovering available models.
    """
    features_config = config["features"]["derived_probability_features"]
    
    if features_config["enabled"]:
        models_directory = features_config["models_directory"]
        file_pattern = features_config["file_pattern"]
        
        # Auto-discover models
        discovered_models = auto_discover_models(models_directory, file_pattern)
        
        # Update the config
        features_config["models_to_use"] = discovered_models
        
        print(f"\n📊 Auto-discovery complete!")
        print(f"   Total models configured: {len(discovered_models)}")
        
        # Group by category for summary
        categories = {}
        for model_info in discovered_models.values():
            category = model_info.get("target_type", "unknown")
            categories.setdefault(category, 0)
            categories[category] += 1
        
        for category, count in categories.items():
            print(f"   {category}: {count} models")
    
    return config

# --- Initialize the configuration ---
CONFIG = setup_derived_probability_features(CONFIG)

# Derived settings
CONFIG["hpt"]["model_engine"] = CONFIG["model"]["engine"]

# --- Path Setup ---
current_output_root_str = CONFIG["output"]["directory"]
current_output_root_path = Path(current_output_root_str)

paths = {
    "root": current_output_root_path,
    "feature_selection": current_output_root_path / CONFIG["output"]["subdirectories"]["features"],
    "trained_models": current_output_root_path / CONFIG["output"]["subdirectories"]["models"],
    "hpt_studies": current_output_root_path / CONFIG["output"]["subdirectories"]["hpt"],
    "feature_cache": current_output_root_path / CONFIG["output"]["subdirectories"]["cache"]
}

# Create directories if they don't exist
for path_obj in paths.values():
    path_obj.mkdir(parents=True, exist_ok=True)

# --- Storage Dict ---
storage = {
    "models": {}, "scores": {}, "features": {}, "predictions": {},
    "metadata": {}, "selected_features": {}, "trained_models": {},
    "validation_scores": {}, "validation_predictions": {},
    "hpt_params": {}, "hpt_studies": {}
}

# --- Optional: Function to refresh models dynamically ---
def refresh_available_models():
    """
    Refresh the list of available models (useful if new models are trained).
    """
    global CONFIG
    CONFIG = setup_derived_probability_features(CONFIG)
    print("🔄 Model list refreshed!")

# --- Optional: Function to filter models by criteria ---
def filter_models_by_criteria(criteria_func=None, stage=None, target_type=None):
    """
    Filter available models by custom criteria.
    
    Args:
        criteria_func: Custom function that takes model_name and returns True/False
        stage: Filter by stage (e.g., "stage1", "stage2", "stage3")
        target_type: Filter by target type (e.g., "abs_exp1", "abs_exp2", "logret")
    
    Returns:
        dict: Filtered models dictionary
    """
    all_models = CONFIG["features"]["derived_probability_features"]["models_to_use"]
    filtered_models = {}
    
    for model_name, model_info in all_models.items():
        include = True

        # Apply stage filter
        if stage:
            known_stage = model_info.get("stage") if isinstance(model_info, dict) else None
            stage_match = (known_stage and stage == known_stage)
            if not stage_match:
                include = False

        # Apply target type filter
        if include and target_type:
            known_target = model_info.get("target_type") if isinstance(model_info, dict) else None
            sanitized_name = model_info.get("feature_base_name") if isinstance(model_info, dict) else model_name.split("::")[-1]
            target_match = (known_target and known_target.startswith(target_type)) or sanitized_name.startswith(target_type)
            if not target_match:
                include = False

        # Apply custom criteria function
        if include and criteria_func:
            try:
                include = bool(criteria_func(model_name, model_info))
            except TypeError:
                include = bool(criteria_func(model_name))

        if include:
            filtered_models[model_name] = model_info
    
    return filtered_models

# --- Example usage functions ---
def get_stage1_models():
    """Get only stage1 models"""
    return filter_models_by_criteria(stage="stage1")

def get_abs_exp1_models():
    """Get only abs_exp1 models"""
    return filter_models_by_criteria(target_type="abs_exp1")

def get_high_percentile_models():
    """Get only models with high percentiles (p80+)"""
    def high_percentile_filter(_, model_info):
        percentile = None
        if isinstance(model_info, dict):
            percentile = model_info.get("percentile")
        if percentile is None:
            return False
        return percentile >= 80
    
    return filter_models_by_criteria(criteria_func=high_percentile_filter)

# Print summary when module is loaded
if CONFIG["features"]["derived_probability_features"]["enabled"]:
    print(f"\n📋 SUMMARY OF AVAILABLE MODELS:")
    print(f"   Stage 1 models: {len(get_stage1_models())}")
    print(f"   abs_exp1 models: {len(get_abs_exp1_models())}")
    print(f"   High percentile models (p80+): {len(get_high_percentile_models())}")
