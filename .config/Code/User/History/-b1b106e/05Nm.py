# Standard Library Imports
import functools
import json
import time
import warnings
from pathlib import Path

import functools
import json
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Any

import catboost as cb
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_pipeline import load_data
from featureEngineer import FeatureEngineer
from ML_general_tools import *
from ML_general_tools import (
    calculate_weights,
    count_consecutive_nan_rows,
    optuna_objective_func_regression,
)
from ML_setup import CONFIG, paths, storage
from targetEngineer import ExpirationTargetEngineer
# --- Utility Functions ---
def debug_print(msg): print(f"\033[96m{msg}\033[0m")
def warn_print(msg): print(f"\033[93m{msg}\033[0m")
def error_print(msg): print(f"\033[91m{msg}\033[0m")

def get_percentiles(target):
    # Use CONFIG targets to_process if available
    from ML_setup import CONFIG
    to_process = CONFIG["targets"]["to_process"]
    if target in to_process:
        return to_process[target]


# --- Setup Output Paths ---
current_output_root_path = Path(CONFIG["output"]["directory"])
paths = {
    "root": current_output_root_path,
    "feature_selection": current_output_root_path / CONFIG["output"]["subdirectories"]["features"],
    "trained_models": current_output_root_path / CONFIG["output"]["subdirectories"]["models"],
    "hpt_studies": current_output_root_path / CONFIG["output"]["subdirectories"]["hpt"],
    "feature_cache": current_output_root_path / CONFIG["output"]["subdirectories"]["cache"]
}

for path_obj in paths.values():
    path_obj.mkdir(parents=True, exist_ok=True)

DERIVED_PROBA_CONFIG = CONFIG["features"].get("derived_probability_features", {})
if DERIVED_PROBA_CONFIG.get("enabled", False):
    print(
        "[INFO] Derived probability features enabled; will augment base dataset with pre-trained bundles once features are loaded."
    )

# --- Stage & Feature Configuration Helpers ---
TARGETS_TO_PROCESS = CONFIG["targets"]["to_process"]
STAGES_CONFIG = CONFIG.get("stages", {})


def _format_list_preview(items, *, max_items: int = 6) -> str:
    sequence = [str(item) for item in items if item is not None]
    if not sequence:
        return "none"
    preview = ", ".join(sequence[:max_items])
    if len(sequence) > max_items:
        preview += ", ..."
    return preview


def _load_feature_list(file_spec: str, default_dir: Path) -> tuple[list[str], Path]:
    resolved_path = Path(file_spec)
    if not resolved_path.is_absolute():
        resolved_path = default_dir / file_spec
    if not resolved_path.exists():
        raise FileNotFoundError(f"Feature list file not found: {resolved_path}")
    with resolved_path.open("r") as f:
        payload = json.load(f)
    features = payload.get("features", payload) if isinstance(payload, dict) else payload
    if isinstance(features, dict):
        features = list(features)
    if not isinstance(features, list):
        raise ValueError(f"Feature list in {resolved_path} must resolve to a list of column names.")
    return features, resolved_path


def _resolve_dir(path_str: str | None, default_dir: Path) -> Path:
    if not path_str:
        return default_dir

    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate

    output_candidate = current_output_root_path / candidate
    if output_candidate.exists():
        return output_candidate

    project_candidate = PROJECT_ROOT / candidate
    if project_candidate.exists():
        return project_candidate

    return output_candidate


def _is_model_target(target_name: str) -> bool:
    value = TARGETS_TO_PROCESS.get(target_name)
    return isinstance(value, (list, tuple)) and len(value) > 0


def _resolve_stage_targets(stage_cfg: dict) -> list[str]:
    if not stage_cfg:
        return []
    explicit_targets = [
        t for t in stage_cfg.get("targets", [])
        if t in TARGETS_TO_PROCESS and _is_model_target(t)
    ]
    prefixes = stage_cfg.get("target_prefixes", [])
    stage_targets: list[str] = []
    seen: set[str] = set()
    for target in explicit_targets:
        if target not in seen:
            stage_targets.append(target)
            seen.add(target)
    for target in TARGETS_TO_PROCESS:
        if target in seen or not _is_model_target(target):
            continue
        if any(target.startswith(prefix) for prefix in prefixes):
            stage_targets.append(target)
            seen.add(target)
    return stage_targets


def _load_target_specific_feature_list(target_name: str) -> tuple[list[str], Path | None]:
    if target_name in TARGET_FEATURE_LIST_CACHE:
        return TARGET_FEATURE_LIST_CACHE[target_name]

    if not ROBUST_FEATURE_SETS_DIR.exists():
        TARGET_FEATURE_LIST_CACHE[target_name] = ([], None)
        return TARGET_FEATURE_LIST_CACHE[target_name]

    candidate = ROBUST_FEATURE_SETS_DIR / f"robust_auto_{target_name}.csv"
    if candidate.exists():
        try:
            df = pd.read_csv(candidate)
        except Exception as exc:
            warn_print(
                f"[WARN] Failed to load target-specific feature list for '{target_name}' from {candidate}: {exc}"
            )
            TARGET_FEATURE_LIST_CACHE[target_name] = ([], None)
            return TARGET_FEATURE_LIST_CACHE[target_name]

        if df.empty or not df.columns.tolist():
            warn_print(
                f"[WARN] Target-specific feature list for '{target_name}' at {candidate} is empty."
            )
            TARGET_FEATURE_LIST_CACHE[target_name] = ([], None)
            return TARGET_FEATURE_LIST_CACHE[target_name]

        column_to_use = "feature" if "feature" in df.columns else df.columns[0]
        features = df[column_to_use].dropna().astype(str).tolist()
        features = [feat for feat in features if feat]
        TARGET_FEATURE_LIST_CACHE[target_name] = (features, candidate)
        print(
            f"[INFO] Target-specific feature list for '{target_name}' loaded from {candidate} "
            f"({len(features)} features)."
        )
        return TARGET_FEATURE_LIST_CACHE[target_name]

    TARGET_FEATURE_LIST_CACHE[target_name] = ([], None)
    return TARGET_FEATURE_LIST_CACHE[target_name]


def _format_percentile_suffix(percentile: float) -> str:
    value = int(round(percentile * 100))
    return f"p{value:02d}"


def _load_percentile_specific_feature_list(target_name: str, percentile: float) -> tuple[list[str], Path | None]:
    suffix = _format_percentile_suffix(percentile)
    cache_key = (target_name, suffix)
    if cache_key in PERCENTILE_FEATURE_LIST_CACHE:
        return PERCENTILE_FEATURE_LIST_CACHE[cache_key]

    feature_dir = paths.get("feature_selection")
    if not feature_dir:
        PERCENTILE_FEATURE_LIST_CACHE[cache_key] = ([], None)
        return PERCENTILE_FEATURE_LIST_CACHE[cache_key]

    candidates = [
        feature_dir / f"selected_features_rfecv_{target_name}_{suffix}.json",
        feature_dir / f"selected_features_{target_name}_{suffix}.json",
        feature_dir / f"features_{target_name}_{suffix}.json",
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with candidate.open("r") as handle:
                payload = json.load(handle)
        except Exception as exc:
            warn_print(
                f"[WARN] Failed to load percentile-specific feature list for '{target_name}_{suffix}' from {candidate}: {exc}"
            )
            continue

        features = payload.get("features", payload) if isinstance(payload, dict) else payload
        if isinstance(features, dict):
            features = list(features)
        if not isinstance(features, list):
            warn_print(
                f"[WARN] Percentile-specific feature list for '{target_name}_{suffix}' in {candidate} is invalid."
            )
            continue

        PERCENTILE_FEATURE_LIST_CACHE[cache_key] = (list(features), candidate)
        print(
            f"[INFO] Percentile-specific feature list for '{target_name}_{suffix}' loaded from {candidate} "
            f"({len(features)} features)."
        )
        return PERCENTILE_FEATURE_LIST_CACHE[cache_key]

    PERCENTILE_FEATURE_LIST_CACHE[cache_key] = ([], None)
    return PERCENTILE_FEATURE_LIST_CACHE[cache_key]


def _format_feature_source_label(source: Path | None, stage_label: str, note: str) -> str:
    if source:
        try:
            display_path = str(source.relative_to(PROJECT_ROOT))
        except ValueError:
            display_path = str(source)
    else:
        display_path = f"all available {stage_label.lower()} features"
    if note and note.lower() not in display_path.lower():
        return f"{display_path} [{note}]"
    return display_path


def _collect_stage_percentiles(stage_targets: list[str]) -> list[str]:
    unique: set[int] = set()
    for target in stage_targets:
        percentiles = get_percentiles(target) or []
        for pct in percentiles:
            unique.add(int(round(pct * 100)))
    return [f"p{value:02d}" for value in sorted(unique)]


def _infer_family(target_name: str, stage_cfg: dict) -> str:
    depth = stage_cfg.get("family_depth", 2)
    parts = target_name.split("_")
    if depth <= 0:
        depth = 1
    depth = min(depth, len(parts))
    return "_".join(parts[:depth]) if parts else target_name


def _load_feature_map(feature_map: dict[str, str], stage_cfg: dict, stage_label: str) -> tuple[dict[str, list[str]], dict[str, Path]]:
    if not feature_map:
        return {}, {}
    lists_dir = _resolve_dir(stage_cfg.get("feature_lists_dir"), FEATURE_SUBSET_LIST_DIR)
    feature_lists: dict[str, list[str]] = {}
    feature_sources: dict[str, Path] = {}
    for family, file_spec in feature_map.items():
        features, resolved_path = _load_feature_list(file_spec, lists_dir)
        feature_lists[family] = features
        feature_sources[family] = resolved_path
        print(
            f"[INFO] {stage_label} feature list for '{family}' loaded from {resolved_path} "
            f"({len(features)} features)."
        )
    return feature_lists, feature_sources


def _load_feature_file_from_stage(stage_cfg: dict, stage_label: str) -> tuple[list[str], Path | None]:
    feature_file = stage_cfg.get("feature_file")
    if not feature_file:
        return [], None
    base_dir = _resolve_dir(stage_cfg.get("feature_file_dir"), paths["feature_selection"])
    try:
        features, resolved_path = _load_feature_list(feature_file, base_dir)
        print(
            f"[INFO] {stage_label} feature list loaded from {resolved_path} "
            f"({len(features)} features)."
        )
        return features, resolved_path
    except FileNotFoundError:
        error_print(
            f"[ERROR] {stage_label} feature file '{feature_file}' not found in {base_dir}."
        )
        raise


def _print_pipeline_overview(stage_summaries: list[dict[str, Any]]) -> None:
    print("\n=== PIPELINE CONFIGURATION OVERVIEW ===")
    print(f"Output root: {current_output_root_path}")
    print(f"Percentile-specific feature lists: {paths['feature_selection']}")
    print(f"Robust target feature lists: {ROBUST_FEATURE_SETS_DIR} (exists={ROBUST_FEATURE_SETS_DIR.exists()})")
    print(f"Trained model bundles will be saved to: {paths['trained_models']}")

    for summary in stage_summaries:
        label = summary["label"]
        cfg = summary["cfg"]
        targets = summary["targets"]
        feature_lists: dict[str, list[str]] = summary["feature_lists"]
        feature_sources: dict[str, Path] = summary["feature_sources"]
        fallback_source: Path | None = summary["fallback_source"]
        allow_target_specific: bool = summary["allow_target_specific"]

        print(f"\n{label}:")
        print(f"  Target prefixes: {_format_list_preview(cfg.get('target_prefixes', []))}")
        print(f"  Targets ({len(targets)}): {_format_list_preview(targets, max_items=8)}")
        percentile_labels = _collect_stage_percentiles(targets)
        print(f"  Percentiles covered: {_format_list_preview(percentile_labels, max_items=12)}")

        if allow_target_specific:
            location_note = "available" if ROBUST_FEATURE_SETS_DIR.exists() else "configured (directory missing)"
            print(f"  Target-specific robust CSV lists: {location_note} -> {ROBUST_FEATURE_SETS_DIR}")
        else:
            print("  Target-specific robust CSV lists: disabled")

        if feature_lists:
            families = sorted(feature_lists.keys())
            unique_dirs = sorted({str(path.parent) for path in feature_sources.values() if isinstance(path, Path)})
            print(f"  Family feature lists: {len(families)} families ({_format_list_preview(families, max_items=8)})")
            print(f"    Source directories: {_format_list_preview(unique_dirs, max_items=3)}")
        elif fallback_source:
            print(f"  Stage feature file: {_format_feature_source_label(fallback_source, label, '')}")
        elif percentile_labels:
            print(f"  Percentile-specific feature lists directory: {paths['feature_selection']}")
            print("  Stage feature file: none (percentile overrides will be applied)")
        else:
            print("  Stage feature file: all available engineered features")

        if label == "Stage 2":
            print("  Derived feature inputs: Stage 1 model outputs (model_* columns)")
        elif label == "Stage 3":
            print("  Derived feature inputs: Stage 1 & Stage 2 model outputs")

# --- Feature Subset Artifacts ---
PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_INSIGHTS_DIR = PROJECT_ROOT / "improved_versions2" / "feature_selection" / "insights"
FEATURE_SUBSET_LIST_DIR = FEATURE_INSIGHTS_DIR / "subset_feature_lists"
# Resolve robust feature directory relative to configured output root
robust_feature_sets_subdir = CONFIG["output"].get("robust_feature_sets_dir", "robust_feature_sets")
robust_candidate = Path(robust_feature_sets_subdir)
if robust_candidate.is_absolute():
    ROBUST_FEATURE_SETS_DIR = robust_candidate
else:
    ROBUST_FEATURE_SETS_DIR = current_output_root_path / robust_candidate
TARGET_FEATURE_LIST_CACHE: dict[str, tuple[list[str], Path | None]] = {}
PERCENTILE_FEATURE_LIST_CACHE: dict[tuple[str, str], tuple[list[str], Path | None]] = {}

# --- Stage Configuration Overview (pre-data generation) ---
stage1_cfg = STAGES_CONFIG.get("stage1", {})
stage2_cfg = STAGES_CONFIG.get("stage2", {})
stage3_cfg = STAGES_CONFIG.get("stage3", {})

stage1_allow_target_specific = bool(stage1_cfg.get("allow_target_specific", False))
stage2_allow_target_specific = bool(stage2_cfg.get("allow_target_specific", False))
stage3_allow_target_specific = bool(stage3_cfg.get("allow_target_specific", False))

stage1_targets = _resolve_stage_targets(stage1_cfg)
stage2_targets = _resolve_stage_targets(stage2_cfg)
stage3_targets = _resolve_stage_targets(stage3_cfg)

stage1_feature_lists, stage1_feature_sources = _load_feature_map(stage1_cfg.get("feature_lists", {}), stage1_cfg, "Stage 1")
stage1_feature_list, stage1_feature_file_source = ([], None)
if not stage1_feature_lists:
    stage1_feature_list, stage1_feature_file_source = _load_feature_file_from_stage(stage1_cfg, "Stage 1")

stage2_feature_lists, stage2_feature_sources = _load_feature_map(stage2_cfg.get("feature_lists", {}), stage2_cfg, "Stage 2")
stage2_feature_list, stage2_feature_file_source = ([], None)
if not stage2_feature_lists:
    stage2_feature_list, stage2_feature_file_source = _load_feature_file_from_stage(stage2_cfg, "Stage 2")

stage3_feature_lists, stage3_feature_sources = _load_feature_map(stage3_cfg.get("feature_lists", {}), stage3_cfg, "Stage 3")
stage3_feature_list, stage3_feature_file_source = ([], None)
if not stage3_feature_lists:
    stage3_feature_list, stage3_feature_file_source = _load_feature_file_from_stage(stage3_cfg, "Stage 3")

stage_summaries = [
    {
        "label": "Stage 1",
        "cfg": stage1_cfg,
        "targets": stage1_targets,
        "feature_lists": stage1_feature_lists,
        "feature_sources": stage1_feature_sources,
        "fallback_source": stage1_feature_file_source,
        "allow_target_specific": stage1_allow_target_specific,
    },
    {
        "label": "Stage 2",
        "cfg": stage2_cfg,
        "targets": stage2_targets,
        "feature_lists": stage2_feature_lists,
        "feature_sources": stage2_feature_sources,
        "fallback_source": stage2_feature_file_source,
        "allow_target_specific": stage2_allow_target_specific,
    },
    {
        "label": "Stage 3",
        "cfg": stage3_cfg,
        "targets": stage3_targets,
        "feature_lists": stage3_feature_lists,
        "feature_sources": stage3_feature_sources,
        "fallback_source": stage3_feature_file_source,
        "allow_target_specific": stage3_allow_target_specific,
    },
]

_print_pipeline_overview(stage_summaries)

# --- Load or Generate Data ---
features_path = paths["root"] / "features.pkl"
targets_path = paths["root"] / "targets.pkl"
combined_df_path = paths["root"] / "combined_df.pkl"

if all(p.exists() for p in [features_path, targets_path, combined_df_path]) and not CONFIG["data"]["force_refresh"]:
    print("Loading features, targets, and combined_df from cache...")
    features = pickle.load(open(features_path, "rb"))
    targets = pickle.load(open(targets_path, "rb"))
    combined_df = pickle.load(open(combined_df_path, "rb"))
else:
    print("Generating features and targets from scratch...")
    df = load_data(CONFIG["data"]["path"])

    feature_params = dict(CONFIG["features"]["params"])
    verbose = feature_params.pop("verbose", False)

    feature_generator = FeatureEngineer(
        **feature_params,
    )


    feature_generator.fit_transform(df)
    features = feature_generator._reference_features.copy()

    target_engineer = ExpirationTargetEngineer(**CONFIG["targets"]["params"])
    target_engineer.fit(features)
    targets = target_engineer.transform(features)
    combined_df = pd.concat([features, targets], axis=1)
    drop_cols = [col for col in ['o', 'h', 'l', 'c'] if col in features.columns]
    if drop_cols: features = features.drop(columns=drop_cols)
    pickle.dump(features, open(features_path, "wb"))
    pickle.dump(targets, open(targets_path, "wb"))
    pickle.dump(combined_df, open(combined_df_path, "wb"))

if DERIVED_PROBA_CONFIG.get("enabled", False):
    print("\n=== DERIVED PROBABILITY FEATURE AUGMENTATION ===")
    print("[INFO] Adding derived probability features from pre-trained bundles before HPT.")
    features, newly_added_proba_features = generate_and_add_derived_probability_features(
        features.copy(),
        DERIVED_PROBA_CONFIG,
        paths,
        convert_prices=False,
    )
    if newly_added_proba_features:
        preview = ", ".join(newly_added_proba_features[:10])
        if len(newly_added_proba_features) > 10:
            preview += ", ..."
        print(f"[INFO] Added derived probability features ({len(newly_added_proba_features)}): {preview}")
        for feat in newly_added_proba_features:
            if feat in features.columns:
                n_nans = features[feat].isna().sum()
                if n_nans:
                    print(f"    [NaN check] {feat}: {n_nans} NaNs ({n_nans / len(features):.2%} of rows)")
    else:
        print("[WARN] Derived probability features were enabled but no columns were added.")
    combined_df = pd.concat([features, targets], axis=1)
else:
    print("[INFO] Derived probability features disabled; skipping bundle augmentation.")

# --- Intelligent Data Cleaning Based on Actual Engineering Parameters ---
def calculate_intelligent_nan_boundaries(config, features_df=None, targets_df=None):
    """
    Calculate intelligent NaN cleaning boundaries based on actual feature and target engineering parameters.
    
    Returns:
        dict: Contains 'start_boundary_hours' and 'end_boundary_hours' for intelligent cleaning
    """
    
    # === FEATURE ENGINEERING BACKWARD LOOKBACK REQUIREMENTS ===
    feature_params = config["features"]["params"]
    
    # Get maximum window sizes from vol and vlm calculations
    max_vol_window = max(feature_params.get("vol_window_sizes", [0]))
    max_vlm_window = max(feature_params.get("vlm_window_sizes", [0]))
    
    print(f"Max volatility window: {max_vol_window} hours")
    print(f"Max volume window: {max_vlm_window} hours")
    
    # Calculate maximum rolling window requirement
    max_rolling_window_hours = max(max_vol_window, max_vlm_window)
    
    # Add buffer for additional feature engineering requirements
    # Include previous week cycle features if enabled
    prev_week_buffer = 0
    if feature_params.get("include_prev_week_cycle", False):
        # Assume 30 weeks as mentioned in the engineering code
        prev_week_buffer = 30 * 7 * 24  # 30 weeks in hours
        print(f"Previous week cycle buffer: {prev_week_buffer} hours ({prev_week_buffer // (7*24)} weeks)")
    
    # Calculate total backward lookback requirement
    total_backward_hours = max(max_rolling_window_hours, prev_week_buffer)
    
    # Add safety margin for robust feature calculation
    safety_multiplier = 1.5  # More intelligent than the 3x used in hard-coded version
    start_boundary_hours = int(total_backward_hours * safety_multiplier)
    
    print(f"Calculated start boundary: {start_boundary_hours} hours ({start_boundary_hours // (30*24):.1f} months)")
    
    # === TARGET ENGINEERING FORWARD-LOOKING REQUIREMENTS ===
    
    # Check for target configuration
    target_config = config.get("targets", {})
    
    # Look for time-to-expiry configuration
    time_to_expiry_col = config.get("hpt", {}).get("time_to_expiry_col", "time_to_exp1_hr")
    
    # Estimate forward-looking requirements based on model naming patterns
    # From the model names, we can see exp1, exp2, and various hour horizons (1h, 3h, 6h, 12h, 24h)
    max_forward_hours = 0
    
    # Check derived model configurations for forward-looking periods
    derived_models = config.get("features", {}).get("derived_probability_features", {}).get("models_to_use", {})

    forward_horizons = []
    for model_name, model_info in derived_models.items():
        if isinstance(model_info, dict):
            horizon_token = model_info.get("horizon")
            target_type_token = model_info.get("target_type", "")
        else:
            horizon_token = None
            target_type_token = model_name

        if horizon_token and horizon_token.endswith('h') and horizon_token[:-1].isdigit():
            forward_horizons.append(int(horizon_token[:-1]))
        elif "_1h_" in model_name:
            forward_horizons.append(1)
        elif "_3h_" in model_name:
            forward_horizons.append(3)
        elif "_6h_" in model_name:
            forward_horizons.append(6)
        elif "_12h_" in model_name:
            forward_horizons.append(12)
        elif "_24h_" in model_name:
            forward_horizons.append(24)
        elif "exp1" in target_type_token or "exp1" in model_name:
            forward_horizons.append(168)
        elif "exp2" in target_type_token or "exp2" in model_name:
            forward_horizons.append(336)
    
    if forward_horizons:
        max_forward_hours = max(forward_horizons)
        print(f"Maximum forward-looking requirement: {max_forward_hours} hours")
    else:
        # Conservative estimate if no models found
        max_forward_hours = 168  # 1 week
        print(f"Using conservative forward estimate: {max_forward_hours} hours")
    
    # Add safety margin for forward-looking requirements
    end_boundary_hours = int(max_forward_hours * 1.2)  # 20% safety margin
    
    print(f"Calculated end boundary: {end_boundary_hours} hours")
    
    return {
        'start_boundary_hours': start_boundary_hours,
        'end_boundary_hours': end_boundary_hours,
        'max_rolling_window_hours': max_rolling_window_hours,
        'max_forward_hours': max_forward_hours,
        'prev_week_buffer_hours': prev_week_buffer
    }

def apply_intelligent_nan_cleaning(df, boundaries_dict, datetime_col='datetime'):
    """
    Apply intelligent NaN cleaning based on calculated boundaries.
    
    Args:
        df: DataFrame to clean
        boundaries_dict: Result from calculate_intelligent_nan_boundaries
        datetime_col: Name of datetime column
    
    Returns:
        DataFrame: Cleaned data
    """
    original_length = len(df)
    
    # Calculate start and end indices
    start_rows_to_drop = boundaries_dict['start_boundary_hours']
    end_rows_to_drop = boundaries_dict['end_boundary_hours'] 
    
    # Apply cleaning
    if end_rows_to_drop > 0:
        cleaned_df = df.iloc[start_rows_to_drop:-end_rows_to_drop].copy()
    else:
        cleaned_df = df.iloc[start_rows_to_drop:].copy()
    
    cleaned_length = len(cleaned_df)
    dropped_start = start_rows_to_drop
    dropped_end = end_rows_to_drop
    
    print(f"\n=== INTELLIGENT NaN CLEANING RESULTS ===")
    print(f"Original data length: {original_length:,} rows")
    print(f"Dropped from start: {dropped_start:,} rows ({dropped_start/24:.1f} days)")
    print(f"Dropped from end: {dropped_end:,} rows ({dropped_end/24:.1f} days)")
    print(f"Final data length: {cleaned_length:,} rows")
    print(f"Data retention: {(cleaned_length/original_length)*100:.1f}%")
    
    if datetime_col in df.columns:
        print(f"Date range: {cleaned_df[datetime_col].min()} to {cleaned_df[datetime_col].max()}")
    elif hasattr(df.index, 'min'):  # DateTime index
        print(f"Date range: {cleaned_df.index.min()} to {cleaned_df.index.max()}")
    
    return cleaned_df

# Calculate intelligent boundaries
print("\n=== CALCULATING INTELLIGENT NaN CLEANING BOUNDARIES ===")
boundaries = calculate_intelligent_nan_boundaries(CONFIG)

print(f"\n=== BOUNDARY SUMMARY ===")
print(f"Start boundary: {boundaries['start_boundary_hours']:,} hours ({boundaries['start_boundary_hours']/(30*24):.1f} months)")
print(f"End boundary: {boundaries['end_boundary_hours']:,} hours ({boundaries['end_boundary_hours']/24:.1f} days)")
print(f"vs. Hard-coded: 4,320 hours (6 months) start, 11 rows end")

# Apply intelligent cleaning
combined_df_clean = apply_intelligent_nan_cleaning(combined_df, boundaries)

# --- Essential Data Checks ---
freq = pd.infer_freq(combined_df_clean.index)
print(f"Inferred frequency: {freq}")
expected_range = pd.date_range(combined_df_clean.index.min(), combined_df_clean.index.max(), freq=freq or 'H')
missing_timestamps = expected_range.difference(combined_df_clean.index)
if len(missing_timestamps) == 0:
    print("No missing timestamps. Index is consecutive.")
else:
    print(f"Missing {len(missing_timestamps)} timestamps.")

problem_cols = []
for col in combined_df_clean.columns:
    sample = combined_df_clean[col].dropna().head(100)
    for v in sample:
        if isinstance(v, (np.ndarray, list, tuple)):
            print(f"Column '{col}' contains non-scalar value: type={type(v)}, example={v}")
            problem_cols.append(col)
            break
if not problem_cols:
    print("All columns contain only scalar values.")

nan_diag_df = count_consecutive_nan_rows(combined_df_clean)
nan_cols = nan_diag_df[nan_diag_df['total_nans'] != 0]
print(nan_cols)

# --- Train/Val/Test Split ---
train_pct = CONFIG["splitting"]["train_pct"]
val_pct = CONFIG["splitting"]["val_pct"]

# Override for HPT - temporarily use validation split
if val_pct == 0:
    print("Overriding CONFIG splits to enable HPT...")
    train_pct = 0.99
    val_pct = 0.01
    print(f"Using HPT splits: train_pct={train_pct}, val_pct={val_pct}, test_pct={1-train_pct-val_pct}")

n_samples = len(combined_df_clean)
train_end_idx = int(n_samples * train_pct)
val_end_idx = train_end_idx + int(n_samples * val_pct)

features_clean = combined_df_clean[features.columns.intersection(combined_df_clean.columns)]
targets_clean = combined_df_clean[targets.columns.intersection(combined_df_clean.columns)]

X_train = features_clean.iloc[:train_end_idx]
X_val = features_clean.iloc[train_end_idx:val_end_idx]
X_test = features_clean.iloc[val_end_idx:]
y_train = targets_clean.iloc[:train_end_idx]
y_val = targets_clean.iloc[train_end_idx:val_end_idx]
y_test = targets_clean.iloc[val_end_idx:]

print(f"Split shapes: X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")

debug_issues = []  # Add this near the top

# --- Stage Feature Fallbacks After Data Prep ---
if not stage1_feature_lists and not stage1_feature_list:
    stage1_feature_list = list(features_clean.columns)
    stage1_feature_file_source = None
    print(f"[INFO] Stage 1 using all available features ({len(stage1_feature_list)}).")

if not stage2_feature_lists and not stage2_feature_list:
    stage2_feature_list = list(features_clean.columns)
    stage2_feature_file_source = None
    print(f"[INFO] Stage 2 using all available features ({len(stage2_feature_list)}).")

if not stage3_feature_lists and not stage3_feature_list:
    stage3_feature_list = list(features_clean.columns)
    stage3_feature_file_source = None
    print(f"[INFO] Stage 3 using all available features ({len(stage3_feature_list)}).")


def _prepare_stage1_feature_plan(
    allow_target_specific: bool,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    plan: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    available_columns = set(X_train.columns)
    for target in stage1_targets:
        percentiles = list(get_percentiles(target) or [])
        family = _infer_family(target, stage1_cfg)
        note = ""
        target_specific_features: list[str] = []
        target_specific_source: Path | None = None
        selected_list = []
        source: Path | None = None

        if allow_target_specific:
            target_specific_features, target_specific_source = _load_target_specific_feature_list(target)

        if target_specific_features:
            selected_list = list(target_specific_features)
            source = target_specific_source
            note = "target-specific feature list"
        elif stage1_feature_lists:
            selected_list = list(stage1_feature_lists.get(family, []))
            source = stage1_feature_sources.get(family)
            if not selected_list:
                fallback_features = stage1_feature_list if stage1_feature_list else list(features_clean.columns)
                selected_list = list(fallback_features)
                source = stage1_feature_file_source
                note = "fallback to stage feature list" if stage1_feature_file_source else "fallback to all features"
            else:
                note = "stage family feature list"
        else:
            fallback_features = stage1_feature_list if stage1_feature_list else list(features_clean.columns)
            selected_list = list(fallback_features)
            source = stage1_feature_file_source
            note = "stage feature file" if stage1_feature_file_source else "all available features"

        if not selected_list:
            errors.append(
                f"Stage 1 target '{target}' has no features configured."
            )
            continue

        missing_features = [col for col in selected_list if col not in available_columns]
        if missing_features:
            preview = ", ".join(missing_features[:10])
            suffix = "..." if len(missing_features) > 10 else ""
            source_label = _format_feature_source_label(source, "Stage 1", note)
            errors.append(
                f"Stage 1 target '{target}' feature source '{source_label}' references {len(missing_features)} missing columns: {preview}{suffix}"
            )
            continue

        plan[target] = {
            "features": selected_list,
            "source": source,
            "note": note,
            "percentiles": percentiles,
            "use_all_when_empty": False,
            "percentile_source": paths.get("feature_selection") if percentiles else None,
        }

    return plan, errors


def _prepare_stage_feature_plan(
    stage_label: str,
    stage_targets: list[str],
    stage_cfg: dict,
    feature_lists: dict[str, list[str]],
    feature_sources: dict[str, Path],
    fallback_list: list[str],
    fallback_source: Path | None,
    allow_target_specific: bool = False,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    plan: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    for target in stage_targets:
        percentiles = list(get_percentiles(target) or [])
        family = _infer_family(target, stage_cfg)
        note = ""
        source: Path | None = None
        selected_list: list[str] = []

        if allow_target_specific:
            target_specific_features, target_specific_source = _load_target_specific_feature_list(target)
            if target_specific_features:
                selected_list = list(target_specific_features)
                source = target_specific_source
                note = "target-specific robust feature list"

        if not selected_list and feature_lists:
            selected_list = list(feature_lists.get(family, []))
            source = feature_sources.get(family)
            if not selected_list:
                selected_list = list(fallback_list)
                source = fallback_source
                note = "fallback to stage feature list" if fallback_source else "fallback to all features"
            else:
                note = "stage family feature list"
        elif not selected_list:
            selected_list = list(fallback_list)
            source = fallback_source
            note = "stage feature file" if fallback_source else "all available features"

        plan[target] = {
            "features": selected_list,
            "source": source,
            "note": note,
            "percentiles": percentiles,
            "use_all_when_empty": not bool(selected_list),
            "percentile_source": paths.get("feature_selection") if percentiles else None,
        }

    return plan, errors


def _print_feature_plan(stage_label: str, plan: dict[str, dict[str, Any]]) -> None:
    print(f"\n=== {stage_label.upper()} FEATURE PLAN ===")
    if not plan:
        print(f"- No targets configured for {stage_label}.")
        return
    for target, info in plan.items():
        percentiles = info.get("percentiles") or []
        percentile_labels = ", ".join(
            f"p{int(round(p * 100)):02d}" for p in percentiles
        ) if percentiles else "none"
        use_all = info.get("use_all_when_empty", False) and not info["features"]
        feature_desc = "all available features" if use_all else f"{len(info['features'])} features"
        source_desc = _format_feature_source_label(info.get("source"), stage_label, info.get("note", ""))
        print(
            f"- {source_desc} -> {target} (features: {feature_desc}, percentiles: {percentile_labels})"
        )
        percent_override = info.get("percentile_source")
        if percentiles and percent_override:
            print(f"    Percentile overrides: {percent_override}")



def _expected_model_features_from_plan(plan: dict[str, dict[str, Any]]) -> set[str]:
    expected: set[str] = set()
    for target, info in plan.items():
        percentiles = info.get("percentiles") or []
        for percentile in percentiles:
            p_suffix = f"p{int(round(percentile * 100)):02d}"
            expected.add(f"model_{target}_{p_suffix}")
    return expected


def _collect_required_model_features(plan: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    required: dict[str, list[str]] = {}
    for target, info in plan.items():
        features = info.get("features", []) or []
        model_features = [feat for feat in features if isinstance(feat, str) and feat.startswith("model_")]
        if model_features:
            required[target] = model_features
    return required


stage1_feature_plan, stage1_plan_errors = _prepare_stage1_feature_plan(stage1_allow_target_specific)
stage2_feature_plan, stage2_plan_errors = _prepare_stage_feature_plan(
    "Stage 2",
    stage2_targets,
    stage2_cfg,
    stage2_feature_lists,
    stage2_feature_sources,
    stage2_feature_list,
    stage2_feature_file_source,
    allow_target_specific=stage2_allow_target_specific,
)
stage3_feature_plan, stage3_plan_errors = _prepare_stage_feature_plan(
    "Stage 3",
    stage3_targets,
    stage3_cfg,
    stage3_feature_lists,
    stage3_feature_sources,
    stage3_feature_list,
    stage3_feature_file_source,
    allow_target_specific=stage3_allow_target_specific,
)

base_feature_columns = set(features_clean.columns)
stage1_expected_model_features = _expected_model_features_from_plan(stage1_feature_plan)
stage2_required_model_features = _collect_required_model_features(stage2_feature_plan)
for target, model_features in stage2_required_model_features.items():
    unresolved_full = [
        feat for feat in model_features
        if feat not in base_feature_columns and feat not in stage1_expected_model_features
    ]
    if unresolved_full:
        preview = ", ".join(unresolved_full[:10])
        if len(unresolved_full) > 10:
            preview += "..."
        stage2_plan_errors.append(
            f"Stage 2 target '{target}' requires derived features ({preview}) but no Stage 1 plan will generate them."
        )

stage2_expected_model_features = _expected_model_features_from_plan(stage2_feature_plan)
stage3_required_model_features = _collect_required_model_features(stage3_feature_plan)
for target, model_features in stage3_required_model_features.items():
    unresolved_full = [
        feat for feat in model_features
        if feat not in base_feature_columns
        and feat not in stage1_expected_model_features
        and feat not in stage2_expected_model_features
    ]
    if unresolved_full:
        preview = ", ".join(unresolved_full[:10])
        if len(unresolved_full) > 10:
            preview += "..."
        stage3_plan_errors.append(
            f"Stage 3 target '{target}' requires derived features ({preview}) but no earlier stage plan will generate them."
        )

_print_feature_plan("Stage 1", stage1_feature_plan)
_print_feature_plan("Stage 2", stage2_feature_plan)
_print_feature_plan("Stage 3", stage3_feature_plan)

fatal_plan_errors = stage1_plan_errors + stage2_plan_errors + stage3_plan_errors
if fatal_plan_errors:
    print("\n=== FEATURE PLAN VALIDATION FAILED ===")
    for message in fatal_plan_errors:
        error_print(message)
    raise SystemExit("Aborting due to feature plan validation errors.")

# --- Weighted Loss Config ---
stage_weighted_loss = {1: False, 2: False, 3: False}  # Set True to enable weighted loss per stage
weight_method = "linear"
max_weight = 5

# --- Model Training Functions ---
def run_hpt_for_target(target, percentile, X_train, y_train, X_val, y_val, p_suffix, stage, model_params, paths, use_weighted_loss):
    import optuna

    # Check if we have validation data for HPT
    if len(X_val) == 0 or len(y_val) == 0:
        warn_print(f"No validation data available for HPT on {target}_{p_suffix}. Training with pre-configured parameters.")
        return run_direct_training(target, percentile, X_train, y_train, p_suffix, model_params)

    # Optionally calculate sample weights
    sample_weights_train = None
    sample_weights_val = None
    if use_weighted_loss:
        # Example: Use a column 'time_to_expiry' if available
        if "time_to_expiry" in X_train.columns:
            sample_weights_train = calculate_weights(X_train["time_to_expiry"], method=weight_method, max_weight=max_weight)
        if "time_to_expiry" in X_val.columns:
            sample_weights_val = calculate_weights(X_val["time_to_expiry"], method=weight_method, max_weight=max_weight)

    # Optuna objective
    def objective(trial):
        return optuna_objective_func_regression(
            trial,
            X_train, y_train, X_val, y_val,
            None,  # time_to_expiry_val_series (optional)
            "catboost",
            model_params,
            early_stopping_rounds_config=50,
            hpt_use_weighted_loss_config=use_weighted_loss,
            hpt_weight_method_config=weight_method,
            hpt_weight_max_multiplier_config=max_weight,
            optimization_metric='mae'
        )

    study_name = f"catboost_{target}_{p_suffix}_stage{stage}_hpt"
    storage_path = paths["hpt_studies"] / f"{study_name}.db"
    storage_url = f"sqlite:///{storage_path}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="minimize",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    best_params = study.best_trial.params
    final_params = model_params.copy()
    final_params.update(best_params)
    # Check if regression mode (percentile == -1 sentinel)
    if percentile == -1:
        final_params['loss_function'] = 'RMSE'
    else:
        final_params['loss_function'] = f'Quantile:alpha={percentile}'
    final_params['iterations'] = final_params.get('iterations', 2000)
    final_params['verbose'] = 0

    # Train final model with sample weights if available
    model = cb.CatBoostRegressor(**final_params)
    fit_kwargs = {
        "X": X_train, "y": y_train,
        "eval_set": (X_val, y_val),
        "early_stopping_rounds": 50,
        "verbose": 100
    }
    if sample_weights_train is not None:
        fit_kwargs["sample_weight"] = sample_weights_train
    if sample_weights_val is not None:
        fit_kwargs["sample_weight_eval_set"] = [sample_weights_val]
    model.fit(**fit_kwargs)

    preds = model.predict(X_val)
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    return {
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "target": target,
        "p_suffix": p_suffix,
        "features": list(X_train.columns)
    }

def run_baseline_for_target(target, percentile, X_train, y_train, X_val, y_val, model_params):
    # Skip baseline if no validation data
    if len(X_val) == 0 or len(y_val) == 0:
        return {"mae": None, "rmse": None, "preds": None}
    
    q = percentile
    baseline_pred = np.full_like(y_val, np.nanquantile(y_train, q), dtype=float)
    mae = mean_absolute_error(y_val, baseline_pred)
    rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))
    return {"mae": mae, "rmse": rmse, "preds": baseline_pred}

def run_direct_training(target, percentile, X_train, y_train, p_suffix, model_params):
    """
    Train model directly with pre-configured parameters (no HPT).
    Used when validation data is not available or HPT is disabled.
    """
    final_params = model_params.copy()
    final_params['loss_function'] = f'Quantile:alpha={percentile}'
    final_params['iterations'] = final_params.get('iterations', 2000)
    final_params['verbose'] = 0

    # Train model on all training data
    model = cb.CatBoostRegressor(**final_params)
    model.fit(X_train, y_train, verbose=False)

    print(f"Trained {target}_{p_suffix} with pre-configured parameters (no validation)")
    2
    return {
        "model": model,
        "mae": None,  # No validation data to calculate MAE
        "rmse": None,  # No validation data to calculate RMSE
        "target": target,
        "p_suffix": p_suffix,
        "features": list(X_train.columns)
    }

def summarize_results(baseline_results, hpt_results, stage_name):
    import pandas as pd
    if not hpt_results:
        return f"No HPT results for {stage_name}"
    df = pd.DataFrame([
        {
            "target": hpt["target"],
            "p_suffix": hpt["p_suffix"],
            "HPT_MAE": hpt["mae"],
            "HPT_RMSE": hpt["rmse"],
        }
        for hpt in hpt_results
    ])
    if baseline_results:
        df_baseline = pd.DataFrame(baseline_results)
        df = df.merge(
            df_baseline[["target", "p_suffix", "mae", "rmse"]],
            on=["target", "p_suffix"],
            how="left",
            suffixes=("", "_baseline")
        )
        df.rename(columns={"mae": "Baseline_MAE", "rmse": "Baseline_RMSE"}, inplace=True)
        df["MAE_Improvement_%"] = 100 * (df["Baseline_MAE"] - df["HPT_MAE"]) / df["Baseline_MAE"]
        df["RMSE_Improvement_%"] = 100 * (df["Baseline_RMSE"] - df["HPT_RMSE"]) / df["Baseline_RMSE"]
    return df

# --- Stage 1 ---
stage1_models, stage1_baseline_results, stage1_hpt_results, stage1_preds_val = {}, [], [], {}
stage1_model_features: dict[str, list[str]] = {}
for target, plan_entry in stage1_feature_plan.items():
    percentiles = plan_entry.get("percentiles") or []
    if not percentiles:
        warn_print(f"[WARN] No percentiles configured for {target} in Stage 1. Skipping.")
        continue
    preds_val_dict = {}
    base_features = list(plan_entry["features"])
    for percentile in percentiles:
        print(f"\n[INFO] Stage 1 - Processing target '{target}' at percentile {percentile}...")
        p_suffix = f"p{int(round(percentile * 100)):02d}"
        override_features, override_source = _load_percentile_specific_feature_list(target, percentile)
        features_this = list(base_features)
        feature_source_note = plan_entry.get("note", "")
        feature_source_path = plan_entry.get("source")
        if override_features:
            print(
                f"[INFO] Using percentile-specific feature list for {target}_{p_suffix} from {override_source} "
                f"({len(override_features)} features)."
            )
            missing_override = [col for col in override_features if col not in X_train.columns]
            if missing_override:
                missing_preview = ", ".join(missing_override)
                error_print(
                    f"[ERROR] Percentile-specific list for {target}_{p_suffix} is missing {len(missing_override)} columns: {missing_preview}"
                )
                raise SystemExit("Aborting due to missing Stage 1 percentile-specific features.")
            else:
                features_this = list(override_features)
                feature_source_path = override_source
                feature_source_note = "percentile-specific feature list"

        missing_final = [col for col in features_this if col not in X_train.columns]
        if missing_final:
            preview = ", ".join(missing_final[:10])
            suffix = "..." if len(missing_final) > 10 else ""
            source_label = _format_feature_source_label(feature_source_path, "Stage 1", feature_source_note)
            error_print(
                f"[ERROR] Stage 1 target '{target}_{p_suffix}' feature source '{source_label}' references {len(missing_final)} missing columns: {preview}{suffix}"
            )
            raise SystemExit("Aborting due to missing Stage 1 features.")

        X_train_fs = X_train[features_this]
        X_val_fs = X_val[features_this]
        y_train_target = y_train[target]
        y_val_target = y_val[target]
        baseline = run_baseline_for_target(
            target,
            percentile,
            X_train_fs,
            y_train_target,
            X_val_fs,
            y_val_target,
            CONFIG["model"]["params"]["catboost"],
        )
        stage1_baseline_results.append(
            {
                "target": target,
                "percentile": percentile,
                "p_suffix": p_suffix,
                "mae": baseline["mae"],
                "rmse": baseline["rmse"],
            }
        )
        hpt = run_hpt_for_target(
            target,
            percentile,
            X_train_fs,
            y_train_target,
            X_val_fs,
            y_val_target,
            p_suffix,
            1,
            CONFIG["model"]["params"]["catboost"],
            paths,
            stage_weighted_loss[1],
        )
        stage1_models[f"{target}_{p_suffix}"] = hpt["model"]
        stage1_model_features[f"{target}_{p_suffix}"] = list(features_this)
        stage1_hpt_results.append(hpt)
        preds_val_dict[p_suffix] = hpt["model"].predict(X_val_fs)
    stage1_preds_val[target] = preds_val_dict

# --- Stage 2 ---
print("\n[INFO] Generating derived features from Stage 1 models for Stage 2...")
derived_train_1 = pd.DataFrame(index=X_train.index)
derived_val_1 = pd.DataFrame(index=X_val.index)
for target, plan_entry in stage1_feature_plan.items():
    percentiles = plan_entry.get("percentiles") or []
    for percentile in percentiles:
        p_suffix = f"p{int(round(percentile*100)):02d}"
        model_key = f"{target}_{p_suffix}"
        if model_key not in stage1_models:
            warn_print(
                f"[WARN] Stage 1 model '{model_key}' not available when generating derived features. Skipping."
            )
            continue
        model = stage1_models[model_key]
        feature_cols = stage1_model_features.get(model_key, [])
        feature_name = f"model_{model_key}"
        print(f'[INFO] Generating derived feature: {feature_name}')
        if feature_cols:
            derived_train_1[feature_name] = model.predict(X_train[feature_cols])
            derived_val_1[feature_name] = model.predict(X_val[feature_cols])
        else:
            derived_train_1[feature_name] = model.predict(X_train)
            derived_val_1[feature_name] = model.predict(X_val)
print(f"[INFO] Derived features for Stage 2: {list(derived_train_1.columns)}")

stage2_train_full = pd.concat([X_train, derived_train_1], axis=1)
stage2_val_full = pd.concat([X_val, derived_val_1], axis=1)
stage2_available_columns = set(stage2_train_full.columns)

stage2_models, stage2_baseline_results, stage2_hpt_results, stage2_preds_val = {}, [], [], {}
stage2_model_features: dict[str, list[str]] = {}
for target, plan_entry in stage2_feature_plan.items():
    percentiles = plan_entry.get("percentiles") or []
    if not percentiles:
        warn_print(f"[WARN] No percentiles configured for {target} in Stage 2. Skipping.")
        continue
    preds_val_dict = {}
    feature_source = plan_entry.get("source")
    configured_features = plan_entry.get("features", [])
    use_all = plan_entry.get("use_all_when_empty", False)
    missing_features = [col for col in configured_features if col not in stage2_available_columns]
    if missing_features:
        preview = ", ".join(missing_features[:10])
        suffix = "..." if len(missing_features) > 10 else ""
        source_label = _format_feature_source_label(feature_source, "Stage 2", plan_entry.get("note", ""))
        print(f"[DEBUG] Stage 2 missing columns: {missing_features}")
        error_print(
            f"[ERROR] Stage 2 target '{target}' feature source '{source_label}' references {len(missing_features)} missing columns: {preview}{suffix}"
        )
        raise SystemExit("Aborting due to missing Stage 2 features.")

    for percentile in percentiles:
        p_suffix = f"p{int(round(percentile * 100)):02d}"
        if use_all or not configured_features:
            base_source_list = stage2_train_full.columns.tolist()
        else:
            base_source_list = list(configured_features)

        override_features, override_source = _load_percentile_specific_feature_list(target, percentile)
        features_this: list[str]
        if override_features:
            missing_override = [col for col in override_features if col not in stage2_available_columns]
            if missing_override:
                missing_preview = ", ".join(missing_override[:12])
                suffix = "..." if len(missing_override) > 12 else ""
                print(f"[DEBUG] Stage 2 percentile override missing columns: {missing_override}")
                error_print(
                    f"[ERROR] Percentile-specific list for {target}_{p_suffix} references {len(missing_override)} missing columns: {missing_preview}{suffix}."
                )
                debug_issues.append(
                    f"Stage 2 override {target}_{p_suffix} missing columns: {missing_preview}{suffix}"
                )
                raise SystemExit("Aborting due to missing Stage 2 percentile-specific features.")
            else:
                features_this = [col for col in override_features if col in stage2_available_columns]
                print(
                    f"[INFO] Stage 2 using percentile-specific feature list for {target}_{p_suffix} "
                    f"from {override_source} ({len(features_this)} features)."
                )
        else:
            features_this = [col for col in base_source_list if col in stage2_available_columns]

        if not features_this:
            error_print(
                f"[ERROR] No features selected for {target}_{p_suffix} in Stage 2 after applying configured lists."
            )
            raise SystemExit("Aborting due to empty Stage 2 feature set.")

        X_train_fs = stage2_train_full[features_this]
        X_val_fs = stage2_val_full[features_this]
        y_train_target = y_train[target]
        y_val_target = y_val[target]
        baseline = run_baseline_for_target(
            target,
            percentile,
            X_train_fs,
            y_train_target,
            X_val_fs,
            y_val_target,
            CONFIG["model"]["params"]["catboost"],
        )
        stage2_baseline_results.append(
            {
                "target": target,
                "percentile": percentile,
                "p_suffix": p_suffix,
                "mae": baseline["mae"],
                "rmse": baseline["rmse"],
            }
        )
        hpt = run_hpt_for_target(
            target,
            percentile,
            X_train_fs,
            y_train_target,
            X_val_fs,
            y_val_target,
            p_suffix,
            2,
            CONFIG["model"]["params"]["catboost"],
            paths,
            stage_weighted_loss[2],
        )
        stage2_models[f"{target}_{p_suffix}"] = hpt["model"]
        stage2_model_features[f"{target}_{p_suffix}"] = list(features_this)
        stage2_hpt_results.append(hpt)
        preds_val_dict[p_suffix] = hpt["model"].predict(X_val_fs)
    stage2_preds_val[target] = preds_val_dict

# --- Stage 3 ---
print("\n[INFO] Generating derived features from Stage 2 models for Stage 3...")
derived_train_2 = pd.DataFrame(index=X_train.index)
derived_val_2 = pd.DataFrame(index=X_val.index)
X_train_stage2 = stage2_train_full
X_val_stage2 = stage2_val_full
for target, plan_entry in stage2_feature_plan.items():
    percentiles = plan_entry.get("percentiles") or []
    for percentile in percentiles:
        p_suffix = f"p{int(round(percentile * 100)):02d}"
        model_key = f"{target}_{p_suffix}"
        if model_key not in stage2_models:
            error_print(f"[ERROR] Model {model_key} not found in Stage 2. Skipping derived feature.")
            continue
        model = stage2_models[model_key]
        feature_cols = stage2_model_features.get(model_key, [])
        feature_name = f"model_{model_key}"
        print(f'[INFO] Generating derived feature: {feature_name}')
        if feature_cols:
            derived_train_2[feature_name] = model.predict(X_train_stage2[feature_cols])
            derived_val_2[feature_name] = model.predict(X_val_stage2[feature_cols])
        else:
            derived_train_2[feature_name] = model.predict(X_train_stage2)
            derived_val_2[feature_name] = model.predict(X_val_stage2)
print(f"[INFO] Derived features for Stage 3: {list(derived_train_2.columns)}")

stage3_train_full = pd.concat([X_train, derived_train_1, derived_train_2], axis=1)
stage3_val_full = pd.concat([X_val, derived_val_1, derived_val_2], axis=1)
stage3_available_columns = set(stage3_train_full.columns)

stage3_models, stage3_baseline_results, stage3_hpt_results, stage3_preds_val = {}, [], [], {}
stage3_model_features: dict[str, list[str]] = {}
for target, plan_entry in stage3_feature_plan.items():
    percentiles = plan_entry.get("percentiles") or []
    if not percentiles:
        warn_print(f"[WARN] No percentiles configured for {target} in Stage 3. Skipping.")
        continue
    preds_val_dict = {}
    feature_source = plan_entry.get("source")
    configured_features = plan_entry.get("features", [])
    use_all = plan_entry.get("use_all_when_empty", False)
    missing_features = [col for col in configured_features if col not in stage3_available_columns]
    if missing_features:
        preview = ", ".join(missing_features[:10])
        suffix = "..." if len(missing_features) > 10 else ""
        source_label = _format_feature_source_label(feature_source, "Stage 3", plan_entry.get("note", ""))
        print(f"[DEBUG] Stage 3 missing columns: {missing_features}")
        error_print(
            f"[ERROR] Stage 3 target '{target}' feature source '{source_label}' references {len(missing_features)} missing columns: {preview}{suffix}"
        )
        raise SystemExit("Aborting due to missing Stage 3 features.")

    for percentile in percentiles:
        p_suffix = f"p{int(round(percentile * 100)):02d}"
        if use_all or not configured_features:
            base_source_list = stage3_train_full.columns.tolist()
        else:
            base_source_list = list(configured_features)

        override_features, override_source = _load_percentile_specific_feature_list(target, percentile)
        features_this: list[str]
        if override_features:
            missing_override = [col for col in override_features if col not in stage3_available_columns]
            if missing_override:
                missing_preview = ", ".join(missing_override[:12])
                suffix = "..." if len(missing_override) > 12 else ""
                print(f"[DEBUG] Stage 3 percentile override missing columns: {missing_override}")
                error_print(
                    f"[ERROR] Percentile-specific list for {target}_{p_suffix} references {len(missing_override)} missing columns: {missing_preview}{suffix}."
                )
                debug_issues.append(
                    f"Stage 3 override {target}_{p_suffix} missing columns: {missing_preview}{suffix}"
                )
                raise SystemExit("Aborting due to missing Stage 3 percentile-specific features.")
            else:
                features_this = [col for col in override_features if col in stage3_available_columns]
                print(
                    f"[INFO] Stage 3 using percentile-specific feature list for {target}_{p_suffix} "
                    f"from {override_source} ({len(features_this)} features)."
                )
        else:
            features_this = [col for col in base_source_list if col in stage3_available_columns]

        if not features_this:
            error_print(
                f"[ERROR] No features selected for {target}_{p_suffix} in Stage 3 after applying configured lists."
            )
            raise SystemExit("Aborting due to empty Stage 3 feature set.")

        X_train_fs = stage3_train_full[features_this]
        X_val_fs = stage3_val_full[features_this]
        y_train_target = y_train[target]
        y_val_target = y_val[target]
        baseline = run_baseline_for_target(
            target,
            percentile,
            X_train_fs,
            y_train_target,
            X_val_fs,
            y_val_target,
            CONFIG["model"]["params"]["catboost"],
        )
        stage3_baseline_results.append(
            {
                "target": target,
                "percentile": percentile,
                "p_suffix": p_suffix,
                "mae": baseline["mae"],
                "rmse": baseline["rmse"],
            }
        )
        hpt = run_hpt_for_target(
            target,
            percentile,
            X_train_fs,
            y_train_target,
            X_val_fs,
            y_val_target,
            p_suffix,
            3,
            CONFIG["model"]["params"]["catboost"],
            paths,
            stage_weighted_loss[3],
        )
        stage3_models[f"{target}_{p_suffix}"] = hpt["model"]
        stage3_model_features[f"{target}_{p_suffix}"] = list(features_this)
        stage3_hpt_results.append(hpt)
        preds_val_dict[p_suffix] = hpt["model"].predict(X_val_fs)
    stage3_preds_val[target] = preds_val_dict

# --- Save Model Bundles ---
def save_model_bundles(stage_models, stage_hpt_results, stage_num):
    for hpt in stage_hpt_results:
        key = f"{hpt['target']}_{hpt['p_suffix']}"
        model = hpt["model"]
        features = hpt["features"]
        bundle = {"model": model, "features": features}
        model_filename = f"catboost_{hpt['target']}_{hpt['p_suffix']}_stage{stage_num}_hpt_bundle.pkl"
        model_save_path = paths["trained_models"] / model_filename
        joblib.dump(bundle, model_save_path)
        debug_print(f"Saved model bundle: {model_save_path}")

save_model_bundles(stage1_models, stage1_hpt_results, 1)
save_model_bundles(stage2_models, stage2_hpt_results, 2)
save_model_bundles(stage3_models, stage3_hpt_results, 3)

# --- Print Summaries ---
print("\n=== Stage 1 HPT Summary ===")
print(summarize_results(stage1_baseline_results, stage1_hpt_results, "Stage 1"))
print("\n=== Stage 2 HPT Summary ===")
print(summarize_results(stage2_baseline_results, stage2_hpt_results, "Stage 2"))
print("\n=== Stage 3 HPT Summary ===")
print(summarize_results(stage3_baseline_results, stage3_hpt_results, "Stage 3"))

# --- Debugging Summary ---
print("\n\033[95m=== MODEL TRAINING DEBUGGING SUMMARY ===\033[0m")
if debug_issues:
    for issue in debug_issues:
        error_print(f"- {issue}")
else:
    debug_print("No major issues detected in model training pipeline.")

print("Stage feature configuration summary:")
stage1_available_columns = set(X_train.columns)
if stage1_feature_plan:
    for target, info in sorted(stage1_feature_plan.items()):
        features = info.get("features", [])
        intersection = [col for col in features if col in stage1_available_columns]
        source_label = _format_feature_source_label(info.get("source"), "Stage 1", info.get("note", ""))
        print(
            f"  Stage 1::{target}: {len(features)} features from {source_label}"
            f" ({len(intersection)} present in training data)."
        )
if stage1_feature_lists:
    for family, features in stage1_feature_lists.items():
        source = stage1_feature_sources.get(family)
        intersection = [col for col in features if col in stage1_available_columns]
        source_name = source.name if isinstance(source, Path) else str(source)
        print(
            f"  Stage 1::{family}: {len(features)} features from {source_name} "
            f"({len(intersection)} present in training data)."
        )
elif stage1_feature_list:
    present = [col for col in stage1_feature_list if col in stage1_available_columns]
    source_name = (
        stage1_feature_file_source.name if isinstance(stage1_feature_file_source, Path)
        else str(stage1_feature_file_source)
        if stage1_feature_file_source else "all features"
    )
    print(
        f"  Stage 1: {len(stage1_feature_list)} features from {source_name} "
        f"({len(present)} present)."
    )
else:
    print(f"  Stage 1: using all available features ({len(stage1_available_columns)}).")

if stage2_feature_lists:
    for family, features in stage2_feature_lists.items():
        source = stage2_feature_sources.get(family)
        intersection = [col for col in features if col in stage2_available_columns]
        source_name = source.name if isinstance(source, Path) else str(source)
        print(
            f"  Stage 2::{family}: {len(features)} features from {source_name} "
            f"({len(intersection)} present in training data)."
        )
elif stage2_feature_plan:
    for target, info in sorted(stage2_feature_plan.items()):
        features = info.get("features", [])
        intersection = [col for col in features if col in stage2_available_columns]
        source_label = _format_feature_source_label(info.get("source"), "Stage 2", info.get("note", ""))
        print(
            f"  Stage 2::{target}: {len(features)} features from {source_label}"
            f" ({len(intersection)} present in training data)."
        )
elif stage2_feature_list:
    present = [col for col in stage2_feature_list if col in stage2_available_columns]
    source_name = (
        stage2_feature_file_source.name if isinstance(stage2_feature_file_source, Path)
        else str(stage2_feature_file_source)
        if stage2_feature_file_source else "all features"
    )
    print(
        f"  Stage 2: {len(stage2_feature_list)} features from {source_name} "
        f"({len(present)} present)."
    )
else:
    print(f"  Stage 2: using all available features ({len(stage2_available_columns)}).")

if stage3_feature_lists:
    for family, features in stage3_feature_lists.items():
        source = stage3_feature_sources.get(family)
        intersection = [col for col in features if col in stage3_available_columns]
        source_name = source.name if isinstance(source, Path) else str(source)
        print(
            f"  Stage 3::{family}: {len(features)} features from {source_name} "
            f"({len(intersection)} present in training data)."
        )
elif stage3_feature_plan:
    for target, info in sorted(stage3_feature_plan.items()):
        features = info.get("features", [])
        intersection = [col for col in features if col in stage3_available_columns]
        source_label = _format_feature_source_label(info.get("source"), "Stage 3", info.get("note", ""))
        print(
            f"  Stage 3::{target}: {len(features)} features from {source_label}"
            f" ({len(intersection)} present in training data)."
        )
elif stage3_feature_list:
    present = [col for col in stage3_feature_list if col in stage3_available_columns]
    source_name = (
        stage3_feature_file_source.name if isinstance(stage3_feature_file_source, Path)
        else str(stage3_feature_file_source)
        if stage3_feature_file_source else "all features"
    )
    print(
        f"  Stage 3: {len(stage3_feature_list)} features from {source_name} "
        f"({len(present)} present)."
    )
else:
    print(f"  Stage 3: using all available features ({len(stage3_available_columns)}).")
