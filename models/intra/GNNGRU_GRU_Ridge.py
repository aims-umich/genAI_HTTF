#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-split deterministic reconstruction comparison for PG29 TF sensors.

Models
------
1. GNN-GRU: graph encoding at every time step, followed by a one-layer GRU.
2. GRU: one global GRU over the flattened sparse sensor history (no graph).
3. Ridge: location-query Ridge regression baseline (linear, classic ML).

Protocol
--------
1. The 20-sensor fixed test set is manually specified and NEVER used as
   model input and NEVER used as a training target for any model.
       total TF sensors = 107
       fixed test        = 20
       remaining pool     = 87 (107 - 20)
2. The 87 remaining sensors are split ONCE, randomly (seeded, reproducible),
   into:
       known        = 66 sensors
       train-target = 21 sensors  (66 + 21 = 87, so every remaining sensor
                                    is used)
   This is a single split -- there is no mask-fold mixing of multiple masks
   within one training run.
3. All three models see only the known sensors as input and are trained (or
   fit) using a loss/objective computed only on the train-target sensors.
4. Neural hyperparameters (GNN-GRU, GRU) are tuned using a temporal
   train/validation split of the train-target sensors only. The Ridge alpha
   is tuned the same way (same temporal split, same train-target sensors),
   so all three models select their hyperparameters under an identical
   no-leakage rule.
5. Official final evaluation uses the split's known sensors as input and
   the fixed test sensors as targets, for all three models.

Default settings match the requested configuration:
- temporal window L: 40
- graph neighbors k: 8
- conditioning GNN: 2 -> 64 -> 64
- batch size: 64
- one-layer GRU: 64 -> 256 (GNN-GRU) / 256 (GRU)
- gradient clipping: 1.0
- optimizer: Adam (neural), Ridge closed-form (linear)
- normalization: sklearn StandardScaler
- final training epochs: 150

All comments and strings use plain ASCII.
"""

import os
import time
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================
# CPU THREAD LIMITS - set before numpy/sklearn/torch imports
# ============================================================
THREADS = os.environ.get("THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", THREADS)
os.environ.setdefault("MKL_NUM_THREADS", THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", THREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", THREADS)
os.environ.setdefault("BLIS_NUM_THREADS", THREADS)

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 0. CONFIGURATION
# ============================================================
TF_CSV = os.environ.get(
    "TF_CSV",
    "/home/unabila/PG29/pg29_TF_30s_common.csv",
)
COORD_CSV = os.environ.get(
    "COORD_CSV",
    "/home/unabila/PG29/core_TC_coordinates_common_norm.csv",
)
OUT_DIR = Path(
    os.environ.get(
        "OUT_DIR",
        "/home/unabila/PG29/TF_20test_66known_21target_GNNGRU_GRU_Ridge",
    )
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed TF split sizes.
N_TF_EXPECTED = int(os.environ.get("N_TF_EXPECTED", "107"))
N_TEST_EXPECTED = int(os.environ.get("N_TEST_EXPECTED", "20"))
N_KNOWN_EXPECTED = int(os.environ.get("N_KNOWN_EXPECTED", "66"))
N_TRAIN_TARGET_EXPECTED = int(os.environ.get("N_TRAIN_TARGET_EXPECTED", "21"))  # 107 - 20 - 66

# This script always uses a single split (no mask-fold mixing). K_FOLDS=1 is
# kept only so the dataset classes below (written generically over "folds")
# do not need to change.
K_FOLDS = 1
FINAL_EVAL_FOLD = 0
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", "42"))

# Fixed requested architecture and training settings.
WINDOW = int(os.environ.get("WINDOW", "40"))
GNN_HIDDEN = int(os.environ.get("GNN_HIDDEN", "64"))
GRU_HIDDEN = int(os.environ.get("GRU_HIDDEN", "256"))
K_NEIGHBORS_GRAPH = int(os.environ.get("K_NEIGHBORS_GRAPH", "8"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
N_EPOCHS = int(os.environ.get("N_EPOCHS", "150"))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", "1.0"))

# Tunable neural settings. These are tuned only on train-target sensors.
LR = float(os.environ.get("LR", "0.001"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.0"))
DROPOUT = float(os.environ.get("DROPOUT", "0.05"))
COORD_HIDDEN = int(os.environ.get("COORD_HIDDEN", "64"))
HEAD_HIDDEN = int(os.environ.get("HEAD_HIDDEN", "192"))

GRAPH_SIGMA = os.environ.get("GRAPH_SIGMA", "").strip()
GRAPH_SIGMA = None if GRAPH_SIGMA == "" else float(GRAPH_SIGMA)
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "10"))

# Hyperparameter tuning controls (neural models only; Ridge is tuned
# separately in section 10, using the same temporal train/val split).
TUNE_HPARAMS = os.environ.get("TUNE_HPARAMS", "1").strip().lower() not in {"0", "false", "no"}
TUNE_EPOCHS = int(os.environ.get("TUNE_EPOCHS", "50"))
VAL_FRACTION = float(os.environ.get("VAL_FRACTION", "0.15"))
TUNE_METHODS_RAW = os.environ.get("TUNE_METHODS", "GNN-GRU,GRU")
TUNE_METHODS = [x.strip() for x in TUNE_METHODS_RAW.split(",") if x.strip()]

# Independent final neural seeds. The known/train-target split does not change.
BASE_MODEL_SEED = int(os.environ.get("BASE_MODEL_SEED", "42"))
N_SEEDS = int(os.environ.get("N_SEEDS", "1"))
SEED_STRIDE = int(os.environ.get("SEED_STRIDE", "1000"))

# Ridge settings. Hyperparameters are selected on train-target sensors only,
# using the same temporal train/validation split as the neural tuning.
RUN_RIDGE = os.environ.get("RUN_RIDGE", "1").strip().lower() not in {"0", "false", "no"}
RIDGE_ALPHA_GRID = [
    float(x) for x in os.environ.get("RIDGE_ALPHA_GRID", "0.01,0.1,1,3,10,30,100,300,1000").split(",")
    if x.strip()
]
RIDGE_FIT_INTERCEPT = os.environ.get("RIDGE_FIT_INTERCEPT", "1").strip().lower() not in {"0", "false", "no"}
RIDGE_STANDARDIZE_FEATURES = os.environ.get("RIDGE_STANDARDIZE_FEATURES", "1").strip().lower() not in {"0", "false", "no"}
RIDGE_USE_WINDOW_FEATURES = os.environ.get("RIDGE_USE_WINDOW_FEATURES", "1").strip().lower() not in {"0", "false", "no"}
RIDGE_MAX_TRAIN_ROWS = int(os.environ.get("RIDGE_MAX_TRAIN_ROWS", "400000"))
RIDGE_MAX_TUNE_ROWS = int(os.environ.get("RIDGE_MAX_TUNE_ROWS", "120000"))
RIDGE_MAX_REFIT_ROWS = int(os.environ.get("RIDGE_MAX_REFIT_ROWS", "400000"))
RIDGE_MAX_EVAL_ROWS_PER_BLOCK = int(os.environ.get("RIDGE_MAX_EVAL_ROWS_PER_BLOCK", "250000"))

# Plot and output settings.
PLOT_MAX_SENSORS = int(os.environ.get("PLOT_MAX_SENSORS", "9"))
SAMPLE_INTERVAL_SECONDS = float(os.environ.get("SAMPLE_INTERVAL_SECONDS", "30.0"))
SAVE_CHECKPOINTS = os.environ.get("SAVE_CHECKPOINTS", "1").strip().lower() not in {"0", "false", "no"}
MODEL_ORDER = ["GNN-GRU", "GRU"] + (["Ridge"] if RUN_RIDGE else [])


def parse_hparam_candidates() -> List[Dict[str, float]]:
    """Parse optional HPARAM_CANDIDATES or return a compact default grid."""
    raw = os.environ.get("HPARAM_CANDIDATES", "").strip()
    if raw:
        candidates = []
        for block in raw.split(";"):
            block = block.strip()
            if not block:
                continue
            row = {
                "lr": LR,
                "dropout": DROPOUT,
                "weight_decay": WEIGHT_DECAY,
                "coord_hidden": COORD_HIDDEN,
                "head_hidden": HEAD_HIDDEN,
            }
            for item in block.split(","):
                key, val = item.split("=")
                key = key.strip()
                val = val.strip()
                if key in {"coord_hidden", "head_hidden"}:
                    row[key] = int(float(val))
                else:
                    row[key] = float(val)
            candidates.append(row)
        if not candidates:
            raise ValueError("HPARAM_CANDIDATES was provided but no candidates were parsed")
        return candidates

    # Compact default grid for important non-architecture hyperparameters.
    # GNN_HIDDEN=64 and GRU_HIDDEN=256 stay fixed as requested.
    return [
        {"lr": 1.0e-3, "dropout": 0.05, "weight_decay": 0.0, "coord_hidden": 64, "head_hidden": 192},
        {"lr": 7.0e-4, "dropout": 0.05, "weight_decay": 0.0, "coord_hidden": 64, "head_hidden": 192},
        {"lr": 5.0e-4, "dropout": 0.05, "weight_decay": 1.0e-5, "coord_hidden": 64, "head_hidden": 192},
        {"lr": 1.0e-3, "dropout": 0.10, "weight_decay": 0.0, "coord_hidden": 64, "head_hidden": 192},
        {"lr": 7.0e-4, "dropout": 0.10, "weight_decay": 1.0e-5, "coord_hidden": 64, "head_hidden": 192},
        {"lr": 1.0e-3, "dropout": 0.05, "weight_decay": 1.0e-5, "coord_hidden": 64, "head_hidden": 256},
    ]


HPARAM_CANDIDATES = parse_hparam_candidates()


# ============================================================
# 0.1 FIXED TF TEST SENSORS (20 sensors, never used as input or target)
# ============================================================
FIXED_TEST = [
    # Core Block #1
    #"TF-1125",

    # Core Block #3
    "TF-1334",
    "TF-1306",

    # Core Block #5
    "TF-1518",
    "TF-1536",

    # Core Block #7
    "TF-1720",
    "TF-1708",

    # Core Block #9
    "TF-1920",

    # Upper Reflector #2
    "TF-1008",

    # Upper Plenum
    "TF-8207",
    "TF-8209",
    "TF-8105",
    "TF-8203",

    # Lower Plenum
    "TF-2203",
    "TF-2209",
    "TF-2214",
    "TF-2304",
    "TF-2310",
    "TF-2315",
    "TF-2404",
    "TF-2411",
]


# ============================================================
# 1. REPRODUCIBILITY AND DEVICE
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, int(THREADS)))
try:
    torch.set_num_interop_threads(max(1, min(2, int(THREADS))))
except RuntimeError:
    pass

print("=" * 78)
print("Single-split deterministic reconstruction comparison (PG29 TF sensors)")
print("Models: " + ", ".join(MODEL_ORDER))
print("No GAN, no diffusion, no RBF. GNN-GRU and GRU are deterministic;")
print("Ridge is a classic linear location-query baseline.")
print("=" * 78)
print(f"Device       : {DEVICE}")
print(f"TF_CSV       : {TF_CSV}")
print(f"COORD_CSV    : {COORD_CSV}")
print(f"OUT_DIR      : {OUT_DIR.resolve()}")
print(f"Split seed   : {SPLIT_SEED}")
print(f"Window L     : {WINDOW}")
print(f"Graph k      : {K_NEIGHBORS_GRAPH}")
print(f"GNN hidden   : 2 -> {GNN_HIDDEN} -> {GNN_HIDDEN}")
print(f"GRU hidden   : {GNN_HIDDEN} -> {GRU_HIDDEN} (GNN-GRU) / {GRU_HIDDEN} (GRU)")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Epochs       : {N_EPOCHS}")
print(f"Tune hparams : {TUNE_HPARAMS}")
print(f"Tune epochs  : {TUNE_EPOCHS}")
print(f"Grad clip    : {MAX_GRAD_NORM}")
print("Optimizer    : Adam (neural), closed-form (Ridge)")
print("Normalization: sklearn.StandardScaler")


# ============================================================
# 2. DATA LOADING AND CLEANING
# ============================================================
def find_sensor_name_column(df: pd.DataFrame) -> str:
    candidates = [
        "sensor", "Sensor", "SENSOR",
        "TC_tag", "TC", "tc_tag",
        "TF_tag", "TF", "tf_tag",
        "tag", "Tag", "name", "Name",
    ]
    name_col = next((c for c in candidates if c in df.columns), None)
    if name_col is None:
        raise ValueError(
            "Could not find a sensor-name column in coordinate CSV. "
            f"Available columns: {list(df.columns)}"
        )
    return name_col


def find_first_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find any of {candidates} in coordinate CSV. "
        f"Available columns: {list(df.columns)}"
    )


def clean_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Temperature dataframe is empty after column selection")

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    bad_all_nan = numeric_df.columns[numeric_df.isna().all(axis=0)].tolist()
    if bad_all_nan:
        raise ValueError(
            "These temperature columns contain no numeric values: "
            f"{bad_all_nan[:20]}"
        )

    if numeric_df.isna().any().any():
        print("WARNING: Missing temperature values found. Applying time interpolation.")
        numeric_df = numeric_df.interpolate(axis=0, limit_direction="both")
        numeric_df = numeric_df.fillna(numeric_df.median(axis=0))

    if numeric_df.isna().any().any():
        raise ValueError("Temperature data still contain NaN values after cleaning.")

    return numeric_df


def prepare_temperature_dataframe(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"Temperature CSV is empty: {path}")

    # Use only TF sensor columns. This prevents timestamp/index columns
    # from accidentally being interpreted as sensors.
    tf_cols = [c for c in raw.columns if str(c).strip().upper().startswith("TF-")]
    if not tf_cols:
        raise ValueError(f"No TF-* sensor columns found in {path}")

    return clean_numeric_dataframe(raw[tf_cols].copy())


def prepare_coordinates(coord_path: str, sensor_names: List[str]) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    raw = pd.read_csv(coord_path)
    name_col = find_sensor_name_column(raw)
    raw[name_col] = raw[name_col].astype(str).str.strip()
    coord_df = raw.set_index(name_col).reindex(sensor_names)

    missing = coord_df.index[coord_df.isna().all(axis=1)].tolist()
    if missing:
        raise ValueError(
            f"Missing coordinate rows for {len(missing)} sensors: {missing[:20]}"
        )

    r_col = find_first_column(coord_df, ["r_norm", "radius_norm", "r_cm", "radius", "r"])
    theta_col = find_first_column(coord_df, ["theta_deg", "theta_rad", "theta_norm", "theta"])
    z_col = find_first_column(coord_df, ["z_norm", "z_cm", "height_norm", "height", "z"])

    r = pd.to_numeric(coord_df[r_col], errors="coerce").to_numpy(dtype=np.float32)
    theta_raw = pd.to_numeric(coord_df[theta_col], errors="coerce").to_numpy(dtype=np.float32)
    z = pd.to_numeric(coord_df[z_col], errors="coerce").to_numpy(dtype=np.float32)

    if np.isnan(r).any() or np.isnan(theta_raw).any() or np.isnan(z).any():
        raise ValueError("Selected coordinate columns contain NaN values.")

    theta_name = theta_col.lower()
    if "deg" in theta_name:
        theta = np.deg2rad(theta_raw)
    elif "rad" in theta_name:
        theta = theta_raw
    elif "norm" in theta_name:
        theta = theta_raw * np.pi
    elif np.nanmax(np.abs(theta_raw)) > 2.5:
        theta = np.deg2rad(theta_raw)
    else:
        theta = theta_raw

    xyz = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1).astype(np.float32)
    xyz_mean = xyz.mean(axis=0, keepdims=True)
    xyz_std = xyz.std(axis=0, keepdims=True)
    xyz_std[xyz_std < 1e-8] = 1.0
    xyz_scaled = ((xyz - xyz_mean) / xyz_std).astype(np.float32)
    return coord_df, xyz_scaled, [r_col, theta_col, z_col]


ts_df = prepare_temperature_dataframe(TF_CSV)
sensor_names = [str(c).strip() for c in ts_df.columns]
ts_df.columns = sensor_names
X_original = ts_df.to_numpy(dtype=np.float32)
T_total, N_sensors = X_original.shape
all_idx = np.arange(N_sensors, dtype=int)

if N_sensors != N_TF_EXPECTED:
    raise ValueError(
        f"Expected {N_TF_EXPECTED} TF sensors, found {N_sensors}. Check {TF_CSV}."
    )

coord_df, coords_np, coord_cols = prepare_coordinates(COORD_CSV, sensor_names)
coords_torch = torch.tensor(coords_np, dtype=torch.float32, device=DEVICE)

print(f"Temperature shape : {X_original.shape}")
print(f"Coordinate shape  : {coords_np.shape}")
print(f"Coordinate columns: {coord_cols}")


# ============================================================
# 3. FIXED TEST SENSORS + SINGLE RANDOM KNOWN/TRAIN-TARGET SPLIT
# ============================================================
def sensor_tags_to_indices(tags: List[str], role_name: str) -> np.ndarray:
    clean = [str(x).strip() for x in tags if str(x).strip()]
    duplicates = pd.Series(clean).value_counts()
    duplicates = duplicates[duplicates > 1].index.tolist()
    if duplicates:
        raise ValueError(f"Duplicate sensors in {role_name}: {duplicates}")

    name_to_idx = {name: i for i, name in enumerate(sensor_names)}
    missing = [tag for tag in clean if tag not in name_to_idx]
    if missing:
        raise ValueError(f"These {role_name} sensors were not found in TF_CSV: {missing}")
    return np.sort(np.asarray([name_to_idx[tag] for tag in clean], dtype=int))


def save_sensor_index_csv(path: Path, indices: np.ndarray) -> None:
    indices = np.asarray(indices, dtype=int)
    pd.DataFrame({"idx": indices, "sensor": [sensor_names[i] for i in indices]}).to_csv(path, index=False)


fixed_test_idx = sensor_tags_to_indices(FIXED_TEST, "fixed_test")
if len(fixed_test_idx) != N_TEST_EXPECTED:
    raise ValueError(
        f"FIXED_TEST must contain exactly {N_TEST_EXPECTED} unique sensors. "
        f"Found {len(fixed_test_idx)}."
    )

remaining_idx = np.asarray(sorted(set(all_idx.tolist()) - set(fixed_test_idx.tolist())), dtype=int)
n_remaining_expected = N_TF_EXPECTED - N_TEST_EXPECTED
if len(remaining_idx) != n_remaining_expected:
    raise ValueError(f"Expected {n_remaining_expected} non-test sensors, got {len(remaining_idx)}.")
if N_KNOWN_EXPECTED + N_TRAIN_TARGET_EXPECTED != len(remaining_idx):
    raise ValueError(
        f"N_KNOWN_EXPECTED ({N_KNOWN_EXPECTED}) + N_TRAIN_TARGET_EXPECTED "
        f"({N_TRAIN_TARGET_EXPECTED}) must equal the {len(remaining_idx)} non-test sensors."
    )


def make_known_train_target_split(remaining_idx: np.ndarray, n_train_target: int, split_seed: int):
    """
    Deterministically partition remaining_idx into a train-target subset
    (n_train_target sensors) and a known subset (the rest). Pure function
    of split_seed, so re-running reproduces the same split.
    """
    rng_split = np.random.default_rng(split_seed)
    train_target = np.sort(rng_split.choice(remaining_idx, size=n_train_target, replace=False))
    known = np.asarray(sorted(set(remaining_idx.tolist()) - set(train_target.tolist())), dtype=int)
    return known, train_target


official_known_idx, official_train_target_idx = make_known_train_target_split(
    remaining_idx=remaining_idx,
    n_train_target=N_TRAIN_TARGET_EXPECTED,
    split_seed=SPLIT_SEED,
)

assert len(official_known_idx) == N_KNOWN_EXPECTED
assert len(official_train_target_idx) == N_TRAIN_TARGET_EXPECTED
assert len(fixed_test_idx) + len(official_known_idx) + len(official_train_target_idx) == N_sensors
assert set(fixed_test_idx.tolist()).isdisjoint(official_known_idx.tolist())
assert set(fixed_test_idx.tolist()).isdisjoint(official_train_target_idx.tolist())
assert set(official_known_idx.tolist()).isdisjoint(official_train_target_idx.tolist())

# A single fold, so the dataset classes below (written generically over
# "folds") do not need any structural change from the mask-fold version.
folds = [{"known_idx": official_known_idx, "train_target_idx": official_train_target_idx}]

used_idx = np.unique(np.concatenate([fixed_test_idx, official_known_idx, official_train_target_idx]))
unused_idx = np.asarray(sorted(set(all_idx.tolist()) - set(used_idx.tolist())), dtype=int)

save_sensor_index_csv(OUT_DIR / "fixed_test_sensors.csv", fixed_test_idx)
save_sensor_index_csv(OUT_DIR / "official_known_sensors.csv", official_known_idx)
save_sensor_index_csv(OUT_DIR / "official_train_target_sensors.csv", official_train_target_idx)
save_sensor_index_csv(OUT_DIR / "remaining_sensors.csv", remaining_idx)
save_sensor_index_csv(OUT_DIR / "unused_sensors.csv", unused_idx)
save_sensor_index_csv(OUT_DIR / "fold1_known_sensors.csv", official_known_idx)
save_sensor_index_csv(OUT_DIR / "fold1_train_target_sensors.csv", official_train_target_idx)

print("\nSingle random split:")
print(f"Total sensors          : {N_sensors}")
print(f"Fixed test sensors     : {len(fixed_test_idx)}")
print(f"Remaining sensors      : {len(remaining_idx)}")
print(f"Known sensors          : {len(official_known_idx)}")
print(f"Train-target sensors   : {len(official_train_target_idx)}")
print(f"Unused sensors         : {len(unused_idx)} (expected 0, since known + train-target = remaining)")


# ============================================================
# 4. NO-LEAKAGE NORMALIZATION
# ============================================================
# Use sklearn StandardScaler exactly, but fit it only on sensors that can be
# used during training/tuning. Fixed-test values do not determine the scaler.
permitted_scaler_idx = np.unique(np.concatenate([official_known_idx, official_train_target_idx]))
if set(fixed_test_idx.tolist()) & set(permitted_scaler_idx.tolist()):
    raise ValueError("Leakage: fixed test sensors were included in scaler fit set")
scale_values = X_original[:, permitted_scaler_idx].reshape(-1, 1)
NORMALIZER = StandardScaler()
NORMALIZER.fit(scale_values)
GLOBAL_MEAN = float(NORMALIZER.mean_[0])
GLOBAL_STD = float(NORMALIZER.scale_[0])
if not np.isfinite(GLOBAL_STD) or GLOBAL_STD < 1e-8:
    GLOBAL_STD = 1.0
    NORMALIZER.scale_[0] = 1.0

X_scaled = NORMALIZER.transform(X_original.reshape(-1, 1)).reshape(X_original.shape).astype(np.float32)


def inverse_standardize(arr_scaled: np.ndarray) -> np.ndarray:
    arr_scaled = np.asarray(arr_scaled, dtype=np.float32)
    return NORMALIZER.inverse_transform(arr_scaled.reshape(-1, 1)).reshape(arr_scaled.shape).astype(np.float32)


pd.DataFrame([{
    "normalization": "sklearn.StandardScaler",
    "fit_scope": "union_of_known_and_train_target_sensors_flattened",
    "global_mean": GLOBAL_MEAN,
    "global_std": GLOBAL_STD,
    "fit_sensor_count": len(permitted_scaler_idx),
    "fixed_test_used_for_fit": False,
}]).to_csv(OUT_DIR / "normalization.csv", index=False)


# ============================================================
# 5. GRAPH CONSTRUCTION
# ============================================================
def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12).astype(np.float32)


def build_knn_adjacency(coords: np.ndarray, k: int = 8, sigma: float = None) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    dist = pairwise_distances(coords, coords)
    dist2 = dist * dist
    nonzero = dist[dist > 1e-8]
    if sigma is None:
        sigma = float(np.median(nonzero)) if len(nonzero) else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    n = coords.shape[0]
    k = min(max(1, int(k)), max(1, n - 1))
    adjacency = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        neighbors = np.argsort(dist[i])[1:k + 1]
        weights = np.exp(-dist2[i, neighbors] / (2.0 * sigma * sigma))
        adjacency[i, neighbors] = weights.astype(np.float32)

    adjacency = np.maximum(adjacency, adjacency.T)
    adjacency += np.eye(n, dtype=np.float32)
    degree = adjacency.sum(axis=1)
    degree_inv_sqrt = 1.0 / np.sqrt(degree + 1e-8)
    adjacency = degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]
    return adjacency.astype(np.float32)


A_np = build_knn_adjacency(coords_np, k=K_NEIGHBORS_GRAPH, sigma=GRAPH_SIGMA)
A_torch = torch.tensor(A_np, dtype=torch.float32, device=DEVICE)


# ============================================================
# 6. PLOTTING HELPERS
# ============================================================
def plot_sensor_split_3d() -> None:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        coords_np[official_known_idx, 0], coords_np[official_known_idx, 1], coords_np[official_known_idx, 2],
        s=45, alpha=0.75, marker="o", label=f"Known ({len(official_known_idx)})",
    )
    ax.scatter(
        coords_np[official_train_target_idx, 0], coords_np[official_train_target_idx, 1], coords_np[official_train_target_idx, 2],
        s=45, alpha=0.65, marker="^", label=f"Train target ({len(official_train_target_idx)})",
    )
    ax.scatter(
        coords_np[fixed_test_idx, 0], coords_np[fixed_test_idx, 1], coords_np[fixed_test_idx, 2],
        s=70, alpha=0.9, marker="x", label=f"Fixed test ({len(fixed_test_idx)})",
    )
    if len(unused_idx) > 0:
        ax.scatter(
            coords_np[unused_idx, 0], coords_np[unused_idx, 1], coords_np[unused_idx, 2],
            s=18, alpha=0.25, marker=".", label=f"Unused ({len(unused_idx)})",
        )
    ax.set_xlabel("x normalized")
    ax.set_ylabel("y normalized")
    ax.set_zlabel("z normalized")
    ax.set_title("Single random split used for training and final evaluation")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=22, azim=45)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "split_3d.png", dpi=240)
    plt.close(fig)


plot_sensor_split_3d()


# ============================================================
# 7. MASK-FOLD WINDOW DATASETS (single fold in this script)
# ============================================================
class MaskFoldWindowDataset(Dataset):
    """Return sparse known-sensor history, full target, known mask, and train mask."""

    def __init__(self, X: np.ndarray, folds: List[Dict[str, np.ndarray]], window: int, base_indices: np.ndarray) -> None:
        self.X = np.asarray(X, dtype=np.float32)
        self.folds = folds
        self.window = int(window)
        self.base_indices = np.asarray(base_indices, dtype=int)
        self.K = len(folds)
        if self.window < 1:
            raise ValueError("WINDOW must be at least 1")
        if self.X.shape[0] < self.window:
            raise ValueError(f"Data length {self.X.shape[0]} is smaller than WINDOW={self.window}")
        if len(self.base_indices) == 0:
            raise ValueError("base_indices is empty")

    def __len__(self) -> int:
        return len(self.base_indices) * self.K

    def __getitem__(self, idx: int):
        base_pos = idx // self.K
        fold_k = idx % self.K
        base_idx = int(self.base_indices[base_pos])
        fold = self.folds[fold_k]
        known_idx = fold["known_idx"]
        train_idx = fold["train_target_idx"]

        x_window = self.X[base_idx:base_idx + self.window].copy()
        x0_true = x_window[-1].copy()

        known_mask = np.zeros(self.X.shape[1], dtype=np.float32)
        known_mask[known_idx] = 1.0
        train_mask = np.zeros(self.X.shape[1], dtype=bool)
        train_mask[train_idx] = True

        x_known = np.zeros_like(x_window, dtype=np.float32)
        x_known[:, known_idx] = x_window[:, known_idx]
        mask_seq = np.repeat(known_mask[None, :], self.window, axis=0)
        known_seq = np.stack([x_known, mask_seq], axis=-1).astype(np.float32)

        return (
            torch.from_numpy(x0_true),
            torch.from_numpy(known_seq),
            torch.from_numpy(known_mask.copy()),
            torch.from_numpy(train_mask),
        )


class FixedKnownEvalDataset(Dataset):
    """Evaluation dataset using one fixed known mask."""

    def __init__(self, X: np.ndarray, known_idx: np.ndarray, window: int) -> None:
        self.X = np.asarray(X, dtype=np.float32)
        self.known_idx = np.asarray(known_idx, dtype=int)
        self.window = int(window)
        if self.X.shape[0] < self.window:
            raise ValueError(f"Data length {self.X.shape[0]} is smaller than WINDOW={self.window}")
        self.known_mask = np.zeros(self.X.shape[1], dtype=np.float32)
        self.known_mask[self.known_idx] = 1.0

    def __len__(self) -> int:
        return self.X.shape[0] - self.window + 1

    def __getitem__(self, idx: int):
        x_window = self.X[idx:idx + self.window].copy()
        x0_true = x_window[-1].copy()
        x_known = np.zeros_like(x_window, dtype=np.float32)
        x_known[:, self.known_idx] = x_window[:, self.known_idx]
        mask_seq = np.repeat(self.known_mask[None, :], self.window, axis=0)
        known_seq = np.stack([x_known, mask_seq], axis=-1).astype(np.float32)
        return (
            torch.from_numpy(x0_true),
            torch.from_numpy(known_seq),
            torch.from_numpy(self.known_mask.copy()),
        )


n_windows = X_scaled.shape[0] - WINDOW + 1
all_base_indices = np.arange(n_windows, dtype=int)
val_count = max(1, int(round(n_windows * VAL_FRACTION)))
if val_count >= n_windows:
    val_count = max(1, n_windows // 5)
train_base_indices = all_base_indices[:-val_count]
val_base_indices = all_base_indices[-val_count:]

print(f"Total windows        : {n_windows}")
print(f"Tune train windows   : {len(train_base_indices)}")
print(f"Tune validation win. : {len(val_base_indices)}")


def make_maskfold_loader(base_indices: np.ndarray, shuffle: bool, seed: int, drop_last: bool) -> DataLoader:
    ds = MaskFoldWindowDataset(X=X_scaled, folds=folds, window=WINDOW, base_indices=base_indices)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
        generator=generator,
    )


def make_eval_loader() -> DataLoader:
    ds = FixedKnownEvalDataset(X=X_scaled, known_idx=official_known_idx, window=WINDOW)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
    )


# ============================================================
# 8. NEURAL MODEL COMPONENTS: GNN-GRU and GRU
# ============================================================
class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        h = torch.einsum("ij,bjf->bif", adjacency, x)
        return self.linear(h)


class CoordEncoder(nn.Module):
    def __init__(self, coord_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)


class ReconstructionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GNNGRUReconstructor(nn.Module):
    """Spatial GNN at every time step followed by a per-sensor one-layer GRU."""

    def __init__(self, n_sensors: int, coord_dim: int, gnn_hidden: int, gru_hidden: int, coord_hidden: int, head_hidden: int, dropout: float) -> None:
        super().__init__()
        self.n_sensors = n_sensors
        self.gnn_hidden = gnn_hidden
        self.gru_hidden = gru_hidden
        self.gcn1 = GraphConv(2, gnn_hidden, dropout=dropout)
        self.gcn2 = GraphConv(gnn_hidden, gnn_hidden, dropout=dropout)
        self.gru = nn.GRU(input_size=gnn_hidden, hidden_size=gru_hidden, num_layers=1, batch_first=True)
        self.coord_encoder = CoordEncoder(coord_dim, coord_hidden)
        head_in = gru_hidden + gru_hidden + coord_hidden + 2
        self.head = ReconstructionHead(head_in, head_hidden, dropout)

    def forward(self, known_seq: torch.Tensor, known_mask: torch.Tensor, coords: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, window, n_sensors, channels = known_seq.shape
        if n_sensors != self.n_sensors or channels != 2:
            raise ValueError(f"Expected known_seq shape (B,L,{self.n_sensors},2), received {tuple(known_seq.shape)}")
        x = known_seq.reshape(batch_size * window, n_sensors, channels)
        h = F.silu(self.gcn1(x, adjacency))
        h = F.silu(self.gcn2(h, adjacency))
        h = h.view(batch_size, window, n_sensors, self.gnn_hidden)
        h_sensor = h.permute(0, 2, 1, 3).contiguous()
        h_sensor = h_sensor.view(batch_size * n_sensors, window, self.gnn_hidden)
        gru_out, _ = self.gru(h_sensor)
        node_context = gru_out[:, -1, :].view(batch_size, n_sensors, self.gru_hidden)
        mask = known_mask.unsqueeze(-1)
        global_context = (node_context * mask).sum(dim=1)
        global_context = global_context / mask.sum(dim=1).clamp_min(1.0)
        coord_context = self.coord_encoder(coords)
        current_sparse = known_seq[:, -1, :, :]
        features = torch.cat([
            node_context,
            global_context.unsqueeze(1).expand(-1, n_sensors, -1),
            coord_context.unsqueeze(0).expand(batch_size, -1, -1),
            current_sparse,
        ], dim=-1)
        return self.head(features)


class GRUReconstructor(nn.Module):
    """Global temporal reconstruction without graph message passing (the
    ablation of removing the GNN component from GNNGRUReconstructor above)."""

    def __init__(self, n_sensors: int, coord_dim: int, gru_hidden: int, coord_hidden: int, head_hidden: int, dropout: float) -> None:
        super().__init__()
        self.n_sensors = n_sensors
        self.gru = nn.GRU(input_size=n_sensors * 2, hidden_size=gru_hidden, num_layers=1, batch_first=True)
        self.coord_encoder = CoordEncoder(coord_dim, coord_hidden)
        head_in = gru_hidden + coord_hidden + 2
        self.head = ReconstructionHead(head_in, head_hidden, dropout)

    def forward(self, known_seq: torch.Tensor, known_mask: torch.Tensor, coords: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        del known_mask, adjacency
        batch_size, window, n_sensors, channels = known_seq.shape
        if n_sensors != self.n_sensors or channels != 2:
            raise ValueError(f"Expected known_seq shape (B,L,{self.n_sensors},2), received {tuple(known_seq.shape)}")
        x = known_seq.reshape(batch_size, window, n_sensors * channels)
        gru_out, _ = self.gru(x)
        global_context = gru_out[:, -1, :]
        coord_context = self.coord_encoder(coords)
        current_sparse = known_seq[:, -1, :, :]
        features = torch.cat([
            global_context.unsqueeze(1).expand(-1, n_sensors, -1),
            coord_context.unsqueeze(0).expand(batch_size, -1, -1),
            current_sparse,
        ], dim=-1)
        return self.head(features)


def build_model(method: str, hparams: Dict[str, float]) -> nn.Module:
    common = {
        "n_sensors": N_sensors,
        "coord_dim": coords_np.shape[1],
        "coord_hidden": int(hparams["coord_hidden"]),
        "head_hidden": int(hparams["head_hidden"]),
        "dropout": float(hparams["dropout"]),
    }
    if method == "GNN-GRU":
        return GNNGRUReconstructor(gnn_hidden=GNN_HIDDEN, gru_hidden=GRU_HIDDEN, **common).to(DEVICE)
    if method == "GRU":
        return GRUReconstructor(gru_hidden=GRU_HIDDEN, **common).to(DEVICE)
    raise ValueError(f"Unknown neural method: {method}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# 9. NEURAL TRAINING, TUNING, AND EVALUATION
# ============================================================
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total_squared_error = 0.0
    total_target_values = 0
    for x0_true, known_seq, known_mask, train_mask in loader:
        x0_true = x0_true.to(DEVICE, non_blocking=True)
        known_seq = known_seq.to(DEVICE, non_blocking=True)
        known_mask = known_mask.to(DEVICE, non_blocking=True)
        train_mask = train_mask.to(DEVICE, non_blocking=True)
        prediction = model(known_seq=known_seq, known_mask=known_mask, coords=coords_torch, adjacency=A_torch)
        loss = F.mse_loss(prediction[train_mask], x0_true[train_mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        n_values = int(train_mask.sum().item())
        total_squared_error += float(loss.item()) * n_values
        total_target_values += n_values
    return total_squared_error / max(total_target_values, 1)


@torch.no_grad()
def evaluate_train_target_scaled_mse(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    total_squared_error = 0.0
    total_target_values = 0
    for x0_true, known_seq, known_mask, train_mask in loader:
        x0_true = x0_true.to(DEVICE, non_blocking=True)
        known_seq = known_seq.to(DEVICE, non_blocking=True)
        known_mask = known_mask.to(DEVICE, non_blocking=True)
        train_mask = train_mask.to(DEVICE, non_blocking=True)
        prediction = model(known_seq=known_seq, known_mask=known_mask, coords=coords_torch, adjacency=A_torch)
        se = (prediction[train_mask] - x0_true[train_mask]).pow(2).sum().item()
        n_values = int(train_mask.sum().item())
        total_squared_error += float(se)
        total_target_values += n_values
    return total_squared_error / max(total_target_values, 1)


@torch.no_grad()
def collect_neural_predictions(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    true_rows = []
    pred_rows = []
    for x0_true, known_seq, known_mask in loader:
        x0_true = x0_true.to(DEVICE, non_blocking=True)
        known_seq = known_seq.to(DEVICE, non_blocking=True)
        known_mask = known_mask.to(DEVICE, non_blocking=True)
        prediction = model(known_seq=known_seq, known_mask=known_mask, coords=coords_torch, adjacency=A_torch)
        true_rows.append(x0_true.cpu().numpy())
        pred_rows.append(prediction.cpu().numpy())
    y_true_scaled = np.concatenate(true_rows, axis=0)
    y_pred_scaled = np.concatenate(pred_rows, axis=0)
    y_true = inverse_standardize(y_true_scaled)
    y_pred = inverse_standardize(y_pred_scaled)
    return y_true[:, fixed_test_idx], y_pred[:, fixed_test_idx]


def tune_neural_hparams(method: str) -> Dict[str, float]:
    if not TUNE_HPARAMS or method not in TUNE_METHODS:
        default = {"lr": LR, "dropout": DROPOUT, "weight_decay": WEIGHT_DECAY, "coord_hidden": COORD_HIDDEN, "head_hidden": HEAD_HIDDEN}
        return default

    print("\n" + "=" * 78)
    print(f"Tuning hyperparameters for {method}")
    print("Tuning objective: train-target validation RMSE only, no fixed-test use")
    print("=" * 78)

    tune_train_loader = make_maskfold_loader(train_base_indices, shuffle=True, seed=BASE_MODEL_SEED, drop_last=True)
    tune_val_loader = make_maskfold_loader(val_base_indices, shuffle=False, seed=BASE_MODEL_SEED, drop_last=False)
    rows = []
    best_row = None

    for cand_id, hparams in enumerate(HPARAM_CANDIDATES, start=1):
        set_seed(BASE_MODEL_SEED + cand_id * 97)
        model = build_model(method, hparams)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(hparams["lr"]),
            weight_decay=float(hparams["weight_decay"]),
        )
        start = time.time()
        last_train_mse = np.nan
        for epoch in range(1, TUNE_EPOCHS + 1):
            last_train_mse = train_one_epoch(model, tune_train_loader, optimizer)
        val_mse = evaluate_train_target_scaled_mse(model, tune_val_loader)
        row = {
            "method": method,
            "candidate_id": cand_id,
            "lr": float(hparams["lr"]),
            "dropout": float(hparams["dropout"]),
            "weight_decay": float(hparams["weight_decay"]),
            "coord_hidden": int(hparams["coord_hidden"]),
            "head_hidden": int(hparams["head_hidden"]),
            "tune_epochs": TUNE_EPOCHS,
            "train_rmse_scaled": math.sqrt(max(float(last_train_mse), 0.0)),
            "val_rmse_scaled": math.sqrt(max(float(val_mse), 0.0)),
            "tune_runtime_sec": time.time() - start,
        }
        rows.append(row)
        print(
            f"{method} cand {cand_id}: val_RMSE_scaled={row['val_rmse_scaled']:.6f} "
            f"lr={row['lr']} dropout={row['dropout']} wd={row['weight_decay']} "
            f"coord={row['coord_hidden']} head={row['head_hidden']}",
            flush=True,
        )
        if best_row is None or row["val_rmse_scaled"] < best_row["val_rmse_scaled"]:
            best_row = row

    tune_df = pd.DataFrame(rows).sort_values("val_rmse_scaled")
    tune_path = OUT_DIR / f"hparam_tuning_{method.lower().replace('-', '_')}.csv"
    tune_df.to_csv(tune_path, index=False)

    best = {
        "lr": float(best_row["lr"]),
        "dropout": float(best_row["dropout"]),
        "weight_decay": float(best_row["weight_decay"]),
        "coord_hidden": int(best_row["coord_hidden"]),
        "head_hidden": int(best_row["head_hidden"]),
    }
    print(f"Best {method} hparams: {best}")
    return best


def train_and_evaluate_neural(method: str, seed: int, hparams: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    set_seed(seed)
    model = build_model(method, hparams)
    parameter_count = count_parameters(model)
    train_loader = make_maskfold_loader(all_base_indices, shuffle=True, seed=seed, drop_last=True)
    eval_loader = make_eval_loader()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(hparams["lr"]),
        weight_decay=float(hparams["weight_decay"]),
    )

    method_dir = OUT_DIR / method.lower().replace("-", "_") / f"seed_{seed}"
    method_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 78)
    print(f"Training final {method} | seed={seed} | parameters={parameter_count:,}")
    print(f"Hparams: {hparams}")
    print("=" * 78)

    start_train = time.time()
    log_rows = []
    for epoch in range(1, N_EPOCHS + 1):
        train_mse = train_one_epoch(model, train_loader, optimizer)
        train_rmse_scaled = math.sqrt(max(train_mse, 0.0))
        log_rows.append({"epoch": epoch, "train_mse_scaled": train_mse, "train_rmse_scaled": train_rmse_scaled, "lr": optimizer.param_groups[0]["lr"]})
        if epoch == 1 or epoch % LOG_EVERY == 0 or epoch == N_EPOCHS:
            print(f"{method} seed={seed} epoch {epoch:04d}/{N_EPOCHS} train_RMSE_scaled={train_rmse_scaled:.6f}", flush=True)
    train_runtime = time.time() - start_train
    pd.DataFrame(log_rows).to_csv(method_dir / "training_log.csv", index=False)

    start_eval = time.time()
    y_true, y_pred = collect_neural_predictions(model, eval_loader)
    eval_runtime = time.time() - start_eval

    if SAVE_CHECKPOINTS:
        torch.save({
            "method": method,
            "seed": seed,
            "model_state_dict": model.state_dict(),
            "sensor_names": sensor_names,
            "folds": folds,
            "official_known_idx": official_known_idx,
            "official_train_target_idx": official_train_target_idx,
            "fixed_test_idx": fixed_test_idx,
            "coord_cols": coord_cols,
            "normalization": "sklearn.StandardScaler",
            "global_mean": GLOBAL_MEAN,
            "global_std": GLOBAL_STD,
            "hyperparameters": {
                "window": WINDOW,
                "gnn_hidden": GNN_HIDDEN,
                "gru_hidden": GRU_HIDDEN,
                "coord_hidden": int(hparams["coord_hidden"]),
                "head_hidden": int(hparams["head_hidden"]),
                "dropout": float(hparams["dropout"]),
                "k_neighbors_graph": K_NEIGHBORS_GRAPH,
                "lr": float(hparams["lr"]),
                "weight_decay": float(hparams["weight_decay"]),
                "batch_size": BATCH_SIZE,
                "n_epochs": N_EPOCHS,
            },
        }, method_dir / "model.pt")

    runtime = {
        "train_runtime_sec": train_runtime,
        "eval_runtime_sec": eval_runtime,
        "total_runtime_sec": train_runtime + eval_runtime,
        "parameter_count": parameter_count,
        "lr": float(hparams["lr"]),
        "dropout": float(hparams["dropout"]),
        "weight_decay": float(hparams["weight_decay"]),
        "coord_hidden": int(hparams["coord_hidden"]),
        "head_hidden": int(hparams["head_hidden"]),
    }
    return y_true, y_pred, runtime


# ============================================================
# 10. RIDGE BASELINE (location-query linear regression)
# ============================================================
def make_ridge_time_arrays(X_scaled_arr: np.ndarray, window: int):
    """
    Build the per-time-step feature arrays used by the Ridge model:
    current values, and (optionally) a windowed mean and a windowed delta.
    t_idx aligns 1:1 with the neural "x0_true" time positions (the last
    index of each window), so Ridge sees the exact same evaluation times
    and true values as the neural models.
    """
    t_idx = np.arange(window - 1, X_scaled_arr.shape[0], dtype=int)
    current = X_scaled_arr[t_idx].astype(np.float32)
    if not RIDGE_USE_WINDOW_FEATURES:
        return t_idx, current, None, None
    csum = np.vstack([
        np.zeros((1, X_scaled_arr.shape[1]), dtype=np.float64),
        np.cumsum(X_scaled_arr.astype(np.float64), axis=0),
    ])
    start = t_idx - window + 1
    end = t_idx + 1
    mean_w = ((csum[end] - csum[start]) / float(window)).astype(np.float32)
    delta_w = (X_scaled_arr[t_idx] - X_scaled_arr[start]).astype(np.float32)
    return t_idx, current, mean_w, delta_w


def build_ridge_base_feature(current, mean_w, delta_w, known_idx, n_sensors):
    known_mask = np.zeros(n_sensors, dtype=np.float32)
    known_mask[np.asarray(known_idx, dtype=int)] = 1.0
    parts = [current * known_mask[None, :]]
    if RIDGE_USE_WINDOW_FEATURES:
        parts.append(mean_w * known_mask[None, :])
        parts.append(delta_w * known_mask[None, :])
    parts.append(np.repeat(known_mask[None, :], current.shape[0], axis=0))
    return np.concatenate(parts, axis=1).astype(np.float32)


def build_ridge_feature_rows(current, mean_w, delta_w, known_idx, target_idx, coords, time_pos, target_pos):
    n_sensors = current.shape[1]
    known_mask = np.zeros(n_sensors, dtype=np.float32)
    known_mask[np.asarray(known_idx, dtype=int)] = 1.0
    parts = [current[time_pos] * known_mask[None, :]]
    if RIDGE_USE_WINDOW_FEATURES:
        parts.append(mean_w[time_pos] * known_mask[None, :])
        parts.append(delta_w[time_pos] * known_mask[None, :])
    parts.append(np.repeat(known_mask[None, :], len(time_pos), axis=0))
    queried_sensors = np.asarray(target_idx, dtype=int)[target_pos]
    parts.append(coords[queried_sensors])
    X_feat = np.concatenate(parts, axis=1).astype(np.float32)
    return X_feat, queried_sensors


def sample_ridge_rows(rng, current, mean_w, delta_w, X_scaled_arr, t_idx, allowed_base_positions, known_idx, train_target_idx, coords, n_rows):
    """
    Sample (time, target-sensor) rows restricted to allowed_base_positions
    (indices into t_idx / current), so tuning uses a genuine temporal
    holdout rather than resampling the same time range used for fitting.
    """
    n_allowed = len(allowed_base_positions)
    n_target = len(train_target_idx)
    total_possible = n_allowed * n_target
    n_rows = min(int(n_rows), int(total_possible))
    if n_rows <= 0:
        raise ValueError("No training rows requested")
    pos_in_allowed = rng.integers(0, n_allowed, size=n_rows, endpoint=False)
    time_pos = allowed_base_positions[pos_in_allowed]
    target_pos = rng.integers(0, n_target, size=n_rows, endpoint=False)
    X_feat, queried = build_ridge_feature_rows(
        current=current,
        mean_w=mean_w,
        delta_w=delta_w,
        known_idx=known_idx,
        target_idx=train_target_idx,
        coords=coords,
        time_pos=time_pos,
        target_pos=target_pos,
    )
    y = X_scaled_arr[t_idx[time_pos], queried].astype(np.float32)
    return X_feat, y


def predict_ridge_targets_for_known(model, feature_scaler, current, mean_w, delta_w, known_idx, eval_idx, coords):
    n_time = current.shape[0]
    n_eval = len(eval_idx)
    y_pred_scaled = np.zeros((n_time, n_eval), dtype=np.float32)
    base = build_ridge_base_feature(current, mean_w, delta_w, known_idx, current.shape[1])
    max_rows = max(1000, int(RIDGE_MAX_EVAL_ROWS_PER_BLOCK))
    sensors_per_block = max(1, min(n_eval, max_rows // max(1, n_time)))
    for start in range(0, n_eval, sensors_per_block):
        stop = min(n_eval, start + sensors_per_block)
        block_idx = np.asarray(eval_idx[start:stop], dtype=int)
        feats = []
        for sensor_i in block_idx:
            coord_block = np.repeat(coords[sensor_i][None, :], n_time, axis=0)
            feats.append(np.concatenate([base, coord_block], axis=1))
        X_block = np.vstack(feats).astype(np.float32)
        if feature_scaler is not None:
            X_block = feature_scaler.transform(X_block)
        pred_block = model.predict(X_block).astype(np.float32)
        pred_block = pred_block.reshape(len(block_idx), n_time).T
        y_pred_scaled[:, start:stop] = pred_block
    return inverse_standardize(y_pred_scaled)


def tune_and_train_ridge() -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    print("\n" + "=" * 78)
    print("Tuning and training Ridge (location-query linear regression)")
    print("Ridge alpha selection uses train-target sensors and the same")
    print("temporal train/validation split used for neural hyperparameter tuning.")
    print("=" * 78)
    start_all = time.time()

    ridge_rng = np.random.default_rng(BASE_MODEL_SEED + 55555)
    t_idx, current, mean_w, delta_w = make_ridge_time_arrays(X_scaled, WINDOW)

    build_start = time.time()
    X_train, y_train = sample_ridge_rows(
        rng=ridge_rng,
        current=current, mean_w=mean_w, delta_w=delta_w,
        X_scaled_arr=X_scaled, t_idx=t_idx,
        allowed_base_positions=train_base_indices,
        known_idx=official_known_idx,
        train_target_idx=official_train_target_idx,
        coords=coords_np,
        n_rows=RIDGE_MAX_TRAIN_ROWS,
    )
    X_tune, y_tune = sample_ridge_rows(
        rng=ridge_rng,
        current=current, mean_w=mean_w, delta_w=delta_w,
        X_scaled_arr=X_scaled, t_idx=t_idx,
        allowed_base_positions=val_base_indices,
        known_idx=official_known_idx,
        train_target_idx=official_train_target_idx,
        coords=coords_np,
        n_rows=RIDGE_MAX_TUNE_ROWS,
    )

    if RIDGE_STANDARDIZE_FEATURES:
        feature_scaler = StandardScaler()
        X_train_fit = feature_scaler.fit_transform(X_train)
        X_tune_fit = feature_scaler.transform(X_tune)
    else:
        feature_scaler = None
        X_train_fit = X_train
        X_tune_fit = X_tune
    build_runtime = time.time() - build_start

    print(f"Built X_train={X_train.shape}, X_tune={X_tune.shape}")

    grid_rows = []
    best_alpha = None
    best_rmse = np.inf
    for alpha in RIDGE_ALPHA_GRID:
        t0 = time.time()
        model = Ridge(alpha=float(alpha), fit_intercept=RIDGE_FIT_INTERCEPT, random_state=BASE_MODEL_SEED)
        model.fit(X_train_fit, y_train)
        pred_tune = model.predict(X_tune_fit).astype(np.float32)
        mask = np.isfinite(y_tune) & np.isfinite(pred_tune)
        rmse_t = math.sqrt(mean_squared_error(y_tune[mask], pred_tune[mask])) if mask.sum() >= 2 else np.nan
        grid_rows.append({"alpha": float(alpha), "tune_rmse_scaled": rmse_t, "fit_and_tune_sec": time.time() - t0})
        print(f"Ridge alpha={alpha:g} | tune_rmse_scaled={rmse_t:.6f}", flush=True)
        if np.isfinite(rmse_t) and rmse_t < best_rmse:
            best_rmse = rmse_t
            best_alpha = float(alpha)
    if best_alpha is None:
        best_alpha = float(RIDGE_ALPHA_GRID[0])
    grid_runtime = time.time() - build_start - build_runtime
    pd.DataFrame(grid_rows).sort_values("tune_rmse_scaled").to_csv(OUT_DIR / "ridge_alpha_grid.csv", index=False)

    # Final refit on rows sampled from the FULL time range (train + val),
    # mirroring how the neural models retrain on all_base_indices after
    # hyperparameters are chosen on the temporal holdout.
    refit_start = time.time()
    X_refit, y_refit = sample_ridge_rows(
        rng=ridge_rng,
        current=current, mean_w=mean_w, delta_w=delta_w,
        X_scaled_arr=X_scaled, t_idx=t_idx,
        allowed_base_positions=all_base_indices,
        known_idx=official_known_idx,
        train_target_idx=official_train_target_idx,
        coords=coords_np,
        n_rows=RIDGE_MAX_REFIT_ROWS,
    )
    if RIDGE_STANDARDIZE_FEATURES:
        final_feature_scaler = StandardScaler()
        X_refit_fit = final_feature_scaler.fit_transform(X_refit)
    else:
        final_feature_scaler = None
        X_refit_fit = X_refit
    final_model = Ridge(alpha=best_alpha, fit_intercept=RIDGE_FIT_INTERCEPT, random_state=BASE_MODEL_SEED)
    final_model.fit(X_refit_fit, y_refit)
    refit_runtime = time.time() - refit_start

    eval_start = time.time()
    y_pred_test = predict_ridge_targets_for_known(
        model=final_model,
        feature_scaler=final_feature_scaler,
        current=current, mean_w=mean_w, delta_w=delta_w,
        known_idx=official_known_idx,
        eval_idx=fixed_test_idx,
        coords=coords_np,
    )
    y_true_test = X_original[t_idx][:, fixed_test_idx]
    eval_runtime = time.time() - eval_start

    pd.DataFrame([{
        "best_alpha": best_alpha,
        "fit_intercept": RIDGE_FIT_INTERCEPT,
        "standardize_features": RIDGE_STANDARDIZE_FEATURES,
        "use_window_features": RIDGE_USE_WINDOW_FEATURES,
        "train_rows": int(X_train.shape[0]),
        "tune_rows": int(X_tune.shape[0]),
        "refit_rows": int(X_refit.shape[0]),
    }]).to_csv(OUT_DIR / "ridge_selected_settings.csv", index=False)

    n_coef = int(np.asarray(final_model.coef_).size)
    total_runtime = time.time() - start_all
    runtime = {
        "train_runtime_sec": build_runtime + grid_runtime + refit_runtime,
        "eval_runtime_sec": eval_runtime,
        "total_runtime_sec": total_runtime,
        "parameter_count": n_coef,
        "best_alpha": best_alpha,
    }
    return y_true_test, y_pred_test, runtime


# ============================================================
# 11. METRICS
# ============================================================
def safe_global_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() < 2:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    return {
        "mae": float(mean_absolute_error(yt[mask], yp[mask])),
        "rmse": float(math.sqrt(mean_squared_error(yt[mask], yp[mask]))),
        "r2": float(r2_score(yt[mask], yp[mask])),
    }


def per_sensor_metrics(y_true: np.ndarray, y_pred: np.ndarray, tags: List[str], method: str, seed: int) -> pd.DataFrame:
    rows = []
    for j, tag in enumerate(tags):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.sum() < 2:
            mae = rmse = r2 = peak_error = peak_time_error = max_abs = np.nan
        else:
            yt_m = yt[mask]
            yp_m = yp[mask]
            mae = float(mean_absolute_error(yt_m, yp_m))
            rmse = float(math.sqrt(mean_squared_error(yt_m, yp_m)))
            r2 = float(r2_score(yt_m, yp_m))
            peak_error = float(abs(np.max(yt_m) - np.max(yp_m)))
            true_peak_idx = int(np.argmax(yt_m))
            pred_peak_idx = int(np.argmax(yp_m))
            peak_time_error = abs(true_peak_idx - pred_peak_idx) * SAMPLE_INTERVAL_SECONDS / 3600.0
            max_abs = float(np.max(np.abs(yt_m - yp_m)))
        rows.append({
            "method": method,
            "seed": seed,
            "sensor": tag,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "peak_temperature_abs_error": peak_error,
            "peak_timing_abs_error_h": peak_time_error,
            "max_abs_error": max_abs,
        })
    return pd.DataFrame(rows)


def build_run_summary(method: str, seed: int, y_true: np.ndarray, y_pred: np.ndarray, runtime: Dict[str, float], sensor_df: pd.DataFrame) -> Dict[str, float]:
    metrics = safe_global_metrics(y_true, y_pred)
    return {
        "method": method,
        "seed": seed,
        **metrics,
        "mean_sensor_rmse": float(sensor_df["rmse"].mean()),
        "median_sensor_rmse": float(sensor_df["rmse"].median()),
        "p95_sensor_rmse": float(sensor_df["rmse"].quantile(0.95)),
        "worst_sensor_rmse": float(sensor_df["rmse"].max()),
        "mean_peak_temperature_abs_error": float(sensor_df["peak_temperature_abs_error"].mean()),
        "mean_peak_timing_abs_error_h": float(sensor_df["peak_timing_abs_error_h"].mean()),
        "max_abs_error": float(sensor_df["max_abs_error"].max()),
        **runtime,
    }


# ============================================================
# 12. RUN ALL METHODS
# ============================================================
test_tags = [sensor_names[i] for i in fixed_test_idx]
model_seeds = [BASE_MODEL_SEED + i * SEED_STRIDE for i in range(N_SEEDS)]

best_hparam_rows = []
best_hparams_by_method = {}
for method in ["GNN-GRU", "GRU"]:
    best = tune_neural_hparams(method)
    best_hparams_by_method[method] = best
    best_hparam_rows.append({"method": method, **best})
pd.DataFrame(best_hparam_rows).to_csv(OUT_DIR / "best_hyperparameters.csv", index=False)

all_run_rows = []
all_sensor_frames = []
predictions_by_method_seed: Dict[str, List[np.ndarray]] = {method: [] for method in MODEL_ORDER}
y_true_reference = None

for method in ["GNN-GRU", "GRU"]:
    hparams = best_hparams_by_method[method]
    for seed in model_seeds:
        y_true, y_pred, runtime = train_and_evaluate_neural(method, seed, hparams)
        if y_true_reference is None:
            y_true_reference = y_true
        elif not np.allclose(y_true_reference, y_true, atol=1e-5, equal_nan=True):
            raise RuntimeError("Neural methods produced misaligned truth arrays")
        sensor_df = per_sensor_metrics(y_true=y_true, y_pred=y_pred, tags=test_tags, method=method, seed=seed)
        run_row = build_run_summary(method=method, seed=seed, y_true=y_true, y_pred=y_pred, runtime=runtime, sensor_df=sensor_df)
        all_sensor_frames.append(sensor_df)
        all_run_rows.append(run_row)
        predictions_by_method_seed[method].append(y_pred)
        method_slug = method.lower().replace("-", "_")
        sensor_df.to_csv(OUT_DIR / f"per_sensor_metrics_{method_slug}_seed_{seed}.csv", index=False)
        print(f"{method} seed={seed}: MAE={run_row['mae']:.4f}, RMSE={run_row['rmse']:.4f}, R2={run_row['r2']:.4f}", flush=True)

if RUN_RIDGE:
    y_true_ridge, y_pred_ridge, ridge_runtime = tune_and_train_ridge()
    if y_true_reference is None:
        y_true_reference = y_true_ridge
    elif not np.allclose(y_true_reference, y_true_ridge, atol=1e-3, equal_nan=True):
        raise RuntimeError("Ridge and neural methods produced misaligned truth arrays")
    ridge_sensor_df = per_sensor_metrics(y_true=y_true_ridge, y_pred=y_pred_ridge, tags=test_tags, method="Ridge", seed=-1)
    ridge_run_row = build_run_summary(method="Ridge", seed=-1, y_true=y_true_ridge, y_pred=y_pred_ridge, runtime=ridge_runtime, sensor_df=ridge_sensor_df)
    all_sensor_frames.append(ridge_sensor_df)
    all_run_rows.append(ridge_run_row)
    predictions_by_method_seed["Ridge"].append(y_pred_ridge)
    ridge_sensor_df.to_csv(OUT_DIR / "per_sensor_metrics_ridge.csv", index=False)
    print(f"Ridge: MAE={ridge_run_row['mae']:.4f}, RMSE={ridge_run_row['rmse']:.4f}, R2={ridge_run_row['r2']:.4f}", flush=True)

all_runs_df = pd.DataFrame(all_run_rows)
all_runs_df["method"] = pd.Categorical(all_runs_df["method"], categories=MODEL_ORDER, ordered=True)
all_runs_df = all_runs_df.sort_values(["method", "seed"])
all_runs_df.to_csv(OUT_DIR / "metrics_all_runs.csv", index=False)

all_sensor_df = pd.concat(all_sensor_frames, ignore_index=True)
all_sensor_df["method"] = pd.Categorical(all_sensor_df["method"], categories=MODEL_ORDER, ordered=True)
all_sensor_df = all_sensor_df.sort_values(["sensor", "method", "seed"])
all_sensor_df.to_csv(OUT_DIR / "per_sensor_metrics_all_methods.csv", index=False)


# ============================================================
# 13. AGGREGATION AND PREDICTION TABLE
# ============================================================
metric_columns = [
    "mae", "rmse", "r2", "mean_sensor_rmse", "median_sensor_rmse", "p95_sensor_rmse",
    "worst_sensor_rmse", "mean_peak_temperature_abs_error", "mean_peak_timing_abs_error_h",
    "max_abs_error", "train_runtime_sec", "eval_runtime_sec", "total_runtime_sec", "parameter_count",
]
aggregate_rows = []
for method in MODEL_ORDER:
    subset = all_runs_df[all_runs_df["method"] == method].copy()
    if subset.empty:
        continue
    row = {"method": method, "n_runs": len(subset)}
    for col in metric_columns:
        if col in subset.columns:
            row[f"{col}_mean"] = float(subset[col].mean())
            row[f"{col}_std"] = float(subset[col].std(ddof=1)) if len(subset) > 1 else 0.0
    aggregate_rows.append(row)
aggregate_df = pd.DataFrame(aggregate_rows)
aggregate_df["method"] = pd.Categorical(aggregate_df["method"], categories=MODEL_ORDER, ordered=True)
aggregate_df = aggregate_df.sort_values("method")
aggregate_df.to_csv(OUT_DIR / "metrics_aggregate.csv", index=False)

mean_prediction_by_method = {}
for method in MODEL_ORDER:
    prediction_list = predictions_by_method_seed.get(method, [])
    if prediction_list:
        mean_prediction_by_method[method] = np.mean(np.stack(prediction_list, axis=0), axis=0)

prediction_columns = {}
for j, tag in enumerate(test_tags):
    prediction_columns[f"{tag}_true"] = y_true_reference[:, j]
    for method in MODEL_ORDER:
        if method in mean_prediction_by_method:
            slug = method.lower().replace("-", "_")
            prediction_columns[f"{tag}_{slug}"] = mean_prediction_by_method[method][:, j]
pd.DataFrame(prediction_columns).to_csv(OUT_DIR / "test_predictions_all_methods.csv", index=False)


# ============================================================
# 14. COMPARISON PLOTS
# ============================================================
def plot_global_metric(metric: str, ylabel: str, filename: str) -> None:
    plot_df = aggregate_df.copy()
    values = plot_df[f"{metric}_mean"].to_numpy(dtype=float)
    errors = plot_df[f"{metric}_std"].to_numpy(dtype=float)
    labels = plot_df["method"].astype(str).tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errors, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=240)
    plt.close(fig)


plot_global_metric("rmse", "RMSE (C)", "global_rmse_comparison.png")
plot_global_metric("mae", "MAE (C)", "global_mae_comparison.png")


def plot_per_sensor_rmse() -> None:
    mean_sensor = all_sensor_df.groupby(["sensor", "method"], observed=True)["rmse"].mean().reset_index()
    pivot = mean_sensor.pivot(index="sensor", columns="method", values="rmse")
    pivot = pivot.reindex(index=test_tags, columns=MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(MODEL_ORDER))
    offset_center = (len(MODEL_ORDER) - 1) / 2.0
    for i, method in enumerate(MODEL_ORDER):
        ax.bar(x + (i - offset_center) * width, pivot[method].to_numpy(dtype=float), width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=90)
    ax.set_ylabel("RMSE (C)")
    ax.legend(ncol=max(1, len(MODEL_ORDER)))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_sensor_rmse_comparison.png", dpi=240)
    plt.close(fig)


plot_per_sensor_rmse()


def choose_representative_sensor_indices() -> List[int]:
    reference_pred = mean_prediction_by_method.get("GNN-GRU")
    if reference_pred is None:
        reference_pred = next(iter(mean_prediction_by_method.values()))
    rmse_values = [math.sqrt(mean_squared_error(y_true_reference[:, j], reference_pred[:, j])) for j in range(len(test_tags))]
    order = np.argsort(rmse_values)
    n_plot = min(PLOT_MAX_SENSORS, len(order))
    if n_plot == len(order):
        return order.tolist()
    positions = np.linspace(0, len(order) - 1, n_plot, dtype=int)
    return [int(order[p]) for p in positions]


def plot_reconstruction_grid() -> None:
    pick = choose_representative_sensor_indices()
    n_panels = len(pick)
    n_cols = 3
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.2 * n_rows), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    t_axis = np.arange(y_true_reference.shape[0])
    for ax_i, ax in enumerate(axes):
        if ax_i >= n_panels:
            ax.axis("off")
            continue
        j = pick[ax_i]
        tag = test_tags[j]
        ax.plot(t_axis, y_true_reference[:, j], label="True", linewidth=1.4)
        for method in MODEL_ORDER:
            pred = mean_prediction_by_method.get(method)
            if pred is not None:
                ax.plot(t_axis, pred[:, j], label=method, linewidth=1.0, linestyle="--")
        ax.set_title(tag, fontsize=9)
        ax.set_ylabel("T (C)")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout(rect=[0, 0, 0.88, 1.0])
    fig.savefig(OUT_DIR / "test_sensor_reconstruction_grid_all_methods.png", dpi=220)
    plt.close(fig)


plot_reconstruction_grid()


# ============================================================
# 15. SUMMARY
# ============================================================
with open(OUT_DIR / "summary.txt", "w") as f:
    f.write("Single-split deterministic reconstruction comparison (PG29 TF sensors)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"TF_CSV: {TF_CSV}\n")
    f.write(f"COORD_CSV: {COORD_CSV}\n")
    f.write(f"Split seed: {SPLIT_SEED}\n")
    f.write(f"Total sensors: {N_sensors}\n")
    f.write(f"Fixed test sensors: {len(fixed_test_idx)}\n")
    f.write(f"Known sensors: {len(official_known_idx)}\n")
    f.write(f"Train-target sensors: {len(official_train_target_idx)}\n")
    f.write(f"WINDOW: {WINDOW}\n")
    f.write(f"K_NEIGHBORS_GRAPH: {K_NEIGHBORS_GRAPH}\n")
    f.write(f"GNN_HIDDEN: {GNN_HIDDEN}\n")
    f.write(f"GRU_HIDDEN: {GRU_HIDDEN}\n")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
    f.write(f"N_EPOCHS: {N_EPOCHS}\n")
    f.write(f"MAX_GRAD_NORM: {MAX_GRAD_NORM}\n")
    f.write("Optimizer: Adam (neural), closed-form (Ridge)\n")
    f.write("Normalization: sklearn.StandardScaler\n\n")
    f.write("No-leakage rule:\n")
    f.write("  Fixed test sensors are removed before the known/train-target split is built.\n")
    f.write("  Neural loss and the Ridge fit objective are computed only on train-target sensors.\n")
    f.write("  Hyperparameter tuning (neural lr/dropout/etc. and Ridge alpha) uses a temporal\n")
    f.write("  train/validation split of train-target sensors only, never the fixed test sensors.\n")
    f.write("  Official evaluation uses only fixed test sensors after training/fitting.\n")
    f.write("  StandardScaler is fit without fixed-test sensors.\n\n")
    f.write("Aggregate metrics:\n")
    f.write(aggregate_df.to_string(index=False))
    f.write("\n")

with open(OUT_DIR / "case_done.txt", "w") as f:
    f.write("done\n")

print("\n" + "=" * 78)
print(f"[OK] Saved all outputs to: {OUT_DIR.resolve()}")
print("Aggregate metrics:")
print(aggregate_df.to_string(index=False))
print("=" * 78)
