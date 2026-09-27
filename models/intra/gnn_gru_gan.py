#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean fixed-test mask-fold GNN-GRU conditional GAN experiment.

Protocol
--------
1) Take the manually specified fixed test sensors first.
   - These sensors are always separated.
   - They are never visible as model input.
   - They are never used in the GAN training loss.
   - They are used only for final evaluation.

2) From the remaining sensors, create K mask folds.
   For HTTF TS with 153 sensors and 29 fixed test sensors:
       remaining = 122 sensors
       each fold: known = 91 sensors, train_target = 31 sensors

3) Train ONE conditional GAN using all K folds.
   - Each training sample uses one fold mask.
   - Input sees only that fold's known sensors through known_seq.
   - Generator reconstruction/adversarial loss is computed only on that fold's train-target sensors.
   - Discriminator sees real/fake values only on train-target sensors, plus known conditioning.
   - All folds update the same generator and discriminator weights.

4) Official final evaluation uses Option B:
   - final known input = fold 1 known sensors, 91 sensors
   - final target/evaluation = fixed test sensors only

5) Optional diagnostic:
   - evaluate the same final generator using all 10 fold-known masks
   - report mean/std/best/worst as sensitivity only
   - official final metric remains fold 1.

Run examples
------------
Smoke test, one case:
    N_CASES=1 python clean_maskfold_fixedtest_optionB_GAN.py

Multiple independent cases:
    N_CASES=20 PARALLEL_CASES=4 GPU_IDS=0,1,2,3 \
    BASE_OUT_DIR=/home/unabila/PG29/fixedtest_optionB_GAN_Ncases \
    python clean_maskfold_fixedtest_optionB_GAN.py

Notes
-----
- No Optuna tuning.
- Hyperparameters are fixed but can be overridden with environment variables.
- This script is intentionally parallel to the GNN-GRU diffusion script, but the generative core is a conditional GAN.
- All comments/strings are plain ASCII to avoid encoding issues.
"""

import os
import sys
import time
import math
import random
import subprocess
from pathlib import Path
from typing import List

# ============================================================
# CPU THREAD LIMITS - set before numpy/sklearn/torch imports
# ============================================================
THREADS_PER_CASE = os.environ.get("THREADS_PER_CASE", "4")
os.environ.setdefault("OMP_NUM_THREADS", THREADS_PER_CASE)
os.environ.setdefault("MKL_NUM_THREADS", THREADS_PER_CASE)
os.environ.setdefault("OPENBLAS_NUM_THREADS", THREADS_PER_CASE)
os.environ.setdefault("NUMEXPR_NUM_THREADS", THREADS_PER_CASE)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", THREADS_PER_CASE)
os.environ.setdefault("BLIS_NUM_THREADS", THREADS_PER_CASE)


# ============================================================
# CONTROLLER MODE
# ============================================================
def _discover_gpu_ids():
    env_gpu_ids = os.environ.get("GPU_IDS", "").strip()
    if env_gpu_ids:
        return [x.strip() for x in env_gpu_ids.split(",") if x.strip()]

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        ids = [line.strip() for line in out.splitlines() if line.strip()]
        return ids if ids else [None]
    except Exception:
        return [None]


def _controller_main():
    n_cases = int(os.environ.get("N_CASES", "600"))
    base_out_dir = BASE_OUT_DIR
    base_out_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = _discover_gpu_ids()
    default_parallel = len(gpu_ids) if gpu_ids != [None] else 1
    parallel_cases = int(os.environ.get("PARALLEL_CASES", str(default_parallel)))
    parallel_cases = max(1, min(parallel_cases, n_cases))

    seed_base = int(os.environ.get("SEED_BASE", "42"))

    print("=" * 78, flush=True)
    print("Launching fixed-test Option-B mask-fold GNN-GRU GAN cases", flush=True)
    print(f"N_CASES        : {n_cases}", flush=True)
    print(f"PARALLEL_CASES : {parallel_cases}", flush=True)
    print(f"GPU_IDS        : {gpu_ids}", flush=True)
    print(f"BASE_OUT_DIR   : {base_out_dir.resolve()}", flush=True)
    print("=" * 78, flush=True)

    pending = list(range(1, n_cases + 1))
    running = []
    completed = []
    failed = []

    while pending or running:
        while pending and len(running) < parallel_cases:
            case_id = pending.pop(0)
            gpu_id = gpu_ids[(case_id - 1) % len(gpu_ids)] if gpu_ids else None
            case_dir = base_out_dir / f"case_{case_id:03d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            log_path = case_dir / "run.log"

            env = os.environ.copy()
            env["WORKER_CASE_ID"] = str(case_id)
            env["CASE_SEED"] = str(seed_base + 1000 * (case_id - 1))
            env["BASE_OUT_DIR"] = str(base_out_dir)
            env["THREADS_PER_CASE"] = os.environ.get("THREADS_PER_CASE", "4")
            env["OMP_NUM_THREADS"] = env["THREADS_PER_CASE"]
            env["MKL_NUM_THREADS"] = env["THREADS_PER_CASE"]
            env["OPENBLAS_NUM_THREADS"] = env["THREADS_PER_CASE"]
            env["NUMEXPR_NUM_THREADS"] = env["THREADS_PER_CASE"]
            env["VECLIB_MAXIMUM_THREADS"] = env["THREADS_PER_CASE"]
            env["BLIS_NUM_THREADS"] = env["THREADS_PER_CASE"]
            env["NUM_WORKERS"] = os.environ.get("NUM_WORKERS", "0")

            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            log_f = open(log_path, "w", buffering=1)
            cmd = [sys.executable, str(Path(__file__).resolve())]
            p = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(Path.cwd()),
            )
            running.append({
                "process": p,
                "case_id": case_id,
                "gpu_id": gpu_id,
                "log_f": log_f,
                "log_path": log_path,
            })
            print(f"[LAUNCH] case_{case_id:03d} on GPU {gpu_id}; log: {log_path}", flush=True)

        time.sleep(5)

        still_running = []
        for item in running:
            p = item["process"]
            rc = p.poll()
            if rc is None:
                still_running.append(item)
                continue

            item["log_f"].close()
            case_id = item["case_id"]
            if rc == 0:
                completed.append(case_id)
                print(f"[DONE] case_{case_id:03d}", flush=True)
            else:
                failed.append(case_id)
                print(f"[FAIL] case_{case_id:03d} returned code {rc}. Check {item['log_path']}", flush=True)
        running = still_running

    # Aggregate case summaries.
    try:
        import pandas as pd
        rows = []
        for case_id in range(1, n_cases + 1):
            summary_path = base_out_dir / f"case_{case_id:03d}" / "metrics_summary.csv"
            if summary_path.exists():
                df = pd.read_csv(summary_path)
                if len(df) > 0:
                    rows.append(df.iloc[0].to_dict())
        if rows:
            all_df = pd.DataFrame(rows).sort_values("official_test_rmse")
            all_df.to_csv(base_out_dir / "all_cases_summary.csv", index=False)
            print(f"[OK] Saved aggregate summary: {base_out_dir / 'all_cases_summary.csv'}", flush=True)
    except Exception as exc:
        print(f"[WARN] Could not aggregate summaries: {exc}", flush=True)

    print("=" * 78, flush=True)
    print(f"Completed cases: {completed}", flush=True)
    print(f"Failed cases   : {failed}", flush=True)
    print("=" * 78, flush=True)

    if failed:
        sys.exit(1)


if "WORKER_CASE_ID" not in os.environ:
    _controller_main()
    sys.exit(0)


# ============================================================
# WORKER MODE IMPORTS
# ============================================================
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Limit PyTorch CPU thread pools inside each worker process.
THREADS_PER_CASE_INT = int(os.environ.get("THREADS_PER_CASE", "4"))
torch.set_num_threads(THREADS_PER_CASE_INT)
torch.set_num_interop_threads(max(1, min(2, THREADS_PER_CASE_INT)))


ROOT = Path(__file__).resolve().parents[2]

TS_CSV = Path(
    os.environ.get(
        "TS_CSV",
        ROOT / "data" / "pg29_TS_30s_common.csv"
    )
)

COORD_CSV = Path(
    os.environ.get(
        "COORD_CSV",
        ROOT / "data" / "core_TC_coordinates_common_norm.csv"
    )
)

BASE_OUT_DIR = Path(
    os.environ.get(
        "BASE_OUT_DIR",
        ROOT / "results" / "intra" / "gnn_gru_GAN"
    )
)
SEED = int(os.environ.get("CASE_SEED", str(42 + 1000 * (CASE_ID - 1))))

# Split sizes for 153 sensors.
N_TEST_EXPECTED = int(os.environ.get("N_TEST_EXPECTED", "29"))
N_KNOWN_PER_FOLD = int(os.environ.get("N_KNOWN_PER_FOLD", "91"))
K_FOLDS = int(os.environ.get("K_FOLDS", "10"))
FINAL_EVAL_FOLD = int(os.environ.get("FINAL_EVAL_FOLD", "0"))  # 0 means fold 1.

# Fixed GNN-GRU conditional GAN hyperparameters. No tuning.
WINDOW = int(os.environ.get("WINDOW", "40"))
GNN_HIDDEN = int(os.environ.get("GNN_HIDDEN", "64"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "192"))
GRU_HIDDEN = int(os.environ.get("GRU_HIDDEN", "256"))
Z_DIM = int(os.environ.get("Z_DIM", "32"))
LR_G = float(os.environ.get("LR_G", "0.0005"))
LR_D = float(os.environ.get("LR_D", "0.0005"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
N_EPOCHS = int(os.environ.get("N_EPOCHS", "150"))
N_SAMPLES_EVAL = int(os.environ.get("N_SAMPLES_EVAL", "3"))
N_SAMPLES_DIAGNOSTIC = int(os.environ.get("N_SAMPLES_DIAGNOSTIC", "1"))
K_NEIGHBORS_GRAPH = int(os.environ.get("K_NEIGHBORS_GRAPH", "8"))
GRAPH_SIGMA = os.environ.get("GRAPH_SIGMA", "").strip()
GRAPH_SIGMA = None if GRAPH_SIGMA == "" else float(GRAPH_SIGMA)

# GAN/training constants.
ADV_LOSS = os.environ.get("ADV_LOSS", "bce").strip().lower()  # bce or lsgan
LAMBDA_ADV = float(os.environ.get("LAMBDA_ADV", "0.1"))
LAMBDA_L1 = float(os.environ.get("LAMBDA_L1", "10.0"))
LAMBDA_MSE = float(os.environ.get("LAMBDA_MSE", "1.0"))
LABEL_SMOOTH_REAL = float(os.environ.get("LABEL_SMOOTH_REAL", "0.9"))
LABEL_SMOOTH_FAKE = float(os.environ.get("LABEL_SMOOTH_FAKE", "0.0"))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", "1.0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
PLOT_MAX_SENSORS = int(os.environ.get("PLOT_MAX_SENSORS", "9"))
SAVE_ALL_MASK_TEST_DIAGNOSTIC = os.environ.get("SAVE_ALL_MASK_TEST_DIAGNOSTIC", "1").strip() not in ["0", "false", "False"]
USE_MULTI_GPU = False


# ============================================================
# 0.1 MANUAL FIXED TEST SET
# ============================================================
# IMPORTANT:
# This list must contain exactly 31 unique sensors for the 20% fixed test set.
MANUAL_TEST_SENSORS = [
    "TS-1105",
    "TS-1119",
    "TS-1133",
    "TS-1602",
    "TS-1113",
    "TS-1312",
    "TS-1327",
    "TS-1309",
    "TS-1315",
    "TS-1343",
    "TS-1501",
    "TS-1505",
    "TS-1515",
    "TS-1537",
    "TS-1528",
    "TS-1539",
    "TS-1543",
    "TS-1724",
    "TS-1705",
    "TS-1715",
    "TS-2103",
    "TS-2107",
    "TS-1915",
    "TS-2111",
    "TS-1902",
    "TS-1329",
    "TS-1742",
    "TS-1727",
    "TS-1723",
]


# ============================================================
# 1. REPRODUCIBILITY + DEVICE
# ============================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count()

print(f"Device     : {DEVICE}")
print(f"GPUs found : {N_GPUS}")
print(f"Case ID    : {CASE_ID}")
print(f"Case seed  : {SEED}")
print(f"OUT_DIR    : {OUT_DIR.resolve()}")


# ============================================================
# 2. DATA LOADING
# ============================================================
ts_df = pd.read_csv(TS_CSV)
coord_df_raw = pd.read_csv(COORD_CSV)

print("TS shape   :", ts_df.shape)
print("Coord shape:", coord_df_raw.shape)

sensor_names = list(ts_df.columns)
N_sensors = len(sensor_names)
all_idx = np.arange(N_sensors, dtype=int)

possible_name_cols = [
    "sensor", "Sensor", "SENSOR",
    "TC_tag", "TC", "tc_tag",
    "tag", "Tag", "name", "Name",
]
name_col = next((c for c in possible_name_cols if c in coord_df_raw.columns), None)
if name_col is None:
    raise ValueError(
        f"Could not find sensor-name column in {COORD_CSV}. "
        f"Available columns: {list(coord_df_raw.columns)}"
    )

coord_df = coord_df_raw.copy()
coord_df[name_col] = coord_df[name_col].astype(str).str.strip()
coord_df = coord_df.set_index(name_col)

missing_from_coord = [s for s in sensor_names if s not in coord_df.index]
if missing_from_coord:
    print(f"WARNING: {len(missing_from_coord)} sensors missing coordinates.")
coord_df = coord_df.reindex(sensor_names)

coord_cols_candidates = ["r_norm", "theta_deg", "z_norm", "z_cm", "theta_rad"]
coord_cols = [c for c in coord_cols_candidates if c in coord_df.columns]
if len(coord_cols) < 3:
    num_cols = coord_df.select_dtypes(include=[np.number]).columns.tolist()
    coord_cols = num_cols[:3]
if len(coord_cols) == 0:
    raise ValueError("No numeric coordinate columns found.")

coords_np = coord_df[coord_cols].values.astype(np.float32)
for i, c in enumerate(coord_cols):
    if "theta" in c.lower() and np.nanmax(np.abs(coords_np[:, i])) > 2.5:
        coords_np[:, i] = coords_np[:, i] / 180.0
coords_np = np.nan_to_num(coords_np, nan=0.0)
coords_torch = torch.tensor(coords_np, dtype=torch.float32, device=DEVICE)


def build_knn_adjacency(coords, k=8, sigma=None):
    """
    Build a normalized kNN adjacency matrix from sensor coordinates.

    The graph is used by the GNN encoder only. It does not change the
    sensor split, the known mask, or the train/test no-leakage rule.
    """
    coords = np.asarray(coords, dtype=np.float32)

    if coords.shape[1] >= 3:
        r = coords[:, 0]
        theta_raw = coords[:, 1]
        z = coords[:, 2]
    elif coords.shape[1] == 2:
        r = np.ones(coords.shape[0], dtype=np.float32)
        theta_raw = coords[:, 0]
        z = coords[:, 1]
    else:
        raise ValueError("coords must have at least 2 columns to build graph adjacency.")

    if np.nanmax(np.abs(theta_raw)) > 2.5:
        theta = np.deg2rad(theta_raw)
    else:
        theta = theta_raw * np.pi

    xyz = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1).astype(np.float32)
    diff = xyz[:, None, :] - xyz[None, :, :]
    dist2 = np.sum(diff ** 2, axis=-1)
    dist = np.sqrt(dist2 + 1e-12)

    if sigma is None:
        nonzero = dist[dist > 0]
        sigma = np.median(nonzero) if len(nonzero) > 0 else 1.0
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0

    N = coords.shape[0]
    A = np.zeros((N, N), dtype=np.float32)
    k = min(int(k), max(1, N - 1))

    for i in range(N):
        nn_idx = np.argsort(dist[i])[1:k + 1]
        weights = np.exp(-dist2[i, nn_idx] / (2.0 * sigma ** 2))
        A[i, nn_idx] = weights.astype(np.float32)

    A = np.maximum(A, A.T)
    A = A + np.eye(N, dtype=np.float32)
    deg = np.sum(A, axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(deg + 1e-8)
    A_norm = deg_inv_sqrt[:, None] * A * deg_inv_sqrt[None, :]
    return A_norm.astype(np.float32)


A_np = build_knn_adjacency(coords_np, k=K_NEIGHBORS_GRAPH, sigma=GRAPH_SIGMA)
A_torch = torch.tensor(A_np, dtype=torch.float32, device=DEVICE)

X_np = ts_df.values.astype(np.float32)
T_total = X_np.shape[0]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_np).astype(np.float32)

print("Using coord columns:", coord_cols)
print("coords_np shape    :", coords_np.shape)
print("Adjacency shape    :", A_np.shape)
print("Graph k neighbors  :", K_NEIGHBORS_GRAPH)
print("Total timesteps    :", T_total)
print("Total sensors      :", N_sensors)


# ============================================================
# 3. BASIC HELPERS + SPLIT
# ============================================================
def save_sensor_index_csv(path, idx):
    idx = np.asarray(idx, dtype=int)
    pd.DataFrame({
        "idx": idx,
        "sensor": [sensor_names[i] for i in idx],
    }).to_csv(path, index=False)


def write_python_sensor_list(path: Path, var_name: str, sensors: List[str]):
    with open(path, "w") as f:
        f.write(f"{var_name} = [\n")
        for s in sensors:
            f.write(f'    "{s}",\n')
        f.write("]\n")


def sensor_tags_to_indices(tags, role_name):
    clean_tags = [str(t).strip() for t in tags if str(t).strip()]
    duplicate_tags = pd.Series(clean_tags).value_counts()
    duplicate_tags = duplicate_tags[duplicate_tags > 1].index.tolist()
    if duplicate_tags:
        raise ValueError(f"Duplicate sensors in {role_name}: {duplicate_tags}")

    name_to_idx = {name.strip(): i for i, name in enumerate(sensor_names)}
    missing = [tag for tag in clean_tags if tag not in name_to_idx]
    if missing:
        raise ValueError(
            f"These {role_name} sensors were not found in TS_CSV columns: {missing}\n"
            f"First available sensors: {sensor_names[:20]}"
        )
    return np.sort(np.asarray([name_to_idx[tag] for tag in clean_tags], dtype=int))


fixed_test_idx = sensor_tags_to_indices(MANUAL_TEST_SENSORS, "fixed_test")
if len(fixed_test_idx) != N_TEST_EXPECTED:
    raise ValueError(
        f"MANUAL_TEST_SENSORS must contain exactly {N_TEST_EXPECTED} unique sensors. "
        f"Found {len(fixed_test_idx)}."
    )

remaining_idx = np.array(sorted(set(all_idx.tolist()) - set(fixed_test_idx.tolist())), dtype=int)
if len(remaining_idx) <= N_KNOWN_PER_FOLD:
    raise ValueError(
        f"N_KNOWN_PER_FOLD={N_KNOWN_PER_FOLD} is too large for remaining sensors={len(remaining_idx)}."
    )

n_train_target_per_fold = len(remaining_idx) - N_KNOWN_PER_FOLD
if n_train_target_per_fold != N_TEST_EXPECTED:
    print(
        f"[WARN] train-target count per fold is {n_train_target_per_fold}, "
        f"not {N_TEST_EXPECTED}. Check test count and N_KNOWN_PER_FOLD.",
        flush=True,
    )

print("\nFixed split:")
print(f"Total sensors          : {N_sensors}")
print(f"Fixed test sensors     : {len(fixed_test_idx)}")
print(f"Remaining sensors      : {len(remaining_idx)}")
print(f"Known per fold         : {N_KNOWN_PER_FOLD}")
print(f"Train-target per fold  : {n_train_target_per_fold}")
print(f"Mask folds             : {K_FOLDS}")
print(f"Official eval fold     : {FINAL_EVAL_FOLD + 1}")


def make_fold_split(remaining_idx, n_known, fold_seed):
    rng_fold = np.random.default_rng(fold_seed)
    known_idx = np.sort(rng_fold.choice(remaining_idx, n_known, replace=False))
    train_target_idx = np.array(sorted(set(remaining_idx.tolist()) - set(known_idx.tolist())), dtype=int)
    assert len(set(known_idx.tolist()) & set(train_target_idx.tolist())) == 0
    assert len(set(fixed_test_idx.tolist()) & set(known_idx.tolist())) == 0
    assert len(set(fixed_test_idx.tolist()) & set(train_target_idx.tolist())) == 0
    return {
        "known_idx": known_idx,
        "train_target_idx": train_target_idx,
        "test_target_idx": fixed_test_idx,
        "trainable_pool_idx": remaining_idx,
    }


folds = [make_fold_split(remaining_idx, N_KNOWN_PER_FOLD, SEED + k) for k in range(K_FOLDS)]
if not (0 <= FINAL_EVAL_FOLD < K_FOLDS):
    raise ValueError(f"FINAL_EVAL_FOLD must be between 0 and {K_FOLDS - 1}.")

print("\nGenerated mask folds:")
for k, fold in enumerate(folds):
    print(
        f"  Fold {k + 1}: known={len(fold['known_idx'])}, "
        f"train_target={len(fold['train_target_idx'])}, test={len(fixed_test_idx)}"
    )

save_sensor_index_csv(OUT_DIR / "fixed_test_sensors.csv", fixed_test_idx)
save_sensor_index_csv(OUT_DIR / "remaining_sensors.csv", remaining_idx)
write_python_sensor_list(OUT_DIR / "MANUAL_TEST_SENSORS.py", "MANUAL_TEST_SENSORS", [sensor_names[i] for i in fixed_test_idx])
for k, fold in enumerate(folds):
    save_sensor_index_csv(OUT_DIR / f"fold{k + 1}_known_sensors.csv", fold["known_idx"])
    save_sensor_index_csv(OUT_DIR / f"fold{k + 1}_train_target_sensors.csv", fold["train_target_idx"])

# Fold 1 official known set.
official_known_idx = folds[FINAL_EVAL_FOLD]["known_idx"]
save_sensor_index_csv(OUT_DIR / "official_eval_known_sensors_fold1.csv", official_known_idx)
write_python_sensor_list(OUT_DIR / "OFFICIAL_EVAL_KNOWN_SENSORS.py", "OFFICIAL_EVAL_KNOWN_SENSORS", [sensor_names[i] for i in official_known_idx])


# ============================================================
# 4. PLOTTING HELPERS
# ============================================================
def get_xyz_from_coords(coords):
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape[1] >= 3:
        r = coords[:, 0]
        theta_raw = coords[:, 1]
        z = coords[:, 2]
    elif coords.shape[1] == 2:
        r = np.ones(coords.shape[0], dtype=np.float32)
        theta_raw = coords[:, 0]
        z = coords[:, 1]
    else:
        raise ValueError("coords_np must have at least 2 columns.")

    if np.nanmax(np.abs(theta_raw)) > 2.5:
        theta = np.deg2rad(theta_raw)
    else:
        theta = theta_raw * np.pi

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1).astype(np.float32)


XYZ = get_xyz_from_coords(coords_np)


def plot_sensor_split_3d(filename, known_idx, train_target_idx, test_idx, title):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        XYZ[train_target_idx, 0], XYZ[train_target_idx, 1], XYZ[train_target_idx, 2],
        s=35, alpha=0.55, marker="^", label=f"Train target ({len(train_target_idx)})"
    )
    ax.scatter(
        XYZ[known_idx, 0], XYZ[known_idx, 1], XYZ[known_idx, 2],
        s=45, alpha=0.75, marker="o", label=f"Known input ({len(known_idx)})"
    )
    ax.scatter(
        XYZ[test_idx, 0], XYZ[test_idx, 1], XYZ[test_idx, 2],
        s=75, alpha=0.85, marker="x", label=f"Fixed test ({len(test_idx)})"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=22, azim=45)
    fig.tight_layout()
    fig.savefig(filename, dpi=240)
    plt.close(fig)


plot_sensor_split_3d(
    OUT_DIR / "official_fold1_split_3d.png",
    known_idx=official_known_idx,
    train_target_idx=folds[FINAL_EVAL_FOLD]["train_target_idx"],
    test_idx=fixed_test_idx,
    title="Official Option-B split: fold 1 known, fold 1 train-target, fixed test",
)


def plot_sensors_grid(y_true, y_pred, target_tags, r2_per_sensor, filename, label="GAN"):
    n_sensors = len(target_tags)
    if n_sensors == 0:
        return

    order = np.argsort(np.nan_to_num(r2_per_sensor, nan=-999.0))
    if n_sensors <= PLOT_MAX_SENSORS:
        indices_to_plot = order.tolist()
    else:
        worst3 = order[:3].tolist()
        best3 = order[-3:].tolist()
        mid = n_sensors // 2
        mid3 = sorted({max(0, mid - 1), mid, min(n_sensors - 1, mid + 1)})
        seen, indices_to_plot = set(), []
        for idx in worst3 + mid3 + best3:
            if idx not in seen:
                seen.add(idx)
                indices_to_plot.append(idx)
        indices_to_plot = indices_to_plot[:PLOT_MAX_SENSORS]

    t_axis = np.arange(y_true.shape[0])
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    for ax_idx in range(9):
        if ax_idx < len(indices_to_plot):
            j = indices_to_plot[ax_idx]
            axes[ax_idx].plot(t_axis, y_true[:, j], label="GT", linewidth=1.5)
            axes[ax_idx].plot(t_axis, y_pred[:, j], label=label, linewidth=1.0, linestyle="--")
            axes[ax_idx].set_title(f"{target_tags[j]} R2={r2_per_sensor[j]:.3f}", fontsize=9)
            axes[ax_idx].grid(True, alpha=0.3)
            axes[ax_idx].set_ylabel("T")
        else:
            axes[ax_idx].axis("off")

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")
            break
    fig.tight_layout(rect=[0, 0, 0.85, 1.0])
    fig.savefig(filename, dpi=200)
    plt.close(fig)


# ============================================================
# 5. DATASETS
# ============================================================
class BaseWindowDataset(Dataset):
    def __init__(self, X_scaled, window):
        self.X = X_scaled.astype(np.float32)
        self.window = int(window)
        if self.X.shape[0] < self.window:
            raise ValueError(f"Data length {self.X.shape[0]} < window {self.window}")

    def __len__(self):
        return self.X.shape[0] - self.window + 1

    def __getitem__(self, idx):
        x_window = self.X[idx:idx + self.window].copy()
        x0_true = x_window[-1].copy()
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(x0_true, dtype=torch.float32)


class MaskFoldFullFieldGANDataset(Dataset):
    """
    Training dataset using all mask folds.

    Returns:
        x0_true    : (N,)
        known_seq  : (L, N, 2), channels = [known temperature, known mask]
        train_mask : (N,) bool, true only for the fold train-target sensors
        known_mask : (N,) float, true only for the fold known sensors
    """
    def __init__(self, X_scaled, folds, window, fast=False):
        self.base = BaseWindowDataset(X_scaled, window)
        self.folds = folds
        self.K = len(folds)
        self.fast = bool(fast)

    def __len__(self):
        if self.fast:
            return len(self.base)
        return len(self.base) * self.K

    def __getitem__(self, idx):
        if self.fast:
            base_idx = idx
            fold_k = random.randint(0, self.K - 1)
        else:
            base_idx = idx // self.K
            fold_k = idx % self.K

        x_window, x0_true = self.base[base_idx]
        fold = self.folds[fold_k]
        known_idx = fold["known_idx"]
        train_idx = fold["train_target_idx"]

        known_mask = torch.zeros(x_window.shape[1], dtype=torch.float32)
        known_mask[known_idx] = 1.0

        train_mask = torch.zeros(x_window.shape[1], dtype=torch.bool)
        train_mask[train_idx] = True

        x_known = x_window.clone()
        x_known[:, known_mask == 0] = 0.0
        known_mask_seq = known_mask.unsqueeze(0).expand_as(x_known)
        known_seq = torch.stack([x_known, known_mask_seq], dim=-1)

        return x0_true, known_seq, train_mask, known_mask


class FixedKnownFullFieldGANDataset(Dataset):
    """
    Evaluation dataset using one fixed known mask.
    """
    def __init__(self, X_scaled, known_idx, window):
        self.base = BaseWindowDataset(X_scaled, window)
        self.known_idx = np.asarray(known_idx, dtype=int)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x_window, x0_true = self.base[idx]
        known_mask = torch.zeros(x_window.shape[1], dtype=torch.float32)
        known_mask[self.known_idx] = 1.0
        x_known = x_window.clone()
        x_known[:, known_mask == 0] = 0.0
        known_mask_seq = known_mask.unsqueeze(0).expand_as(x_known)
        known_seq = torch.stack([x_known, known_mask_seq], dim=-1)
        return x0_true, known_seq, known_mask


def make_train_loader(window, batch_size, fast=False):
    ds = MaskFoldFullFieldGANDataset(X_scaled=X_scaled, folds=folds, window=window, fast=fast)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
    )


def make_eval_loader(known_idx, window, batch_size):
    ds = FixedKnownFullFieldGANDataset(X_scaled=X_scaled, known_idx=known_idx, window=window)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
    )


# ============================================================
# 6. GNN-GRU CONDITIONAL GAN MODEL
# ============================================================
class GraphConv(nn.Module):
    """
    Simple graph convolution using the fixed sensor adjacency matrix.

    x: (B, N, F)
    A: (N, N)
    output: (B, N, H)
    """
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A):
        x = self.dropout(x)
        h = torch.einsum("ij,bjf->bif", A, x)
        return self.linear(h)


class GNNKnownSequenceEncoder(nn.Module):
    """
    Encodes the known sparse sensor history with a GNN at each time step,
    followed by a temporal GRU over each sensor's graph embedding sequence.

    known_seq: (B, L, N, 2), channels = [known temperature, known mask]
    A        : (N, N), normalized adjacency

    Returns:
        graph_context: (B, gru_hidden), pooled global temporal context
        node_context : (B, N, gru_hidden), per-sensor temporal context
    """
    def __init__(self, n_sensors, gnn_hidden, gru_hidden, dropout=0.05):
        super().__init__()
        self.n_sensors = n_sensors
        self.gnn_hidden = gnn_hidden
        self.gru_hidden = gru_hidden

        self.gcn1 = GraphConv(2, gnn_hidden, dropout=dropout)
        self.gcn2 = GraphConv(gnn_hidden, gnn_hidden, dropout=dropout)
        self.temporal_gru = nn.GRU(
            input_size=gnn_hidden,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, known_seq, A):
        B, L, N, C = known_seq.shape
        if N != self.n_sensors:
            raise ValueError(f"Expected {self.n_sensors} sensors, got {N}")
        if C != 2:
            raise ValueError(f"Expected 2 channels, got {C}")

        gnn_outputs = []
        for step in range(L):
            xt = known_seq[:, step, :, :]
            h = F.silu(self.gcn1(xt, A))
            h = F.silu(self.gcn2(h, A))
            gnn_outputs.append(h)

        h_seq = torch.stack(gnn_outputs, dim=1)
        h_seq_sensor = h_seq.permute(0, 2, 1, 3).contiguous()
        h_seq_sensor = h_seq_sensor.view(B * N, L, self.gnn_hidden)

        gru_out, _ = self.temporal_gru(h_seq_sensor)
        node_context = gru_out[:, -1, :].view(B, N, self.gru_hidden)
        graph_context = node_context.mean(dim=1)
        return graph_context, node_context


class CoordEncoder(nn.Module):
    def __init__(self, coord_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )

    def forward(self, coords):
        return self.net(coords)


class FullFieldGNNGRUGenerator(nn.Module):
    """
    Full-field conditional GNN-GRU generator.

    The generator predicts every sensor value at the current time step, but
    training losses are evaluated only on the current fold's train-target sensors.
    Known sensors are hard-conditioned to their observed current values.
    """
    def __init__(
        self,
        n_sensors,
        coord_dim,
        z_dim=32,
        gnn_hidden=64,
        hidden_dim=192,
        gru_hidden=256,
        dropout=0.05,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.z_dim = z_dim
        self.temp_enc = GNNKnownSequenceEncoder(
            n_sensors=n_sensors,
            gnn_hidden=gnn_hidden,
            gru_hidden=gru_hidden,
            dropout=dropout,
        )
        self.coord_enc = CoordEncoder(coord_dim, hidden_dim)
        self.z_global = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )

        per_sensor_in = 1 + 1 + hidden_dim + hidden_dim + gru_hidden + gru_hidden + z_dim
        self.head = nn.Sequential(
            nn.Linear(per_sensor_in, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, known_seq, known_mask, coords, A, z_node=None, hard_condition_known=True):
        B, _, N, _ = known_seq.shape
        if N != self.n_sensors:
            raise ValueError(f"Expected {self.n_sensors} sensors, got {N}")
        if z_node is None:
            z_node = torch.randn(B, N, self.z_dim, device=known_seq.device)

        known_current = known_seq[:, -1, :, 0]
        graph_context, node_context = self.temp_enc(known_seq, A)
        h_coord = self.coord_enc(coords)
        z_pool = z_node.mean(dim=1)
        h_z_global = self.z_global(z_pool)

        features = torch.cat([
            known_current.unsqueeze(-1),
            known_mask.unsqueeze(-1),
            h_coord.unsqueeze(0).expand(B, -1, -1),
            h_z_global.unsqueeze(1).expand(-1, N, -1),
            graph_context.unsqueeze(1).expand(-1, N, -1),
            node_context,
            z_node,
        ], dim=-1)
        raw = self.head(features).squeeze(-1)

        if hard_condition_known:
            raw = known_mask * known_current + (1.0 - known_mask) * raw
        return raw


class FullFieldGNNGRUDiscriminator(nn.Module):
    """
    Full-field conditional GNN-GRU discriminator.

    It produces one real/fake logit per sensor. During training, adversarial
    losses are evaluated only on the current fold's train-target sensors.
    """
    def __init__(
        self,
        n_sensors,
        coord_dim,
        gnn_hidden=64,
        hidden_dim=192,
        gru_hidden=256,
        dropout=0.05,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.temp_enc = GNNKnownSequenceEncoder(
            n_sensors=n_sensors,
            gnn_hidden=gnn_hidden,
            gru_hidden=gru_hidden,
            dropout=dropout,
        )
        self.coord_enc = CoordEncoder(coord_dim, hidden_dim)

        per_sensor_in = 1 + 1 + 1 + 1 + hidden_dim + gru_hidden + gru_hidden
        self.head = nn.Sequential(
            nn.Linear(per_sensor_in, hidden_dim), nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidate_x, known_seq, known_mask, target_mask, coords, A):
        B, N = candidate_x.shape
        if N != self.n_sensors:
            raise ValueError(f"Expected {self.n_sensors} sensors, got {N}")

        known_current = known_seq[:, -1, :, 0]
        graph_context, node_context = self.temp_enc(known_seq, A)
        h_coord = self.coord_enc(coords)
        target_float = target_mask.float()

        features = torch.cat([
            candidate_x.unsqueeze(-1),
            known_current.unsqueeze(-1),
            known_mask.unsqueeze(-1),
            target_float.unsqueeze(-1),
            h_coord.unsqueeze(0).expand(B, -1, -1),
            graph_context.unsqueeze(1).expand(-1, N, -1),
            node_context,
        ], dim=-1)
        return self.head(features).squeeze(-1)


def build_models():
    generator = FullFieldGNNGRUGenerator(
        n_sensors=N_sensors,
        coord_dim=coords_torch.shape[1],
        z_dim=Z_DIM,
        gnn_hidden=GNN_HIDDEN,
        hidden_dim=HIDDEN_DIM,
        gru_hidden=GRU_HIDDEN,
    ).to(DEVICE)
    discriminator = FullFieldGNNGRUDiscriminator(
        n_sensors=N_sensors,
        coord_dim=coords_torch.shape[1],
        gnn_hidden=GNN_HIDDEN,
        hidden_dim=HIDDEN_DIM,
        gru_hidden=GRU_HIDDEN,
    ).to(DEVICE)
    if USE_MULTI_GPU and torch.cuda.device_count() > 1:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
    return generator, discriminator


def get_core(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def masked_adv_loss(logits, mask, target_value):
    if ADV_LOSS == "lsgan":
        target = torch.full_like(logits, float(target_value))
        return F.mse_loss(logits[mask], target[mask])

    target = torch.full_like(logits, float(target_value))
    return F.binary_cross_entropy_with_logits(logits[mask], target[mask])


def make_real_candidate(x0_true, known_seq, known_mask, train_mask):
    known_current = known_seq[:, -1, :, 0]
    train_float = train_mask.float()
    return known_mask * known_current + train_float * x0_true


def make_fake_candidate(fake_full, known_seq, known_mask, train_mask):
    known_current = known_seq[:, -1, :, 0]
    train_float = train_mask.float()
    return known_mask * known_current + train_float * fake_full


# ============================================================
# 7. TRAINING + EVALUATION
# ============================================================
def train_one_epoch(generator, discriminator, loader, optimizer_g, optimizer_d):
    generator.train()
    discriminator.train()
    totals = {
        "d_loss": 0.0,
        "d_real": 0.0,
        "d_fake": 0.0,
        "g_loss": 0.0,
        "g_adv": 0.0,
        "g_l1": 0.0,
        "g_mse": 0.0,
    }
    n_batches = 0

    for x0_true, known_seq, train_mask, known_mask in loader:
        x0_true = x0_true.to(DEVICE, non_blocking=True)
        known_seq = known_seq.to(DEVICE, non_blocking=True)
        train_mask = train_mask.to(DEVICE, non_blocking=True)
        known_mask = known_mask.to(DEVICE, non_blocking=True)

        B = x0_true.shape[0]

        # ------------------------------
        # Train discriminator.
        # ------------------------------
        with torch.no_grad():
            z_node = torch.randn(B, N_sensors, Z_DIM, device=DEVICE)
            fake_full = generator(
                known_seq=known_seq,
                known_mask=known_mask,
                coords=coords_torch,
                A=A_torch,
                z_node=z_node,
                hard_condition_known=True,
            )
            fake_candidate = make_fake_candidate(fake_full, known_seq, known_mask, train_mask)

        real_candidate = make_real_candidate(x0_true, known_seq, known_mask, train_mask)

        logits_real = discriminator(
            candidate_x=real_candidate,
            known_seq=known_seq,
            known_mask=known_mask,
            target_mask=train_mask,
            coords=coords_torch,
            A=A_torch,
        )
        logits_fake = discriminator(
            candidate_x=fake_candidate.detach(),
            known_seq=known_seq,
            known_mask=known_mask,
            target_mask=train_mask,
            coords=coords_torch,
            A=A_torch,
        )

        d_real = masked_adv_loss(logits_real, train_mask, LABEL_SMOOTH_REAL)
        d_fake = masked_adv_loss(logits_fake, train_mask, LABEL_SMOOTH_FAKE)
        d_loss = 0.5 * (d_real + d_fake)

        optimizer_d.zero_grad()
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(get_core(discriminator).parameters(), MAX_GRAD_NORM)
        optimizer_d.step()

        # ------------------------------
        # Train generator.
        # ------------------------------
        z_node = torch.randn(B, N_sensors, Z_DIM, device=DEVICE)
        fake_full = generator(
            known_seq=known_seq,
            known_mask=known_mask,
            coords=coords_torch,
            A=A_torch,
            z_node=z_node,
            hard_condition_known=True,
        )
        fake_candidate = make_fake_candidate(fake_full, known_seq, known_mask, train_mask)
        logits_fake_for_g = discriminator(
            candidate_x=fake_candidate,
            known_seq=known_seq,
            known_mask=known_mask,
            target_mask=train_mask,
            coords=coords_torch,
            A=A_torch,
        )

        g_adv = masked_adv_loss(logits_fake_for_g, train_mask, LABEL_SMOOTH_REAL)
        g_l1 = F.l1_loss(fake_full[train_mask], x0_true[train_mask])
        g_mse = F.mse_loss(fake_full[train_mask], x0_true[train_mask])
        g_loss = LAMBDA_ADV * g_adv + LAMBDA_L1 * g_l1 + LAMBDA_MSE * g_mse

        optimizer_g.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(get_core(generator).parameters(), MAX_GRAD_NORM)
        optimizer_g.step()

        totals["d_loss"] += d_loss.item()
        totals["d_real"] += d_real.item()
        totals["d_fake"] += d_fake.item()
        totals["g_loss"] += g_loss.item()
        totals["g_adv"] += g_adv.item()
        totals["g_l1"] += g_l1.item()
        totals["g_mse"] += g_mse.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def sample_full_field(generator, known_seq, known_mask, n_samples=1):
    generator.eval()
    B, _, N, _ = known_seq.shape
    samples_out = []

    for _ in range(n_samples):
        z_node = torch.randn(B, N, Z_DIM, device=DEVICE)
        pred = generator(
            known_seq=known_seq,
            known_mask=known_mask,
            coords=coords_torch,
            A=A_torch,
            z_node=z_node,
            hard_condition_known=True,
        )
        samples_out.append(pred.detach().cpu())

    return torch.stack(samples_out, dim=0)


@torch.no_grad()
def collect_predictions(generator, loader, eval_idx, n_samples):
    generator.eval()
    all_true, all_pred = [], []

    for x0_true, known_seq, known_mask in loader:
        x0_true = x0_true.to(DEVICE, non_blocking=True)
        known_seq = known_seq.to(DEVICE, non_blocking=True)
        known_mask = known_mask.to(DEVICE, non_blocking=True)

        samples = sample_full_field(
            generator=generator,
            known_seq=known_seq,
            known_mask=known_mask,
            n_samples=n_samples,
        )
        pred_mean = samples.mean(dim=0).numpy()
        all_true.append(x0_true.cpu().numpy())
        all_pred.append(pred_mean)

    y_true_scaled = np.concatenate(all_true, axis=0)
    y_pred_scaled = np.concatenate(all_pred, axis=0)
    y_true = scaler.inverse_transform(y_true_scaled)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    return y_true[:, eval_idx], y_pred[:, eval_idx]


def compute_global_rmse(y_true, y_pred):
    yt, yp = y_true.flatten(), y_pred.flatten()
    mask = np.isfinite(yt) & np.isfinite(yp)
    return math.sqrt(mean_squared_error(yt[mask], yp[mask])) if mask.sum() > 0 else 1e6


def compute_global_mae(y_true, y_pred):
    yt, yp = y_true.flatten(), y_pred.flatten()
    mask = np.isfinite(yt) & np.isfinite(yp)
    return mean_absolute_error(yt[mask], yp[mask]) if mask.sum() > 0 else np.nan


def compute_global_r2(y_true, y_pred):
    yt, yp = y_true.flatten(), y_pred.flatten()
    mask = np.isfinite(yt) & np.isfinite(yp)
    return r2_score(yt[mask], yp[mask]) if mask.sum() >= 2 else np.nan


def compute_per_sensor(y_true, y_pred):
    D = y_true.shape[1]
    rmse_arr = np.zeros(D)
    mae_arr = np.zeros(D)
    r2_arr = np.zeros(D)
    for j in range(D):
        yt, yp = y_true[:, j], y_pred[:, j]
        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.sum() < 2:
            rmse_arr[j] = np.nan
            mae_arr[j] = np.nan
            r2_arr[j] = np.nan
        else:
            rmse_arr[j] = math.sqrt(mean_squared_error(yt[mask], yp[mask]))
            mae_arr[j] = mean_absolute_error(yt[mask], yp[mask])
            r2_arr[j] = r2_score(yt[mask], yp[mask])
    return r2_arr, rmse_arr, mae_arr


# ============================================================
# 8. TRAIN ONE FINAL MODEL USING ALL MASK FOLDS
# ============================================================
print("\n" + "=" * 78)
print("Training one full-field GNN-GRU conditional GAN using all mask folds")
print("=" * 78)
print(f"WINDOW          : {WINDOW}")
print(f"GNN_HIDDEN      : {GNN_HIDDEN}")
print(f"HIDDEN_DIM      : {HIDDEN_DIM}")
print(f"GRU_HIDDEN      : {GRU_HIDDEN}")
print(f"Z_DIM           : {Z_DIM}")
print(f"LR_G            : {LR_G}")
print(f"LR_D            : {LR_D}")
print(f"BATCH_SIZE      : {BATCH_SIZE}")
print(f"N_EPOCHS        : {N_EPOCHS}")
print(f"N_SAMPLES_EVAL  : {N_SAMPLES_EVAL}")
print(f"ADV_LOSS        : {ADV_LOSS}")
print(f"LAMBDA_ADV      : {LAMBDA_ADV}")
print(f"LAMBDA_L1       : {LAMBDA_L1}")
print(f"LAMBDA_MSE      : {LAMBDA_MSE}")
print("=" * 78)

train_loader = make_train_loader(window=WINDOW, batch_size=BATCH_SIZE, fast=False)
generator, discriminator = build_models()
optimizer_g = torch.optim.Adam(get_core(generator).parameters(), lr=LR_G, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(get_core(discriminator).parameters(), lr=LR_D, betas=(0.5, 0.999))

epoch_logs = []
start_train = time.time()

for epoch in range(1, N_EPOCHS + 1):
    stats = train_one_epoch(
        generator=generator,
        discriminator=discriminator,
        loader=train_loader,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
    )
    if epoch == 1 or epoch % 10 == 0 or epoch == N_EPOCHS:
        row = {"epoch": epoch, **stats}
        epoch_logs.append(row)
        print(
            f"Epoch {epoch:04d}/{N_EPOCHS} | "
            f"D={stats['d_loss']:.6f} | "
            f"G={stats['g_loss']:.6f} | "
            f"Adv={stats['g_adv']:.6f} | "
            f"L1={stats['g_l1']:.6f} | "
            f"MSE={stats['g_mse']:.6f}",
            flush=True,
        )

train_runtime_sec = time.time() - start_train
pd.DataFrame(epoch_logs).to_csv(OUT_DIR / "training_log.csv", index=False)

# Save final model before evaluation.
torch.save(
    {
        "generator_state_dict": get_core(generator).state_dict(),
        "discriminator_state_dict": get_core(discriminator).state_dict(),
        "sensor_names": sensor_names,
        "coord_cols": coord_cols,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "fixed_test_idx": fixed_test_idx,
        "remaining_idx": remaining_idx,
        "folds": folds,
        "official_eval_fold": FINAL_EVAL_FOLD,
        "official_known_idx": official_known_idx,
        "hyperparameters": {
            "window": WINDOW,
            "gnn_hidden": GNN_HIDDEN,
            "hidden_dim": HIDDEN_DIM,
            "gru_hidden": GRU_HIDDEN,
            "z_dim": Z_DIM,
            "lr_g": LR_G,
            "lr_d": LR_D,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "adv_loss": ADV_LOSS,
            "lambda_adv": LAMBDA_ADV,
            "lambda_l1": LAMBDA_L1,
            "lambda_mse": LAMBDA_MSE,
        },
    },
    OUT_DIR / "final_gnn_gru_gan_model.pt",
)


# ============================================================
# 9. OFFICIAL FINAL TEST: FOLD 1 KNOWN INPUT -> FIXED TEST TARGETS
# ============================================================
print("\n" + "=" * 78)
print("Official final test: fold 1 known sensors -> fixed test sensors")
print("=" * 78)

start_eval = time.time()
official_loader = make_eval_loader(known_idx=official_known_idx, window=WINDOW, batch_size=BATCH_SIZE)
y_true_test, y_pred_test = collect_predictions(
    generator=generator,
    loader=official_loader,
    eval_idx=fixed_test_idx,
    n_samples=N_SAMPLES_EVAL,
)
official_eval_runtime_sec = time.time() - start_eval

official_test_rmse = compute_global_rmse(y_true_test, y_pred_test)
official_test_mae = compute_global_mae(y_true_test, y_pred_test)
official_test_r2 = compute_global_r2(y_true_test, y_pred_test)
r2_per_sensor, rmse_per_sensor, mae_per_sensor = compute_per_sensor(y_true_test, y_pred_test)
test_tags = [sensor_names[i] for i in fixed_test_idx]

print(f"Official known sensors: {len(official_known_idx)}")
print(f"Fixed test sensors    : {len(fixed_test_idx)}")
print(f"Official test RMSE    : {official_test_rmse:.6f}")
print(f"Official test MAE     : {official_test_mae:.6f}")
print(f"Official test R2      : {official_test_r2:.6f}")

pd.DataFrame({
    "idx": fixed_test_idx,
    "sensor": test_tags,
    "rmse": rmse_per_sensor,
    "mae": mae_per_sensor,
    "r2": r2_per_sensor,
}).to_csv(OUT_DIR / "metrics_test_sensors_official_fold1.csv", index=False)

pred_cols = []
pred_data = {}
for j, tag in enumerate(test_tags):
    true_col = f"{tag}_true"
    pred_col = f"{tag}_pred"
    pred_cols.extend([true_col, pred_col])
    pred_data[true_col] = y_true_test[:, j]
    pred_data[pred_col] = y_pred_test[:, j]
pd.DataFrame(pred_data, columns=pred_cols).to_csv(OUT_DIR / "test_predictions_official_fold1.csv", index=False)

plot_sensors_grid(
    y_true=y_true_test,
    y_pred=y_pred_test,
    target_tags=test_tags,
    r2_per_sensor=r2_per_sensor,
    filename=OUT_DIR / "test_sensor_reconstruction_grid_official_fold1.png",
    label="GAN",
)


# ============================================================
# 10. OPTIONAL DIAGNOSTIC: FINAL GENERATOR TESTED UNDER ALL KNOWN MASKS
# ============================================================
diagnostic_rows = []
if SAVE_ALL_MASK_TEST_DIAGNOSTIC:
    print("\nDiagnostic: evaluating final generator under all fold-known masks")
    for k, fold in enumerate(folds):
        fold_loader = make_eval_loader(known_idx=fold["known_idx"], window=WINDOW, batch_size=BATCH_SIZE)
        y_true_k, y_pred_k = collect_predictions(
            generator=generator,
            loader=fold_loader,
            eval_idx=fixed_test_idx,
            n_samples=N_SAMPLES_DIAGNOSTIC,
        )
        rmse_k = compute_global_rmse(y_true_k, y_pred_k)
        mae_k = compute_global_mae(y_true_k, y_pred_k)
        r2_k = compute_global_r2(y_true_k, y_pred_k)
        diagnostic_rows.append({
            "fold": k + 1,
            "n_known": len(fold["known_idx"]),
            "n_train_target": len(fold["train_target_idx"]),
            "test_rmse": rmse_k,
            "test_mae": mae_k,
            "test_r2": r2_k,
            "official_metric": (k == FINAL_EVAL_FOLD),
        })
        print(f"  Fold {k + 1}: RMSE={rmse_k:.4f}, MAE={mae_k:.4f}, R2={r2_k:.4f}", flush=True)

    pd.DataFrame(diagnostic_rows).to_csv(OUT_DIR / "test_metrics_by_known_mask_diagnostic.csv", index=False)


# ============================================================
# 11. SUMMARY
# ============================================================
case_runtime_sec = train_runtime_sec + official_eval_runtime_sec
summary = {
    "case_id": CASE_ID,
    "seed": SEED,
    "out_dir": str(OUT_DIR.resolve()),
    "model": "full_field_gnn_gru_conditional_gan",
    "n_sensors": N_sensors,
    "n_fixed_test": len(fixed_test_idx),
    "n_remaining": len(remaining_idx),
    "n_known_per_fold": N_KNOWN_PER_FOLD,
    "n_train_target_per_fold": n_train_target_per_fold,
    "k_folds": K_FOLDS,
    "official_eval_fold": FINAL_EVAL_FOLD + 1,
    "official_test_rmse": official_test_rmse,
    "official_test_mae": official_test_mae,
    "official_test_r2": official_test_r2,
    "window": WINDOW,
    "gnn_hidden": GNN_HIDDEN,
    "hidden_dim": HIDDEN_DIM,
    "gru_hidden": GRU_HIDDEN,
    "z_dim": Z_DIM,
    "lr_g": LR_G,
    "lr_d": LR_D,
    "batch_size": BATCH_SIZE,
    "n_epochs": N_EPOCHS,
    "n_samples_eval": N_SAMPLES_EVAL,
    "adv_loss": ADV_LOSS,
    "lambda_adv": LAMBDA_ADV,
    "lambda_l1": LAMBDA_L1,
    "lambda_mse": LAMBDA_MSE,
    "train_runtime_sec": train_runtime_sec,
    "official_eval_runtime_sec": official_eval_runtime_sec,
    "case_runtime_sec": case_runtime_sec,
}

if diagnostic_rows:
    diag_df = pd.DataFrame(diagnostic_rows)
    summary.update({
        "diagnostic_mean_test_rmse": float(diag_df["test_rmse"].mean()),
        "diagnostic_std_test_rmse": float(diag_df["test_rmse"].std()),
        "diagnostic_best_test_rmse": float(diag_df["test_rmse"].min()),
        "diagnostic_worst_test_rmse": float(diag_df["test_rmse"].max()),
    })

pd.DataFrame([summary]).to_csv(OUT_DIR / "metrics_summary.csv", index=False)

with open(OUT_DIR / "summary.txt", "w") as f:
    f.write("Fixed-test Option-B mask-fold GNN-GRU conditional GAN model\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"TS_CSV: {TS_CSV}\n")
    f.write(f"COORD_CSV: {COORD_CSV}\n")
    f.write(f"Total sensors: {N_sensors}\n")
    f.write(f"Fixed test sensors: {len(fixed_test_idx)}\n")
    f.write(f"Remaining sensors: {len(remaining_idx)}\n")
    f.write(f"Known per fold: {N_KNOWN_PER_FOLD}\n")
    f.write(f"Train-target per fold: {n_train_target_per_fold}\n")
    f.write(f"K folds: {K_FOLDS}\n")
    f.write(f"Official final known input: fold {FINAL_EVAL_FOLD + 1}\n\n")
    f.write("No-leakage rule:\n")
    f.write("  Fixed test sensors are removed before fold construction.\n")
    f.write("  Fixed test sensors are never visible in known_seq.\n")
    f.write("  Fixed test sensors are never included in train_mask.\n")
    f.write("  GAN adversarial and reconstruction losses are computed only on each fold's train-target sensors.\n")
    f.write("  Official final metric uses fold 1 known sensors as input and fixed test sensors as targets.\n\n")
    f.write("Official final test performance:\n")
    f.write(f"  RMSE: {official_test_rmse}\n")
    f.write(f"  MAE : {official_test_mae}\n")
    f.write(f"  R2  : {official_test_r2}\n")

with open(OUT_DIR / "case_done.txt", "w") as f:
    f.write("done\n")

print(f"\n[OK] All outputs saved to: {OUT_DIR.resolve()}")
