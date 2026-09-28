#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Section-context conditional VAE model for PG27 -> PG29 TS reconstruction.

"""

import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["BLIS_NUM_THREADS"] = "8"

import math
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(2)

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIG
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]

PG27_PATH = Path(
    os.environ.get(
        "PG27_PATH",
        ROOT / "data" / "pg27_TS_30s_common_temp.csv"
    )
)

PG29_PATH = Path(
    os.environ.get(
        "PG29_PATH",
        ROOT / "data" / "pg29_TS_30s_common.csv"
    )
)

COORD_PATH = Path(
    os.environ.get(
        "COORD_PATH",
        ROOT / "data" / "core_TC_coordinates_common_norm.csv"
    )
)

OUT_DIR = Path(
    os.environ.get(
        "OUT_DIR",
        ROOT / "results" / "inter" / "section_context_local_vae"
    )
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = int(os.environ.get("SEED", "42"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MANUAL_TEST_SENSORS = [
    "TS-1105", "TS-1119", "TS-1133", "TS-1602", "TS-1113",
    "TS-1312", "TS-1327", "TS-1309", "TS-1315",
    "TS-1343", "TS-1501", "TS-1505", "TS-1515",
    "TS-1537", "TS-1528", "TS-1539", "TS-1543", "TS-1724",
    "TS-1705", "TS-1715", "TS-2103", "TS-2107", "TS-1915",
    "TS-2111", "TS-1902", "TS-1329", "TS-1742", "TS-1727",
    "TS-1723",
]

# Fast training defaults. Increase EPOCHS or SAMPLES_PER_EPOCH only if needed.
WINDOW = int(os.environ.get("WINDOW", "40"))
PHYS_K = int(os.environ.get("PHYS_K", "8"))
SECTION_K = int(os.environ.get("SECTION_K", "6"))
CONTEXT_K = PHYS_K + SECTION_K
N_SECTIONS = int(os.environ.get("N_SECTIONS", "6"))

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
EPOCHS = int(os.environ.get("EPOCHS", "80"))
SAMPLES_PER_EPOCH = int(os.environ.get("SAMPLES_PER_EPOCH", "30000"))
LR = float(os.environ.get("LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-6"))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", "1.0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))

# VAE sampling and latent controls.
N_SAMPLES_EVAL = int(os.environ.get("N_SAMPLES_EVAL", "4"))
LATENT_DIM = int(os.environ.get("LATENT_DIM", "24"))
BETA_KL = float(os.environ.get("BETA_KL", "0.015"))
KL_WARMUP_EPOCHS = int(os.environ.get("KL_WARMUP_EPOCHS", "30"))
KL_FREE_BITS = float(os.environ.get("KL_FREE_BITS", "0.00"))
PRIOR_SAMPLE_STD_SCALE = float(os.environ.get("PRIOR_SAMPLE_STD_SCALE", "0.75"))

# Model size.
SEQ_HIDDEN = int(os.environ.get("SEQ_HIDDEN", "128"))
SET_HIDDEN = int(os.environ.get("SET_HIDDEN", "256"))
TARGET_HIDDEN = int(os.environ.get("TARGET_HIDDEN", "128"))
DROPOUT = float(os.environ.get("DROPOUT", "0.05"))

# Geometry distance.
DIST_R_WEIGHT = 1.0
DIST_THETA_WEIGHT = 1.0
DIST_Z_WEIGHT = 10.0

# Section selection scoring.
SECTION_SCORE_BEHAVIOR = float(os.environ.get("SECTION_SCORE_BEHAVIOR", "1.00"))
SECTION_SCORE_GEO = float(os.environ.get("SECTION_SCORE_GEO", "0.10"))
SECTION_MISMATCH_PENALTY = float(os.environ.get("SECTION_MISMATCH_PENALTY", "0.75"))
SECTION_WEIGHT_STRENGTH = float(os.environ.get("SECTION_WEIGHT_STRENGTH", "1.50"))
SECTION_MISSING_STRENGTH = float(os.environ.get("SECTION_MISSING_STRENGTH", "1.00"))

# Training target sampling.
TRAIN_TARGET_MODE = os.environ.get("TRAIN_TARGET_MODE", "fixed_plus_random")
FIXED_TARGET_PROB = float(os.environ.get("FIXED_TARGET_PROB", "0.60"))

# Loss weights.
LAMBDA_RECON = float(os.environ.get("LAMBDA_RECON", "1.00"))
LAMBDA_PRIOR_RECON = float(os.environ.get("LAMBDA_PRIOR_RECON", "1.00"))
LAMBDA_TEMP = float(os.environ.get("LAMBDA_TEMP", "1.20"))
LAMBDA_LOCAL_DELTA = float(os.environ.get("LAMBDA_LOCAL_DELTA", "0.10"))
LAMBDA_PHYS_DELTA = float(os.environ.get("LAMBDA_PHYS_DELTA", "0.20"))
LAMBDA_SECTION_DELTA = float(os.environ.get("LAMBDA_SECTION_DELTA", "0.35"))
LAMBDA_SLOPE = float(os.environ.get("LAMBDA_SLOPE", "0.15"))
LAMBDA_RES_MAG = float(os.environ.get("LAMBDA_RES_MAG", "0.001"))
LAMBDA_KL = float(os.environ.get("LAMBDA_KL", "1.00"))

# Clever weighted-MSE controls. These affect the VAE loss directly.
# The main temperature losses use local_scale, not pg27_global_std.
LOCAL_SCALE_MIN_C = float(os.environ.get("LOCAL_SCALE_MIN_C", "25.0"))
LOCAL_SCALE_MAX_C = float(os.environ.get("LOCAL_SCALE_MAX_C", "250.0"))

BLOCK_WEIGHT_STRENGTH = float(os.environ.get("BLOCK_WEIGHT_STRENGTH", "0.75"))
RADIAL_CENTER_STRENGTH = float(os.environ.get("RADIAL_CENTER_STRENGTH", "1.25"))
RADIAL_EDGE_STRENGTH = float(os.environ.get("RADIAL_EDGE_STRENGTH", "0.25"))
DIST_KNOWN_STRENGTH = float(os.environ.get("DIST_KNOWN_STRENGTH", "1.25"))
MISSING_NEIGHBOR_STRENGTH = float(os.environ.get("MISSING_NEIGHBOR_STRENGTH", "1.00"))
MISSING_NEIGHBOR_SIGMA = float(os.environ.get("MISSING_NEIGHBOR_SIGMA", "0.35"))
SENSOR_WEIGHT_MIN = float(os.environ.get("SENSOR_WEIGHT_MIN", "0.35"))
SENSOR_WEIGHT_MAX = float(os.environ.get("SENSOR_WEIGHT_MAX", "4.00"))

EVENT_RESID_STRENGTH = float(os.environ.get("EVENT_RESID_STRENGTH", "1.00"))
EVENT_SLOPE_STRENGTH = float(os.environ.get("EVENT_SLOPE_STRENGTH", "0.80"))

# Anchor and calibration.
ANCHOR_K = int(os.environ.get("ANCHOR_K", "8"))
ANCHOR_ALPHA = float(os.environ.get("ANCHOR_ALPHA", "1.0"))
PSEUDO_MAX_TARGETS = int(os.environ.get("PSEUDO_MAX_TARGETS", "40"))
ADAPT_STRIDE = int(os.environ.get("ADAPT_STRIDE", "8"))
LOCAL_CAL_NEIGHBORS = int(os.environ.get("LOCAL_CAL_NEIGHBORS", "8"))
LOCAL_GATE_NEIGHBORS = int(os.environ.get("LOCAL_GATE_NEIGHBORS", "8"))
RELIABILITY_TAU_C = float(os.environ.get("RELIABILITY_TAU_C", "25.0"))

PLOT_MAX_SENSORS = 12


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 100)
print("Section-context local-scale LocationQuery-VAE reconstruction")
print("Device:", DEVICE)
print("Output:", OUT_DIR.resolve())
print("=" * 100)


# ============================================================
# 3. BASIC HELPERS
# ============================================================
def ts_cols(df):
    return sorted([c for c in df.columns if str(c).strip().upper().startswith("TS-")])


def find_name_col(df):
    candidates = [
        "tag", "Tag", "sensor", "Sensor", "SENSOR",
        "TC_tag", "TC", "tc_tag", "name", "Name",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        vals = df[c].dropna().astype(str).str.strip()
        if len(vals) == 0:
            continue
        hit = vals.str.match(r"^(TS|TF|TC)-?\d+", na=False).mean()
        if hit > 0.30:
            return c

    raise ValueError(f"No sensor-name column found. Columns: {list(df.columns)}")


def angular_distance(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def get_cylindrical_features(coord_df, sensor_names):
    name_col = find_name_col(coord_df)
    cdf = coord_df.copy()
    cdf[name_col] = cdf[name_col].astype(str).str.strip()
    cdf = cdf.set_index(name_col).reindex(sensor_names)

    numeric_cols = cdf.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric coordinate columns found.")

    if "r_norm" in cdf.columns:
        r = cdf["r_norm"].astype(float).values
    else:
        r = cdf[numeric_cols[0]].astype(float).values

    if "theta_deg" in cdf.columns:
        theta = np.deg2rad(cdf["theta_deg"].astype(float).values)
    elif "theta_rad" in cdf.columns:
        theta = cdf["theta_rad"].astype(float).values
    else:
        raw = cdf[numeric_cols[min(1, len(numeric_cols) - 1)]].astype(float).values
        theta = np.deg2rad(raw) if np.nanmax(np.abs(raw)) > 2.5 else raw * np.pi

    if "z_norm" in cdf.columns:
        z = cdf["z_norm"].astype(float).values
    elif "z_cm" in cdf.columns:
        zr = cdf["z_cm"].astype(float).values
        z = (zr - np.nanmin(zr)) / (np.nanmax(zr) - np.nanmin(zr) + 1e-12)
    else:
        z = cdf[numeric_cols[min(2, len(numeric_cols) - 1)]].astype(float).values

    r = np.nan_to_num(r, nan=0.0).astype(np.float32)
    theta = np.nan_to_num(theta, nan=0.0).astype(np.float32)
    z = np.nan_to_num(z, nan=0.0).astype(np.float32)
    return r, theta, z


def sensor_number(sensor):
    m = re.search(r"(\d+)", str(sensor))
    if m is None:
        return None
    return int(m.group(1))


def sensor_parity(sensor):
    n = sensor_number(sensor)
    if n is None:
        return -1
    return n % 2


def block_prefix(sensor):
    n = sensor_number(sensor)
    if n is None:
        return -1
    return n // 100


def metric_values(y_true, y_pred):
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    m = np.isfinite(yt) & np.isfinite(yp)
    if int(m.sum()) == 0:
        return np.nan, np.nan, np.nan
    mae = mean_absolute_error(yt[m], yp[m])
    rmse = math.sqrt(mean_squared_error(yt[m], yp[m]))
    r2 = r2_score(yt[m], yp[m]) if int(m.sum()) >= 2 else np.nan
    return mae, rmse, r2


def corrcoef_safe(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3:
        return np.nan
    aa = a[m]
    bb = b[m]
    if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def normalize_shape(y):
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(y)
    if int(m.sum()) == 0:
        return np.zeros_like(y), 0.0, 1.0
    mu = float(np.nanmean(y[m]))
    sd = float(np.nanstd(y[m]) + 1e-12)
    z = (y - mu) / sd
    z[~np.isfinite(z)] = 0.0
    return z, mu, sd


def affine_fit(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 5:
        return 1.0, 0.0

    X = np.column_stack([x[m], np.ones(int(m.sum()))])
    coef, _, _, _ = np.linalg.lstsq(X, y[m], rcond=None)
    a = float(np.clip(coef[0], -3.0, 3.0))
    b = float(coef[1])
    return a, b


def relation_label(i, j):
    same_group = int(group_id[i]) == int(group_id[j])
    same_z = int(z_bin[i]) == int(z_bin[j])
    same_r = int(r_bin[i]) == int(r_bin[j])
    adj_z = abs(int(z_bin[i]) - int(z_bin[j])) == 1
    same_parity = sensor_parity(sensor_names[i]) == sensor_parity(sensor_names[j])
    same_prefix = block_prefix(sensor_names[i]) == block_prefix(sensor_names[j])

    labels = []
    if same_group:
        labels.append("same_group")
    if same_z:
        labels.append("same_axial")
    if same_r:
        labels.append("same_radial")
    if adj_z:
        labels.append("adjacent_axial")
    if same_prefix:
        labels.append("same_prefix")
    if same_parity:
        labels.append("same_odd_even")
    if not labels:
        labels.append("other")
    return "+".join(labels)


def relation_priority(i, j):
    lab = relation_label(i, j)
    score = 0
    if "same_group" in lab:
        score += 50
    if "same_axial" in lab:
        score += 20
    if "same_radial" in lab:
        score += 15
    if "adjacent_axial" in lab:
        score += 8
    if "same_prefix" in lab:
        score += 8
    if "same_odd_even" in lab:
        score += 4
    return score


def save_idx_csv(path, idx):
    pd.DataFrame({
        "idx": np.asarray(idx, dtype=int),
        "sensor": [sensor_names[int(i)] for i in idx],
    }).to_csv(path, index=False)


def replace_bad_with_fallback(arr, fallback):
    arr = np.asarray(arr, dtype=np.float32).copy()
    fallback = np.asarray(fallback, dtype=np.float32)
    bad = ~np.isfinite(arr)
    if int(bad.sum()) > 0:
        arr[bad] = fallback[bad]
    bad2 = ~np.isfinite(arr)
    if int(bad2.sum()) > 0:
        arr[bad2] = 0.0
    return arr


# ============================================================
# 4. LOAD DATA
# ============================================================
pg27 = pd.read_csv(PG27_PATH)
pg29 = pd.read_csv(PG29_PATH)
coord_df = pd.read_csv(COORD_PATH)

pg27_ts = set(ts_cols(pg27))
pg29_ts = set(ts_cols(pg29))
name_col = find_name_col(coord_df)
coord_ts = set([
    s for s in coord_df[name_col].astype(str).str.strip().values
    if str(s).upper().startswith("TS")
])

sensor_names = sorted(pg27_ts & pg29_ts & coord_ts)
if len(sensor_names) < 10:
    raise ValueError(f"Too few common TS sensors: {len(sensor_names)}")

pg27 = pg27[sensor_names].astype(float).ffill().bfill()
pg29 = pg29[sensor_names].astype(float).ffill().bfill()

X27_raw = pg27.values.astype(np.float32)
X29_raw = pg29.values.astype(np.float32)
N_sensors = len(sensor_names)

r_feat, theta_feat, z_feat = get_cylindrical_features(coord_df, sensor_names)

sensor_to_idx = {s: i for i, s in enumerate(sensor_names)}
manual_available = [s for s in MANUAL_TEST_SENSORS if s in sensor_to_idx]
manual_unavailable = [s for s in MANUAL_TEST_SENSORS if s not in sensor_to_idx]

FIXED_MISSING_IDX = np.array(sorted([sensor_to_idx[s] for s in manual_available]), dtype=int)
FIXED_KNOWN_IDX = np.array(
    sorted(set(range(N_sensors)) - set(FIXED_MISSING_IDX.tolist())),
    dtype=int,
)

print("\nLoaded:")
print("PG27:", X27_raw.shape)
print("PG29:", X29_raw.shape)
print("Common sensors:", N_sensors)
print("Known:", len(FIXED_KNOWN_IDX), "Missing:", len(FIXED_MISSING_IDX))
print("Unavailable manual:", manual_unavailable)

pd.DataFrame({"sensor": sensor_names}).to_csv(OUT_DIR / "common_sensors_TS.csv", index=False)
save_idx_csv(OUT_DIR / "fixed_known_sensors.csv", FIXED_KNOWN_IDX)
save_idx_csv(OUT_DIR / "fixed_missing_sensors.csv", FIXED_MISSING_IDX)

# Groups.
N_AXIAL_BINS = 6
N_RADIAL_BINS = 4
z_edges = np.quantile(z_feat, np.linspace(0.0, 1.0, N_AXIAL_BINS + 1))
r_edges = np.quantile(r_feat, np.linspace(0.0, 1.0, N_RADIAL_BINS + 1))
z_edges[0] -= 1e-6
z_edges[-1] += 1e-6
r_edges[0] -= 1e-6
r_edges[-1] += 1e-6

z_bin = np.digitize(z_feat, z_edges[1:-1], right=False)
r_bin = np.digitize(r_feat, r_edges[1:-1], right=False)
group_id = z_bin * N_RADIAL_BINS + r_bin

# Distance matrix.
D2_np = np.zeros((N_sensors, N_sensors), dtype=np.float32)
for i in range(N_sensors):
    dr = r_feat[i] - r_feat
    dt = angular_distance(theta_feat[i], theta_feat) / np.pi
    dz = z_feat[i] - z_feat
    D2_np[i] = DIST_R_WEIGHT * dr ** 2 + DIST_THETA_WEIGHT * dt ** 2 + DIST_Z_WEIGHT * dz ** 2
    D2_np[i, i] = np.inf

def make_behavior_features(X_raw):
    X = np.asarray(X_raw, dtype=np.float32)
    Tn, Ns = X.shape
    mu = np.mean(X, axis=0, keepdims=True)
    sd = np.std(X, axis=0, keepdims=True) + 1e-6
    Z = (X - mu) / sd
    dZ = np.diff(Z, axis=0)

    peak_idx = np.argmax(Z, axis=0).astype(np.float32) / max(float(Tn - 1), 1.0)
    valley_idx = np.argmin(Z, axis=0).astype(np.float32) / max(float(Tn - 1), 1.0)
    peak_z = np.max(Z, axis=0)
    valley_z = np.min(Z, axis=0)
    max_heat = np.max(dZ, axis=0)
    max_cool = np.min(dZ, axis=0)
    rms_slope = np.sqrt(np.mean(dZ ** 2, axis=0))
    heat_idx = np.argmax(dZ, axis=0).astype(np.float32) / max(float(Tn - 2), 1.0)
    cool_idx = np.argmin(dZ, axis=0).astype(np.float32) / max(float(Tn - 2), 1.0)

    q = np.linspace(0, Tn, 7).astype(int)
    seg_mean = []
    seg_slope = []
    for a, b in zip(q[:-1], q[1:]):
        b = max(b, a + 1)
        part = Z[a:b]
        seg_mean.append(np.mean(part, axis=0))
        if part.shape[0] >= 2:
            seg_slope.append(np.mean(np.diff(part, axis=0), axis=0))
        else:
            seg_slope.append(np.zeros(Ns, dtype=np.float32))

    feats = [
        peak_idx,
        valley_idx,
        peak_z,
        valley_z,
        max_heat,
        max_cool,
        rms_slope,
        heat_idx,
        cool_idx,
    ]
    feats.extend(seg_mean)
    feats.extend(seg_slope)
    F = np.stack(feats, axis=1).astype(np.float32)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F


print("\nClustering sensors into transient/peak sections from PG27...")
BEHAVIOR_FEATURES = make_behavior_features(X27_raw)
behavior_scaler = StandardScaler()
BEHAVIOR_FEATURES_Z = behavior_scaler.fit_transform(BEHAVIOR_FEATURES).astype(np.float32)
SECTION_COUNT = int(min(max(2, N_SECTIONS), N_sensors))
section_model = KMeans(n_clusters=SECTION_COUNT, random_state=SEED, n_init=20)
SECTION_ID = section_model.fit_predict(BEHAVIOR_FEATURES_Z).astype(int)
section_centers = section_model.cluster_centers_.astype(np.float32)

Bdiff = BEHAVIOR_FEATURES_Z[:, None, :] - BEHAVIOR_FEATURES_Z[None, :, :]
B2_np = np.mean(Bdiff ** 2, axis=2).astype(np.float32)
np.fill_diagonal(B2_np, np.inf)

section_onehot = np.eye(SECTION_COUNT, dtype=np.float32)[SECTION_ID]
section_id_norm = SECTION_ID.astype(np.float32) / max(float(SECTION_COUNT - 1), 1.0)

pd.DataFrame({
    "idx": np.arange(N_sensors),
    "sensor": sensor_names,
    "section_id": SECTION_ID,
    "r_norm": r_feat,
    "z_norm": z_feat,
    "peak_time_norm": BEHAVIOR_FEATURES[:, 0],
    "valley_time_norm": BEHAVIOR_FEATURES[:, 1],
    "peak_z": BEHAVIOR_FEATURES[:, 2],
    "valley_z": BEHAVIOR_FEATURES[:, 3],
    "max_heat_rate_z": BEHAVIOR_FEATURES[:, 4],
    "max_cool_rate_z": BEHAVIOR_FEATURES[:, 5],
    "rms_slope_z": BEHAVIOR_FEATURES[:, 6],
}).to_csv(OUT_DIR / "pg27_peak_transient_sections.csv", index=False)

base_geo = np.stack(
    [
        r_feat,
        np.sin(theta_feat),
        np.cos(theta_feat),
        z_feat,
        z_bin / max(float(np.max(z_bin)), 1.0),
        r_bin / max(float(np.max(r_bin)), 1.0),
        (r_feat < 0.10).astype(np.float32),
        section_id_norm.astype(np.float32),
    ],
    axis=1,
).astype(np.float32)
sensor_geo_np = np.concatenate([base_geo, section_onehot], axis=1).astype(np.float32)

GEO_DIM = sensor_geo_np.shape[1]
PAIR_GEO_DIM = GEO_DIM * 2 + 4 + 5

# These are retained only for reporting/backward compatibility.
# The core reconstruction loss and sample scaling below use local scale only.
pg27_global_mean = float(np.mean(X27_raw))
pg27_global_std = float(np.std(X27_raw) + 1e-6)
pg27_diff_std = float(np.std(np.diff(X27_raw, axis=0)) + 1e-6)


def make_sensor_loss_weights():
    # 1. Block sparsity weight. Sparse axial-radial blocks get higher loss weight.
    gid = group_id.astype(int)
    n_blocks = int(N_AXIAL_BINS * N_RADIAL_BINS)
    counts = np.bincount(gid, minlength=n_blocks).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    block_sparse = 1.0 / np.sqrt(counts[gid])
    block_sparse = block_sparse / (np.mean(block_sparse) + 1e-12)
    block_w = 1.0 + BLOCK_WEIGHT_STRENGTH * (block_sparse - 1.0)

    # 2. Radial weight. Center sensors get stronger weight by default.
    rmin = float(np.min(r_feat))
    rmax = float(np.max(r_feat))
    r_norm = (r_feat - rmin) / (rmax - rmin + 1e-12)
    radial_w = 1.0 + RADIAL_CENTER_STRENGTH * (1.0 - r_norm) ** 2
    radial_w += RADIAL_EDGE_STRENGTH * (r_norm ** 2)
    radial_w = radial_w / (np.mean(radial_w) + 1e-12)

    # 3. Distance-to-known weight. Hard-to-anchor targets get higher weight.
    d_known = np.min(D2_np[:, FIXED_KNOWN_IDX], axis=1)
    finite = np.isfinite(d_known)
    if np.any(finite):
        max_finite = float(np.max(d_known[finite]))
    else:
        max_finite = 1.0
    d_known = np.nan_to_num(d_known, nan=max_finite, posinf=max_finite, neginf=0.0)
    d_known_norm = d_known / (np.mean(d_known) + 1e-12)
    known_w = 1.0 + DIST_KNOWN_STRENGTH * d_known_norm
    known_w = known_w / (np.mean(known_w) + 1e-12)

    # 4. Missing-neighborhood weight. Training targets near final missing sensors matter more.
    Dtmp = D2_np.copy()
    np.fill_diagonal(Dtmp, 0.0)
    d_missing = np.min(Dtmp[:, FIXED_MISSING_IDX], axis=1)
    missing_w = 1.0 + MISSING_NEIGHBOR_STRENGTH * np.exp(
        -d_missing / (2.0 * MISSING_NEIGHBOR_SIGMA ** 2)
    )
    missing_w = missing_w / (np.mean(missing_w) + 1e-12)

    # 5. Peak/transient section weight. Sections are built from PG27 peak time,
    # transient slope, and shape features. Sparse sections and sections that
    # contain final missing sensors get higher loss weight.
    sid = SECTION_ID.astype(int)
    sec_counts = np.bincount(sid, minlength=SECTION_COUNT).astype(np.float32)
    sec_counts = np.maximum(sec_counts, 1.0)
    sec_sparse = 1.0 / np.sqrt(sec_counts[sid])
    sec_sparse = sec_sparse / (np.mean(sec_sparse) + 1e-12)
    section_w = 1.0 + SECTION_WEIGHT_STRENGTH * (sec_sparse - 1.0)

    missing_sec = np.bincount(SECTION_ID[FIXED_MISSING_IDX].astype(int), minlength=SECTION_COUNT).astype(np.float32)
    if float(np.max(missing_sec)) > 0.0:
        missing_sec = missing_sec / (np.max(missing_sec) + 1e-12)
    section_missing_w = 1.0 + SECTION_MISSING_STRENGTH * missing_sec[sid]
    section_missing_w = section_missing_w / (np.mean(section_missing_w) + 1e-12)

    sensor_w = block_w * radial_w * known_w * missing_w * section_w * section_missing_w
    sensor_w = np.clip(sensor_w, SENSOR_WEIGHT_MIN, SENSOR_WEIGHT_MAX)
    sensor_w = sensor_w / (np.mean(sensor_w) + 1e-12)

    pd.DataFrame({
        "idx": np.arange(N_sensors),
        "sensor": sensor_names,
        "block_id": gid,
        "r_norm": r_norm,
        "block_w": block_w,
        "radial_w": radial_w,
        "known_distance_w": known_w,
        "missing_neighborhood_w": missing_w,
        "section_id": SECTION_ID,
        "section_w": section_w,
        "section_missing_w": section_missing_w,
        "final_sensor_loss_w": sensor_w,
    }).to_csv(OUT_DIR / "sensor_loss_weights_block_radial_distance_section.csv", index=False)

    return torch.tensor(sensor_w.astype(np.float32), device=DEVICE)


SENSOR_LOSS_W = make_sensor_loss_weights()


# ============================================================
# 5. SECTION AND CONTEXT SELECTION
# ============================================================
def context_physical(target_idx, allowed_idx, k):
    allowed_idx = np.asarray(allowed_idx, dtype=int)
    allowed_idx = allowed_idx[allowed_idx != int(target_idx)]
    dist = D2_np[int(target_idx), allowed_idx]
    order = np.argsort(dist)[:min(int(k), len(allowed_idx))]
    out = allowed_idx[order]
    return out.astype(int)


def section_score_pg27(target_i, cand_j):
    target_i = int(target_i)
    cand_j = int(cand_j)
    same_section = float(SECTION_ID[target_i] == SECTION_ID[cand_j])
    beh = float(B2_np[target_i, cand_j])
    geo = float(D2_np[target_i, cand_j])
    mismatch = 0.0 if same_section > 0.5 else SECTION_MISMATCH_PENALTY
    rel_bonus = 0.01 * relation_priority(target_i, cand_j)
    score = SECTION_SCORE_BEHAVIOR * beh + SECTION_SCORE_GEO * min(geo, 5.0) + mismatch - rel_bonus
    return score, beh, geo, same_section


def context_section(target_idx, allowed_idx, k):
    allowed_idx = np.asarray(allowed_idx, dtype=int)
    allowed_idx = allowed_idx[allowed_idx != int(target_idx)]

    rows = []
    for cand_j in allowed_idx:
        score, beh, geo, same_section = section_score_pg27(int(target_idx), int(cand_j))
        rows.append((int(cand_j), score, beh, geo, same_section))

    rows = sorted(rows, key=lambda x: x[1])
    out = [r[0] for r in rows[:min(int(k), len(rows))]]
    return np.asarray(out, dtype=int), rows[:min(int(k), len(rows))]


def build_context_cache(target_list, allowed_idx, label):
    cache = {}
    section_rows = []
    for target_idx in target_list:
        phys = context_physical(int(target_idx), allowed_idx, PHYS_K)
        section_ctx, section_info = context_section(int(target_idx), allowed_idx, SECTION_K)

        # Keep physical first and section-similar second. Remove duplicates while
        # preserving order, then pad to CONTEXT_K.
        seq = []
        ctype = []
        for idx in phys:
            if int(idx) not in seq:
                seq.append(int(idx))
                ctype.append(0.0)
        for idx in section_ctx:
            if int(idx) not in seq:
                seq.append(int(idx))
                ctype.append(1.0)

        if len(seq) == 0:
            raise ValueError("No context sensors found.")

        while len(seq) < CONTEXT_K:
            seq.append(seq[0])
            ctype.append(0.0)

        seq = np.asarray(seq[:CONTEXT_K], dtype=int)
        ctype = np.asarray(ctype[:CONTEXT_K], dtype=np.float32)

        cache[int(target_idx)] = {
            "ctx": seq,
            "ctx_type": ctype,
        }

        for rank, info in enumerate(section_info):
            cand_j, score, beh, geo, same_section = info
            section_rows.append({
                "cache": label,
                "target_idx": int(target_idx),
                "target_sensor": sensor_names[int(target_idx)],
                "target_section": int(SECTION_ID[int(target_idx)]),
                "rank": int(rank + 1),
                "section_idx": int(cand_j),
                "section_sensor": sensor_names[int(cand_j)],
                "candidate_section": int(SECTION_ID[int(cand_j)]),
                "same_section": float(same_section),
                "score": float(score),
                "behavior_distance": float(beh),
                "geometry_distance": float(geo),
                "relation": relation_label(int(target_idx), int(cand_j)),
            })

    return cache, pd.DataFrame(section_rows)


all_idx = np.arange(N_sensors, dtype=int)
print("\nBuilding PG27 peak/transient section context cache...")
train_context_cache, train_section_df = build_context_cache(all_idx, all_idx, "train_all")
eval_context_cache, eval_section_df = build_context_cache(FIXED_MISSING_IDX, FIXED_KNOWN_IDX, "eval_missing")
train_section_df.to_csv(OUT_DIR / "train_all_section_context_candidates.csv", index=False)
eval_section_df.to_csv(OUT_DIR / "final_missing_section_context_candidates.csv", index=False)


def relative_features(target_idx, ctx_idx, ctx_type):
    target_idx = int(target_idx)
    ctx_idx = np.asarray(ctx_idx, dtype=int)
    ctx_type = np.asarray(ctx_type, dtype=np.float32)

    tg = sensor_geo_np[target_idx]
    cg = sensor_geo_np[ctx_idx]

    dr = r_feat[ctx_idx] - r_feat[target_idx]
    dt = angular_distance(theta_feat[ctx_idx], theta_feat[target_idx]) / np.pi
    dz = z_feat[ctx_idx] - z_feat[target_idx]
    dist = np.sqrt(np.maximum(D2_np[target_idx, ctx_idx], 0.0))

    same_z = (z_bin[ctx_idx] == z_bin[target_idx]).astype(np.float32)
    same_r = (r_bin[ctx_idx] == r_bin[target_idx]).astype(np.float32)
    same_group = (group_id[ctx_idx] == group_id[target_idx]).astype(np.float32)
    same_section = (SECTION_ID[ctx_idx] == SECTION_ID[target_idx]).astype(np.float32)

    rel = np.stack([dr, dt, dz, dist], axis=1).astype(np.float32)
    flags = np.stack([ctx_type, same_z, same_r, same_group, same_section], axis=1).astype(np.float32)
    tg_rep = np.repeat(tg.reshape(1, -1), len(ctx_idx), axis=0)

    return np.concatenate([cg, tg_rep, rel, flags], axis=1).astype(np.float32)


# ============================================================
# 6. SAMPLE BUILDER AND DATASETS
# ============================================================
def build_sample(X_raw, t, target_idx, cache):
    target_idx = int(target_idx)
    ctx = cache[target_idx]["ctx"]
    ctx_type = cache[target_idx]["ctx_type"]

    win = X_raw[t - WINDOW + 1:t + 1, ctx].astype(np.float32)

    # Local scaling is set at the beginning from physical-near sensors only.
    # Section sensors provide shape/context, but they do not set the scale.
    phys_mask = ctx_type < 0.5
    section_mask = ctx_type >= 0.5
    if np.any(phys_mask):
        scale_win = win[:, phys_mask]
    else:
        scale_win = win

    local_mean_seq = scale_win.mean(axis=1, keepdims=True)
    local_std_seq = scale_win.std(axis=1, keepdims=True) + 1e-6
    ctx_seq_norm = (win - local_mean_seq) / local_std_seq
    ctx_seq_norm = np.clip(ctx_seq_norm, -8.0, 8.0).astype(np.float32)

    y_raw = float(X_raw[t, target_idx])
    local_mean_now = float(local_mean_seq[-1, 0])
    local_std_now = float(local_std_seq[-1, 0])
    y_resid = (y_raw - local_mean_now) / local_std_now
    y_resid = float(np.clip(y_resid, -8.0, 8.0))

    if t > WINDOW - 1:
        prev = build_sample_no_prev(X_raw, t - 1, target_idx, cache)
        y_prev = prev["y_resid"]
        true_slope = y_raw - float(X_raw[t - 1, target_idx])
    else:
        y_prev = y_resid
        true_slope = 0.0

    phase = float(t) / max(float(X_raw.shape[0] - 1), 1.0)
    local_mean_prev = float(local_mean_seq[-2, 0]) if WINDOW >= 2 else local_mean_now
    local_slope = local_mean_now - local_mean_prev

    vals_now = X_raw[t, ctx].astype(np.float32)
    phys_mean = float(vals_now[phys_mask].mean()) if np.any(phys_mask) else local_mean_now
    section_mean = float(vals_now[section_mask].mean()) if np.any(section_mask) else local_mean_now
    section_std = float(vals_now[section_mask].std() + 1e-6) if np.any(section_mask) else local_std_now
    section_minus_phys = section_mean - phys_mean

    local_scale_for_aux = max(local_std_now, 1e-6)
    target_section_norm = float(SECTION_ID[target_idx]) / max(float(SECTION_COUNT - 1), 1.0)
    aux = np.array(
        [
            np.log1p(local_std_now / 50.0),
            local_slope / local_scale_for_aux,
            phase,
            section_minus_phys / local_scale_for_aux,
            section_std / local_scale_for_aux,
            target_section_norm,
            local_mean_now / 1000.0,
        ],
        dtype=np.float32,
    )

    pair_geo = relative_features(target_idx, ctx, ctx_type)
    target_geo = sensor_geo_np[target_idx].astype(np.float32)

    return {
        "ctx_seq": ctx_seq_norm.T.astype(np.float32),
        "pair_geo": pair_geo.astype(np.float32),
        "target_geo": target_geo.astype(np.float32),
        "aux": aux.astype(np.float32),
        "y_resid": np.float32(y_resid),
        "y_prev": np.float32(y_prev),
        "local_mean": np.float32(local_mean_now),
        "local_std": np.float32(local_std_now),
        "phys_mean": np.float32(phys_mean),
        "section_mean": np.float32(section_mean),
        "section_std": np.float32(section_std),
        "section_minus_phys": np.float32(section_minus_phys),
        "true_T": np.float32(y_raw),
        "true_slope": np.float32(true_slope),
        "target_idx": np.int64(target_idx),
        "t": np.int64(t),
    }


def build_sample_no_prev(X_raw, t, target_idx, cache):
    target_idx = int(target_idx)
    ctx = cache[target_idx]["ctx"]
    ctx_type = cache[target_idx]["ctx_type"]
    win = X_raw[t - WINDOW + 1:t + 1, ctx].astype(np.float32)
    phys_mask = ctx_type < 0.5
    if np.any(phys_mask):
        vals = win[-1, phys_mask]
    else:
        vals = win[-1]
    local_mean = float(vals.mean())
    local_std = float(vals.std() + 1e-6)
    y_raw = float(X_raw[t, target_idx])
    y_resid = float(np.clip((y_raw - local_mean) / local_std, -8.0, 8.0))
    return {"y_resid": np.float32(y_resid)}


class SectionContextTrainDataset(Dataset):
    def __init__(self, X_raw, target_pool, cache):
        self.X = X_raw.astype(np.float32)
        self.target_pool = np.asarray(target_pool, dtype=int)
        self.cache = cache
        self.times = np.arange(WINDOW - 1, self.X.shape[0], dtype=int)

    def __len__(self):
        return SAMPLES_PER_EPOCH

    def __getitem__(self, idx):
        t = int(random.choice(self.times.tolist()))

        if TRAIN_TARGET_MODE == "fixed_plus_random" and random.random() < FIXED_TARGET_PROB:
            target_idx = int(random.choice(FIXED_MISSING_IDX.tolist()))
        else:
            target_idx = int(random.choice(self.target_pool.tolist()))

        return build_sample(self.X, t, target_idx, self.cache)


class SectionContextEvalDataset(Dataset):
    def __init__(self, X_raw, target_idx, cache, stride=1):
        self.X = X_raw.astype(np.float32)
        self.target_idx = np.asarray(target_idx, dtype=int)
        self.cache = cache
        self.times = np.arange(WINDOW - 1, self.X.shape[0], int(stride), dtype=int)

        pairs = []
        for j, idx in enumerate(self.target_idx):
            for t in self.times:
                pairs.append((int(t), int(idx), int(j)))
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        t, target_idx, local_j = self.pairs[idx]
        sample = build_sample(self.X, t, target_idx, self.cache)
        sample["local_j"] = np.int64(local_j)
        return sample


def collate_train(batch):
    keys_float = [
        "ctx_seq", "pair_geo", "target_geo", "aux",
        "y_resid", "y_prev", "local_mean", "local_std",
        "phys_mean", "section_mean", "section_std", "section_minus_phys",
        "true_T", "true_slope",
    ]
    out = {}
    for k in keys_float:
        out[k] = torch.tensor(np.stack([b[k] for b in batch]), dtype=torch.float32)
    out["target_idx"] = torch.tensor([b["target_idx"] for b in batch], dtype=torch.long)
    out["t"] = torch.tensor([b["t"] for b in batch], dtype=torch.long)
    if "local_j" in batch[0]:
        out["local_j"] = torch.tensor([b["local_j"] for b in batch], dtype=torch.long)
    return out


def batch_to_device(batch):
    return {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}


train_ds = SectionContextTrainDataset(X27_raw, all_idx, train_context_cache)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_train,
)



# ============================================================
# 7. MODEL
# ============================================================
class SectionContextLocationVAE(nn.Module):
    def __init__(self, pair_geo_dim, target_geo_dim, aux_dim):
        super().__init__()

        self.sensor_gru = nn.GRU(
            input_size=1,
            hidden_size=SEQ_HIDDEN,
            num_layers=1,
            batch_first=True,
        )

        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_geo_dim, SET_HIDDEN),
            nn.SiLU(),
            nn.Linear(SET_HIDDEN, SET_HIDDEN),
            nn.SiLU(),
        )

        self.item_encoder = nn.Sequential(
            nn.Linear(SEQ_HIDDEN + SET_HIDDEN, SET_HIDDEN),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(SET_HIDDEN, SET_HIDDEN),
            nn.SiLU(),
        )

        self.target_encoder = nn.Sequential(
            nn.Linear(target_geo_dim + aux_dim, TARGET_HIDDEN),
            nn.SiLU(),
            nn.Linear(TARGET_HIDDEN, TARGET_HIDDEN),
            nn.SiLU(),
        )

        self.attn_score = nn.Sequential(
            nn.Linear(SET_HIDDEN + TARGET_HIDDEN, SET_HIDDEN),
            nn.SiLU(),
            nn.Linear(SET_HIDDEN, 1),
        )

        cond_dim = SET_HIDDEN + TARGET_HIDDEN + aux_dim
        self.cond_dim = cond_dim

        self.prior_net = nn.Sequential(
            nn.Linear(cond_dim, SET_HIDDEN),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(SET_HIDDEN, SET_HIDDEN),
            nn.SiLU(),
        )
        self.prior_mu = nn.Linear(SET_HIDDEN, LATENT_DIM)
        self.prior_logvar = nn.Linear(SET_HIDDEN, LATENT_DIM)

        self.post_net = nn.Sequential(
            nn.Linear(cond_dim + 1, SET_HIDDEN),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(SET_HIDDEN, SET_HIDDEN),
            nn.SiLU(),
        )
        self.post_mu = nn.Linear(SET_HIDDEN, LATENT_DIM)
        self.post_logvar = nn.Linear(SET_HIDDEN, LATENT_DIM)

        self.decoder = nn.Sequential(
            nn.Linear(cond_dim + LATENT_DIM, SET_HIDDEN),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(SET_HIDDEN, SET_HIDDEN),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(SET_HIDDEN, 1),
        )

    def encode_context(self, ctx_seq, pair_geo, target_geo, aux):
        B, K, L = ctx_seq.shape

        seq_in = ctx_seq.reshape(B * K, L, 1)
        _, h = self.sensor_gru(seq_in)
        seq_h = h[-1].reshape(B, K, SEQ_HIDDEN)

        pair_h = self.pair_encoder(pair_geo)
        item_h = self.item_encoder(torch.cat([seq_h, pair_h], dim=-1))

        target_h = self.target_encoder(torch.cat([target_geo, aux], dim=-1))
        target_rep = target_h.unsqueeze(1).expand(-1, K, -1)

        scores = self.attn_score(torch.cat([item_h, target_rep], dim=-1)).squeeze(-1)
        weights = torch.softmax(scores, dim=1)

        context_h = torch.sum(item_h * weights.unsqueeze(-1), dim=1)
        cond = torch.cat([context_h, target_h, aux], dim=-1)
        return cond

    def prior(self, cond):
        h = self.prior_net(cond)
        mu = self.prior_mu(h)
        logvar = torch.clamp(self.prior_logvar(h), -8.0, 4.0)
        return mu, logvar

    def posterior(self, cond, y0):
        h = self.post_net(torch.cat([cond, y0.unsqueeze(-1)], dim=-1))
        mu = self.post_mu(h)
        logvar = torch.clamp(self.post_logvar(h), -8.0, 4.0)
        return mu, logvar

    def reparameterize(self, mu, logvar, sample=True, scale=1.0):
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar) * float(scale)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, cond, z):
        y = self.decoder(torch.cat([cond, z], dim=-1)).squeeze(-1)
        y = torch.nan_to_num(y, nan=0.0, posinf=8.0, neginf=-8.0)
        return torch.clamp(y, -8.0, 8.0)

    def forward_train(self, ctx_seq, pair_geo, target_geo, aux, y0):
        cond = self.encode_context(ctx_seq, pair_geo, target_geo, aux)
        prior_mu, prior_logvar = self.prior(cond)
        post_mu, post_logvar = self.posterior(cond, y0)
        z = self.reparameterize(post_mu, post_logvar, sample=True, scale=1.0)
        y0_post = self.decode(cond, z)

        # Train the conditional prior directly because PG29 inference can only
        # use p(z | context), not q(z | y0, context).
        z_prior_mean = self.reparameterize(prior_mu, prior_logvar, sample=False)
        y0_prior = self.decode(cond, z_prior_mean)
        return y0_post, y0_prior, post_mu, post_logvar, prior_mu, prior_logvar

    def sample_prior(self, ctx_seq, pair_geo, target_geo, aux, n_samples):
        cond = self.encode_context(ctx_seq, pair_geo, target_geo, aux)
        prior_mu, prior_logvar = self.prior(cond)
        samples = []

        # First sample is the prior mean. This stabilizes the VAE prediction and
        # keeps the reported mean from being dominated by random latent draws.
        z0 = self.reparameterize(prior_mu, prior_logvar, sample=False)
        samples.append(self.decode(cond, z0).detach().cpu())

        for _ in range(max(int(n_samples) - 1, 0)):
            z = self.reparameterize(
                prior_mu,
                prior_logvar,
                sample=True,
                scale=PRIOR_SAMPLE_STD_SCALE,
            )
            samples.append(self.decode(cond, z).detach().cpu())

        return torch.stack(samples, dim=0)


model = SectionContextLocationVAE(
    pair_geo_dim=PAIR_GEO_DIM,
    target_geo_dim=GEO_DIM,
    aux_dim=7,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Latent dim: {LATENT_DIM} | beta_kl: {BETA_KL} | KL warmup epochs: {KL_WARMUP_EPOCHS}")


# ============================================================
# 8. TRAINING
# ============================================================
def norm_weight(w):
    w = torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)
    w = torch.clamp(w, min=1e-6)
    return w / (torch.mean(w).detach() + 1e-8)


def weighted_mse_error(err, w):
    ww = norm_weight(w)
    return torch.mean(ww * err ** 2)


def weighted_mse(pred, target, w):
    return weighted_mse_error(pred - target, w)


def kl_diag_gaussian(q_mu, q_logvar, p_mu, p_logvar):
    q_var = torch.exp(q_logvar)
    p_var = torch.exp(p_logvar)
    kl_dim = 0.5 * (
        p_logvar - q_logvar
        + (q_var + (q_mu - p_mu) ** 2) / (p_var + 1e-8)
        - 1.0
    )
    if KL_FREE_BITS > 0.0:
        kl_dim = torch.clamp(kl_dim, min=KL_FREE_BITS)
    kl_dim = torch.nan_to_num(kl_dim, nan=0.0, posinf=100.0, neginf=0.0)
    return torch.sum(kl_dim, dim=1)


def compute_loss(batch, epoch):
    y0 = batch["y_resid"]

    y0_post, y0_prior, q_mu, q_logvar, p_mu, p_logvar = model.forward_train(
        ctx_seq=batch["ctx_seq"],
        pair_geo=batch["pair_geo"],
        target_geo=batch["target_geo"],
        aux=batch["aux"],
        y0=y0,
    )

    # Use the prior-mean residual for all Celsius/physics losses, because this
    # is the path used during PG29 missing-sensor inference.
    y0_pred = y0_prior
    pred_T = batch["local_mean"] + y0_pred * batch["local_std"]
    true_T = batch["true_T"]

    # Local scale only. No pg27_global_std in the core MSE losses.
    local_scale = torch.clamp(batch["local_std"], min=LOCAL_SCALE_MIN_C, max=LOCAL_SCALE_MAX_C)

    # Sensor weight from block, radial position, distance to known sensors,
    # and closeness to the final missing-sensor region.
    sensor_w = SENSOR_LOSS_W[batch["target_idx"]]

    # Event weight from local residual difficulty and local transient slope.
    resid_event = torch.clamp(torch.abs(y0).detach(), 0.0, 4.0) / 4.0
    slope_event = torch.clamp(torch.abs(batch["true_slope"]).detach() / local_scale, 0.0, 3.0) / 3.0
    event_w = 1.0 + EVENT_RESID_STRENGTH * resid_event + EVENT_SLOPE_STRENGTH * slope_event
    w_recon = sensor_w * event_w

    post_recon_loss = weighted_mse(y0_post, y0, w_recon)
    prior_recon_loss = weighted_mse(y0_prior, y0, w_recon)
    recon_loss = 0.35 * post_recon_loss + 0.65 * LAMBDA_PRIOR_RECON * prior_recon_loss

    # Weighted Celsius MSE in local units, not global units.
    temp_err_local = (pred_T - true_T) / local_scale
    temp_loss = weighted_mse_error(temp_err_local, w_recon)

    # Local relation losses, also in local units.
    pred_local_delta = (pred_T - batch["local_mean"]) / local_scale
    true_local_delta = (true_T - batch["local_mean"]) / local_scale
    local_delta_loss = weighted_mse(pred_local_delta, true_local_delta, w_recon)

    pred_phys_delta = (pred_T - batch["phys_mean"]) / local_scale
    true_phys_delta = (true_T - batch["phys_mean"]) / local_scale
    phys_delta_loss = weighted_mse(pred_phys_delta, true_phys_delta, w_recon)

    pred_section_delta = (pred_T - batch["section_mean"]) / local_scale
    true_section_delta = (true_T - batch["section_mean"]) / local_scale
    section_delta_loss = weighted_mse(pred_section_delta, true_section_delta, w_recon)

    pred_section_z = (pred_T - batch["section_mean"]) / torch.clamp(batch["section_std"], min=LOCAL_SCALE_MIN_C, max=LOCAL_SCALE_MAX_C)
    true_section_z = (true_T - batch["section_mean"]) / torch.clamp(batch["section_std"], min=LOCAL_SCALE_MIN_C, max=LOCAL_SCALE_MAX_C)
    section_z_loss = weighted_mse(pred_section_z, true_section_z, w_recon)

    pred_res_slope = y0_pred - batch["y_prev"]
    true_res_slope = y0 - batch["y_prev"]
    slope_loss = weighted_mse(pred_res_slope, true_res_slope, w_recon)

    res_mag_loss = weighted_mse_error(y0_pred, sensor_w)

    kl_row = kl_diag_gaussian(q_mu, q_logvar, p_mu, p_logvar)
    kl_loss = torch.mean(norm_weight(sensor_w) * kl_row) / max(float(LATENT_DIM), 1.0)
    kl_warm = min(1.0, float(epoch) / max(float(KL_WARMUP_EPOCHS), 1.0))
    kl_term = BETA_KL * kl_warm * kl_loss

    total = (
        LAMBDA_RECON * recon_loss
        + LAMBDA_TEMP * temp_loss
        + LAMBDA_LOCAL_DELTA * local_delta_loss
        + LAMBDA_PHYS_DELTA * phys_delta_loss
        + LAMBDA_SECTION_DELTA * (section_delta_loss + 0.50 * section_z_loss)
        + LAMBDA_SLOPE * slope_loss
        + LAMBDA_RES_MAG * res_mag_loss
        + LAMBDA_KL * kl_term
    )

    parts = {
        "total": float(total.detach().cpu()),
        "recon": float(recon_loss.detach().cpu()),
        "post_recon": float(post_recon_loss.detach().cpu()),
        "prior_recon": float(prior_recon_loss.detach().cpu()),
        "temp": float(temp_loss.detach().cpu()),
        "local_delta": float(local_delta_loss.detach().cpu()),
        "phys_delta": float(phys_delta_loss.detach().cpu()),
        "section_delta": float(section_delta_loss.detach().cpu()),
        "section_z": float(section_z_loss.detach().cpu()),
        "slope": float(slope_loss.detach().cpu()),
        "res_mag": float(res_mag_loss.detach().cpu()),
        "kl": float(kl_loss.detach().cpu()),
        "kl_term": float(kl_term.detach().cpu()),
        "kl_warm": float(kl_warm),
        "mean_sensor_w": float(torch.mean(sensor_w).detach().cpu()),
        "mean_event_w": float(torch.mean(event_w).detach().cpu()),
    }

    return total, parts


def train_one_epoch(epoch):
    model.train()
    totals = None
    n_batches = 0

    for batch in train_loader:
        batch = batch_to_device(batch)
        loss, parts = compute_loss(batch, epoch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        if totals is None:
            totals = {k: 0.0 for k in parts.keys()}
        for k in totals:
            totals[k] += parts[k]
        n_batches += 1

    if totals is None:
        totals = {"total": 0.0}
    for k in totals:
        totals[k] /= max(n_batches, 1)

    return totals


train_log = []
start_train = time.time()

print("\nTraining SectionContext LocationQuery-VAE...")
for epoch in range(1, EPOCHS + 1):
    parts = train_one_epoch(epoch)
    row = {"epoch": epoch}
    row.update(parts)
    train_log.append(row)

    if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
        print(
            f"Epoch {epoch:04d}/{EPOCHS} | "
            f"total={parts['total']:.6f} recon={parts['recon']:.6f} "
            f"prior={parts['prior_recon']:.6f} temp={parts['temp']:.6f} "
            f"kl={parts['kl']:.6f} "
            f"kl_term={parts['kl_term']:.6f} "
            f"phys={parts.get('phys_delta', 0.0):.6f} "
            f"section={parts.get('section_delta', 0.0):.6f} "
            f"slope={parts['slope']:.6f}",
            flush=True,
        )

train_seconds = time.time() - start_train
train_df = pd.DataFrame(train_log)
train_df.to_csv(OUT_DIR / "train_log.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
for col in [
    "total", "recon", "post_recon", "prior_recon", "temp", "local_delta",
    "phys_delta", "section_delta", "section_z", "slope",
    "res_mag", "kl", "kl_term",
]:
    if col in train_df.columns:
        ax.plot(train_df["epoch"], train_df[col], label=col)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Section-context LocationQuery-VAE training loss")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "train_loss_components.png", dpi=160)
plt.close(fig)


# ============================================================
# 9. SAMPLING AND COLLECTION
# ============================================================
@torch.no_grad()
def sample_residual(batch, n_samples):
    model.eval()
    return model.sample_prior(
        ctx_seq=batch["ctx_seq"],
        pair_geo=batch["pair_geo"],
        target_geo=batch["target_geo"],
        aux=batch["aux"],
        n_samples=n_samples,
    )


@torch.no_grad()
def collect_predictions(X_raw, target_idx, cache, stride=1, n_samples=N_SAMPLES_EVAL):
    ds = SectionContextEvalDataset(X_raw, target_idx, cache, stride=stride)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_train,
    )

    times = ds.times
    Tn = len(times)
    M = len(target_idx)
    time_to_row = {int(t): i for i, t in enumerate(times)}

    y_true = np.zeros((Tn, M), dtype=np.float32)
    y_pred = np.zeros((Tn, M), dtype=np.float32)
    y_std = np.zeros((Tn, M), dtype=np.float32)

    for batch in loader:
        batch_dev = batch_to_device(batch)
        samples = sample_residual(batch_dev, n_samples=n_samples)
        samples_np = samples.numpy()

        resid_mean = samples_np.mean(axis=0)
        resid_std = samples_np.std(axis=0)

        lm = batch["local_mean"].numpy()
        ls = batch["local_std"].numpy()
        yp = lm + resid_mean * ls
        ys = resid_std * ls

        tt = batch["t"].numpy().astype(int)
        jj = batch["local_j"].numpy().astype(int)
        yt = batch["true_T"].numpy()

        for i in range(len(tt)):
            r = time_to_row[int(tt[i])]
            c = int(jj[i])
            y_true[r, c] = yt[i]
            y_pred[r, c] = yp[i]
            y_std[r, c] = ys[i]

    return y_true, y_pred, y_std, times


print("\nCollecting PG29 final missing predictions...")
y29_true, y29_vae, y29_std, eval_times = collect_predictions(
    X29_raw,
    FIXED_MISSING_IDX,
    eval_context_cache,
    stride=1,
    n_samples=N_SAMPLES_EVAL,
)

# PG27 sanity check with fixed-known context.
y27_true, y27_vae, _, eval_times27 = collect_predictions(
    X27_raw,
    FIXED_MISSING_IDX,
    eval_context_cache,
    stride=1,
    n_samples=max(1, min(2, N_SAMPLES_EVAL)),
)


# ============================================================
# 10. ANCHOR BASELINE
# ============================================================
def context_physical_only(target_idx, allowed_idx, k):
    return context_physical(target_idx, allowed_idx, k)


def build_anchor_features(X_raw, target_idx, allowed_idx):
    ctx = context_physical_only(target_idx, allowed_idx, ANCHOR_K)
    Xc = X_raw[:, ctx].astype(np.float32)

    local_mean = Xc.mean(axis=1, keepdims=True)
    local_std = Xc.std(axis=1, keepdims=True)
    local_min = Xc.min(axis=1, keepdims=True)
    local_max = Xc.max(axis=1, keepdims=True)

    vaeean = np.zeros_like(local_mean)
    vaeean[1:] = local_mean[1:] - local_mean[:-1]

    dist = D2_np[int(target_idx), ctx].reshape(1, -1).astype(np.float32)
    dist_rep = np.repeat(dist, X_raw.shape[0], axis=0)
    phase = np.linspace(0.0, 1.0, X_raw.shape[0], dtype=np.float32).reshape(-1, 1)

    return np.concatenate([Xc, local_mean, local_std, local_min, local_max, vaeean, dist_rep, phase], axis=1)


def fit_predict_anchor(train_raw, test_raw, target_idx, train_context_idx, test_context_idx):
    pred_train = np.zeros((train_raw.shape[0], len(target_idx)), dtype=np.float32)
    pred_test = np.zeros((test_raw.shape[0], len(target_idx)), dtype=np.float32)
    rows = []

    for j, idx in enumerate(target_idx):
        Ftr = build_anchor_features(train_raw, int(idx), train_context_idx)
        Fte = build_anchor_features(test_raw, int(idx), test_context_idx)

        scaler = StandardScaler()
        Ftrs = scaler.fit_transform(Ftr)
        Ftes = scaler.transform(Fte)

        y = train_raw[:, int(idx)]
        reg = Ridge(alpha=ANCHOR_ALPHA, fit_intercept=True, solver="lsqr", max_iter=10000, tol=1e-6)
        reg.fit(Ftrs, y)

        pred_train[:, j] = reg.predict(Ftrs).astype(np.float32)
        pred_test[:, j] = reg.predict(Ftes).astype(np.float32)

        rows.append({
            "target_idx": int(idx),
            "target_sensor": sensor_names[int(idx)],
            "context_sensors": ";".join([sensor_names[int(i)] for i in context_physical_only(int(idx), test_context_idx, ANCHOR_K)]),
        })

    return pred_train, pred_test, pd.DataFrame(rows)


print("\nFitting anchor baseline...")
anchor27_full, anchor29_full, anchor_info = fit_predict_anchor(
    X27_raw,
    X29_raw,
    FIXED_MISSING_IDX,
    FIXED_KNOWN_IDX,
    FIXED_KNOWN_IDX,
)
anchor_info.to_csv(OUT_DIR / "anchor_context_info.csv", index=False)

y29_anchor = anchor29_full[eval_times]
y27_anchor = anchor27_full[eval_times27]

y29_vae = replace_bad_with_fallback(y29_vae, y29_anchor)
y27_vae = replace_bad_with_fallback(y27_vae, y27_anchor)


# ============================================================
# 11. PG29 KNOWN PSEUDO CALIBRATION AND GATE
# ============================================================
def select_pseudo_targets():
    dist = np.min(D2_np[np.ix_(FIXED_KNOWN_IDX, FIXED_MISSING_IDX)], axis=1)
    order = np.argsort(dist)
    return FIXED_KNOWN_IDX[order[:min(PSEUDO_MAX_TARGETS, len(order))]].astype(int)


def fit_affine_cal(y_pred, y_true):
    x = np.asarray(y_pred).reshape(-1)
    y = np.asarray(y_true).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 10:
        return 1.0, 0.0
    X = np.column_stack([x[m], np.ones(int(m.sum()))])
    coef, _, _, _ = np.linalg.lstsq(X, y[m], rcond=None)
    a = float(np.clip(coef[0], 0.25, 2.5))
    b = float(coef[1])
    return a, b


def geometry_weighted_average(target_idx, source_idx, source_values, k):
    source_idx = np.asarray(source_idx, dtype=int)
    source_values = np.asarray(source_values, dtype=np.float32)

    if len(source_idx) == 0:
        if source_values.ndim == 1:
            return np.float32(0.0)
        return np.ones(source_values.shape[1], dtype=np.float32) / float(source_values.shape[1])

    dist = D2_np[int(target_idx), source_idx]
    nn = np.argsort(dist)[:min(int(k), len(source_idx))]
    w = np.exp(-dist[nn] / (2.0 * 0.35 ** 2))
    w = w / (w.sum() + 1e-12)

    if source_values.ndim == 1:
        return np.float32(np.sum(w * source_values[nn]))

    out = np.sum(source_values[nn] * w.reshape(-1, 1), axis=0)
    return (out / (out.sum() + 1e-12)).astype(np.float32)


def soft_weights_from_rmse(rmses):
    rmses = np.asarray(rmses, dtype=np.float64)
    rmses = np.nan_to_num(rmses, nan=1e6, posinf=1e6, neginf=1e6)
    score = np.exp(-rmses / max(RELIABILITY_TAU_C, 1e-6)) + 1e-6
    return (score / score.sum()).astype(np.float32)


print("\nRunning PG29 known-sensor pseudo calibration/gate...")
pseudo_targets = select_pseudo_targets()
pseudo_context = np.array(
    sorted(set(FIXED_KNOWN_IDX.tolist()) - set(pseudo_targets.tolist())),
    dtype=int,
)

pseudo_cache, pseudo_section_df = build_context_cache(pseudo_targets, pseudo_context, "pseudo_known")
pseudo_section_df.to_csv(OUT_DIR / "pseudo_known_section_context_candidates.csv", index=False)

y_pseudo_true, y_pseudo_vae, _, pseudo_times = collect_predictions(
    X29_raw,
    pseudo_targets,
    pseudo_cache,
    stride=ADAPT_STRIDE,
    n_samples=max(1, min(2, N_SAMPLES_EVAL)),
)

_, anchor_pseudo_full, pseudo_anchor_info = fit_predict_anchor(
    X27_raw,
    X29_raw,
    pseudo_targets,
    all_idx,
    pseudo_context,
)
pseudo_anchor_info.to_csv(OUT_DIR / "pseudo_anchor_context_info.csv", index=False)
y_pseudo_anchor = anchor_pseudo_full[pseudo_times]
y_pseudo_vae = replace_bad_with_fallback(y_pseudo_vae, y_pseudo_anchor)

# Local affine calibration for VAE.
pseudo_a = []
pseudo_b = []
pseudo_perf_rows = []
pseudo_weights = []

for j, idx in enumerate(pseudo_targets):
    a, b = fit_affine_cal(y_pseudo_vae[:, [j]], y_pseudo_true[:, [j]])
    pseudo_a.append(a)
    pseudo_b.append(b)

pseudo_a = np.asarray(pseudo_a, dtype=np.float32)
pseudo_b = np.asarray(pseudo_b, dtype=np.float32)

target_a = []
target_b = []
for idx in FIXED_MISSING_IDX:
    target_a.append(geometry_weighted_average(idx, pseudo_targets, pseudo_a, LOCAL_CAL_NEIGHBORS))
    target_b.append(geometry_weighted_average(idx, pseudo_targets, pseudo_b, LOCAL_CAL_NEIGHBORS))

target_a = np.asarray(target_a, dtype=np.float32)
target_b = np.asarray(target_b, dtype=np.float32)

y29_vae_cal = y29_vae * target_a.reshape(1, -1) + target_b.reshape(1, -1)
y_pseudo_vae_cal = y_pseudo_vae * pseudo_a.reshape(1, -1) + pseudo_b.reshape(1, -1)

# Mean/spatial candidate.
def mean_anchor_spatial(vae_pred, anchor_pred):
    vae_mean = vae_pred.mean(axis=1, keepdims=True)
    anchor_mean = anchor_pred.mean(axis=1, keepdims=True)
    return vae_mean + (anchor_pred - anchor_mean)

y29_vaemean_anchor = mean_anchor_spatial(y29_vae_cal, y29_anchor)
y_pseudo_vaemean_anchor = mean_anchor_spatial(y_pseudo_vae_cal, y_pseudo_anchor)

# Pseudo reliability among VAE_cal, Anchor, VAEmean_Anchor.
for j, idx in enumerate(pseudo_targets):
    _, r_vae, _ = metric_values(y_pseudo_true[:, [j]], y_pseudo_vae_cal[:, [j]])
    _, r_anchor, _ = metric_values(y_pseudo_true[:, [j]], y_pseudo_anchor[:, [j]])
    _, r_vaemean, _ = metric_values(y_pseudo_true[:, [j]], y_pseudo_vaemean_anchor[:, [j]])

    w = soft_weights_from_rmse([r_vae, r_anchor, r_vaemean])
    pseudo_weights.append(w)

    best_vals = {
        "SectionContextVAE_calibrated": r_vae,
        "Anchor_only": r_anchor,
        "VAEmean_AnchorSpatial": r_vaemean,
    }
    best_name = min(best_vals, key=best_vals.get)

    pseudo_perf_rows.append({
        "idx": int(idx),
        "sensor": sensor_names[int(idx)],
        "rmse_vae_cal": r_vae,
        "rmse_anchor": r_anchor,
        "rmse_vaemean_anchor": r_vaemean,
        "w_vae_cal": float(w[0]),
        "w_anchor": float(w[1]),
        "w_vaemean_anchor": float(w[2]),
        "best_candidate": best_name,
        "best_rmse": float(best_vals[best_name]),
    })

pseudo_weights = np.asarray(pseudo_weights, dtype=np.float32)
pd.DataFrame(pseudo_perf_rows).to_csv(OUT_DIR / "pseudo_known_candidate_reliability.csv", index=False)

target_weights = []
for idx in FIXED_MISSING_IDX:
    target_weights.append(
        geometry_weighted_average(idx, pseudo_targets, pseudo_weights, LOCAL_GATE_NEIGHBORS)
    )

target_weights = np.asarray(target_weights, dtype=np.float32)
target_weights = target_weights / (target_weights.sum(axis=1, keepdims=True) + 1e-12)

pd.DataFrame({
    "idx": FIXED_MISSING_IDX,
    "sensor": [sensor_names[int(i)] for i in FIXED_MISSING_IDX],
    "a_local": target_a,
    "b_local": target_b,
    "w_vae_cal": target_weights[:, 0],
    "w_anchor": target_weights[:, 1],
    "w_vaemean_anchor": target_weights[:, 2],
}).to_csv(OUT_DIR / "final_missing_calibration_and_gate_weights.csv", index=False)

y29_gate = (
    target_weights[:, 0].reshape(1, -1) * y29_vae_cal
    + target_weights[:, 1].reshape(1, -1) * y29_anchor
    + target_weights[:, 2].reshape(1, -1) * y29_vaemean_anchor
)


# ============================================================
# 12. METRICS
# ============================================================
methods = {
    "SectionContextVAE_only": y29_vae,
    "SectionContextVAE_calibrated": y29_vae_cal,
    "Anchor_only": y29_anchor,
    "VAEmean_AnchorSpatial": y29_vaemean_anchor,
    "SectionContextVAE_LocalGate": y29_gate,
}

summary_rows = []
for name, pred in methods.items():
    mae, rmse, r2 = metric_values(y29_true, pred)
    summary_rows.append({
        "domain": "PG29",
        "method": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

summary_df = pd.DataFrame(summary_rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
summary_df.to_csv(OUT_DIR / "method_summary_global_metrics.csv", index=False)

combined = pd.DataFrame({
    "idx": FIXED_MISSING_IDX,
    "sensor": [sensor_names[int(i)] for i in FIXED_MISSING_IDX],
})

for name, pred in methods.items():
    maes = []
    rmses = []
    r2s = []
    for j in range(len(FIXED_MISSING_IDX)):
        mae, rmse, r2 = metric_values(y29_true[:, [j]], pred[:, [j]])
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)
    combined[f"{name}_MAE"] = maes
    combined[f"{name}_RMSE"] = rmses
    combined[f"{name}_R2"] = r2s

oracle_rmse = []
oracle_method = []
for j in range(len(FIXED_MISSING_IDX)):
    best_r = np.inf
    best_name = None
    for name, pred in methods.items():
        if name == "SectionContextVAE_LocalGate":
            continue
        _, r, _ = metric_values(y29_true[:, [j]], pred[:, [j]])
        if r < best_r:
            best_r = r
            best_name = name
    oracle_rmse.append(best_r)
    oracle_method.append(best_name)

combined["OracleBestCandidate_Diagnostic_RMSE"] = oracle_rmse
combined["OracleBestCandidate_Diagnostic_Method"] = oracle_method
combined.to_csv(OUT_DIR / "combined_per_sensor_pg29_all_methods.csv", index=False)

print("\nFinal PG29 method comparison")
print("=" * 100)
print(summary_df.to_string(index=False))
print("=" * 100)

print("\nWorst SectionContextVAE_LocalGate sensors:")
print(
    combined[
        [
            "sensor",
            "SectionContextVAE_LocalGate_RMSE",
            "OracleBestCandidate_Diagnostic_RMSE",
            "OracleBestCandidate_Diagnostic_Method",
        ]
    ]
    .sort_values("SectionContextVAE_LocalGate_RMSE", ascending=False)
    .head(12)
    .to_string(index=False)
)


# ============================================================
# 13. PLOTS AND SAVE ARRAYS
# ============================================================
np.savez_compressed(
    OUT_DIR / "final_predictions_arrays.npz",
    eval_times=eval_times,
    fixed_missing_idx=FIXED_MISSING_IDX,
    y29_true=y29_true,
    pred_vae=y29_vae,
    pred_vae_cal=y29_vae_cal,
    pred_anchor=y29_anchor,
    pred_vaemean_anchor=y29_vaemean_anchor,
    pred_gate=y29_gate,
    pred_std=y29_std,
    target_weights=target_weights,
)


def plot_rmse_bar():
    df = combined.sort_values("SectionContextVAE_LocalGate_RMSE", ascending=True).reset_index(drop=True)
    method_cols = [
        ("SectionContextVAE_only", "VAE_only"),
        ("Anchor_only", "Regression anchor"),
        ("SectionContextVAE_LocalGate", "Best VAE gate"),
    ]

    x = np.arange(len(df))
    width = 0.24
    offsets = np.linspace(-width, width, len(method_cols))

    fig, ax = plt.subplots(figsize=(max(14, 0.55 * len(df)), 6))
    for off, (method, label) in zip(offsets, method_cols):
        ax.bar(x + off, df[f"{method}_RMSE"].values, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(df["sensor"].str.replace("TS-", "", regex=False), rotation=90, fontsize=8)
    ax.set_ylabel("RMSE (C)")
    ax.set_title("PG29 per-sensor RMSE: section-context local-scale LocationQuery-VAE")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pg29_rmse_all_methods_bar.png", dpi=170)
    plt.close(fig)


def plot_mean():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y29_true.mean(axis=1), label="True", linewidth=1.5)
    ax.plot(y29_gate.mean(axis=1), label="Best VAE gate", linestyle="--", linewidth=1.2)
    ax.plot(y29_anchor.mean(axis=1), label="Regression anchor", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Window index")
    ax.set_ylabel("Mean fixed-missing temperature (C)")
    ax.set_title("PG29 fixed-missing mean comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pg29_missing_mean_comparison.png", dpi=170)
    plt.close(fig)


def plot_grid():
    df = combined.sort_values("SectionContextVAE_LocalGate_RMSE", ascending=False).head(PLOT_MAX_SENSORS)
    n = len(df)
    ncols = 4
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.1 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    t = np.arange(len(y29_true))

    for k, ax in enumerate(axes):
        if k >= n:
            ax.axis("off")
            continue

        row = df.iloc[k]
        sensor = row["sensor"]
        j = list(FIXED_MISSING_IDX).index(int(row["idx"]))

        ax.plot(t, y29_true[:, j], label="True", linewidth=1.4)
        ax.plot(t, y29_gate[:, j], label="Best VAE gate", linestyle="--", linewidth=1.1)
        ax.plot(t, y29_anchor[:, j], label="Regression anchor", linestyle=":", linewidth=1.0)

        ax.set_title(
            f"{sensor} | gate={row['SectionContextVAE_LocalGate_RMSE']:.1f} | oracle={row['OracleBestCandidate_Diagnostic_RMSE']:.1f}",
            fontsize=8,
        )
        ax.set_ylabel("T (C)")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Section-context LocationQuery-VAE: worst sensors", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    fig.savefig(OUT_DIR / "pg29_worst_sensor_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


plot_rmse_bar()
plot_mean()
plot_grid()

with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
    f.write("Section-context LocationQuery-VAE no-cheat reconstruction\n")
    f.write("=" * 80 + "\n\n")
    f.write("VAE uses physical nearby sensors plus PG27 peak/transient section sensors.\n")
    f.write("Training uses PG27 only.\n")
    f.write("PG29 known sensors are used only for pseudo-target calibration/gating.\n")
    f.write("PG29 fixed missing sensors are used only for final reporting.\n\n")
    f.write(f"Training seconds: {train_seconds:.2f}\n\n")
    f.write("Global metrics:\n")
    for _, row in summary_df.iterrows():
        f.write(
            f"  {row['method']}: "
            f"MAE={row['MAE']:.4f}, RMSE={row['RMSE']:.4f}, R2={row['R2']:.4f}\n"
        )

print("\nDone.")
print("Training seconds:", train_seconds)
print("Outputs saved to:", OUT_DIR.resolve())
print("Key files:")
print("  method_summary_global_metrics.csv")
print("  combined_per_sensor_pg29_all_methods.csv")
print("  final_missing_section_context_candidates.csv")
print("  pseudo_known_candidate_reliability.csv")
print("  final_missing_calibration_and_gate_weights.csv")
print("  pg29_rmse_all_methods_bar.png")
print("  pg29_missing_mean_comparison.png")
print("  pg29_worst_sensor_grid.png")
print("  final_predictions_arrays.npz")
print("  summary.txt")
