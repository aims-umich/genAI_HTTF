#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. INPUTS
# ============================================================
NPZ_FILES = {
    "GAN": "/home/unabila/pg27/section_context_local_gan_v2/final_predictions_arrays.npz",
    "VAE": "/home/unabila/pg27/section_context_local_vae/final_predictions_arrays.npz",
    "DM": "/home/unabila/pg27/section_context_local_dm_v2_section_sweep/sections_6/final_predictions_arrays.npz",
}

TS_CSV = os.environ.get("TS_CSV", "/home/unabila/PG29/pg29_TS_30s_common.csv")

# ------------------------------------------------------------
# FIX: source files needed to recompute the *exact* reduced,
# sorted sensor_names list each training script used
# (sensor_names = sorted(pg27_ts & pg29_ts & coord_ts)).
# fixed_missing_idx stored in each npz is an index into THAT
# reduced list, not into the raw PG29 CSV column list. If any
# family used a different PG27_PATH/PG29_PATH/COORD_PATH at
# training time, add a per-family override in
# SENSOR_SOURCE_OVERRIDES below.
# ------------------------------------------------------------
PG27_PATH = os.environ.get("PG27_PATH", "/home/unabila/pg27/pg27_TS_30s_common_temp.csv")
PG29_PATH = os.environ.get("PG29_PATH", "/home/unabila/PG29/pg29_TS_30s_common.csv")
COORD_PATH = os.environ.get("COORD_PATH", "/home/unabila/PG29/core_TC_coordinates_common_norm.csv")

# Optional per-file-label overrides, e.g.:
# SENSOR_SOURCE_OVERRIDES = {
#     "DM": {"PG27_PATH": "...", "PG29_PATH": "...", "COORD_PATH": "..."},
# }
SENSOR_SOURCE_OVERRIDES = {}

OUT_DIR = Path("/home/unabila/pg27/cross_recon_best_DM_VAE_GAN_anchor_regression_smoothDM")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. PLOT SETTINGS
# ============================================================
DPI = 600
FONT_SIZE = 14
TIME_STEP_SECONDS = 30.0

# Plot-only smoothing. Metrics still use raw predictions.
SMOOTH_DM_FOR_PLOT = True
SMOOTH_ALL_FOR_PLOT = False
DM_SMOOTH_WINDOW = 21

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
})

# Okabe-Ito style colorblind-friendly palette.
MODEL_COLORS = {
    "True": "black",
    "DM": "#0072B2",
    "VAE": "#D55E00",
    "GAN": "#009E73",
    "Regression": "#CC79A7",
}

MODEL_LINESTYLES = {
    "DM": "-",
    "VAE": "--",
    "GAN": "-.",
    "Regression": ":",
}

MODEL_MARKERS = {
    "DM": "o",
    "VAE": "s",
    "GAN": "^",
    "Regression": "D",
}

MODEL_LINEWIDTHS = {
    "DM": 1.25,
    "VAE": 1.35,
    "GAN": 1.35,
    "Regression": 1.35,
}

MODEL_HATCHES = {
    "DM": "",
    "VAE": "//",
    "GAN": "\\\\",
    "Regression": "..",
}

PLOT_ORDER = ["DM", "VAE", "GAN", "Regression"]


# ============================================================
# 3. KEY RULES
# ============================================================
REGRESSION_KEY_CANDIDATES = [
    "Anchor_only",
    "y29_anchor",
    "pred_anchor",
    "anchor",
    "Anchor",
    "anchor_pred",
]

METHOD_NAME_MAP = {
    "pred_safe_raw_gan_anchor": "SafeRawGAN_AnchorGate",
    "pred_anchor": "Anchor_only",
    "Anchor_only": "Anchor_only",
    "y29_anchor": "Anchor_only",
    "pred_vae": "SectionContextVAE_only",
    "pred_vae_cal": "SectionContextVAE_calibrated",
    "pred_vae_gate": "SectionContextVAE_Gate",
    "pred_dm": "SectionContextDM_only",
    "pred_dm_cal": "SectionContextDM_calibrated",
    "pred_dm_gate": "SectionContextDM_Gate",
}


# ============================================================
# 4. BASIC HELPERS
# ============================================================
def safe_name(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_key(text):
    return str(text).lower().replace("_", "").replace("-", "").replace(" ", "")


def safe_metrics(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)

    mask = np.isfinite(yt) & np.isfinite(yp)
    if int(mask.sum()) < 2:
        return np.nan, np.nan, np.nan

    yt = yt[mask]
    yp = yp[mask]

    mae = mean_absolute_error(yt, yp)
    rmse = math.sqrt(mean_squared_error(yt, yp))
    r2 = r2_score(yt, yp)

    return mae, rmse, r2


def smooth_for_plot(y, window=21):
    y = np.asarray(y, dtype=float).reshape(-1)

    if window is None or window <= 1 or len(y) < 3:
        return y.copy()

    if window % 2 == 0:
        window += 1

    return (
        pd.Series(y)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def print_npz_keys(file_label, path):
    data = np.load(path, allow_pickle=True)

    print("\n" + "=" * 80)
    print("File label:", file_label)
    print("File:", path)
    print("Keys:")
    for key in data.files:
        arr = data[key]
        print(f"  {key}: shape={getattr(arr, 'shape', None)}, dtype={getattr(arr, 'dtype', None)}")
    print("=" * 80)

    return data


def is_numeric_array(arr):
    arr = np.asarray(arr)
    return np.issubdtype(arr.dtype, np.number)


def find_true_key(data):
    preferred = [
        "y29_true",
        "y_true",
        "true",
        "truth",
        "test_true",
        "pg29_true",
        "Y_true",
    ]

    for key in preferred:
        if key in data.files:
            arr = np.asarray(data[key])
            if arr.ndim == 2 and is_numeric_array(arr):
                return key

    for key in data.files:
        lk = key.lower()
        arr = np.asarray(data[key])
        if ("true" in lk or "truth" in lk) and arr.ndim == 2 and is_numeric_array(arr):
            return key

    raise KeyError("Could not find a 2D true array in this npz file.")


def is_prediction_key(key):
    lk = key.lower()

    bad_tokens = [
        "true",
        "truth",
        "std",
        "sigma",
        "uncert",
        "weight",
        "weights",
        "safe_w",
        "idx",
        "time",
        "times",
        "sensor",
        "tag",
        "mask",
    ]

    for bad in bad_tokens:
        if bad in lk:
            return False

    good_tokens = [
        "pred",
        "dm",
        "vae",
        "gan",
        "anchor",
        "gate",
        "cal",
        "only",
        "mean",
        "y29_anchor",
    ]

    return any(good in lk for good in good_tokens)


def find_prediction_keys(data, true_shape):
    pred_keys = []

    for key in data.files:
        arr = np.asarray(data[key])

        if arr.shape != true_shape:
            continue

        if not is_numeric_array(arr):
            continue

        if not is_prediction_key(key):
            continue

        pred_keys.append(key)

    return pred_keys


# ------------------------------------------------------------
# FIX: helper copied from the training scripts. Needed to
# recompute the coordinate-file sensor-tag column the same way
# training did, so the reduced sensor_names list matches exactly.
# ------------------------------------------------------------
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


def _ts_cols_from_columns(cols):
    return sorted([c for c in cols if str(c).strip().upper().startswith("TS-")])


_TRUE_SENSOR_NAMES_CACHE = {}


def get_true_sensor_names(file_label=None):
    """
    Recompute the exact reduced/sorted sensor_names list the training
    script used (sensor_names = sorted(pg27_ts & pg29_ts & coord_ts)),
    so fixed_missing_idx can be mapped back to the correct sensor tag
    instead of indexing into the raw, unfiltered PG29 CSV column list.

    Cached per file_label since SENSOR_SOURCE_OVERRIDES may differ by
    family (DM / VAE / GAN).
    """
    cache_key = file_label if file_label in SENSOR_SOURCE_OVERRIDES else "__default__"

    if cache_key in _TRUE_SENSOR_NAMES_CACHE:
        return _TRUE_SENSOR_NAMES_CACHE[cache_key]

    paths = SENSOR_SOURCE_OVERRIDES.get(file_label, {})
    pg27_path = paths.get("PG27_PATH", PG27_PATH)
    pg29_path = paths.get("PG29_PATH", PG29_PATH)
    coord_path = paths.get("COORD_PATH", COORD_PATH)

    pg27_cols = list(pd.read_csv(pg27_path, nrows=1).columns)
    pg29_cols = list(pd.read_csv(pg29_path, nrows=1).columns)
    coord_df_head = pd.read_csv(coord_path)

    pg27_ts = set(_ts_cols_from_columns(pg27_cols))
    pg29_ts = set(_ts_cols_from_columns(pg29_cols))

    name_col = find_name_col(coord_df_head)
    coord_ts = set([
        s for s in coord_df_head[name_col].astype(str).str.strip().values
        if str(s).upper().startswith("TS")
    ])

    sensor_names = sorted(pg27_ts & pg29_ts & coord_ts)
    _TRUE_SENSOR_NAMES_CACHE[cache_key] = sensor_names
    return sensor_names


def get_sensor_names_from_npz(data, n_sensors):
    candidates = [
        "target_sensors",
        "test_sensors",
        "missing_sensors",
        "fixed_missing_sensors",
        "sensor_names",
        "sensors",
        "target_tags",
        "test_tags",
        "missing_tags",
    ]

    for key in candidates:
        if key in data.files:
            arr = np.asarray(data[key])
            if arr.ndim == 1 and len(arr) == n_sensors:
                out = []
                for x in arr:
                    if isinstance(x, bytes):
                        out.append(x.decode("utf-8"))
                    else:
                        out.append(str(x))
                return out

    for key in data.files:
        lk = key.lower()
        arr = np.asarray(data[key])
        if ("sensor" in lk or "tag" in lk) and arr.ndim == 1 and len(arr) == n_sensors:
            out = []
            for x in arr:
                if isinstance(x, bytes):
                    out.append(x.decode("utf-8"))
                else:
                    out.append(str(x))
            return out

    return None


def get_sensor_names_from_fixed_idx(data, n_sensors, file_label=None):
    """
    FIXED: previously indexed into the raw PG29 CSV's column list
    (which includes every TS sensor in the file, in file order). But
    fixed_missing_idx was built at training time as an index into
    sensor_names = sorted(pg27_ts & pg29_ts & coord_ts) -- a reduced,
    alphabetically sorted subset. Indexing the wrong list silently
    returns the wrong sensor tag (e.g. "TS-1721" instead of the real
    missing sensor). Now we recompute that same reduced/sorted list
    from the original source CSVs and index into that instead.
    """
    if "fixed_missing_idx" not in data.files:
        return None

    try:
        fixed_idx = np.asarray(data["fixed_missing_idx"], dtype=int).reshape(-1)
        if len(fixed_idx) != n_sensors:
            return None

        sensor_names = get_true_sensor_names(file_label=file_label)
        if len(sensor_names) == 0:
            return None

        if int(np.max(fixed_idx)) >= len(sensor_names):
            return None

        return [sensor_names[int(i)] for i in fixed_idx]
    except Exception as e:
        print(f"WARNING: get_sensor_names_from_fixed_idx failed for {file_label}: {e}")
        return None


def get_sensor_names(data, n_sensors, file_label=None):
    names = get_sensor_names_from_npz(data, n_sensors)
    if names is not None:
        return names

    names = get_sensor_names_from_fixed_idx(data, n_sensors, file_label=file_label)
    if names is not None:
        return names

    return [f"sensor_{i:03d}" for i in range(n_sensors)]


def get_time_values(data, n_time):
    candidates = [
        "eval_times",
        "times",
        "time",
        "t",
        "time_index",
        "test_times",
    ]

    for key in candidates:
        if key in data.files:
            arr = np.asarray(data[key])
            if arr.ndim == 1 and len(arr) == n_time:
                return arr

    for key in data.files:
        lk = key.lower()
        arr = np.asarray(data[key])
        if ("time" in lk or lk == "t") and arr.ndim == 1 and len(arr) == n_time:
            return arr

    return np.arange(n_time)


def make_time_hours(time_values):
    time_values = np.asarray(time_values, dtype=float).reshape(-1)
    if len(time_values) == 0:
        return time_values

    time_values = time_values - time_values[0]
    return time_values * TIME_STEP_SECONDS / 3600.0


def clean_method_name(file_label, key):
    if key in METHOD_NAME_MAP:
        return METHOD_NAME_MAP[key]
    return f"{file_label}_{key}"


def is_family_method(file_label, npz_key, method_name):
    fl = str(file_label).lower()
    key = str(npz_key).lower()
    method = str(method_name).lower()

    if "anchor" in key or method == "anchor" or "anchor_only" in method:
        return False

    if fl == "dm":
        return ("dm" in key) or ("dm" in method)

    if fl == "vae":
        return ("vae" in key) or ("vae" in method)

    if fl == "gan":
        gan_tokens = [
            "gan",
            "gate",
            "safe_raw",
            "safe",
        ]
        return any(tok in key for tok in gan_tokens) or any(tok in method for tok in gan_tokens)

    return False


def find_regression_entry_from_dm(all_entries):
    dm_entries = [e for e in all_entries if e["file_label"] == "DM"]

    for cand in REGRESSION_KEY_CANDIDATES:
        for entry in dm_entries:
            if entry["npz_key"] == cand:
                return entry

    cand_norms = [normalize_key(c) for c in REGRESSION_KEY_CANDIDATES]
    for entry in dm_entries:
        key_norm = normalize_key(entry["npz_key"])
        if key_norm in cand_norms:
            return entry

    for entry in dm_entries:
        key_norm = normalize_key(entry["npz_key"])
        method_norm = normalize_key(entry["method"])
        if "anchor" in key_norm or "anchor" in method_norm:
            return entry

    print("\nAvailable DM entries:")
    for entry in dm_entries:
        print(f"  {entry['npz_key']} -> {entry['method']}")

    raise KeyError(
        "Could not find regression anchor from DM npz. "
        "Tried Anchor_only, y29_anchor, pred_anchor, and other anchor-like keys."
    )


def align_entries(entries):
    sensor_sets = [set(entry["sensors"]) for entry in entries]
    common_sensors = sorted(set.intersection(*sensor_sets))

    if len(common_sensors) == 0:
        print("\nWARNING: No common sensor names found. Falling back to index alignment.")
        min_sensors = min(entry["y_pred"].shape[1] for entry in entries)
        common_sensors = [f"sensor_{i:03d}" for i in range(min_sensors)]

        for entry in entries:
            entry["y_true_aligned"] = entry["y_true"][:, :min_sensors]
            entry["y_pred_aligned"] = entry["y_pred"][:, :min_sensors]
            entry["sensors_aligned"] = common_sensors
    else:
        for entry in entries:
            sensor_to_col = {s: i for i, s in enumerate(entry["sensors"])}
            cols = [sensor_to_col[s] for s in common_sensors]
            entry["y_true_aligned"] = entry["y_true"][:, cols]
            entry["y_pred_aligned"] = entry["y_pred"][:, cols]
            entry["sensors_aligned"] = common_sensors

    min_time = min(entry["y_pred_aligned"].shape[0] for entry in entries)

    for entry in entries:
        entry["y_true_aligned"] = entry["y_true_aligned"][:min_time, :]
        entry["y_pred_aligned"] = entry["y_pred_aligned"][:min_time, :]
        entry["times_aligned"] = entry["times"][:min_time]

    return entries, common_sensors, min_time


def get_plot_array(label, y_raw):
    if SMOOTH_ALL_FOR_PLOT:
        return smooth_for_plot(y_raw, window=DM_SMOOTH_WINDOW)

    if label == "DM" and SMOOTH_DM_FOR_PLOT:
        return smooth_for_plot(y_raw, window=DM_SMOOTH_WINDOW)

    return y_raw


def plot_one_sensor(ax, sensor, y_true, method_arrays, time_hours):
    markevery = max(1, len(time_hours) // 18)

    ax.plot(
        time_hours,
        y_true,
        label="True",
        linewidth=1.9,
        color=MODEL_COLORS["True"],
        linestyle="-",
    )

    for label in PLOT_ORDER:
        y_raw = method_arrays[label]
        y_plot = get_plot_array(label, y_raw)

        ax.plot(
            time_hours,
            y_plot,
            label=label,
            linewidth=MODEL_LINEWIDTHS.get(label, 1.35),
            linestyle=MODEL_LINESTYLES.get(label, "-"),
            marker=MODEL_MARKERS.get(label, None),
            markevery=markevery,
            markersize=3.0,
            color=MODEL_COLORS.get(label, None),
            alpha=0.95,
        )

    ax.set_title(sensor)
    ax.grid(True, alpha=0.3)


def add_shared_legend(fig, axes, ncol=5):
    axes_flat = np.asarray(axes).reshape(-1)

    for ax in axes_flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=ncol,
                frameon=False,
            )
            return


def choose_best_mid_worst_sensors(metrics_df):
    sort_df = metrics_df.sort_values("DM_RMSE", ascending=True).reset_index(drop=True)

    best = sort_df.head(4)["sensor"].tolist()
    worst = sort_df.tail(4)["sensor"].tolist()

    n = len(sort_df)
    mid_start = max(0, n // 2 - 2)
    mid = sort_df.iloc[mid_start:mid_start + 4]["sensor"].tolist()

    used = set(best) | set(worst)
    mid_clean = []

    for sensor in mid:
        if sensor not in used:
            mid_clean.append(sensor)

    if len(mid_clean) < 4:
        for sensor in sort_df["sensor"].tolist():
            if sensor not in used and sensor not in mid_clean:
                mid_clean.append(sensor)
            if len(mid_clean) == 4:
                break

    selected = []
    group_labels = []

    for sensor in best:
        selected.append(sensor)
        group_labels.append("Best")

    for sensor in mid_clean[:4]:
        selected.append(sensor)
        group_labels.append("Middle")

    for sensor in worst:
        selected.append(sensor)
        group_labels.append("Worst")

    return selected, group_labels


# ============================================================
# 5. LOAD ALL FILES AND COMPUTE ALL MODEL METRICS
# ============================================================
all_entries = []

for file_label, path in NPZ_FILES.items():
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing npz file for {file_label}: {path}")

    data = print_npz_keys(file_label, path)

    true_key = find_true_key(data)
    y_true = np.asarray(data[true_key], dtype=float)

    n_time, n_sensors = y_true.shape
    sensors = get_sensor_names(data, n_sensors, file_label=file_label)
    times = get_time_values(data, n_time)

    pred_keys = find_prediction_keys(data, true_shape=y_true.shape)

    print(f"\nDetected true key for {file_label}: {true_key}")
    print(f"Detected prediction keys for {file_label}:")
    for key in pred_keys:
        print(f"  {key}")
    print(f"Resolved sensor names for {file_label}: {sensors}")

    if len(pred_keys) == 0:
        print(f"WARNING: No prediction arrays found in {file_label}")
        continue

    for pred_key in pred_keys:
        y_pred = np.asarray(data[pred_key], dtype=float)
        mae, rmse, r2 = safe_metrics(y_true, y_pred)

        method_name = clean_method_name(file_label, pred_key)

        all_entries.append({
            "file_label": file_label,
            "method": method_name,
            "method_id": safe_name(f"{file_label}_{pred_key}"),
            "npz_key": pred_key,
            "true_key": true_key,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "n_time": n_time,
            "n_sensors": n_sensors,
            "source_path": path,
            "y_true": y_true,
            "y_pred": y_pred,
            "sensors": sensors,
            "times": times,
        })

if len(all_entries) == 0:
    raise RuntimeError("No prediction arrays were found in any npz file.")


# ============================================================
# 6. SAVE ALL GLOBAL METRICS
# ============================================================
metrics_rows = []

for entry in all_entries:
    metrics_rows.append({
        "file_label": entry["file_label"],
        "method": entry["method"],
        "method_id": entry["method_id"],
        "npz_key": entry["npz_key"],
        "true_key": entry["true_key"],
        "MAE": entry["MAE"],
        "RMSE": entry["RMSE"],
        "R2": entry["R2"],
        "n_time": entry["n_time"],
        "n_sensors": entry["n_sensors"],
        "source_path": entry["source_path"],
    })

metrics_df_all = pd.DataFrame(metrics_rows).sort_values("RMSE", ascending=True)
metrics_all_path = OUT_DIR / "all_npz_model_global_metrics_ranked.csv"
metrics_df_all.to_csv(metrics_all_path, index=False)

print("\n" + "=" * 80)
print("All prediction arrays ranked by RMSE:")
print(metrics_df_all[["file_label", "method", "npz_key", "MAE", "RMSE", "R2"]].to_string(index=False))
print("=" * 80)


# ============================================================
# 7. SELECT PAPER METHODS AND DM-ANCHOR REGRESSION
# ============================================================
# Use the raw model output for DM and VAE, the safe anchor-gated output for
# GAN, and the anchor-only prediction stored in the DM file as regression.
# This prevents automatic RMSE ranking from selecting an unintended variant.
PREFERRED_METHOD_KEYS = {
    "DM": "pred_dm",
    "VAE": "pred_vae",
    "GAN": "pred_safe_raw_gan_anchor",
}


def find_preferred_entry(all_model_entries, file_label, preferred_key):
    """Return one exact preferred prediction entry from the requested file."""
    preferred_norm = normalize_key(preferred_key)
    matches = [
        entry
        for entry in all_model_entries
        if entry["file_label"] == file_label
        and normalize_key(entry["npz_key"]) == preferred_norm
    ]

    if len(matches) == 0:
        available = sorted(
            entry["npz_key"]
            for entry in all_model_entries
            if entry["file_label"] == file_label
        )
        raise KeyError(
            f"Could not find preferred {file_label} prediction key "
            f"'{preferred_key}'. Available keys: {available}"
        )

    if len(matches) > 1:
        source_paths = sorted({entry["source_path"] for entry in matches})
        raise RuntimeError(
            f"Found multiple matches for {file_label} key '{preferred_key}' "
            f"in sources: {source_paths}"
        )

    return matches[0]


selected_entries = {
    family: find_preferred_entry(
        all_model_entries=all_entries,
        file_label=family,
        preferred_key=preferred_key,
    )
    for family, preferred_key in PREFERRED_METHOD_KEYS.items()
}

reg_entry = find_regression_entry_from_dm(all_entries)
selected_entries["Regression"] = reg_entry

selected_rows = []
for label in PLOT_ORDER:
    entry = selected_entries[label]
    selected_rows.append({
        "plot_label": label,
        "source_file": entry["file_label"],
        "method": entry["method"],
        "npz_key": entry["npz_key"],
        "MAE": entry["MAE"],
        "RMSE": entry["RMSE"],
        "R2": entry["R2"],
        "source_path": entry["source_path"],
    })

selected_global_df = pd.DataFrame(selected_rows)
selected_global_path = OUT_DIR / "selected_DM_VAE_GAN_anchor_regression_global_metrics.csv"
selected_global_df.to_csv(selected_global_path, index=False)

print("\n" + "=" * 80)
print("Selected methods for plotting:")
print(selected_global_df[["plot_label", "method", "npz_key", "MAE", "RMSE", "R2"]].to_string(index=False))
print("=" * 80)


# ============================================================
# 8. ALIGN SELECTED METHODS
# ============================================================
entries_for_alignment = [selected_entries[label] for label in PLOT_ORDER]
entries_for_alignment, sensors_plot, min_time = align_entries(entries_for_alignment)

for label, entry in zip(PLOT_ORDER, entries_for_alignment):
    selected_entries[label] = entry

# Use DM truth as common reference.
y_true_plot = selected_entries["DM"]["y_true_aligned"]

print("\nAligned comparison:")
print(f"  common sensors: {len(sensors_plot)}")
print(f"  common time length: {min_time}")

for label in PLOT_ORDER:
    entry = selected_entries[label]
    diff = np.nanmax(np.abs(y_true_plot - entry["y_true_aligned"]))
    print(f"Max true difference vs DM truth for {label}: {diff:.6g}")
    if diff > 1e-5:
        print("WARNING: true arrays differ. Metrics and plots use DM truth as reference.")

time_hours = make_time_hours(selected_entries["DM"]["times_aligned"])


# ============================================================
# 9. ALIGNED GLOBAL METRICS
# ============================================================
aligned_global_rows = []

for label in PLOT_ORDER:
    entry = selected_entries[label]
    mae, rmse, r2 = safe_metrics(y_true_plot, entry["y_pred_aligned"])

    aligned_global_rows.append({
        "method": label,
        "source_method": entry["method"],
        "npz_key": entry["npz_key"],
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "n_time": y_true_plot.shape[0],
        "n_sensors": y_true_plot.shape[1],
        "source_path": entry["source_path"],
    })

aligned_global_df = pd.DataFrame(aligned_global_rows).sort_values("RMSE")
aligned_global_path = OUT_DIR / "aligned_global_metrics_DM_VAE_GAN_Regression.csv"
aligned_global_df.to_csv(aligned_global_path, index=False)

print("\nAligned global metrics:")
print(aligned_global_df[["method", "source_method", "npz_key", "MAE", "RMSE", "R2"]].to_string(index=False))


# ============================================================
# 10. PER-SENSOR METRICS
# ============================================================
per_sensor_rows = []

for j, sensor in enumerate(sensors_plot):
    row = {"sensor": sensor}

    for label in PLOT_ORDER:
        entry = selected_entries[label]
        mae, rmse, r2 = safe_metrics(y_true_plot[:, j], entry["y_pred_aligned"][:, j])

        row[f"{label}_MAE"] = mae
        row[f"{label}_RMSE"] = rmse
        row[f"{label}_R2"] = r2

    per_sensor_rows.append(row)

per_sensor_df = pd.DataFrame(per_sensor_rows)
per_sensor_path = OUT_DIR / "per_sensor_metrics_DM_VAE_GAN_Regression.csv"
per_sensor_df.to_csv(per_sensor_path, index=False)


# ============================================================
# 11. SAVE COMBINED PREDICTIONS
# ============================================================
pred_data = {}

for j, sensor in enumerate(sensors_plot):
    pred_data[f"{sensor}_true"] = y_true_plot[:, j]

    for label in PLOT_ORDER:
        pred_data[f"{sensor}_{label}_pred"] = selected_entries[label]["y_pred_aligned"][:, j]

pred_df = pd.DataFrame(pred_data)
pred_path = OUT_DIR / "combined_predictions_DM_VAE_GAN_Regression.csv"
pred_df.to_csv(pred_path, index=False)


# ============================================================
# 12. RMSE BAR PLOT
# ============================================================
plot_df = per_sensor_df.sort_values("DM_RMSE", ascending=True).reset_index(drop=True)

x = np.arange(len(plot_df))
width = 0.8 / len(PLOT_ORDER)

fig, ax = plt.subplots(figsize=(20, 6))

for i, label in enumerate(PLOT_ORDER):
    offset = (i - (len(PLOT_ORDER) - 1) / 2.0) * width

    ax.bar(
        x + offset,
        plot_df[f"{label}_RMSE"],
        width=width,
        label=label,
        color=MODEL_COLORS.get(label, None),
        hatch=MODEL_HATCHES.get(label, ""),
        edgecolor="black",
        linewidth=0.4,
    )

ax.set_xticks(x)
ax.set_xticklabels(plot_df["sensor"], rotation=60, ha="right")
ax.set_ylabel("RMSE ($^\\circ$C)")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(ncol=4)
fig.tight_layout()

rmse_png = OUT_DIR / "per_sensor_RMSE_bar_DM_VAE_GAN_Regression.png"
rmse_pdf = OUT_DIR / "per_sensor_RMSE_bar_DM_VAE_GAN_Regression.pdf"

fig.savefig(rmse_png, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
fig.savefig(rmse_pdf, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)


# ============================================================
# 13. ALL-SENSOR TREND PLOT
# ============================================================
n_sensors = len(sensors_plot)
ncols = 5
nrows = int(math.ceil(n_sensors / ncols))

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(24, 3.0 * nrows),
    sharex=True,
)

axes = np.asarray(axes).reshape(-1)

for ax_idx, sensor in enumerate(sensors_plot):
    ax = axes[ax_idx]
    method_arrays = {}

    for label in PLOT_ORDER:
        method_arrays[label] = selected_entries[label]["y_pred_aligned"][:, ax_idx]

    plot_one_sensor(
        ax=ax,
        sensor=sensor,
        y_true=y_true_plot[:, ax_idx],
        method_arrays=method_arrays,
        time_hours=time_hours,
    )

for k in range(n_sensors, len(axes)):
    axes[k].axis("off")

add_shared_legend(fig, axes, ncol=5)
fig.supxlabel("Time (h)")
fig.supylabel("Temperature ($^\\circ$C)")
fig.tight_layout(rect=[0.03, 0.04, 0.98, 0.95])

all_trend_png = OUT_DIR / "trend_all_sensors_true_DM_VAE_GAN_Regression.png"
all_trend_pdf = OUT_DIR / "trend_all_sensors_true_DM_VAE_GAN_Regression.pdf"

fig.savefig(all_trend_png, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
fig.savefig(all_trend_pdf, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)


# ============================================================
# 14. 3X4 TREND PLOT: 4 BEST, 4 MIDDLE, 4 WORST BASED ON DM RMSE
# ============================================================
selected_sensors, group_labels = choose_best_mid_worst_sensors(per_sensor_df)

fig, axes = plt.subplots(
    3,
    4,
    figsize=(18, 10),
    sharex=True,
)

axes = np.asarray(axes).reshape(-1)

for ax_idx, sensor in enumerate(selected_sensors):
    ax = axes[ax_idx]
    j = sensors_plot.index(sensor)

    method_arrays = {}
    for label in PLOT_ORDER:
        method_arrays[label] = selected_entries[label]["y_pred_aligned"][:, j]

    plot_one_sensor(
        ax=ax,
        sensor=sensor,
        y_true=y_true_plot[:, j],
        method_arrays=method_arrays,
        time_hours=time_hours,
    )

add_shared_legend(fig, axes, ncol=5)
fig.supxlabel("Time (h)")
fig.supylabel("Temperature ($^\\circ$C)")
fig.tight_layout(rect=[0.03, 0.04, 0.98, 0.95])

selected_trend_png = OUT_DIR / "trend_3x4_best_middle_worst_by_DM_true_DM_VAE_GAN_Regression.png"
selected_trend_pdf = OUT_DIR / "trend_3x4_best_middle_worst_by_DM_true_DM_VAE_GAN_Regression.pdf"

fig.savefig(selected_trend_png, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
fig.savefig(selected_trend_pdf, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

selected_table = pd.DataFrame({
    "group": group_labels,
    "sensor": selected_sensors,
})
selected_table_path = OUT_DIR / "selected_3x4_sensors_best_middle_worst_by_DM.csv"
selected_table.to_csv(selected_table_path, index=False)


# ============================================================
# 15. SAVE SENSOR RANKING
# ============================================================
rank_path = OUT_DIR / "sensor_ranking_by_DM_RMSE.csv"

per_sensor_df.sort_values("DM_RMSE", ascending=True)[[
    "sensor",
    "DM_RMSE",
    "VAE_RMSE",
    "GAN_RMSE",
    "Regression_RMSE",
    "DM_MAE",
    "VAE_MAE",
    "GAN_MAE",
    "Regression_MAE",
    "DM_R2",
    "VAE_R2",
    "GAN_R2",
    "Regression_R2",
]].to_csv(rank_path, index=False)


# ============================================================
# 16. DONE
# ============================================================
print("\nSaved outputs:")
print(metrics_all_path)
print(selected_global_path)
print(aligned_global_path)
print(per_sensor_path)
print(pred_path)
print(rmse_png)
print(rmse_pdf)
print(all_trend_png)
print(all_trend_pdf)
print(selected_trend_png)
print(selected_trend_pdf)
print(selected_table_path)
print(rank_path)
print("\nDone.")
