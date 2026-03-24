# =======================
# End-to-End Dataset Prep (PTB-XL, EchoNext, MIMIC-IV)
# =======================
import os
import gc
import ast
import sys
import glob
import numpy as np
import torch
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, medfilt
from scipy.stats import t as student_t  # if SciPy is not available, see note below
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.utils.data import TensorDataset, Subset

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fastai.metrics import mae
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.callback.core import Callback
from fastai.callback.fp16 import MixedPrecision
from fastai.callback.schedule import fit_one_cycle
from fastai.callback.training import GradientClip
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback

import os, glob
from typing import List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# =======================
# Models
# =======================
sys.path.append("/data/BCS/Services/Ronika_De/ImageDS/Analysis/ecg_ptbxl_benchmarking-master/code/")
from models.xresnet1d_custom import xresnet1d50_ultradeep
from models.resnet1d import resnet1d18, resnet1d34, resnet1d50, resnet1d101, resnet1d152, resnet1d_wang, wrn1d_22
from models.xresnet1d import xresnet1d18, xresnet1d34, xresnet1d50, xresnet1d101, xresnet1d152, xresnet1d18_deep, xresnet1d34_deep, xresnet1d50_deep, xresnet1d18_deeper, xresnet1d34_deeper, xresnet1d50_deeper
from models.inception1d import inception1d

# ============================================================
# GLOBALS
# ============================================================
SEED = 42
rng = np.random.default_rng(SEED)

MIN_AGE = 18
MAX_AGE = 89

AGE_BINS = np.arange(18, 95, 5)  # for sex ratio plots
MISSING_SEX_CODE = -1            # 0=male, 1=female, -1=unknown

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE, TEST_BS, EPOCHS, LR = 256, 256, 50, 1e-2
age_bins = np.arange(18, 95, 5)

NUM_CHANNELS = 12
NUM_OUTPUT = 1

# ============================================================
# HELPERS
# ============================================================
def make_sex_tensor(values,
                    male_tokens=("male", "m", "M", 0),
                    female_tokens=("female", "f", "F", 1),
                    missing_code=MISSING_SEX_CODE) -> torch.LongTensor:
    """
    Convert a 1D array-like of sex tokens (strings or numbers) into a torch.long tensor:
      male -> 0, female -> 1, unknown -> missing_code.
    Works for PTB-XL ("male"/"female"), EchoNext ("Male"/"Female"), MIMIC ("M"/"F").
    """
    arr = np.asarray(values, dtype=object)
    out = np.full(arr.shape, missing_code, dtype=np.int64)
    low = np.char.lower(arr.astype(str))

    male = np.array([str(t).lower() for t in male_tokens], dtype=object)
    female = np.array([str(t).lower() for t in female_tokens], dtype=object)

    male_mask = np.isin(low, male) | np.isin(arr, [0])
    female_mask = np.isin(low, female) | np.isin(arr, [1])

    out[male_mask] = 0
    out[female_mask] = 1
    return torch.from_numpy(out)

def ensure_channel_first_torch(X: torch.Tensor) -> torch.FloatTensor:
    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X)
    X = X.float()
    if X.ndim != 3:
        raise ValueError(f"Expected 3D tensor (N,*,*); got shape {tuple(X.shape)}")
    N, A, B = X.shape
    # ECG channels are always small (<= 32)
    if A <= 32 and B > A:
        return X  # already (N, C, T)
    elif B <= 32 and A > B:
        return X.permute(0, 2, 1)  # (N, T, C) → (N, C, T)
    else:
        raise ValueError(f"Cannot infer channel dimension from shape {X.shape}")

def safe_np_save(path: str, arr: np.ndarray):
    """
    Atomic np.save to avoid truncated files on shared filesystems.
    """
    tmp = path + ".tmp.npy"
    np.save(tmp, arr, allow_pickle=False)
    os.replace(tmp, path)  # atomic on POSIX

def load_npy_concat(file_glob: str,
                    require_ndim: Optional[int] = None,
                    axis: int = 0,
                    name: str = "arr") -> Optional[np.ndarray]:
    """
    Load multiple .npy files matching pattern, skipping 0-byte/corrupt files,
    and concatenating along axis. Returns None if no good files found.
    """
    files = sorted(glob.glob(file_glob))
    if not files:
        print(f"[WARN] No files for pattern: {file_glob}")
        return None

    arrs: List[np.ndarray] = []
    bad = 0
    ref_shape = None
    for f in files:
        try:
            if os.path.getsize(f) == 0:
                bad += 1
                print(f"[SKIP] 0-byte file: {f}")
                continue
            arr = np.load(f, allow_pickle=False)
            if require_ndim is not None and arr.ndim != require_ndim:
                bad += 1
                print(f"[SKIP] ndim mismatch in {f}: got {arr.ndim}, want {require_ndim}")
                continue
            if ref_shape is None:
                ref_shape = arr.shape[1:]
            else:
                if arr.shape[1:] != ref_shape:
                    bad += 1
                    print(f"[SKIP] shape mismatch in {f}: {arr.shape} vs ref (?,?,{ref_shape})")
                    continue
            arrs.append(arr)
        except Exception as e:
            bad += 1
            print(f"[SKIP] {f} -> {type(e).__name__}: {e}")
            continue

    if not arrs:
        print(f"[ERROR] No good {name} files found for pattern: {file_glob}")
        return None

    cat = np.concatenate(arrs, axis=axis)
    print(f"[OK] Loaded {name}: {len(arrs)} files, {bad} skipped, concatenated shape = {cat.shape}")
    return cat

def assign_age_bins(ages):
    return np.digitize(ages, AGE_BINS) - 1

def sex_counts_by_bin(ages, sex):
    bins = assign_age_bins(ages)
    n = len(AGE_BINS) - 1
    male = np.zeros(n, dtype=int)
    female = np.zeros(n, dtype=int)
    for b in range(n):
        idx = bins == b
        male[b] = np.sum(sex[idx] == 0)
        female[b] = np.sum(sex[idx] == 1)
    return male, female

def plot_sex_ratio(male, female, title, fname):
    labels = [f"{AGE_BINS[i]}–{AGE_BINS[i+1]-1}" for i in range(len(AGE_BINS)-1)]
    x = np.arange(len(labels))
    w = 0.4
    plt.figure(figsize=(14,5))
    plt.bar(x-w/2, male, w, label="Male")
    plt.bar(x+w/2, female, w, label="Female")
    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Segments")
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

# ============================================================
# AGE / OVERLAP AUDIT (METADATA ONLY, HEALTHY-ONLY REMOVAL)
# ============================================================
def audit_age_and_overlap_metadata(
    name,
    ages,
    pids,
    labels,
    min_age=MIN_AGE,
    max_age=MAX_AGE,
):
    """
    Memory‑safe audit:
    - prints counts before/after age filtering
    - removes overlapping patients ONLY from healthy set
    - returns final_mask (boolean array)
    """

    labels = labels.astype(int)

    # ---------------- BEFORE ----------------
    H_before = int((labels == 0).sum())
    D_before = int((labels == 1).sum())
    Hp_before = len(np.unique(pids[labels == 0]))
    Dp_before = len(np.unique(pids[labels == 1]))

    # ---------------- AGE FILTER ----------------
    missing = np.isnan(ages)
    valid_age = (~missing) & (ages >= min_age) & (ages <= max_age)

    ages_v = ages[valid_age]
    pids_v = pids[valid_age]
    labs_v = labels[valid_age]

    H_after_age = int((labs_v == 0).sum())
    D_after_age = int((labs_v == 1).sum())
    Hp_after_age = len(np.unique(pids_v[labs_v == 0]))
    Dp_after_age = len(np.unique(pids_v[labs_v == 1]))

    # ---------------- OVERLAP ----------------
    pid_H = set(np.unique(pids_v[labs_v == 0]))
    pid_D = set(np.unique(pids_v[labs_v == 1]))
    overlap = pid_H.intersection(pid_D)
    n_overlap = len(overlap)

    # Remove overlapping patients ONLY from healthy
    keep = np.ones_like(labs_v, dtype=bool)
    if n_overlap > 0:
        ov = np.array(list(overlap))
        keep = keep & ~((labs_v == 0) & np.isin(pids_v, ov))

    final_mask = valid_age.copy()
    final_mask[valid_age] = keep

    # ---------------- AFTER OVERLAP ----------------
    ages_f = ages[final_mask]
    pids_f = pids[final_mask]
    labs_f = labels[final_mask]

    H_after_ov = int((labs_f == 0).sum())
    D_after_ov = int((labs_f == 1).sum())
    Hp_after_ov = len(np.unique(pids_f[labs_f == 0]))
    Dp_after_ov = len(np.unique(pids_f[labs_f == 1]))

    # ---------------- PRINT REPORT ----------------
    print(f"\n================ {name} — AGE/OVERLAP AUDIT ================")
    print("--- BEFORE ---")
    print(f"Healthy segments   : {H_before}")
    print(f"Diseased segments  : {D_before}")
    print(f"Healthy patients   : {Hp_before}")
    print(f"Diseased patients  : {Dp_before}")

    print("\n--- AFTER AGE FILTER ---")
    print(f"Healthy segments   : {H_after_age}")
    print(f"Diseased segments  : {D_after_age}")
    print(f"Healthy patients   : {Hp_after_age}")
    print(f"Diseased patients  : {Dp_after_age}")

    print("\n--- OVERLAP ---")
    print(f"Overlapping patients (H∩D): {n_overlap}")

    print("\n--- AFTER OVERLAP REMOVAL (healthy only) ---")
    print(f"Healthy segments   : {H_after_ov}")
    print(f"Diseased segments  : {D_after_ov}")
    print(f"Healthy patients   : {Hp_after_ov}")
    print(f"Diseased patients  : {Dp_after_ov}")

    return final_mask

# ============================================================
# PTB-XL (IN-MEMORY)
# ============================================================
print("\n================ PTB-XL ====================")
ptbxl_meta = pd.read_csv("/data/BCS/Services/Ronika_De/ImageDS/PTB/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv")

ptbxl_path = "/data/BCS/Services/Ronika_De/ImageDS/PTB/physionet.org/files/ptb-xl/1.0.3/cache/ptbxl_ecg_segments.npz"
ptbxl_npz = np.load(ptbxl_path, allow_pickle=False)

X_ptbxl           = torch.from_numpy(ptbxl_npz["X"]).float()                  # (N, T, C)
ages_ptbxl        = torch.from_numpy(ptbxl_npz["y"]).float()                  # (N,)
patient_ids_ptbxl = torch.from_numpy(ptbxl_npz["patient_ids"]).long()         # (N,)
is_diseased_np    = ptbxl_npz["is_diseased"]
labels_bool_ptbxl = torch.from_numpy(is_diseased_np.astype(np.bool_))         # (N,) False=healthy, True=diseased

# Sex from metadata (map by patient_id)
sex_map_ptbxl = ptbxl_meta.set_index("patient_id")["sex"].to_dict()
sex_arr_ptbxl = np.array([sex_map_ptbxl.get(int(pid.item()), np.nan) for pid in patient_ids_ptbxl], dtype=object)
sex_ptbxl = make_sex_tensor(sex_arr_ptbxl, male_tokens=("male","m",0), female_tokens=("female","f",1))

# Audit + mask (metadata only, healthy-only overlap removal)
final_mask_ptbxl = audit_age_and_overlap_metadata(
    "PTB-XL",
    ages_ptbxl.numpy(),
    patient_ids_ptbxl.numpy(),
    labels_bool_ptbxl.numpy().astype(int)
)
mask_t = torch.from_numpy(final_mask_ptbxl)

X_ptbxl = X_ptbxl[mask_t]
ages_ptbxl = ages_ptbxl[mask_t]
patient_ids_ptbxl = patient_ids_ptbxl[mask_t]
labels_bool_ptbxl = labels_bool_ptbxl[mask_t]
sex_ptbxl = sex_ptbxl[mask_t]

# Channel-first and split
X_ptbxl_cf = ensure_channel_first_torch(X_ptbxl)
h_mask_t = ~labels_bool_ptbxl
d_mask_t = labels_bool_ptbxl

Xw_h_torch_ptbxl   = X_ptbxl_cf[h_mask_t]
Xw_d_torch_ptbxl   = X_ptbxl_cf[d_mask_t]
yh_age_torch_ptbxl = ages_ptbxl[h_mask_t]
yd_age_torch_ptbxl = ages_ptbxl[d_mask_t]
pid_h_torch_ptbxl  = patient_ids_ptbxl[h_mask_t]
pid_d_torch_ptbxl  = patient_ids_ptbxl[d_mask_t]
sex_h_torch_ptbxl  = sex_ptbxl[h_mask_t]
sex_d_torch_ptbxl  = sex_ptbxl[d_mask_t]

# QC plots
m, f = sex_counts_by_bin(yh_age_torch_ptbxl.cpu().numpy(), sex_h_torch_ptbxl.cpu().numpy())
plot_sex_ratio(m, f, "PTB-XL Healthy — Sex by Age", "ptbxl_healthy_sex_age.pdf")
m, f = sex_counts_by_bin(yd_age_torch_ptbxl.cpu().numpy(), sex_d_torch_ptbxl.cpu().numpy())
plot_sex_ratio(m, f, "PTB-XL Diseased — Sex by Age", "ptbxl_diseased_sex_age.pdf")


# ============================================================
# EchoNext (IN-MEMORY)
# ============================================================
print("\n================ EchoNext ====================")
echo_path = "/data/BCS/Services/David_Leone/ACHA-2025/raw_data/EchoNext/upsampled_cache"
echonext_meta = pd.read_csv("/data/BCS/Services/Ronika_De/ImageDS/EchoNext/physionet.org/files/echonext/1.1.0/echonext_metadata_100k.csv")
echonext_meta = echonext_meta.set_index("patient_key")

Xw_h_echonext = np.load(os.path.join(echo_path, "Xw_healthy.npy"), mmap_mode="r")  # (Nh, T, C)
y_h_echonext  = np.load(os.path.join(echo_path, "y_healthy.npy"))                  # (Nh,) ages
pid_h_echonext= np.load(os.path.join(echo_path, "pid_healthy.npy"))                # (Nh,) keys

Xw_d_echonext = np.load(os.path.join(echo_path, "Xw_diseased.npy"), mmap_mode="r") # (Nd, T, C)
y_d_echonext  = np.load(os.path.join(echo_path, "y_diseased.npy"))                 # (Nd,) ages
pid_d_echonext= np.load(os.path.join(echo_path, "pid_diseased.npy"))               # (Nd,) keys

# Build combined metadata for audit
pid_echon_cat = np.concatenate([pid_h_echonext, pid_d_echonext], axis=0)
lab_echon_cat = np.concatenate([
    np.zeros_like(y_h_echonext, dtype=np.int64),
    np.ones_like(y_d_echonext, dtype=np.int64)
], axis=0)

age_map_echon = echonext_meta["age_at_ecg"].to_dict()
sex_map_echon = echonext_meta["sex"].to_dict()
ages_echon_cat = np.array([age_map_echon.get(int(pid), np.nan) for pid in pid_echon_cat], dtype=np.float32)
sex_echon_vals = np.array([sex_map_echon.get(int(pid), np.nan) for pid in pid_echon_cat], dtype=object)

# Audit + mask
final_mask_echon = audit_age_and_overlap_metadata(
    "EchoNext",
    ages_echon_cat,
    pid_echon_cat,
    lab_echon_cat
)
mask_t = torch.from_numpy(final_mask_echon)

# Apply mask to concatenated arrays
X_echon_cat = np.concatenate([Xw_h_echonext, Xw_d_echonext], axis=0)  # still memmap-backed
X_echon = torch.from_numpy(X_echon_cat[final_mask_echon]).float()
ages_echon = torch.from_numpy(ages_echon_cat[final_mask_echon]).float()
pids_echon = torch.from_numpy(pid_echon_cat[final_mask_echon].astype(np.int64)).long()
labels_echon = torch.from_numpy(lab_echon_cat[final_mask_echon].astype(np.int64)).bool()
sex_echon = make_sex_tensor(sex_echon_vals[final_mask_echon], male_tokens=("male","m",0), female_tokens=("female","f",1))

# Channel-first and split
X_echon_cf = ensure_channel_first_torch(X_echon)
h_mask_t = ~labels_echon
d_mask_t = labels_echon

Xw_h_torch_echonext   = X_echon_cf[h_mask_t]
Xw_d_torch_echonext   = X_echon_cf[d_mask_t]
yh_age_torch_echonext = ages_echon[h_mask_t]
yd_age_torch_echonext = ages_echon[d_mask_t]
pid_h_torch_echonext  = pids_echon[h_mask_t]
pid_d_torch_echonext  = pids_echon[d_mask_t]
sex_h_torch_echonext  = sex_echon[h_mask_t]
sex_d_torch_echonext  = sex_echon[d_mask_t]

# QC plots
m, f = sex_counts_by_bin(yh_age_torch_echonext.cpu().numpy(), sex_h_torch_echonext.cpu().numpy())
plot_sex_ratio(m, f, "EchoNext Healthy — Sex by Age", "echonext_healthy_sex_age.pdf")
m, f = sex_counts_by_bin(yd_age_torch_echonext.cpu().numpy(), sex_d_torch_echonext.cpu().numpy())
plot_sex_ratio(m, f, "EchoNext Diseased — Sex by Age", "echonext_diseased_sex_age.pdf")

# ============================================================
# MIMIC-IV (CHUNKED + STREAMING)
# ============================================================
print("\n================ MIMIC-IV ====================")
BASE_DIR_mimic = "/data/BCS/Services/Ronika_De/ImageDS/MIMIC_IV/segments_fast"
OUT_DIR_mimic = "./cache_age89"
os.makedirs(OUT_DIR_mimic, exist_ok=True)

OUT_HEALTHY_mimic  = os.path.join(OUT_DIR_mimic, "healthy")
OUT_DISEASED_mimic = os.path.join(OUT_DIR_mimic, "diseased")
os.makedirs(OUT_HEALTHY_mimic, exist_ok=True)
os.makedirs(OUT_DISEASED_mimic, exist_ok=True)

META_PATH_mimic = (
    "/data/BCS/Services/Ronika_De/ImageDS/MIMIC_IV/"
    "physionet.org/files/mimic-iv-ecg/1.0/records_w_diag_icd10.csv"
)

print("Loading MIMIC-IV metadata...")
mimic_meta = pd.read_csv(META_PATH_mimic)
age_map_mimic = mimic_meta.groupby("subject_id")["age"].first().to_dict()
sex_map_mimic = mimic_meta.groupby("subject_id")["gender"].first().to_dict()

def normalize_sex_mimic(x):
    if x == "M": return 0
    if x == "F": return 1
    return MISSING_SEX_CODE

# Process node directories robustly (preprocessing step)
node_dirs = sorted(glob.glob(os.path.join(BASE_DIR_mimic, "node_*")))
assert len(node_dirs) > 0, "No node directories found!"

healthy_chunk_id = 0
diseased_chunk_id = 0

# ============================================================
# PASS 1 — COLLECT GLOBAL PATIENT SETS
# ============================================================
print("\n[PASS 1] Collecting global patient sets...")

all_pid_H = set()
all_pid_D = set()

for node_dir in node_dirs:
    X_path = os.path.join(node_dir, "X.npy")
    y_path = os.path.join(node_dir, "y.npy")
    g_path = os.path.join(node_dir, "g.npy")

    if not (os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(g_path)):
        continue

    try:
        y = np.load(y_path, allow_pickle=False)
        g = np.load(g_path, allow_pickle=False)
    except Exception:
        continue

    # Age filter (same as your pipeline)
    ages = np.array([age_map_mimic.get(int(sid), np.nan) for sid in g], dtype=float)
    valid_age_mask = (~np.isnan(ages)) & (ages >= MIN_AGE) & (ages <= MAX_AGE)

    yv = y[valid_age_mask]
    gv = g[valid_age_mask]

    all_pid_H.update(gv[yv == 0])
    all_pid_D.update(gv[yv == 1])

# Compute global overlap
global_overlap = all_pid_H.intersection(all_pid_D)
print(f"[GLOBAL OVERLAP] patients: {len(global_overlap)}")

for node_dir in node_dirs:
    print(f"\nProcessing {node_dir}")
    X_path = os.path.join(node_dir, "X.npy")  # (N, T, C)
    y_path = os.path.join(node_dir, "y.npy")  # 0=healthy, 1=diseased
    g_path = os.path.join(node_dir, "g.npy")  # subject_id

    # Basic existence and size checks
    if not (os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(g_path)):
        print("  → Skipping incomplete node")
        continue
    if os.path.getsize(X_path) == 0 or os.path.getsize(y_path) == 0 or os.path.getsize(g_path) == 0:
        print("  → Skipping node with 0-byte file")
        continue

    try:
        X = np.load(X_path, allow_pickle=False, mmap_mode="r")
        y = np.load(y_path, allow_pickle=False)  # labels 0/1
        g = np.load(g_path, allow_pickle=False)  # subject_ids
    except Exception as e:
        print(f"  → Skipping node due to load error: {type(e).__name__}: {e}")
        continue

    # ============================
    # BEFORE AGE FILTER (RAW COUNTS)
    # ============================
    H_raw = int((y == 0).sum())
    D_raw = int((y == 1).sum())

    Hp_raw = len(np.unique(g[y == 0]))
    Dp_raw = len(np.unique(g[y == 1]))

    print("  --- BEFORE AGE FILTER ---")
    print(f"  Healthy segments   : {H_raw}")
    print(f"  Diseased segments  : {D_raw}")
    print(f"  Healthy patients   : {Hp_raw}")
    print(f"  Diseased patients  : {Dp_raw}")

    # Age lookup (patient-level)
    ages = np.array([age_map_mimic.get(int(sid), np.nan) for sid in g], dtype=float)

    # Age valid mask
    missing_mask = np.isnan(ages)
    valid_age_mask = (~missing_mask) & (ages >= MIN_AGE) & (ages <= MAX_AGE)

    # Filter by valid age
    Xv = np.asarray(X)[valid_age_mask]  # materialize after boolean indexing
    yv = y[valid_age_mask].astype(np.int64)
    gv = g[valid_age_mask].astype(np.int64)
    av = ages[valid_age_mask].astype(np.float32)
    
    # Ensure Xv is (N, T, C)
    if Xv.ndim == 3:
        N, A, B = Xv.shape
        # If channels accidentally first, fix it
        if A <= 32 and B > A:   # (N, C, T)
            Xv = np.transpose(Xv, (0, 2, 1))  # → (N, T, C)

    # Drop non-finite waveforms
    finite_mask = np.isfinite(Xv).all(axis=(1, 2))
    if not finite_mask.all():
        n_bad = int((~finite_mask).sum())
        print(f"  → Dropping {n_bad} segments with NaN/Inf in waveform")
        Xv = Xv[finite_mask]
        yv = yv[finite_mask]
        gv = gv[finite_mask]
        av = av[finite_mask]

    if len(yv) == 0:
        print("  → No valid segments after filtering; skipping save")
        continue

    # Sex per segment
    sex_vals = np.array([normalize_sex_mimic(sex_map_mimic.get(int(sid), None)) for sid in gv], dtype=np.int64)

    # -------------------------------------------------
    # GLOBAL OVERLAP REMOVAL (healthy only)
    # -------------------------------------------------
    if len(global_overlap) > 0:
        ov = np.array(list(global_overlap))
        keep_mask = ~((yv == 0) & np.isin(gv, ov))

        Xv = Xv[keep_mask]
        yv = yv[keep_mask]
        gv = gv[keep_mask]
        av = av[keep_mask]
        sex_vals = sex_vals[keep_mask]

    # Split indices AFTER cleaning
    h_idx = np.where(yv == 0)[0]
    d_idx = np.where(yv == 1)[0]

    assert Xv.shape[1] > Xv.shape[2], f"Unexpected shape {Xv.shape}, expected (N, T, C)"

    # Save healthy chunks (atomic)
    if len(h_idx) > 0:
        safe_np_save(os.path.join(OUT_HEALTHY_mimic, f"X_healthy_{healthy_chunk_id}.npy"), Xv[h_idx])
        safe_np_save(os.path.join(OUT_HEALTHY_mimic, f"y_healthy_{healthy_chunk_id}.npy"), av[h_idx])
        safe_np_save(os.path.join(OUT_HEALTHY_mimic, f"pid_healthy_{healthy_chunk_id}.npy"), gv[h_idx])
        safe_np_save(os.path.join(OUT_HEALTHY_mimic, f"sex_healthy_{healthy_chunk_id}.npy"), sex_vals[h_idx])
        healthy_chunk_id += 1

    # Save diseased chunks (atomic)
    if len(d_idx) > 0:
        safe_np_save(os.path.join(OUT_DISEASED_mimic, f"X_diseased_{diseased_chunk_id}.npy"), Xv[d_idx])
        safe_np_save(os.path.join(OUT_DISEASED_mimic, f"y_diseased_{diseased_chunk_id}.npy"), av[d_idx])
        safe_np_save(os.path.join(OUT_DISEASED_mimic, f"pid_diseased_{diseased_chunk_id}.npy"), gv[d_idx])
        safe_np_save(os.path.join(OUT_DISEASED_mimic, f"sex_diseased_{diseased_chunk_id}.npy"), sex_vals[d_idx])
        diseased_chunk_id += 1

    # Free loop vars early
    del X, y, g, ages, Xv, yv, gv, av, finite_mask, sex_vals

# Reload metadata for audit
yh_mimic_np = load_npy_concat(os.path.join(OUT_HEALTHY_mimic, "y_healthy_*.npy"), require_ndim=1, name="yh")
yd_mimic_np = load_npy_concat(os.path.join(OUT_DISEASED_mimic, "y_diseased_*.npy"), require_ndim=1, name="yd")
pid_h_np    = load_npy_concat(os.path.join(OUT_HEALTHY_mimic, "pid_healthy_*.npy"), require_ndim=1, name="pid_h")
pid_d_np    = load_npy_concat(os.path.join(OUT_DISEASED_mimic, "pid_diseased_*.npy"), require_ndim=1, name="pid_d")

for name_arr, arr in [
    ("yh_mimic_np", yh_mimic_np),
    ("yd_mimic_np", yd_mimic_np),
    ("pid_h_np", pid_h_np),
    ("pid_d_np", pid_d_np),
]:
    if arr is None:
        raise RuntimeError(f"[FATAL] Missing MIMIC array: {name_arr}")

ages_all = np.concatenate([yh_mimic_np, yd_mimic_np])
pids_all = np.concatenate([pid_h_np, pid_d_np])
labels_all = np.concatenate([
    np.zeros_like(yh_mimic_np, dtype=np.int64),
    np.ones_like(yd_mimic_np, dtype=np.int64)
])

_ = audit_age_and_overlap_metadata(
    "MIMIC-IV",
    ages_all,
    pids_all,
    labels_all
)

# Load sex arrays
sex_h_np = load_npy_concat(
    os.path.join(OUT_HEALTHY_mimic, "sex_healthy_*.npy"),
    require_ndim=1,
    name="sex_h"
)

sex_d_np = load_npy_concat(
    os.path.join(OUT_DISEASED_mimic, "sex_diseased_*.npy"),
    require_ndim=1,
    name="sex_d"
)


# QC Plots
m, f = sex_counts_by_bin(yh_mimic_np, sex_h_np)
plot_sex_ratio( m, f, "MIMIC-IV Healthy — Sex by Age", "mimic_healthy_sex_age.pdf")
m, f = sex_counts_by_bin(yd_mimic_np, sex_d_np)
plot_sex_ratio(m, f, "MIMIC-IV Diseased — Sex by Age", "mimic_diseased_sex_age.pdf")

# ============================================================
# DATASETS (STREAMING + IN-MEMORY)
# ============================================================
class MIMICNPYDataset(Dataset):
    """
    Streaming dataset over chunked MIMIC .npy files.
    Each sample: dict with X (C,T), age, pid, sex, label.
    """
    def __init__(self, base_dir, split: str):
        assert split in ("healthy", "diseased")
        self.base_dir = base_dir
        self.split = split

        if split == "healthy":
            X_glob   = os.path.join(base_dir, "healthy", "X_healthy_*.npy")
            y_glob   = os.path.join(base_dir, "healthy", "y_healthy_*.npy")
            pid_glob = os.path.join(base_dir, "healthy", "pid_healthy_*.npy")
            sex_glob = os.path.join(base_dir, "healthy", "sex_healthy_*.npy")
            self.label_value = 0
        else:
            X_glob   = os.path.join(base_dir, "diseased", "X_diseased_*.npy")
            y_glob   = os.path.join(base_dir, "diseased", "y_diseased_*.npy")
            pid_glob = os.path.join(base_dir, "diseased", "pid_diseased_*.npy")
            sex_glob = os.path.join(base_dir, "diseased", "sex_diseased_*.npy")
            self.label_value = 1

        self.X_files   = sorted(glob.glob(X_glob))
        self.y_files   = sorted(glob.glob(y_glob))
        self.pid_files = sorted(glob.glob(pid_glob))
        self.sex_files = sorted(glob.glob(sex_glob))

        # =================================================
        # Memory-mapped / preloaded tensors
        # =================================================
        self.X_memmaps   = [torch.from_numpy(np.load(f, mmap_mode="r")) for f in self.X_files]
        self.y_memmaps   = [torch.from_numpy(np.load(f, mmap_mode="r")) for f in self.y_files]
        self.pid_memmaps = [torch.from_numpy(np.load(f, mmap_mode="r")) for f in self.pid_files]
        self.sex_memmaps = [torch.from_numpy(np.load(f, mmap_mode="r")) for f in self.sex_files]

        # Build global index: (file_idx, local_idx)
        self.index = []
        for fi, x_path in enumerate(self.X_files):
            x_mm = np.load(x_path, mmap_mode="r")
            n = x_mm.shape[0]
            self.index.extend([(fi, i) for i in range(n)])
            del x_mm

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, li = self.index[idx]

        X   = self.X_memmaps[fi][li]
        age = self.y_memmaps[fi][li]
        pid = self.pid_memmaps[fi][li]
        sex = self.sex_memmaps[fi][li]

        X = ensure_channel_first_torch(X.unsqueeze(0)).squeeze(0)
        age = age.float()
        label = torch.tensor(self.label_value, dtype=torch.long)

        return {"X": X, "age": age, "pid": pid, "sex": sex, "label": label}

class InMemoryECGDataset(Dataset):
    """
    Simple in-memory dataset for PTB-XL and EchoNext tensors.
    """
    def __init__(self, Xw, age, pid, sex, label_value):
        self.Xw = Xw
        self.age = age
        self.pid = pid
        self.sex = sex
        self.label_value = label_value

    def __len__(self):
        return self.Xw.shape[0]

    def __getitem__(self, idx):
        X = self.Xw[idx]          # (C,T)
        age = self.age[idx]
        pid = self.pid[idx]
        sex = self.sex[idx]
        label = torch.tensor(self.label_value, dtype=torch.long)
        return {"X": X, "age": age, "pid": pid, "sex": sex, "label": label}

# Build datasets
ptbxl_healthy_ds = InMemoryECGDataset(Xw_h_torch_ptbxl, yh_age_torch_ptbxl, pid_h_torch_ptbxl, sex_h_torch_ptbxl, label_value=0)
ptbxl_diseased_ds = InMemoryECGDataset(Xw_d_torch_ptbxl, yd_age_torch_ptbxl, pid_d_torch_ptbxl, sex_d_torch_ptbxl, label_value=1)

echonext_healthy_ds = InMemoryECGDataset(Xw_h_torch_echonext, yh_age_torch_echonext, pid_h_torch_echonext, sex_h_torch_echonext, label_value=0)
echonext_diseased_ds = InMemoryECGDataset(Xw_d_torch_echonext, yd_age_torch_echonext, pid_d_torch_echonext, sex_d_torch_echonext, label_value=1)

mimic_healthy_ds = MIMICNPYDataset(OUT_DIR_mimic, split="healthy")
mimic_diseased_ds = MIMICNPYDataset(OUT_DIR_mimic, split="diseased")

# Combined dataset (all sources)
train_ds = ConcatDataset([ptbxl_healthy_ds, ptbxl_diseased_ds, echonext_healthy_ds, echonext_diseased_ds, mimic_healthy_ds, mimic_diseased_ds])

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=35,
    pin_memory=True,
    persistent_workers=True,
)

# ============================================================
# SANITY SUMMARY
# ============================================================
def summary(name, Xh, Xd, yh, yd, sh, sd):
    print(f"\n--- {name} ---")
    print(f"Xw_h shape: {tuple(Xh.shape)} | Xw_d shape: {tuple(Xd.shape)}")
    print(f"Age range H: {float(yh.min()):.1f}–{float(yh.max()):.1f} | D: {float(yd.min()):.1f}–{float(yd.max()):.1f}")
    print(f"Sex (H) counts -> male:{int((sh==0).sum())} female:{int((sh==1).sum())} unknown:{int((sh<0).sum())}")
    print(f"Sex (D) counts -> male:{int((sd==0).sum())} female:{int((sd==1).sum())} unknown:{int((sd<0).sum())}")

summary("PTB-XL", Xw_h_torch_ptbxl, Xw_d_torch_ptbxl, yh_age_torch_ptbxl, yd_age_torch_ptbxl, sex_h_torch_ptbxl, sex_d_torch_ptbxl)
summary("EchoNext", Xw_h_torch_echonext, Xw_d_torch_echonext, yh_age_torch_echonext, yd_age_torch_echonext, sex_h_torch_echonext, sex_d_torch_echonext)
print(f"\nMIMIC-IV streaming dataset sizes: H={len(mimic_healthy_ds)}, D={len(mimic_diseased_ds)}")

# ============================================================
# TRAINING UTILITIES (YOU CAN REUSE YOUR EXISTING MODEL CODE)
# ============================================================
class NanGuard(Callback):
    "Stops training and reports where NaNs/Infs appear."
    def before_batch(self):
        xb = self.xb[0]
        yb = self.yb[0]
        if not torch.isfinite(xb).all():
            bad = xb[~torch.isfinite(xb)]
            raise RuntimeError(f"Non-finite values in X batch. Example: {bad[:5]}")
        if not torch.isfinite(yb).all():
            bad = yb[~torch.isfinite(yb)]
            raise RuntimeError(f"Non-finite values in y batch. Example: {bad[:5]}")
    def after_pred(self):
        pred = self.pred
        if not torch.isfinite(pred).all():
            bad = pred[~torch.isfinite(pred)]
            raise RuntimeError(f"Non-finite values in predictions. Example: {bad[:5]}")
    def after_loss(self):
        if not torch.isfinite(self.loss):
            raise RuntimeError(f"Loss became non-finite: {self.loss}")

def mse_flat(pred, targ):
    return F.mse_loss(pred.flatten(), targ.flatten())

# =====================================================================
# ======================= DATA WRAPPERS (SAFE) ========================
# =====================================================================

class AgeRegWrapper(Dataset):
    """Wraps any dataset returning dict → tuple (X, age)."""
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        d = self.base[idx]
        return d["X"], d["age"]


def build_concat_from_sources(sources, split):
    """
    Returns:
        ConcatDataset (streaming-safe)
        list of component datasets
        list of names matching datasets
    """
    subdatasets = []
    names = []

    if "PTBXL" in sources:
        subdatasets.append(ptbxl_healthy_ds if split == "healthy" else ptbxl_diseased_ds)
        names.append("PTBXL")

    if "ECHONEXT" in sources:
        subdatasets.append(echonext_healthy_ds if split == "healthy" else echonext_diseased_ds)
        names.append("ECHONEXT")

    if "MIMIC" in sources:
        subdatasets.append(mimic_healthy_ds if split == "healthy" else mimic_diseased_ds)
        names.append("MIMIC")

    return ConcatDataset(subdatasets), subdatasets, names


def get_num_channels_from_any(subdatasets):
    """Look at a single sample to infer #channels (no memory spike)."""
    for ds in subdatasets:
        if len(ds) > 0:
            return int(ds[0]["X"].shape[0])
    raise RuntimeError("No samples available to infer channels.")


# =====================================================================
# ============= PATIENT → AGE TABLE FOR STRATIFICATION ================
# =====================================================================

def patient_agg_from_concat(subdatasets, agg="min"):
    """
    Build patient→age aggregate mapping using metadata only.
    Supports 'min', 'mean', 'median' aggregation.
    """
    agg = agg.lower()
    patient_age = {}

    # --- In-memory datasets (PTBXL, EchoNext) ---
    for ds in subdatasets:
        if isinstance(ds, InMemoryECGDataset) and len(ds) > 0:
            df = pd.DataFrame({
                "pid": ds.pid.cpu().numpy(),
                "age": ds.age.cpu().numpy()
            })

            if agg == "min":
                s = df.groupby("pid")["age"].min()
            elif agg == "mean":
                s = df.groupby("pid")["age"].mean()
            else:
                s = df.groupby("pid")["age"].median()

            for pid, val in s.items():
                if pid not in patient_age:
                    patient_age[pid] = float(val)
                else:
                    if agg == "min":
                        patient_age[pid] = min(patient_age[pid], float(val))
                    else:
                        patient_age[pid] = float(val)

    # --- MIMIC streaming datasets ---
    for ds in subdatasets:
        if isinstance(ds, MIMICNPYDataset) and len(ds) > 0:
            for fi in range(len(ds.pid_files)):
                pid_arr = np.load(ds.pid_files[fi], mmap_mode="r")
                age_arr = np.load(ds.y_files[fi], mmap_mode="r")
                df = pd.DataFrame({"pid": pid_arr, "age": age_arr})

                if agg == "min":
                    s = df.groupby("pid")["age"].min()
                elif agg == "mean":
                    s = df.groupby("pid")["age"].mean()
                else:
                    s = df.groupby("pid")["age"].median()

                for pid, val in s.items():
                    if pid not in patient_age:
                        patient_age[pid] = float(val)
                    else:
                        if agg == "min":
                            patient_age[pid] = min(patient_age[pid], float(val))
                        else:
                            patient_age[pid] = float(val)

    return patient_age


# =====================================================================
# ============= ASSIGN INDICES TO TRAIN/VAL WITHOUT LEAKAGE ===========
# =====================================================================

def iter_indices_for_patients(concat_ds, subdatasets, train_patients, val_patients):
    """
    Builds train/val segment index lists without ever collecting full arrays.
    Ensures *no patient leakage*.
    """
    tr_inds, va_inds = [], []
    offset = 0

    for sub in subdatasets:
        n = len(sub)

        if isinstance(sub, InMemoryECGDataset):
            pids = sub.pid.cpu().numpy()

            tr_local = np.nonzero(np.isin(pids, list(train_patients)))[0]
            va_local = np.nonzero(np.isin(pids, list(val_patients)))[0]

            tr_inds.extend((offset + tr_local).tolist())
            va_inds.extend((offset + va_local).tolist())

        elif isinstance(sub, MIMICNPYDataset):
            pos = 0
            for fi, li in sub.index:
                pid = int(np.load(sub.pid_files[fi], mmap_mode="r")[li])
                if pid in train_patients:
                    tr_inds.append(offset + pos)
                elif pid in val_patients:
                    va_inds.append(offset + pos)
                pos += 1

        offset += n

    return tr_inds, va_inds

# =====================================================================
# ============================= METRICS ================================
# =====================================================================

def regression_metrics(y_true, y_pred):
    return (
        mean_absolute_error(y_true, y_pred),
        np.mean(y_pred - y_true),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        r2_score(y_true, y_pred)
    )


def summarize_cv(df, metric_cols=None, n_folds=None):
    if metric_cols is None:
        metric_cols = df.select_dtypes(include=[np.number]).columns.drop("fold")
    if n_folds is None:
        n_folds = df["fold"].nunique()

    mean = df[metric_cols].mean()
    sd = df[metric_cols].std(ddof=1)
    se = sd / np.sqrt(n_folds)

    return (
        pd.DataFrame({"metric": metric_cols, "mean": mean, "sd": sd, "se": se}),
        n_folds
    )


# =====================================================================
# ============================= PLOTTING ===============================
# =====================================================================

def plot_segment_mean_age(df, model_name, dataset_name=None, save_dir="./plots", CI=None):
    os.makedirs(save_dir, exist_ok=True)

    dfp = df if dataset_name is None else df[df["dataset"] == dataset_name]
    if len(dfp) == 0:
        return

    x = dfp["age"].astype(float).values
    y = dfp["y_pred_mean"].astype(float).values

    xmin, xmax = min(x.min(), y.min()), max(x.max(), y.max())

    plt.figure(figsize=(6,6))
    plt.scatter(x, y, alpha=0.3, s=8, edgecolor="none")
    plt.plot([xmin,xmax], [xmin,xmax], "r--", lw=2)

    # Fit
    a, b = np.polyfit(x, y, 1)
    xline = np.linspace(xmin, xmax, 200)
    yline = a*xline + b
    plt.plot(xline, yline, "k", lw=2)

    # R2
    yfit = a*x + b
    ss_res = ((y - yfit)**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    plt.text(0.04, 0.95, f"$R^2={r2:.3f}$", transform=plt.gca().transAxes)

    # Optional CI band
    if CI is not None and len(x) > 2:
        dof = len(x) - 2
        s2 = ss_res / dof
        s = np.sqrt(s2)
        xbar = x.mean()
        Sxx = ((x - xbar)**2).sum()
        alpha = 1 - CI/100
        tcrit = student_t.ppf(1 - alpha/2, df=dof)
        se = s * np.sqrt((1/len(x)) + ((xline - xbar)**2)/Sxx)
        plt.fill_between(xline, yline - tcrit*se, yline + tcrit*se, alpha=0.2)

    plt.xlim(xmin, xmax)
    plt.ylim(xmin, xmax)
    plt.gca().set_aspect("equal")
    plt.xlabel("Chronological Age")
    plt.ylabel("Mean Predicted Age")
    suffix = dataset_name or "ALL"
    plt.title(f"{model_name} — {suffix}")

    out = f"{model_name}_scatter_{suffix}_{CI or 'noCI'}.png"
    plt.savefig(os.path.join(save_dir, out), dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# ====================== FULL CV PIPELINE ====================
# ============================================================

def run_full_experiment(
    train_sources,
    model_fn,
    n_splits=10,
    test_size=0.10,
    epochs=EPOCHS,
    bs=BATCH_SIZE,
    test_bs=TEST_BS,
    results_dir="./results",
    run_name="experiment",
):
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(results_dir, f"{run_name}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # ==========================================================
    # 1️⃣ BUILD HEALTHY + DISEASED DATASETS
    # ==========================================================
    healthy_concat, healthy_subs, _ = build_concat_from_sources(train_sources, "healthy")
    diseased_concat, diseased_subs, _ = build_concat_from_sources(train_sources, "diseased")

    full_subs = healthy_subs + diseased_subs
    full_concat = ConcatDataset(full_subs)

    # ==========================================================
    # 2️⃣ PATIENT → AGE TABLE
    # ==========================================================
    healthy_age = patient_agg_from_concat(healthy_subs, agg="mean")
    diseased_age = patient_agg_from_concat(diseased_subs, agg="mean")

    df_h = pd.DataFrame({"patient_id": list(healthy_age.keys()),
                         "age": list(healthy_age.values())})
    df_d = pd.DataFrame({"patient_id": list(diseased_age.keys()),
                         "age": list(diseased_age.values())})

    df_h["age_bin"] = np.digitize(df_h["age"], AGE_BINS) - 1
    df_d["age_bin"] = np.digitize(df_d["age"], AGE_BINS) - 1

    # ==========================================================
    # 3️⃣ STRATIFIED HOLD-OUT SPLITS (SEPARATE)
    # ==========================================================
    def stratified_split(df):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
        dummy = np.zeros(len(df))
        tr_idx, ho_idx = next(sss.split(dummy, df["age_bin"]))
        return (
            set(df.iloc[tr_idx]["patient_id"]),
            set(df.iloc[ho_idx]["patient_id"])
        )

    healthy_train_pats, healthy_holdout_pats = stratified_split(df_h)
    diseased_train_pats, diseased_holdout_pats = stratified_split(df_d)

    train_patients = healthy_train_pats.union(diseased_train_pats)

    # Convert to segment indices
    train_inds, _ = iter_indices_for_patients(full_concat, full_subs, train_patients, set())
    healthy_hold_inds, _ = iter_indices_for_patients(
        healthy_concat, healthy_subs, healthy_holdout_pats, set()
    )
    diseased_hold_inds, _ = iter_indices_for_patients(
        diseased_concat, diseased_subs, diseased_holdout_pats, set()
    )

    train_concat = Subset(full_concat, train_inds)
    healthy_holdout = Subset(healthy_concat, healthy_hold_inds)
    diseased_holdout = Subset(diseased_concat, diseased_hold_inds)

    print(f"Train segments: {len(train_concat)}")
    print(f"Healthy hold-out segments: {len(healthy_holdout)}")
    print(f"Diseased hold-out segments: {len(diseased_holdout)}")

    # ==========================================================
    # 4️⃣ 10-FOLD CV
    # ==========================================================
    df_train_pat = pd.concat([
        df_h[df_h["patient_id"].isin(healthy_train_pats)],
        df_d[df_d["patient_id"].isin(diseased_train_pats)]
    ]).reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    fold_models = []
    cv_rows = []

    dummy = np.zeros(len(df_train_pat))

    for fold_idx, (tr_idx, va_idx) in enumerate(
        skf.split(dummy, df_train_pat["age_bin"])
    ):
        print(f"\n==== Fold {fold_idx} ====")

        tr_pats = set(df_train_pat.iloc[tr_idx]["patient_id"])
        va_pats = set(df_train_pat.iloc[va_idx]["patient_id"])

        tr_inds, va_inds = iter_indices_for_patients(
            full_concat, full_subs, tr_pats, va_pats
        )

        tr_ds = AgeRegWrapper(Subset(full_concat, tr_inds))
        va_ds = AgeRegWrapper(Subset(full_concat, va_inds))

        dls = DataLoaders.from_dsets(
            tr_ds, va_ds,
            bs=bs,
            shuffle=True,
            num_workers=35,
            pin_memory=(DEVICE=="cuda")
        ).to(DEVICE)

        model = model_fn().to(DEVICE)

        #learn = Learner(dls, model, loss_func=mse_flat, cbs=[NanGuard(), GradientClip(0.25)])
        learn = Learner(
            dls,
            model,
            loss_func=mse_flat,
            metrics=[mae],
            cbs=[NanGuard(), GradientClip(0.25), SaveModelCallback(monitor='valid_loss', comp=np.less), EarlyStoppingCallback(monitor='valid_loss', patience=5)]
        )

        if DEVICE == "cuda":
            learn = learn.to_fp16()

        learn.fit_one_cycle(epochs, LR)

        preds, targs = learn.get_preds(dl=dls.valid)

        val_metrics = regression_metrics(
            targs.cpu().numpy().flatten(),
            preds.cpu().numpy().flatten()
        )

        cv_rows.append({
            "fold": fold_idx,
            "val_mae": val_metrics[0],
            "val_bias": val_metrics[1],
            "val_rmse": val_metrics[2],
            "val_r2": val_metrics[3],
        })

        fold_models.append(learn.model.eval())

        del learn
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    df_cv = pd.DataFrame(cv_rows)
    df_cv.to_csv(os.path.join(out_dir, "cv_fold_metrics.csv"), index=False)

    # ==========================================================
    # 5️⃣ HOLD-OUT PREDICTIONS (ALL FOLDS)
    # ==========================================================
    def predict_stack(ds):
        dl = DataLoader(AgeRegWrapper(ds), batch_size=test_bs, shuffle=False)
        stack = []

        with torch.no_grad():
            for model in fold_models:
                model.to(DEVICE)
                preds_fold = []
                for xb, _ in dl:
                    xb = xb.to(DEVICE)
                    preds_fold.append(model(xb).cpu().numpy().flatten())
                stack.append(np.concatenate(preds_fold))

        return np.vstack(stack)

    healthy_stack = predict_stack(healthy_holdout)
    diseased_stack = predict_stack(diseased_holdout)

    healthy_true = np.array([d["age"].item() for d in healthy_holdout])
    diseased_true = np.array([d["age"].item() for d in diseased_holdout])

    # Fold-wise hold-out metrics
    healthy_fold_metrics = [
        regression_metrics(healthy_true, healthy_stack[i])
        for i in range(n_splits)
    ]

    diseased_fold_metrics = [
        regression_metrics(diseased_true, diseased_stack[i])
        for i in range(n_splits)
    ]

    # SAVE HOLD-OUT CV METRICS
    healthy_rows = []
    diseased_rows = []

    for i in range(n_splits):
        h_mae, h_bias, h_rmse, h_r2 = healthy_fold_metrics[i]
        d_mae, d_bias, d_rmse, d_r2 = diseased_fold_metrics[i]

        healthy_rows.append({
            "fold": i,
            "MAE": h_mae,
            "BIAS": h_bias,
            "RMSE": h_rmse,
            "R2": h_r2
        })

        diseased_rows.append({
            "fold": i,
            "MAE": d_mae,
            "BIAS": d_bias,
            "RMSE": d_rmse,
            "R2": d_r2
        })

    df_healthy_cv = pd.DataFrame(healthy_rows)
    df_diseased_cv = pd.DataFrame(diseased_rows)

    df_healthy_cv.to_csv(os.path.join(out_dir, "healthy_holdout_cv_metrics.csv"), index=False)
    df_diseased_cv.to_csv(os.path.join(out_dir, "diseased_holdout_cv_metrics.csv"), index=False)

    # ==========================================================
    # 6️⃣ ENSEMBLE METRICS
    # ==========================================================
    healthy_ens = healthy_stack.mean(axis=0)
    diseased_ens = diseased_stack.mean(axis=0)

    healthy_metrics = regression_metrics(healthy_true, healthy_ens)
    diseased_metrics = regression_metrics(diseased_true, diseased_ens)

    # Save ensemble metrics
    pd.DataFrame([healthy_metrics],
        columns=["MAE","BIAS","RMSE","R2"]
    ).to_csv(os.path.join(out_dir, "healthy_holdout_ensemble.csv"), index=False)

    pd.DataFrame([diseased_metrics],
        columns=["MAE","BIAS","RMSE","R2"]
    ).to_csv(os.path.join(out_dir, "diseased_holdout_ensemble.csv"), index=False)

    print("\nHealthy Ensemble:", healthy_metrics)
    print("Diseased Ensemble:", diseased_metrics)

    return {
        "cv_results": df_cv,
        "healthy_holdout_cv": df_healthy_cv,
        "diseased_holdout_cv": df_diseased_cv,
        "healthy_holdout_metrics": healthy_metrics,
        "diseased_holdout_metrics": diseased_metrics,
        "healthy_true": healthy_true,
        "diseased_true": diseased_true,
        "healthy_ensemble": healthy_ens,
        "diseased_ensemble": diseased_ens
    }


# =====================================================================
# ================================ RUN ================================
# =====================================================================
# MODEL ZOO
MODELS = {
    "XResNet1d50_Deeper": lambda: xresnet1d50_deeper(input_channels=NUM_CHANNELS, num_classes=NUM_OUTPUT),
}

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA version:", torch.version.cuda)
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

    out = run_full_experiment(
        train_sources=["PTBXL" , "ECHONEXT", "MIMIC"],
        model_fn=MODELS["XResNet1d50_Deeper"],
        run_name="XResNet1d50_Deeper_Combined"
    )

    print("\nCV Results:")
    print(out["cv_results"])

    print("\nHealthy Hold-Out Metrics:")
    print(out["healthy_holdout_metrics"])

    print("\nDiseased Hold-Out Metrics:")
    print(out["diseased_holdout_metrics"])

    # Plot for Healthy hold-out
    df_healthy = pd.DataFrame({
        "age": out["healthy_true"],
        "y_pred_mean": out["healthy_ensemble"],
        "dataset": "Healthy"
    })

    os.makedirs("./plots/healthy", exist_ok=True)
    os.makedirs("./plots/diseased", exist_ok=True)

    plot_segment_mean_age(df_healthy, model_name="XResNet1d50_Deeper", dataset_name=None, save_dir="./plots/healthy", CI=None)
    plot_segment_mean_age(df_healthy, model_name="XResNet1d50_Deeper", dataset_name=None, save_dir="./plots/healthy", CI=95)

    # Plot for Healthy hold-out
    df_diseased = pd.DataFrame({
        "age": out["diseased_true"],
        "y_pred_mean": out["diseased_ensemble"],
        "dataset": "Diseased"
    })

    plot_segment_mean_age(df_diseased, model_name="XResNet1d50_Deeper", dataset_name=None, save_dir="./plots/diseased", CI=None)
    plot_segment_mean_age(df_diseased, model_name="XResNet1d50_Deeper", dataset_name=None, save_dir="./plots/diseased", CI=95)

