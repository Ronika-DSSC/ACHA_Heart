import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import wfdb
import ast
import os
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt, medfilt, resample

################################################################################
#                               ECG UTILITIES                                  #
################################################################################

def load_raw_data(path):
    sig, _ = wfdb.rdsamp(path)
    return sig.astype(np.float32)


def butter_bandpass_filter(data, lowcut=0.5, highcut=40, fs=500, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data, axis=0)


def remove_baseline_wander(sig):
    baseline = medfilt(sig, kernel_size=501)
    return sig - baseline


def segment_ecg(signal, seg_len=5000):
    """Segment long signals WITHOUT truncation."""
    segs = []
    n = signal.shape[0]

    if n <= seg_len:
        pad = np.pad(signal, ((0, seg_len-n),(0,0)))
        segs.append(pad)
        return np.array(segs, dtype=np.float32)

    for start in range(0, n, seg_len):
        seg = signal[start:start+seg_len]
        if seg.shape[0] < seg_len:
            seg = np.pad(seg, ((0, seg_len-seg.shape[0]),(0,0)))
        segs.append(seg)

    return np.array(segs, dtype=np.float32)


def global_zscore(signal):
    """Compute per-lead z-score using FULL signal, BEFORE segmentation."""
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0) + 1e-8
    return (signal - mean) / std


def minmax_scale(signal):
    """Per-lead min-max scaling to [0,1]"""
    min_val = np.min(signal, axis=0)
    max_val = np.max(signal, axis=0)
    return (signal - min_val) / (max_val - min_val + 1e-8)


def preprocess_ecg(signal, fs=500):
    """
    FINAL CORRECT ORDER:
    1) filter
    2) baseline removal
    3) global z-score
    4) segment long recording
    """
    signal = butter_bandpass_filter(signal, 0.5, 40, fs)
    signal = remove_baseline_wander(signal)
    signal = global_zscore(signal)
    #signal = minmax_scale(signal)
    return segment_ecg(signal, seg_len=5000)


################################################################################
#                                  DATA LOAD                                    #
################################################################################

path = '/data/BCS/Services/David_Leone/ACHA-2025/raw_data/PTB_XL/PTB/physionet.org/files/ptb-xl/1.0.3/'
sampling_rate=500 # 500Hz records

Y = pd.read_csv(path + "ptbxl_database.csv", index_col="ecg_id")
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

files = [path + f for f in Y.filename_hr]
ages = Y["age"].astype(np.float32).values

# Healthy/diseased classification
agg_df = pd.read_csv(path + "scp_statements.csv", index_col=0)
agg_df = agg_df[agg_df.diagnostic==1]

def aggr(dic):
    out = []
    for k in dic:
        if k in agg_df.index:
            out.append(agg_df.loc[k].diagnostic_class)
    return list(set(out))

Y["diagnostic_superclass"] = Y.scp_codes.apply(aggr); print(Y["diagnostic_superclass"].head(50));
Y["is_healthy"] = Y["diagnostic_superclass"].apply(lambda x: 1 if x==['NORM'] else 0)
Y["is_diseased"] = Y["diagnostic_superclass"].apply(lambda x: 1 if (x!=['NORM'] and x!=[]) else 0)
# Healthy = is_healthy == 1
num_healthy = (Y["is_healthy"] == 1).sum()
# Diseased = is_diseased == 1
num_diseased = (Y["is_diseased"] == 1).sum()
print("Healthy patients:", num_healthy)
print("Diseased patients:", num_diseased)
#patient_ids = Y["patient_id"].astype(int).values


################################################################################
#                          PREPROCESS ALL RECORDINGS                            #
################################################################################

def preprocess_wrapper(path, age, healthy_flag, diseased_flag):
    sig = load_raw_data(path)
    segments = preprocess_ecg(sig, fs=500)
    return segments, age, healthy_flag, diseased_flag

# n_jobs=-1
results = Parallel(n_jobs=28, prefer="processes")(
    delayed(preprocess_wrapper)(f, age, healthy, diseased)
    for f, age, healthy, diseased in zip(files, ages, Y.is_healthy.values, Y.is_diseased.values)
)

# Flatten segment-level arrays
X_segments, y_segments, is_healthy_seg, is_diseased_seg = [], [], [], []
seg_counts = []

for segments, age, healthy_flag, diseased_flag in results:
    X_segments.append(segments)
    seg_len = len(segments)
    y_segments.append(np.repeat(age, seg_len))
    is_healthy_seg.append(np.repeat(healthy_flag, seg_len))
    is_diseased_seg.append(np.repeat(diseased_flag, seg_len))
    seg_counts.append(seg_len)

X = np.concatenate(X_segments)
y = np.concatenate(y_segments)
is_healthy_seg = np.concatenate(is_healthy_seg)
is_diseased_seg = np.concatenate(is_diseased_seg)

# Build group vector (patient-level grouping for each segment)
patient_ids = Y["patient_id"].values
groups = np.concatenate([
    np.repeat(pid, count) for pid, count in zip(patient_ids, seg_counts)
])

assert len(groups) == len(X), "Group assignment mismatch!"

print("Segment-level healthy:", is_healthy_seg.sum())
print("Segment-level diseased:", is_diseased_seg.sum())

cache_path = "/data/BCS/Services/David_Leone/ACHA-2025/raw_data/PTB_XL/"
cache_dir = os.path.join(cache_path, "cache")
os.makedirs(cache_dir, exist_ok=True)

cached_file = os.path.join(cache_dir, "ptbxl_ecg_segments.npz")

np.savez_compressed(
    cached_file,
    X=X.astype(np.float32),                     # (N, 5000, 12)
    y=y.astype(np.float32),                     # (N,)
    patient_ids=groups.astype(np.int32),        # (N,)
    is_healthy=is_healthy_seg.astype(np.int8),
    is_diseased=is_diseased_seg.astype(np.int8),
    fs=sampling_rate,
    seg_len=5000,
    channels=12
)

print("Preprocessed & Segmented NumPy dataset saved:", cached_file)

# For Pytorch models, modify shape of X.
X_pt = torch.tensor(X, dtype=torch.float32)       # X.shape = (N, 5000, 12)
X_pt = X_pt.permute(0, 2, 1)                      # now (N, 12, 5000)

def ecg_preprocess(X, y=None, fs=500):
    """
    X: list/array of raw ECGs, shape (n_samples, T, C)
    y: optional labels
    Returns:
        X_proc: (N, C, T)
        y_proc: (N,)
    """
    X_segments, y_segments = [], []

    for i, x in enumerate(X):
        x = x.astype(np.float32)
        # Apply ECG preprocessing + segmentation
        segs = preprocess_ecg(x, fs=fs)      # (n_seg, T, C)
        # Convert to PyTorch layout
        segs = np.transpose(segs, (0, 2, 1)) # (n_seg, C, T)
        X_segments.append(segs)
        if y is not None:
            y_segments.append(np.repeat(y[i], len(segs)))

    X_proc = np.concatenate(X_segments, axis=0)

    if y is not None:
        y_proc = np.concatenate(y_segments, axis=0).astype(np.float32)
        return X_proc, y_proc

    return X_proc

# Create cache folder
os.makedirs("cache", exist_ok=True)

# Wrapper to preprocess a single ECG file using ecg_preprocess
def preprocess_wrapper_with_ecg_preprocess(f, age, healthy, diseased):
    # Load raw ECG
    sig = load_raw_data(f)
    # ecg_preprocess expects a list/array of ECGs
    X_proc, y_proc = ecg_preprocess([sig], y=[age], fs=500)  # returns (n_seg, C, T), (n_seg,)
    n_seg = X_proc.shape[0]
    healthy_seg = np.repeat(healthy, n_seg)
    diseased_seg = np.repeat(diseased, n_seg)
    return X_proc, y_proc, healthy_seg, diseased_seg, n_seg

# Run preprocessing in parallel
results = Parallel(n_jobs=28, prefer="processes")(
    delayed(preprocess_wrapper_with_ecg_preprocess)(f, a, h, d)
    for f, a, h, d in zip(files, ages, healthy_flag, diseased_flag)
)

# Flatten all segments
X_segments, y_segments, healthy_segments, diseased_segments, seg_counts = [], [], [], [], []
for X_proc, y_proc, h_seg, d_seg, n_seg in results:
    X_segments.append(X_proc)
    y_segments.append(y_proc)
    healthy_segments.append(h_seg)
    diseased_segments.append(d_seg)
    seg_counts.append(n_seg)

# Concatenate into arrays
X_all = np.concatenate(X_segments, axis=0)         # (N_segments, C, T)
y_all = np.concatenate(y_segments, axis=0)
is_healthy_all = np.concatenate(healthy_segments)
is_diseased_all = np.concatenate(diseased_segments)
groups_all = np.concatenate([np.repeat(pid, c) for pid, c in zip(patient_ids, seg_counts)])

# Save as PyTorch tensors
torch.save({
    "X": torch.tensor(X_all, dtype=torch.float32),         # preprocessed & segmented
    "y": torch.tensor(y_all, dtype=torch.float32),
    "patient_ids": torch.tensor(groups_all, dtype=torch.int32),
    "is_healthy": torch.tensor(is_healthy_all, dtype=torch.int8),
    "is_diseased": torch.tensor(is_diseased_all, dtype=torch.int8)
}, "cache/ptbxl_ecg_segments_preprocessed.pt")

print("Preprocessed & segmented ECG data saved in PyTorch format (shape: N_segments, C, T)!")