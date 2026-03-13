################################################################################
#                                IMPORTS                                       #
################################################################################
import os
import ast
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, medfilt
from joblib import Parallel, delayed
from tqdm import tqdm

################################################################################
#                            HPC CONFIG                                        #
################################################################################
NODE_ID = int(os.environ.get("LSB_JOBINDEX", 1)) - 1
NUM_NODES = 35
N_JOBS = 28                      # must match BSUB -n
SEG_LEN = 5000
FS = 500

OUT_DIR = f"/data/BCS/Services/Ronika_De/ImageDS/MIMIC_IV/segments_fast/node_{NODE_ID}"
os.makedirs(OUT_DIR, exist_ok=True)

################################################################################
#                            ECG UTILITIES                                     #
################################################################################
def load_raw_data(record_path):
    """Load WFDB ECG signal."""
    sig, _ = wfdb.rdsamp(record_path)
    return sig.astype(np.float32)

def bandpass_filter(sig, fs=FS):
    """Bandpass filter to remove noise and baseline components."""
    b, a = butter(4, [0.5 / (fs / 2), 40 / (fs / 2)], btype="band")
    return filtfilt(b, a, sig, axis=0)

def remove_baseline_wander(sig, kernel_size=501):
    """
    Baseline wander removal using median filtering.
    NOTE: This is computationally expensive but clinically standard.
    """
    baseline = medfilt(sig, kernel_size=(kernel_size, 1))
    return sig - baseline

def global_zscore(sig):
    return (sig - sig.mean(axis=0)) / (sig.std(axis=0) + 1e-8)

def segment_ecg(sig, seg_len=SEG_LEN):
    """Non-overlapping segmentation with zero-padding."""
    n = sig.shape[0]
    pad = (-n) % seg_len
    if pad:
        sig = np.pad(sig, ((0, pad), (0, 0)))
    return sig.reshape(-1, seg_len, sig.shape[1]).astype(np.float32)

def preprocess_one(record_path):
    """
    Full preprocessing for one ECG:
    load → bandpass → baseline removal → z-score → segment
    """
    try:
        sig = load_raw_data(record_path)
        sig = bandpass_filter(sig)
        sig = remove_baseline_wander(sig)   # <<< baseline wander INCLUDED
        sig = global_zscore(sig)
        return segment_ecg(sig)
    except Exception:
        return None

################################################################################
#                        LOAD & PREPARE METADATA                                #
################################################################################
CSV_PATH = (
    "/data/BCS/Services/Ronika_De/ImageDS/MIMIC_IV/"
    "physionet.org/files/mimic-iv-ecg/1.0/records_w_diag_icd10.csv"
)
BASE_PATH = (
    "/data/BCS/Services/Ronika_De/ImageDS/MIMIC_IV/"
    "physionet.org/files/mimic-iv-ecg/1.0"
)

df = pd.read_csv(CSV_PATH)

# Parse ICD codes safely
df["all_diag_all"] = df["all_diag_all"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)

# Cardiac disease label (ICD-10 I*)
df["label"] = df["all_diag_all"].apply(
    lambda codes: int(any(c.startswith("I") for c in codes))
)

def build_record_path(file_name):
    rel = file_name.split("files/")[1]
    return os.path.join(BASE_PATH, "files", rel)

df["record_path"] = df["file_name"].apply(build_record_path)

################################################################################
#                    SPLIT WORK ACROSS ARRAY JOBS                               #
################################################################################
chunks = np.array_split(df, NUM_NODES)
df_node = chunks[NODE_ID].reset_index(drop=True)

print(f"Node {NODE_ID}: processing {len(df_node)} ECG records")

################################################################################
#                      PARALLEL PREPROCESSING                                   #
################################################################################
segments = []
labels = []
subjects = []

results = Parallel(
    n_jobs=N_JOBS,
    prefer="processes",
    backend="loky"
)(
    delayed(preprocess_one)(p)
    for p in tqdm(df_node["record_path"], desc=f"Node {NODE_ID}")
)

for segs, label, subj in zip(
    results,
    df_node["label"].values,
    df_node["subject_id"].values,
):
    if segs is None:
        continue
    n = len(segs)
    segments.append(segs)
    labels.append(np.full(n, label, dtype=np.uint8))
    subjects.append(np.full(n, subj, dtype=np.uint32))

################################################################################
#                         SAVE OUTPUT (ONCE)                                    #
################################################################################
if len(segments) == 0:
    raise RuntimeError(f"Node {NODE_ID}: no valid ECGs processed")

X = np.concatenate(segments, axis=0)
y = np.concatenate(labels, axis=0)
g = np.concatenate(subjects, axis=0)

np.save(os.path.join(OUT_DIR, "X.npy"), X)
np.save(os.path.join(OUT_DIR, "y.npy"), y)
np.save(os.path.join(OUT_DIR, "g.npy"), g)

################################################################################
#                         SUMMARY                                               #
################################################################################
n_healthy = np.sum(y == 0)
n_diseased = np.sum(y == 1)
u_healthy = len(np.unique(g[y == 0]))
u_diseased = len(np.unique(g[y == 1]))

print("\n===== NODE SUMMARY =====")
print(f"Total segments: {len(y)}")
print(f"Healthy segments: {n_healthy}")
print(f"Diseased segments: {n_diseased}")
print(f"Unique healthy patients: {u_healthy}")
print(f"Unique diseased patients: {u_diseased}")
print("========================")
