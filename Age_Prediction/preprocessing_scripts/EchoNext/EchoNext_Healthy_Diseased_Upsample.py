################################################################################
#                               IMPORTS                                        #
################################################################################
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils import shuffle
from scipy.signal import butter, filtfilt, medfilt, resample, resample_poly
from tqdm import tqdm

################################################################################
#                               PATHS                                          #
################################################################################
DATA_DIR = "/data/BCS/Services/David_Leone/ACHA-2025/raw_data/EchoNext/"
CACHE_DIR = os.path.join(DATA_DIR, "upsampled_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

################################################################################
#                               UP-SAMPLING                                    #
################################################################################
# FFT‑Based Resampling
def upsample_to_500hz(ecg_250hz):
    n_samples_250 = ecg_250hz.shape[-1]
    n_samples_500 = n_samples_250 * 2  # because 500 Hz is 2× 250 Hz
    return resample(ecg_250hz, n_samples_500, axis=-1)

# Polyphase‑Based Resampling
def upsamplePoly_to_500hz(ecg_250hz):
    return resample_poly(ecg_250hz, up=2, down=1, axis=0)

################################################################################
#                         ECG PREPROCESSING FUNCTIONS                           #
################################################################################
def butter_bandpass_filter(data, lowcut=0.5, highcut=40, fs=500, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data, axis=0)

def remove_baseline_wander(sig, kernel_size=501):
    baseline = medfilt(sig, kernel_size=kernel_size)
    return sig - baseline

def global_zscore(signal):
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0) + 1e-8
    return (signal - mean) / std

def preprocess_ecg(signal, fs=500):
    signal = butter_bandpass_filter(signal, 0.5, 40, fs)
    signal = remove_baseline_wander(signal)
    signal = global_zscore(signal)
    return signal.astype(np.float32)

################################################################################
#                         SEGMENTATION FUNCTION                                 #
################################################################################
def segment_ecg(signal, seg_len=5000):
    n = signal.shape[0]
    n_seg = int(np.ceil(n / seg_len))
    segments = np.zeros((n_seg, seg_len, signal.shape[1]), dtype=np.float32)

    for i in range(n_seg):
        start = i * seg_len
        end = min(start + seg_len, n)
        segments[i, : end - start] = signal[start:end]

    return segments

################################################################################
#                         LOAD & PREPROCESS SPLIT                               #
################################################################################
def load_and_preprocess_split(split, n_jobs=28):
    print(f"\n===== Processing {split.upper()} =====")

    wave_file = os.path.join(DATA_DIR, f"EchoNext_{split}_waveforms.npy")
    tab_file  = os.path.join(DATA_DIR, f"EchoNext_{split}_tabular_features.npy")
    meta_file = os.path.join(DATA_DIR, "echonext_metadata_100k.csv")

    #X_wave = np.load(wave_file, mmap_mode="r")[:, 0, :, :] #####
    X_wave = np.load(wave_file, mmap_mode="r")[:, 0]; print(X_wave.shape);
    # Ensure shape is (N, T, 12)
    if X_wave.shape[-1] != 12 and X_wave.shape[-2] == 12:
        X_wave = np.transpose(X_wave, (0, 2, 1))

    # Final safety check
    assert X_wave.shape[-1] == 12, f"Expected 12 leads, got {X_wave.shape}"
    
    X_tab  = np.load(tab_file)
    meta   = pd.read_csv(meta_file).iloc[: len(X_wave)]

    y = meta["shd_moderate_or_greater_flag"].astype(int).values
    patient_ids = meta["patient_key"].astype(int).values

    cache_file = os.path.join(CACHE_DIR, f"{split}_wave_preprocessed.npy")

    if os.path.exists(cache_file):
        print(" Loading cached preprocessed ECGs")
        X_wave_proc = np.load(cache_file, mmap_mode="r")
    else:
        print(" Upsampling ECGs (parallel CPU)...")
        X_wave_proc = np.array(
            Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(upsamplePoly_to_500hz)(sig)
                for sig in tqdm(X_wave)
            ),
            dtype=np.float32
        )
        
        np.save(cache_file, X_wave_proc)

    return X_wave_proc, X_tab, y, patient_ids

################################################################################
#                         LOAD ALL SPLITS                                      #
################################################################################
Xw_train, Xt_train, y_train, pid_train = load_and_preprocess_split("train")
Xw_val,   Xt_val,   y_val,   pid_val   = load_and_preprocess_split("val")
Xw_test,  Xt_test,  y_test,  pid_test  = load_and_preprocess_split("test")

################################################################################
#                         COMBINE ALL SPLITS                                   #
################################################################################
X_wave_all = np.concatenate([Xw_train, Xw_val, Xw_test])
X_tab_all  = np.concatenate([Xt_train, Xt_val, Xt_test])
y_all      = np.concatenate([y_train, y_val, y_test])
patient_all = np.concatenate([pid_train, pid_val, pid_test])

print("\nCombined ECGs:", X_wave_all.shape)

################################################################################
#                         SPLIT HEALTHY / DISEASED                             #
################################################################################
healthy = y_all == 0
diseased = y_all == 1

Xw_h, Xt_h, y_h, pid_h = shuffle(
    X_wave_all[healthy],
    X_tab_all[healthy],
    y_all[healthy],
    patient_all[healthy],
    random_state=42
)

Xw_d, Xt_d, y_d, pid_d = shuffle(
    X_wave_all[diseased],
    X_tab_all[diseased],
    y_all[diseased],
    patient_all[diseased],
    random_state=42
)

################################################################################
#                               SUMMARY                                        #
################################################################################
print("\n===== FINAL SUMMARY =====")
print("Healthy samples:", len(y_h))
print("Diseased samples:", len(y_d))
print("Unique healthy patients:", len(np.unique(pid_h)))
print("Unique diseased patients:", len(np.unique(pid_d)))

################################################################################
#                               SAVE FINAL DATA                                #
################################################################################
np.save(os.path.join(CACHE_DIR, "Xw_healthy.npy"), Xw_h)
np.save(os.path.join(CACHE_DIR, "Xt_healthy.npy"), Xt_h)
np.save(os.path.join(CACHE_DIR, "y_healthy.npy"), y_h)
np.save(os.path.join(CACHE_DIR, "pid_healthy.npy"), pid_h)

np.save(os.path.join(CACHE_DIR, "Xw_diseased.npy"), Xw_d)
np.save(os.path.join(CACHE_DIR, "Xt_diseased.npy"), Xt_d)
np.save(os.path.join(CACHE_DIR, "y_diseased.npy"), y_d)
np.save(os.path.join(CACHE_DIR, "pid_diseased.npy"), pid_d)

print("\n Preprocessing complete and cached.")


