import os
import glob
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = "/data/BCS/Services/David_Leone/ACHA-2025/raw_data/MIMIC_IV/segments_fast"

node_dirs = sorted(glob.glob(os.path.join(BASE_DIR, "node_*")))
assert len(node_dirs) > 0, "No node directories found!"

Xs, ys, gs = [], [], []

print(f"Loading data from {len(node_dirs)} nodes")

for d in node_dirs:
    X_path = os.path.join(d, "X.npy")
    y_path = os.path.join(d, "y.npy")
    g_path = os.path.join(d, "g.npy")

    if not (os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(g_path)):
        print(f"Skipping incomplete node dir: {d}")
        continue

    Xs.append(np.load(X_path, mmap_mode="r"))
    ys.append(np.load(y_path))
    gs.append(np.load(g_path))

X_all = np.concatenate(Xs, axis=0)
y_all = np.concatenate(ys, axis=0)
groups = np.concatenate(gs, axis=0)

print("Total segments:", len(y_all))

# -----------------------
# Torch formatting
# -----------------------
X_pt = torch.from_numpy(X_all).permute(0, 2, 1)  # (N, C, T)
y_pt = torch.from_numpy(y_all)

# -----------------------
# Patient-level split
# -----------------------
healthy_idx = np.where(y_all == 0)[0]
diseased_idx = np.where(y_all == 1)[0]

# -----------------------
# Summary statistics AFTER segmentation
# -----------------------
n_healthy_segments = len(healthy_idx)
n_diseased_segments = len(diseased_idx)

healthy_patients = np.unique(groups[healthy_idx])
diseased_patients = np.unique(groups[diseased_idx])

print("\n===== DATASET SUMMARY (AFTER SEGMENTATION) =====")
print(f"Healthy segments:   {n_healthy_segments}")
print(f"Diseased segments:  {n_diseased_segments}")
print(f"Unique healthy patients:  {len(healthy_patients)}")
print(f"Unique diseased patients: {len(diseased_patients)}")
print("===============================================\n")

X_healthy = X_pt[healthy_idx]
y_healthy = y_pt[healthy_idx]
groups_healthy = groups[healthy_idx]

gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, val_idx = next(
    gss.split(X_healthy, y_healthy, groups_healthy)
)

X_train = X_healthy[train_idx]
X_val   = X_healthy[val_idx]
y_train = y_healthy[train_idx]
y_val   = y_healthy[val_idx]

X_test = X_pt[diseased_idx]
y_test = y_pt[diseased_idx]

np.save("X_train.npy", X_train.numpy())
np.save("y_train.npy", y_train.numpy())
np.save("X_val.npy", X_val.numpy())
np.save("y_val.npy", y_val.numpy())
np.save("X_test.npy", X_test.numpy())
np.save("y_test.npy", y_test.numpy())

print("Merge complete.")


