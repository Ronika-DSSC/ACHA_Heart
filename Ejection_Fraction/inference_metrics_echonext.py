import numpy as np
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# Import your exact EchoNext model code
sys.path.append("/data/BCS/Services/Ronika_De/ImageDS/EchoNext/Ejection_Fraction/IntroECG-master/7-EchoNext Minimodel/cradlenet/models")
from resnet1d_tabular import ResNet1dWithTabular

# -------------------------------
# Load waveform data and labels
# -------------------------------
Xw_healthy = np.load("/data/BCS/Services/David_Leone/ACHA-2025/raw_data/EchoNext/upsampled_cache/Xw_healthy.npy")
Xw_diseased = np.load("/data/BCS/Services/David_Leone/ACHA-2025/raw_data/EchoNext/upsampled_cache/Xw_diseased.npy")
pid_healthy = np.load("/data/BCS/Services/David_Leone/ACHA-2025/raw_data/EchoNext/upsampled_cache/pid_healthy.npy")
pid_diseased = np.load("/data/BCS/Services/David_Leone/ACHA-2025/raw_data/EchoNext/upsampled_cache/pid_diseased.npy")

X = np.concatenate([Xw_healthy, Xw_diseased]) # X.shape = (samples, 5000, 12)
# Before converting to tensors, permute the waveform
X = np.transpose(X, (0, 2, 1))  # now X.shape = (samples, 12, 5000)

pids = np.concatenate([pid_healthy, pid_diseased])

# Load metadata for LVEF
import pandas as pd
meta = pd.read_csv("/data/BCS/Services/Ronika_De/ImageDS/EchoNext/physionet.org/files/echonext/1.1.0/echonext_metadata_100k.csv")
meta["lvef_lte_45"] = (meta["lvef_value"] <= 45).astype(int)
label_dict = dict(zip(meta["patient_key"], meta["lvef_lte_45"]))
y = np.array([label_dict[pid] for pid in pids])

# -------------------------------
# Device
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# 10-Fold Stratified CV
# -------------------------------
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
aucs, precisions, recalls, f1s = [], [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n=== Fold {fold+1} ===")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    # -------------------------------
    # Create empty tabular features
    # -------------------------------
    X_tab_train = torch.zeros((X_train.shape[0], 0), dtype=torch.float32).to(device)
    X_tab_val = torch.zeros((X_val.shape[0], 0), dtype=torch.float32).to(device)

    # -------------------------------
    # DataLoader
    # -------------------------------
    train_loader = DataLoader(TensorDataset(X_train, X_tab_train, y_train), batch_size=128, shuffle=True)

    # -------------------------------
    # Initialize model (exact EchoNext)
    # -------------------------------
    model = ResNet1dWithTabular(
        len_tabular_feature_vector=0,  # zero because no tabular features
        input_channels=X_train.shape[1],
        num_classes=1
    ).to(device)

    # -------------------------------
    # Class imbalance handling
    # -------------------------------
    pos_weight = torch.tensor([(len(y_train)-y_train.sum())/y_train.sum()]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # -------------------------------
    # Early stopping params
    # -------------------------------
    best_auc = 0
    patience = 3
    trigger_times = 0
    best_model_state = None
    max_epochs = 50

    # -------------------------------
    # Training loop
    # -------------------------------
    for epoch in range(max_epochs):
        model.train()
        for xb, tab, yb in train_loader:
            optimizer.zero_grad()
            outputs = model((xb, tab))  # Pass waveform + empty tabular
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model((X_val, X_tab_val))
            probs = torch.sigmoid(logits).cpu().numpy()
        y_true = y_val.cpu().numpy()

        auc = roc_auc_score(y_true, probs)

        if auc > best_auc:
            best_auc = auc
            trigger_times = 0
            best_model_state = model.state_dict()
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}, best AUROC={best_auc:.4f}")
                break

    # Load best model
    model.load_state_dict(best_model_state)

    # -------------------------------
    # Metrics
    # -------------------------------
    with torch.no_grad():
        logits = model((X_val, X_tab_val))
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)

    precision = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)
    f1 = f1_score(y_true, preds)

    print(f"Fold {fold+1} - AUROC: {best_auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    aucs.append(best_auc)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

# -------------------------------
# CV Summary
# -------------------------------
print("\n=== 10-Fold CV Summary ===")
print(f"Mean AUROC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"Mean Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Mean Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")