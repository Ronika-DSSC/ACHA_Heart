import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score
from resnet1d_waveform import ResNet1dWaveform

# -------------------------------
# Load your waveform data
# -------------------------------
Xw_healthy = np.load("/data/BCS/Services/David_Leone/ACHA-2025/raw_data/EchoNext/upsampled_cache/Xw_healthy.npy")
Xw_diseased = np.load("/data/BCS/Services/David_Leone/ACHA-2025/raw_data/EchoNext/upsampled_cache/Xw_diseased.npy")
pid_healthy = np.load("/data/BCS/Services/David_Leone/ACHA-2025/raw_data/EchoNext/upsampled_cache/pid_healthy.npy")
pid_diseased = np.load("/data/BCS/Services/David_Leone/ACHA-2025/raw_data/EchoNext/upsampled_cache/pid_diseased.npy")

# Merge
X = np.concatenate([Xw_healthy, Xw_diseased])
print(X.shape)  # (samples, 5000, 12)
pids = np.concatenate([pid_healthy, pid_diseased])

# Permute to [samples, channels, timesteps] for ResNet1dWaveform
X = np.transpose(X, (0, 2, 1))  # now X.shape = (samples, 12, 5000)

# Load LVEF metadata and binarize EF <=45%
meta = pd.read_csv("/data/BCS/Services/Ronika_De/ImageDS/EchoNext/physionet.org/files/echonext/1.1.0/echonext_metadata_100k.csv")
meta["lvef_lte_45"] = (meta["lvef_value"] <= 45).astype(int)
label_dict = dict(zip(meta["patient_key"], meta["lvef_lte_45"]))
y = np.array([label_dict[pid] for pid in pids])

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 10-Fold CV
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Store metrics
aucs, precisions, recalls, f1s = [], [], [], []

# -------------------------------
# 10-Fold Loop
# -------------------------------
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n=== Fold {fold+1} ===")

    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=128,
        shuffle=True
    )

    # Initialize model
    model = ResNet1dWaveform().to(device)

    # Handle class imbalance
    pos_weight = torch.tensor([(len(y_train)-y_train.sum())/y_train.sum()]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Early stopping parameters
    best_auc = 0
    patience = 3
    trigger_times = 0
    best_model_state = None
    max_epochs = 50

    # -------------------------------
    # Training Loop with Early Stopping
    # -------------------------------
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(X_val)
            probs = torch.sigmoid(logits).cpu().numpy()
        y_true = y_val.cpu().numpy()

        # Compute AUROC for early stopping
        auc = roc_auc_score(y_true, probs)

        if auc > best_auc:
            best_auc = auc
            trigger_times = 0
            best_model_state = model.state_dict()  # save best weights
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}, best AUROC={best_auc:.4f}")
                break

    # Load best model weights
    model.load_state_dict(best_model_state)

    # -------------------------------
    # Metrics for this fold
    # -------------------------------
    with torch.no_grad():
        logits = model(X_val)
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
    # ROC curve for this fold
    # -------------------------------
    fpr, tpr, thresholds = roc_curve(y_true, probs)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'Fold {fold+1} AUROC = {best_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold+1}')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Save figure
    plt.savefig(f'auroc_fold_{fold+1}.png')
    plt.close()

# -------------------------------
# Summary across all folds
# -------------------------------
print("\n=== 10-Fold CV Summary ===")
print(f"Mean AUROC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"Mean Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Mean Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")