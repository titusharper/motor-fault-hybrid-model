import os
import copy
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from motor_fault.dataset_fusion import FusionDataset
from motor_fault.model_fusion_film import FusionFiLMModel
from motor_fault.train_utils import EarlyStopping

csv_path = REPO_ROOT / "data/processed/all_segments_1d_features_new.csv"
img_root = REPO_ROOT / "data/processed/reassigned_new_1p5_4p5s"
REPO_ROOT = Path(__file__).resolve().parents[1]

labels = ['healthy', 'imbalance', 'loose', 'misalignment']
BATCH_SIZE = 32
EPOCHS = 200
NUM_CLASSES = len(labels)
FEATURE_DIM = 19

df = pd.read_csv(csv_path)
feature_cols = [c for c in df.columns if c.startswith('feat_')]
expected_splits = {'train','validation','test'}
actual_splits = set(df['split'].unique())
missing = expected_splits - actual_splits
if missing:
    raise ValueError(f"Missing sets in 'split' column: {missing}.")

means = df[df['split']=='train'][feature_cols].mean()
stds  = df[df['split']=='train'][feature_cols].std().replace(0, 1e-6)
df[feature_cols] = (df[feature_cols] - means) / stds

train_dataset = FusionDataset(df, img_root, split="train", label_list=labels, feature_cols=feature_cols, augment=True)
val_dataset   = FusionDataset(df, img_root, split="validation", label_list=labels, feature_cols=feature_cols, augment=False)
test_dataset  = FusionDataset(df, img_root, split="test", label_list=labels, feature_cols=feature_cols, augment=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionFiLMModel(num_numeric_features=FEATURE_DIM, num_classes=NUM_CLASSES, freeze_backbone=False).to(device)
def set_bn_momentum(model, momentum=0.01):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.momentum = momentum
set_bn_momentum(model, 0.01)
# Parameter groups
backbone_params = list(model.backbone.parameters())
backbone_param_ids = {id(p) for p in backbone_params}
head_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]
optimizer = torch.optim.AdamW(
    [
        {"params": backbone_params, "lr": 5e-5, "weight_decay": 1e-6},
        {"params": head_params,     "lr": 1e-4, "weight_decay": 1e-5},
    ]
)

from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
MAX_NORM = 1.0
NUMERIC_L2 = 1e-4  
# Warm-up
WARMUP_EPOCHS = 3
base_lrs = [pg["lr"] for pg in optimizer.param_groups]
def warmup(epoch: int):
    if epoch <= WARMUP_EPOCHS:
        scale = epoch / float(WARMUP_EPOCHS)
        for pg, base in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = base * scale

early = EarlyStopping(patience=20, min_delta=0.0)

train_acc_hist, val_acc_hist = [], []

def set_freeze(module, freeze=True):
    for p in module.parameters():
        p.requires_grad = not freeze

UNFREEZE_EPOCH = 1 
for epoch in range(1, EPOCHS+1):
    if epoch == UNFREEZE_EPOCH:
        set_freeze(model.backbone, freeze=False)

    # Warmup
    warmup(epoch)
    # ---- Train ----
    model.train()
    tr_loss, tr_correct, tr_total = 0.0, 0, 0
    for X_img, X_feat, y in train_loader:
        X_img, X_feat, y = X_img.to(device), X_feat.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X_img, X_feat)
        loss = criterion(outputs, y)
        numeric_l2 = (model.fc1.weight.pow(2).sum() + model.fc2.weight.pow(2).sum())
        loss = loss + NUMERIC_L2 * numeric_l2
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
        optimizer.step()
        _, preds = outputs.max(1)
        tr_loss += loss.item() * y.size(0)
        tr_total += y.size(0)
        tr_correct += (preds == y).sum().item()
      
    train_acc = 100.0 * tr_correct / max(1, tr_total)
    avg_tr_loss = tr_loss / max(1, tr_total)

    # ---- Validation ----
    model.eval()
    va_loss, va_correct, va_total = 0.0, 0, 0
    with torch.no_grad():
        for X_img, X_feat, y in val_loader:
            X_img, X_feat, y = X_img.to(device), X_feat.to(device), y.to(device)
            outputs = model(X_img, X_feat)
            loss = criterion(outputs, y) 
            _, preds = outputs.max(1)
            va_loss += loss.item() * y.size(0)
            va_total += y.size(0)
            va_correct += (preds == y).sum().item()

    val_acc = 100.0 * va_correct / max(1, va_total)
    avg_va_loss = va_loss / max(1, va_total)
    train_acc_hist.append(train_acc)
    val_acc_hist.append(val_acc)

    print(f"Epoch {epoch:02d} | TrainLoss={avg_tr_loss:.4f} Acc={train_acc:.2f}% | ValLoss={avg_va_loss:.4f} Acc={val_acc:.2f}%")
    scheduler.step()

    if early.step(avg_va_loss, model):
        print(f"[EarlyStop] No improvement for {early.patience} checks. Stop at epoch {epoch}.")
        break

# Best weights
if early.best_state is not None:
    model.load_state_dict(early.best_state)
  
torch.save(model.state_dict(), "best_hybrid_film.pth")
print("Best model saved to best_hybrid_film.pth")

model.eval()
test_correct, test_total = 0, 0
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for X_img, X_feat, y in test_loader:
        X_img, X_feat, y = X_img.to(device), X_feat.to(device), y.to(device)
        outputs = model(X_img, X_feat)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
        test_total += y.size(0)
        test_correct += (preds == y).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

all_probs = np.vstack(all_probs)   # (N, num_classes)
test_acc = 100.0 * test_correct / max(1, test_total)
print(f"\n[TEST] Accuracy = {test_acc:.2f}%")
print("\n--- Classification Report (Test Set) ---")
print(classification_report(all_labels, all_preds, target_names=labels))

# Confusion Matrix (count)
plt.figure(figsize=(6,6))
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Reds', values_format='d')
plt.title('Confusion Matrix (Test)')
plt.show()

# Confusion Matrix (normalized)
plt.figure(figsize=(6,6))
cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
disp.plot(cmap="Blues", values_format=".2f", ax=plt.gca(), colorbar=True)
plt.title("Normalized Confusion Matrix")
plt.show()

# Learning curves
plt.figure(figsize=(7,5))
plt.plot(train_acc_hist, label="Train Acc")
plt.plot(val_acc_hist, label="Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
plt.title("Train vs Validation Accuracy")
plt.legend(); plt.grid(True); plt.show()

# ROC–AUC (one-vs-rest)
y_true = np.array(all_labels)
y_score = np.array(all_probs)   # (N, NUM_CLASSES)
y_bin = label_binarize(y_true, classes=range(NUM_CLASSES))

plt.figure(figsize=(7,6))
for i, label in enumerate(labels):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.2f})")
plt.plot([0,1],[0,1],'k--',lw=1)
plt.xlim([0.0,1.0]); plt.ylim([0.0,1.05])
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves (One-vs-Rest)")
plt.legend(loc="lower right"); plt.grid(True); plt.show()
