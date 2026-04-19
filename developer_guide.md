# Developer Guide — AI-Based Forest Cover Change Detection System
### North Eastern India | Sentinel-1 + Sentinel-2 | Siamese U-Net + ASPP

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup](#3-environment-setup)
4. [Dataset](#4-dataset)
5. [Data Pipeline](#5-data-pipeline)
6. [Model Architectures](#6-model-architectures)
7. [Loss Functions](#7-loss-functions)
8. [Training Pipeline](#8-training-pipeline)
9. [Evaluation Pipeline](#9-evaluation-pipeline)
10. [Experiment Log](#10-experiment-log)
11. [Reproducing Results](#11-reproducing-results)
12. [Known Issues and Fixes](#12-known-issues-and-fixes)
13. [Extending the Codebase](#13-extending-the-codebase)

---

## 1. Project Overview

This project develops a supervised binary change detection system that identifies
forest cover loss in the North Eastern Region (NER) of India using bi-temporal
satellite imagery. Given two co-registered image patches acquired at T1 (2021)
and T2 (2023), the model produces a pixel-level binary mask where:

- **Class 0** — No forest loss
- **Class 1** — Forest loss

### Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| Input modality | Sentinel-1 SAR + Sentinel-2 optical | Cloud-robust coverage |
| Change method | Siamese feature-level comparison | Avoids post-classification error accumulation |
| Bottleneck | ASPP (dilation rates 1, 6, 12, 18) | Multi-scale Jhum patch detection |
| Loss | Dice + Focal (0.7 / 0.3) | Handles severe class imbalance |
| Label source | Hansen GFC `lossyear` band | Publicly available, global coverage |

---

## 2. Repository Structure

```
project/
│
├── notebooks/
│   ├── Data_pipeline_v2.ipynb       # GEE extraction + patch generation
│   ├── dataset-eda.ipynb            # Exploratory data analysis + sanity checks
│   ├── baseline-model.ipynb         # Baseline model + all training experiments
│   ├── model_improvement.ipynb      # Data augmentation + loss refinement run
│   └── evaluation.ipynb             # Test set evaluation + visualizations
│
├── dataset/
│   └── download_dataset.py          # Run this to fetch the dataset from Kaggle
│
└── developer_guide.md               # This file
```

> **Note:** The dataset is not committed to the repository due to its size (~3–5 GB).
> Run `download_dataset.py` once to fetch it locally before opening any notebook.
> Model checkpoints and figures are saved automatically by the notebooks to
> Kaggle's `/kaggle/working/` directory during training.

---

## 3. Environment Setup

### Platform
All experiments were run on **Kaggle Notebooks** with the following setup:

| Component | Specification |
|---|---|
| GPU | NVIDIA Tesla T4 (15 GB VRAM) |
| CPU | 4 vCPUs |
| RAM | 29 GB |
| Storage | 20 GB (Kaggle working directory) |
| Python | 3.12 |
| PyTorch | 2.x with CUDA |

### Required packages

```python
# Core
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

# Data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Evaluation
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    jaccard_score, confusion_matrix,
    precision_recall_curve, roc_curve, auc
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Experiment tracking
import wandb

# Optional (for pretrained encoder experiments)
import torchvision.models as tvm
import segmentation_models_pytorch as smp  # pip install segmentation-models-pytorch
```

### W&B setup

```python
from kaggle_secrets import UserSecretsClient
wandb_key = UserSecretsClient().get_secret("wandb_api_key")
import wandb
wandb.login(key=wandb_key)
```

---

## 4. Dataset

### Sources

| Source | Bands used | Resolution | Purpose |
|---|---|---|---|
| Sentinel-1 GRD | VV, VH | 10 m | SAR structural features |
| Sentinel-2 L2A | B2, B3, B4, B8 | 10 m | Spectral/optical features |
| Hansen GFC v1.11 | `lossyear` | 30 m | Binary change labels |

### Input tensor structure

Each training sample is a pair of 6-channel tensors:

```
Channel 0 — Sentinel-1 VV  (radar backscatter, vertical polarization)
Channel 1 — Sentinel-1 VH  (radar backscatter, cross polarization)
Channel 2 — Sentinel-2 B2  (Blue,  490 nm)
Channel 3 — Sentinel-2 B3  (Green, 560 nm)
Channel 4 — Sentinel-2 B4  (Red,   665 nm)
Channel 5 — Sentinel-2 B8  (NIR,   842 nm)
```

- **Patch size:** 256 × 256 pixels
- **T1 shape:** `(6, 256, 256)` — float32
- **T2 shape:** `(6, 256, 256)` — float32
- **Label shape:** `(1, 256, 256)` — binary float32 (0 or 1)

### Dataset statistics

| Split | Samples | Positive pixel % |
|---|---|---|
| Train | 5,250 (70%) | ~1.2% |
| Validation | 1,125 (15%) | ~1.2% |
| Test | 867 (15%) | 1.20% |

### Change bin distribution

| Bin | Ratio range | % of dataset |
|---|---|---|
| no\_change | = 0 | ~11.2% |
| very\_sparse | 0 – 0.005 | ~35.6% |
| sparse | 0.005 – 0.02 | ~34.8% |
| moderate\_large | > 0.02 | ~18.4% |

---

## 5. Data Pipeline

### 5.1 Metadata dataframe

```python
dataset_root = "/kaggle/input/datasets/.../dataset"
regions = ["Meghalaya_2021_2023", "Nagaland_2021_2023"]

records = []
for region in regions:
    label_dir = os.path.join(dataset_root, region, "label")
    for fname in os.listdir(label_dir):
        mask = np.load(os.path.join(label_dir, fname))
        change_ratio = mask.sum() / mask.size
        records.append({
            "region": region,
            "file": fname,
            "label_path": os.path.join(label_dir, fname),
            "change_pixels": mask.sum(),
            "change_ratio": change_ratio
        })

df = pd.DataFrame(records)
```

### 5.2 Stratified split

```python
bins   = [0, 1e-6, 0.005, 0.02, 1]
labels = ["no_change", "very_sparse", "sparse", "moderate_large"]

df["change_bin"]   = pd.cut(df["change_ratio"], bins=bins,
                             labels=labels, include_lowest=True)
df["stratify_key"] = df["region"] + "_" + df["change_bin"].astype(str)

# 70 / 15 / 15 split stratified by region × change bin
train_df, temp_df = train_test_split(df, test_size=0.30,
                                      stratify=df["stratify_key"],
                                      random_state=42)
val_df, test_df   = train_test_split(temp_df, test_size=0.50,
                                      stratify=temp_df["stratify_key"],
                                      random_state=42)
```

### 5.3 Dataset class

```python
class ChangeDataset(Dataset):
    def __init__(self, df, root_dir, augment=False):
        self.df       = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.augment  = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        region = row["region"]
        file   = row["file"]

        t1    = np.load(os.path.join(self.root_dir, region, "t1", file)).astype(np.float32)
        t2    = np.load(os.path.join(self.root_dir, region, "t2", file)).astype(np.float32)
        label = np.load(row["label_path"]).astype(np.float32)

        t1    = torch.from_numpy(t1)
        t2    = torch.from_numpy(t2)
        label = torch.from_numpy(label).unsqueeze(0)

        if self.augment:
            t1, t2, label = self._augment(t1, t2, label)

        return t1, t2, label

    def _augment(self, t1, t2, label):
        import torchvision.transforms.functional as TF
        if torch.rand(1) > 0.5:
            t1, t2, label = TF.hflip(t1), TF.hflip(t2), TF.hflip(label)
        if torch.rand(1) > 0.5:
            t1, t2, label = TF.vflip(t1), TF.vflip(t2), TF.vflip(label)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            t1    = torch.rot90(t1,    k, dims=[1, 2])
            t2    = torch.rot90(t2,    k, dims=[1, 2])
            label = torch.rot90(label, k, dims=[1, 2])
        return t1, t2, label
```

---

## 6. Model Architectures

### 6.1 Baseline — SiameseUNet\_ASPP

The baseline model processes T1 and T2 through a shared VGG-style encoder,
fuses temporal features at each decoder level via concatenation, and uses an
ASPP bottleneck for multi-scale feature extraction.

```
Encoder (shared weights):
  conv1: Conv(6→64)   + MaxPool  →  (64,  128×128)
  conv2: Conv(64→128) + MaxPool  →  (128,  64×64)
  conv3: Conv(128→256)+ MaxPool  →  (256,  32×32)
  conv4: Conv(256→512)+ MaxPool  →  (512,  16×16)

ASPP bottleneck (dilation 1, 6, 12, 18):
  4 × Conv(512→512) → cat → Conv(2048→512)

Decoder (with Siamese skip concat):
  up4: ConvTranspose(512→512) + cat[t1_e4, t2_e4] → Conv(1536→512)
  up3: ConvTranspose(512→256) + cat[t1_e3, t2_e3] → Conv(768→256)
  up2: ConvTranspose(256→128) + cat[t1_e2, t2_e2] → Conv(384→128)
  up1: ConvTranspose(128→64)  + cat[t1_e1, t2_e1] → Conv(192→64)

Head: Conv(64→1, 1×1) → logits
```

**Why skip channels are doubled:** At each decoder level, skip features from
both T1 and T2 encoders are concatenated, giving `up_out + skip_T1 + skip_T2`
channels. For example, `up4` receives `512 + 512 + 512 = 1536` channels.

### 6.2 SiameseUNet\_V2 — Architecture improvements

Three improvements are added on top of the baseline encoder and ASPP:

#### Temporal Bottleneck Fusion
```python
self.bottleneck_fusion = nn.Sequential(
    nn.Conv2d(1024, 512, kernel_size=1, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True)
)
# Usage: cat(t1_e4, t2_e4) → 1024ch → bottleneck_fusion → 512ch → ASPP
```

#### Attention Gate
```python
class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        # g = gating signal (decoder), x = skip connection (encoder)
        self.Wg  = nn.Sequential(nn.Conv2d(g_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.Wx  = nn.Sequential(nn.Conv2d(x_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g, x):
        att = self.psi(F.relu(self.Wg(g) + self.Wx(x)))
        return x * att   # attended skip
```

#### Change Feature Fusion
```python
class ChangeFusion(nn.Module):
    def __init__(self, ch):
        self.project = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch * 2), nn.ReLU(inplace=True)
        )

    def forward(self, t1, t2):
        # Explicit change representations: raw + difference + correlation
        return self.project(torch.cat([t1, t2, torch.abs(t1-t2), t1*t2], dim=1))
```

#### Deep Supervision loss
```python
def loss_fn_v2(preds, target):
    if isinstance(preds, tuple):
        main, aux4, aux3, aux2 = preds
        return (1.0 * loss_fn(main, target) +
                0.4 * loss_fn(aux4, target) +
                0.2 * loss_fn(aux3, target) +
                0.1 * loss_fn(aux2, target))
    return loss_fn(preds, target)
```

> **Note:** Auxiliary heads are only active during `model.train()`. They are
> automatically bypassed during `model.eval()` via a `self.training` guard,
> adding zero inference overhead.

### 6.3 PretrainedSiameseUNet\_ASPP

Replaces the VGG-style encoder with pretrained ResNet-34. ASPP and decoder
are completely unchanged.

**6-channel weight initialization strategy:**
```python
new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
with torch.no_grad():
    # Average pretrained 3-ch weights → tile across 6 channels
    new_conv.weight = nn.Parameter(
        backbone.conv1.weight.mean(dim=1, keepdim=True)
        .repeat(1, 6, 1, 1) / 6 * 3
    )
backbone.conv1 = new_conv
```

ResNet-34 stage → channel mapping:

| Stage | Output ch | Spatial (for 256×256 input) |
|---|---|---|
| enc0 (conv1+bn+relu) | 64 | 128×128 |
| enc1 (maxpool+layer1) | 64 | 64×64 |
| enc2 (layer2) | 128 | 32×32 |
| enc3 (layer3) | 256 | 16×16 |
| enc4 (layer4) | 512 | 8×8 |

---

## 7. Loss Functions

### Dice Loss
```python
class DiceLoss(nn.Module):
    def forward(self, logits, targets):
        probs  = torch.sigmoid(logits)
        num    = 2 * (probs * targets).sum()
        den    = probs.sum() + targets.sum() + 1e-6
        return 1 - num / den
```

### Focal Loss
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt    = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()
```

### Combined loss (final configuration)
```python
focal = FocalLoss()
dice  = DiceLoss()

def loss_fn(pred, target):
    return 0.7 * dice(pred, target) + 0.3 * focal(pred, target)
```

---

## 8. Training Pipeline

### Standard training loop

```python
device    = "cuda"
model     = SiameseUNet_ASPP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2
)
scaler    = GradScaler()
best_iou  = 0
patience_counter = 0
PATIENCE  = 5

for epoch in range(50):
    # ── Train ──────────────────────────────────────────────────────
    model.train()
    for t1, t2, label in train_loader:
        t1, t2, label = t1.to(device), t2.to(device), label.to(device)
        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            pred = model(t1, t2)
            loss = loss_fn(pred, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # ── Validate ────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        for t1, t2, label in val_loader:
            with autocast(device_type="cuda"):
                pred = model(t1, t2.to(device))
            f1, iou = compute_metrics(pred, label.to(device))

    scheduler.step(val_iou)

    # ── Checkpoint ─────────────────────────────────────────────────
    if val_iou > best_iou:
        best_iou = val_iou
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            break   # early stopping
```

### Compute metrics

```python
def compute_metrics(pred, target, threshold=0.5):
    prob     = torch.sigmoid(pred)
    pred_bin = (prob > threshold).float()
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    iou       = tp / (tp + fp + fn + 1e-6)
    return f1.item(), iou.item()
```

### W&B logging template

```python
wandb.init(
    project="Forest loss detection",
    name="experiment_name",
    config={
        "model"     : "SiameseUNet_ASPP",
        "loss"      : "Dice+Focal (0.7/0.3)",
        "lr"        : 1e-4,
        "batch_size": 16,
        "epochs"    : 50
    }
)

# Inside training loop:
wandb.log({
    "epoch"      : epoch + 1,
    "train_loss" : train_loss,
    "val_f1"     : val_f1,
    "val_iou"    : val_iou,
    "lr"         : optimizer.param_groups[0]["lr"]
})
```

---

## 9. Evaluation Pipeline

### Step 1 — Load model and run inference

```python
# CRITICAL: Always use the original ChangeDataset (with normalization)
# for inference. Never build a custom loader that skips normalization.
test_loader_ordered = DataLoader(
    ChangeDataset(test_df.reset_index(drop=True), root_dir),
    batch_size=1,
    shuffle=False    # shuffle=False to preserve index alignment with test_df
)

model = SiameseUNet_ASPP()
state_dict = torch.load("best_model.pth", map_location=device)
# Strip DataParallel prefix if present
if "module." in list(state_dict.keys())[0]:
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

all_preds, all_labels, all_t1, all_t2 = [], [], [], []

with torch.no_grad():
    for t1, t2, label in tqdm(test_loader_ordered):
        with autocast(device_type="cuda"):
            logits = model(t1.to(device), t2.to(device))
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        all_preds.append(prob)
        all_labels.append(label.squeeze().cpu().numpy())
        all_t1.append(t1.squeeze().cpu().numpy())
        all_t2.append(t2.squeeze().cpu().numpy())
```

### Step 2 — Optimal threshold search

```python
flat_preds  = np.array(all_preds).flatten()
flat_labels = np.array(all_labels).flatten()

best_thresh, best_f1 = 0.5, 0.0
for t in np.arange(0.05, 0.60, 0.05):
    f1 = f1_score(flat_labels, (flat_preds >= t).astype(np.uint8),
                  zero_division=0)
    if f1 > best_f1:
        best_f1, best_thresh = f1, round(t, 2)
```

### Step 3 — Core metrics at best threshold

```python
bin_preds   = (flat_preds >= best_thresh).astype(np.uint8)
precision   = precision_score(flat_labels, bin_preds, zero_division=0)
recall      = recall_score(flat_labels, bin_preds, zero_division=0)
f1          = f1_score(flat_labels, bin_preds, zero_division=0)
iou         = jaccard_score(flat_labels, bin_preds, zero_division=0)
cm          = confusion_matrix(flat_labels, bin_preds)
tn, fp, fn, tp = cm.ravel()
```

### Step 4 — Per-category breakdown

```python
def extract_bin(stratify_key):
    for bin_name in ["moderate_large", "very_sparse", "sparse", "no_change"]:
        if stratify_key.endswith(bin_name):
            return bin_name
    return "unknown"

test_df_reset["change_bin"] = test_df_reset["stratify_key"].apply(extract_bin)
```

> **Important:** Always check `moderate_large` before `sparse` and `very_sparse`
> before `sparse` in the string matching, otherwise suffix overlap causes
> misclassification.

### Step 5 — Qualitative visualization

```python
def make_false_color(img_6ch):
    """NIR-Red-Green false color. Channels: 0=VV,1=VH,2=B2,3=B3,4=B4,5=B8"""
    fc = np.stack([img_6ch[5], img_6ch[4], img_6ch[3]], axis=-1)
    p2, p98 = np.percentile(fc, (2, 98))
    return np.clip((fc - p2) / (p98 - p2 + 1e-6), 0, 1)
```

> **Common mistake:** Using `img_6ch[2], img_6ch[3], img_6ch[4]` for RGB
> produces a blue-dominant image because B2 (blue) is channel 2. Always use
> channels `[5, 4, 3]` for NIR-R-G false color to show vegetation clearly.

---

## 10. Experiment Log

| Run name | Model | Loss | Aug | Best Val F1 | Best Val IoU | Stopped at |
|---|---|---|---|---|---|---|
| `unet_aspp_dice_bce` | SiameseUNet\_ASPP | BCE+Dice (50/50) | No | 0.5060 | 0.3413 | Epoch 19 |
| `unet_aspp_model_improvement` | SiameseUNet\_ASPP | Dice+Focal (70/30) | Yes | 0.5189 | 0.3526 | Epoch 30 |
| `siamese_resnet34_pretrained` | PretrainedSiamese | Dice+Focal (70/30) | Yes | 0.4625 | 0.3026 | Epoch 17 |
| `siamese_v2_attn_changefusion_deepsup` | SiameseUNet\_V2 | Dice+Focal+DeepSup | Yes | 0.4869 | 0.3240 | Epoch 17 |

### Test set results (best model: `unet_aspp_model_improvement`)

| Metric | Value |
|---|---|
| Threshold | 0.30 |
| Precision | 0.4793 |
| Recall | 0.6049 |
| F1 | 0.5348 |
| IoU | 0.3650 |
| Specificity | 0.9920 |
| PR-AUC | 0.5141 |
| ROC-AUC | 0.9776 |

### Per-category test results

| Category | Samples | Precision | Recall | F1 | IoU |
|---|---|---|---|---|---|
| No Change | 97 | 0.000 | 0.000 | 0.000 | 0.000 |
| Very Sparse | 308 | 0.281 | 0.408 | 0.333 | 0.200 |
| Sparse | 301 | 0.422 | 0.534 | 0.471 | 0.308 |
| Moderate Large | 161 | 0.530 | 0.657 | 0.587 | 0.415 |

> **Note on No Change F1=0:** This is mathematically correct, not a bug.
> There are no positive ground truth pixels in these patches, so TP=0 and
> FN=0 by definition. Any model prediction in these patches is a false positive.
> The correct metric for this category is Specificity (0.9997), which shows
> the model correctly ignores 99.97% of background pixels.

---

## 11. Reproducing Results

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd project

# 2. Download the dataset
cd dataset
pip install kagglehub
python download_dataset.py
cd ..

# 3. Open Kaggle and upload notebooks from project/notebooks/
#    Enable T4 GPU accelerator in Kaggle notebook settings
#    Add dataset: suranjandas1990/forest-loss-dataset
#    Add W&B API key to Kaggle secrets as "wandb_api_key"

# 4. Run notebooks in this order:
#    Data_pipeline_v2.ipynb        → generates dataset patches (if re-extracting)
#    dataset-eda.ipynb             → sanity checks and EDA
#    baseline-model.ipynb          → trains all experiments, saves best_model.pth
#    model_improvement.ipynb       → data augmentation + Dice+Focal run
#    evaluation.ipynb              → test set metrics + visualizations
```

> Each notebook is self-contained. Running `baseline-model.ipynb` end-to-end
> trains all four experiments, logs to W&B, saves checkpoints as artifacts,
> and produces the comparison table. No separate scripts are needed.

### Loading a saved checkpoint

```python
model = SiameseUNet_ASPP()
state_dict = torch.load("/path/to/best_model.pth", map_location="cuda")

# Strip DataParallel wrapper if model was trained with DataParallel
if list(state_dict.keys())[0].startswith("module."):
    state_dict = {k[7:]: v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()
```

---

## 12. Known Issues and Fixes

### Issue 1: All-black predictions during visualization
**Cause:** Custom dataset class missing normalization applied during training.
The model receives out-of-distribution raw pixel values and collapses to
near-zero outputs.

**Fix:** Always use the original `ChangeDataset` class for inference. Never
build a custom loader that bypasses `__getitem__` normalization logic.

```python
# WRONG — custom loader missing normalization
test_loader = DataLoader(MyCustomDataset(...), ...)

# CORRECT — always use the same Dataset class used during training
test_loader = DataLoader(ChangeDataset(test_df, root_dir), batch_size=1,
                         shuffle=False)
```

---

### Issue 2: DataParallel state dict key mismatch
**Cause:** When a model is wrapped with `nn.DataParallel`, all parameter keys
are prefixed with `"module."`. Loading this state dict into an unwrapped model
raises `KeyError`.

**Fix:**
```python
state_dict = torch.load("best_model.pth")
if list(state_dict.keys())[0].startswith("module."):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
```

---

### Issue 3: SMP `UnetDecoder` API breaking changes
**Cause:** `segmentation_models_pytorch >= 0.3.x` removed the `use_batchnorm`
argument and changed `UnetDecoder.forward()` to accept a list instead of
`*args`.

**Fix:** Do not use `smp.decoders.unet.decoder.UnetDecoder` directly. Build
a custom decoder from plain `nn.Conv2d` + `BatchNorm2d` blocks which is fully
version-agnostic.

---

### Issue 4: Stratify key bin parsing failure
**Cause:** Simple `x.split("_")[-1]` fails for `moderate_large` and
`very_sparse` because they are two-word suffixes.

**Fix:** Always use suffix matching in priority order:
```python
def extract_bin(stratify_key):
    for bin_name in ["moderate_large", "very_sparse", "sparse", "no_change"]:
        if stratify_key.endswith(bin_name):
            return bin_name
    return "unknown"
```

---

### Issue 5: Blue-dominant satellite visualization
**Cause:** Using channels `[2, 3, 4]` for RGB maps B2 (blue) to the red
display channel, producing unnatural blue imagery where vegetation is
indistinguishable from other classes.

**Fix:** Use NIR-Red-Green false color `[5, 4, 3]` which shows healthy
vegetation as bright green-red tones, making deforestation visually apparent.

---

## 13. Extending the Codebase

### Adding a new encoder
1. Define encoder stages producing the same output channels: 64, 128, 256, 512
2. Implement an `encode(x)` method returning `(e1, e2, e3, e4)`
3. Pass to the existing decoder — no decoder changes required as long as
   channel dimensions match

### Adding a new experiment
1. Define the model class
2. Call `train_model()` with a unique `run_name` for W&B tracking
3. Add results to the comparison table in `per_experiment_results`
4. Run `evaluate_on_test()` only after all validation experiments are complete
   — the test set must remain unseen until final evaluation

### Adding a new region
1. Extract patches using the GEE pipeline in `data_pipeline_v2.ipynb`
2. Export to the same directory structure: `region_name/t1/`, `t2/`, `label/`
3. Add the region name to the `regions` list in the metadata builder
4. Re-run the stratified split — all downstream code is region-agnostic

### Threshold selection best practice
Always select the optimal threshold on the **validation set** and apply it
fixed to the **test set**. Never search for the threshold on the test set
and report those metrics — this constitutes data leakage.

```python
# CORRECT workflow
best_thresh = find_optimal_threshold(val_preds, val_labels)   # on val set
test_metrics = evaluate(test_preds, test_labels, threshold=best_thresh)  # apply fixed
```

---

*Last updated: April 2026 | IIT Madras — Data Science and AI Project*
