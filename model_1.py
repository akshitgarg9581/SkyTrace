"""
model_1.py  —  Improved Contrail Detection Pipeline (v2)
─────────────────────────────────────────────────────────
Key improvements over model.py:
  1. Multi-frame temporal input  (t-1, t, t+1 → 9-channel Ash-RGB stack)
  2. 8,000 training samples      (up from 4,000)
  3. Stronger data augmentation  (ElasticTransform, GridDistortion, CoarseDropout, …)
  4. OneCycleLR scheduler        (warm-up → cosine decay for faster convergence)
  5. Early stopping              (patience=5 to stop if no improvement)
  6. Gradient accumulation       (effective batch = 32 even with batch_size=16)
  7. Mixed-precision training    (AMP for 2× speed on GPU)
  8. Threshold tuning            (auto-search best threshold on validation set)
  
Designed to run on Kaggle with GPU.  Copy-paste entire file into a notebook cell.
"""

import os, json, glob, time, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
class CFG:
    # ── paths (Kaggle default)
    DATA_DIR   = "/kaggle/input/competitions/google-research-identify-contrails-reduce-global-warming"
    TRAIN_DIR  = os.path.join(DATA_DIR, "train")
    VALID_DIR  = os.path.join(DATA_DIR, "validation")
    SAVE_PATH  = "/kaggle/working/contrail_model.pth"
    HIST_PATH  = "/kaggle/working/history.json"

    # ── model architecture
    ENCODER      = "efficientnet-b3"       # pretrained ImageNet backbone
    IN_CHANS     = 9                       # 3 timestamps × 3 Ash-RGB channels
    NUM_CLASSES  = 1
    DECODER_ATTN = "scse"                  # Squeeze-and-Excitation attention in decoder

    # ── training hyperparameters
    IMAGE_SIZE   = 256
    BATCH_SIZE   = 16
    ACCUM_STEPS  = 2                       # gradient accumulation → effective batch = 32
    EPOCHS       = 25
    LR           = 3e-4                    # peak LR for OneCycleLR
    WEIGHT_DECAY = 1e-4
    TRAIN_SIZE   = 8000                    # training records to use
    VALID_SIZE   = 1000                    # validation records to use

    # ── early stopping
    PATIENCE     = 5                       # stop if val Dice doesn't improve for 5 epochs

    # ── inference
    THRESHOLD    = 0.35                    # will be auto-tuned after training

    # ── misc
    SEED         = 42
    NUM_WORKERS  = 2
    USE_AMP      = True                    # mixed precision (set False if no GPU)
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed=CFG.SEED):
    """Reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()
print(f"[CFG] Device: {CFG.DEVICE}  |  Encoder: {CFG.ENCODER}  |  "
      f"In-channels: {CFG.IN_CHANS}  |  Epochs: {CFG.EPOCHS}")
print(f"[CFG] Train size: {CFG.TRAIN_SIZE}  |  Valid size: {CFG.VALID_SIZE}  |  "
      f"Effective batch: {CFG.BATCH_SIZE * CFG.ACCUM_STEPS}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ASH-COLOR HELPER  (same physics, multi-frame support)
# ══════════════════════════════════════════════════════════════════════════════
BAND_RANGES = {
    "r": (-4.0,  2.0),
    "g": (-4.0,  5.0),
    "b": (243.0, 303.0),
}


def load_band(record_dir: str, band: str, t: int) -> np.ndarray:
    """Load a single timestamp slice from a band file (memory-mapped).
    Returns float32 array of shape (H, W)."""
    path = os.path.join(record_dir, f"{band}.npy")
    arr = np.load(path, mmap_mode='r')
    return arr[..., t].astype(np.float32)


def ash_color(record_dir: str, t: int = 4) -> np.ndarray:
    """
    Compute 3-channel Ash-Color image for a single timestamp t.
    Returns float32 (H, W, 3) in [0, 1].
    """
    b11 = load_band(record_dir, "band_11", t)
    b14 = load_band(record_dir, "band_14", t)
    b15 = load_band(record_dir, "band_15", t)

    R = b15 - b14
    G = b14 - b11
    B = b14

    def _norm(x, lo, hi):
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    R = _norm(R, *BAND_RANGES["r"])
    G = _norm(G, *BAND_RANGES["g"])
    B = _norm(B, *BAND_RANGES["b"])

    return np.stack([R, G, B], axis=-1)


def ash_color_multiframe(record_dir: str, timestamps=(3, 4, 5)) -> np.ndarray:
    """
    Stack Ash-Color images from multiple timestamps into a single
    multi-channel image.  Default = (t-1, t, t+1) = 3 × 3 = 9 channels.
    
    This gives the model temporal context — it can detect motion patterns
    that distinguish contrails (which move/persist) from normal clouds.
    
    Returns float32 (H, W, 9) in [0, 1].
    """
    frames = []
    for t in timestamps:
        frames.append(ash_color(record_dir, t=t))
    return np.concatenate(frames, axis=-1)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DATASET  (multi-frame + stronger augmentation)
# ══════════════════════════════════════════════════════════════════════════════
class ContrailDataset(Dataset):
    """
    Loads multi-frame Ash-Color images (t-1, t, t+1) → 9-channel tensor.
    Ground-truth mask from human_pixel_masks.npy.
    """

    def __init__(self, data_dir: str, transform=None, max_size=None):
        all_paths = glob.glob(os.path.join(data_dir, "*"))
        self.records = sorted([p for p in all_paths if os.path.isdir(p)])
        if len(self.records) == 0:
            raise RuntimeError(
                f"No record folders found in: {data_dir}\n"
                "Make sure the Kaggle dataset is attached and the path is correct."
            )
        # Apply size limit
        if max_size and len(self.records) > max_size:
            self.records = self.records[:max_size]
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # ── multi-frame Ash image: t=3, t=4, t=5  →  (256, 256, 9)
        image = ash_color_multiframe(rec, timestamps=(3, 4, 5))

        # ── ground truth mask
        mask_path = os.path.join(rec, "human_pixel_masks.npy")
        mask = np.load(mask_path).astype(np.float32)
        if mask.ndim == 3:
            mask = (mask.mean(axis=-1) > 0).astype(np.float32)

        # ── augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1))
            mask  = torch.from_numpy(mask)

        mask = mask.unsqueeze(0)
        return image, mask


def get_transforms(train: bool):
    """
    Stronger augmentation pipeline for training.
    Validation uses only ToTensorV2 (no augmentation).
    Uses up-to-date albumentations API to avoid deprecation warnings.
    """
    if train:
        return A.Compose([
            # ── spatial transforms (applied identically to image + mask)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent=0.08,
                scale=(0.85, 1.15),
                rotate=(-20, 20),
                p=0.5,
            ),
            A.ElasticTransform(
                alpha=50, sigma=10,
                p=0.2,
            ),
            A.GridDistortion(
                num_steps=5, distort_limit=0.2,
                p=0.2,
            ),
            # ── pixel-level transforms (image only, mask unchanged)
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=0.5,
            ),
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=0.2,
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([ToTensorV2()])


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL  (U-Net + EfficientNet-B3 + SE attention)
# ══════════════════════════════════════════════════════════════════════════════
def get_model():
    """
    U-Net with EfficientNet-B3 encoder and Squeeze-and-Excitation (scSE)
    attention blocks in the decoder for better feature recalibration.
    
    9-channel input to capture temporal information from 3 timestamps.
    """
    model = smp.Unet(
        encoder_name    = CFG.ENCODER,
        encoder_weights = "imagenet",
        in_channels     = CFG.IN_CHANS,
        classes         = CFG.NUM_CLASSES,
        activation      = None,
        decoder_attention_type = CFG.DECODER_ATTN,
    )
    return model.to(CFG.DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  LOSS  (Dice + Focal — numerically stable version)
# ══════════════════════════════════════════════════════════════════════════════
class CombinedLoss(nn.Module):
    """0.5 × Dice + 0.5 × Focal.  Numerically stable — forces float32."""
    def __init__(self):
        super().__init__()
        self.dice  = DiceLoss(mode="binary", from_logits=True, smooth=1.0)
        self.focal = FocalLoss(mode="binary", gamma=2.0)

    def forward(self, logits, targets):
        # Force float32 to prevent NaN from mixed precision
        logits  = logits.float()
        targets = targets.float()
        # Clamp logits to prevent extreme values that cause overflow
        logits = torch.clamp(logits, -50.0, 50.0)
        return 0.5 * self.dice(logits, targets) + 0.5 * self.focal(logits, targets)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  METRICS
# ══════════════════════════════════════════════════════════════════════════════
def dice_score(preds: torch.Tensor, targets: torch.Tensor,
               threshold: float = CFG.THRESHOLD, eps: float = 1e-6) -> float:
    """Compute binary Dice score from raw logits."""
    preds   = preds.float()      # ensure float32
    targets = targets.float()    # ensure float32
    preds   = (torch.sigmoid(preds) > threshold).float()
    inter   = (preds * targets).sum()
    return (2.0 * inter / (preds.sum() + targets.sum() + eps)).item()


# ══════════════════════════════════════════════════════════════════════════════
# 7.  TRAINING LOOP  (with AMP + gradient accumulation)
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion, scaler, scheduler):
    """
    Train for one epoch with:
      - Mixed precision (AMP) for speed
      - Gradient accumulation for larger effective batch size
      - OneCycleLR stepping per batch (not per epoch)
    """
    model.train()
    total_loss, total_dice = 0.0, 0.0
    optimizer.zero_grad()

    for step, (images, masks) in enumerate(tqdm(loader, desc="Train", leave=False)):
        images = images.to(CFG.DEVICE, non_blocking=True)
        masks  = masks.to(CFG.DEVICE, non_blocking=True)

        # ── forward pass with AMP
        with torch.amp.autocast("cuda", enabled=CFG.USE_AMP):
            logits = model(images)

        # ── compute loss in float32 (criterion handles casting internally)
        loss = criterion(logits.float(), masks.float()) / CFG.ACCUM_STEPS

        # ── backward pass with gradient scaling
        scaler.scale(loss).backward()

        # ── gradient accumulation: only step every ACCUM_STEPS
        if (step + 1) % CFG.ACCUM_STEPS == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * CFG.ACCUM_STEPS
        total_dice += dice_score(logits.detach().float(), masks.detach().float())

    n = len(loader)
    return total_loss / n, total_dice / n


@torch.no_grad()
def validate(model, loader, criterion):
    """Validate one epoch — all loss computed in float32 for stability."""
    model.eval()
    total_loss, total_dice = 0.0, 0.0

    for images, masks in tqdm(loader, desc="Valid", leave=False):
        images = images.to(CFG.DEVICE, non_blocking=True)
        masks  = masks.to(CFG.DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=CFG.USE_AMP):
            logits = model(images)

        # Cast to float32 before loss computation to prevent NaN
        logits_f32 = logits.float()
        masks_f32  = masks.float()

        loss = criterion(logits_f32, masks_f32)

        # Guard against NaN — skip batch if loss is invalid
        if not torch.isfinite(loss):
            continue

        total_loss += loss.item()
        total_dice += dice_score(logits_f32, masks_f32)

    n = len(loader)
    return total_loss / max(n, 1), total_dice / max(n, 1)


def train():
    """
    Full training pipeline:
      1. Build datasets & dataloaders
      2. Initialise model, optimizer, scheduler, loss
      3. Train with early stopping
      4. Save best model + history
    """
    # ── datasets & loaders
    train_ds = ContrailDataset(
        CFG.TRAIN_DIR, transform=get_transforms(True), max_size=CFG.TRAIN_SIZE
    )
    valid_ds = ContrailDataset(
        CFG.VALID_DIR, transform=get_transforms(False), max_size=CFG.VALID_SIZE
    )
    print(f"[Data] Train: {len(train_ds)} records  |  Valid: {len(valid_ds)} records")

    train_loader = DataLoader(
        train_ds, batch_size=CFG.BATCH_SIZE,
        shuffle=True, num_workers=CFG.NUM_WORKERS,
        pin_memory=True, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=CFG.BATCH_SIZE,
        shuffle=False, num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
    )

    # ── model, optimizer, scheduler, loss
    model     = get_model()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY
    )

    # OneCycleLR: warm-up to peak LR then cosine decay to near zero
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CFG.LR,
        steps_per_epoch=steps_per_epoch,
        epochs=CFG.EPOCHS,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    criterion = CombinedLoss()
    scaler    = torch.amp.GradScaler("cuda", enabled=CFG.USE_AMP)

    # ── history tracking
    history   = {
        "train_loss": [], "train_dice": [],
        "val_loss":   [], "val_dice":   [],
        "lr":         [],
    }
    best_dice = 0.0
    patience_counter = 0

    # ── header
    print(f"\n{'Epoch':>6} {'TrLoss':>10} {'TrDice':>10} "
          f"{'VlLoss':>10} {'VlDice':>10} {'LR':>12} {'Status':>10}")
    print("─" * 75)

    t_start = time.time()

    for epoch in range(1, CFG.EPOCHS + 1):
        # ── train + validate
        tr_loss, tr_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, scheduler
        )
        vl_loss, vl_dice = validate(model, valid_loader, criterion)

        # ── current LR (from scheduler)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── record history
        history["train_loss"].append(tr_loss)
        history["train_dice"].append(tr_dice)
        history["val_loss"].append(vl_loss)
        history["val_dice"].append(vl_dice)
        history["lr"].append(current_lr)

        # ── check for improvement
        status = ""
        if vl_dice > best_dice:
            best_dice = vl_dice
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch":       epoch,
                    "best_dice":   best_dice,
                    "cfg": {k: str(v) for k, v in vars(CFG).items()
                            if not k.startswith("_")},
                },
                CFG.SAVE_PATH,
            )
            status = "✓ SAVED"
        else:
            patience_counter += 1
            status = f"wait {patience_counter}/{CFG.PATIENCE}"

        print(f"{epoch:>6} {tr_loss:>10.4f} {tr_dice:>10.4f} "
              f"{vl_loss:>10.4f} {vl_dice:>10.4f} {current_lr:>12.2e} {status:>10}")

        # ── early stopping
        if patience_counter >= CFG.PATIENCE:
            print(f"\n[Early Stop] No improvement for {CFG.PATIENCE} epochs. Stopping.")
            break

    elapsed = time.time() - t_start
    print(f"\n[Done] Best Val Dice = {best_dice:.4f}  |  "
          f"Total time: {elapsed/60:.1f} min")

    # ── save training history
    with open(CFG.HIST_PATH, "w") as f:
        json.dump({"history": history, "best_dice": best_dice}, f, indent=2)
    print(f"[Saved] history.json")

    # ── plot training curves
    plot_training_curves(history, best_dice)

    return model, valid_loader


# ══════════════════════════════════════════════════════════════════════════════
# 8.  TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════
def plot_training_curves(history: dict, best_dice: float):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("Training History (Improved Model v2)", fontsize=14, fontweight="bold")

    # ── Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "b-o", ms=4, label="Train")
    ax.plot(epochs, history["val_loss"],   "r-o", ms=4, label="Val")
    ax.set_title("Loss (0.5×Dice + 0.5×Focal)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # ── Dice Score
    ax = axes[1]
    ax.plot(epochs, history["train_dice"], "b-o", ms=4, label="Train")
    ax.plot(epochs, history["val_dice"],   "r-o", ms=4, label="Val")
    ax.axhline(best_dice, color="green", ls="--",
               label=f"Best Val: {best_dice:.4f}")
    ax.set_title("Dice Score")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Dice")
    ax.legend(); ax.grid(alpha=0.3)

    # ── Learning Rate
    ax = axes[2]
    if "lr" in history:
        ax.plot(epochs, history["lr"], "g-o", ms=4)
    ax.set_title("Learning Rate (OneCycleLR)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.grid(alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    plt.tight_layout()
    plt.savefig("/kaggle/working/training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → training_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  THRESHOLD TUNING
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def find_best_threshold(model, loader, thresholds=None):
    """
    Search for the optimal binarization threshold on the validation set.
    Tests a range of thresholds and picks the one with highest Dice.
    """
    if thresholds is None:
        thresholds = np.arange(0.20, 0.65, 0.05)

    model.eval()

    # collect all probabilities and masks
    all_probs, all_masks = [], []
    for images, masks in tqdm(loader, desc="Threshold Search", leave=False):
        images = images.to(CFG.DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=CFG.USE_AMP):
            logits = model(images)
        probs = torch.sigmoid(logits.float()).cpu()
        all_probs.append(probs)
        all_masks.append(masks)

    all_probs = torch.cat(all_probs)
    all_masks = torch.cat(all_masks)

    best_thr, best_dice = 0.35, 0.0
    print(f"\n{'Threshold':>10} {'Dice':>10}")
    print("─" * 25)

    for thr in thresholds:
        preds = (all_probs > thr).float()
        inter = (preds * all_masks).sum()
        dice  = (2.0 * inter / (preds.sum() + all_masks.sum() + 1e-6)).item()
        marker = " ◄" if dice > best_dice else ""
        print(f"{thr:>10.2f} {dice:>10.4f}{marker}")
        if dice > best_dice:
            best_dice = dice
            best_thr  = thr

    print(f"\n[Threshold] Best = {best_thr:.2f}  →  Dice = {best_dice:.4f}")
    return best_thr


# ══════════════════════════════════════════════════════════════════════════════
# 10.  EVALUATION / INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model=None, valid_loader=None, threshold=None):
    """
    Load best saved weights, run inference on validation set,
    compute pixel-level metrics, auto-tune threshold, and visualise.
    """
    # ── rebuild loader if not provided
    if valid_loader is None:
        valid_ds = ContrailDataset(
            CFG.VALID_DIR, transform=get_transforms(False), max_size=CFG.VALID_SIZE
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=CFG.BATCH_SIZE,
            shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True,
        )

    # ── load best model weights
    if model is None:
        model = get_model()
    checkpoint = torch.load(CFG.SAVE_PATH, map_location=CFG.DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print(f"[Eval] Loaded best model from epoch {checkpoint['epoch']}  "
          f"(saved Dice = {checkpoint['best_dice']:.4f})")
    model.eval()

    # ── auto-tune threshold if not provided
    if threshold is None:
        threshold = find_best_threshold(model, valid_loader)
    print(f"[Eval] Using threshold = {threshold:.2f}")

    # ── collect predictions
    all_preds, all_masks, sample_batches = [], [], []
    for i, (images, masks) in enumerate(tqdm(valid_loader, desc="Inference")):
        images = images.to(CFG.DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=CFG.USE_AMP):
            logits = model(images)
        probs = torch.sigmoid(logits.float()).cpu()
        preds = (probs > threshold).float()

        all_preds.append(preds.flatten().numpy())
        all_masks.append(masks.flatten().numpy())

        if i < 5:
            sample_batches.append((
                images.cpu(), masks.cpu(), probs.cpu(), preds.cpu()
            ))

    all_preds = np.concatenate(all_preds).astype(np.uint8)
    all_masks = np.concatenate(all_masks).astype(np.uint8)

    # ── pixel-level confusion matrix
    TP = int(((all_preds == 1) & (all_masks == 1)).sum())
    TN = int(((all_preds == 0) & (all_masks == 0)).sum())
    FP = int(((all_preds == 1) & (all_masks == 0)).sum())
    FN = int(((all_preds == 0) & (all_masks == 1)).sum())
    EPS = 1e-6

    metrics = {
        "Accuracy"   : (TP + TN) / (TP + TN + FP + FN + EPS),
        "Precision"  : TP / (TP + FP + EPS),
        "Recall"     : TP / (TP + FN + EPS),
        "F1 / Dice"  : 2 * TP / (2 * TP + FP + FN + EPS),
        "IoU"        : TP / (TP + FP + FN + EPS),
        "Specificity": TN / (TN + FP + EPS),
    }

    # ── print summary
    print("\n" + "=" * 55)
    print("         EVALUATION SUMMARY  (Improved Model v2)")
    print("=" * 55)
    for name, val in metrics.items():
        bar = "█" * int(val * 30)
        print(f"  {name:<12} {val:.4f}  {bar}")
    print("=" * 55)
    print(f"  Threshold: {threshold:.2f}")
    print(f"  TP: {TP:>12,}   (correct contrail pixels)")
    print(f"  TN: {TN:>12,}   (correct background pixels)")
    print(f"  FP: {FP:>12,}   (false alarms)")
    print(f"  FN: {FN:>12,}   (missed contrails)")
    print("=" * 55)

    # ── plots
    _plot_confusion_and_metrics(TP, TN, FP, FN, metrics, threshold)
    _plot_sample_predictions(sample_batches, threshold)

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 11.  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════
def _plot_confusion_and_metrics(TP, TN, FP, FN, metrics, threshold):
    cm = np.array([[TN, FP], [FN, TP]])
    labels  = np.array([["TN", "FP"], ["FN", "TP"]])
    colours = ["#2ecc71", "#e74c3c", "#e67e22", "#3498db"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Model Evaluation — Validation Set  (threshold={threshold:.2f})",
                 fontsize=15, fontweight="bold", y=1.02)

    # ── confusion matrix
    for idx, (i, j) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
        ax1.add_patch(plt.Rectangle((j, 1-i), 1, 1,
                      color=colours[idx], ec="white", lw=2))
        val = cm[i, j]
        pct = val / cm.sum() * 100
        ax1.text(j+0.5, 1.5-i,
                 f"{labels[i,j]}\n{val:,}\n({pct:.1f}%)",
                 ha="center", va="center",
                 fontsize=12, fontweight="bold", color="white")

    ax1.set_xlim(0, 2); ax1.set_ylim(0, 2)
    ax1.set_xticks([0.5, 1.5])
    ax1.set_xticklabels(["Predicted\nNo Contrail", "Predicted\nContrail"], fontsize=11)
    ax1.set_yticks([0.5, 1.5])
    ax1.set_yticklabels(["Actual\nContrail", "Actual\nNo Contrail"], fontsize=11)
    ax1.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    ax1.tick_params(length=0)

    # ── metrics bar chart
    names  = list(metrics.keys())
    values = list(metrics.values())
    bar_colors = ["#3498db","#2ecc71","#e67e22","#9b59b6","#1abc9c","#e74c3c"]
    bars = ax2.barh(names, values, color=bar_colors, edgecolor="white", height=0.55)
    for bar, val in zip(bars, values):
        ax2.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, 1.15)
    ax2.set_xlabel("Score", fontsize=12)
    ax2.set_title("Evaluation Metrics", fontsize=13, fontweight="bold", pad=12)
    ax2.axvline(0.5, color="gray", ls="--", alpha=0.5, label="0.5 baseline")
    ax2.grid(axis="x", alpha=0.3)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig("/kaggle/working/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → confusion_matrix.png")


def _plot_sample_predictions(sample_batches, threshold):
    """Show up to 5 sample prediction comparisons."""
    n = min(5, len(sample_batches))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle("Validation — Ground Truth vs Prediction (Improved Model v2)",
                 fontsize=14, fontweight="bold")

    for row, (images, masks, probs, preds) in enumerate(sample_batches[:n]):
        img = images[0]
        # Use the middle frame (channels 3-5 = t=4) for display
        ash = img[3:6].permute(1, 2, 0).numpy()
        gt  = masks[0, 0].numpy()
        pr  = probs[0, 0].numpy()

        ax_img, ax_gt, ax_pr = axes[row]

        ax_img.imshow(ash)
        ax_img.set_title("Input (Ash RGB, t=4)")
        ax_img.axis("off")

        ax_gt.imshow(ash, alpha=0.7)
        ax_gt.imshow(gt, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
        ax_gt.set_title("Ground Truth")
        ax_gt.axis("off")

        ax_pr.imshow(ash, alpha=0.7)
        ax_pr.imshow(pr, cmap="Blues", alpha=0.5, vmin=0, vmax=1)
        ax_pr.set_title(f"Prediction (thr={threshold:.2f})")
        ax_pr.axis("off")

    plt.tight_layout()
    plt.savefig("/kaggle/working/predictions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → predictions.png")


# ══════════════════════════════════════════════════════════════════════════════
# 12.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  CONTRAIL DETECTION — Improved Training Pipeline (v2)")
    print(f"  Encoder     : {CFG.ENCODER}")
    print(f"  Input       : {CFG.IN_CHANS} channels (3 frames × 3 Ash bands)")
    print(f"  Device      : {CFG.DEVICE}")
    print(f"  Epochs      : {CFG.EPOCHS}  |  Batch: {CFG.BATCH_SIZE} × {CFG.ACCUM_STEPS} accum = {CFG.BATCH_SIZE * CFG.ACCUM_STEPS}")
    print(f"  Train size  : {CFG.TRAIN_SIZE}  |  Valid size: {CFG.VALID_SIZE}")
    print(f"  Scheduler   : OneCycleLR (peak={CFG.LR})")
    print(f"  Loss        : 0.5×Dice + 0.5×Focal")
    print(f"  Early stop  : patience={CFG.PATIENCE}")
    print(f"  AMP         : {CFG.USE_AMP}")
    print("=" * 65 + "\n")

    trained_model, val_loader = train()
    evaluate(trained_model, val_loader)
