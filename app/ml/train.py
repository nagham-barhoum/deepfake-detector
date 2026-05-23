import os
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from sklearn.metrics import classification_report, roc_auc_score # مقياس لجودة المودل وتقييمو كل ماكانت اقرب لل1 كان احسسن

from dataset_loader import get_dataloaders #لتحضير DataLoader للتدريب/التحقق
from model import DeepfakeDetector # كـ model


# ══════════════════════════════════════════════════════════════════════
#  ⚙️ Config (CPU Optimized)
# ══════════════════════════════════════════════════════════════════════

CONFIG = {
    "dataset_dir": "../../dataset",
    "save_dir": "models",
    "model_name": "efficientnet_detector.pth",

    "epochs": 10,

    # CPU OPTIMIZED
    "batch_size": 8,          # 🔥 مهم للـ CPU
    "lr": 3e-4, #معدل التعلم
    "weight_decay": 1e-4, #L2 regularization

    "patience": 3, #Early Stopping
    "min_delta": 1e-4,

    "save_best_only": True,

    # 🔥 CPU tuning
    "num_workers": 2
}


# ══════════════════════════════════════════════════════════════════════
#  🔧 Device
# ══════════════════════════════════════════════════════════════════════

def get_device():
    device = torch.device("cpu")
    print("Running on CPU (optimized mode)")
    print(f"batch_size = {CONFIG['batch_size']}")
    print(f"num_workers = {CONFIG['num_workers']}")
    return device


# ══════════════════════════════════════════════════════════════════════
#  🚀 Train one epoch (CPU optimized)
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device, epoch_idx):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for step, (images, labels) in enumerate(loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)        # 1. Forward Pass
        loss   = criterion(logits, labels)  # 2. حساب الـ Loss
        loss.backward()               # 3. Backward Pass
        optimizer.step()              # 4. تحديث الأوزان

        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        # 🔥 أقل logging لتقليل overhead
        if (step + 1) % 50 == 0:
            acc = correct / total * 100
            print(f"step [{step+1}/{len(loader)}] loss={loss.item():.4f} acc={acc:.1f}%")

    return total_loss / total, correct / total * 100


# ══════════════════════════════════════════════════════════════════════
#  📊 Evaluate (CPU optimized)
# ══════════════════════════════════════════════════════════════════════

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)[:, 1]

            total_loss += loss.item() * images.size(0)

            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            # 🔥 CPU optimized collection
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.tolist())

    avg_loss = total_loss / total
    avg_acc = correct / total * 100
    auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, avg_acc, auc, np.array(all_labels), np.array(all_probs)


# ══════════════════════════════════════════════════════════════════════
#  🚀 Main
# ══════════════════════════════════════════════════════════════════════

def main():

    device = get_device()

    print("\n==============================")
    print(" Deepfake Detector (CPU Mode)")
    print("==============================\n")

    train_loader, val_loader, class_weights = get_dataloaders(
        dataset_dir=CONFIG["dataset_dir"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )

    print("Loading model...")
    model = DeepfakeDetector(pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    save_path = os.path.join(CONFIG["save_dir"], CONFIG["model_name"])
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    best_loss = float("inf")
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}

    start = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):

        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")

        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        v_loss, v_acc, v_auc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Train loss: {t_loss:.4f} acc: {t_acc:.2f}%")
        print(f"Val loss:   {v_loss:.4f} acc: {v_acc:.2f}% AUC: {v_auc:.4f}")

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["val_auc"].append(v_auc)

        # Early stopping
        if best_loss - v_loss > CONFIG["min_delta"]:
            best_loss = v_loss
            patience_counter = 0

            torch.save(model.state_dict(), save_path)
            print("Saved best model")
        else:
            patience_counter += 1
            print(f"No improvement {patience_counter}/{CONFIG['patience']}")

            if patience_counter >= CONFIG["patience"]:
                print("Early stopping")
                break

    print("\nTraining finished")
    print("Time:", time.time() - start)

    with open(os.path.join(CONFIG["save_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()