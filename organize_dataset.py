"""
organize_dataset.py
Organizes all real and AI images into train/val folders
while guaranteeing no duplication between splits
"""

import os
import shutil
import random

# ══════════════════════════════════════════════════════════════════════
#  ⚙️  Settings
# ══════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42          # Always fixed → same result on every run
RAW_DIR     = "dataset/raw"
OUTPUT_DIR  = "dataset"
TRAIN_RATIO = 0.8

AI_FOLDERS = [
    "ADM",
    "BigGAN",
    "glide",
    "Midjourney",
    "stable_diffusion_v_1_5",   # ← correct folder name
    "VQDM",
    "wukong",
]

REAL_FOLDERS = [
    "Nature",
    "CelebA",
    "COCO_real",
]

EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# ══════════════════════════════════════════════════════════════════════
#  🧹  Step 1: Remove old folders completely
#      Necessary to guarantee no leftover files from previous runs
# ══════════════════════════════════════════════════════════════════════

print("\n" + "═" * 55)
print("  🗂️   Organizing Dataset (Clean Setup)")
print("═" * 55)

print("\n  🧹  Removing old folders...")
for split in ["train", "val"]:
    for label in ["ai", "real"]:
        folder_path = os.path.join(OUTPUT_DIR, split, label)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"  ✗  Deleted: {folder_path}")
        os.makedirs(folder_path)
        print(f"  ✓  Created: {folder_path}")

# ══════════════════════════════════════════════════════════════════════
#  🎲  Step 2: Set random seed
# ══════════════════════════════════════════════════════════════════════

random.seed(RANDOM_SEED)
print(f"\n  🎲  Random seed = {RANDOM_SEED}  (same result every run)")

# ══════════════════════════════════════════════════════════════════════
#  📁  Step 3: Distribution function
# ══════════════════════════════════════════════════════════════════════

def organize(src_folder: str, label: str) -> tuple[int, int]:
    """
    Splits images from one folder into train and val
    - returns (n_train, n_val)
    - filename format = {folder_name}_{original_name} to avoid conflicts
    """
    if not os.path.exists(src_folder):
        print(f"  ⚠️  Folder not found: {src_folder}")
        return 0, 0

    files = sorted([
        f for f in os.listdir(src_folder)
        if f.lower().endswith(EXTENSIONS)
    ])

    if not files:
        print(f"  ⚠️  No images found in: {src_folder}")
        return 0, 0

    # shuffle after sort → reproducible with same seed
    random.shuffle(files)

    split_idx   = int(len(files) * TRAIN_RATIO)
    train_files = files[:split_idx]
    val_files   = files[split_idx:]

    folder_name = os.path.basename(src_folder)

    for f in train_files:
        dst = os.path.join(OUTPUT_DIR, "train", label, f"{folder_name}_{f}")
        shutil.copy2(os.path.join(src_folder, f), dst)

    for f in val_files:
        dst = os.path.join(OUTPUT_DIR, "val", label, f"{folder_name}_{f}")
        shutil.copy2(os.path.join(src_folder, f), dst)

    print(f"  ✅ {folder_name:<25}: train={len(train_files):>5,} | val={len(val_files):>4,}")
    return len(train_files), len(val_files)

# ══════════════════════════════════════════════════════════════════════
#  🤖  Step 4: Distribute images
# ══════════════════════════════════════════════════════════════════════

print("\n  🤖  AI folders:")
total_train_ai = total_val_ai = 0
for folder in AI_FOLDERS:
    tr, vl = organize(os.path.join(RAW_DIR, folder), "ai")
    total_train_ai += tr
    total_val_ai   += vl

print("\n  📷  Real image folders:")
total_train_real = total_val_real = 0
for folder in REAL_FOLDERS:
    tr, vl = organize(os.path.join(RAW_DIR, folder), "real")
    total_train_real += tr
    total_val_real   += vl

# ══════════════════════════════════════════════════════════════════════
#  🔍  Step 5: Verify no duplication
#      Compare filenames between train and val
# ══════════════════════════════════════════════════════════════════════

print("\n" + "─" * 55)
print("  🔍  Checking for duplicates...")

overlap_found = False
for label in ["ai", "real"]:
    train_files = set(os.listdir(os.path.join(OUTPUT_DIR, "train", label)))
    val_files   = set(os.listdir(os.path.join(OUTPUT_DIR, "val",   label)))
    overlap     = train_files & val_files

    if overlap:
        overlap_found = True
        print(f"  ❌  [{label}] Duplicate detected! {len(overlap)} files exist in both train and val:")
        for f in list(overlap)[:5]:   # print first 5 only
            print(f"       - {f}")
    else:
        print(f"  ✅  [{label}] No duplicates between train and val")

# ══════════════════════════════════════════════════════════════════════
#  📊  Step 6: Final statistics
# ══════════════════════════════════════════════════════════════════════

print("\n" + "═" * 55)
print("  📊  Final Results")
print("═" * 55)
print(f"  Train AI   : {total_train_ai:>6,}")
print(f"  Train Real : {total_train_real:>6,}")
print(f"  Val   AI   : {total_val_ai:>6,}")
print(f"  Val   Real : {total_val_real:>6,}")

total_train = total_train_ai + total_train_real
total_val   = total_val_ai   + total_val_real
print(f"  {'─'*30}")
print(f"  Total Train : {total_train:>6,}")
print(f"  Total Val   : {total_val:>6,}")

# AI/Real ratio
ratio = total_train_ai / max(total_train_real, 1)
if 0.7 <= ratio <= 1.3:
    balance = "✅ Perfect"
elif 0.5 <= ratio <= 1.5:
    balance = "🟡 Acceptable"
else:
    balance = "⚠️  Imbalanced — review dataset sizes"

print(f"\n  AI/Real ratio (train): {ratio:.2f}  {balance}")

if overlap_found:
    print("\n  ❌  Duplicate issue detected — review the script!")
else:
    print("\n  ✅  Dataset is clean — ready for training!")
    print(f"  ← Next step: python train.py\n")