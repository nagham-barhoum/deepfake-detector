import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import torch

class GenImageDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", transform=None):
        self.samples   = []
        self.transform = transform

        split_dir = os.path.join(root_dir, split)
        ai_dir = os.path.join(split_dir, "ai")
        for fname in os.listdir(ai_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                self.samples.append((os.path.join(ai_dir, fname), 1))

        real_dir = os.path.join(split_dir, "real")
        for fname in os.listdir(real_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                self.samples.append((os.path.join(real_dir, fname), 0))

        n_ai     = sum(1 for _, l in self.samples if l == 1)
        n_real = sum(1 for _, l in self.samples if l == 0)
        print(f"[{split}] AI: {n_ai} | real: {n_real} | Total: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def get_class_weights(dataset: GenImageDataset) -> torch.Tensor:
    n_total   = len(dataset.samples)
    n_ai      = sum(1 for _, l in dataset.samples if l == 1)
    n_real  = sum(1 for _, l in dataset.samples if l == 0)

    weight_real = n_total / (2 * n_real)   # 16662 / (2 * 4662) = 1.79
    weight_ai     = n_total / (2 * n_ai)       # 16662 / (2 * 12000) = 0.69

    print(f"Class weights → real: {weight_real:.2f} | AI: {weight_ai:.2f}")
    return torch.tensor([weight_real, weight_ai], dtype=torch.float)


def get_sampler(dataset: GenImageDataset) -> WeightedRandomSampler:
    """
    الخيار 2 — WeightedRandomSampler
    بيخلي الـ real تتكرر أكثر في كل batch
    """
    n_ai     = sum(1 for _, l in dataset.samples if l == 1)
    n_real = sum(1 for _, l in dataset.samples if l == 0)

    weight_per_class = {
        1: 1.0 / n_ai,
        0: 1.0 / n_real
    }

    sample_weights = [weight_per_class[label] for _, label in dataset.samples]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)

    return WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True
    )


def get_dataloaders(dataset_dir: str, batch_size: int = 32, num_workers: int = 0):
    train_tf, val_tf = get_transforms()

    train_dataset = GenImageDataset(dataset_dir, split="train", transform=train_tf)
    val_dataset   = GenImageDataset(dataset_dir, split="val",   transform=val_tf)

    sampler = get_sampler(train_dataset)
    class_weights = get_class_weights(train_dataset)

    use_persistent = num_workers > 0
    prefetch       = 2 if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size        = batch_size,
        sampler           = sampler,
        num_workers       = num_workers,
        pin_memory        = torch.cuda.is_available(),
        persistent_workers= use_persistent,
        prefetch_factor   = prefetch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size        = batch_size,
        shuffle           = False,
        num_workers       = num_workers,
        pin_memory        = torch.cuda.is_available(),
        persistent_workers= use_persistent,
        prefetch_factor   = prefetch,
    )

    return train_loader, val_loader, class_weights