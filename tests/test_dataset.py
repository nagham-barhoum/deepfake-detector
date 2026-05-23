from app.ml.dataset_loader import get_dataloaders

train_loader, val_loader, class_weights = get_dataloaders(
    dataset_dir = "dataset",
    batch_size  = 32
)

print(f"\nClass weights: {class_weights}")

images, labels = next(iter(train_loader))
ai_count     = (labels == 1).sum().item()
nature_count = (labels == 0).sum().item()
print(f"Batch — AI: {ai_count} | Nature: {nature_count}")
print("✅ Dataset + Balancing شغّال!")