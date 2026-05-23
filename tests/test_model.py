import torch
from app.ml.model import DeepfakeDetector

model = DeepfakeDetector(pretrained=True)

# صورة وهمية للاختبار
dummy = torch.randn(1, 3, 224, 224)

output = model.predict_proba(dummy)
print(f"Output shape : {output.shape}")
print(f"AI probability: {output.item():.4f}")
print("✅ Model شغّال!")