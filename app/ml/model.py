import torch
import torch.nn as nn
from torchvision import models

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

class DeepfakeDetector(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        # تحميل EfficientNet-B0 مدرّب مسبقاً على ImageNet
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # استبدال آخر طبقة لتناسب مشكلتنا (2 classes فقط)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 2)
        )

    def forward(self, x):
        return self.backbone(x)

    def predict_proba(self, x):
        """يرجع احتمال كون الصورة AI بين 0 و 1"""
        logits = self.forward(x)
        probs  = torch.softmax(logits, dim=1)
        return probs[:, 1]  # احتمال الكلاس AI