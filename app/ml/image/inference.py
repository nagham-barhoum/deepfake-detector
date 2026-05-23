"""
inference.py — Deepfake / AI Image Detection Inference Pipeline
"""

import os
import io
import argparse
import numpy as np
from PIL import Image

# ──────────────────────────────────────────────────────────
# TORCH
# ──────────────────────────────────────────────────────────

import torch
import torch.nn.functional as F
from torchvision import transforms

# ──────────────────────────────────────────────────────────
# LOCAL IMPORTS
# كل الملفات موجودة داخل app/ml
# ──────────────────────────────────────────────────────────

from model import DeepfakeDetector

from ela import compute_ela, get_ela_interpretation
from noise_analysis import compute_noise, get_noise_interpretation
from fft_analysis import compute_fft, get_fft_interpretation
from metadata_analysis import (
    analyze_metadata,
    get_metadata_interpretation
)

# ──────────────────────────────────────────────────────────
# DEVICE
# ──────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────────────────
# MODEL PATH
# ──────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",        # ← طلع من image/ لـ ml/
    "models",
    "efficientnet_detector.pth"
)

# ──────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────

def load_model():

    model = DeepfakeDetector(pretrained=False)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            epoch    = checkpoint.get("epoch", "?")
            val_loss = checkpoint.get("val_loss", "?")
            print(f"✅ Model loaded from Epoch {epoch} | val_loss={val_loss}")
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("✅ Model loaded successfully")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Model loaded successfully")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Model loaded successfully")

    model.to(DEVICE)
    model.eval()
    return model


model = load_model()

# ──────────────────────────────────────────────────────────
# IMAGE TRANSFORM
# ──────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ──────────────────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────────────────

def normalize_to_jpeg(image: Image.Image) -> Image.Image:
    """
    الـ model تدرّب على JPEG/PNG فقط.
    WEBP يستخدم VP8 compression يخلق artifacts مختلفة تضلل الـ model.
    الحل: نعيد encode الصورة كـ JPEG quality=95 قبل الـ model.
    """
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def predict_image(image_path: str):

    image = Image.open(image_path).convert("RGB")

    # WEBP fix: normalize to JPEG distribution (training was JPEG/PNG only)
    if os.path.splitext(image_path)[1].lower() in (".webp", ".avif", ".heic"):
        image = normalize_to_jpeg(image)

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(tensor).item()
        else:
            logits = model(tensor)
            if logits.shape[-1] == 1:
                prob = torch.sigmoid(logits)[0].item()
            else:
                probs = F.softmax(logits, dim=1)
                prob  = probs[0][1].item()

    return {"fake_probability": float(prob)}

# ──────────────────────────────────────────────────────────
# LABELS
# ──────────────────────────────────────────────────────────

def get_label(score: float):
    if score >= 0.85:
        return "🔴 Likely AI-Generated"
    elif score >= 0.65:
        return "🟠 Probably AI-Generated"
    elif score >= 0.45:
        return "🟡 Unclear / Mixed Signals"
    elif score >= 0.25:
        return "🟢 Probably Real"
    return "✅ Likely Real"


def get_confidence(score: float):
    distance = abs(score - 0.5)
    if distance >= 0.40:
        return "High"
    elif distance >= 0.22:
        return "Medium"
    return "Low"

# ══════════════════════════════════════════════════════════
# SCORE FUSION — Weighted Average (Dynamic Weights)
#
# المبدأ:
#   بدل ما نضيف modifier صغير على الـ ML score،
#   نعطي كل طبقة وزن حقيقي في المعدل النهائي.
#
# الأوزان تتغير بناءً على قوة الأدلة:
#
#  ┌─────────────────────┬────────────┬────────────────────┐
#  │ الحالة              │ وزن ML     │ وزن Metadata       │
#  ├─────────────────────┼────────────┼────────────────────┤
#  │ AI keyword موجود    │ 0.30       │ 0.40 (أدلة قاطعة) │
#  │ كاميرا حقيقية      │ 0.40       │ 0.25 (أدلة قوية)  │
#  │ لا معلومات metadata │ 0.55       │ 0.10 (أدلة ضعيفة) │
#  └─────────────────────┴────────────┴────────────────────┘
# ══════════════════════════════════════════════════════════

def compute_final_score(
    ml_score: float,
    ela_result: dict,
    fft_result: dict,
    noise_result: dict,
    metadata_result: dict,
):
    ela_score      = ela_result.get("ela_score", 0.5)
    fft_score      = fft_result.get("fft_score", 0.5)
    noise_score    = noise_result.get("noise_score", 0.5)
    metadata_score = metadata_result.get("metadata_score", 0.5)

    has_camera       = metadata_result.get("has_camera", False)
    ai_keyword_found = metadata_result.get("ai_keyword_found", False)

    # ── تحديد الأوزان بناءً على قوة الأدلة ───────────────────────────────
    if ai_keyword_found:
        # أدلة قاطعة: اسم أداة AI موجود في الـ metadata
        # → نعطي الـ metadata وزن أكبر، ونخفف الـ ML
        w_ml       = 0.30
        w_ela      = 0.10
        w_fft      = 0.08
        w_noise    = 0.07
        w_metadata = 0.45

    elif has_camera:
        # كاميرا حقيقية مؤكدة: Make + Model موجودان
        # → الـ metadata أدلة قوية إنها حقيقية، نوازن مع ML لكن لا نسمح للـ ML بالهيمنة
        w_ml       = 0.25
        w_ela      = 0.18
        w_fft      = 0.17
        w_noise    = 0.15
        w_metadata = 0.25


    elif metadata_result.get("has_exif", False):
        # EXIF موجود بس بدون كاميرا (Photoshop مثلاً)
        w_ml       = 0.50
        w_ela      = 0.13
        w_fft      = 0.11
        w_noise    = 0.10
        w_metadata = 0.16

    else:
        # لا معلومات metadata على الإطلاق
        # → لا نترك ML وحده يقرر، خفّض وزنه قليلاً
        w_ml       = 0.40
        w_ela      = 0.16
        w_fft      = 0.18
        w_noise    = 0.14
        w_metadata = 0.12


    # ── الـ Weighted Average ──────────────────────────────────────────────
    final_score = (
        ml_score       * w_ml       +
        ela_score      * w_ela      +
        fft_score      * w_fft      +
        noise_score    * w_noise    +
        metadata_score * w_metadata
    )

    # ── Hard Override للحالات القاطعة فقط ────────────────────────────────
    # (بس بسقف معقول، مش override كامل)
    # ملاحظة: نمنع سيناريوه/تسميم الـ ML كثيراً.
    if ai_keyword_found:
        final_score = max(final_score, 0.90)

    # Safety 1: إذا الـ ML واصل 1.0 لكن بقية الطبقات كلها "real-like" بشكل قوي،
    # قلّل الثقة حتى لا تحصل false positives كثيرة.
    non_ml_really_real = (
        (ela_score < 0.12) and
        (fft_score < 0.18) and
        (noise_score < 0.18) and
        (metadata_score < 0.20)
    )
    if non_ml_really_real and ml_score >= 0.95:
        final_score = min(final_score, 0.80)

    # Safety 2: إذا الـ ML واثق جداً لكن metadata ما دعمت لا AI ولا camera حقيقية،
    # لا نخلي الـ fusion يحوّلها إلى "Real-like / Mixed" تلقائياً.
    # (مهم لحالات Leonardo عندما الـ metadata لا تلتقط إشارات كافية)
    no_real_metadata_signal = (
        (not ai_keyword_found) and
        (not has_camera) and
        (metadata_score < 0.25)
    )
    if (ml_score >= 0.95) and no_real_metadata_signal:
        final_score = max(final_score, 0.70)


    final_score = float(np.clip(final_score, 0.0, 1.0))

    # ── حساب الـ modifier للـ debug (الفرق عن ML) ────────────────────────
    modifier = final_score - ml_score

    # أوزان للعرض
    weights_used = {
        "ml": w_ml, "ela": w_ela, "fft": w_fft,
        "noise": w_noise, "metadata": w_metadata
    }

    return round(final_score, 4), round(modifier, 4), weights_used

# ──────────────────────────────────────────────────────────
# DISPLAY
# ──────────────────────────────────────────────────────────

def progress_bar(score: float, length: int = 30):
    filled = int(score * length)
    empty  = length - filled
    return "█" * filled + "░" * empty


def print_header():
    print("\n" + "═" * 60)
    print("  🔍 Deepfake Detector — Multi-Layer Analysis")
    print("═" * 60)


def print_result_line(name, score, desc):
    print(
        f"  ║  {name:<15}: "
        f"{score * 100:>5.1f}%   "
        f"{desc:<20}║"
    )

# ──────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ──────────────────────────────────────────────────────────

def analyze_image(image_path: str):

    print_header()
    print(f"\n  📸 Image: {image_path}")

    print("\n  ⏳ Layer 1/5: ML Detection...")
    prediction = predict_image(image_path)
    ml_score   = float(prediction["fake_probability"])

    print("  ⏳ Layer 2/5: ELA Analysis...")
    ela_result = compute_ela(image_path)

    print("  ⏳ Layer 3/5: FFT Analysis...")
    fft_result = compute_fft(image_path)

    print("  ⏳ Layer 4/5: Noise Analysis...")
    noise_result = compute_noise(image_path)

    print("  ⏳ Layer 5/5: Metadata Analysis...")
    metadata_result = analyze_metadata(image_path)

    final_score, modifier, weights = compute_final_score(
        ml_score,
        ela_result,
        fft_result,
        noise_result,
        metadata_result
    )

    label      = get_label(final_score)
    confidence = get_confidence(final_score)

    print("\n  ╔══════════════════════════════════════════════════════╗")
    print("  ║             Detailed Analysis Results               ║")
    print("  ╠══════════════════════════════════════════════════════╣")
    print("  ║                                                      ║")

    print_result_line("🧠 ML Score",    ml_score,                      f"(w={weights['ml']:.0%})")
    print_result_line("🔬 ELA Score",   ela_result["ela_score"],        f"(w={weights['ela']:.0%})")
    print_result_line("📡 FFT Score",   fft_result["fft_score"],        f"(w={weights['fft']:.0%})")
    print_result_line("🌊 Noise Score", noise_result["noise_score"],    f"(w={weights['noise']:.0%})")
    print_result_line("📋 Metadata",    metadata_result["metadata_score"], f"(w={weights['metadata']:.0%})")

    print("  ║                                                      ║")

    bar = progress_bar(final_score)
    print(f"  ║  [{bar}] ║")
    print("  ║                                                      ║")

    print(f"  ║  📊 Final AI Probability : {final_score * 100:>5.1f}%                      ║")
    print(f"  ║  🏷️ Verdict              : {label:<28}║")
    print(f"  ║  📈 Confidence           : {confidence:<28}║")
    print("  ╠══════════════════════════════════════════════════════╣")

    fmt        = metadata_result.get('image_format', 'Unknown')
    has_camera = '✅' if metadata_result.get('has_camera') else '❌'
    print(f"  ║  📁 Format : {fmt:<12}Camera EXIF: {has_camera}{' ' * 19}║")

    print("  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  {get_ela_interpretation(ela_result['ela_score']):<52}║")
    print(f"  ║  {get_fft_interpretation(fft_result['fft_score']):<52}║")
    print(f"  ║  {get_noise_interpretation(noise_result['noise_score']):<52}║")
    print(f"  ║  {get_metadata_interpretation(metadata_result):<52}║")
    print("  ╚══════════════════════════════════════════════════════╝")

    print("\n  ── Fusion Debug ────────────────────────────────────")
    print(f"  ML Score    : {ml_score:.4f}  ×  {weights['ml']:.0%}  =  {ml_score * weights['ml']:.4f}")
    print(f"  ELA Score   : {ela_result['ela_score']:.4f}  ×  {weights['ela']:.0%}  =  {ela_result['ela_score'] * weights['ela']:.4f}")
    print(f"  FFT Score   : {fft_result['fft_score']:.4f}  ×  {weights['fft']:.0%}  =  {fft_result['fft_score'] * weights['fft']:.4f}")
    print(f"  Noise Score : {noise_result['noise_score']:.4f}  ×  {weights['noise']:.0%}  =  {noise_result['noise_score'] * weights['noise']:.4f}")
    print(f"  Metadata    : {metadata_result['metadata_score']:.4f}  ×  {weights['metadata']:.0%}  =  {metadata_result['metadata_score'] * weights['metadata']:.4f}")
    print(f"  {'─' * 44}")
    print(f"  Final Score : {final_score:.4f}  (Δ from ML: {modifier:+.4f})")

# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    args = parser.parse_args()

    analyze_image(args.image)