"""
ela.py — Error Level Analysis
بيحلل بصمة الضغط بالصورة لكشف التلاعب أو التوليد الاصطناعي
"""

import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io


def compute_ela(image_path: str, quality: int = 90) -> dict:
    """
    يحسب ELA Score للصورة
    
    الفكرة:
    - نضغط الصورة بجودة معروفة (90%)
    - نطرح النسخة المضغوطة من الأصلية
    - الفرق = Error Level
    - صور AI → error موزّع بشكل موحّد (مشبوه)
    - صور حقيقية → error متفاوت (طبيعي)
    """
    try:
        # تحميل الصورة الأصلية
        original = Image.open(image_path).convert("RGB")
        
        # ضغط وفك ضغط بجودة محددة
        buffer = io.BytesIO()
        original.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")
        
        # حساب الفرق
        ela_image = ImageChops.difference(original, compressed)
        
        # تحويل لـ numpy
        ela_array = np.array(ela_image, dtype=np.float32)
        
        # ── المقاييس ──────────────────────────────────────────────────────
        
        # 1. متوسط الـ error
        mean_error = float(ela_array.mean())
        
        # 2. الانحراف المعياري — AI بيكون أقل (error موحّد)
        std_error = float(ela_array.std())
        
        # 3. أعلى error — مناطق التلاعب
        max_error = float(ela_array.max())
        
        # 4. نسبة البكسلات يلي فيها error عالي
        high_error_ratio = float((ela_array > 10).mean())
        
        # ── حساب الـ Score ────────────────────────────────────────────────
        # AI → mean_error عالي + std_error منخفض (موحّد)
        # Real → mean_error أقل + std_error أعلى (متفاوت)
        
        # Uniformity = كيف الـ error موحّد (كلما عالي → أكثر شك)
        uniformity = mean_error / (std_error + 1e-6)
        
        # نحوّل لـ score بين 0 و 1
        # بناءً على مشاهدات: AI عادة uniformity > 3
        ela_score = float(np.clip(uniformity / 10.0, 0.0, 1.0))
        
        return {
            "ela_score"        : round(ela_score, 4),
            "mean_error"       : round(mean_error, 4),
            "std_error"        : round(std_error, 4),
            "max_error"        : round(max_error, 4),
            "high_error_ratio" : round(high_error_ratio, 4),
            "uniformity"       : round(uniformity, 4),
            "success"          : True,
        }

    except Exception as e:
        return {
            "ela_score" : 0.5,  # قيمة محايدة عند الفشل
            "error"     : str(e),
            "success"   : False,
        }


def get_ela_interpretation(ela_score: float) -> str:
    """Interpret the ELA score and return a human-readable label."""
    if ela_score >= 0.7:
        return "🔴 ELA: Highly suspicious compression fingerprint"
    elif ela_score >= 0.5:
        return "🟡 ELA: Suspicious compression fingerprint"
    elif ela_score >= 0.3:
        return "🟠 ELA: Slightly abnormal compression fingerprint"
    else:
        return "🟢 ELA: Normal compression fingerprint"