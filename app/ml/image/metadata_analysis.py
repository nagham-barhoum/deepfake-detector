"""
metadata_analysis.py — Metadata & EXIF Analysis (v2)

التحسينات عن v1:
  - لا نعتمد فقط على غياب EXIF (كل صور الإنترنت بدون EXIF)
  - نضيف تحليل الأبعاد: AI tools تنتج أبعاد محددة (512, 768, 1024...)
  - نضيف كشف Bing/Google thumbnails (474px = thumbnail حقيقي)
  - نضيف compression ratio كمؤشر إضافي
"""

import os
import numpy as np
from PIL import Image
import piexif

# ── AI tool keywords in metadata ──────────────────────────────────────────────
AI_KEYWORDS = [
    "stable diffusion", "midjourney", "dall-e", "dall·e",
    "generative", "generated", "ai-generated", "artificial",
    "diffusion", "comfyui", "automatic1111", "invokeai",
    "canva", "canva.com",
    "adobe firefly", "firefly", "ideogram", "leonardo",
    "nightcafe", "dreamstudio", "runwayml", "bing image creator",
    "stylegan", "biggan", "vqdm", "wukong",
]

# ── Standard AI output dimensions ─────────────────────────────────────────────
# Stable Diffusion, SDXL, Midjourney, DALL-E كلهم يستخدموا هاد الأبعاد
AI_STANDARD_SIZES = {512, 640, 768, 832, 896, 960, 1024, 1152, 1280, 1344,
                     1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 2048}

# ── Known thumbnail widths from image search engines ──────────────────────────
# هاد ما يعني AI — يعني thumbnail حقيقي من محرك بحث
THUMBNAIL_WIDTHS = {474, 480, 320, 400, 300, 640, 160, 200,50, 64, 75, 100, 128} 


def _is_ai_dimension(w: int, h: int) -> tuple[bool, str]:
    """
    هل الأبعاد تطابق output نماذج AI؟

    قواعد:
      1. أي بُعد في قائمة AI_STANDARD_SIZES → قوي
      2. كلا البُعدين قابلَين للقسمة على 64 → متوسط (AI tools تستخدم multiples of 64)
      3. الصورة مربعة وصغيرة (≤ 1024) → مشبوه
    """
    w_ai = w in AI_STANDARD_SIZES
    h_ai = h in AI_STANDARD_SIZES

    if w_ai and h_ai:
        return True, f"both dims are standard AI sizes ({w}×{h})"
    if w_ai or h_ai:
        return True, f"one dim is standard AI size ({w}×{h})"
    if w % 64 == 0 and h % 64 == 0 and w <= 2048 and h <= 2048:
        return True, f"both dims divisible by 64 ({w}×{h})"

    return False, ""


def _is_thumbnail(w: int, h: int) -> bool:
    """
    هل الصورة thumbnail من محرك بحث؟
    → حقيقية على الأرجح، مش AI
    """
    return w in THUMBNAIL_WIDTHS or h in THUMBNAIL_WIDTHS


def analyze_metadata(image_path: str) -> dict:
    """
    يحلل الـ metadata للكشف عن صور AI.

    الفرق عن v1:
      - نستخدم أبعاد الصورة كمؤشر (AI sizes vs real/thumbnail sizes)
      - ما نعاقب بشدة على غياب EXIF (معظم صور الإنترنت بدون EXIF)
      - نعطي bonus للـ thumbnails المعروفة (أدلة على حقيقيتها)
    """
    try:
        img = Image.open(image_path)
        fmt = img.format
        w, h = img.size
        file_size = os.path.getsize(image_path)

        flags = {
            "has_exif"        : False,
            "has_camera_make" : False,
            "has_camera_model": False,
            "has_gps"         : False,
            "has_datetime"    : False,
            "software"        : None,
            "ai_keyword_found": False,
            "ai_keyword"      : None,
            "is_png"          : fmt == "PNG",
            "is_thumbnail"    : _is_thumbnail(w, h),
            "ai_dimension"    : False,
            "ai_dim_reason"   : "",
        }

        raw_strings = []

        # ── 1. EXIF ───────────────────────────────────────────────────────────
        try:
            exif_bytes = img.info.get("exif")
            if exif_bytes:
                exif_dict = piexif.load(exif_bytes)
                flags["has_exif"] = True
                ifd0 = exif_dict.get("0th", {})

                make = ifd0.get(piexif.ImageIFD.Make, b"")
                if make:
                    make_str = make.decode("utf-8", errors="ignore").strip()
                    flags["has_camera_make"] = bool(make_str)
                    raw_strings.append(make_str.lower())

                model_tag = ifd0.get(piexif.ImageIFD.Model, b"")
                if model_tag:
                    model_str = model_tag.decode("utf-8", errors="ignore").strip()
                    flags["has_camera_model"] = bool(model_str)
                    raw_strings.append(model_str.lower())

                software = ifd0.get(piexif.ImageIFD.Software, b"")
                if software:
                    sw_str = software.decode("utf-8", errors="ignore").strip()
                    flags["software"] = sw_str
                    raw_strings.append(sw_str.lower())

                dt = ifd0.get(piexif.ImageIFD.DateTime, b"")
                flags["has_datetime"] = bool(dt)

                gps = exif_dict.get("GPS", {})
                flags["has_gps"] = bool(gps)

        except Exception:
            pass

        # ── 2. PNG chunks (SD embeds prompts here) ────────────────────────────
        if fmt == "PNG":
            for key, val in img.info.items():
                if isinstance(val, str):
                    raw_strings.append(val.lower())
                elif isinstance(val, bytes):
                    raw_strings.append(val.decode("utf-8", errors="ignore").lower())

        # ── 3. AI keyword scan ────────────────────────────────────────────────
        combined_text = " ".join(raw_strings)
        for keyword in AI_KEYWORDS:
            if keyword in combined_text:
                flags["ai_keyword_found"] = True
                flags["ai_keyword"]       = keyword
                break

        # ── 4. Dimension analysis ─────────────────────────────────────────────
        ai_dim, ai_dim_reason = _is_ai_dimension(w, h)
        flags["ai_dimension"]  = ai_dim
        flags["ai_dim_reason"] = ai_dim_reason

        # ── Score Calculation ─────────────────────────────────────────────────
        score = 0.0

        # الحالة 1: اسم أداة AI في الـ metadata → شبه مؤكد
        if flags["ai_keyword_found"]:
            score = 0.95

        else:
            # ── أدلة تدل على AI ───────────────────────────────────────────────

            # PNG بدون كاميرا → AI tools تحب PNG
            if flags["is_png"] and not flags["has_camera_make"]:
                score += 0.35

            # أبعاد AI قياسية (512, 1024, multiples of 64...)
            if flags["ai_dimension"] and not flags["is_thumbnail"]:
                score += 0.30

            # Software موجود بس بدون كاميرا (Photoshop, Canva...)
            sw = (flags["software"] or "").lower()
            if sw and not flags["has_camera_make"]:
                score += 0.15

            # EXIF موجود بس بدون كاميرا (edited/exported)
            if flags["has_exif"] and not flags["has_camera_make"]:
                score += 0.15

            # JPEG بدون EXIF خالص → مشبوه بشكل خفيف فقط
            # (كثير من الصور الحقيقية بتفقد EXIF عند المشاركة)
            if not flags["has_exif"] and not flags["is_png"]:
                score += 0.10

            # ── أدلة تدل على حقيقية → نخفض الـ score ──────────────────────

            # كاميرا حقيقية → قوي جداً
            if flags["has_camera_make"]:
                score -= 0.40

            # GPS → مستحيل تقريباً في AI
            if flags["has_gps"]:
                score -= 0.20

            # Thumbnail من محرك بحث → الصورة من الإنترنت، مش AI output
            if flags["is_thumbnail"]:
                score -= 0.15

            score = float(np.clip(score, 0.0, 0.90))

        return {
            "metadata_score"  : round(score, 4),
            "has_camera"      : flags["has_camera_make"] and flags["has_camera_model"],
            "has_gps"         : flags["has_gps"],
            "has_datetime"    : flags["has_datetime"],
            "software"        : flags["software"],
            "ai_keyword_found": flags["ai_keyword_found"],
            "ai_keyword"      : flags["ai_keyword"],
            "ai_dimension"    : flags["ai_dimension"],
            "ai_dim_reason"   : flags["ai_dim_reason"],
            "is_thumbnail"    : flags["is_thumbnail"],
            "image_format"    : fmt,
            "image_size"      : f"{w}×{h}",
            "has_exif"        : flags["has_exif"],
            "success"         : True,
        }

    except Exception as e:
        return {
            "metadata_score": 0.4,
            "error"         : str(e),
            "success"       : False,
        }


def get_metadata_interpretation(result: dict) -> str:
    score   = result.get("metadata_score", 0)
    keyword = result.get("ai_keyword")
    sw      = result.get("software", "") or ""
    ai_dim  = result.get("ai_dimension", False)
    is_thumb = result.get("is_thumbnail", False)

    if keyword:
        return f"🔴 Metadata: AI tool '{keyword}' detected"
    elif sw and not result.get("has_camera"):
        return f"🔴 Metadata: Software='{sw}' with no camera data"
    elif ai_dim and not is_thumb:
        return f"🟠 Metadata: Standard AI output dimensions detected"
    elif result.get("has_camera"):
        return "🟢 Metadata: Real camera EXIF present"
    elif is_thumb:
        return "🟢 Metadata: Image search thumbnail (likely real)"
    elif score >= 0.50:
        return "🟡 Metadata: Missing camera data — suspicious"
    elif score >= 0.20:
        return "🟠 Metadata: Inconclusive — no camera info"
    else:
        return "🟢 Metadata: No AI indicators found"