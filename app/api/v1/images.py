"""
api/v1/images.py — Image Analysis Endpoint

POST /api/v1/analyze/image
  - Accepts image file upload
  - Runs 5-layer forensic analysis
  - Returns structured AnalysisResponse
"""
import os
import uuid
import tempfile
from pathlib import Path

# ✅ هاد السطر ضروري
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

# ── ML imports ──────────────────────────────────────────────────────────────
import sys

_APP_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_APP_DIR / "ml"))
sys.path.insert(0, str(_APP_DIR / "ml" / "image"))

from inference import (
    predict_image,
    compute_final_score,
)
from ela import compute_ela
from fft_analysis import compute_fft
from noise_analysis import compute_noise
from metadata_analysis import analyze_metadata


router = APIRouter(prefix="/analyze", tags=["Image Analysis"])

# ─── Allowed file types ───────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_FILE_SIZE_MB   = 20


# ─── Response models ──────────────────────────────────────────────────────────

class LayerScoresOut(BaseModel):
    ml_score      : float = Field(..., ge=0, le=1, description="EfficientNet probability")
    ela_score     : float = Field(..., ge=0, le=1)
    fft_score     : float = Field(..., ge=0, le=1)
    noise_score   : float = Field(..., ge=0, le=1)
    metadata_score: float = Field(..., ge=0, le=1)


class FusionWeightsOut(BaseModel):
    ml      : float
    ela     : float
    fft     : float
    noise   : float
    metadata: float


class MetadataOut(BaseModel):
    image_format    : Optional[str]
    image_size      : Optional[str]
    has_camera_exif : bool
    ai_keyword      : Optional[str]
    ai_dimension    : bool
    software        : Optional[str]


class ImageAnalysisResponse(BaseModel):
    analysis_id    : str
    media_type     : str = "image"
    final_score    : float = Field(..., ge=0, le=1, description="AI probability 0→1")
    verdict        : str
    confidence     : str
    confidence_pct : float
    is_ai          : bool
    modifier       : float   # delta from raw ML score to final score
    layer_scores   : LayerScoresOut
    fusion_weights : FusionWeightsOut
    metadata       : MetadataOut


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_verdict(score: float) -> tuple[str, str]:
    if score >= 0.85: label = "Likely AI-Generated"
    elif score >= 0.65: label = "Probably AI-Generated"
    elif score >= 0.45: label = "Unclear / Mixed Signals"
    elif score >= 0.25: label = "Probably Real"
    else: label = "Likely Real"

    dist = abs(score - 0.5)
    conf = "High" if dist >= 0.40 else ("Medium" if dist >= 0.22 else "Low")
    return label, conf


def _cleanup(path: str):
    try:
        os.unlink(path)
    except OSError:
        pass


# ─── Route ────────────────────────────────────────────────────────────────────

@router.post(
    "/image",
    response_model=ImageAnalysisResponse,
    summary="Analyze an image for AI generation / deepfake",
    description="""
Runs a 5-layer forensic analysis on the uploaded image:

1. **ML** — EfficientNet-B0 trained on GenImage dataset
2. **ELA** — Error Level Analysis (compression fingerprinting)
3. **FFT** — Frequency domain artifact detection
4. **Noise** — Sensor noise pattern analysis
5. **Metadata** — EXIF, AI keywords, dimension analysis

Returns a fused probability score with dynamic weights based on evidence strength.
    """,
)
async def analyze_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Image file to analyze"),
) -> ImageAnalysisResponse:

    # ── Validate file type ────────────────────────────────────────────────
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # ── Read file content ─────────────────────────────────────────────────
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB",
        )

    # ── Save to temp file ─────────────────────────────────────────────────
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(content)
    tmp.close()
    tmp_path = tmp.name

    # Ensure cleanup even if analysis fails
    background_tasks.add_task(_cleanup, tmp_path)

    # ── Run analysis ──────────────────────────────────────────────────────
    try:
        ml_result       = predict_image(tmp_path)
        ela_result      = compute_ela(tmp_path)
        fft_result      = compute_fft(tmp_path)
        noise_result    = compute_noise(tmp_path)
        metadata_result = analyze_metadata(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    ml_score = float(ml_result["fake_probability"])

    # ── Score fusion ──────────────────────────────────────────────────────
    final_score, modifier, weights = compute_final_score(
        ml_score       = ml_score,
        ela_result     = ela_result,
        fft_result     = fft_result,
        noise_result   = noise_result,
        metadata_result= metadata_result,
    )

    verdict, confidence = _get_verdict(final_score)

    return ImageAnalysisResponse(
        analysis_id    = str(uuid.uuid4()),
        final_score    = final_score,
        verdict        = verdict,
        confidence     = confidence,
        confidence_pct = round(abs(final_score - 0.5) * 200, 1),
        is_ai          = final_score >= 0.50,
        modifier       = modifier,
        layer_scores   = LayerScoresOut(
            ml_score       = round(ml_score, 4),
            ela_score      = ela_result.get("ela_score", 0.5),
            fft_score      = fft_result.get("fft_score", 0.5),
            noise_score    = noise_result.get("noise_score", 0.5),
            metadata_score = metadata_result.get("metadata_score", 0.5),
        ),
        fusion_weights = FusionWeightsOut(
            ml       = weights["ml"],
            ela      = weights["ela"],
            fft      = weights["fft"],
            noise    = weights["noise"],
            metadata = weights["metadata"],
        ),
        metadata = MetadataOut(
            image_format    = metadata_result.get("image_format"),
            image_size      = metadata_result.get("image_size"),
            has_camera_exif = metadata_result.get("has_camera", False),
            ai_keyword      = metadata_result.get("ai_keyword"),
            ai_dimension    = metadata_result.get("ai_dimension", False),
            software        = metadata_result.get("software"),
        ),
    )