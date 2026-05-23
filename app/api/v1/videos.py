"""
api/v1/videos.py — Video Analysis Endpoint

POST /api/v1/analyze/video
  - Accepts video file upload (mp4, avi, mov, webm, mkv)
  - Runs frame extraction + per-frame forensics + temporal analysis
  - Returns structured VideoAnalysisResponse
"""

import os
import uuid
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import Optional, List

import sys

_APP_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_APP_DIR / "ml"))
sys.path.insert(0, str(_APP_DIR / "ml" / "image"))

from video.video_inference import analyze_video, VideoAnalysisResult


router = APIRouter(prefix="/analyze", tags=["Video Analysis"])


# ─── Config ───────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS  = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".m4v"}
MAX_FILE_SIZE_MB    = 200
DEFAULT_MAX_FRAMES  = 60
DEFAULT_INTERVAL    = 1.0


# ─── Response models ──────────────────────────────────────────────────────────

class TemporalSummaryOut(BaseModel):
    score_variance   : float
    mean_gradient    : float
    suspicious_ratio : float
    consistency_score: float
    interpretation   : str


class SuspiciousFrameOut(BaseModel):
    timestamp    : float
    frame_score  : float
    is_suspicious: bool
    is_keyframe  : bool


class VideoLayerScoresOut(BaseModel):
    frame_aggregate: float   # weighted aggregate of per-frame scores
    temporal_score : float


class VideoInfoOut(BaseModel):
    duration_sec      : float
    fps               : float
    resolution        : Optional[str]
    codec             : Optional[str]
    n_frames_analyzed : int
    processing_time_s : float


class VideoAnalysisResponse(BaseModel):
    analysis_id      : str
    media_type       : str = "video"
    final_score      : float = Field(..., ge=0, le=1)
    verdict          : str
    confidence       : str
    confidence_pct   : float
    is_ai            : bool
    layer_scores     : VideoLayerScoresOut
    temporal         : Optional[TemporalSummaryOut]
    suspicious_frames: List[SuspiciousFrameOut]
    video_info       : VideoInfoOut


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _cleanup(path: str):
    try:
        os.unlink(path)
    except OSError:
        pass


# ─── Route ────────────────────────────────────────────────────────────────────

@router.post(
    "/video",
    response_model=VideoAnalysisResponse,
    summary="Analyze a video for AI generation / deepfake",
    description="""
Runs a multi-stage forensic analysis on the uploaded video:

1. **Frame extraction** — samples frames at 1fps + scene-change keyframes
2. **Per-frame forensics** — ELA + FFT + Noise on each frame
3. **Temporal analysis** — detects flickering/inconsistency between frames (deepfake signature)
4. **Score aggregation** — weighted fusion of frame aggregate + temporal score

Ideal for detecting:
- Face-swap deepfakes (DeepFaceLab, Roop, etc.)
- AI-generated video (Sora, Runway, Pika, etc.)
- Spliced / manipulated footage
    """,
)
async def analyze_video_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file to analyze"),
    max_frames: int = Query(
        default=DEFAULT_MAX_FRAMES,
        ge=10, le=120,
        description="Max frames to analyze (10–120, default 60)",
    ),
    frame_interval: float = Query(
        default=DEFAULT_INTERVAL,
        ge=0.5, le=5.0,
        description="Seconds between uniform frame samples (0.5–5.0)",
    ),
) -> VideoAnalysisResponse:

    # ── Validate file type ────────────────────────────────────────────────
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video format '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # ── Read + validate size ──────────────────────────────────────────────
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum: {MAX_FILE_SIZE_MB}MB",
        )

    # ── Save to temp file ─────────────────────────────────────────────────
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(content)
    tmp.close()
    tmp_path = tmp.name

    background_tasks.add_task(_cleanup, tmp_path)

    # ── Run video analysis ────────────────────────────────────────────────
    try:
        result: VideoAnalysisResult = analyze_video(
            video_path     = tmp_path,
            max_frames     = max_frames,
            frame_interval = frame_interval,
            verbose        = False,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")

    # ── Build response ────────────────────────────────────────────────────
    temporal_out = None
    if result.temporal:
        temporal_out = TemporalSummaryOut(
            score_variance   = result.temporal.score_variance,
            mean_gradient    = result.temporal.mean_gradient,
            suspicious_ratio = result.temporal.suspicious_ratio,
            consistency_score= result.temporal.consistency_score,
            interpretation   = result.temporal.interpretation,
        )

    suspicious_out = [
        SuspiciousFrameOut(
            timestamp    = f["timestamp"],
            frame_score  = f["frame_score"],
            is_suspicious= True,
            is_keyframe  = f["is_keyframe"],
        )
        for f in result.suspicious_frames
    ]

    return VideoAnalysisResponse(
        analysis_id      = str(uuid.uuid4()),
        final_score      = result.final_score,
        verdict          = result.verdict,
        confidence       = result.confidence,
        confidence_pct   = round(abs(result.final_score - 0.5) * 200, 1),
        is_ai            = result.final_score >= 0.50,
        layer_scores     = VideoLayerScoresOut(
            frame_aggregate = result.frame_aggregate,
            temporal_score  = result.temporal_score,
        ),
        temporal         = temporal_out,
        suspicious_frames= suspicious_out,
        video_info       = VideoInfoOut(
            duration_sec      = result.video_meta.duration_sec,
            fps               = result.video_meta.fps,
            resolution        = f"{result.video_meta.width}×{result.video_meta.height}",
            codec             = result.video_meta.codec,
            n_frames_analyzed = result.n_frames_analyzed,
            processing_time_s = result.processing_time_s,
        ),
    )