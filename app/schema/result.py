"""
schemas/result.py — Unified Detection Result Schema

Single source of truth for how analysis results are structured.
Used by both the API response and the DB model.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json


# ─── Layer scores (shared by image and video) ─────────────────────────────────

@dataclass
class LayerScores:
    ml_score       : Optional[float] = None   # None for video (no single-image ML pass)
    ela_score      : float = 0.5
    fft_score      : float = 0.5
    noise_score    : float = 0.5
    metadata_score : float = 0.5
    temporal_score : Optional[float] = None   # None for images


@dataclass
class FusionWeights:
    ml       : Optional[float] = None
    ela      : float = 0.0
    fft      : float = 0.0
    noise    : float = 0.0
    metadata : float = 0.0
    temporal : Optional[float] = None


# ─── Verdict ──────────────────────────────────────────────────────────────────

@dataclass
class Verdict:
    label      : str    # e.g. "🔴 Likely AI-Generated"
    confidence : str    # "High" | "Medium" | "Low"
    final_score: float  # 0.0 → 1.0

    @property
    def is_ai(self) -> bool:
        return self.final_score >= 0.50

    @property
    def confidence_pct(self) -> float:
        """Distance from 0.5, scaled to 0–100%."""
        return round(abs(self.final_score - 0.5) * 200, 1)


# ─── Image result ─────────────────────────────────────────────────────────────

@dataclass
class ImageAnalysisResult:
    verdict        : Verdict
    layer_scores   : LayerScores
    fusion_weights : FusionWeights
    modifier       : float           # delta from raw ML score to final_score
    media_type     : str = "image"

    # Metadata details (optional, for frontend display)
    image_format   : Optional[str] = None
    image_size     : Optional[str] = None
    has_camera_exif: bool = False
    ai_keyword     : Optional[str] = None
    ai_dimension   : bool = False

    def to_dict(self) -> dict:
        return {
            "media_type"     : self.media_type,
            "final_score"    : self.verdict.final_score,
            "verdict"        : self.verdict.label,
            "confidence"     : self.verdict.confidence,
            "confidence_pct" : self.verdict.confidence_pct,
            "is_ai"          : self.verdict.is_ai,
            "layer_scores"   : asdict(self.layer_scores),
            "fusion_weights" : asdict(self.fusion_weights),
            "modifier"       : self.modifier,
            "metadata"       : {
                "format"         : self.image_format,
                "size"           : self.image_size,
                "has_camera_exif": self.has_camera_exif,
                "ai_keyword"     : self.ai_keyword,
                "ai_dimension"   : self.ai_dimension,
            },
        }


# ─── Video result ─────────────────────────────────────────────────────────────

@dataclass
class PerFrameSummary:
    timestamp    : float
    frame_score  : float
    is_suspicious: bool
    is_keyframe  : bool


@dataclass
class VideoTemporalSummary:
    score_variance   : float
    mean_gradient    : float
    suspicious_ratio : float
    consistency_score: float
    interpretation   : str


@dataclass
class VideoAnalysisResult:
    verdict          : Verdict
    layer_scores     : LayerScores
    fusion_weights   : FusionWeights
    media_type       : str = "video"

    # Video-specific
    frame_aggregate  : float = 0.5
    n_frames         : int = 0
    suspicious_frames: list[PerFrameSummary] = field(default_factory=list)
    temporal         : Optional[VideoTemporalSummary] = None

    # Video file info
    duration_sec     : float = 0.0
    fps              : float = 0.0
    resolution       : Optional[str] = None
    codec            : Optional[str] = None
    processing_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "media_type"      : self.media_type,
            "final_score"     : self.verdict.final_score,
            "verdict"         : self.verdict.label,
            "confidence"      : self.verdict.confidence,
            "confidence_pct"  : self.verdict.confidence_pct,
            "is_ai"           : self.verdict.is_ai,
            "layer_scores"    : asdict(self.layer_scores),
            "fusion_weights"  : asdict(self.fusion_weights),
            "frame_aggregate" : self.frame_aggregate,
            "n_frames"        : self.n_frames,
            "suspicious_frames": [
                {
                    "timestamp"   : f.timestamp,
                    "frame_score" : f.frame_score,
                    "is_suspicious": f.is_suspicious,
                    "is_keyframe" : f.is_keyframe,
                }
                for f in self.suspicious_frames
            ],
            "temporal"        : asdict(self.temporal) if self.temporal else None,
            "video_info"      : {
                "duration_sec"  : self.duration_sec,
                "fps"           : self.fps,
                "resolution"    : self.resolution,
                "codec"         : self.codec,
                "n_frames_analyzed": self.n_frames,
                "processing_time_s": self.processing_time_s,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)