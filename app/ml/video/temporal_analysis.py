"""
video/temporal_analysis.py — Temporal Consistency Analysis

Why temporal analysis matters for video deepfakes:
  - Face-swap tools (DeepFaceLab, FaceSwap, Roop) process FRAMES INDEPENDENTLY.
    This causes subtle score flickering across frames that real videos don't have.
  - Real videos have smooth optical flow and consistent noise profiles.
  - AI-generated video (Sora, Runway, Pika) introduces periodic texture artifacts
    that manifest as score spikes at frame boundaries.

Signals we compute:
  1. Score variance         — high variance across frames = flickering = AI indicator
  2. Temporal gradient      — frame-to-frame score change (abrupt jumps = splice)
  3. Score drift            — systematic score increase over time (GAN blending)
  4. Suspicious frame ratio — % of frames above high-confidence AI threshold
  5. Consistency score      — inverse of overall temporal instability
"""

import numpy as np
from dataclasses import dataclass


# ─── Thresholds ──────────────────────────────────────────────────────────────

HIGH_AI_THRESHOLD     = 0.70   # frame score above this = "suspicious frame"
VARIANCE_NORMAL_MAX   = 0.02   # real video: score variance usually < 0.02
GRADIENT_SPIKE_THRESH = 0.25   # abrupt change > 25% between frames = splice


@dataclass
class TemporalResult:
    temporal_score      : float   # 0=real, 1=AI, based on temporal patterns
    score_variance      : float   # variance of per-frame scores
    mean_gradient       : float   # average frame-to-frame score jump
    max_gradient        : float   # largest single frame jump
    suspicious_ratio    : float   # % of frames above HIGH_AI_THRESHOLD
    score_drift         : float   # linear regression slope (drift over time)
    consistency_score   : float   # 1 - instability (inverse metric for display)
    frame_scores        : list[float]
    n_frames            : int
    interpretation      : str


def compute_temporal_analysis(frame_scores: list[float]) -> TemporalResult:
    """
    Analyze temporal patterns across per-frame AI probability scores.

    Args:
        frame_scores: list of per-frame AI probability (0→1), in time order

    Returns:
        TemporalResult with temporal_score and diagnostic metrics
    """

    if len(frame_scores) < 2:
        # Can't do temporal analysis with a single frame
        score = frame_scores[0] if frame_scores else 0.5
        return TemporalResult(
            temporal_score    = score,
            score_variance    = 0.0,
            mean_gradient     = 0.0,
            max_gradient      = 0.0,
            suspicious_ratio  = float(score >= HIGH_AI_THRESHOLD),
            score_drift       = 0.0,
            consistency_score = 1.0 - score,
            frame_scores      = frame_scores,
            n_frames          = len(frame_scores),
            interpretation    = "⚠️ Too few frames for temporal analysis",
        )

    scores = np.array(frame_scores, dtype=float)
    n      = len(scores)

    # ── 1. Score variance ─────────────────────────────────────────────────────
    score_variance = float(scores.var())

    # ── 2. Frame-to-frame gradients ───────────────────────────────────────────
    gradients     = np.abs(np.diff(scores))
    mean_gradient = float(gradients.mean())
    max_gradient  = float(gradients.max())

    n_spikes = int((gradients > GRADIENT_SPIKE_THRESH).sum())

    # ── 3. Suspicious frame ratio ─────────────────────────────────────────────
    suspicious_ratio = float((scores >= HIGH_AI_THRESHOLD).mean())

    # ── 4. Score drift (linear regression) ───────────────────────────────────
    x          = np.arange(n, dtype=float)
    slope      = float(np.polyfit(x, scores, 1)[0])
    score_drift= slope * n   # total drift across all frames

    # ── 5. Temporal score ────────────────────────────────────────────────────
    # AI deepfakes → high variance, high gradients, many suspicious frames
    # Real videos → low variance, smooth progression, few AI-flagged frames
    mean_score = float(scores.mean())
    
    if score_variance < 0.001 and mean_score > 0.40:
        # ناعم بشكل غير طبيعي + ML شايفه AI → مشبوه جداً
        smoothness_signal = float(np.clip(mean_score * 1.5, 0, 1))
    elif score_variance < 0.005 and mean_score > 0.35:
        smoothness_signal = float(np.clip(mean_score * 1.2, 0, 1))
    else:
        smoothness_signal = 0.0
    
    variance_signal   = np.clip(score_variance / 0.06, 0, 1)
    gradient_signal   = np.clip(mean_gradient / 0.15, 0, 1)
    suspicious_signal = float(suspicious_ratio)
    spike_signal      = np.clip(n_spikes / max(n * 0.2, 1), 0, 1)
    
    temporal_score = float(np.clip(
        smoothness_signal * 0.40 +   # ← الأهم للـ AI الحديث
        variance_signal   * 0.15 +
        gradient_signal   * 0.10 +
        suspicious_signal * 0.25 +
        spike_signal      * 0.10,
        0.0, 1.0
    ))

    # Consistency: how stable and "real" the temporal pattern is
    consistency_score = float(np.clip(1.0 - (variance_signal * 0.5 + gradient_signal * 0.5), 0, 1))

    # ── Interpretation ────────────────────────────────────────────────────────
    if temporal_score >= 0.70:
        interpretation = "🔴 Temporal: Strong flickering — frame-by-frame deepfake signature"
    elif temporal_score >= 0.50:
        interpretation = "🟡 Temporal: Suspicious temporal inconsistency across frames"
    elif temporal_score >= 0.30:
        interpretation = "🟠 Temporal: Slightly irregular temporal pattern"
    else:
        interpretation = "🟢 Temporal: Consistent frame progression — looks real"

    return TemporalResult(
        temporal_score   = round(temporal_score, 4),
        score_variance   = round(score_variance, 4),
        mean_gradient    = round(mean_gradient, 4),
        max_gradient     = round(max_gradient, 4),
        suspicious_ratio = round(suspicious_ratio, 4),
        score_drift      = round(score_drift, 4),
        consistency_score= round(consistency_score, 4),
        frame_scores     = [round(s, 4) for s in scores.tolist()],
        n_frames         = n,
        interpretation   = interpretation,
    )


def aggregate_frame_scores(
    frame_scores: list[float],
    method: str = "weighted",
) -> float:
    """
    Aggregate per-frame scores into a single video-level AI probability.

    Methods:
        weighted   — emphasizes high-scoring frames (conservative, fewer false positives)
        mean       — simple average
        max_pool   — top-10% average (catches even brief deepfake segments)
    """

    if not frame_scores:
        return 0.5

    scores = np.array(frame_scores)

    if method == "mean":
        return float(scores.mean())

    elif method == "max_pool":
        # Average of top 10% of frames
        k = max(1, int(len(scores) * 0.10))
        return float(np.sort(scores)[-k:].mean())

    else:  # weighted — default
        # Higher-scoring frames get more weight
        weights = scores ** 2 + 0.01   # square to amplify high scores
        weights /= weights.sum()
        return float((scores * weights).sum())