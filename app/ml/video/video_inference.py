"""
video/video_inference.py — Full Video Deepfake Detection Pipeline

Flow:
  1. Extract frames (frame_extractor.py)
  2. Run per-frame analysis: ML (full frame + face crop) + ELA + FFT + Noise
  3. Run temporal consistency analysis (temporal_analysis.py)
  4. Fuse per-frame aggregate + temporal score → final video verdict

"""

import os
import io
import time
import sys
import tempfile
from dataclasses import dataclass

import numpy as np
from PIL import Image

# ─── Paths ────────────────────────────────────────────────────────────────────
_ML_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
_IMG_DIR = os.path.join(_ML_DIR, 'image')
sys.path.insert(0, _ML_DIR)
sys.path.insert(0, _IMG_DIR)

from video.frame_extractor import (
    extract_frames, frames_to_temp_images, cleanup_temp_files, VideoMeta, ExtractedFrame
)
from video.temporal_analysis import (
    compute_temporal_analysis, aggregate_frame_scores, TemporalResult
)

from ela import compute_ela
from fft_analysis import compute_fft
from noise_analysis import compute_noise
from inference import predict_image

import cv2

AI_STANDARD_SIZES = {512, 640, 768, 832, 896, 960, 1024, 1152,
                     1280, 1344, 1408, 1472, 1536, 1600, 1664,
                     1728, 1792, 1856, 1920, 2048}
NORMAL_FPS = {24, 25, 29.97, 30, 48, 50, 59.94, 60}

def _video_metadata_score(meta: "VideoMeta") -> float:
    score = 0.0
    w, h  = meta.width, meta.height
    if w in AI_STANDARD_SIZES and h in AI_STANDARD_SIZES:
        score += 0.60
    elif w in AI_STANDARD_SIZES or h in AI_STANDARD_SIZES:
        score += 0.30
    if not any(abs(meta.fps - n) < 0.5 for n in NORMAL_FPS):
        score += 0.25   # FPS غير طبيعي (مثل 32fps)
    return round(min(score, 0.95), 4)
# ─── Config ───────────────────────────────────────────────────────────────────

# Temperature scaling: the higher the T, the more scores are pulled toward 0.5
# T=1.0 = no change, T=2.5 = moderate, T=4.0 = aggressive
ML_TEMPERATURE = 2.5

# Final video fusion weights
WEIGHT_FRAME_AGG = 0.75   # was 0.90
WEIGHT_TEMPORAL  = 0.25   # was 0.10  ← temporal matters more now

# Per-frame ML vs forensic weights
WEIGHT_ML_IN_FRAME        = 0.55   # was 0.85
WEIGHT_FORENSIC_IN_FRAME  = 0.45   # was 0.15

# Face detector (loaded once)
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


# ─── ML Score calibration for video ──────────────────────────────────────────

def _calibrate_ml_score(prob: float, temperature: float = ML_TEMPERATURE) -> float:
    """
    Temperature scaling in logit space.

    The model was trained on still images (JPEG/PNG).
    Video frames — even from real cameras — have h264 blocking artifacts
    that the model confuses with AI generation, pushing scores toward 1.0.

    Temperature scaling pulls extreme scores back toward 0.5:
      T=2.5:  prob=0.999 → 0.83  |  prob=0.90 → 0.74  |  prob=0.50 → 0.50
              prob=0.10  → 0.26  |  prob=0.01 → 0.17
    """
    p = float(np.clip(prob, 1e-6, 1 - 1e-6))
    logit = np.log(p / (1.0 - p))
    logit_scaled = logit / temperature
    return float(1.0 / (1.0 + np.exp(-logit_scaled)))


# ─── Frame normalization (removes h264 codec artifacts) ───────────────────────

def _normalize_frame_for_ml(tmp_path: str) -> str:
    """
    Re-encode a video frame as JPEG quality=92 before passing to ML.

    Why: The model was trained on JPEG/PNG still images. Video frames saved as PNG
    preserve h264 blocking/ringing artifacts that are foreign to the model and push
    it toward predicting 'AI'. Re-encoding as JPEG simulates the same compression
    distribution the model saw during training.

    Returns a new temp path (caller must delete it).
    """
    try:
        img = Image.open(tmp_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        buf.seek(0)
        normalized = Image.open(buf).convert("RGB")

        tmp = tempfile.NamedTemporaryFile(suffix="_norm.jpg", delete=False)
        normalized.save(tmp.name, format="JPEG", quality=92)
        tmp.close()
        return tmp.name
    except Exception:
        return tmp_path  # fallback: use original if normalization fails


# ─── Face crop helper ─────────────────────────────────────────────────────────

def crop_face(tmp_path: str) -> str | None:
    try:
        img = cv2.imread(tmp_path)
        if img is None:
            return None

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = _FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60),
        )

        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad_x = int(w * 0.25)
        pad_y = int(h * 0.25)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img.shape[1], x + w + pad_x)
        y2 = min(img.shape[0], y + h + pad_y)

        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        face_path = tmp_path.replace('.png', '_face.png').replace('.jpg', '_face.jpg')
        cv2.imwrite(face_path, face_crop)
        return face_path

    except Exception:
        return None


# ─── Per-frame analysis ───────────────────────────────────────────────────────

@dataclass
class FrameAnalysisResult:
    timestamp     : float
    frame_idx     : int
    is_keyframe   : bool
    ml_score_raw  : float   # raw ML output before calibration
    ml_score      : float   # calibrated ML score
    ml_face_score : float   # face ML (-1 if no face)
    ela_score     : float
    fft_score     : float
    noise_score   : float
    face_found    : bool
    frame_score   : float


def analyze_single_frame(
    frame: ExtractedFrame,
    tmp_path: str,
) -> FrameAnalysisResult:

    # ── Normalize frame for ML (removes h264 artifacts) ───────────────────────
    norm_path = _normalize_frame_for_ml(tmp_path)
    norm_is_different = norm_path != tmp_path

    try:
        # ── Layer 1: Full-frame ML ─────────────────────────────────────────────
        try:
            ml_result   = predict_image(norm_path)
            ml_score_raw = float(ml_result["fake_probability"])
            ml_score     = _calibrate_ml_score(ml_score_raw)
        except Exception:
            ml_score_raw = 0.5
            ml_score     = 0.5

        # ── Layer 2: Face-crop ML ─────────────────────────────────────────────
        face_found    = False
        ml_face_score = -1.0

        face_path = crop_face(norm_path)
        if face_path and os.path.exists(face_path):
            try:
                face_result   = predict_image(face_path)
                raw_face      = float(face_result["fake_probability"])
                ml_face_score = _calibrate_ml_score(raw_face)
                face_found    = True
            except Exception:
                ml_face_score = -1.0
            finally:
                try:
                    os.unlink(face_path)
                except OSError:
                    pass

        # ── Layers 3-5: Forensic signals (run on original PNG — better for ELA/FFT) ──
        try:
            ela_score = float(compute_ela(tmp_path).get("ela_score", 0.5))
        except Exception:
            ela_score = 0.5

        try:
            fft_score = float(compute_fft(tmp_path).get("fft_score", 0.5))
        except Exception:
            fft_score = 0.5

        try:
            noise_score = float(compute_noise(tmp_path).get("noise_score", 0.5))
        except Exception:
            noise_score = 0.5

    finally:
        if norm_is_different:
            try:
                os.unlink(norm_path)
            except OSError:
                pass

    # ── ML combination (full frame + face) ────────────────────────────────────
    if face_found and ml_face_score >= 0:
        diff = abs(ml_score - ml_face_score)
        if diff > 0.45:
            ml_combined = min(ml_score, ml_face_score)
        else:
            ml_combined = ml_score * 0.30 + ml_face_score * 0.70
    else:
        ml_combined = ml_score

    # ── Forensic aggregate ────────────────────────────────────────────────────
    forensic_avg = (ela_score + fft_score + noise_score) / 3.0

    # ── Frame score fusion (ML 55% + Forensic 45%) ───────────────────────────
    frame_score = (
        ml_combined  * WEIGHT_ML_IN_FRAME +
        forensic_avg * WEIGHT_FORENSIC_IN_FRAME
    )

    # Small boost if multiple forensic layers strongly agree with ML
    high_forensics = sum(s >= 0.70 for s in [ela_score, fft_score, noise_score])
    if high_forensics >= 2 and ml_combined >= 0.65:
        frame_score = min(frame_score + 0.04, 1.0)

    # Dampen: if forensic layers are all "real-like", don't let ML alone dominate
    all_forensic_real = ela_score < 0.25 and fft_score < 0.25 and noise_score < 0.25
    if all_forensic_real and ml_combined >= 0.75:
        frame_score = min(frame_score, ml_combined * 0.75)

    frame_score = float(np.clip(frame_score, 0.0, 1.0))

    return FrameAnalysisResult(
        timestamp=frame.timestamp,
        frame_idx=frame.frame_idx,
        is_keyframe=frame.is_keyframe,
        ml_score_raw=round(ml_score_raw, 4),
        ml_score=round(ml_score, 4),
        ml_face_score=round(ml_face_score, 4),
        ela_score=round(ela_score, 4),
        fft_score=round(fft_score, 4),
        noise_score=round(noise_score, 4),
        face_found=face_found,
        frame_score=round(frame_score, 4),
    )


# ─── Full Video Analysis ──────────────────────────────────────────────────────

@dataclass
class VideoAnalysisResult:
    final_score       : float
    verdict           : str
    confidence        : str
    frame_aggregate   : float
    temporal_score    : float
    weights           : dict
    temporal          : TemporalResult
    frame_results     : list[FrameAnalysisResult]
    suspicious_frames : list[dict]
    n_frames_analyzed : int
    video_meta        : VideoMeta
    processing_time_s : float
    faces_detected    : int


def _get_verdict(score: float) -> tuple[str, str]:
    distance = abs(score - 0.5)

    if score >= 0.75:
        label = "🔴 Likely AI-Generated / Deepfake"
    elif score >= 0.55:
        label = "🟠 Probably AI-Generated / Deepfake"
    elif score >= 0.40:
        label = "🟡 Unclear / Mixed Signals"
    elif score >= 0.25:
        label = "🟢 Probably Real"
    else:
        label = "✅ Likely Real"

    if distance >= 0.30:
        conf = "High"
    elif distance >= 0.15:
        conf = "Medium"
    else:
        conf = "Low"

    return label, conf


def analyze_video(
    video_path: str,
    max_frames: int = 60,
    frame_interval: float = 1.0,
    verbose: bool = True,
) -> VideoAnalysisResult:

    t_start = time.time()

    if verbose:
        print(f"\n  📹 Video: {video_path}")
        print("  ⏳ Step 1/4: Extracting frames...")

    frames, meta = extract_frames(
        video_path=video_path,
        max_frames=max_frames,
        interval=frame_interval,
    )

    if verbose:
        print(f"  ✅ Extracted {len(frames)} frames from {meta.duration_sec}s video")

    tmp_paths = frames_to_temp_images(frames)

    try:

        if verbose:
            print(
                f"  ⏳ Step 2/4: Analyzing {len(frames)} frames "
                f"(ML T={ML_TEMPERATURE} + Face + ELA + FFT + Noise)..."
            )

        frame_results: list[FrameAnalysisResult] = []

        for i, (frame, tmp_path) in enumerate(zip(frames, tmp_paths)):
            result = analyze_single_frame(frame, tmp_path)
            frame_results.append(result)

            if verbose:
                face_info = (
                    f"face={result.ml_face_score:.3f}"
                    if result.face_found
                    else "no_face"
                )
                print(
                    f"    [{i+1}/{len(frames)}] "
                    f"ml_raw={result.ml_score_raw:.3f} "
                    f"ml_cal={result.ml_score:.3f} "
                    f"{face_info} "
                    f"ela={result.ela_score:.3f} "
                    f"fft={result.fft_score:.3f} "
                    f"noise={result.noise_score:.3f} "
                    f"→ frame={result.frame_score:.3f}"
                )

        if verbose:
            print("  ⏳ Step 3/4: Temporal consistency analysis...")

        raw_scores = [r.frame_score for r in frame_results]

        # Median smoothing
        smoothed_scores = []
        for i in range(len(raw_scores)):
            neighbors = raw_scores[max(0, i-1): min(len(raw_scores), i+2)]
            smoothed_scores.append(float(np.median(neighbors)))

        frame_scores = smoothed_scores

        temporal = compute_temporal_analysis(frame_scores)

        frame_aggregate = aggregate_frame_scores(frame_scores, method="mean")

        if verbose:
            print("  ⏳ Step 4/4: Score fusion...")

        # ← أضيف هون
        video_meta_score = _video_metadata_score(meta)
        if verbose:
            print(f"  📐 Video metadata score: {video_meta_score:.3f} "
                  f"({meta.width}×{meta.height} | {meta.fps}fps)")

        # Dynamic weights بناءً على قوة الـ metadata signal
        if video_meta_score >= 0.50:
            # resolution وFPS واضحين → metadata يأخذ وزن أكبر
            w_frame    = 0.50
            w_temporal = 0.20
            w_meta     = 0.30
        else:
            # metadata ضعيفة → نعتمد على frames أكثر
            w_frame    = 0.70
            w_temporal = 0.25
            w_meta     = 0.05

        final_score = (
            frame_aggregate         * w_frame    +
            temporal.temporal_score * w_temporal +
            video_meta_score        * w_meta
        )
        final_score = round(float(np.clip(final_score, 0.0, 1.0)), 4)

        verdict, confidence = _get_verdict(final_score)

        faces_detected = sum(1 for r in frame_results if r.face_found)

        suspicious_frames = [
            {
                "timestamp"    : r.timestamp,
                "frame_idx"    : r.frame_idx,
                "frame_score"  : r.frame_score,
                "ml_score_raw" : r.ml_score_raw,
                "ml_score_cal" : r.ml_score,
                "ml_face_score": r.ml_face_score,
                "face_found"   : r.face_found,
                "is_keyframe"  : r.is_keyframe,
            }
            for r in frame_results
            if r.frame_score >= 0.55
        ]

        processing_time = round(time.time() - t_start, 2)

        if verbose:
            _print_result(
                final_score, verdict, confidence,
                frame_aggregate, temporal, meta,
                processing_time, suspicious_frames,
                frame_results, faces_detected,
            )

        return VideoAnalysisResult(
            final_score=final_score,
            verdict=verdict,
            confidence=confidence,
            frame_aggregate=round(frame_aggregate, 4),
            temporal_score=temporal.temporal_score,
            weights={
                "frame_agg": WEIGHT_FRAME_AGG,
                "temporal" : WEIGHT_TEMPORAL,
            },
            temporal=temporal,
            frame_results=frame_results,
            suspicious_frames=suspicious_frames,
            n_frames_analyzed=len(frame_results),
            video_meta=meta,
            processing_time_s=processing_time,
            faces_detected=faces_detected,
        )

    finally:
        try:
            cleanup_temp_files(tmp_paths)
        except Exception:
            # prevent cleanup errors from breaking the whole request
            pass



# ─── Display ──────────────────────────────────────────────────────────────────

def _bar(score: float, length: int = 28) -> str:
    filled = int(score * length)
    return "█" * filled + "░" * (length - filled)


def _print_result(
    final_score, verdict, confidence, frame_agg,
    temporal: TemporalResult, meta: VideoMeta,
    proc_time, suspicious_frames,
    frame_results: list[FrameAnalysisResult],
    faces_detected: int,
):
    n           = len(frame_results)
    avg_ml_raw  = sum(r.ml_score_raw for r in frame_results) / n if n else 0
    avg_ml_cal  = sum(r.ml_score for r in frame_results) / n if n else 0
    face_frames = [r for r in frame_results if r.face_found]
    avg_face_ml = sum(r.ml_face_score for r in face_frames) / len(face_frames) if face_frames else 0
    avg_ela     = sum(r.ela_score for r in frame_results) / n if n else 0
    avg_fft     = sum(r.fft_score for r in frame_results) / n if n else 0
    avg_noise   = sum(r.noise_score for r in frame_results) / n if n else 0

    print("\n  ╔══════════════════════════════════════════════════════╗")
    print("  ║           Video Deepfake Analysis Results           ║")
    print("  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  📊 Final AI Probability : {final_score * 100:>5.1f}%                      ║")
    print(f"  ║  🏷️  Verdict             : {verdict:<28}║")
    print(f"  ║  📈 Confidence           : {confidence:<28}║")
    print("  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  🧠 ML raw (avg)         : {avg_ml_raw * 100:>5.1f}%                         ║")
    print(f"  ║  🧠 ML calibrated (avg)  : {avg_ml_cal * 100:>5.1f}%  (T={ML_TEMPERATURE})                ║")
    if faces_detected > 0:
        print(f"  ║  👤 Face ML (avg)        : {avg_face_ml * 100:>5.1f}%  ({faces_detected}/{n} frames)       ║")
    else:
        print(f"  ║  👤 Face Detection       : No faces found              ║")
    print(f"  ║  🔬 ELA (avg)            : {avg_ela * 100:>5.1f}%                         ║")
    print(f"  ║  📡 FFT (avg)            : {avg_fft * 100:>5.1f}%                         ║")
    print(f"  ║  🌊 Noise (avg)          : {avg_noise * 100:>5.1f}%                         ║")
    print("  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  🎬 Frame Aggregate      : {frame_agg * 100:>5.1f}%  (w={WEIGHT_FRAME_AGG:.0%})            ║")
    print(f"  ║  ⏱️  Temporal Score       : {temporal.temporal_score * 100:>5.1f}%  (w={WEIGHT_TEMPORAL:.0%})            ║")
    print("  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  [{_bar(final_score)}]     ║")
    print("  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  📹 Duration : {meta.duration_sec}s  |  Frames: {n}  |  Suspicious: {len(suspicious_frames)}       ║")
    print(f"  ║  🎞️  Codec   : {meta.codec:<8}  |  Resolution: {meta.width}×{meta.height}           ║")
    print(f"  ║  {temporal.interpretation:<52}║")
    print(f"  ║  Score variance: {temporal.score_variance:.4f}  |  Mean gradient: {temporal.mean_gradient:.4f}       ║")
    print("  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  ⏱️  Processing time: {proc_time}s                                 ║")
    print("  ╚══════════════════════════════════════════════════════╝")

    if suspicious_frames:
        print(f"\n  🚨 Top suspicious frames (ml_raw → ml_cal):")
        for f in sorted(suspicious_frames, key=lambda x: -x["frame_score"])[:5]:
            face_info = f"face={f['ml_face_score']:.3f}" if f["face_found"] else "no_face"
            print(
                f"     t={f['timestamp']:.1f}s  "
                f"frame={f['frame_score']:.3f}  "
                f"ml_raw={f['ml_score_raw']:.3f}→cal={f['ml_score_cal']:.3f}  "
                f"{face_info}  "
                f"{'[keyframe]' if f['is_keyframe'] else ''}"
            )