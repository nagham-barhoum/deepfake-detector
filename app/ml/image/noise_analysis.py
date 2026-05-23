"""
noise_analysis.py — Noise Pattern Analysis
AI-generated images are unnaturally smooth compared to real camera photos,
which always carry sensor noise (shot noise, read noise, pattern noise).
"""

import numpy as np
from PIL import Image
from scipy import ndimage


def compute_noise(image_path: str) -> dict:
    """
    Analyzes the noise pattern of an image to detect AI generation.

    Intuition:
    - Real cameras produce natural noise: Gaussian, spatially random,
      and non-uniform across regions (varies with ISO, exposure, etc.)
    - AI-generated images are either nearly noiseless or have
      unnaturally uniform, regular noise from the denoiser.

    Indicators:
    1. Overall noise level    — AI images have very low noise std
    2. Local variance CoV     — AI images have unnaturally uniform variance
    3. Laplacian variance     — Measures edge sharpness / texture richness
    """
    try:
        img = Image.open(image_path).convert("L")
        arr = np.array(img, dtype=np.float32)

        # ── 1. Extract noise via high-pass filter ─────────────────────────
        # Subtract Gaussian-blurred version -> residual is the noise component
        blurred   = ndimage.gaussian_filter(arr, sigma=1.5)
        noise_map = arr - blurred

        # ── 2. Global noise statistics ─────────────────────────────────────
        noise_std  = float(noise_map.std())
        noise_mean = float(np.abs(noise_map).mean())

        # ── 3. Local variance map (patch-based) ───────────────────────────
        # Divide image into patches and compute variance per patch.
        # Real images: high variance between patches (CoV > 0.6)
        # AI images:   uniform variance across patches (CoV < 0.5)
        patch_size = 16
        h, w = arr.shape
        variances = []
        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch = noise_map[y:y + patch_size, x:x + patch_size]
                variances.append(float(patch.var()))

        variances = np.array(variances)
        var_mean  = float(variances.mean())
        var_std   = float(variances.std())

        # Coefficient of Variation: measures spatial non-uniformity of noise
        # Real: CoV ~ 0.6-1.5 | AI: CoV ~ 0.1-0.5
        cov = var_std / (var_mean + 1e-8)

        # ── 4. Laplacian variance — edge / texture richness ───────────────
        laplacian = ndimage.laplace(arr)
        lap_var   = float(laplacian.var())

        # ── Final Score Calculation ───────────────────────────────────────
        # AI signatures:
        #   noise_std -> very low (< 5 is suspicious, < 3 is strong signal)
        #   cov       -> low     (< 0.5 means unnaturally uniform noise)
        #
        # Reference ranges (empirical):
        #   Real: noise_std ~ 6-20,  CoV ~ 0.6-1.5
        #   AI:   noise_std ~ 1-6,   CoV ~ 0.1-0.5

        noise_signal      = float(np.clip((8.0 - noise_std) / 8.0, 0, 1))
        uniformity_signal = float(np.clip((0.6 - cov) / 0.6, 0, 1))

        noise_score = float(np.clip(
            noise_signal * 0.55 + uniformity_signal * 0.45,
            0.0, 1.0
        ))

        return {
            "noise_score" : round(noise_score, 4),
            "noise_std"   : round(noise_std, 4),
            "noise_mean"  : round(noise_mean, 4),
            "var_cov"     : round(cov, 4),
            "lap_variance": round(lap_var, 2),
            "success"     : True,
        }

    except Exception as e:
        return {
            "noise_score": 0.5,
            "error"      : str(e),
            "success"    : False,
        }


def get_noise_interpretation(noise_score: float) -> str:
    if noise_score >= 0.70:
        return "🔴 Noise: Unnaturally clean — strong AI signature"
    elif noise_score >= 0.50:
        return "🟡 Noise: Abnormal noise pattern detected"
    elif noise_score >= 0.30:
        return "🟠 Noise: Slightly unusual noise characteristics"
    else:
        return "🟢 Noise: Natural camera-like noise pattern"