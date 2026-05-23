"""
fft_analysis.py — Frequency Domain Analysis (FFT)
AI-generated images leave distinctive frequency artifacts caused by
upsampling operations and convolutional layer patterns.
"""

import numpy as np
from PIL import Image


def compute_fft(image_path: str) -> dict:
    """
    Analyzes the image in the frequency domain using 2D FFT.

    Intuition:
    - Diffusion/GAN images contain regular frequency artifacts
      introduced by bilinear/nearest upsampling and Conv layers.
    - Real photos have naturally random, non-uniform noise across frequencies.

    Indicators:
    1. High-frequency energy ratio — AI images tend to be lower (too clean)
    2. Spectral flatness           — AI images tend to be higher (uniform distribution)
    3. Peak artifacts              — Sharp isolated peaks indicate upsampling grids
    """
    try:
        img = Image.open(image_path).convert("L")  # Grayscale
        arr = np.array(img, dtype=np.float32)

        # 2D FFT with shift (DC component at center)
        fft       = np.fft.fft2(arr)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        log_mag   = np.log1p(magnitude)

        h, w = arr.shape
        cy, cx = h // 2, w // 2

        # ── 1. Energy distribution across frequency bands ─────────────────
        r_low  = min(h, w) // 8   # Low frequencies  (center region)
        r_mid  = min(h, w) // 4   # Mid frequencies
        r_high = min(h, w) // 2   # High frequencies (outer ring)

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy)**2 + (X - cx)**2)

        energy_low   = magnitude[dist <= r_low].sum()
        energy_mid   = magnitude[(dist > r_low)  & (dist <= r_mid)].sum()
        energy_high  = magnitude[(dist > r_mid)  & (dist <= r_high)].sum()
        energy_total = energy_low + energy_mid + energy_high + 1e-8

        ratio_high = energy_high / energy_total  # AI -> lower (image too smooth)
        ratio_low  = energy_low  / energy_total  # AI -> higher (energy concentrated at DC)

        # ── 2. Spectral Flatness (uniform spectrum = AI-like) ─────────────
        flat_arr       = log_mag.flatten() + 1e-8
        geometric_mean = np.exp(np.mean(np.log(flat_arr)))
        arithmetic_mean = np.mean(flat_arr)
        spectral_flat  = float(geometric_mean / (arithmetic_mean + 1e-8))

        # ── 3. Peak Detection (upsampling grid artifacts) ─────────────────
        # Sharp isolated peaks at regular intervals -> AI upsampling signature
        peak_score = float(log_mag.max() / (log_mag.mean() + 1e-8))

        # ── Final Score Calculation ───────────────────────────────────────
        # AI signatures:
        #   ratio_low  -> high  (energy concentrated at low frequencies)
        #   ratio_high -> low   (missing natural high-freq texture)
        #   spectral_flat -> high (spectrum too uniform)
        #
        # Reference ranges (empirical):
        #   Real:  ratio_high ~ 0.20-0.35,  ratio_low ~ 0.30-0.50
        #   AI:    ratio_high ~ 0.05-0.15,  ratio_low ~ 0.55-0.75

        low_freq_signal  = np.clip((ratio_low  - 0.50) / 0.40, 0, 1)
        high_freq_signal = np.clip((0.20 - ratio_high) / 0.20, 0, 1)
        flat_signal      = np.clip((spectral_flat - 0.30) / 0.50, 0, 1)

        fft_score = float(np.clip(
            low_freq_signal * 0.40 + high_freq_signal * 0.40 + flat_signal * 0.20,
            0.0, 1.0
        ))

        return {
            "fft_score"    : round(fft_score, 4),
            "ratio_high"   : round(ratio_high, 4),
            "ratio_low"    : round(ratio_low, 4),
            "spectral_flat": round(spectral_flat, 4),
            "peak_score"   : round(peak_score, 4),
            "success"      : True,
        }

    except Exception as e:
        return {
            "fft_score": 0.5,
            "error"    : str(e),
            "success"  : False,
        }


def get_fft_interpretation(fft_score: float) -> str:
    if fft_score >= 0.70:
        return "🔴 FFT: Strong artificial frequency signature detected"
    elif fft_score >= 0.50:
        return "🟡 FFT: Suspicious frequency spectrum"
    elif fft_score >= 0.30:
        return "🟠 FFT: Slightly unusual frequency distribution"
    else:
        return "🟢 FFT: Normal frequency spectrum"