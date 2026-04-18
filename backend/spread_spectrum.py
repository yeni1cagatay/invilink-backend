"""
Brandion Spread Spectrum Watermark
====================================
ID → Pseudo-random Noise → piksel overlay
Tek frame decode, platform bağımsız, post prodüksiyon dostu.
"""

from __future__ import annotations
import numpy as np
from PIL import Image
from typing import Optional

# ─── SABITLER ────────────────────────────────────────────────────────────────

NUM_IDS     = 256       # 0-255 arası unique ID
EPSILON     = 20        # piksel başına sinyal gücü — ekran→kamera için yüksek
SEED_OFFSET = 0xB4A710  # gizli seed — brute force zorlaştırır
THRESHOLD   = 8.0       # korelasyon eşiği
DECODE_W    = 1920      # decode için standart genişlik
DECODE_H    = 1080      # decode için standart yükseklik

# ─── PN DİZİSİ ───────────────────────────────────────────────────────────────

def _pn(id_val: int, size: int) -> np.ndarray:
    """ID için tekrarlanabilir ±1 pseudo-random dizi."""
    rng = np.random.default_rng(seed=id_val ^ SEED_OFFSET)
    return rng.choice(np.array([-1, 1], dtype=np.float32), size=size)

# ─── ENCODE ──────────────────────────────────────────────────────────────────

def encode_overlay(id_val: int, width: int = 1920, height: int = 1080) -> Image.Image:
    """
    Post prodüksiyon overlay PNG üret.
    128 gri zemin + PN noise. Add blend modda üste koy.
    """
    size = width * height
    pn = _pn(id_val, size)
    overlay = np.clip(128 + EPSILON * pn, 0, 255).astype(np.uint8)
    overlay = overlay.reshape(height, width)
    # RGB'ye çevir (bazı NLE'ler mono kabul etmez)
    rgb = np.stack([overlay] * 3, axis=-1)
    img = Image.fromarray(rgb, mode="RGB")
    print(f"[SS] overlay id={id_val} size={width}x{height} ε={EPSILON}")
    return img


def encode_frame(frame: Image.Image, id_val: int) -> Image.Image:
    """
    Var olan bir frame'e doğrudan gömme (video encode için).
    """
    arr = np.array(frame.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]
    pn = _pn(id_val, w * h).reshape(h, w)
    for c in range(3):
        arr[:, :, c] = np.clip(arr[:, :, c] + EPSILON * pn, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")

# ─── DECODE ──────────────────────────────────────────────────────────────────

def decode(frame: Image.Image, num_ids: int = NUM_IDS) -> Optional[int]:
    """
    Tek frame → ID.
    Kamera fotoğrafı overlay boyutuna indirgenir, sonra korelasyon.
    """
    frame = frame.resize((DECODE_W, DECODE_H), Image.LANCZOS)
    gray = np.array(frame.convert("L")).astype(np.float32).flatten()
    # DC bileşeni çıkar
    gray -= gray.mean()
    norm = np.linalg.norm(gray)
    if norm < 1e-6:
        return None

    best_id, best_corr = None, -np.inf

    for id_val in range(num_ids):
        pn = _pn(id_val, len(gray))
        corr = float(np.dot(gray, pn)) / norm
        if corr > best_corr:
            best_corr = corr
            best_id = id_val

    print(f"[SS] best_corr={best_corr:.3f} id={best_id} (threshold={THRESHOLD})")
    return best_id if best_corr > THRESHOLD else None

