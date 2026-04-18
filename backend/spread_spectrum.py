"""
Brandion Spread Spectrum Watermark
====================================
Block-based PN — ekran→kamera, 16:9 crop ile dayanıklı decode.
"""

from __future__ import annotations
import numpy as np
from PIL import Image
from typing import Optional

# ─── SABITLER ────────────────────────────────────────────────────────────────

NUM_IDS     = 256
EPSILON     = 50        # güçlü sinyal — ekran→kamera kanal kaybı için
SEED_OFFSET = 0xB4A710
BLOCK_SIZE  = 64        # büyük blok — ölçek değişimine dayanıklı
THRESHOLD   = 5.0
DECODE_W    = 1920
DECODE_H    = 1080

# ─── BLOK PN ─────────────────────────────────────────────────────────────────

def _pn_blocks(id_val: int, height: int, width: int) -> np.ndarray:
    bh = (height + BLOCK_SIZE - 1) // BLOCK_SIZE
    bw = (width  + BLOCK_SIZE - 1) // BLOCK_SIZE
    rng = np.random.default_rng(seed=id_val ^ SEED_OFFSET)
    blocks = rng.choice(np.array([-1, 1], dtype=np.float32), size=(bh, bw))
    pattern = np.repeat(np.repeat(blocks, BLOCK_SIZE, axis=0), BLOCK_SIZE, axis=1)
    return pattern[:height, :width]

# ─── ENCODE ──────────────────────────────────────────────────────────────────

def encode_overlay(id_val: int, width: int = 1920, height: int = 1080) -> Image.Image:
    pn = _pn_blocks(id_val, height, width)
    overlay = np.clip(128 + EPSILON * pn, 0, 255).astype(np.uint8)
    rgb = np.stack([overlay] * 3, axis=-1)
    print(f"[SS] overlay id={id_val} ε={EPSILON} block={BLOCK_SIZE}")
    return Image.fromarray(rgb, mode="RGB")


def encode_frame(frame: Image.Image, id_val: int) -> Image.Image:
    arr = np.array(frame.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]
    pn = _pn_blocks(id_val, h, w)
    for c in range(3):
        arr[:, :, c] = np.clip(arr[:, :, c] + EPSILON * pn, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")

# ─── DECODE ──────────────────────────────────────────────────────────────────

def _crop_16_9(img: Image.Image) -> Image.Image:
    """Merkezi 16:9 crop — dikey telefon fotoğrafını düzeltir."""
    w, h = img.size
    target_ratio = 16 / 9
    if w / h > target_ratio:
        new_w = int(h * target_ratio)
        x = (w - new_w) // 2
        return img.crop((x, 0, x + new_w, h))
    else:
        new_h = int(w / target_ratio)
        y = (h - new_h) // 2
        return img.crop((0, y, w, y + new_h))


def decode(frame: Image.Image, num_ids: int = NUM_IDS) -> Optional[int]:
    """Kamera fotoğrafı → ID. 16:9 crop → 1920x1080 resize → korelasyon."""
    frame = _crop_16_9(frame)
    frame = frame.resize((DECODE_W, DECODE_H), Image.LANCZOS)
    gray = np.array(frame.convert("L")).astype(np.float32)
    h, w = gray.shape
    gray -= gray.mean()
    flat = gray.flatten()
    norm = np.linalg.norm(flat)
    if norm < 1e-6:
        return None

    best_id, best_corr = None, -np.inf
    for id_val in range(num_ids):
        pn = _pn_blocks(id_val, h, w).flatten()
        corr = float(np.dot(flat, pn)) / norm
        if corr > best_corr:
            best_corr = corr
            best_id = id_val

    print(f"[SS] best_corr={best_corr:.3f} id={best_id} threshold={THRESHOLD}")
    return best_id if best_corr > THRESHOLD else None
