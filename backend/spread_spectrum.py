"""
Brandion Spread Spectrum Watermark
====================================
Block-based PN — ekran→kamera, multi-scale + moiré-filtered decode.
"""

from __future__ import annotations
import numpy as np
from PIL import Image, ImageFilter
from typing import Optional, Tuple
import functools

# ─── SABITLER ────────────────────────────────────────────────────────────────

NUM_IDS     = 256
EPSILON     = 50
SEED_OFFSET = 0xB4A710
BLOCK_SIZE  = 64
THRESHOLD   = 3.5       # random noise max ~3.1 (max of 256 normals), margin check handles FP
MARGIN      = 1.2       # best - second_best must be > MARGIN
DECODE_W    = 1920
DECODE_H    = 1080

# ─── PN CACHE ────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _pn_matrix(height: int, width: int) -> np.ndarray:
    """Tüm ID'ler için PN pattern matrix — (NUM_IDS, H*W). Bir kez hesaplanır."""
    rows = []
    for id_val in range(NUM_IDS):
        rows.append(_pn_blocks(id_val, height, width).flatten())
    return np.stack(rows, axis=0)  # (256, N)


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

# ─── DECODE HELPERS ──────────────────────────────────────────────────────────

def _crop_16_9(img: Image.Image) -> Image.Image:
    w, h = img.size
    target = 16 / 9
    if w / h > target:
        new_w = int(h * target)
        x = (w - new_w) // 2
        return img.crop((x, 0, x + new_w, h))
    else:
        new_h = int(w / target)
        y = (h - new_h) // 2
        return img.crop((0, y, w, y + new_h))


def _prepare_candidates(frame: Image.Image) -> list[np.ndarray]:
    """
    Farklı scale/crop stratejileri ile hazırlanmış gri görsel listesi.
    Her biri (DECODE_H, DECODE_W) boyutunda, sıfır ortalamalı.
    Strateji: TV ekranının telefon frame'inde farklı büyüklüklerde göründüğünü varsay.
    """
    candidates = []

    # Scale oranları: TV'nin frame alanının kaçını kapladığını tahmin et.
    # 1.0 = tam frame, 0.7 = frame'in %70'i kadar TV görünüyor.
    for scale in [1.0, 0.90, 0.80, 0.70, 1.10]:
        w, h = frame.size
        nw = int(w * scale)
        nh = int(h * scale)
        nw = min(nw, w)
        nh = min(nh, h)
        x = (w - nw) // 2
        y = (h - nh) // 2
        cropped = frame.crop((x, y, x + nw, y + nh))

        # 16:9'a getir, moiré filtrele, resize
        c16 = _crop_16_9(cropped)
        blurred = c16.filter(ImageFilter.GaussianBlur(radius=1.2))
        resized = blurred.resize((DECODE_W, DECODE_H), Image.LANCZOS)

        gray = np.array(resized.convert("L")).astype(np.float32)
        gray -= gray.mean()
        norm = np.linalg.norm(gray)
        if norm > 1e-6:
            candidates.append(gray / norm * np.sqrt(DECODE_W * DECODE_H))  # normalize

    return candidates


def _correlate(gray_flat: np.ndarray, norm: float) -> Tuple[int, float, float]:
    """
    Tüm ID'lerle batch korelasyon. NumPy matmul ile tek seferde.
    Döner: (best_id, best_corr, margin)
    """
    pn_mat = _pn_matrix(DECODE_H, DECODE_W)   # (256, N)
    corrs = pn_mat @ gray_flat / norm           # (256,)
    sorted_idx = np.argsort(corrs)[::-1]
    best_id = int(sorted_idx[0])
    best_corr = float(corrs[sorted_idx[0]])
    second_corr = float(corrs[sorted_idx[1]])
    return best_id, best_corr, best_corr - second_corr

# ─── PUBLIC DECODE ───────────────────────────────────────────────────────────

def decode_scores(frame: Image.Image) -> Tuple[Optional[int], float, float]:
    """
    Kamera frame → (id veya None, best_corr, margin).
    Multi-frame voting için skoru da döndürür.
    """
    candidates = _prepare_candidates(frame)
    if not candidates:
        return None, 0.0, 0.0

    overall_best_id   = None
    overall_best_corr = -np.inf
    overall_margin    = 0.0

    for gray in candidates:
        flat = gray.flatten()
        norm = np.linalg.norm(flat)
        if norm < 1e-6:
            continue
        bid, bcorr, margin = _correlate(flat, norm)
        if bcorr > overall_best_corr:
            overall_best_corr  = bcorr
            overall_best_id    = bid
            overall_margin     = margin

    print(f"[SS] best_corr={overall_best_corr:.3f} margin={overall_margin:.3f} "
          f"id={overall_best_id} thr={THRESHOLD} req_margin={MARGIN}")

    if overall_best_corr > THRESHOLD and overall_margin > MARGIN:
        return overall_best_id, overall_best_corr, overall_margin
    return None, overall_best_corr, overall_margin


def decode(frame: Image.Image, num_ids: int = NUM_IDS) -> Optional[int]:
    """Geriye dönük uyumluluk wrapper."""
    result_id, _, _ = decode_scores(frame)
    return result_id


def decode_multi(frames: list[Image.Image]) -> Tuple[Optional[int], float]:
    """
    Birden fazla frame'den korelasyon ortalaması alarak decode.
    Her frame'den per-ID skor toplar, average üzerinden karar verir.
    """
    if not frames:
        return None, 0.0

    # Her ID için toplam skor biriktirir
    score_accum = np.zeros(NUM_IDS, dtype=np.float32)
    n_valid = 0

    for frame in frames:
        candidates = _prepare_candidates(frame)
        for gray in candidates:
            flat = gray.flatten()
            norm = np.linalg.norm(flat)
            if norm < 1e-6:
                continue
            pn_mat = _pn_matrix(DECODE_H, DECODE_W)
            corrs = pn_mat @ flat / norm
            score_accum += corrs
            n_valid += 1

    if n_valid == 0:
        return None, 0.0

    avg_scores = score_accum / n_valid
    sorted_idx = np.argsort(avg_scores)[::-1]
    best_id    = int(sorted_idx[0])
    best_avg   = float(avg_scores[sorted_idx[0]])
    margin     = float(avg_scores[sorted_idx[0]] - avg_scores[sorted_idx[1]])

    print(f"[SS-MULTI] frames={len(frames)} valid={n_valid} "
          f"best_avg={best_avg:.3f} margin={margin:.3f} id={best_id}")

    # Multi-frame ile eşik biraz düşer (SNR artar)
    multi_threshold = THRESHOLD * 0.75
    if best_avg > multi_threshold and margin > MARGIN * 0.75:
        return best_id, best_avg
    return None, best_avg
