"""
Brandion DCT Watermark
======================
brandion_engine üzerine kurulu, wm_id tabanlı encode/decode.
Screen detection + perspective correction ile kamera dayanıklılığı.

Payload (56 bit):
  [16 magic][32 wm_id][8 checksum]
  ECC: 3x repeat → toplam 168 bit
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple

from brandion_engine import (
    BLOCK_SIZE, FREQ_BAND, PAYLOAD_BITS,
    int_to_bits, bits_to_int,
    add_error_correction, decode_error_correction,
    get_embedding_positions,
    embed_bit_in_block, extract_bit_from_block,
)

# ─── SABITLER ────────────────────────────────────────────────────────────────

MAGIC       = 0xB8A3
STRENGTH    = 40.0          # JPEG q75 + kamera distorsiyonuna dayanıklı
SECRET_KEY  = 0x42726164    # "Bran" — 32-bit max
ENCODE_W    = 1280
ENCODE_H    = 720

# ─── PAYLOAD ─────────────────────────────────────────────────────────────────

def _build_payload(wm_id: int) -> list[int]:
    """56 bit: [16 magic][32 wm_id][8 checksum]"""
    checksum = ((MAGIC >> 8) ^ (MAGIC & 0xFF) ^ (wm_id & 0xFF) ^ ((wm_id >> 8) & 0xFF)) & 0xFF
    bits = int_to_bits(MAGIC, 16) + int_to_bits(wm_id, 32) + int_to_bits(checksum, 8)
    assert len(bits) == PAYLOAD_BITS
    return bits


def _parse_payload(bits: list[int]) -> Optional[int]:
    """Payload'ı doğrula → wm_id veya None."""
    if len(bits) < PAYLOAD_BITS:
        return None
    magic    = bits_to_int(bits[0:16])
    wm_id    = bits_to_int(bits[16:48])
    checksum = bits_to_int(bits[48:56])

    if magic != MAGIC:
        return None

    expected = ((MAGIC >> 8) ^ (MAGIC & 0xFF) ^ (wm_id & 0xFF) ^ ((wm_id >> 8) & 0xFF)) & 0xFF
    if checksum != expected:
        return None

    return wm_id

# ─── ENCODE ──────────────────────────────────────────────────────────────────

def encode(img: Image.Image, wm_id: int) -> Image.Image:
    """PIL görsel + wm_id → filigranlanmış PIL görsel."""
    frame = np.array(img.convert("RGB"))
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Standart boyuta resize
    frame_bgr = cv2.resize(frame_bgr, (ENCODE_W, ENCODE_H), interpolation=cv2.INTER_LANCZOS4)

    ycbcr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycbcr[:, :, 0].astype(np.float64)
    h, w = y.shape

    payload     = _build_payload(wm_id)
    payload_ecc = add_error_correction(payload)
    positions   = get_embedding_positions(h, w, len(payload_ecc), seed=SECRET_KEY)

    y_enc = y.copy()
    for (row, col), bit in zip(positions, payload_ecc):
        block = y_enc[row:row + BLOCK_SIZE, col:col + BLOCK_SIZE]
        if block.shape == (BLOCK_SIZE, BLOCK_SIZE):
            y_enc[row:row + BLOCK_SIZE, col:col + BLOCK_SIZE] = embed_bit_in_block(block, bit, STRENGTH)

    ycbcr_enc = ycbcr.copy()
    ycbcr_enc[:, :, 0] = np.clip(y_enc, 0, 255).astype(np.uint8)
    enc_bgr = cv2.cvtColor(ycbcr_enc, cv2.COLOR_YCrCb2BGR)
    enc_rgb = cv2.cvtColor(enc_bgr, cv2.COLOR_BGR2RGB)

    psnr = cv2.PSNR(frame_bgr, enc_bgr)
    print(f"[DCT] encode wm_id={wm_id} PSNR={psnr:.1f}dB strength={STRENGTH}")
    return Image.fromarray(enc_rgb)


# ─── SCREEN DETECTION (spread_spectrum'dan uyarlandı) ────────────────────────

def _warp_to_standard(arr: np.ndarray, pts: np.ndarray) -> Optional[np.ndarray]:
    """4 köşeyi alıp ENCODE_W×ENCODE_H'ye warp et."""
    try:
        pts = pts.astype(np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        ordered = np.array([
            pts[s.argmin()], pts[d.argmin()],
            pts[s.argmax()], pts[d.argmax()]
        ], dtype=np.float32)
        dst = np.array([[0, 0], [ENCODE_W, 0], [ENCODE_W, ENCODE_H], [0, ENCODE_H]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(arr, M, (ENCODE_W, ENCODE_H))
    except Exception:
        return None


def _detect_screen_regions(frame_arr: np.ndarray) -> list[np.ndarray]:
    """Kamera frame'inde dikdörtgen ekranları tespit et, warp edilmiş BGR listesi döner."""
    gray = cv2.cvtColor(frame_arr, cv2.COLOR_RGB2GRAY)
    frame_area = gray.shape[0] * gray.shape[1]
    results = []

    for blur_k in [21, 31, 51]:
        blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        edges   = cv2.Canny(blurred, 10, 40)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:6]:
            peri = cv2.arcLength(c, True)
            for eps in [0.02, 0.04, 0.06]:
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) == 4:
                    area = cv2.contourArea(approx)
                    if area > frame_area * 0.05:
                        results.append((area, approx.reshape(4, 2)))
                    break

    seen, unique = set(), []
    for area, pts in sorted(results, key=lambda x: -x[0]):
        key = round(area / frame_area, 1)
        if key not in seen:
            seen.add(key)
            unique.append(pts)

    warped = []
    for pts in unique[:3]:
        arr = np.array(frame_arr)
        w = _warp_to_standard(arr, pts)
        if w is not None:
            coverage = cv2.contourArea(pts) / frame_area
            print(f"[DCT] screen candidate coverage={coverage:.1%}")
            warped.append(cv2.cvtColor(w, cv2.COLOR_RGB2BGR))
    return warped


# ─── NOISE OVERLAY ───────────────────────────────────────────────────────────

def generate_noise_overlay(wm_id: int, width: int = ENCODE_W, height: int = ENCODE_H) -> Image.Image:
    """
    Standalone noise overlay PNG üret.
    Nötr gri zemin üzerine DCT watermark gömülür → TV static görünümü.
    Video editörde Screen/Overlay blend mode ile içerik üstüne bindirilebilir.
    """
    rng = np.random.default_rng(seed=wm_id ^ 0xBDA3)
    # TV static grain → encode'un host image'ı
    grain = rng.standard_normal((height, width, 3)).astype(np.float32) * 55
    base_arr = np.clip(128 + grain, 0, 255).astype(np.uint8)
    base = Image.fromarray(base_arr, mode="RGB")
    # Watermark'ı grain texture'ın içine göm
    encoded = encode(base, wm_id)
    print(f"[DCT] noise overlay wm_id={wm_id} size={width}×{height}")
    return encoded


# ─── NEURAL DECODER (lazy load) ──────────────────────────────────────────────

_neural_decoder = None  # HiDDeNDecoder instance, ilk decode'da yüklenir

def _get_neural_decoder():
    global _neural_decoder
    if _neural_decoder is not None:
        return _neural_decoder
    model_path = Path(__file__).parent / "hidden_decoder.pt"
    if not model_path.exists():
        return None
    try:
        from hidden_decoder import HiDDeNDecoder
        _neural_decoder = HiDDeNDecoder.load(model_path)
        print(f"[DCT] neural decoder yüklendi: {model_path}")
    except Exception as e:
        print(f"[DCT] neural decoder yüklenemedi: {e}")
    return _neural_decoder


def _neural_decode(img: Image.Image) -> Optional[int]:
    """Neural decoder ile multi-crop voting."""
    decoder = _get_neural_decoder()
    if decoder is None:
        return None

    from hidden_decoder import _pil_to_tensor
    import torch

    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]

    crops = []
    # Tam frame + farklı crop'lar
    for scale in [1.0, 0.85, 0.70]:
        nh, nw = int(h * scale), int(w * scale)
        y0, x0 = (h - nh) // 2, (w - nw) // 2
        crop = Image.fromarray(arr[y0:y0+nh, x0:x0+nw])
        crops.append(crop)

    # Ekran tespiti varsa ona da bak
    screens = _detect_screen_regions(arr)
    for screen_bgr in screens:
        screen_rgb = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2RGB)
        crops.append(Image.fromarray(screen_rgb))

    votes: dict[int, int] = {}
    for crop in crops[:6]:
        wm_id = decoder.decode_image(crop)
        if wm_id is not None:
            votes[wm_id] = votes.get(wm_id, 0) + 1

    if not votes:
        return None

    best_id = max(votes, key=votes.__getitem__)
    if votes[best_id] >= 1:
        print(f"[DCT] neural decode OK wm_id={best_id} votes={votes}")
        return best_id
    return None


# ─── DECODE ──────────────────────────────────────────────────────────────────

def _decode_bgr(frame_bgr: np.ndarray) -> Optional[int]:
    """BGR frame'den wm_id çöz."""
    frame_bgr = cv2.resize(frame_bgr, (ENCODE_W, ENCODE_H), interpolation=cv2.INTER_LANCZOS4)
    ycbcr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycbcr[:, :, 0].astype(np.float64)
    h, w = y.shape

    n_bits_ecc = PAYLOAD_BITS * 3
    positions  = get_embedding_positions(h, w, n_bits_ecc, seed=SECRET_KEY)

    raw_bits = []
    for (row, col) in positions:
        block = y[row:row + BLOCK_SIZE, col:col + BLOCK_SIZE]
        if block.shape == (BLOCK_SIZE, BLOCK_SIZE):
            raw_bits.append(extract_bit_from_block(block, STRENGTH))

    corrected = decode_error_correction(raw_bits)
    return _parse_payload(corrected[:PAYLOAD_BITS])


def decode(img: Image.Image) -> Optional[int]:
    """
    PIL kamera frame → wm_id veya None.
    1) Neural decoder varsa önce onu dene (distorsiyon-robust)
    2) Ekran tespiti + perspective correction
    3) Fallback: tüm frame + crop variantları
    """
    # Neural decoder (eğitilmiş model varsa)
    wm_id = _neural_decode(img)
    if wm_id is not None:
        return wm_id

    arr = np.array(img.convert("RGB"))

    # Ekran tespiti ile warp
    screens = _detect_screen_regions(arr)
    for screen_bgr in screens:
        wm_id = _decode_bgr(screen_bgr)
        if wm_id is not None:
            print(f"[DCT] decode OK via screen detection wm_id={wm_id}")
            return wm_id

    # Fallback: farklı crop oranları
    for scale in [1.0, 0.85, 0.7]:
        h, w = arr.shape[:2]
        nh, nw = int(h * scale), int(w * scale)
        y0, x0 = (h - nh) // 2, (w - nw) // 2
        crop = arr[y0:y0 + nh, x0:x0 + nw]
        bgr  = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        wm_id = _decode_bgr(bgr)
        if wm_id is not None:
            print(f"[DCT] decode OK via fallback crop scale={scale} wm_id={wm_id}")
            return wm_id

    print("[DCT] decode FAIL — watermark not found")
    return None
