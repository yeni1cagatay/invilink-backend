"""
Brandion Spread Spectrum Watermark
====================================
Tiled PN encoding — scale-invariant decode.
Küçük bir tile oluşturulur, 1920×1080'e tekrar edilir.
Decode'da FFT ile tile periyodu bulunur → mesafeden bağımsız.
"""

from __future__ import annotations
import numpy as np
from PIL import Image, ImageFilter
from typing import Optional, Tuple
import functools
import cv2

# ─── SABITLER ────────────────────────────────────────────────────────────────

NUM_IDS     = 256
EPSILON     = 50
SEED_OFFSET = 0xB4A710
BLOCK_SIZE  = 64           # tile içindeki blok boyutu — 64px kamera bulanıklığına dayanıklı
TILE_BLOCKS = 6            # tile = 6×6 blok = 384×384px
TILE_W      = BLOCK_SIZE * TILE_BLOCKS   # 384
TILE_H      = BLOCK_SIZE * TILE_BLOCKS   # 384
THRESHOLD   = 300.0
MARGIN      = 50.0
ENCODE_W    = 1920
ENCODE_H    = 1080

# ─── TILE PN ─────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=NUM_IDS)
def _tile_pn(id_val: int) -> np.ndarray:
    """ID için TILE_H×TILE_W boyutunda PN tile üret. Cache'lenir."""
    rng = np.random.default_rng(seed=id_val ^ SEED_OFFSET)
    blocks = rng.choice(np.array([-1, 1], dtype=np.float32),
                        size=(TILE_BLOCKS, TILE_BLOCKS))
    tile = np.repeat(np.repeat(blocks, BLOCK_SIZE, axis=0), BLOCK_SIZE, axis=1)
    return tile   # (TILE_H, TILE_W)


def _tile_to_full(tile: np.ndarray, h: int, w: int) -> np.ndarray:
    """Tile'ı h×w boyutuna tekrar ederek döşe."""
    th, tw = tile.shape[:2]
    reps_h = int(np.ceil(h / th))
    reps_w = int(np.ceil(w / tw))
    tiled = np.tile(tile, (reps_h, reps_w))
    return tiled[:h, :w]

# ─── ENCODE ──────────────────────────────────────────────────────────────────

def encode_overlay(id_val: int, width: int = ENCODE_W, height: int = ENCODE_H) -> Image.Image:
    tile = _tile_pn(id_val)
    full = _tile_to_full(tile, height, width)
    overlay = np.clip(128 + 128 * full, 0, 255).astype(np.uint8)  # ε=128 → siyah-beyaz, kamera odaklanabilir
    rgb = np.stack([overlay] * 3, axis=-1)
    print(f"[SS] overlay id={id_val} ε=128 tile={TILE_W}×{TILE_H} block={BLOCK_SIZE}")
    return Image.fromarray(rgb, mode="RGB")


def encode_frame(frame: Image.Image, id_val: int) -> Image.Image:
    arr = np.array(frame.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]
    tile = _tile_pn(id_val)
    full = _tile_to_full(tile, h, w)
    for c in range(3):
        arr[:, :, c] = np.clip(arr[:, :, c] + EPSILON * full, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")

# ─── DECODE ──────────────────────────────────────────────────────────────────

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


def _find_tile_scale(gray: np.ndarray) -> float:
    """
    FFT ile görüntüdeki tile periyodunu bul.
    Döner: scale faktörü (beklenen TILE boyutunun kaç katı).
    Bulamazsa 1.0 döner.
    """
    h, w = gray.shape
    F = np.fft.fft2(gray)
    power = np.abs(np.fft.fftshift(F))
    cx, cy = w // 2, h // 2
    # DC baskıla
    power[cy-3:cy+3, cx-3:cx+3] = 0

    # Yatay eksende güç profili (orta banttan)
    band = max(1, h // 20)
    horiz = power[cy-band:cy+band, :].mean(axis=0)
    horiz[cx-3:cx+3] = 0

    top_idx = np.argmax(horiz)
    freq = abs(top_idx - cx)
    if freq < 2:
        return 1.0

    detected_period = w / freq
    scale = detected_period / TILE_W
    print(f"[SS] FFT detected period={detected_period:.1f}px  scale={scale:.3f}")
    return float(scale)


@functools.lru_cache(maxsize=64)
def _pattern_matrix(actual_tile_w: int, actual_tile_h: int,
                    target_h: int, target_w: int) -> np.ndarray:
    """Tüm 256 ID için normalize pattern matris. Cache'lenir → scale başına 1 kez hesaplanır."""
    rows = []
    for id_val in range(NUM_IDS):
        tile = _tile_pn(id_val)
        tile_r = cv2.resize(tile, (actual_tile_w, actual_tile_h),
                            interpolation=cv2.INTER_LINEAR)
        full = _tile_to_full(tile_r, target_h, target_w).flatten().astype(np.float32)
        n = np.linalg.norm(full)
        rows.append(full / n if n > 1e-6 else full)
    return np.stack(rows)  # (256, H*W)


def _decode_at_scale(gray: np.ndarray, scale: float) -> Tuple[int, float, float]:
    """Batch matmul ile tüm ID'leri tek seferde korelasyon."""
    h, w = gray.shape
    actual_tile_h = int(round(TILE_H * scale))
    actual_tile_w = int(round(TILE_W * scale))
    if actual_tile_h < 8 or actual_tile_w < 8:
        return 0, 0.0, 0.0

    gray_flat = gray.flatten().astype(np.float32)
    norm_g = np.linalg.norm(gray_flat)
    if norm_g < 1e-6:
        return 0, 0.0, 0.0

    pmat = _pattern_matrix(actual_tile_w, actual_tile_h, h, w)  # (256, H*W)
    corrs = pmat @ gray_flat / norm_g * np.sqrt(h * w)           # (256,)

    idx = np.argsort(corrs)[::-1]
    return int(idx[0]), float(corrs[idx[0]]), float(corrs[idx[0]] - corrs[idx[1]])


def _prepare_gray_candidates(frame: Image.Image) -> list[np.ndarray]:
    """Frame'den gri görsel adayları üret (screen detection + fallback crops)."""
    candidates = []

    def _to_gray(img: Image.Image, target_w: int = 960, target_h: int = 540) -> Optional[np.ndarray]:
        blurred = img.filter(ImageFilter.GaussianBlur(radius=1.0))
        resized = blurred.resize((target_w, target_h), Image.LANCZOS)
        gray = np.array(resized.convert("L")).astype(np.float32)
        gray -= gray.mean()
        return gray

    # Screen detection
    screens = _detect_screen_candidates(frame)
    for screen in screens:
        candidates.append(_to_gray(screen))

    # Fallback crops
    for scale in [1.0, 0.80, 0.65]:
        w, h = frame.size
        nw, nh = int(w * scale), int(h * scale)
        x, y = (w - nw) // 2, (h - nh) // 2
        cropped = _crop_16_9(frame.crop((x, y, x + nw, y + nh)))
        candidates.append(_to_gray(cropped))

    return [g for g in candidates if g is not None]


# ─── SCREEN DETECTION ────────────────────────────────────────────────────────

def _warp_quad(arr: np.ndarray, pts: np.ndarray) -> Image.Image:
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).ravel()
    ordered = np.array([pts[s.argmin()], pts[d.argmin()],
                        pts[s.argmax()], pts[d.argmax()]], dtype=np.float32)
    dst = np.array([[0, 0], [ENCODE_W, 0], [ENCODE_W, ENCODE_H], [0, ENCODE_H]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(arr, M, (ENCODE_W, ENCODE_H))
    return Image.fromarray(warped)


def _detect_screen_candidates(frame: Image.Image) -> list[Image.Image]:
    arr = np.array(frame.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    frame_area = arr.shape[0] * arr.shape[1]
    results = []

    for blur_k in [31, 21, 51]:
        blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        edges = cv2.Canny(blurred, 10, 40)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        edges_d = cv2.dilate(edges, kernel, iterations=2)
        cnts, _ = cv2.findContours(edges_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:8]
        for c in cnts:
            peri = cv2.arcLength(c, True)
            for eps in [0.02, 0.04, 0.06]:
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) == 4:
                    area = cv2.contourArea(approx)
                    if area > frame_area * 0.08:
                        results.append((area, approx.reshape(4, 2)))
                    break

    seen, unique = set(), []
    for area, pts in sorted(results, key=lambda x: x[0], reverse=True):
        key = round(area / frame_area, 1)
        if key not in seen:
            seen.add(key)
            unique.append((area, pts))

    warped = []
    for area, pts in unique[:4]:
        print(f"[SS] screen candidate coverage={area/frame_area:.1%}")
        warped.append(_warp_quad(arr, pts))
    return warped


# ─── PUBLIC API ──────────────────────────────────────────────────────────────

def decode_scores(frame: Image.Image) -> Tuple[Optional[int], float, float]:
    """
    Kamera frame → (id veya None, best_corr, margin).
    Scale-invariant: FFT ile tile periyodunu bulur, doğru scale'de decode eder.
    """
    candidates = _prepare_gray_candidates(frame)
    if not candidates:
        return None, 0.0, 0.0
    # Bulanık / düz frame'i reddet
    candidates = [g for g in candidates if g.std() > 4.0]
    if not candidates:
        return None, 0.0, 0.0

    overall_best_id   = None
    overall_best_corr = -np.inf
    overall_margin    = 0.0

    for gray in candidates:
        scale = _find_tile_scale(gray)

        for s in [scale, 0.85, 1.15, scale * 2.0, scale * 3.0, 0.5, 1.0]:
            if s < 0.3 or s > 5.0:
                continue
            bid, bcorr, margin = _decode_at_scale(gray, s)
            if bcorr > overall_best_corr:
                overall_best_corr = bcorr
                overall_best_id   = bid
                overall_margin    = margin

    print(f"[SS] best_corr={overall_best_corr:.3f} margin={overall_margin:.3f} "
          f"id={overall_best_id} thr={THRESHOLD}")

    if overall_best_corr > THRESHOLD and overall_margin > MARGIN:
        return overall_best_id, overall_best_corr, overall_margin
    return None, overall_best_corr, overall_margin


def decode(frame: Image.Image, num_ids: int = NUM_IDS) -> Optional[int]:
    """Geriye dönük uyumluluk wrapper."""
    result_id, _, _ = decode_scores(frame)
    return result_id


def decode_multi(frames: list[Image.Image]) -> Tuple[Optional[int], float]:
    """Birden fazla frame → korelasyon ortalaması."""
    if not frames:
        return None, 0.0

    score_accum = np.zeros(NUM_IDS, dtype=np.float32)
    n_valid = 0

    for frame in frames:
        candidates = _prepare_gray_candidates(frame)
        for gray in candidates:
            scale = _find_tile_scale(gray)
            for s in [scale, 0.5, 1.0]:
                if s < 0.3 or s > 5.0:
                    continue
                atw = int(round(TILE_W * s)); ath = int(round(TILE_H * s))
                if atw < 8 or ath < 8:
                    continue
                h, w = gray.shape
                gf = gray.flatten().astype(np.float32)
                ng = np.linalg.norm(gf)
                if ng < 1e-6:
                    continue
                pmat = _pattern_matrix(atw, ath, h, w)
                score_accum += pmat @ gf / ng
                n_valid += 1

    if n_valid == 0:
        return None, 0.0

    avg = score_accum / n_valid
    idx = np.argsort(avg)[::-1]
    best_id = int(idx[0])
    best_avg = float(avg[idx[0]])
    margin = float(avg[idx[0]] - avg[idx[1]])

    print(f"[SS-MULTI] best_avg={best_avg:.3f} margin={margin:.3f} id={best_id}")

    if best_avg > THRESHOLD * 0.75 and margin > MARGIN * 0.75:
        return best_id, best_avg
    return None, best_avg
