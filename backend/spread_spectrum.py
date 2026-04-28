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
THRESHOLD   = 200.0
MARGIN      = 60.0
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


def encode_noise_overlay(id_val: int, width: int = ENCODE_W, height: int = ENCODE_H,
                          epsilon: int = 25) -> Image.Image:
    """
    Pure RGB static görünümlü SS overlay.
    Her piksel tamamen random RGB — insan gözü pattern görmez.
    SS sinyali noise altında gömülü, korelatör tarafından okunabilir.
    """
    rng = np.random.default_rng(seed=id_val ^ 0xDEAD)
    noise = rng.integers(0, 256, (height, width, 3), dtype=np.uint8).astype(np.float32)
    tile = _tile_pn(id_val)
    full = _tile_to_full(tile, height, width)
    for c in range(3):
        noise[:, :, c] = np.clip(noise[:, :, c] + epsilon * full, 0, 255)
    print(f"[SS] noise_overlay id={id_val} ε={epsilon} pure-RGB tile={TILE_W}×{TILE_H}")
    return Image.fromarray(noise.astype(np.uint8), mode="RGB")


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
        resized = img.resize((target_w, target_h), Image.LANCZOS)
        gray = np.array(resized.convert("L")).astype(np.float32)
        # High-pass filter: video içeriğini (düşük frekans) sil, watermark sinyali bırak
        low = cv2.GaussianBlur(gray, (0, 0), sigmaX=24)
        gray = gray - low  # yalnızca yüksek/orta frekans kalır
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

        # Fine grid 0.45-1.0 covers all perspective-warped tile periods
        fine_grid = [round(0.45 + i * 0.05, 2) for i in range(12)]  # 0.45..1.0
        for s in [scale, scale * 2.0, scale * 3.0, 0.333, 0.25] + fine_grid:
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
            fine_grid = [round(0.45 + i * 0.05, 2) for i in range(12)]
            for s in [scale, scale * 2.0, scale * 3.0] + fine_grid:
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


# ─── LAB b-CHANNEL WATERMARK ─────────────────────────────────────────────────
# Göz mavi-sarı (Lab b) değişimine en kör. 64px blok b kanalına gömülü →
# gözde "renk dalgalanması" gibi görünür, beyin filtreler.
# Decoder: b kanalı çıkar + HPF → sadece watermark sinyali kalır.

LAB_BLOCK   = 64
LAB_BLOCKS  = TILE_W // LAB_BLOCK   # 6
LAB_SEED    = 0xB4C7E2A9
LAB_THRESH  = 120.0
LAB_MARGIN  = 30.0


@functools.lru_cache(maxsize=NUM_IDS)
def _lab_tile_pn(id_val: int) -> np.ndarray:
    rng = np.random.default_rng(seed=id_val ^ LAB_SEED)
    blocks = rng.choice(np.array([-1, 1], dtype=np.float32),
                        size=(LAB_BLOCKS, LAB_BLOCKS))
    return np.repeat(np.repeat(blocks, LAB_BLOCK, axis=0), LAB_BLOCK, axis=1)


@functools.lru_cache(maxsize=64)
def _lab_pattern_matrix(actual_tile_w: int, actual_tile_h: int,
                         target_h: int, target_w: int) -> np.ndarray:
    rows = []
    for id_val in range(NUM_IDS):
        tile = _lab_tile_pn(id_val)
        tile_r = cv2.resize(tile, (actual_tile_w, actual_tile_h),
                            interpolation=cv2.INTER_LINEAR)
        full = _tile_to_full(tile_r, target_h, target_w).flatten().astype(np.float32)
        n = np.linalg.norm(full)
        rows.append(full / n if n > 1e-6 else full)
    return np.stack(rows)


def encode_lab_overlay(id_val: int, width: int = ENCODE_W,
                        height: int = ENCODE_H, epsilon: float = 8.0) -> Image.Image:
    """
    64px PN pattern → Lab b kanalına gömülü noise overlay.
    Göz b kanalı değişimini algılamaz → invisible.
    Kamera rengi yakalar → decode edilebilir.
    epsilon: Lab b birim cinsinden (0-255 aralığı). 8 ≈ JND sınırı.
    """
    rng = np.random.default_rng(seed=id_val ^ 0xCAFE1AB1)
    noise_rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)

    lab = cv2.cvtColor(noise_rgb, cv2.COLOR_RGB2Lab).astype(np.float32)
    l_ch, a_ch, b_ch = lab[:,:,0], lab[:,:,1], lab[:,:,2]

    tile = _lab_tile_pn(id_val)
    full = _tile_to_full(tile, height, width)

    # Adaptive masking: dokulu bölgelerde daha güçlü, düz bölgelerde zayıf
    edges = cv2.Laplacian(l_ch, cv2.CV_32F)
    mask = cv2.GaussianBlur(np.abs(edges), (11, 11), 0)
    if mask.max() > 1e-6:
        mask = mask / mask.max()   # 0-1 normalize
    mask = 0.3 + 0.7 * mask       # min 0.3, max 1.0

    b_wm = np.clip(b_ch + epsilon * full * mask, 0, 255)

    lab_out = np.stack([l_ch, a_ch, b_wm], axis=2).astype(np.uint8)
    rgb_out = cv2.cvtColor(lab_out, cv2.COLOR_Lab2RGB)
    print(f"[SS-LAB] encode id={id_val} ε={epsilon} block={LAB_BLOCK}px")
    return Image.fromarray(rgb_out, "RGB")


def decode_lab_scores(frame: Image.Image) -> Tuple[Optional[int], float, float]:
    """
    Lab b kanalı + high-pass filter → PN korelasyon decode.
    Video içeriği (düşük frekans) filtreden geçemez, sadece watermark kalır.
    """
    def _prep(img: Image.Image, target_w: int = 960, target_h: int = 540):
        resized = img.resize((target_w, target_h), Image.LANCZOS)
        arr = cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2Lab).astype(np.float32)
        b = arr[:, :, 2]
        # High-pass: video içeriğini sil
        low = cv2.GaussianBlur(b, (0, 0), sigmaX=24)
        return b - low

    candidates = []
    screens = _detect_screen_candidates(frame)
    for s in screens:
        candidates.append(_prep(s))
    for scale in [1.0, 0.80, 0.65]:
        w, h = frame.size
        nw, nh = int(w * scale), int(h * scale)
        x, y = (w - nw) // 2, (h - nh) // 2
        crop = _crop_16_9(frame.crop((x, y, x + nw, y + nh)))
        candidates.append(_prep(crop))

    candidates = [g for g in candidates if g.std() > 0.5]
    if not candidates:
        return None, 0.0, 0.0

    overall_best_id   = None
    overall_best_corr = -np.inf
    overall_margin    = 0.0

    for gray in candidates:
        h, w = gray.shape
        gray_flat = gray.flatten().astype(np.float32)
        norm_g = np.linalg.norm(gray_flat)
        if norm_g < 1e-6:
            continue
        fine_grid = [round(0.45 + i * 0.05, 2) for i in range(12)]
        for s in [1.0, 0.5, 2.0, 0.333, 0.25] + fine_grid:
            if s < 0.3 or s > 5.0:
                continue
            atw = int(round(TILE_W * s))
            ath = int(round(TILE_H * s))
            if atw < 8 or ath < 8:
                continue
            pmat = _lab_pattern_matrix(atw, ath, h, w)
            corrs = pmat @ gray_flat / norm_g * np.sqrt(h * w)
            idx = np.argsort(corrs)[::-1]
            bid = int(idx[0]); bcorr = float(corrs[idx[0]]); margin = float(corrs[idx[0]] - corrs[idx[1]])
            if bcorr > overall_best_corr:
                overall_best_corr = bcorr; overall_best_id = bid; overall_margin = margin

    print(f"[SS-LAB] best_corr={overall_best_corr:.1f} margin={overall_margin:.1f} "
          f"id={overall_best_id} thr={LAB_THRESH}")

    if overall_best_corr > LAB_THRESH and overall_margin > LAB_MARGIN:
        return overall_best_id, overall_best_corr, overall_margin
    return None, overall_best_corr, overall_margin


# ─── MED-GRAIN WATERMARK (24px) ─────────────────────────────────────────────
# 24px bloklar — kamera testinde geçti (corr=3550), 64px'e göre görsel olarak finer.
# Tile boyutu aynı (384px) → mevcut scale detection değişmez.

MED_BLOCK   = 24
MED_BLOCKS  = TILE_W // MED_BLOCK   # 16
MED_SEED    = 0xA3C7E9F1
MED_THRESH  = 150.0
MED_MARGIN  = 40.0


@functools.lru_cache(maxsize=NUM_IDS)
def _med_tile_pn(id_val: int) -> np.ndarray:
    """24px bloklu PN tile (384×384)."""
    rng = np.random.default_rng(seed=id_val ^ MED_SEED)
    blocks = rng.choice(np.array([-1, 1], dtype=np.float32),
                        size=(MED_BLOCKS, MED_BLOCKS))
    return np.repeat(np.repeat(blocks, MED_BLOCK, axis=0), MED_BLOCK, axis=1)


@functools.lru_cache(maxsize=64)
def _med_pattern_matrix(actual_tile_w: int, actual_tile_h: int,
                         target_h: int, target_w: int) -> np.ndarray:
    rows = []
    for id_val in range(NUM_IDS):
        tile = _med_tile_pn(id_val)
        tile_r = cv2.resize(tile, (actual_tile_w, actual_tile_h),
                            interpolation=cv2.INTER_LINEAR)
        full = _tile_to_full(tile_r, target_h, target_w).flatten().astype(np.float32)
        n = np.linalg.norm(full)
        rows.append(full / n if n > 1e-6 else full)
    return np.stack(rows)


def encode_med_overlay(id_val: int, width: int = ENCODE_W,
                        height: int = ENCODE_H, epsilon: int = 22,
                        blur_sigma: float = 18.0) -> Image.Image:
    """
    24px blok + Gaussian blur + random noise overlay.
    Blur blok kenarlarını yumuşatır → göze saf TV static gibi görünür.
    Kamera için yeterli büyük-ölçekli sinyal korunur.
    """
    rng = np.random.default_rng(seed=id_val ^ 0xBEEF2401)
    noise = rng.integers(0, 256, (height, width, 3), dtype=np.uint8).astype(np.float32)
    tile = _med_tile_pn(id_val)
    full = _tile_to_full(tile, height, width)  # sert kenarlar ±1

    # Blok kenarlarını Gaussian blur ile yumuşat → görünür grid yok
    full_img = Image.fromarray(((full + 1) * 127.5).astype(np.uint8), mode="L")
    full_blurred = full_img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    full_smooth = (np.array(full_blurred, dtype=np.float32) / 127.5) - 1.0  # tekrar ±1 aralığı

    for c in range(3):
        noise[:, :, c] = np.clip(noise[:, :, c] + epsilon * full_smooth, 0, 255)
    print(f"[SS-MED] encode id={id_val} ε={epsilon} blur_σ={blur_sigma} block={MED_BLOCK}px tile={TILE_W}px")
    return Image.fromarray(noise.astype(np.uint8), mode="RGB")


def decode_med_scores(frame: Image.Image) -> Tuple[Optional[int], float, float]:
    """24px blok decode. FFT scale detection ile scale-invariant."""
    candidates = _prepare_gray_candidates(frame)
    if not candidates:
        return None, 0.0, 0.0
    candidates = [g for g in candidates if g.std() > 4.0]
    if not candidates:
        return None, 0.0, 0.0

    overall_best_id   = None
    overall_best_corr = -np.inf
    overall_margin    = 0.0

    for gray in candidates:
        scale = _find_tile_scale(gray)
        fine_grid = [round(0.45 + i * 0.05, 2) for i in range(12)]
        for s in [scale, scale * 2.0, scale * 3.0, 0.333, 0.25] + fine_grid:
            if s < 0.3 or s > 5.0:
                continue
            h, w = gray.shape
            atw = int(round(TILE_W * s))
            ath = int(round(TILE_H * s))
            if atw < 8 or ath < 8:
                continue
            gray_flat = gray.flatten().astype(np.float32)
            norm_g = np.linalg.norm(gray_flat)
            if norm_g < 1e-6:
                continue
            pmat = _med_pattern_matrix(atw, ath, h, w)
            corrs = pmat @ gray_flat / norm_g * np.sqrt(h * w)
            idx = np.argsort(corrs)[::-1]
            bid, bcorr, margin = int(idx[0]), float(corrs[idx[0]]), float(corrs[idx[0]] - corrs[idx[1]])
            if bcorr > overall_best_corr:
                overall_best_corr = bcorr
                overall_best_id   = bid
                overall_margin    = margin

    print(f"[SS-MED] best_corr={overall_best_corr:.1f} margin={overall_margin:.1f} "
          f"id={overall_best_id} thr={MED_THRESH}")

    if overall_best_corr > MED_THRESH and overall_margin > MED_MARGIN:
        return overall_best_id, overall_best_corr, overall_margin
    return None, overall_best_corr, overall_margin


# ─── TEMPORAL MODULATION WATERMARK ───────────────────────────────────────────
# Her frame tek başına pure random noise.  Sinyal frame FARKINDA gizli.
# Çift frame: noise + ε×pattern   |   Tek frame: noise − ε×pattern
# Göz ikisini temporal integrate eder → pattern görmez.
# Kamera: 2 ardışık frame → diff = ±2ε×pattern → decode.

TEMP_BLOCK   = 32
TEMP_SEED    = 0xAE19C47B
TEMP_EPSILON = 30


@functools.lru_cache(maxsize=NUM_IDS)
def _temp_tile_pn(id_val: int) -> np.ndarray:
    """32px bloklu PN tile (384×384) — temporal encode için."""
    NBLK = TILE_W // TEMP_BLOCK   # 12
    rng = np.random.default_rng(seed=id_val ^ TEMP_SEED)
    blocks = rng.choice(np.array([-1, 1], dtype=np.float32), size=(NBLK, NBLK))
    return np.repeat(np.repeat(blocks, TEMP_BLOCK, axis=0), TEMP_BLOCK, axis=1)


@functools.lru_cache(maxsize=64)
def _temp_pattern_matrix(actual_tile_w: int, actual_tile_h: int,
                          target_h: int, target_w: int) -> np.ndarray:
    rows = []
    for id_val in range(NUM_IDS):
        tile = _temp_tile_pn(id_val)
        tile_r = cv2.resize(tile, (actual_tile_w, actual_tile_h),
                            interpolation=cv2.INTER_LINEAR)
        full = _tile_to_full(tile_r, target_h, target_w).flatten().astype(np.float32)
        n = np.linalg.norm(full)
        rows.append(full / n if n > 1e-6 else full)
    return np.stack(rows)


def encode_temporal_pair(id_val: int, width: int = ENCODE_W, height: int = ENCODE_H,
                          epsilon: int = TEMP_EPSILON) -> tuple[Image.Image, Image.Image]:
    """
    İki frame üret.  Her biri tek başına pure random noise görünümünde.
    Fark: ±2×epsilon×PN_pattern  →  decode edilebilir.

    Stüdyo: frame_even'i çift karelere, frame_odd'u tek karelere ekler.
    """
    rng = np.random.default_rng(seed=id_val ^ 0xBA5E0015)
    noise = rng.integers(0, 256, (height, width, 3), dtype=np.uint8).astype(np.float32)
    tile = _temp_tile_pn(id_val)
    full = _tile_to_full(tile, height, width)          # (H, W)  ±1
    pat3 = full[:, :, np.newaxis]                      # broadcast to 3 ch

    frame_even = np.clip(noise + epsilon * pat3, 0, 255).astype(np.uint8)
    frame_odd  = np.clip(noise - epsilon * pat3, 0, 255).astype(np.uint8)
    print(f"[SS-TEMP] encode id={id_val} ε={epsilon} block={TEMP_BLOCK}px")
    return Image.fromarray(frame_even, "RGB"), Image.fromarray(frame_odd, "RGB")


def decode_temporal_scores(frame1: Image.Image, frame2: Image.Image,
                            ) -> Tuple[Optional[int], float, float]:
    """
    2 ardışık kamera karesi → temporal fark → decode.
    Fark = ±2ε×pattern (even−odd veya odd−even, büyüklük aynı).
    """
    a1 = np.array(frame1.convert("RGB")).astype(np.float32)
    a2 = np.array(frame2.convert("RGB")).astype(np.float32)

    # Gri fark: (H, W)
    diff = (a1 - a2).mean(axis=2)   # even−odd → +2ε×pat
    diff -= diff.mean()

    h, w = diff.shape
    diff_flat = diff.flatten()
    norm_d = np.linalg.norm(diff_flat)
    if norm_d < 1e-6:
        return None, 0.0, 0.0

    # Her iki yönü de dene (even−odd ve odd−even)
    overall_best_id   = None
    overall_best_corr = -np.inf
    overall_margin    = 0.0

    for sign in [1.0, -1.0]:
        signed_flat = sign * diff_flat

        fine_grid = [round(0.45 + i * 0.05, 2) for i in range(12)]
        for s in [1.0, 0.5, 2.0, 0.333] + fine_grid:
            if s < 0.3 or s > 5.0:
                continue
            atw = int(round(TILE_W * s))
            ath = int(round(TILE_H * s))
            if atw < 8 or ath < 8:
                continue
            pmat = _temp_pattern_matrix(atw, ath, h, w)
            corrs = pmat @ signed_flat / norm_d * np.sqrt(h * w)
            idx = np.argsort(corrs)[::-1]
            bid = int(idx[0])
            bcorr = float(corrs[idx[0]])
            margin = float(corrs[idx[0]] - corrs[idx[1]])
            if bcorr > overall_best_corr:
                overall_best_corr = bcorr
                overall_best_id   = bid
                overall_margin    = margin

    TEMP_THRESH  = 150.0
    TEMP_MARGIN  = 40.0
    print(f"[SS-TEMP] best_corr={overall_best_corr:.1f} margin={overall_margin:.1f} "
          f"id={overall_best_id} thr={TEMP_THRESH}")

    if overall_best_corr > TEMP_THRESH and overall_margin > TEMP_MARGIN:
        return overall_best_id, overall_best_corr, overall_margin
    return None, overall_best_corr, overall_margin


# ─── FINE-GRAIN INVISIBLE WATERMARK ──────────────────────────────────────────
# 8px bloklar — 64px'e göre 8× daha ince, normal TV izleme mesafesinde görünmez.
# Tile boyutu aynı (384px) → mevcut scale detection değişmez.

FINE_BLOCK    = 8
FINE_BLOCKS   = TILE_W // FINE_BLOCK   # 48
FINE_SEED     = 0xF1AE2BCD
FINE_THRESH   = 180.0
FINE_MARGIN   = 40.0

@functools.lru_cache(maxsize=NUM_IDS)
def _fine_tile_pn(id_val: int) -> np.ndarray:
    """8px bloklu PN tile (384×384). Görsel noise gibi görünür."""
    rng = np.random.default_rng(seed=id_val ^ FINE_SEED)
    blocks = rng.choice(np.array([-1, 1], dtype=np.float32),
                        size=(FINE_BLOCKS, FINE_BLOCKS))
    return np.repeat(np.repeat(blocks, FINE_BLOCK, axis=0), FINE_BLOCK, axis=1)


@functools.lru_cache(maxsize=64)
def _fine_pattern_matrix(actual_tile_w: int, actual_tile_h: int,
                          target_h: int, target_w: int) -> np.ndarray:
    rows = []
    for id_val in range(NUM_IDS):
        tile = _fine_tile_pn(id_val)
        tile_r = cv2.resize(tile, (actual_tile_w, actual_tile_h),
                            interpolation=cv2.INTER_LINEAR)
        full = _tile_to_full(tile_r, target_h, target_w).flatten().astype(np.float32)
        n = np.linalg.norm(full)
        rows.append(full / n if n > 1e-6 else full)
    return np.stack(rows)


def encode_fine_overlay(id_val: int, width: int = ENCODE_W,
                         height: int = ENCODE_H, epsilon: int = 80) -> Image.Image:
    """
    Pure RGB static görünümlü watermark.
    8px bloklar + random noise → göze TV static, kameraya okunabilir.
    """
    rng = np.random.default_rng(seed=id_val ^ 0xCAFE31AE)
    noise = rng.integers(0, 256, (height, width, 3), dtype=np.uint8).astype(np.float32)
    tile = _fine_tile_pn(id_val)
    full = _tile_to_full(tile, height, width)
    for c in range(3):
        noise[:, :, c] = np.clip(noise[:, :, c] + epsilon * full, 0, 255)
    print(f"[SS-FINE] encode id={id_val} ε={epsilon} block={FINE_BLOCK}px tile={TILE_W}px")
    return Image.fromarray(noise.astype(np.uint8), mode="RGB")


def decode_fine_scores(frame: Image.Image) -> Tuple[Optional[int], float, float]:
    """Fine-grain (8px) decode. Mevcut FFT scale detection'ı kullanır."""
    candidates = _prepare_gray_candidates(frame)
    if not candidates:
        return None, 0.0, 0.0
    candidates = [g for g in candidates if g.std() > 4.0]
    if not candidates:
        return None, 0.0, 0.0

    overall_best_id   = None
    overall_best_corr = -np.inf
    overall_margin    = 0.0

    for gray in candidates:
        scale = _find_tile_scale(gray)
        fine_grid = [round(0.45 + i * 0.05, 2) for i in range(12)]
        for s in [scale, scale * 2.0, scale * 3.0, 0.333, 0.25] + fine_grid:
            if s < 0.3 or s > 5.0:
                continue
            h, w = gray.shape
            atw = int(round(TILE_W * s))
            ath = int(round(TILE_H * s))
            if atw < 8 or ath < 8:
                continue
            gray_flat = gray.flatten().astype(np.float32)
            norm_g = np.linalg.norm(gray_flat)
            if norm_g < 1e-6:
                continue
            pmat = _fine_pattern_matrix(atw, ath, h, w)
            corrs = pmat @ gray_flat / norm_g * np.sqrt(h * w)
            idx = np.argsort(corrs)[::-1]
            bid, bcorr, margin = int(idx[0]), float(corrs[idx[0]]), float(corrs[idx[0]] - corrs[idx[1]])
            if bcorr > overall_best_corr:
                overall_best_corr = bcorr
                overall_best_id   = bid
                overall_margin    = margin

    print(f"[SS-FINE] best_corr={overall_best_corr:.1f} margin={overall_margin:.1f} "
          f"id={overall_best_id} thr={FINE_THRESH}")

    if overall_best_corr > FINE_THRESH and overall_margin > FINE_MARGIN:
        return overall_best_id, overall_best_corr, overall_margin
    return None, overall_best_corr, overall_margin
