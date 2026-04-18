"""
Brandion Watermark — 4-Quadrant Redundant Differential
=======================================================
Aynı kod 4 kez gömülür (2x2 grid, her tile 512x512).
Decode sırasında 4 quadrant bağımsız denenir.
Telefon açısına ve kısmi ekran görünümüne karşı dayanıklı.
"""

from __future__ import annotations
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# ─── SABITLER ────────────────────────────────────────────────────────────────

IMAGE_SIZE   = 1024
TILE_SIZE    = 512       # 4 quadrant, her biri 512x512
BLOCK_SIZE   = 28        # 512/28 = ~18 blok/kenar → 324 blok → 162 çift > 156 bit
STRENGTH     = 80
SECRET_KEY   = 42
MAGIC        = 0xB8A3
CODE_CHARS   = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CODE_LEN     = 6
BITS_PER_CHAR = 6
PAYLOAD_BITS = 16 + CODE_LEN * BITS_PER_CHAR   # 52 bit
ECC_REPEAT   = 3         # 52*3=156 bit, 4 tile = toplam 12x redundancy

# Quadrant pozisyonları (x, y) başlangıç
QUADRANTS = [(0, 0), (TILE_SIZE, 0), (0, TILE_SIZE), (TILE_SIZE, TILE_SIZE)]


# ─── KOD DÖNÜŞÜMÜ ────────────────────────────────────────────────────────────

def _code_to_bits(code: str) -> list:
    code = (code.upper() + " " * CODE_LEN)[:CODE_LEN]
    bits = [(MAGIC >> i) & 1 for i in range(15, -1, -1)]
    for char in code:
        idx = CODE_CHARS.find(char)
        if idx < 0: idx = 0
        for i in range(BITS_PER_CHAR - 1, -1, -1):
            bits.append((idx >> i) & 1)
    return bits


def _bits_to_code(bits: list) -> Optional[str]:
    if len(bits) < PAYLOAD_BITS:
        return None
    magic = sum(bits[i] << (15 - i) for i in range(16))
    if magic != MAGIC:
        return None
    code = ""
    for c in range(CODE_LEN):
        start = 16 + c * BITS_PER_CHAR
        idx = sum(bits[start + i] << (BITS_PER_CHAR - 1 - i) for i in range(BITS_PER_CHAR))
        code += CODE_CHARS[idx] if idx < len(CODE_CHARS) else "A"
    return code.strip()


def _add_ecc(bits: list) -> list:
    return [b for b in bits for _ in range(ECC_REPEAT)]


def _decode_ecc(bits: list) -> list:
    result = []
    for i in range(0, len(bits) - ECC_REPEAT + 1, ECC_REPEAT):
        result.append(1 if sum(bits[i:i+ECC_REPEAT]) > ECC_REPEAT // 2 else 0)
    return result


# ─── BLOK DİFERANSİYEL ───────────────────────────────────────────────────────

def _get_block_pairs(n_bits: int) -> list:
    rng = np.random.RandomState(SECRET_KEY)
    nb = TILE_SIZE // BLOCK_SIZE          # blok/kenar
    total = nb * nb
    idx = rng.permutation(total)
    ga, gb = idx[:total//2], idx[total//2:]
    rng.shuffle(ga); rng.shuffle(gb)
    pairs = []
    for i in range(n_bits):
        a = ga[i % len(ga)]; b = gb[i % len(gb)]
        pairs.append(((a // nb * BLOCK_SIZE, a % nb * BLOCK_SIZE),
                      (b // nb * BLOCK_SIZE, b % nb * BLOCK_SIZE)))
    return pairs


def _embed_bits_in_tile(y: np.ndarray, bits: list, ox: int, oy: int):
    pairs = _get_block_pairs(len(bits))
    for (pa, pb), bit in zip(pairs, bits):
        ra, ca = pa[0]+oy, pa[1]+ox
        rb, cb = pb[0]+oy, pb[1]+ox
        ba = y[ra:ra+BLOCK_SIZE, ca:ca+BLOCK_SIZE].astype(np.float32)
        bb = y[rb:rb+BLOCK_SIZE, cb:cb+BLOCK_SIZE].astype(np.float32)
        diff = ba.mean() - bb.mean()
        q = round(diff / STRENGTH)
        if bit == 1:
            if q % 2 == 0: q += 1
        else:
            if q % 2 != 0: q += 1
        delta = (q * STRENGTH - diff) / 2.0
        y[ra:ra+BLOCK_SIZE, ca:ca+BLOCK_SIZE] = np.clip(ba + delta, 0, 255)
        y[rb:rb+BLOCK_SIZE, cb:cb+BLOCK_SIZE] = np.clip(bb - delta, 0, 255)


def _decode_tile(y_tile: np.ndarray) -> Optional[str]:
    """512x512 Y kanalından kod çöz."""
    pairs = _get_block_pairs(PAYLOAD_BITS * ECC_REPEAT)
    raw = []
    for (pa, pb) in pairs:
        ra, ca = pa; rb, cb = pb
        diff = y_tile[ra:ra+BLOCK_SIZE, ca:ca+BLOCK_SIZE].mean() - \
               y_tile[rb:rb+BLOCK_SIZE, cb:cb+BLOCK_SIZE].mean()
        raw.append(abs(round(diff / STRENGTH)) % 2)
    return _bits_to_code(_decode_ecc(raw)[:PAYLOAD_BITS])


def _to_y(arr_rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)


def _apply_gamma(arr: np.ndarray, gamma: float) -> np.ndarray:
    lut = np.clip((np.arange(256)/255.0)**(1.0/gamma)*255, 0, 255).astype(np.uint8)
    return lut[arr]


# ─── PUBLIC API ──────────────────────────────────────────────────────────────

def encode(image: Image.Image, code: str) -> Image.Image:
    """Görsele kodu 4 quadrant'a göm."""
    img = image.convert("RGB")
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2))
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)

    bits = _add_ecc(_code_to_bits(code))

    for (ox, oy) in QUADRANTS:
        _embed_bits_in_tile(y, bits, ox, oy)

    ycrcb[:, :, 0] = np.clip(y, 0, 255).astype(np.uint8)
    bgr_enc = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    psnr = cv2.PSNR(bgr, bgr_enc)
    print(f"[WM] encode code={code} PSNR={psnr:.1f}dB (4-quadrant)")
    return Image.fromarray(cv2.cvtColor(bgr_enc, cv2.COLOR_BGR2RGB))


def decode(image: Image.Image) -> Optional[str]:
    """
    Kameradan gelen görselden kod oku.
    9 scale × 4 quadrant × 5 gamma = 180 deneme.
    """
    img = image.convert("RGB")
    w, h = img.size

    # Hafif blur (moire giderme)
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    bgr = cv2.GaussianBlur(bgr, (3, 3), 0.8)
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    w, h = img.size

    def _try(img_src, label=""):
        ws, hs = img_src.size
        for ratio in [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]:
            side = int(min(ws, hs) * ratio)
            left, top = (ws-side)//2, (hs-side)//2
            crop = img_src.crop((left, top, left+side, top+side))
            full = crop.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            y_full = _to_y(np.array(full))

            for qi, (ox, oy) in enumerate(QUADRANTS):
                y_tile = y_full[oy:oy+TILE_SIZE, ox:ox+TILE_SIZE]
                code = _decode_tile(y_tile)
                if code:
                    print(f"[WM] decode{label} ratio={ratio} Q{qi}")
                    return code
        return None

    # Gamma=1.0 (dijital / hafif bozulma)
    code = _try(img)
    if code: return code

    # Gamma varyantları (ekran-kamera transfer)
    for gamma in [0.75, 0.85, 1.2, 1.5]:
        arr_g = _apply_gamma(np.array(img), gamma)
        img_g = Image.fromarray(arr_g)
        code = _try(img_g, f" gamma={gamma}")
        if code: return code

    print("[WM] decode result=None")
    return None
