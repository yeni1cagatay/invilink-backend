"""
Temporal Brightness Watermark — Screen-to-Camera
=================================================
Her frame'in Y kanalı ortalamasını ±DELTA ile modüle eder.
Kamera auto-exposure'a karşı dayanıklı: median baseline çıkarımı kullanır.

Encoding:
  Bit 1 → Y += DELTA
  Bit 0 → Y -= DELTA
  FRAMES_PER_BIT = 2  (daha sağlam)

Sequence per cycle (64 frames = ~2.1s @ 30fps):
  [MAGIC 8 bit] [ID 8 bit] × ECC_REPEAT 2 = 32 bit × 2 frame = 64 frame

Decode:
  Telefon 90 frame (~3s) yakalar.
  Median baseline çıkarılır (video içeriği).
  Binarize → sliding window → MAGIC check → ID.
"""

from __future__ import annotations
import numpy as np
import cv2
from PIL import Image
from typing import Optional, List

# ─── SABITLER ────────────────────────────────────────────────────────────────

VIDEO_FPS    = 60           # encode sırasında kullanılan video fps
MAGIC        = 0b10101010   # 8-bit sync pattern
MAGIC_BITS   = 8
ID_BITS      = 8            # 256 unique ID/cycle
ECC_REPEAT   = 2            # her bit 2 kez
FRAMES_PER_BIT = 2          # her bit = 2 frame
DELTA        = 40           # luma ±40 (kameradan güvenilir okunur)

PAYLOAD_BITS = MAGIC_BITS + ID_BITS          # 16
TOTAL_BITS   = PAYLOAD_BITS * ECC_REPEAT     # 32
FRAMES_PER_CYCLE = TOTAL_BITS * FRAMES_PER_BIT  # 64 frame ≈ 2.1s @ 30fps

# ─── BIT DÖNÜŞÜMÜ ────────────────────────────────────────────────────────────

def _id_to_bits(id_val: int) -> list:
    """ID → bit listesi (ECC ile)."""
    magic_bits = [(MAGIC >> (7 - i)) & 1 for i in range(MAGIC_BITS)]
    id_bits    = [(id_val >> (7 - i)) & 1 for i in range(ID_BITS)]
    raw = magic_bits + id_bits
    return [b for b in raw for _ in range(ECC_REPEAT)]  # ECC: her biti tekrarla


def _bits_to_id(bits: list) -> Optional[int]:
    """Bit listesi → ID (MAGIC kontrol + ECC decode)."""
    if len(bits) < TOTAL_BITS:
        return None

    # ECC: çoğunluk oyu
    decoded = []
    for i in range(0, TOTAL_BITS, ECC_REPEAT):
        vote = sum(bits[i:i + ECC_REPEAT])
        decoded.append(1 if vote > ECC_REPEAT // 2 else 0)

    # MAGIC kontrol
    magic_check = sum(decoded[i] << (MAGIC_BITS - 1 - i) for i in range(MAGIC_BITS))
    if magic_check != MAGIC:
        return None

    id_val = sum(decoded[MAGIC_BITS + i] << (ID_BITS - 1 - i) for i in range(ID_BITS))
    return id_val


# ─── ENCODE ──────────────────────────────────────────────────────────────────

def encode_frame(frame_rgb: np.ndarray, bit: int) -> np.ndarray:
    """
    Sol/sağ diferansiyel: bit=1 → sol +DELTA / sağ -DELTA
                          bit=0 → sol -DELTA / sağ +DELTA
    Net ortalama parlaklık değişimi = 0  (global flicker yok)
    """
    arr = frame_rgb.astype(np.float32)
    w   = arr.shape[1]
    mid = w // 2
    delta = DELTA if bit == 1 else -DELTA
    arr[:, :mid]  = np.clip(arr[:, :mid]  + delta, 0, 255)
    arr[:, mid:]  = np.clip(arr[:, mid:]  - delta, 0, 255)
    return arr.astype(np.uint8)


def encode_frames(frames: List[Image.Image], id_val: int) -> List[Image.Image]:
    """
    PIL Image listesine ID göm.
    Cycle her FRAMES_PER_CYCLE frame'de bir tekrar eder.
    """
    bits = _id_to_bits(id_val)
    result = []
    for i, frame in enumerate(frames):
        bit_idx = (i // FRAMES_PER_BIT) % len(bits)
        bit = bits[bit_idx]
        arr = np.array(frame.convert("RGB"))
        result.append(Image.fromarray(encode_frame(arr, bit)))
    return result


def encode_video(input_path: str, output_path: str, id_val: int, fps: float = 30.0):
    """
    Video dosyasına ID göm.
    input_path → output_path (aynı format)
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Video açılamadı: {input_path}")

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, actual_fps, (w, h))

    bits = _id_to_bits(id_val)
    frame_idx = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bit_idx = (frame_idx // FRAMES_PER_BIT) % len(bits)
        rgb_wm = encode_frame(rgb, bits[bit_idx])
        out.write(cv2.cvtColor(rgb_wm, cv2.COLOR_RGB2BGR))
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[TWM] encode id={id_val} frames={frame_idx} → {output_path}")


# ─── DECODE ──────────────────────────────────────────────────────────────────

def _frame_differential(frame_rgb: np.ndarray) -> float:
    """RGB frame → sol_ortalama - sağ_ortalama (Y kanalı)."""
    bgr   = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    y     = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    mid   = y.shape[1] // 2
    return float(y[:, :mid].mean() - y[:, mid:].mean())


def _frame_brightness(frame_rgb: np.ndarray) -> float:
    """Geriye dönük uyumluluk için."""
    return _frame_differential(frame_rgb)


def decode_brightness_series(brightness: list, frames_per_bit: int = FRAMES_PER_BIT) -> Optional[int]:
    """
    Frame parlaklıkları → ID.
    Tüm geçerli pencerelerde decode yapar, majority vote ile ID döner.
    MAGIC periyodikliğinden kaynaklanan yanlış offset sorununu önler.
    frames_per_bit: kamera fps != video fps ise 1 geç.
    """
    if len(brightness) < TOTAL_BITS * frames_per_bit:
        return None

    arr = np.array(brightness, dtype=np.float32)

    n_groups = len(arr) // frames_per_bit
    if n_groups < TOTAL_BITS:
        return None
    groups = np.array([
        arr[i * frames_per_bit:(i + 1) * frames_per_bit].mean()
        for i in range(n_groups)
    ])

    magic_ecc_bits = [b for b in [(MAGIC >> (7 - i)) & 1 for i in range(MAGIC_BITS)]
                      for _ in range(ECC_REPEAT)]
    template = np.array([1.0 if b == 1 else -1.0 for b in magic_ecc_bits])
    magic_len = len(template)
    threshold = magic_len * 0.25

    # Her faz (0..TOTAL_BITS-1) için cycle-aligned pencereleri dene.
    # Doğru faz: tüm pencereleri aynı ID'e decode eder → en yüksek tutarlılık.
    best_phase_id = None
    best_score = -1

    for phase in range(TOTAL_BITS):
        phase_votes: dict = {}
        offsets = range(phase, len(groups) - TOTAL_BITS + 1, TOTAL_BITS)
        for offset in offsets:
            window = groups[offset:offset + TOTAL_BITS]
            window_c = window - window.mean()
            std = window_c.std()
            if std < 0.5:
                continue
            window_n = window_c / std
            corr = float(np.dot(window_n[:magic_len], template))
            if corr < threshold:
                continue
            bits = [1 if v > 0 else 0 for v in window_c]
            id_val = _bits_to_id(bits)
            if id_val is not None:
                phase_votes[id_val] = phase_votes.get(id_val, 0) + 1

        if not phase_votes:
            continue
        top_id = max(phase_votes, key=lambda k: phase_votes[k])
        score = phase_votes[top_id]
        if score > best_score:
            best_score = score
            best_phase_id = top_id

    if best_phase_id is None:
        print("[TWM] hiç geçerli decode yok → None")
        return None

    print(f"[TWM] phase-aligned majority → id={best_phase_id} (score={best_score})")
    return best_phase_id


def decode_frames(frames: List[Image.Image]) -> Optional[int]:
    """PIL Image listesinden ID çöz (spatial differential)."""
    diffs = [_frame_differential(np.array(f.convert("RGB"))) for f in frames]
    return decode_brightness_series(diffs)


def decode_video(video_path: str, source_fps: float = None) -> Optional[int]:
    """
    Video dosyasından ID çöz.
    source_fps: kamera fps'i (bilinmiyorsa None → video'dan okur).
    Kamera 30fps + encode 60fps ise frames_per_bit=1 ile dener.
    """
    cap = cv2.VideoCapture(video_path)
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    brightness = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        brightness.append(_frame_brightness(rgb))
    cap.release()
    print(f"[TWM] decode_video frames={len(brightness)} fps={cam_fps:.1f}")

    # Video fps'ine göre frames_per_bit hesapla; her iki değeri de dene
    fpb_native = max(1, round(cam_fps / (VIDEO_FPS / FRAMES_PER_BIT)))
    candidates = sorted({1, FRAMES_PER_BIT, fpb_native})
    for fpb in candidates:
        result = decode_brightness_series(brightness, frames_per_bit=fpb)
        if result is not None:
            print(f"[TWM] decoded with frames_per_bit={fpb}")
            return result
    return None
