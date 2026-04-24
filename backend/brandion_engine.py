"""
BRANDION Steganography Engine
==============================
Gorünmez veri gömme ve çözme — video frame'leri için.

Yöntem: DCT (Discrete Cosine Transform) tabanlı frekans alanı steganografi
- İnsan gözü yüksek frekanslara duyarsız → biz oraya yazıyoruz
- JPEG/H.264 sıkıştırmaya dayanıklı (orta frekans bandı)
- 56+ bit payload — her içerik için benzersiz ID yeterli

Referans: StegaStamp (Tancik et al., UC Berkeley 2019)
BRANDION adaptasyonu: video frame + TV ekranı + mobil kamera pipeline
"""

import numpy as np
import cv2
from scipy.fftpack import dct, idct
import hashlib
import struct
import json
from datetime import datetime


# ─── SABITLER ────────────────────────────────────────────────────────────────

BLOCK_SIZE = 8          # DCT blok boyutu (JPEG standardı)
PAYLOAD_BITS = 56       # StegaStamp referans bit sayısı
STRENGTH = 18.0         # Gömme gücü (yüksek = dayanıklı ama görünür)
FREQ_BAND = (3, 6)      # Orta frekans bandı — sıkıştırmaya dayanıklı
MAGIC = 0xB8A3          # BRANDION imzası (16 bit)


# ─── BIT OPERASYONLARI ───────────────────────────────────────────────────────

def text_to_bits(text: str) -> list[int]:
    """Metni bit listesine çevirir."""
    bits = []
    for char in text.encode('utf-8'):
        for i in range(7, -1, -1):
            bits.append((char >> i) & 1)
    return bits


def bits_to_text(bits: list[int]) -> str:
    """Bit listesini metne çevirir."""
    chars = []
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        if byte == 0:
            break
        chars.append(chr(byte))
    return ''.join(chars)


def int_to_bits(value: int, n_bits: int) -> list[int]:
    """Integer'ı sabit uzunlukta bit listesine çevirir."""
    return [(value >> i) & 1 for i in range(n_bits - 1, -1, -1)]


def bits_to_int(bits: list[int]) -> int:
    """Bit listesini integer'a çevirir."""
    result = 0
    for b in bits:
        result = (result << 1) | b
    return result


def add_error_correction(bits: list[int]) -> list[int]:
    """Basit tekrar kodu ile hata düzeltme (3x tekrar)."""
    return [b for b in bits for _ in range(3)]


def decode_error_correction(bits: list[int]) -> list[int]:
    """3x tekrar kodunu çözer — majority vote."""
    result = []
    for i in range(0, len(bits) - 2, 3):
        votes = bits[i] + bits[i+1] + bits[i+2]
        result.append(1 if votes >= 2 else 0)
    return result


def build_payload(content_id: str, scene_id: int, timestamp: int) -> list[int]:
    """
    BRANDION payload formatı (56 bit):
    [16 bit magic][16 bit content_hash][8 bit scene_id][16 bit timestamp_short]
    """
    magic_bits = int_to_bits(MAGIC, 16)
    
    content_hash = int(hashlib.md5(content_id.encode()).hexdigest()[:4], 16)
    content_bits = int_to_bits(content_hash, 16)
    
    scene_bits = int_to_bits(scene_id & 0xFF, 8)
    
    ts_short = timestamp & 0xFFFF
    ts_bits = int_to_bits(ts_short, 16)
    
    payload = magic_bits + content_bits + scene_bits + ts_bits
    assert len(payload) == PAYLOAD_BITS
    return payload


def verify_payload(bits: list[int]) -> dict | None:
    """Payload'ı doğrular ve parse eder."""
    if len(bits) < PAYLOAD_BITS:
        return None
    
    magic = bits_to_int(bits[0:16])
    if magic != MAGIC:
        return None
    
    content_hash = bits_to_int(bits[16:32])
    scene_id = bits_to_int(bits[32:40])
    ts_short = bits_to_int(bits[40:56])
    
    return {
        "magic": hex(magic),
        "content_hash": hex(content_hash),
        "scene_id": scene_id,
        "timestamp_short": ts_short,
        "verified": True
    }


# ─── DCT STEGANOGRAFI ────────────────────────────────────────────────────────

def embed_bit_in_block(block: np.ndarray, bit: int, strength: float) -> np.ndarray:
    """
    Tek bir 8x8 DCT bloğuna 1 bit gömer.
    Orta frekans katsayılarını modifiye eder.
    """
    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    r, c = FREQ_BAND
    coeff = dct_block[r, c]
    
    # Katsayıyı quantize et ve bit'e göre modifiye et
    q = strength
    coeff_q = np.round(coeff / q)
    
    if bit == 1:
        # Tek sayıya yuvarla
        if int(coeff_q) % 2 == 0:
            coeff_q += 1
    else:
        # Çift sayıya yuvarla
        if int(coeff_q) % 2 == 1:
            coeff_q += 1
    
    dct_block[r, c] = coeff_q * q
    
    # Geri dönüştür
    result = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
    return np.clip(result, 0, 255)


def extract_bit_from_block(block: np.ndarray, strength: float) -> int:
    """Bir 8x8 DCT bloğundan 1 bit çıkarır."""
    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    r, c = FREQ_BAND
    coeff = dct_block[r, c]
    
    coeff_q = int(np.round(coeff / strength))
    return abs(coeff_q) % 2


def get_embedding_positions(frame_h: int, frame_w: int, n_bits: int, seed: int = 42) -> list[tuple]:
    """
    Bit gömme pozisyonlarını pseudorandom olarak seçer.
    Seed = gizli anahtar → şifreli pozisyon haritası.
    """
    rng = np.random.RandomState(seed)
    
    n_blocks_h = frame_h // BLOCK_SIZE
    n_blocks_w = frame_w // BLOCK_SIZE
    total_blocks = n_blocks_h * n_blocks_w
    
    if total_blocks < n_bits:
        raise ValueError(f"Frame çok küçük: {total_blocks} blok var, {n_bits} bit gerekli")
    
    positions = rng.choice(total_blocks, n_bits, replace=False)
    result = []
    for pos in positions:
        row = (pos // n_blocks_w) * BLOCK_SIZE
        col = (pos % n_blocks_w) * BLOCK_SIZE
        result.append((row, col))
    return result


# ─── ANA FONKSİYONLAR ────────────────────────────────────────────────────────

def encode_frame(
    frame: np.ndarray,
    content_id: str,
    scene_id: int = 1,
    timestamp: int = None,
    strength: float = STRENGTH,
    secret_key: int = 42
) -> tuple[np.ndarray, dict]:
    """
    Video frame'e görünmez BRANDION payload gömer.
    
    Args:
        frame: BGR video frame (numpy array)
        content_id: İçerik kimliği (dizi adı, bölüm vs.)
        scene_id: Sahne numarası
        timestamp: Unix timestamp (None ise şimdiki zaman)
        strength: Gömme gücü
        secret_key: Pozisyon şifreleme anahtarı
    
    Returns:
        (encoded_frame, metadata)
    """
    if timestamp is None:
        timestamp = int(datetime.now().timestamp())
    
    # YCbCr'ye çevir — sadece Y (parlaklık) kanalına göm
    ycbcr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2YCrCb)
    y_channel = ycbcr[:, :, 0].astype(np.float64)
    
    h, w = y_channel.shape
    
    # Payload oluştur
    payload = build_payload(content_id, scene_id, timestamp)
    
    # Hata düzeltme ekle
    payload_ecc = add_error_correction(payload)
    
    # Pozisyonları hesapla
    positions = get_embedding_positions(h, w, len(payload_ecc), seed=secret_key)
    
    # Her biti göm
    y_encoded = y_channel.copy()
    for (row, col), bit in zip(positions, payload_ecc):
        block = y_encoded[row:row+BLOCK_SIZE, col:col+BLOCK_SIZE]
        if block.shape == (BLOCK_SIZE, BLOCK_SIZE):
            y_encoded[row:row+BLOCK_SIZE, col:col+BLOCK_SIZE] = embed_bit_in_block(block, bit, strength)
    
    # Geri BGR'ye çevir
    ycbcr_encoded = ycbcr.copy()
    ycbcr_encoded[:, :, 0] = np.clip(y_encoded, 0, 255).astype(np.uint8)
    encoded_frame = cv2.cvtColor(ycbcr_encoded, cv2.COLOR_YCrCb2BGR)
    
    # PSNR hesapla (görsel kalite ölçütü)
    psnr = cv2.PSNR(frame, encoded_frame)
    
    metadata = {
        "content_id": content_id,
        "scene_id": scene_id,
        "timestamp": timestamp,
        "payload_bits": len(payload),
        "payload_with_ecc": len(payload_ecc),
        "psnr_db": round(psnr, 2),
        "strength": strength,
        "secret_key": secret_key,
        "brandion_magic": hex(MAGIC)
    }
    
    return encoded_frame, metadata


def decode_frame(
    frame: np.ndarray,
    strength: float = STRENGTH,
    secret_key: int = 42
) -> dict:
    """
    Video frame'den BRANDION payload çözer.
    
    Args:
        frame: Potansiyel olarak encode edilmiş BGR frame
        strength: Gömme gücü (encode ile aynı olmalı)
        secret_key: Pozisyon şifreleme anahtarı (encode ile aynı)
    
    Returns:
        Decoded payload dict veya {"verified": False}
    """
    ycbcr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2YCrCb)
    y_channel = ycbcr[:, :, 0].astype(np.float64)
    
    h, w = y_channel.shape
    
    n_bits_with_ecc = PAYLOAD_BITS * 3  # 3x hata düzeltme
    positions = get_embedding_positions(h, w, n_bits_with_ecc, seed=secret_key)
    
    # Bitleri çıkar
    raw_bits = []
    for (row, col) in positions:
        block = y_channel[row:row+BLOCK_SIZE, col:col+BLOCK_SIZE]
        if block.shape == (BLOCK_SIZE, BLOCK_SIZE):
            raw_bits.append(extract_bit_from_block(block, strength))
    
    # Hata düzeltme uygula
    corrected_bits = decode_error_correction(raw_bits)
    
    # Payload doğrula
    result = verify_payload(corrected_bits[:PAYLOAD_BITS])
    
    if result:
        result["raw_bits_sample"] = corrected_bits[:16]
        return result
    
    return {"verified": False, "reason": "Magic byte eslesmiyor — BRANDION imzasi bulunamadi"}


def compute_invisibility_score(original: np.ndarray, encoded: np.ndarray) -> dict:
    """
    Görünmezlik metriklerini hesaplar.
    PSNR > 40dB = insan gözüyle ayırt edilemez.
    """
    psnr = cv2.PSNR(original, encoded)
    
    diff = cv2.absdiff(original, encoded).astype(np.float64)
    mse = np.mean(diff ** 2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Görünmezlik skoru (0-100)
    invisibility = min(100, max(0, (psnr - 30) * 10))
    
    return {
        "psnr_db": round(psnr, 2),
        "mse": round(mse, 4),
        "max_pixel_diff": int(max_diff),
        "mean_pixel_diff": round(mean_diff, 3),
        "invisibility_score": round(invisibility, 1),
        "human_detectable": psnr < 30,
        "rating": "Mukemmel" if psnr > 45 else "Iyi" if psnr > 38 else "Kabul edilebilir" if psnr > 30 else "Gorunur"
    }
