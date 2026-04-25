"""
TrustMark tabanlı watermark engine.

Yaklaşım: gri referans frame üzerine encode et, residual al.
Residual content-independent — herhangi bir frame üzerine additively uygulanabilir.
"""

import io
import numpy as np
from PIL import Image
from functools import lru_cache

ENCODE_W = 1920
ENCODE_H = 1080
_REF_GRAY = None


@lru_cache(maxsize=1)
def _get_tm():
    from trustmark import TrustMark
    return TrustMark(verbose=False, model_type="Q")


def _ref_gray(width: int = ENCODE_W, height: int = ENCODE_H) -> Image.Image:
    """Sabit gri referans frame (reproducible residual için)."""
    arr = np.full((height, width, 3), 128, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def encode_watermark(code: str, width: int = ENCODE_W, height: int = ENCODE_H) -> Image.Image:
    """
    8-char kodu referans gri frame'e embed et, residual noise döndür.
    Studio bu residual'ı 'Add' blend mode ile video üzerine uygular.
    """
    assert len(code) <= 8, f"Kod max 8 karakter olmalı, geldi: {len(code)}"
    tm = _get_tm()
    ref = _ref_gray(width, height)
    watermarked = tm.encode(ref, code, MODE="text")
    ref_arr = np.array(ref, dtype=np.int16)
    wm_arr = np.array(watermarked, dtype=np.int16)
    residual = np.clip(wm_arr - ref_arr + 128, 0, 255).astype(np.uint8)
    return Image.fromarray(residual, mode="RGB")


def decode_watermark(image: Image.Image) -> tuple[str | None, bool, float]:
    """
    Kameradan gelen frame'den kodu decode et.
    Returns: (code, detected, confidence)
    """
    tm = _get_tm()
    img_rgb = image.convert("RGB")
    if img_rgb.width > 1920:
        img_rgb = img_rgb.resize((1920, 1080), Image.LANCZOS)
    code, detected, confidence = tm.decode(img_rgb, MODE="text")
    return code, detected, float(confidence) if confidence is not None else 0.0


def generate_residual_png(code: str) -> bytes:
    """Residual noise PNG bytes döndür."""
    img = encode_watermark(code)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=False)
    return buf.getvalue()


def test_roundtrip(code: str) -> dict:
    """Encode → decode roundtrip testi."""
    tm = _get_tm()
    ref = _ref_gray(512, 512)
    watermarked = tm.encode(ref, code, MODE="text")
    decoded, detected, conf = tm.decode(watermarked, MODE="text")
    return {
        "original": code,
        "decoded": decoded,
        "detected": detected,
        "confidence": conf,
        "match": decoded == code,
    }
