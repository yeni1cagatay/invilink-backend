"""
HiDDeN-style Neural Decoder
============================
DCT encoder aynen kalır — sadece decode tarafı CNN olur.
Kamera → ekran kanalına karşı robust: JPEG + blur + perspektif + gamma.

Kullanım:
  decoder = HiDDeNDecoder.load("hidden_decoder.pt")
  wm_id   = decoder.decode_image(pil_img)   # None veya 0-255
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from brandion_engine import PAYLOAD_BITS, bits_to_int

# ─── SABITLER ────────────────────────────────────────────────────────────────

DECODE_W = 256   # decoder input size — küçük = hızlı, 256 yeterli
DECODE_H = 256
MAGIC    = 0xB8A3


# ─── MİMARİ ──────────────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HiDDeNDecoder(nn.Module):
    """
    7x ConvBNReLU(64) → AdaptiveAvgPool(1,1) → Linear(PAYLOAD_BITS)
    HiDDeN paper'dan uyarlandı (Zhu et al. 2018).
    """

    def __init__(self, payload_bits: int = PAYLOAD_BITS):
        super().__init__()
        self.payload_bits = payload_bits
        self.encoder = nn.Sequential(
            ConvBNReLU(3, 64),
            ConvBNReLU(64, 64),
            ConvBNReLU(64, 64),
            ConvBNReLU(64, 64),
            ConvBNReLU(64, 64),
            ConvBNReLU(64, 64),
            ConvBNReLU(64, 64),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, payload_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) in [0,1] → (B, payload_bits) in [0,1]"""
        feat = self.encoder(x).view(x.size(0), -1)
        return torch.sigmoid(self.fc(feat))

    # ─── INFERENCE ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def decode_tensor(self, x: torch.Tensor) -> list[int]:
        """(B, 3, H, W) → list of raw bits (thresholded at 0.5)."""
        self.eval()
        preds = self(x)  # (B, 56)
        bits = (preds > 0.5).long().squeeze(0).tolist()
        return bits

    @torch.no_grad()
    def decode_image(self, img: Image.Image) -> Optional[int]:
        """PIL image → wm_id veya None (magic+checksum doğrulaması ile)."""
        x = _pil_to_tensor(img).unsqueeze(0)  # (1, 3, H, W)
        bits = self.decode_tensor(x)
        return _validate_bits(bits)

    # ─── SAVE / LOAD ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        torch.save({"state_dict": self.state_dict(),
                    "payload_bits": self.payload_bits}, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "HiDDeNDecoder":
        ck = torch.load(path, map_location=device, weights_only=False)
        model = cls(payload_bits=ck["payload_bits"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        return model


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL → (3, DECODE_H, DECODE_W) float32 tensor in [0,1]."""
    img = img.convert("RGB").resize((DECODE_W, DECODE_H), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))  # CHW


def _validate_bits(bits: list[int]) -> Optional[int]:
    """56-bit payload → wm_id veya None."""
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
