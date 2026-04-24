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

from brandion_engine import CAMERA_PAYLOAD_BITS, bits_to_int

# ─── SABITLER ────────────────────────────────────────────────────────────────

DECODE_W = 128
DECODE_H = 128
PAYLOAD_BITS = CAMERA_PAYLOAD_BITS  # 8 bit — wm_id 0-255


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
    128×128 grayscale (Y kanalı) → 5 katman CNN → PAYLOAD_BITS
    DCT watermark Y kanalında — grayscale yeterli, 3x hızlı.
    """

    def __init__(self, payload_bits: int = PAYLOAD_BITS):
        super().__init__()
        self.payload_bits = payload_bits
        self.encoder = nn.Sequential(
            ConvBNReLU(1, 16),
            nn.MaxPool2d(2),          # 128→64
            ConvBNReLU(16, 32),
            nn.MaxPool2d(2),          # 64→32
            ConvBNReLU(32, 64),
            nn.MaxPool2d(2),          # 32→16
            ConvBNReLU(64, 128),
            nn.MaxPool2d(2),          # 16→8
            ConvBNReLU(128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, payload_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) in [0,1] → (B, payload_bits) in [0,1]"""
        feat = self.encoder(x).view(x.size(0), -1)   # (B, 128)
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
        """PIL image → wm_id (0-255) veya None."""
        x = _pil_to_tensor(img).unsqueeze(0)
        bits = self.decode_tensor(x)
        if len(bits) >= PAYLOAD_BITS:
            return bits_to_int(bits[:PAYLOAD_BITS])
        return None

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
    """PIL → (1, DECODE_H, DECODE_W) float32 grayscale tensor in [0,1].
    Y kanalı kullan — DCT watermark orada."""
    img = img.convert("YCbCr").resize((DECODE_W, DECODE_H), Image.LANCZOS)
    y = np.array(img, dtype=np.float32)[:, :, 0] / 255.0  # Y channel only
    return torch.from_numpy(y).unsqueeze(0)  # (1, H, W)


# No validation needed — caller checks DB for valid wm_id
