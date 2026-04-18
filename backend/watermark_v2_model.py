"""
Brandion Filigran v2 — Neural Network Watermark
=================================================
Encoder + Decoder CNN çifti.
Kamera yoluna dayanikli: JPEG, gamma, gurultu, scale ile egitilir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """
    Görsel + 52-bit mesaji alir, watermarkli görsel döner.
    Çikti = girdi + küçük residual (görünmezlik için).
    """

    def __init__(self, msg_bits: int = 52, img_size: int = 256):
        super().__init__()
        self.img_size = img_size
        self.msg_bits = msg_bits

        # Mesaj → uzamsal harita (img_size x img_size)
        self.msg_expand = nn.Sequential(
            nn.Linear(msg_bits, img_size * img_size // 16),
            nn.ReLU(),
            nn.Linear(img_size * img_size // 16, img_size * img_size),
        )

        # Görsel + mesaj haritası → residual
        self.net = nn.Sequential(
            ConvBnRelu(4, 32),
            ConvBnRelu(32, 32),
            ConvBnRelu(32, 64),
            ConvBnRelu(64, 64),
            ConvBnRelu(64, 32),
            ConvBnRelu(32, 16),
            nn.Conv2d(16, 3, 1),
            nn.Tanh(),
        )

    def forward(self, img, msg):
        """
        img: (B, 3, H, W) — [0, 1] araliginda
        msg: (B, msg_bits) — {0, 1}
        döner: watermarkli img [0, 1]
        """
        B, _, H, W = img.shape
        msg_f = msg.float() * 2 - 1   # {0,1} → {-1,+1}
        msg_map = self.msg_expand(msg_f).view(B, 1, H, W)
        x = torch.cat([img, msg_map], dim=1)   # (B, 4, H, W)
        residual = self.net(x) * 0.1            # küçük değişiklik
        return torch.clamp(img + residual, 0, 1)


class Decoder(nn.Module):
    """
    Bozulmuş görselden 52-bit mesaji çözer.
    """

    def __init__(self, msg_bits: int = 52):
        super().__init__()
        self.msg_bits = msg_bits

        self.net = nn.Sequential(
            ConvBnRelu(3, 32),
            ConvBnRelu(32, 32),
            ConvBnRelu(32, 64),
            ConvBnRelu(64, 64),
            ConvBnRelu(64, 128),
            ConvBnRelu(128, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, msg_bits)

    def forward(self, img):
        """
        img: (B, 3, H, W) — [0, 1]
        döner: logits (B, msg_bits) — sigmoid ile bit olasılığı
        """
        feat = self.net(img).squeeze(-1).squeeze(-1)
        return self.head(feat)
