"""
Brandion Watermark Models — PyTorch
=====================================
StegaStamp mimarisinin Brandion için PyTorch portu.

Encoder : U-Net (image + secret_bits → residual)
Decoder : CNN + Spatial Transformer (distorted image → bits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SECRET_SIZE = 100   # StegaStamp uyumlu; ilk 8 bit = wm_id (0-255)
IMG_SIZE    = 400


def _cbr(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ─── ENCODER ─────────────────────────────────────────────────────────────────

class BrandionEncoder(nn.Module):
    """
    (image, secret) → residual
    encoded = clamp(image + residual, 0, 1)

    secret_bits → FC → reshape (3,50,50) → bilinear upsample → concat with image
    Sonra U-Net ile residual öğrenir.
    """

    def __init__(self, secret_size: int = SECRET_SIZE):
        super().__init__()
        self.secret_fc = nn.Sequential(
            nn.Linear(secret_size, 50 * 50 * 3),
            nn.ReLU(inplace=True),
        )
        # Encoder path  (6 = 3 image + 3 secret)
        self.c1 = _cbr(6,   32)
        self.c2 = _cbr(32,  32,  stride=2)
        self.c3 = _cbr(32,  64,  stride=2)
        self.c4 = _cbr(64,  128, stride=2)
        self.c5 = _cbr(128, 256, stride=2)
        # Decoder path
        self.u6 = _cbr(256, 128)
        self.d6 = _cbr(256, 128)   # concat with c4
        self.u7 = _cbr(128, 64)
        self.d7 = _cbr(128, 64)    # concat with c3
        self.u8 = _cbr(64,  32)
        self.d8 = _cbr(64,  32)    # concat with c2
        self.u9 = _cbr(32,  32)
        self.d9 = _cbr(32 + 32 + 6, 32)   # concat with c1 + input
        self.d10 = _cbr(32, 32)
        self.out = nn.Conv2d(32, 3, 1)

    def forward(self, image: torch.Tensor, secret: torch.Tensor) -> torch.Tensor:
        B = image.size(0)
        sec = self.secret_fc(secret).view(B, 3, 50, 50)
        sec = F.interpolate(sec, size=image.shape[2:], mode='bilinear', align_corners=False)

        x  = torch.cat([image - 0.5, sec], dim=1)   # normalize + concat
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)

        up = lambda t: F.interpolate(t, scale_factor=2, mode='bilinear', align_corners=False)

        d6  = self.d6(torch.cat([self.u6(up(c5)), c4], dim=1))
        d7  = self.d7(torch.cat([self.u7(up(d6)),  c3], dim=1))
        d8  = self.d8(torch.cat([self.u8(up(d7)),  c2], dim=1))
        d9  = self.d9(torch.cat([self.u9(up(d8)),  c1, x], dim=1))
        d10 = self.d10(d9)
        return self.out(d10)                          # residual (B,3,H,W)


# ─── DECODER ─────────────────────────────────────────────────────────────────

class _STN(nn.Module):
    """Spatial Transformer: kameradan gelen perspektif bozulmasını düzelt."""

    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.loc = nn.Sequential(
            nn.Conv2d(in_ch, 32,  3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32,    64,  3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64,    128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(128, 6)
        # Identity başlangıç
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.fc(self.loc(x)).view(-1, 2, 3)
        grid  = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)


class BrandionDecoder(nn.Module):
    """
    (distorted encoded image) → secret_bits (logits)
    STN önce perspektifi düzeltir, sonra CNN bitleri okur.
    """

    def __init__(self, secret_size: int = SECRET_SIZE):
        super().__init__()
        self.stn = _STN()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,   32,  3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32,  32,  3,           padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32,  64,  3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64,  64,  3,           padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64,  64,  3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64,  128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 512), nn.ReLU(inplace=True),
            nn.Linear(512, secret_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stn(x - 0.5)
        return self.head(self.cnn(x))


# ─── INFERENCE HELPERS ───────────────────────────────────────────────────────

def wm_id_from_bits(logits: torch.Tensor) -> int:
    """Decoder çıktısından wm_id (0-255) çıkar — ilk 8 bit."""
    bits = (logits.sigmoid() > 0.5).long().squeeze(0)
    val  = 0
    for b in bits[:8]:
        val = (val << 1) | int(b.item())
    return val


def bits_from_wm_id(wm_id: int, secret_size: int = SECRET_SIZE) -> torch.Tensor:
    """wm_id → secret tensor (8 bit wm_id + sıfır padding)."""
    bits = [(wm_id >> (7 - i)) & 1 for i in range(8)]
    bits += [0] * (secret_size - 8)
    return torch.tensor(bits, dtype=torch.float)
