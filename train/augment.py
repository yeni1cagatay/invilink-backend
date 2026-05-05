"""
Brandion Screen→Camera Augmentation Pipeline
=============================================
StegaStamp'ın transform_net'ini genişletir:
+ Moiré pattern (TV ekran ızgarası × kamera sensörü)
+ Geniş scale aralığı (izleyici 1-5m uzakta)
+ TV gamma eğrisi
+ Adaptif ramp (eğitim ilerledikçe bozulma artar)
"""

import io
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ─── YARDIMCI ────────────────────────────────────────────────────────────────

def _gaussian_kernel(size: int, sigma: float, channels: int, device) -> torch.Tensor:
    x     = torch.arange(size, dtype=torch.float, device=device) - size // 2
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    k     = gauss.view(1, 1, 1, size) * gauss.view(1, 1, size, 1)
    return k.expand(channels, 1, size, size)


def _blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma < 0.15:
        return img
    size   = int(sigma * 4) * 2 + 1
    kernel = _gaussian_kernel(size, sigma, img.size(1), img.device)
    return F.conv2d(img, kernel, padding=size // 2, groups=img.size(1))


def _jpeg(img: torch.Tensor, quality: int) -> torch.Tensor:
    """
    PIL JPEG round-trip. Gradient-friendly: JPEG'i detach ile işle,
    sonra STE (Straight-Through Estimator) ile farkı orijinale ekle.
    Encoder böylece JPEG bozulmasına karşı gradient alır.
    """
    with torch.no_grad():
        out = []
        for i in range(img.size(0)):
            arr = (img[i].permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, "JPEG", quality=quality)
            buf.seek(0)
            arr2 = np.array(Image.open(buf)).astype(np.float32) / 255.0
            out.append(torch.from_numpy(arr2).permute(2, 0, 1))
        compressed = torch.stack(out).to(img.device)
    # STE: forward = compressed, backward = img (gradient akıyor)
    return img + (compressed - img).detach()


# ─── BİREYSEL BOZULMALAR ─────────────────────────────────────────────────────

def perspective_warp(img: torch.Tensor, strength: float) -> torch.Tensor:
    """Rastgele affine + hafif perspektif simülasyonu."""
    B, C, H, W = img.shape
    s  = strength
    dx = random.uniform(-s, s)
    dy = random.uniform(-s, s)
    sc = random.uniform(1.0 - s * 0.5, 1.0 + s * 0.3)
    an = random.uniform(-s * 0.3, s * 0.3)

    cos_a, sin_a = math.cos(an), math.sin(an)
    theta = torch.tensor([
        [sc * cos_a, -sin_a, dx],
        [sin_a,  sc * cos_a, dy],
    ], dtype=torch.float, device=img.device).unsqueeze(0).expand(B, -1, -1)

    grid = F.affine_grid(theta, img.size(), align_corners=False)
    return F.grid_sample(img, grid, align_corners=False, padding_mode='reflection')


def scale_crop(img: torch.Tensor, scale: float) -> torch.Tensor:
    """Uzak izleyici simülasyonu: küçült → pad."""
    if abs(scale - 1.0) < 0.02:
        return img
    B, C, H, W = img.shape
    nh, nw = int(H * scale), int(W * scale)
    small  = F.interpolate(img, size=(nh, nw), mode='bilinear', align_corners=False)
    pad_h  = (H - nh) // 2
    pad_w  = (W - nw) // 2
    out    = torch.zeros_like(img)
    out[:, :, pad_h:pad_h + nh, pad_w:pad_w + nw] = small
    return out


def moire(img: torch.Tensor, strength: float) -> torch.Tensor:
    """
    TV ekranı + kamera sensörü interferans paterni.
    Yatay + dikey sinüs dalgalarının çarpımı.
    """
    B, C, H, W = img.shape
    fx = random.uniform(0.04, 0.18)
    fy = random.uniform(0.04, 0.18)
    px = random.uniform(0, 2 * math.pi)
    py = random.uniform(0, 2 * math.pi)

    xs = torch.linspace(0, W * fx * 2 * math.pi, W, device=img.device)
    ys = torch.linspace(0, H * fy * 2 * math.pi, H, device=img.device)
    pat = torch.sin(xs + px).view(1, 1, W) * torch.sin(ys + py).view(1, H, 1)
    pat = pat.unsqueeze(0).expand(B, C, H, W)
    return torch.clamp(img + strength * pat, 0, 1)


def tv_gamma(img: torch.Tensor, gamma: float) -> torch.Tensor:
    """Ekran parlaklık eğrisi + kamera exposure."""
    return torch.clamp(img.pow(gamma), 0, 1)


def color_shift(img: torch.Tensor, strength: float) -> torch.Tensor:
    """Kamera white balance kayması."""
    shift = torch.tensor(
        [random.uniform(-strength, strength) for _ in range(3)],
        device=img.device
    ).view(1, 3, 1, 1)
    return torch.clamp(img + shift, 0, 1)


# ─── ANA AUGMENTATION SINIFI ─────────────────────────────────────────────────

class ScreenCameraAugment:
    """
    Eğitim boyunca kademeli olarak artan bozulma pipeline'ı.
    step=0'da hiç bozulma yok, ramp_steps sonunda tam şiddette.

    Kullanım:
        aug = ScreenCameraAugment(step=current_step, total_steps=total)
        distorted = aug(encoded_batch)
    """

    def __init__(self, step: int = 0, total_steps: int = 50_000):
        self.step  = step
        self.total = total_steps

    def _r(self, ramp: int) -> float:
        """0→1 ramp: step/ramp adımda 1'e ulaşır."""
        return min(1.0, self.step / max(ramp, 1))

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        r = self._r

        # 1. Uzaklık / ölçek (1m ≈ scale=1.0, 4m ≈ scale=0.3)
        if random.random() > 0.25:
            lo    = 1.0 - 0.65 * r(12_000)
            scale = random.uniform(lo, 1.0)
            img   = scale_crop(img, scale)

        # 2. Perspektif eğrilik
        if random.random() > 0.3:
            strength = random.uniform(0.05, 0.20) * r(8_000)
            img      = perspective_warp(img, strength)

        # 3. Kamera blur (odak + hareket)
        if random.random() > 0.25:
            sigma = random.uniform(0.3, 2.5) * r(5_000)
            img   = _blur(img, sigma)

        # 4. Parlaklık + kontrast
        bri = random.uniform(-0.25, 0.25) * r(3_000)
        con = 1.0 + random.uniform(-0.35, 0.45) * r(3_000)
        img = torch.clamp(img * con + bri, 0, 1)

        # 5. TV gamma eğrisi
        if random.random() > 0.4:
            gamma = random.uniform(0.6, 1.5) * r(4_000) + 1.0 * (1 - r(4_000))
            img   = tv_gamma(img, gamma)

        # 6. Gaussian noise (kamera sensör gürültüsü)
        if random.random() > 0.3:
            std = random.uniform(0.005, 0.04) * r(4_000)
            img = torch.clamp(img + torch.randn_like(img) * std, 0, 1)

        # 7. Moiré (Brandion-specific)
        if random.random() > 0.45:
            strength = random.uniform(0.01, 0.05) * r(6_000)
            img      = moire(img, strength)

        # 8. White balance kayması
        if random.random() > 0.5:
            img = color_shift(img, 0.08 * r(5_000))

        # 9. JPEG sıkıştırma (kamera çıktısı her zaman JPEG)
        quality = int(100 - random.uniform(15, 60) * r(5_000))
        quality = max(quality, 40)
        if quality < 95:
            img = _jpeg(img, quality)

        return img
