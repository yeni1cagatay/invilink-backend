"""
Brandion Watermark — Eğitim Scripti
=====================================
Kullanım:
  python train.py --data ./data --gpu            # GPU (Windows RTX 3060)
  python train.py --data ./data --epochs 20      # CPU test (Mac)

Çıktı: checkpoints/brandion_best.pt
"""

import argparse
import glob
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from augment import ScreenCameraAugment
from models import BrandionDecoder, BrandionEncoder, SECRET_SIZE, bits_from_wm_id, wm_id_from_bits

# ─── SABITLER ────────────────────────────────────────────────────────────────

IMG_SIZE   = 400
BATCH_SIZE = 16     # RTX 3060 12GB GDDR6 için optimize
LR         = 1e-4

W_IMAGE  = 2.0     # YUV image fidelity
W_SECRET = 1.5     # bit accuracy
W_EDGE   = 5.0     # kenar bölgelerinde daha az değişim (görünmezlik)


# ─── DATASET ─────────────────────────────────────────────────────────────────

class ImageFolder(Dataset):
    EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

    def __init__(self, root: str, size: int = IMG_SIZE):
        self.files = []
        for ext in self.EXTS:
            self.files += glob.glob(str(Path(root) / "**" / ext), recursive=True)
        if not self.files:
            raise FileNotFoundError(f"'{root}' altında görsel bulunamadı.")
        print(f"[Dataset] {len(self.files)} görsel yüklendi → {root}")
        self.tf = T.Compose([
            T.RandomResizedCrop(size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
        ])

    def __len__(self):  return len(self.files)

    def __getitem__(self, i):
        try:
            return self.tf(Image.open(self.files[i]).convert("RGB"))
        except Exception:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)


# ─── LOSS FONKSİYONLARI ──────────────────────────────────────────────────────

def yuv_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    YUV renk uzayında MSE.
    U ve V (renk) kanalları Y'den 100x daha ağır →
    renk kanalları korunurken parlaklığa daha fazla izin verilir.
    """
    def rgb2yuv(x):
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        y =  0.299 * r + 0.587 * g + 0.114 * b
        u = -0.147 * r - 0.289 * g + 0.436 * b
        v =  0.615 * r - 0.515 * g - 0.100 * b
        return torch.stack([y, u, v], dim=1)

    diff    = rgb2yuv(pred) - rgb2yuv(target)
    weights = torch.tensor([1.0, 100.0, 100.0], device=pred.device).view(1, 3, 1, 1)
    return (diff ** 2 * weights).mean()


def edge_loss(residual: torch.Tensor) -> torch.Tensor:
    """
    Kenar bölgelerinde (köşe, çerçeve) residual küçük olsun.
    İnsan gözü düz alanlardaki değişimi daha kolay fark eder.
    """
    B, C, H, W = residual.shape
    mask = torch.ones(B, C, H, W, device=residual.device)
    edge = H // 8
    decay = torch.linspace(1.0, 0.0, edge, device=residual.device)
    mask[:, :, :edge,  :] *= decay.view(edge, 1)
    mask[:, :, -edge:, :] *= decay.flip(0).view(edge, 1)
    mask[:, :, :,  :edge] *= decay.view(1, edge)
    mask[:, :, :, -edge:] *= decay.flip(0).view(1, edge)
    return (residual ** 2 * mask).mean()


# ─── EĞİTİM ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"[Train] device={device}  batch={BATCH_SIZE}  epochs={args.epochs}")

    encoder = BrandionEncoder(SECRET_SIZE).to(device)
    decoder = BrandionDecoder(SECRET_SIZE).to(device)

    params    = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(params, lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    dataset = ImageFolder(args.data)
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=6 if args.gpu else 0,   # i5-12400F 12 çekirdek
        pin_memory=args.gpu,
        persistent_workers=args.gpu,
    )

    secret_loss_fn = nn.BCEWithLogitsLoss()
    total_steps    = args.epochs * len(loader)
    step           = 0
    best_id_acc    = 0.0
    # Mixed precision — RTX 3060'ta ~1.5x hız artışı
    scaler = torch.cuda.amp.GradScaler(enabled=args.gpu)

    Path(args.out).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        augment = ScreenCameraAugment(step=step, total_steps=total_steps)

        epoch_img = epoch_sec = epoch_loss = 0.0
        t0 = time.time()

        for images in loader:
            images = images.to(device)
            B      = images.size(0)

            # Rastgele wm_id → 100-bit secret
            wm_ids = [random.randint(0, 255) for _ in range(B)]
            secret = torch.stack([bits_from_wm_id(wid) for wid in wm_ids]).to(device)

            with torch.cuda.amp.autocast(enabled=args.gpu):
                # Encode
                residual = encoder(images, secret)
                encoded  = torch.clamp(images + residual, 0, 1)

                # Distort (screen→camera simülasyonu)
                distorted = augment(encoded)

                # Decode
                pred_logits = decoder(distorted)

                # Losses
                img_l  = yuv_loss(encoded, images)
                edg_l  = edge_loss(residual)
                sec_l  = secret_loss_fn(pred_logits, secret)

                # Image loss için ramp: ilk 3K adımda sadece secret loss
                img_ramp = min(1.0, max(0.0, (step - 3_000) / 5_000))
                loss = W_SECRET * sec_l + img_ramp * (W_IMAGE * img_l + W_EDGE * edg_l)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_img  += img_l.item()
            epoch_sec  += sec_l.item()
            epoch_loss += loss.item()
            step       += 1

        scheduler.step()

        n       = len(loader)
        elapsed = time.time() - t0
        id_acc  = evaluate(encoder, decoder, device, n_samples=200)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={epoch_loss/n:.4f}  img={epoch_img/n:.4f}  sec={epoch_sec/n:.4f} | "
            f"id_acc={id_acc:.1f}%  [{elapsed:.0f}s]"
        )

        if id_acc > best_id_acc:
            best_id_acc = id_acc
            _save(encoder, decoder, args.out, epoch, id_acc)
            print(f"  → kaydedildi  (best id_acc={best_id_acc:.1f}%)")

    print(f"\nEğitim tamamlandı. Best wm_id accuracy: {best_id_acc:.1f}%")
    print(f"Model: {args.out}/brandion_best.pt")


# ─── DEĞERLENDİRME ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    encoder: BrandionEncoder,
    decoder: BrandionDecoder,
    device,
    n_samples: int = 200,
    with_augment: bool = True,
) -> float:
    """wm_id tam eşleşme doğruluğu (tüm 8 bit doğru olmalı)."""
    encoder.eval()
    decoder.eval()
    augment = ScreenCameraAugment(step=99_999, total_steps=100_000)  # tam şiddet

    correct = 0
    for _ in range(n_samples // BATCH_SIZE + 1):
        B      = min(BATCH_SIZE, n_samples - correct)
        if B <= 0: break
        images = torch.rand(B, 3, IMG_SIZE, IMG_SIZE).to(device)
        wm_ids = [random.randint(0, 255) for _ in range(B)]
        secret = torch.stack([bits_from_wm_id(wid) for wid in wm_ids]).to(device)

        residual  = encoder(images, secret)
        encoded   = torch.clamp(images + residual, 0, 1)
        distorted = augment(encoded) if with_augment else encoded
        logits    = decoder(distorted)

        for i in range(B):
            pred = wm_id_from_bits(logits[i].unsqueeze(0))
            if pred == wm_ids[i]:
                correct += 1

    encoder.train()
    decoder.train()
    return correct / n_samples * 100


# ─── KAYDETME ────────────────────────────────────────────────────────────────

def _save(encoder, decoder, out_dir, epoch, acc):
    path = Path(out_dir) / "brandion_best.pt"
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "epoch":   epoch,
        "id_acc":  acc,
        "secret_size": SECRET_SIZE,
        "img_size":    IMG_SIZE,
    }, path)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="./data",        help="Görsel dataset klasörü")
    parser.add_argument("--out",    default="./checkpoints", help="Checkpoint kayıt klasörü")
    parser.add_argument("--epochs", type=int, default=100,   help="Epoch sayısı")
    parser.add_argument("--gpu",    action="store_true",     help="CUDA kullan")
    args = parser.parse_args()
    train(args)
