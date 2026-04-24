"""
HiDDeN Decoder Eğitim Scripti
================================
Kullanım:
  python train_hidden.py                      # 50 epoch, CPU
  python train_hidden.py --epochs 100 --gpu   # GPU varsa

Çıktı: hidden_decoder.pt  (decode için kullanılır)

Strateji:
- DCT encoder ile rastgele wm_id'ler için filigranlanmış görsel üret
- Ekran→kamera kanalını simüle et (perspektif + blur + JPEG + gamma)
- Decoder her seferinde orijinal bit'leri tahmin etmeyi öğrenir
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset

import dct_watermark as dct
from brandion_engine import PAYLOAD_BITS
from hidden_decoder import HiDDeNDecoder, DECODE_H, DECODE_W, _pil_to_tensor

# ─── SABITLER ────────────────────────────────────────────────────────────────

NUM_IDS      = 256
BATCH_SIZE   = 16
LR           = 3e-4
STEPS_EPOCH  = 200   # her epoch'ta kaç batch
SAVE_PATH    = Path(__file__).parent / "hidden_decoder.pt"


# ─── EKRan→KAMERA SİMÜLASYONU ───────────────────────────────────────────────

def simulate_screen_camera(img: Image.Image, rng: random.Random) -> Image.Image:
    """
    Gerçekçi ekran→kamera distorsiyon zinciri:
    perspektif → blur → gamma → JPEG → renk gürültüsü
    """
    arr = np.array(img.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]

    # 1. Perspektif warp — ekrana açılı tutma simülasyonu
    if rng.random() > 0.3:
        strength = rng.uniform(0.04, 0.18)
        pts_src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dx = [rng.uniform(-strength * w, strength * w) for _ in range(4)]
        dy = [rng.uniform(-strength * h, strength * h) for _ in range(4)]
        pts_dst = pts_src + np.float32(list(zip(dx, dy)))
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        arr = cv2.warpPerspective(arr, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)

    # 2. Gaussian blur — kamera odak sorunları
    if rng.random() > 0.2:
        sigma = rng.uniform(0.3, 2.0)
        k = int(sigma * 3) * 2 + 1
        arr = cv2.GaussianBlur(arr, (k, k), sigma)

    # 3. Gamma — ekran parlaklığı / kamera exposure
    gamma = rng.uniform(0.65, 1.45)
    arr = np.clip(arr / 255.0, 0, 1) ** gamma * 255.0

    # 4. Hafif renk gürültüsü — kamera sensör gürültüsü
    if rng.random() > 0.4:
        noise = np.random.normal(0, rng.uniform(2, 8), arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255)

    result = Image.fromarray(arr.astype(np.uint8))

    # 5. JPEG sıkıştırma — kamera çıktısı her zaman JPEG
    quality = rng.randint(55, 92)
    buf = __import__("io").BytesIO()
    result.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    result = Image.open(buf).copy()

    return result


# ─── DATASET ─────────────────────────────────────────────────────────────────

def _random_host_image(rng: random.Random) -> Image.Image:
    """Çeşitli host görsel türleri üret — decoder overfitting önlenir."""
    w, h = dct.ENCODE_W, dct.ENCODE_H
    style = rng.choice(["noise", "gradient", "mixed", "noise_color"])

    if style == "noise":
        arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    elif style == "gradient":
        x = np.linspace(rng.randint(30, 80), rng.randint(150, 230), w)
        y = np.linspace(rng.randint(30, 80), rng.randint(150, 230), h)
        xx, yy = np.meshgrid(x, y)
        base = ((xx + yy) / 2).astype(np.uint8)
        arr = np.stack([base] * 3, axis=-1)
    elif style == "mixed":
        base = np.full((h, w, 3), rng.randint(80, 180), dtype=np.float32)
        noise = np.random.normal(0, 40, (h, w, 3))
        arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    else:  # noise_color
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(3):
            arr[:, :, c] = np.random.randint(
                rng.randint(0, 80), rng.randint(120, 256), (h, w), dtype=np.uint8
            )
    return Image.fromarray(arr)


class WatermarkDataset(IterableDataset):
    """Her iterasyonda taze watermarked görsel + distorsiyon üretir."""

    def __init__(self, steps: int, seed: int = 0):
        self.steps = steps
        self.seed  = seed

    def __iter__(self):
        rng = random.Random(self.seed + torch.initial_seed() % 10000)
        for _ in range(self.steps):
            wm_id = rng.randint(0, NUM_IDS - 1)
            host    = _random_host_image(rng)
            encoded = dct.encode(host, wm_id)
            distorted = simulate_screen_camera(encoded, rng)
            x = _pil_to_tensor(distorted)
            target = _build_target_bits(wm_id)
            yield x, target


class PregenDataset(torch.utils.data.Dataset):
    """generate_training_data.py ile üretilmiş JPEG'lerden okur — on-the-fly encode YOK."""

    def __init__(self, data_dir: Path, seed: int = 0):
        import json
        self.data_dir = data_dir
        with open(data_dir / "labels.json") as f:
            labels = json.load(f)
        self.files  = sorted(labels.keys())
        self.labels = labels
        self.rng    = random.Random(seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        fname = self.files[idx]
        wm_id = self.labels[fname]
        img   = Image.open(self.data_dir / fname).convert("RGB")
        rng   = random.Random(idx)
        distorted = simulate_screen_camera(img, rng)
        x     = _pil_to_tensor(distorted)
        return x, _build_target_bits(wm_id)


def _build_target_bits(wm_id: int) -> torch.Tensor:
    """wm_id → 56-bit float tensor."""
    from brandion_engine import int_to_bits
    MAGIC = 0xB8A3
    checksum = ((MAGIC >> 8) ^ (MAGIC & 0xFF) ^ (wm_id & 0xFF) ^ ((wm_id >> 8) & 0xFF)) & 0xFF
    bits = int_to_bits(MAGIC, 16) + int_to_bits(wm_id, 32) + int_to_bits(checksum, 8)
    return torch.tensor(bits, dtype=torch.float32)


# ─── EĞİTİM ──────────────────────────────────────────────────────────────────

def train(epochs: int = 50, use_gpu: bool = False, data_dir: Path | None = None):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] device={device}  epochs={epochs}  batch={BATCH_SIZE}")

    # Pre-generated data varsa kullan (10x hızlı), yoksa on-the-fly
    pregen = Path(__file__).parent / "training_data"
    if data_dir:
        pregen = Path(data_dir)

    if (pregen / "labels.json").exists():
        dataset_full = PregenDataset(pregen)
        print(f"[TRAIN] PregenDataset: {len(dataset_full)} sample  →  {pregen}")
        use_pregen = True
    else:
        print(f"[TRAIN] On-the-fly dataset (yavaş). "
              f"Hızlandırmak için: python generate_training_data.py")
        use_pregen = False

    model = HiDDeNDecoder(payload_bits=PAYLOAD_BITS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        if use_pregen:
            loader = DataLoader(dataset_full, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=2, pin_memory=False)
        else:
            dataset = WatermarkDataset(STEPS_EPOCH * BATCH_SIZE, seed=epoch)
            loader  = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)

        total_loss = 0.0
        total_bits = 0
        correct_bits = 0

        for x, target in loader:
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            predicted = (preds > 0.5).float()
            correct_bits += (predicted == target).sum().item()
            total_bits   += target.numel()

        scheduler.step()

        bit_acc  = correct_bits / total_bits * 100
        avg_loss = total_loss / len(loader)

        # wm_id doğrulama (tüm payload doğru mu?)
        id_acc = _eval_id_accuracy(model, device, n=100)

        print(f"Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | "
              f"bit_acc={bit_acc:.1f}% | id_acc={id_acc:.1f}%")

        if id_acc > best_acc:
            best_acc = id_acc
            model.save(SAVE_PATH)
            print(f"  → saved (best id_acc={best_acc:.1f}%)")

    print(f"\n[TRAIN] Tamamlandı. Best id_acc={best_acc:.1f}%  →  {SAVE_PATH}")


@torch.no_grad()
def _eval_id_accuracy(model: HiDDeNDecoder, device, n: int = 100) -> float:
    """n adet örnek üzerinde wm_id tam doğruluğu ölç."""
    model.eval()
    rng = random.Random(9999)
    correct = 0
    for _ in range(n):
        wm_id = rng.randint(0, NUM_IDS - 1)
        host  = _random_host_image(rng)
        enc   = dct.encode(host, wm_id)
        dist  = simulate_screen_camera(enc, rng)
        x     = _pil_to_tensor(dist).unsqueeze(0).to(device)
        bits  = (model(x) > 0.5).long().squeeze(0).tolist()
        from hidden_decoder import _validate_bits
        pred_id = _validate_bits(bits)
        if pred_id == wm_id:
            correct += 1
    model.train()
    return correct / n * 100


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()
    train(epochs=args.epochs, use_gpu=args.gpu,
          data_dir=Path(args.data_dir) if args.data_dir else None)
