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
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


ALERT_LOG    = Path(r"C:\Users\cyeni\train_alerts.log")
DISK_PATH    = Path(r"C:\Users\cyeni")
_loss_history: list = []   # son 500 adımın loss'u
_last_alerts: dict  = {}   # aynı alertin spam yapmaması için

def _cooldown(key: str, minutes: int = 30) -> bool:
    """Aynı alert key'i için cooldown süresi geçtiyse True döner."""
    now = time.time()
    if key not in _last_alerts or now - _last_alerts[key] > minutes * 60:
        _last_alerts[key] = now
        return True
    return False

def alert(title: str, message: str, level: str = "UYARI"):
    """Windows toast bildirimi + alert log."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] [{level}] {title}: {message}"
    print(entry, flush=True)
    with open(ALERT_LOG, "a", encoding="utf-8") as f:
        f.write(entry + "\n")
    # Windows toast bildirimi
    ps = (
        f'[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null;'
        f'$template = [Windows.UI.Notifications.ToastTemplateType]::ToastText02;'
        f'$xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent($template);'
        f'$xml.GetElementsByTagName("text")[0].AppendChild($xml.CreateTextNode("{title}")) | Out-Null;'
        f'$xml.GetElementsByTagName("text")[1].AppendChild($xml.CreateTextNode("{message}")) | Out-Null;'
        f'$toast = [Windows.UI.Notifications.ToastNotification]::new($xml);'
        f'[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Brandion AI").Show($toast);'
    )
    try:
        subprocess.Popen(["powershell", "-WindowStyle", "Hidden", "-Command", ps])
    except Exception:
        pass
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
W_SECRET = 5.0     # bit accuracy — increased to prevent encoder collapse
W_EDGE   = 2.0     # kenar bölgelerinde daha az değişim (görünmezlik)


# ─── DATASET ─────────────────────────────────────────────────────────────────

class ImageFolder(Dataset):
    EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

    def __init__(self, root: str, size: int = IMG_SIZE):
        self.files = []
        for ext in self.EXTS:
            self.files += glob.glob(str(Path(root) / "**" / ext), recursive=True)
        if not self.files:
            raise FileNotFoundError(f"'{root}' altında görsel bulunamadı.")
        print(f"[Dataset] {len(self.files)} images loaded -> {root}")
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

    # Phase 2: freeze decoder, train only encoder for invisibility
    if args.phase2:
        for p in decoder.parameters():
            p.requires_grad = False
        params = list(encoder.parameters())
        print("[Phase2] Decoder donduruldu, sadece encoder egitiliyor.")
    else:
        params = list(encoder.parameters()) + list(decoder.parameters())

    optimizer = optim.AdamW(params, lr=LR if not args.phase2 else LR * 0.1, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    dataset = ImageFolder(args.data)
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    secret_loss_fn = nn.BCEWithLogitsLoss()
    total_steps    = args.epochs * len(loader)
    step           = 0
    best_id_acc    = 0.0
    start_epoch    = 1
    # Mixed precision — RTX 3060'ta ~1.5x hız artışı
    scaler = torch.amp.GradScaler('cuda', enabled=args.gpu)

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if available
    resume_path = Path(args.out) / "brandion_resume.pt"
    best_path   = Path(args.out) / "brandion_best.pt"
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt["scaler"])
            scheduler.load_state_dict(ckpt["scheduler"])
        except (ValueError, KeyError):
            print("[Resume] Optimizer state uyumsuz, sadece model ağırlıkları yüklendi.")
            scaler = torch.amp.GradScaler("cuda")
        start_epoch  = ckpt["epoch"] + 1
        step         = ckpt["step"]
        best_id_acc  = ckpt["best_id_acc"]
        print(f"[Resume] Epoch {ckpt['epoch']} noktasindan devam ediliyor. Best id_acc={best_id_acc:.1f}%")
    elif best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        start_epoch = ckpt["epoch"] + 1
        best_id_acc = 0.0  # sıfırla: curriculum augment değişti, eski eşik geçersiz
        print(f"[Resume] brandion_best.pt'den yuklendu (epoch {ckpt['epoch']}), optimizer sifirdan.")

    for epoch in range(start_epoch, args.epochs + 1):
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

            with torch.amp.autocast('cuda', enabled=args.gpu):
                # Encode
                residual = encoder(images, secret)
                encoded  = torch.clamp(images + residual, 0, 1)

            # Augmentation autocast dışında — _jpeg float16/float32 karışımı önlenir
            with torch.no_grad():
                distorted_detached = augment(encoded.float()).to(encoded.dtype)
            # STE: forward = distorted, backward = encoded
            distorted = encoded + (distorted_detached - encoded).detach()

            with torch.amp.autocast('cuda', enabled=args.gpu):
                # Decode
                pred_logits = decoder(distorted)

                # Losses
                img_l  = yuv_loss(encoded, images)
                edg_l  = edge_loss(residual)
                sec_l  = secret_loss_fn(pred_logits, secret)

                # Phase2: img_ramp from epoch 1 (decoder frozen, stable)
                # Phase1: no image loss
                if args.phase2:
                    ramp_start = int(0.10 * total_steps)
                    ramp_end   = int(0.90 * total_steps)
                    img_ramp = min(1.0, max(0.0, (step - ramp_start) / max(1, ramp_end - ramp_start)))
                else:
                    img_ramp = 0.0

                # Anti-collapse: penalize near-zero residual using per-sample L2 norm
                per_sample_norm = residual.flatten(1).norm(dim=1).mean()
                anti_collapse = torch.clamp(10.0 - per_sample_norm, min=0.0) * 2.0

                loss = W_SECRET * sec_l + img_ramp * (W_IMAGE * img_l + W_EDGE * edg_l) + anti_collapse

            if torch.isnan(loss) or torch.isinf(loss):
                alert("NaN Loss Tespit Edildi", f"Epoch {epoch}, step {step} — adim atlandi", "KRITIK")
                continue

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

            if step % 100 == 0:
                batch_idx = step % len(loader) or len(loader)
                pct = batch_idx / len(loader) * 100
                print(f"  Epoch {epoch} | iter {batch_idx}/{len(loader)} ({pct:.1f}%) | loss={loss.item():.4f}", flush=True)
                _step_checks(epoch, step, loss.item())

        scheduler.step()

        n       = len(loader)
        elapsed = time.time() - t0
        id_acc_clean = evaluate(encoder, decoder, device, n_samples=100,
                                aug_step=0, aug_total=total_steps)
        id_acc       = evaluate(encoder, decoder, device, n_samples=200,
                                aug_step=step, aug_total=total_steps)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={epoch_loss/n:.4f}  img={epoch_img/n:.4f}  sec={epoch_sec/n:.4f} | "
            f"id_acc={id_acc:.1f}% (clean={id_acc_clean:.1f}%)  [{elapsed:.0f}s]"
        )

        is_best = id_acc > best_id_acc
        if is_best:
            best_id_acc = id_acc
            _save(encoder, decoder, args.out, epoch, id_acc)
            print(f"  -> saved  (best id_acc={best_id_acc:.1f}%)")
            alert("Yeni En İyi Model", f"Epoch {epoch} | id_acc=%{id_acc:.1f}", "BILGI")

        # id_acc ani düşüş kontrolü
        if epoch > 1 and id_acc < 50.0:
            alert("id_acc Duste", f"Epoch {epoch} | id_acc=%{id_acc:.1f} — model bozulmus olabilir", "KRITIK")

        # Her epoch resume checkpoint kaydet (atomik: temp → rename)
        resume_tmp = Path(args.out) / "brandion_resume.tmp"
        torch.save({
            "encoder":     encoder.state_dict(),
            "decoder":     decoder.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "scaler":      scaler.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "epoch":       epoch,
            "step":        step,
            "best_id_acc": best_id_acc,
        }, resume_tmp)
        resume_tmp.replace(Path(args.out) / "brandion_resume.pt")

        # Epoch geçmişini JSON satırı olarak kaydet
        record = {
            "epoch":       epoch,
            "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
            "loss":        round(epoch_loss / n, 6),
            "img_loss":    round(epoch_img  / n, 6),
            "sec_loss":    round(epoch_sec  / n, 6),
            "id_acc":      round(id_acc, 2),
            "id_acc_clean":round(id_acc_clean, 2),
            "best_id_acc": round(best_id_acc, 2),
            "is_best":     is_best,
            "elapsed_sec": int(elapsed),
            "step":        step,
        }
        history_path = Path(args.out) / "epoch_history.jsonl"
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  [checkpoint] Epoch {epoch} kaydedildi.", flush=True)

    print(f"\nTraining complete. Best wm_id accuracy: {best_id_acc:.1f}%")
    print(f"Model: {args.out}/brandion_best.pt")
    alert("Egitim Tamamlandi", f"100 epoch bitti | Best id_acc=%{best_id_acc:.1f}", "BILGI")


# ─── DEĞERLENDİRME ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    encoder: BrandionEncoder,
    decoder: BrandionDecoder,
    device,
    n_samples: int = 200,
    aug_step: int = 0,
    aug_total: int = 100_000,
) -> float:
    """wm_id tam eşleşme doğruluğu (tüm 8 bit doğru olmalı)."""
    encoder.eval()
    decoder.eval()
    augment = ScreenCameraAugment(step=aug_step, total_steps=aug_total)

    correct = 0
    for _ in range(n_samples // BATCH_SIZE + 1):
        B      = min(BATCH_SIZE, n_samples - correct)
        if B <= 0: break
        images = torch.rand(B, 3, IMG_SIZE, IMG_SIZE).to(device)
        wm_ids = [random.randint(0, 255) for _ in range(B)]
        secret = torch.stack([bits_from_wm_id(wid) for wid in wm_ids]).to(device)

        residual  = encoder(images, secret)
        encoded   = torch.clamp(images + residual, 0, 1)
        distorted = augment(encoded)
        logits    = decoder(distorted)

        for i in range(B):
            pred = wm_id_from_bits(logits[i].unsqueeze(0))
            if pred == wm_ids[i]:
                correct += 1

    encoder.train()
    decoder.train()
    return correct / n_samples * 100


# ─── ADIM KONTROLLERI ────────────────────────────────────────────────────────

def _step_checks(epoch: int, step: int, loss_val: float):
    """Her 100 adımda tüm alertleri kontrol eder."""
    global _loss_history

    # 1. Loss artış trendi
    _loss_history.append(loss_val)
    if len(_loss_history) > 500:
        _loss_history.pop(0)
    if len(_loss_history) >= 200 and _cooldown("loss_trend", 60):
        recent = _loss_history[-100:]
        older  = _loss_history[-200:-100]
        if sum(recent) / len(recent) > sum(older) / len(older) * 2.0:
            alert("Loss Artıyor", f"Epoch {epoch} step {step} — son 100 adim 2x daha yuksek", "UYARI")

    # 2. GPU sıcaklık ve VRAM
    if _cooldown("gpu_temp", 10):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total",
                 "--format=csv,noheader"], text=True).strip().split(",")
            temp     = int(out[0].strip())
            mem_used = int(out[1].strip().replace(" MiB", ""))
            mem_tot  = int(out[2].strip().replace(" MiB", ""))
            mem_pct  = mem_used / mem_tot * 100
            if temp >= 85:
                alert("GPU COK SICAK", f"{temp}C — termal throttle riski!", "KRITIK")
            elif temp >= 78 and _cooldown("gpu_warn", 30):
                alert("GPU Isinıyor", f"{temp}C", "UYARI")
            if mem_pct >= 95:
                alert("VRAM DOLUYOR", f"%{mem_pct:.0f} — OOM riski!", "KRITIK")
        except Exception:
            pass

    # 3. Disk alanı
    if _cooldown("disk", 60):
        try:
            import shutil
            free_gb = shutil.disk_usage(r"C:\Users\cyeni").free / 1e9
            if free_gb < 5:
                alert("DISK KRITIK", f"Yalnizca {free_gb:.1f} GB kaldi — checkpoint kaydedilemez!", "KRITIK")
            elif free_gb < 15:
                alert("Disk Azaliyor", f"{free_gb:.1f} GB kaldi", "UYARI")
        except Exception:
            pass


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
    parser.add_argument("--data",   default="./data",        help="Dataset folder")
    parser.add_argument("--out",    default="./checkpoints", help="Checkpoint output folder")
    parser.add_argument("--epochs", type=int, default=100,   help="Number of epochs")
    parser.add_argument("--gpu",    action="store_true",     help="Use CUDA")
    parser.add_argument("--log",    default=None,            help="Log file path (optional)")
    parser.add_argument("--phase2", action="store_true",     help="Freeze decoder, train encoder for invisibility")
    args = parser.parse_args()

    if args.log:
        log_file = open(args.log, "a", encoding="utf-8", buffering=1)
        log_file.write(f"\n{'='*60}\n[RUN] {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n")
        log_file.flush()
        sys.stdout = log_file
        sys.stderr = log_file

    train(args)
