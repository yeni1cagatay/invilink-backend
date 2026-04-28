"""
COCO 2017 dataset indirme scripti.
Kullanim: python download_data.py

Indirilenler:
  train/data/  →  ~118K görsel (~18GB)

Sadece görseller indirilir, annotationlar atlanır.
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
COCO_URL = "http://images.cocodataset.org/zips/train2017.zip"
ZIP_PATH = Path(__file__).parent / "train2017.zip"


def _progress(count, block_size, total):
    pct  = count * block_size / total * 100
    done = int(pct / 2)
    bar  = "█" * done + "░" * (50 - done)
    gb   = count * block_size / 1e9
    sys.stdout.write(f"\r  [{bar}] {pct:.1f}%  {gb:.2f} GB")
    sys.stdout.flush()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Zaten var mı?
    existing = list(DATA_DIR.glob("*.jpg"))
    if len(existing) > 1000:
        print(f"Dataset zaten mevcut: {len(existing)} görsel  →  {DATA_DIR}")
        return

    print("COCO 2017 train set indiriliyor (~18GB)...")
    print("Tahmini süre: 30-60 dk (bağlantı hızına göre)\n")

    if not ZIP_PATH.exists():
        urllib.request.urlretrieve(COCO_URL, ZIP_PATH, reporthook=_progress)
        print()
    else:
        print(f"ZIP zaten var: {ZIP_PATH}")

    print("\nZIP açılıyor...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        members = z.namelist()
        for i, m in enumerate(members):
            if m.endswith(".jpg"):
                dest = DATA_DIR / Path(m).name
                with z.open(m) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
            if i % 5000 == 0:
                print(f"  {i}/{len(members)}...")

    ZIP_PATH.unlink()  # ZIP'i sil, disk alanı kazan
    print(f"\nTamamlandi! {len(list(DATA_DIR.glob('*.jpg')))} görsel  →  {DATA_DIR}")
    print("\nEğitimi başlatmak için:")
    print("  python train.py --data ./data --gpu --epochs 100")


if __name__ == "__main__":
    main()
