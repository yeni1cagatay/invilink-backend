"""
Eğitim verisi pre-generation scripti.
Çalıştır:  python generate_training_data.py
Çıktı:     training_data/ klasörü (JPEG + labels)

Sonra train_hidden.py bu klasörden okur → encode yok, 10x hızlı.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import dct_watermark as dct
from train_hidden import simulate_screen_camera, _random_host_image

OUT_DIR   = Path(__file__).parent / "training_data"
NUM_IDS   = 256
JPEG_Q    = 92   # kayıpsız benzeri kalite — distorsiyon training sırasında eklenir


def generate(n: int = 5000, seed: int = 42):
    OUT_DIR.mkdir(exist_ok=True)
    labels_path = OUT_DIR / "labels.json"

    # Mevcut kayıtları yükle (devam modunda)
    if labels_path.exists():
        with open(labels_path) as f:
            labels: dict = json.load(f)
    else:
        labels = {}

    start = len(labels)
    if start >= n:
        print(f"[GEN] Zaten {start} sample var, n={n}'e ulaşıldı.")
        return

    rng = random.Random(seed)
    print(f"[GEN] {start} → {n} sample üretiliyor...")

    for i in range(start, n):
        wm_id = rng.randint(0, NUM_IDS - 1)
        host  = _random_host_image(rng)
        enc   = dct.encode(host, wm_id)

        fname = f"{i:05d}.jpg"
        enc.save(OUT_DIR / fname, "JPEG", quality=JPEG_Q)
        labels[fname] = wm_id

        if (i + 1) % 100 == 0 or i == n - 1:
            with open(labels_path, "w") as f:
                json.dump(labels, f)
            pct = (i + 1) / n * 100
            print(f"  {i+1}/{n} ({pct:.0f}%)  wm_id={wm_id}", flush=True)

    print(f"[GEN] Tamamlandı → {OUT_DIR}  ({len(labels)} sample)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(n=args.n, seed=args.seed)
