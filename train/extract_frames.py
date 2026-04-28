"""
Video dosyalarından eğitim frame'i çıkarma.
Kullanim: python extract_frames.py --videos ./videos --out ./data

TV reklamları, dizi sahneleri, tanıtım videoları varsa
COCO'dan çok daha iyi eğitim verisi sağlar.
"""

import argparse
import glob
from pathlib import Path

import cv2


def extract(video_path: str, out_dir: Path, every_n: int = 15) -> int:
    cap   = cv2.VideoCapture(video_path)
    count = saved = 0
    stem  = Path(video_path).stem

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if count % every_n == 0:
            out = out_dir / f"{stem}_{count:06d}.jpg"
            cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
        count += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default="./videos", help="Video klasörü (.mp4, .mov, .avi)")
    parser.add_argument("--out",    default="./data",   help="Frame çıktı klasörü")
    parser.add_argument("--every",  type=int, default=15, help="Her N frame'de bir kaydet (default: 15 = ~2fps)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = (
        glob.glob(str(Path(args.videos) / "**" / "*.mp4"),  recursive=True) +
        glob.glob(str(Path(args.videos) / "**" / "*.mov"),  recursive=True) +
        glob.glob(str(Path(args.videos) / "**" / "*.avi"),  recursive=True) +
        glob.glob(str(Path(args.videos) / "**" / "*.mkv"),  recursive=True)
    )

    if not videos:
        print(f"'{args.videos}' altında video bulunamadı.")
        return

    total = 0
    for v in videos:
        n = extract(v, out_dir, args.every)
        print(f"  {Path(v).name}  →  {n} frame")
        total += n

    print(f"\nToplam: {total} frame  →  {out_dir}")
    print("\nEğitimi başlatmak için:")
    print(f"  python train.py --data {args.out} --gpu --epochs 100")


if __name__ == "__main__":
    main()
