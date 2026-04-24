"""Local video encoder — SS watermark frame by frame."""
import subprocess, sys, os
import numpy as np
from PIL import Image
import spread_spectrum as ss

IN  = "/Users/macbook/Downloads/mobialpp.mp4"
OUT = "/Users/macbook/Downloads/mobialpp_wm.mp4"
WM_ID = 57

# Video bilgisi
probe = subprocess.run(
    ["ffprobe","-v","error","-select_streams","v:0",
     "-show_entries","stream=width,height,r_frame_rate",
     "-of","csv=p=0", IN],
    capture_output=True, text=True
).stdout.strip().split(",")
W, H = int(probe[0]), int(probe[1])
fps_n, fps_d = probe[2].split("/")
FPS = f"{fps_n}/{fps_d}"

print(f"Video: {W}x{H} fps={FPS} wm_id={WM_ID}")

# FFmpeg: decode → pipe
dec = subprocess.Popen(
    ["ffmpeg","-i",IN,"-f","rawvideo","-pix_fmt","rgb24","-"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
)

# FFmpeg: pipe → encode
enc = subprocess.Popen(
    ["ffmpeg","-y","-f","rawvideo","-pix_fmt","rgb24",
     "-s",f"{W}x{H}","-r",FPS,"-i","pipe:0",
     "-i",IN,"-map","0:v","-map","1:a?",
     "-c:v","libx264","-crf","18","-preset","fast",
     "-c:a","aac","-shortest", OUT],
    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
)

frame_size = W * H * 3
frame_n = 0
while True:
    raw = dec.stdout.read(frame_size)
    if len(raw) < frame_size:
        break
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3)
    img = Image.fromarray(arr)
    wm  = ss.encode_frame(img, WM_ID)
    enc.stdin.write(np.array(wm).tobytes())
    frame_n += 1
    if frame_n % 60 == 0:
        print(f"  {frame_n} frame...")

enc.stdin.close()
dec.wait()
enc.wait()
print(f"Bitti! {frame_n} frame → {OUT}")
