from __future__ import annotations

import io
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from PIL import Image
from sqlalchemy.orm import Session

from database import (create_link, get_db, get_link, increment_scan,
                       create_video_link, get_video_link, increment_video_scan, list_video_links)
import watermark
import temporal_watermark as twm
import spread_spectrum as ss

app = FastAPI(title="Brandion API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("dashboard.html", encoding="utf-8") as f:
        return f.read()


@app.post("/api/links")
async def create_watermarked_link(
    url: str = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Gorsel + URL yukle → filigranli gorsel indir."""
    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="Gecersiz URL")

    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Sadece JPEG/PNG/WEBP kabul edilir")

    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes))

    link = create_link(db, url)
    encoded = watermark.encode(img, link.code)

    buf = io.BytesIO()
    encoded.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={
            "Content-Disposition": f'attachment; filename="invilink_{link.code}.png"',
            "X-InviLink-Code": link.code,
        },
    )


@app.post("/api/decode")
async def decode_frame(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Kamera frame'inden URL oku."""
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes))

    # Debug: gelen frame'i kaydet
    import os, time
    debug_path = f"/app/debug_frame_{int(time.time())}.jpg"
    img.save(debug_path, "JPEG", quality=95)
    print(f"[DECODE] frame kaydedildi: {debug_path} size={img.size}")

    code = watermark.decode(img)
    print(f"[DECODE] code={repr(code)} size={img.size}")
    if not code:
        raise HTTPException(status_code=404, detail="Filigran bulunamadi")

    link = get_link(db, code)
    if not link:
        print(f"[DECODE] code {code} not in DB")
        raise HTTPException(status_code=404, detail="Link bulunamadi")

    increment_scan(db, code)

    return {"url": link.url, "code": link.code}


# ─── SPREAD SPECTRUM ─────────────────────────────────────────────────────────

@app.post("/api/ss/decode")
async def ss_decode(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Tek fotoğraf → Spread Spectrum decode → URL."""
    img = Image.open(io.BytesIO(await image.read()))
    wm_id = ss.decode(img)
    if wm_id is None:
        raise HTTPException(status_code=404, detail="Filigran bulunamadı")
    vl = get_video_link(db, wm_id)
    if not vl:
        raise HTTPException(status_code=404, detail="ID kayıtlı değil")
    increment_video_scan(db, wm_id)
    return {"url": vl.url, "wm_id": wm_id, "label": vl.label}


@app.post("/api/ss/overlay")
async def ss_overlay(
    wm_id: int = Form(...),
    width: int = Form(1920),
    height: int = Form(1080),
    db: Session = Depends(get_db),
):
    """wm_id → post prodüksiyon overlay PNG indir."""
    if not get_video_link(db, wm_id):
        raise HTTPException(status_code=404, detail="wm_id bulunamadı")
    img = ss.encode_overlay(wm_id, width, height)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="overlay_{wm_id}.png"'},
    )


# ─── VIDEO WATERMARK ──────────────────────────────────────────────────────────

@app.post("/api/video-links")
async def create_video_link_endpoint(
    url: str = Form(...),
    label: str = Form(""),
    db: Session = Depends(get_db),
):
    """URL kaydet → watermark ID (0-255) döner."""
    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="Geçersiz URL")
    vl = create_video_link(db, url, label)
    return {"wm_id": vl.wm_id, "url": vl.url, "label": vl.label}


@app.get("/api/video-links")
async def list_video_links_endpoint(db: Session = Depends(get_db)):
    """Tüm video linkleri listele."""
    links = list_video_links(db)
    return [{"wm_id": v.wm_id, "url": v.url, "label": v.label,
             "scan_count": v.scan_count, "created_at": v.created_at.isoformat()} for v in links]


@app.post("/api/encode-video")
async def encode_video_endpoint(
    video: UploadFile = File(...),
    wm_id: int = Form(...),
    db: Session = Depends(get_db),
):
    """Video + wm_id → filigranli video (MP4)."""
    if wm_id < 0 or wm_id > 255:
        raise HTTPException(status_code=400, detail="wm_id 0-255 arasında olmalı")
    vl = get_video_link(db, wm_id)
    if not vl:
        raise HTTPException(status_code=404, detail="wm_id bulunamadı")

    import tempfile, os
    suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(await video.read())
        in_path = tmp_in.name

    out_path = in_path.replace(suffix, "_wm.mp4")
    try:
        twm.encode_video(in_path, out_path, wm_id)
        with open(out_path, "rb") as f:
            data = f.read()
    finally:
        os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)

    return StreamingResponse(
        io.BytesIO(data),
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="wm_{wm_id}.mp4"'},
    )


@app.post("/api/decode-video")
async def decode_video_endpoint(
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Kamera videosu → URL.
    Telefon 2-3 saniyelik video kaydeder, buraya gönderir.
    """
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        wm_id = twm.decode_video(tmp_path)
    finally:
        os.unlink(tmp_path)

    if wm_id is None:
        raise HTTPException(status_code=404, detail="Filigran bulunamadı")

    vl = get_video_link(db, wm_id)
    if not vl:
        raise HTTPException(status_code=404, detail="ID bulunamadı")

    increment_video_scan(db, wm_id)
    return {"url": vl.url, "wm_id": wm_id, "label": vl.label}


@app.post("/api/decode-frames")
async def decode_frames_endpoint(
    frames: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    """
    Birden fazla JPEG frame → URL.
    Telefon frame'leri tek tek gönderebilir.
    """
    pil_frames = []
    for f in frames:
        data = await f.read()
        pil_frames.append(Image.open(io.BytesIO(data)))

    wm_id = twm.decode_frames(pil_frames)
    if wm_id is None:
        raise HTTPException(status_code=404, detail="Filigran bulunamadı")

    vl = get_video_link(db, wm_id)
    if not vl:
        raise HTTPException(status_code=404, detail="ID bulunamadı")

    increment_video_scan(db, wm_id)
    return {"url": vl.url, "wm_id": wm_id, "label": vl.label}


@app.get("/r/{code}")
async def redirect_to_url(code: str, db: Session = Depends(get_db)):
    """Kisa kod → tam URL yonlendirme."""
    link = get_link(db, code)
    if not link:
        raise HTTPException(status_code=404, detail="Link bulunamadi")
    increment_scan(db, code)
    return RedirectResponse(url=link.url, status_code=302)


@app.get("/api/stats/{code}")
async def stats(code: str, db: Session = Depends(get_db)):
    """Tarama istatistikleri."""
    link = get_link(db, code)
    if not link:
        raise HTTPException(status_code=404, detail="Link bulunamadi")
    return {
        "code": link.code,
        "url": link.url,
        "scan_count": link.scan_count,
        "created_at": link.created_at.isoformat(),
    }
