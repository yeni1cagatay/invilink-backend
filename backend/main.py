from __future__ import annotations

import io
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from PIL import Image
from sqlalchemy.orm import Session

import zipfile
import json as _json
from database import (create_link, get_db, get_link, increment_scan,
                       create_video_link, get_video_link, increment_video_scan, list_video_links,
                       create_short_link, get_short_link, increment_short_scan,
                       create_project, get_project, list_projects, add_slot)
import trustmark_engine as tm_engine
import watermark
import temporal_watermark as twm
import spread_spectrum as ss
import dct_watermark as dct

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


@app.get("/studio", response_class=HTMLResponse)
async def postprod_studio():
    with open("postprod.html", encoding="utf-8") as f:
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
    debug_path = f"/tmp/debug_frame_{int(time.time())}.jpg"
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
    import os, time as _time
    raw = await image.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    dbg = f"/tmp/debug_ss_{int(_time.time())}.jpg"
    img.save(dbg, "JPEG", quality=95)
    print(f"[SS-DBG] frame saved: {dbg} size={img.size} mode={img.mode}")
    wm_id, best_corr, margin = ss.decode_scores(img)
    if wm_id is None:
        raise HTTPException(
            status_code=404,
            detail=f"Filigran bulunamadı (corr={best_corr:.2f} margin={margin:.2f})",
        )
    vl = get_video_link(db, wm_id)
    if not vl:
        raise HTTPException(status_code=404, detail="ID kayıtlı değil")
    increment_video_scan(db, wm_id)
    return {"url": vl.url, "wm_id": wm_id, "label": vl.label}


@app.post("/api/ss/decode-multi")
async def ss_decode_multi(
    images: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    """2-5 fotoğraf → korelasyon ortalaması → URL. Tek frame'den daha dayanıklı."""
    frames = [Image.open(io.BytesIO(await img.read())) for img in images]
    wm_id, best_avg = ss.decode_multi(frames)
    if wm_id is None:
        raise HTTPException(
            status_code=404,
            detail=f"Filigran bulunamadı (avg_corr={best_avg:.2f})",
        )
    vl = get_video_link(db, wm_id)
    if not vl:
        raise HTTPException(status_code=404, detail="ID kayıtlı değil")
    increment_video_scan(db, wm_id)
    return {"url": vl.url, "wm_id": wm_id, "label": vl.label}


@app.get("/api/ss/overlay/{wm_id}")
async def ss_overlay_get(wm_id: int, db: Session = Depends(get_db)):
    """GET ile overlay indir — telefon tarayıcısından kullanım için."""
    if not get_video_link(db, wm_id):
        raise HTTPException(status_code=404, detail="wm_id bulunamadı")
    img = ss.encode_overlay(wm_id)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="overlay_{wm_id}.png"'})


@app.get("/api/ss/noise/{wm_id}")
async def ss_noise_overlay(wm_id: int, db: Session = Depends(get_db)):
    """Gürültü görünümlü SS overlay — tarayıcıda görüntüle."""
    if not get_video_link(db, wm_id):
        raise HTTPException(status_code=404, detail="wm_id bulunamadı")
    img = ss.encode_noise_overlay(wm_id)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


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


# ─── DCT WATERMARK ───────────────────────────────────────────────────────────

@app.post("/api/dct/encode")
async def dct_encode(
    image: UploadFile = File(...),
    wm_id: int = Form(...),
    db: Session = Depends(get_db),
):
    """Görsel + wm_id → DCT filigranlanmış PNG indir."""
    vl = get_video_link(db, wm_id)
    if not vl:
        raise HTTPException(status_code=404, detail="wm_id bulunamadı")
    img = Image.open(io.BytesIO(await image.read()))
    encoded = dct.encode(img, wm_id)
    buf = io.BytesIO()
    encoded.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="dct_{wm_id}.png"'})


@app.post("/api/dct/decode")
async def dct_decode(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Kamera frame → DCT decode → URL."""
    raw = await image.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    wm_id = dct.decode(img)
    if wm_id is None:
        raise HTTPException(status_code=404, detail="Filigran bulunamadı")
    vl = get_video_link(db, wm_id)
    if not vl:
        raise HTTPException(status_code=404, detail="ID kayıtlı değil")
    increment_video_scan(db, wm_id)
    return {"url": vl.url, "wm_id": wm_id, "label": vl.label}


@app.get("/api/dct/overlay/{wm_id}")
async def dct_overlay(wm_id: int, db: Session = Depends(get_db)):
    """wm_id → noise overlay PNG indir. Video editörde Screen blend ile kullan."""
    if not get_video_link(db, wm_id):
        raise HTTPException(status_code=404, detail="wm_id bulunamadı")
    img = dct.generate_noise_overlay(wm_id)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="overlay_{wm_id}.png"'})


@app.post("/api/ss/diagnose")
async def ss_diagnose(image: UploadFile = File(...)):
    """DB gerektirmez — kamera görselinin ham korelasyon skorlarını döner."""
    import spread_spectrum as ss
    from PIL import Image
    import io, numpy as np

    img = Image.open(io.BytesIO(await image.read()))
    candidates = ss._prepare_candidates(img)

    top5_per_candidate = []
    for idx, gray in enumerate(candidates):
        flat = gray.flatten()
        norm = float(np.linalg.norm(flat))
        if norm < 1e-6:
            continue
        pn_mat = ss._pn_matrix(ss.DECODE_H, ss.DECODE_W)
        corrs = (pn_mat @ flat / norm).tolist()
        sorted_corrs = sorted(enumerate(corrs), key=lambda x: -x[1])[:5]
        top5_per_candidate.append({
            "scale_idx": idx,
            "top5": [{"id": i, "corr": round(c, 3)} for i, c in sorted_corrs],
        })

    return {
        "image_size": list(img.size),
        "threshold": ss.THRESHOLD,
        "margin_req": ss.MARGIN,
        "candidates": top5_per_candidate,
    }


# ─── TRUSTMARK ───────────────────────────────────────────────────────────────

@app.post("/api/tm/decode")
async def tm_decode(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Kamera frame → TrustMark decode → ürün URL'i."""
    raw = await image.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    code, detected, conf = tm_engine.decode_watermark(img)
    if not detected or not code:
        raise HTTPException(status_code=404, detail=f"Watermark bulunamadı (conf={conf:.2f})")
    sl = get_short_link(db, code.strip())
    if not sl:
        raise HTTPException(status_code=404, detail=f"Kod kayıtlı değil: {code}")
    increment_short_scan(db, code.strip())
    return {"url": sl.url, "code": code, "label": sl.label, "confidence": conf}


@app.get("/r/{code}")
async def short_redirect(code: str, db: Session = Depends(get_db)):
    """brd.io/XXXXXXXX → ürün URL'ine redirect."""
    sl = get_short_link(db, code)
    if not sl:
        raise HTTPException(status_code=404, detail="Bağlantı bulunamadı")
    increment_short_scan(db, code)
    return RedirectResponse(url=sl.url, status_code=302)


# ─── WATERMARK PROJELERİ ─────────────────────────────────────────────────────

@app.post("/api/projects")
async def create_project_endpoint(
    title: str = Form(...),
    client: str = Form(""),
    db: Session = Depends(get_db),
):
    """Yeni watermark projesi oluştur."""
    p = create_project(db, title, client)
    return {"id": p.id, "title": p.title, "client": p.client}


@app.get("/api/projects")
async def list_projects_endpoint(db: Session = Depends(get_db)):
    projects = list_projects(db)
    return [
        {
            "id": p.id,
            "title": p.title,
            "client": p.client,
            "slot_count": len(p.slots),
            "created_at": p.created_at.isoformat(),
        }
        for p in projects
    ]


@app.get("/api/projects/{project_id}")
async def get_project_endpoint(project_id: int, db: Session = Depends(get_db)):
    p = get_project(db, project_id)
    if not p:
        raise HTTPException(status_code=404, detail="Proje bulunamadı")
    return {
        "id": p.id,
        "title": p.title,
        "client": p.client,
        "created_at": p.created_at.isoformat(),
        "slots": [
            {
                "id": s.id,
                "ts_start": s.ts_start,
                "ts_end": s.ts_end,
                "product_name": s.product_name,
                "product_url": s.product_url,
                "short_code": s.short_code,
            }
            for s in p.slots
        ],
    }


@app.post("/api/projects/{project_id}/slots")
async def add_slot_endpoint(
    project_id: int,
    ts_start: str = Form(...),
    ts_end: str = Form(...),
    product_url: str = Form(...),
    product_name: str = Form(""),
    db: Session = Depends(get_db),
):
    """Projeye timestamp aralığı + ürün URL'i ekle."""
    p = get_project(db, project_id)
    if not p:
        raise HTTPException(status_code=404, detail="Proje bulunamadı")
    slot = add_slot(db, project_id, ts_start, ts_end, product_url, product_name)
    return {
        "id": slot.id,
        "ts_start": slot.ts_start,
        "ts_end": slot.ts_end,
        "product_name": slot.product_name,
        "short_code": slot.short_code,
    }


@app.get("/api/projects/{project_id}/zip")
async def download_project_zip(project_id: int, db: Session = Depends(get_db)):
    """
    Proje ZIP'i oluştur ve indir.
    İçerik: her slot için residual PNG + uygulama talimatları.
    """
    p = get_project(db, project_id)
    if not p:
        raise HTTPException(status_code=404, detail="Proje bulunamadı")
    if not p.slots:
        raise HTTPException(status_code=400, detail="Projede henüz slot yok")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        readme_lines = [
            f"# Brandion Watermark Paketi",
            f"Proje: {p.title}",
            f"Müşteri: {p.client or '-'}",
            "",
            "## Uygulama Talimatları",
            "1. Her PNG dosyasını After Effects / Premiere / DaVinci'de bir adjustment layer olarak ekleyin.",
            "2. Blend mode: **Add**",
            "3. Opacity: **100%** (zaten görünmez — azaltmayın)",
            "4. Her katmanı yalnızca belirtilen timestamp aralığında aktif yapın.",
            "",
            "## Zaman Aralıkları",
        ]

        for slot in p.slots:
            readme_lines.append(
                f"- [{slot.ts_start} → {slot.ts_end}] {slot.product_name}: {slot.product_url}"
            )
            png_bytes = tm_engine.generate_residual_png(slot.short_code)
            filename = f"wm_{slot.ts_start.replace(':','')}-{slot.ts_end.replace(':','')}.png"
            zf.writestr(filename, png_bytes)

        readme_lines += [
            "",
            "## Teknik Detaylar",
            "- Watermark yöntemi: TrustMark (Adobe Research, CVPR 2024)",
            "- Çözünürlük: 1920×1080",
            "- İzleyici Brandion uygulamasıyla ekranı tararsa otomatik ürün sayfasına yönlendirilir.",
            "",
            "brandion.io",
        ]
        zf.writestr("TALIMATLAR.md", "\n".join(readme_lines))

        slots_data = [
            {
                "ts_start": s.ts_start,
                "ts_end": s.ts_end,
                "product_name": s.product_name,
                "product_url": s.product_url,
                "short_code": s.short_code,
                "file": f"wm_{s.ts_start.replace(':','')}-{s.ts_end.replace(':','')}.png",
            }
            for s in p.slots
        ]
        zf.writestr("slots.json", _json.dumps(slots_data, ensure_ascii=False, indent=2))

    buf.seek(0)
    safe_title = p.title.replace(" ", "_")[:40]
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="brandion_{safe_title}.zip"'},
    )


@app.post("/api/ss/decode-temporal")
async def ss_decode_temporal(
    frame1: UploadFile = File(...),
    frame2: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Temporal modulation decode — 2 ardışık kamera karesi farkından."""
    raw1 = await frame1.read()
    raw2 = await frame2.read()
    img1 = Image.open(io.BytesIO(raw1)).convert("RGB")
    img2 = Image.open(io.BytesIO(raw2)).convert("RGB")
    wm_id, corr, margin = ss.decode_temporal_scores(img1, img2)
    if wm_id is None:
        raise HTTPException(status_code=404, detail=f"Watermark bulunamadı (corr={corr:.1f})")
    vl = get_video_link(db, wm_id)
    if not vl:
        raise HTTPException(status_code=404, detail="ID kayıtlı değil")
    increment_video_scan(db, wm_id)
    return {"url": vl.url, "wm_id": wm_id, "label": vl.label, "corr": corr}


@app.post("/api/ss/decode-med")
async def ss_decode_med(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """24px block SS decode — kamera-dayanıklı, 64px'e göre finer görünüm."""
    raw = await image.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    wm_id, corr, margin = ss.decode_med_scores(img)
    if wm_id is None:
        raise HTTPException(status_code=404, detail=f"Watermark bulunamadı (corr={corr:.1f})")
    vl = get_video_link(db, wm_id)
    if not vl:
        raise HTTPException(status_code=404, detail="ID kayıtlı değil")
    increment_video_scan(db, wm_id)
    return {"url": vl.url, "wm_id": wm_id, "label": vl.label, "corr": corr}


@app.post("/api/ss/decode-fine")
async def ss_decode_fine(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Fine-grain (8px block) SS decode."""
    raw = await image.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    wm_id, corr, margin = ss.decode_fine_scores(img)
    if wm_id is None:
        raise HTTPException(status_code=404, detail=f"Watermark bulunamadı (corr={corr:.1f})")
    vl = get_video_link(db, wm_id)
    if not vl:
        raise HTTPException(status_code=404, detail="ID kayıtlı değil")
    increment_video_scan(db, wm_id)
    return {"url": vl.url, "wm_id": wm_id, "label": vl.label, "corr": corr}
