from __future__ import annotations

import io
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from PIL import Image
from sqlalchemy.orm import Session

from database import create_link, get_db, get_link, increment_scan
from stega import StegaStamp

stega: Optional[StegaStamp] = None

app = FastAPI(title="Brandion API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global stega
    stega = StegaStamp()


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
    encoded = stega.encode(img, link.code)

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

    code = stega.decode(img)
    if not code:
        raise HTTPException(status_code=404, detail="Filigran bulunamadi")

    link = get_link(db, code)
    if not link:
        raise HTTPException(status_code=404, detail="Link bulunamadi")

    increment_scan(db, code)

    return {"url": link.url, "code": link.code}


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
