import os
import random
import string
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////app/invilink.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

CHARSET = string.ascii_uppercase + string.digits


class Base(DeclarativeBase):
    pass


class Link(Base):
    __tablename__ = "links"

    code = Column(String(7), primary_key=True, index=True)
    url = Column(String(2048), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    scan_count = Column(Integer, default=0)


class VideoLink(Base):
    __tablename__ = "video_links"

    wm_id = Column(Integer, primary_key=True, index=True)  # 0-255
    url = Column(String(2048), nullable=False)
    label = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    scan_count = Column(Integer, default=0)


class ShortLink(Base):
    """TrustMark watermark → 8-char kod → ürün URL'i"""
    __tablename__ = "short_links"

    code = Column(String(8), primary_key=True, index=True)
    url = Column(String(2048), nullable=False)
    label = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    scan_count = Column(Integer, default=0)


class WatermarkProject(Base):
    """Stüdyo için watermark projesi (örn: 'Dizi X S01E03')"""
    __tablename__ = "wm_projects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(256), nullable=False)
    client = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    slots = relationship("WatermarkSlot", back_populates="project", cascade="all, delete-orphan")


class WatermarkSlot(Base):
    """Tek bir zaman aralığı: ts_start-ts_end → ürün URL'i"""
    __tablename__ = "wm_slots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("wm_projects.id"), nullable=False)
    ts_start = Column(String(12), nullable=False)   # "01:22:30"
    ts_end = Column(String(12), nullable=False)     # "01:22:59"
    product_url = Column(String(2048), nullable=False)
    product_name = Column(String(256), nullable=True)
    short_code = Column(String(8), ForeignKey("short_links.code"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    project = relationship("WatermarkProject", back_populates="slots")


Base.metadata.create_all(bind=engine)


SHORTLINK_CHARSET = string.ascii_lowercase + string.digits  # a-z0-9, 36 chars


def generate_code(length: int = 6) -> str:
    return "".join(random.choices(CHARSET, k=length))


def generate_short_code() -> str:
    return "".join(random.choices(SHORTLINK_CHARSET, k=8))


def create_short_link(db: Session, url: str, label: str = "") -> ShortLink:
    for _ in range(20):
        code = generate_short_code()
        if not db.query(ShortLink).filter(ShortLink.code == code).first():
            sl = ShortLink(code=code, url=url, label=label or url[:80])
            db.add(sl)
            db.commit()
            db.refresh(sl)
            return sl
    raise RuntimeError("Benzersiz short kod oluşturulamadı")


def get_short_link(db: Session, code: str) -> Optional[ShortLink]:
    return db.query(ShortLink).filter(ShortLink.code == code).first()


def increment_short_scan(db: Session, code: str) -> None:
    sl = get_short_link(db, code)
    if sl:
        sl.scan_count += 1
        db.commit()


def create_project(db: Session, title: str, client: str = "") -> WatermarkProject:
    p = WatermarkProject(title=title, client=client)
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


def get_project(db: Session, project_id: int) -> Optional[WatermarkProject]:
    return db.query(WatermarkProject).filter(WatermarkProject.id == project_id).first()


def list_projects(db: Session):
    return db.query(WatermarkProject).order_by(WatermarkProject.created_at.desc()).all()


def add_slot(db: Session, project_id: int, ts_start: str, ts_end: str,
             product_url: str, product_name: str = "") -> WatermarkSlot:
    sl = create_short_link(db, product_url, product_name)
    slot = WatermarkSlot(
        project_id=project_id,
        ts_start=ts_start,
        ts_end=ts_end,
        product_url=product_url,
        product_name=product_name or product_url[:80],
        short_code=sl.code,
    )
    db.add(slot)
    db.commit()
    db.refresh(slot)
    return slot


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_video_link(db: Session, url: str, label: str = "") -> VideoLink:
    """Boş bir wm_id bul (0-255) ve video link oluştur."""
    used = {r.wm_id for r in db.query(VideoLink.wm_id).all()}
    free = [i for i in range(256) if i not in used]
    if not free:
        raise RuntimeError("Tüm 256 video slot dolu")
    wm_id = random.choice(free)
    vl = VideoLink(wm_id=wm_id, url=url, label=label or url[:80])
    db.add(vl)
    db.commit()
    db.refresh(vl)
    return vl


def get_video_link(db: Session, wm_id: int) -> Optional[VideoLink]:
    return db.query(VideoLink).filter(VideoLink.wm_id == wm_id).first()


def increment_video_scan(db: Session, wm_id: int) -> None:
    vl = get_video_link(db, wm_id)
    if vl:
        vl.scan_count += 1
        db.commit()


def list_video_links(db: Session):
    return db.query(VideoLink).order_by(VideoLink.created_at.desc()).all()


def create_link(db: Session, url: str) -> Link:
    for _ in range(10):
        code = generate_code(6)
        if not db.query(Link).filter(Link.code == code).first():
            link = Link(code=code, url=url)
            db.add(link)
            db.commit()
            db.refresh(link)
            return link
    raise RuntimeError("Benzersiz kod olusturulamadi")


def get_link(db: Session, code: str) -> Optional[Link]:
    return db.query(Link).filter(Link.code == code.upper()).first()


def increment_scan(db: Session, code: str) -> None:
    link = get_link(db, code)
    if link:
        link.scan_count += 1
        db.commit()
