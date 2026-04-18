import os
import random
import string
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////app/data/invilink.db")
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


Base.metadata.create_all(bind=engine)


def generate_code(length: int = 6) -> str:
    return "".join(random.choices(CHARSET, k=length))


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
