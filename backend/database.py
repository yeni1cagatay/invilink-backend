import os
import random
import string
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./invilink.db")
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


Base.metadata.create_all(bind=engine)


def generate_code(length: int = 6) -> str:
    return "".join(random.choices(CHARSET, k=length))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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
