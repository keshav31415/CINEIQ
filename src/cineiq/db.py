import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from cineiq.config import get_config
from pathlib import Path

class Base(DeclarativeBase):
    pass

def get_engine():
    cfg = get_config()
    db_path = cfg['data']['database']
    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}")

def get_session():
    return sessionmaker(bind=get_engine())()
