from __future__ import annotations

import os
from functools import lru_cache
from typing import Generator

from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from influxdb_client import InfluxDBClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

bearer_scheme = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def get_database_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('POSTGRES_USER', 'urjanet')}:{os.getenv('POSTGRES_PASSWORD', 'urjanet-secret')}"
        f"@{os.getenv('POSTGRES_HOST', 'postgres')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'urjanet')}"
    )
    engine = create_engine(url, future=True)
    return engine


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_database_engine())


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@lru_cache(maxsize=1)
def get_influx_client() -> InfluxDBClient:
    return InfluxDBClient(
        url=os.getenv("INFLUXDB_URL", "http://influxdb:8086"),
        token=os.getenv("INFLUXDB_TOKEN", "local-dev-token"),
        org=os.getenv("INFLUXDB_ORG", "urjanet"),
    )


def get_current_subject(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> dict:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing authorization token")
    token = credentials.credentials
    expected = os.getenv("JWT_SECRET", "development-secret")
    if token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")
    return {"sub": "demo-user"}
