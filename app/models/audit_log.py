from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str] = mapped_column(String(36))
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    original_prompt_hash: Mapped[str] = mapped_column(String(64))
    anonymized_prompt: Mapped[str] = mapped_column(Text)
    pii_detected: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    injection_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    llm_provider: Mapped[str] = mapped_column(String(20))
    response_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(20))
