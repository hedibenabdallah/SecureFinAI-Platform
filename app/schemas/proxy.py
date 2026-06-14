from datetime import datetime
from typing import Any

from pydantic import BaseModel


class PIIDetectionResult(BaseModel):
    pii_found: bool
    entities: list[dict[str, Any]]
    anonymized_text: str


class InjectionDetectionResult(BaseModel):
    detected: bool
    severity: str = "LOW"
    matched_patterns: list[str] | None = None


class ProxyRequest(BaseModel):
    prompt: str
    model: str | None = None


class ProxyResponse(BaseModel):
    request_id: str
    response: str
    status: str
    cached: bool = False


class AuditEntry(BaseModel):
    request_id: str
    timestamp: datetime
    original_prompt_hash: str
    anonymized_prompt: str
    pii_detected: list[dict[str, Any]] | None = None
    injection_detected: bool
    llm_provider: str
    response_hash: str | None = None
    latency_ms: int | None = None
    status: str
