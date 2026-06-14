import hashlib
import time
import uuid

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.guardrail.injection_detector import InjectionDetector
from app.guardrail.pii_detector import PIIDetector
from app.models.audit_log import AuditLog
from app.schemas.proxy import ProxyRequest, ProxyResponse
from app.services.cache_service import CacheService

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class LLMProxyService:
    def __init__(
        self,
        settings: Settings,
        pii_detector: PIIDetector,
        injection_detector: InjectionDetector,
        cache_service: CacheService,
    ) -> None:
        self._settings = settings
        self._pii_detector = pii_detector
        self._injection_detector = injection_detector
        self._cache_service = cache_service

    async def handle(
        self,
        request: ProxyRequest,
        provider: str,
        session: AsyncSession,
    ) -> ProxyResponse:
        started = time.perf_counter()
        request_id = str(uuid.uuid4())
        prompt_hash = _hash_text(request.prompt)

        pii_result = await self._pii_detector.detect_and_anonymize(request.prompt)
        injection_result = await self._injection_detector.detect(request.prompt)

        if injection_result.detected and injection_result.severity == "HIGH":
            return await self._finalize(
                session=session,
                request_id=request_id,
                prompt_hash=prompt_hash,
                anonymized_prompt=pii_result.anonymized_text,
                pii_entities=pii_result.entities,
                injection_detected=True,
                provider=provider,
                response_text="",
                response_hash=None,
                status="blocked",
                cached=False,
                started=started,
            )

        cached_response = await self._cache_service.lookup(pii_result.anonymized_text)
        if cached_response:
            return await self._finalize(
                session=session,
                request_id=request_id,
                prompt_hash=prompt_hash,
                anonymized_prompt=pii_result.anonymized_text,
                pii_entities=pii_result.entities,
                injection_detected=injection_result.detected,
                provider=provider,
                response_text=cached_response,
                response_hash=_hash_text(cached_response),
                status="cache_hit",
                cached=True,
                started=started,
            )

        llm_response = await self._forward_to_llm(
            provider=provider,
            prompt=pii_result.anonymized_text,
            model=request.model,
        )
        await self._cache_service.store(pii_result.anonymized_text, llm_response)

        return await self._finalize(
            session=session,
            request_id=request_id,
            prompt_hash=prompt_hash,
            anonymized_prompt=pii_result.anonymized_text,
            pii_entities=pii_result.entities,
            injection_detected=injection_result.detected,
            provider=provider,
            response_text=llm_response,
            response_hash=_hash_text(llm_response),
            status="completed",
            cached=False,
            started=started,
        )

    async def _forward_to_llm(
        self,
        provider: str,
        prompt: str,
        model: str | None,
    ) -> str:
        if provider == "openai":
            return await self._call_openai(prompt, model)
        if provider == "anthropic":
            return await self._call_anthropic(prompt, model)
        raise ValueError(f"Unsupported provider: {provider}")

    async def _call_openai(self, prompt: str, model: str | None) -> str:
        client = AsyncOpenAI(api_key=self._settings.openai_api_key)
        completion = await client.chat.completions.create(
            model=model or DEFAULT_OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content or ""

    async def _call_anthropic(self, prompt: str, model: str | None) -> str:
        client = AsyncAnthropic(api_key=self._settings.anthropic_api_key)
        message = await client.messages.create(
            model=model or DEFAULT_ANTHROPIC_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    async def _finalize(
        self,
        session: AsyncSession,
        request_id: str,
        prompt_hash: str,
        anonymized_prompt: str,
        pii_entities: list,
        injection_detected: bool,
        provider: str,
        response_text: str,
        response_hash: str | None,
        status: str,
        cached: bool,
        started: float,
    ) -> ProxyResponse:
        latency_ms = int((time.perf_counter() - started) * 1000)
        audit_row = AuditLog(
            request_id=request_id,
            original_prompt_hash=prompt_hash,
            anonymized_prompt=anonymized_prompt,
            pii_detected=pii_entities or None,
            injection_detected=injection_detected,
            llm_provider=provider,
            response_hash=response_hash,
            latency_ms=latency_ms,
            status=status,
        )
        session.add(audit_row)
        await session.commit()
        return ProxyResponse(
            request_id=request_id,
            response=response_text,
            status=status,
            cached=cached,
        )


def build_proxy_service() -> LLMProxyService:
    settings = get_settings()
    cache = CacheService(
        qdrant_url=settings.qdrant_url,
        openai_api_key=settings.openai_api_key,
        threshold=settings.cache_similarity_threshold,
    )
    return LLMProxyService(
        settings=settings,
        pii_detector=PIIDetector(),
        injection_detector=InjectionDetector(),
        cache_service=cache,
    )
