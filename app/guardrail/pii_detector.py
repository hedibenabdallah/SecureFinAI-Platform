import asyncio
from typing import Any

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from app.schemas.proxy import PIIDetectionResult

TARGET_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "ACCOUNT_BALANCE",
]

LANGUAGE_CODES = ["en", "fr", "ar"]


def _build_balance_recognizer(language: str, patterns: list[Pattern]) -> PatternRecognizer:
    return PatternRecognizer(
        supported_entity="ACCOUNT_BALANCE",
        name=f"account_balance_{language}",
        patterns=patterns,
        supported_language=language,
    )


def _register_balance_recognizers(analyzer: AnalyzerEngine) -> None:
    english = _build_balance_recognizer(
        "en",
        [
            Pattern("en_balance", r"(?i)(balance|amount)[:\s]*\$?\s*[\d,.]+", 0.75),
            Pattern("en_currency", r"\$\s*[\d,]+\.?\d*", 0.65),
        ],
    )
    french = _build_balance_recognizer(
        "fr",
        [
            Pattern("fr_balance", r"(?i)(solde|montant)[:\s]*€?\s*[\d,.]+", 0.75),
            Pattern("fr_currency", r"€\s*[\d,]+\.?\d*", 0.65),
        ],
    )
    arabic = _build_balance_recognizer(
        "ar",
        [
            Pattern("ar_balance", r"رصيد[:\s]*[\d,.]+", 0.75),
            Pattern("ar_amount", r"[\d,.]+\s*(?:دينار|DT|TND)", 0.65),
        ],
    )
    for recognizer in (english, french, arabic):
        analyzer.registry.add_recognizer(recognizer)


def _create_analyzer() -> AnalyzerEngine:
    config = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_sm"},
            {"lang_code": "fr", "model_name": "fr_core_news_sm"},
            {"lang_code": "ar", "model_name": "ar_core_news_sm"},
        ],
    }
    provider = NlpEngineProvider(nlp_configuration=config)
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=LANGUAGE_CODES,
    )
    _register_balance_recognizers(analyzer)
    return analyzer


def _entity_to_dict(text: str, result: Any) -> dict[str, Any]:
    return {
        "entity_type": result.entity_type,
        "start": result.start,
        "end": result.end,
        "score": result.score,
        "text": text[result.start : result.end],
    }


def _run_analysis(text: str, analyzer: AnalyzerEngine) -> list[Any]:
    findings: list[Any] = []
    for language in LANGUAGE_CODES:
        batch = analyzer.analyze(
            text=text,
            language=language,
            entities=TARGET_ENTITIES,
        )
        findings.extend(batch)
    return findings


class PIIDetector:
    def __init__(self) -> None:
        self._analyzer = _create_analyzer()
        self._anonymizer = AnonymizerEngine()

    async def detect_and_anonymize(self, text: str) -> PIIDetectionResult:
        return await asyncio.to_thread(self._process, text)

    def _process(self, text: str) -> PIIDetectionResult:
        findings = _run_analysis(text, self._analyzer)
        entities = [_entity_to_dict(text, item) for item in findings]
        if not entities:
            return PIIDetectionResult(
                pii_found=False,
                entities=[],
                anonymized_text=text,
            )
        operators = {"DEFAULT": OperatorConfig("replace", {"new_value": "<PII>"})}
        for entity_name in TARGET_ENTITIES:
            operators[entity_name] = OperatorConfig(
                "replace",
                {"new_value": f"<{entity_name}>"},
            )
        anonymized = self._anonymizer.anonymize(
            text=text,
            analyzer_results=findings,
            operators=operators,
        )
        return PIIDetectionResult(
            pii_found=True,
            entities=entities,
            anonymized_text=anonymized.text,
        )
