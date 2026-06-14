import asyncio
import re

from app.schemas.proxy import InjectionDetectionResult

SEVERITY_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

INJECTION_RULES: list[tuple[str, str, str]] = [
    ("jailbreak", r"\bjailbreak\b", "HIGH"),
    ("dan_mode", r"\bdan\s+mode\b", "HIGH"),
    ("ignore_instructions", r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", "HIGH"),
    ("reveal_system_prompt", r"(reveal|show|print|display|repeat)\s+(your\s+)?(system\s+prompt|initial\s+instructions)", "HIGH"),
    ("act_as_system", r"act\s+as\s+(the\s+)?system", "HIGH"),
    ("role_override", r"you\s+are\s+now\s+(a|an|the)\s+", "HIGH"),
    ("developer_mode", r"\bdeveloper\s+mode\b", "HIGH"),
    ("instruction_injection", r"new\s+instructions\s*:", "MEDIUM"),
    ("forget_rules", r"forget\s+(your\s+)?(rules|guidelines|policy)", "MEDIUM"),
    ("disregard_safety", r"disregard\s+(all\s+)?(safety|security)\s+(rules|guidelines)", "MEDIUM"),
    ("override_role", r"override\s+(your\s+)?(role|persona|instructions)", "MEDIUM"),
    ("hidden_instruction", r"<\s*/?\s*system\s*>", "MEDIUM"),
    ("prompt_extraction", r"what\s+(are|were)\s+your\s+(original|hidden)\s+instructions", "MEDIUM"),
    ("pretend_role", r"pretend\s+(you\s+are|to\s+be)\s+", "LOW"),
    ("roleplay_as", r"role\s*play\s+as\s+", "LOW"),
    ("hypothetical_bypass", r"hypothetically[,]?\s+(ignore|disregard|break)\s+", "LOW"),
]


def _compile_rules() -> list[tuple[str, re.Pattern[str], str]]:
    compiled: list[tuple[str, re.Pattern[str], str]] = []
    for name, pattern, severity in INJECTION_RULES:
        compiled.append((name, re.compile(pattern, re.IGNORECASE), severity))
    return compiled


def _highest_severity(current: str, candidate: str) -> str:
    if SEVERITY_RANK[candidate] > SEVERITY_RANK[current]:
        return candidate
    return current


class InjectionDetector:
    def __init__(self) -> None:
        self._rules = _compile_rules()

    async def detect(self, text: str) -> InjectionDetectionResult:
        return await asyncio.to_thread(self._scan, text)

    def _scan(self, text: str) -> InjectionDetectionResult:
        matched: list[str] = []
        severity = "LOW"
        for name, pattern, rule_severity in self._rules:
            if pattern.search(text):
                matched.append(name)
                severity = _highest_severity(severity, rule_severity)
        if not matched:
            return InjectionDetectionResult(
                detected=False,
                severity="LOW",
                matched_patterns=None,
            )
        return InjectionDetectionResult(
            detected=True,
            severity=severity,
            matched_patterns=matched,
        )
