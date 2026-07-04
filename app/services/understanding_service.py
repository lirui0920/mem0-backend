from dataclasses import asdict, dataclass
from typing import Literal


InputType = Literal["chat", "health", "sleep", "emotion", "system"]


@dataclass(frozen=True)
class StructuredUnderstanding:
    input_type: InputType
    intent: str
    entities: list[str]
    severity: float
    should_store: bool
    should_trigger: bool
    should_trigger_hint: str | None = None

    def model_dump(self) -> dict:
        return asdict(self)


class UnderstandingService:
    """Rule-based pre-processing layer for raw user input."""

    _SLEEP_TERMS = ("sleep", "insomnia", "slept", "睡不着", "失眠", "睡眠")
    _SAD_TERMS = ("sad", "depressed", "down", "难过", "伤心", "沮丧")
    _ANXIOUS_TERMS = ("anxious", "anxiety", "worried", "nervous", "焦虑", "担心", "紧张")
    _HEALTH_TERMS = ("heart rate", "headache", "tired", "头痛", "心率", "很累")
    _SYSTEM_TERMS = ("/system", "[system]", "system:")

    def parse(self, message: str) -> StructuredUnderstanding:
        text = message.strip()
        lower = text.lower()

        if self._contains_any(lower, self._SYSTEM_TERMS):
            return StructuredUnderstanding(
                input_type="system",
                intent="system_instruction",
                entities=[],
                severity=0.0,
                should_store=False,
                should_trigger=False,
                should_trigger_hint=None,
            )

        if self._contains_any(lower, self._SLEEP_TERMS):
            return StructuredUnderstanding(
                input_type="sleep",
                intent="sleep_deprivation",
                entities=self._matched_terms(lower, self._SLEEP_TERMS),
                severity=0.75,
                should_store=True,
                should_trigger=True,
                should_trigger_hint="health_sleep_signal",
            )

        if self._contains_any(lower, self._HEALTH_TERMS):
            return StructuredUnderstanding(
                input_type="health",
                intent="health_signal",
                entities=self._matched_terms(lower, self._HEALTH_TERMS),
                severity=0.65,
                should_store=True,
                should_trigger=True,
                should_trigger_hint="health_signal",
            )

        if self._contains_any(lower, self._SAD_TERMS) or self._contains_any(lower, self._ANXIOUS_TERMS):
            entities = [
                *self._matched_terms(lower, self._SAD_TERMS),
                *self._matched_terms(lower, self._ANXIOUS_TERMS),
            ]
            return StructuredUnderstanding(
                input_type="emotion",
                intent="emotional_state",
                entities=entities,
                severity=0.7,
                should_store=True,
                should_trigger=True,
                should_trigger_hint="emotional_signal",
            )

        return StructuredUnderstanding(
            input_type="chat",
            intent="normal_chat",
            entities=[],
            severity=0.1,
            should_store=len(text) >= 3,
            should_trigger=False,
            should_trigger_hint=None,
        )

    @staticmethod
    def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
        return any(term in text for term in terms)

    @staticmethod
    def _matched_terms(text: str, terms: tuple[str, ...]) -> list[str]:
        return [term for term in terms if term in text]
