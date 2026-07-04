from dataclasses import asdict, dataclass, field
from typing import Any

from app.services.understanding_service import StructuredUnderstanding


@dataclass
class MemoryContext:
    intent: Any
    retrieved_memories: list[dict[str, Any]]
    context_messages: list[dict[str, str]]
    memory_state: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return {
            "intent": asdict(self.intent) if hasattr(self.intent, "__dataclass_fields__") else self.intent,
            "retrieved_memories": self.retrieved_memories,
            "context_messages": self.context_messages,
            "memory_state": self.memory_state,
        }


@dataclass(frozen=True)
class TriggerDecision:
    action: str
    reason: str
    event_type: str | None = None
    event_subtype: str | None = None
    confidence: float = 0.0
    priority: int | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SystemBus:
    understanding: StructuredUnderstanding | None = None
    memory_context: MemoryContext | None = None
    trigger_events: list[TriggerDecision] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return {
            "understanding": self.understanding.model_dump() if self.understanding else None,
            "memory_context": self.memory_context.model_dump() if self.memory_context else None,
            "trigger_events": [event.model_dump() for event in self.trigger_events],
        }
