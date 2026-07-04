from typing import Any

from app.core.system_bus import TriggerDecision
from app.services.memory_orchestrator import MemoryOrchestrator


class TriggerProcess:
    """Event execution adapter. AgentCore owns event decisions."""

    def execute(
        self,
        decision: TriggerDecision,
        user_id: str,
        orchestrator: MemoryOrchestrator,
    ) -> dict[str, Any]:
        if decision.action != "emit_event" or not decision.event_type or not decision.event_subtype:
            return {"action": decision.action, "reason": decision.reason, "event": None}

        event = orchestrator.emit_event(
            decision.event_type,
            decision.event_subtype,
            user_id,
            confidence=decision.confidence,
            priority=decision.priority,
            context=decision.context,
        )
        return {"action": decision.action, "reason": decision.reason, "event": event}
