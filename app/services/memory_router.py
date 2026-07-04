from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.memory import (
    MemoryNamespaceKind,
    MemoryType,
    UnifiedMemory,
    UnifiedMemoryMetadata,
    namespace_kind_for,
    normalize_memory_type,
)

MemoryRouteReason = Literal["formatted", "not_stored"]


class MemoryRouteInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    user_id: str = Field(min_length=1, max_length=128)
    agent_id: str | None = Field(default=None, max_length=128)
    message: str = Field(min_length=1, max_length=8000)
    should_store: bool = True
    namespace: MemoryNamespaceKind | None = None
    type: MemoryType = "chat"
    llm_tag: dict[str, Any] = Field(default_factory=dict)


class MemoryRouteDecision(BaseModel):
    should_store: bool
    namespace: MemoryNamespaceKind
    type: MemoryType
    normalized_memory: UnifiedMemory | None = None
    reason: MemoryRouteReason


class MemoryRouter:
    """Deterministic memory formatting utility."""

    def route(self, payload: MemoryRouteInput | dict[str, Any]) -> MemoryRouteDecision:
        route_input = (
            payload if isinstance(payload, MemoryRouteInput) else MemoryRouteInput.model_validate(payload)
        )
        memory_type = normalize_memory_type(route_input.type)
        agent_id = route_input.agent_id if route_input.namespace == "agent" else None
        namespace_kind = route_input.namespace or namespace_kind_for(memory_type, agent_id)

        if not route_input.should_store:
            return MemoryRouteDecision(
                should_store=False,
                namespace=namespace_kind,
                type=memory_type,
                reason="not_stored",
            )

        normalized_memory = self.normalize(route_input, memory_type, namespace_kind)
        return MemoryRouteDecision(
            should_store=True,
            namespace=namespace_kind,
            type=normalized_memory.type,
            normalized_memory=normalized_memory,
            reason="formatted",
        )

    def normalize(
        self,
        route_input: MemoryRouteInput,
        memory_type: MemoryType | None = None,
        namespace_kind: MemoryNamespaceKind | None = None,
    ) -> UnifiedMemory:
        memory_type = normalize_memory_type(memory_type or route_input.type)
        namespace_kind = namespace_kind or route_input.namespace or namespace_kind_for(memory_type, route_input.agent_id)
        agent_id = route_input.agent_id if namespace_kind == "agent" else None
        if namespace_kind == "summary":
            memory_type = "summary"

        llm_tag = dict(route_input.llm_tag or {})
        importance = llm_tag.get("importance") if "importance" in llm_tag else llm_tag.get("importance_score")
        metadata = UnifiedMemoryMetadata.model_validate(
            {
                **self._metadata_extras(llm_tag),
                "timestamp": llm_tag.get("timestamp") or datetime.now(UTC),
                "importance": importance,
                "decay": llm_tag.get("decay", 0.0),
                "feedback_weight": llm_tag.get("feedback_weight", 0.0),
            }
        )
        return UnifiedMemory.from_tagged_memory(
            user_id=route_input.user_id,
            agent_id=agent_id,
            content=route_input.message.strip(),
            memory_type=memory_type,
            metadata=metadata,
            memory_id=llm_tag.get("id"),
        )

    @staticmethod
    def _metadata_extras(llm_tag: dict[str, Any]) -> dict[str, Any]:
        extras: dict[str, Any] = {}
        for key in ("emotion", "topic", "intent", "entities"):
            if key in llm_tag:
                extras[key] = llm_tag[key]
        return extras
