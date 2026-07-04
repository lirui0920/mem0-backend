from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MemoryNamespaceKind = Literal["user", "agent", "summary"]
MemoryType = Literal["chat", "sleep", "preference", "event", "summary"]

ALLOWED_MEMORY_TYPES: set[str] = {"chat", "sleep", "preference", "event", "summary"}
TYPE_NORMALIZATION: dict[str, MemoryType] = {
    "fact": "event",
    "health": "event",
    "emotion": "chat",
    "conversation": "chat",
}


def normalize_memory_type(value: Any) -> MemoryType:
    memory_type = str(value or "chat").strip().lower()
    memory_type = TYPE_NORMALIZATION.get(memory_type, memory_type)
    if memory_type not in ALLOWED_MEMORY_TYPES:
        return "chat"
    return memory_type  # type: ignore[return-value]


def namespace_kind_for(memory_type: MemoryType, agent_id: str | None = None) -> MemoryNamespaceKind:
    if memory_type == "summary":
        return "summary"
    if agent_id:
        return "agent"
    return "user"


def resolve_memory_namespace(user_id: str, memory_type: MemoryType, agent_id: str | None = None) -> str:
    if not user_id:
        raise ValueError("Memory user_id is required.")
    kind = namespace_kind_for(memory_type, agent_id)
    if kind == "summary":
        return f"summary:{user_id}"
    if kind == "agent":
        return f"agent:{user_id}:{agent_id}"
    return f"user:{user_id}"


class UnifiedMemoryMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    decay: float = Field(default=0.0, ge=0.0)
    feedback_weight: float = Field(default=0.0, ge=-0.5, le=0.5)
    event_boost: float = Field(default=0.0, ge=0.0, le=0.3)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _parse_timestamp(cls, value: Any) -> Any:
        if value is None or value == "":
            return datetime.now(UTC)
        return value

    @field_validator("importance", mode="before")
    @classmethod
    def _parse_importance(cls, value: Any) -> float:
        if isinstance(value, int | float):
            score = float(value)
        else:
            score = {"low": 0.2, "medium": 0.5, "high": 0.9}.get(str(value).lower(), 0.5)
        return max(0.0, min(1.0, score))

    @field_validator("decay", mode="before")
    @classmethod
    def _parse_float(cls, value: Any) -> float:
        if value is None or value == "":
            return 0.0
        return float(value)

    @field_validator("feedback_weight", mode="before")
    @classmethod
    def _clamp_feedback_weight(cls, value: Any) -> float:
        if value is None or value == "":
            return 0.0
        return max(-0.5, min(0.5, float(value)))

    @field_validator("event_boost", mode="before")
    @classmethod
    def _clamp_event_boost(cls, value: Any) -> float:
        if value is None or value == "":
            return 0.0
        return max(0.0, min(0.3, float(value)))


class UnifiedMemory(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str = Field(min_length=1, max_length=128)
    agent_id: str | None = Field(default=None, max_length=128)
    namespace: str = ""
    type: MemoryType = "chat"
    content: str = Field(min_length=1, max_length=8000)
    embedding: list[float] | None = None
    metadata: UnifiedMemoryMetadata = Field(default_factory=UnifiedMemoryMetadata)

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_type(cls, value: Any) -> MemoryType:
        return normalize_memory_type(value)

    @model_validator(mode="after")
    def _resolve_namespace(self) -> "UnifiedMemory":
        expected_namespace = resolve_memory_namespace(self.user_id, self.type, self.agent_id)
        if self.namespace and self.namespace != expected_namespace:
            raise ValueError(f"Memory namespace must be {expected_namespace}.")
        self.namespace = expected_namespace
        return self

    @classmethod
    def from_tagged_memory(
        cls,
        *,
        user_id: str,
        content: str,
        memory_type: Any = "chat",
        agent_id: str | None = None,
        metadata: dict[str, Any] | UnifiedMemoryMetadata | None = None,
        memory_id: str | None = None,
    ) -> "UnifiedMemory":
        metadata_object = (
            metadata
            if isinstance(metadata, UnifiedMemoryMetadata)
            else UnifiedMemoryMetadata.model_validate(metadata or {})
        )
        data = {
            "user_id": user_id,
            "agent_id": agent_id,
            "type": normalize_memory_type(memory_type),
            "content": content,
            "metadata": metadata_object,
        }
        if memory_id:
            data["id"] = memory_id
        return cls.model_validate(data)

    def to_mem0_metadata(self, extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata = self.metadata.model_dump(mode="json")
        metadata.update(extra_metadata or {})
        metadata["timestamp"] = self.metadata.timestamp.isoformat()
        metadata["importance"] = self.metadata.importance
        metadata["decay"] = self.metadata.decay
        metadata["feedback_weight"] = self.metadata.feedback_weight
        metadata["event_boost"] = self.metadata.event_boost
        metadata["id"] = self.id
        metadata["user_id"] = self.user_id
        metadata["agent_id"] = self.agent_id
        metadata["namespace"] = self.namespace
        metadata["namespace_kind"] = namespace_kind_for(self.type, self.agent_id)
        metadata["type"] = self.type
        metadata["content"] = self.content
        metadata["memory_object"] = self.model_dump(mode="json")
        return metadata
