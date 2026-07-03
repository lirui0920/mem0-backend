from typing import Any
from typing import Literal

from pydantic import BaseModel, Field


class MemoryTagMetadata(BaseModel):
    emotion: Literal["happy", "sad", "angry", "anxious", "neutral"]
    type: Literal["fact", "chat", "preference", "event", "summary"]
    importance: Literal["low", "medium", "high"]
    topic: str = Field(min_length=1, max_length=80)
    timestamp: str = Field(min_length=1, max_length=64)


class StructuredMemory(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    content: str = Field(min_length=1, max_length=8000)
    metadata: MemoryTagMetadata


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    request_id: str
    user_id: str
    intent: dict[str, Any]
    memory: StructuredMemory | None = None
    response: str
    memories: list[dict[str, Any]]


class MemorySearchRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=10, ge=1, le=50)


class MemorySearchResponse(BaseModel):
    results: list[dict[str, Any]]


class MemoryStabilityTestRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    test_cases: list[str] = Field(min_length=1, max_length=20)
    repeat: int = Field(default=5, ge=2, le=10)


class MemoryAddRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    content: str = Field(min_length=1, max_length=8000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryAddResponse(BaseModel):
    memory: StructuredMemory
    result: dict[str, Any] | list[dict[str, Any]] | str | None


class TimeRange(BaseModel):
    start: str
    end: str
    start_epoch: int
    end_epoch: int


class MemorySummary(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    daily_summary: str = Field(min_length=1)
    emotional_trend: str = Field(min_length=1)
    key_events: list[str] = Field(default_factory=list)
    new_user_preferences: list[str] = Field(default_factory=list)
    time_range: TimeRange


class SummaryRunRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    force: bool = False
    limit: int = Field(default=500, ge=1, le=1000)


class SummaryRunResponse(BaseModel):
    created: bool
    reason: str
    summary: MemorySummary | None = None
    source_memory_count: int = 0
    archived_memory_ids: list[str] = Field(default_factory=list)
    result: dict[str, Any] | list[dict[str, Any]] | str | None = None


class DiaryGenerateRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    timezone: str = Field(default="Asia/Shanghai", max_length=64)
    limit: int = Field(default=100, ge=1, le=500)


class DiaryGenerateResponse(BaseModel):
    user_id: str
    diary: str
    memory_count: int
    memories: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    app: str
