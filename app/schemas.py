from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.models.memory import MemoryType, UnifiedMemory, UnifiedMemoryMetadata


class MemoryTagMetadata(UnifiedMemoryMetadata):
    emotion: str = Field(default="neutral", max_length=40)
    topic: str = Field(min_length=1, max_length=80)


class StructuredMemory(UnifiedMemory):
    type: MemoryType
    metadata: MemoryTagMetadata


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    user_name: str | None = Field(default=None, max_length=128)
    agent_id: str | None = Field(default=None, max_length=128)
    agent_name: str | None = Field(default=None, max_length=128)
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    request_id: str
    user_id: str
    intent: dict[str, Any]
    memory: StructuredMemory | None = None
    response: str
    memories: list[dict[str, Any]]


class SleepInput(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    user_name: str | None = Field(default=None, max_length=128)
    agent_id: str | None = Field(default=None, max_length=128)
    agent_name: str | None = Field(default=None, max_length=128)
    sleep_start: datetime
    sleep_end: datetime
    sleep_duration: float | None = Field(default=None, gt=0, le=24)
    deep_sleep_duration: float | None = Field(default=None, ge=0, le=24)
    awake_count: int | None = Field(default=None, ge=0)
    rem_sleep_duration: float | None = Field(default=None, ge=0, le=24)
    source: Literal["apple_shortcuts", "apple_watch", "manual"]

    @model_validator(mode="after")
    def _validate_sleep_window(self) -> "SleepInput":
        if self.sleep_end <= self.sleep_start:
            raise ValueError("sleep_end must be later than sleep_start.")
        computed_duration = (self.sleep_end - self.sleep_start).total_seconds() / 3600
        if computed_duration <= 0 or computed_duration > 24:
            raise ValueError("sleep window must be greater than 0 and no longer than 24 hours.")
        if self.sleep_duration is None:
            self.sleep_duration = round(computed_duration, 2)
        return self


class SleepResponse(BaseModel):
    status: str
    memory_id: str
    profile_updated: bool
    summary: dict[str, Any]


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
    user_name: str | None = Field(default=None, max_length=128)
    agent_id: str | None = Field(default=None, max_length=128)
    agent_name: str | None = Field(default=None, max_length=128)
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


class AgentSummaryRunRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    agent_id: str = Field(min_length=1, max_length=128)
    force: bool = False
    limit: int = Field(default=200, ge=1, le=1000)


class AgentSummaryRunResponse(BaseModel):
    created: bool
    reason: str
    user_id: str
    agent_id: str
    source_memory_count: int = 0
    created_count: int = 0
    summaries: list[dict[str, Any]] = Field(default_factory=list)
    results: list[Any] = Field(default_factory=list)


class ChatImportMessage(BaseModel):
    message_id: str | None = Field(default=None, max_length=128)
    timestamp: datetime
    sender_role: Literal["user", "agent", "system"]
    sender_id: str | None = Field(default=None, max_length=128)
    sender_name: str | None = Field(default=None, max_length=128)
    content: str = Field(min_length=1, max_length=8000)


class ChatImportRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    user_name: str | None = Field(default=None, max_length=128)
    agent_id: str = Field(min_length=1, max_length=128)
    agent_name: str | None = Field(default=None, max_length=128)
    source: str = Field(default="local_chat_import", max_length=80)
    store_raw: bool = True
    summarize: bool = True
    messages: list[ChatImportMessage] = Field(min_length=1, max_length=1000)


class ChatImportResponse(BaseModel):
    status: str
    import_id: str
    user_id: str
    agent_id: str
    received_count: int
    stored_raw_count: int = 0
    created_event_summary_count: int = 0
    created_user_preference_count: int = 0
    raw_memory_ids: list[str] = Field(default_factory=list)
    event_summaries: list[dict[str, Any]] = Field(default_factory=list)
    user_preferences: list[dict[str, Any]] = Field(default_factory=list)
    results: list[Any] = Field(default_factory=list)


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
