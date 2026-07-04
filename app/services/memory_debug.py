import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.core.config import Settings


class MemoryDebugService:
    def __init__(self, settings: Settings) -> None:
        self._path = Path(settings.memory_debug_log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log_chat_trace(self, trace: dict[str, Any]) -> None:
        self._append({"event": "chat_trace", "created_epoch": int(time.time()), **trace})

    def log_memory_write(self, record: dict[str, Any]) -> None:
        self._append({"event": "memory_write", "created_epoch": int(time.time()), **record})

    def log_agent_trace(self, trace: dict[str, Any]) -> None:
        self._append({"event": "agent_trace", "created_epoch": int(time.time()), **trace})

    def log_memory_lifecycle(self, memory_id: str, lifecycle_event: dict[str, Any]) -> None:
        self._append(
            {
                "event": "memory_lifecycle",
                "created_epoch": int(time.time()),
                "memory_id": str(memory_id),
                "lifecycle_event": lifecycle_event,
            }
        )

    def log_memory_ranking(self, record: dict[str, Any]) -> None:
        self._append({"event": "memory_ranking", "created_epoch": int(time.time()), **record})

    def log_event_trigger(self, record: dict[str, Any]) -> None:
        self._append({"event": "event_trigger", "created_epoch": int(time.time()), **record})

    def get_prompt_trace(self, request_id: str) -> dict[str, Any] | None:
        for record in reversed(self._read_all()):
            if record.get("event") == "chat_trace" and record.get("request_id") == request_id:
                return {
                    "request_id": request_id,
                    "trace_id": record.get("trace_id"),
                    "prompt_context": record.get("prompt_context", {}),
                    "final_prompt": record.get("final_prompt", []),
                    "retrieved_memories": record.get("retrieved_memories", []),
                    "filtered_memories": record.get("filtered_memories", []),
                }
        return None

    def get_agent_trace(self, trace_id: str) -> dict[str, Any] | None:
        for record in reversed(self._read_all()):
            if record.get("event") == "agent_trace" and record.get("trace_id") == trace_id:
                return record
        return None

    def memory_lifecycle(self, memory_id: str) -> list[dict[str, Any]]:
        lifecycle = []
        for record in self._read_all():
            if record.get("event") == "memory_lifecycle" and str(record.get("memory_id")) == str(memory_id):
                lifecycle.append(record.get("lifecycle_event") or {})
        return lifecycle

    def latest_memory_ranking(self, user_id: str | None = None) -> dict[str, Any] | None:
        for record in reversed(self._read_all()):
            if record.get("event") != "memory_ranking":
                continue
            if user_id is None or record.get("user_id") == user_id:
                return record
        return None

    def events_for_user(self, user_id: str) -> list[dict[str, Any]]:
        return [
            record
            for record in self._read_all()
            if record.get("event") == "event_trigger" and record.get("user_id") == user_id
        ]

    def stats(self) -> dict[str, Any]:
        records = self._read_all()
        chat_traces = [record for record in records if record.get("event") == "chat_trace"]
        writes = [record for record in records if record.get("event") == "memory_write"]
        accepted_writes = [record for record in writes if record.get("outcome") == "stored"]
        rejected_writes = [record for record in writes if record.get("outcome") == "rejected"]

        retrieval_times = [
            float(record.get("latency", {}).get("retrieval_ms", 0))
            for record in chat_traces
            if record.get("latency")
        ]
        intents = Counter(
            (record.get("intent") or {}).get("intent_type", "unknown")
            for record in chat_traces
        )
        growth = defaultdict(int)
        for record in accepted_writes:
            day = time.strftime("%Y-%m-%d", time.localtime(record.get("created_epoch", 0)))
            growth[day] += 1

        total_writes = len(accepted_writes) + len(rejected_writes)
        rejection_rate = (len(rejected_writes) / total_writes) if total_writes else 0.0
        average_retrieval_ms = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0.0

        return {
            "average_retrieval_time_ms": round(average_retrieval_ms, 3),
            "memory_write_count": len(accepted_writes),
            "memory_rejection_rate": round(rejection_rate, 4),
            "most_common_intents": intents.most_common(),
            "memory_growth_over_time": dict(sorted(growth.items())),
            "chat_trace_count": len(chat_traces),
            "memory_write_event_count": len(writes),
        }

    def recalled_counts(self) -> Counter:
        counts = Counter()
        for record in self._read_all():
            if record.get("event") != "chat_trace":
                continue
            for memory in record.get("retrieved_memories", []):
                memory_id = memory.get("memory_id")
                if memory_id:
                    counts[str(memory_id)] += 1
        return counts

    def _append(self, record: dict[str, Any]) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _read_all(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        records = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records
