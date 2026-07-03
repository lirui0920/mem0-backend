import os
import time
from datetime import datetime
from typing import Any

from app.core.config import Settings
from app.schemas import MemorySummary, StructuredMemory


class MemoryService:
    _IMPORTANCE_WEIGHTS = {
        "low": 0.2,
        "medium": 0.5,
        "high": 0.9,
    }
    _SECONDS_PER_DAY = 24 * 60 * 60

    def __init__(self, settings: Settings) -> None:
        os.environ.setdefault("MEM0_TELEMETRY", "False")
        from mem0 import Memory

        self._settings = settings
        self._client = Memory.from_config(self._build_config(settings))

    def add_structured_memory(
        self,
        memory: StructuredMemory,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Any:
        importance_score = self._IMPORTANCE_WEIGHTS.get(memory.metadata.importance, 0.5)
        metadata = {
            **memory.metadata.model_dump(),
            "importance_score": importance_score,
            "retrieval_count": 0,
            "status": "active",
            "archived": False,
            "memory_object": memory.model_dump(),
            **(extra_metadata or {}),
        }
        return self._add_memory(
            user_id=memory.user_id,
            content=memory.content,
            metadata=metadata,
        )

    def _add_memory(
        self,
        user_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
    ) -> Any:
        enriched_metadata = {
            **(metadata or {}),
            "user_id": user_id,
            "logged_epoch": int(time.time()),
        }
        messages = [{"role": "user", "content": content}]
        return self._client.add(messages, user_id=user_id, metadata=enriched_metadata, infer=infer)

    def search(self, user_id: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
        active_limit = max(limit * 4, 20)
        summary_limit = max(min(limit * 2, 20), 5)

        active_result = self._client.search(
            query=query,
            top_k=active_limit,
            filters={
                "user_id": user_id,
                "type": {"ne": "summary"},
                "NOT": [{"archived": True}],
            },
        )
        summary_result = self._client.search(
            query=query,
            top_k=summary_limit,
            filters={
                "user_id": user_id,
                "type": "summary",
            },
        )
        candidates = self._dedupe_by_id(
            [
                *self._normalize_results(summary_result),
                *self._normalize_results(active_result),
            ]
        )
        return self.apply_decay_ranking(candidates, limit)

    def search_candidates(
        self,
        user_id: str,
        query: str,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        effective_filters = {"user_id": user_id, **(filters or {})}
        result = self._client.search(
            query=query,
            top_k=limit,
            filters=effective_filters,
        )
        return self._normalize_results(result)

    def memories_since(self, user_id: str, since_epoch: int, limit: int = 100) -> list[dict[str, Any]]:
        result = self._client.search(
            query="recent user events chat preferences tasks emotions plans",
            top_k=limit,
            filters={
                "user_id": user_id,
                "type": {"ne": "summary"},
                "NOT": [{"archived": True}],
                "logged_epoch": {
                    "gte": since_epoch,
                    "lte": int(time.time()),
                },
            },
        )
        return self._normalize_results(result)

    def get_unarchived_memories(self, user_id: str, limit: int = 500) -> list[dict[str, Any]]:
        result = self._client.get_all(
            filters={
                "user_id": user_id,
                "type": {"ne": "summary"},
                "NOT": [{"archived": True}],
            },
            top_k=limit,
        )
        return self._normalize_results(result)

    def get_summary_memories(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        result = self._client.get_all(
            filters={
                "user_id": user_id,
                "type": "summary",
            },
            top_k=limit,
        )
        return self._normalize_results(result)

    def get_all_memories(self, user_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        result = self._client.get_all(
            filters={"user_id": user_id},
            top_k=limit,
        )
        return self._normalize_results(result)

    def update_memory_metadata(self, memory_id: str, metadata: dict[str, Any]) -> Any:
        return self._client.update(memory_id=memory_id, metadata=metadata)

    def add_summary_memory(self, summary: MemorySummary) -> Any:
        content = self._format_summary_content(summary)
        metadata = {
            "emotion": "neutral",
            "type": "summary",
            "importance": "high",
            "importance_score": 0.9,
            "retrieval_count": 0,
            "topic": "summary",
            "timestamp": summary.time_range.end,
            "time_range": summary.time_range.model_dump(),
            "summary_object": summary.model_dump(),
            "archived": False,
            "status": "active",
            "source": "summary",
            "role": "system",
        }
        return self._add_memory(summary.user_id, content, metadata, infer=False)

    def archive_memories(self, memories: list[dict[str, Any]], summary_time_range: dict[str, Any]) -> list[str]:
        archived_ids = []
        for memory in memories:
            memory_id = memory.get("id")
            if not memory_id:
                continue
            metadata = memory.get("metadata") or {}
            metadata.update(
                {
                    "archived": True,
                    "archived_at": int(time.time()),
                    "summary_time_range": summary_time_range,
                }
            )
            self._client.update(memory_id=memory_id, metadata=metadata)
            archived_ids.append(str(memory_id))
        return archived_ids

    def should_summarize(self, user_id: str, now_epoch: int | None = None, limit: int = 500) -> tuple[bool, str, list[dict[str, Any]]]:
        now_epoch = now_epoch or int(time.time())
        memories = self.get_unarchived_memories(user_id, limit=limit)
        if len(memories) >= self._settings.summary_memory_batch_size:
            return True, "memory_count_threshold", memories

        summaries = self.get_summary_memories(user_id, limit=50)
        last_summary_epoch = self._latest_summary_epoch(summaries)
        if memories and now_epoch - last_summary_epoch >= self._settings.summary_interval_seconds:
            return True, "time_threshold", memories

        return False, "threshold_not_met", memories

    @staticmethod
    def _normalize_results(result: Any) -> list[dict[str, Any]]:
        if isinstance(result, dict):
            candidates = result.get("results") or result.get("memories") or []
        else:
            candidates = result or []

        normalized = []
        for item in candidates:
            if isinstance(item, dict):
                normalized.append(item)
            else:
                normalized.append({"memory": str(item)})
        return normalized

    @staticmethod
    def _dedupe_by_id(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen = set()
        deduped = []
        for memory in memories:
            memory_id = memory.get("id")
            key = memory_id or memory.get("memory") or str(memory)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(memory)
        return deduped

    def apply_decay_ranking(
        self,
        memories: list[dict[str, Any]],
        limit: int,
        now_epoch: int | None = None,
    ) -> list[dict[str, Any]]:
        now_epoch = now_epoch or int(time.time())
        scored = []
        for memory in memories:
            ranked = dict(memory)
            metadata = ranked.get("metadata") or {}
            relevance_score = self._relevance_score(ranked)
            importance_weight = self._importance_weight(metadata)
            decay_penalty = self._time_decay_penalty(metadata, ranked, now_epoch)
            if metadata.get("type") == "summary":
                importance_weight += self._settings.summary_retention_boost

            final_score = importance_weight + relevance_score - decay_penalty
            ranked["decay_score"] = final_score
            ranked["score_components"] = {
                "importance_weight": importance_weight,
                "relevance_score": relevance_score,
                "time_decay_penalty": decay_penalty,
            }
            scored.append(ranked)

        scored.sort(key=lambda item: item.get("decay_score", 0.0), reverse=True)
        return scored[:limit]

    def _importance_weight(self, metadata: dict[str, Any]) -> float:
        importance = metadata.get("importance", "medium")
        return self._IMPORTANCE_WEIGHTS.get(str(importance), self._IMPORTANCE_WEIGHTS["medium"])

    def _time_decay_penalty(
        self,
        metadata: dict[str, Any],
        memory: dict[str, Any],
        now_epoch: int,
    ) -> float:
        memory_epoch = self._memory_epoch(metadata, memory)
        if memory_epoch is None:
            return 0.0

        age_days = max(0, (now_epoch - memory_epoch) / self._SECONDS_PER_DAY)
        if age_days < 7:
            return 0.0
        if age_days <= 30:
            return self._settings.decay_medium_penalty
        return self._settings.decay_strong_penalty

    @staticmethod
    def _relevance_score(memory: dict[str, Any]) -> float:
        score = memory.get("score", 0.0)
        if isinstance(score, int | float):
            return float(score)
        return 0.0

    @staticmethod
    def _memory_epoch(metadata: dict[str, Any], memory: dict[str, Any]) -> int | None:
        candidates = [
            metadata.get("logged_epoch"),
            metadata.get("timestamp"),
            memory.get("created_at"),
            metadata.get("created_at"),
        ]
        for candidate in candidates:
            if isinstance(candidate, int):
                return candidate
            if isinstance(candidate, float):
                return int(candidate)
            if isinstance(candidate, str):
                try:
                    return int(datetime.fromisoformat(candidate.replace("Z", "+00:00")).timestamp())
                except ValueError:
                    continue
        return None

    @staticmethod
    def _latest_summary_epoch(summaries: list[dict[str, Any]]) -> int:
        latest = 0
        for summary in summaries:
            metadata = summary.get("metadata") or {}
            time_range = metadata.get("time_range") or {}
            end_epoch = time_range.get("end_epoch") or metadata.get("logged_epoch")
            if isinstance(end_epoch, int):
                latest = max(latest, end_epoch)
        return latest

    @staticmethod
    def _format_summary_content(summary: MemorySummary) -> str:
        key_events = "\n".join(f"- {event}" for event in summary.key_events) or "- None"
        preferences = "\n".join(f"- {preference}" for preference in summary.new_user_preferences) or "- None"
        return (
            f"Daily summary: {summary.daily_summary}\n"
            f"Emotional trend: {summary.emotional_trend}\n"
            f"Key events:\n{key_events}\n"
            f"New user preferences:\n{preferences}"
        )

    @staticmethod
    def _build_config(settings: Settings) -> dict[str, Any]:
        embedder_config: dict[str, Any] = {
            "model": settings.mem0_embedder_model,
            "embedding_dims": settings.mem0_embedder_dims,
        }
        if settings.mem0_embedder_provider == "openai":
            embedder_config.update(
                {
                    "api_key": settings.llm_api_key,
                    "openai_base_url": settings.llm_base_url,
                }
            )

        return {
            "llm": {
                "provider": settings.mem0_llm_provider,
                "config": {
                    "api_key": settings.llm_api_key,
                    "openai_base_url": settings.llm_base_url,
                    "model": settings.mem0_llm_model,
                    "temperature": 0.1,
                },
            },
            "embedder": {
                "provider": settings.mem0_embedder_provider,
                "config": embedder_config,
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": settings.mem0_collection,
                    "path": str(settings.qdrant_path),
                    "embedding_model_dims": settings.mem0_embedder_dims,
                },
            },
            "history_db_path": str(settings.mem0_history_db_path),
        }
