import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.models.memory import (
    MemoryNamespaceKind,
    MemoryType,
    UnifiedMemory,
    UnifiedMemoryMetadata,
    namespace_kind_for,
    normalize_memory_type,
    resolve_memory_namespace,
)
from app.schemas import MemorySummary, SleepInput, StructuredMemory
from app.services.memory_evolution_engine import MemoryEvolutionEngine
from app.services.memory_router import MemoryRouteDecision, MemoryRouteInput, MemoryRouter


class MemoryService:
    _IMPORTANCE_WEIGHTS = {
        "low": 0.2,
        "medium": 0.5,
        "high": 0.9,
    }
    _SECONDS_PER_DAY = 24 * 60 * 60

    def __init__(self, settings: Settings) -> None:
        os.environ.setdefault("MEM0_TELEMETRY", "False")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from mem0 import Memory

        self._settings = settings
        self._client = Memory.from_config(self._build_config(settings))
        self._router = MemoryRouter()
        self._evolution_engine = MemoryEvolutionEngine(settings)

    def add_structured_memory(
        self,
        memory: StructuredMemory,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Any:
        unified_memory = UnifiedMemory.model_validate(memory.model_dump())
        write_decision = self._write_decision(extra_metadata)
        memory_type = normalize_memory_type(write_decision.get("type") or unified_memory.type)
        agent_id = unified_memory.agent_id or self._extra_agent_id(extra_metadata)
        namespace = write_decision.get("namespace")
        namespace_kind = self._namespace_kind(namespace, memory_type, agent_id)
        if namespace_kind != "agent":
            agent_id = None
        importance = write_decision.get("importance", unified_memory.metadata.importance)
        llm_tag = {
            **unified_memory.metadata.model_dump(mode="json"),
            "id": unified_memory.id,
            "type": memory_type,
            "importance": importance,
        }
        route_input = MemoryRouteInput(
            user_id=unified_memory.user_id,
            message=unified_memory.content,
            agent_id=agent_id,
            should_store=bool(write_decision.get("should_store", True)),
            namespace=namespace_kind,
            type=memory_type,
            llm_tag=llm_tag,
        )
        return self._route_and_store(route_input, extra_metadata, infer=True)

    def _route_and_store(
        self,
        route_input: MemoryRouteInput,
        extra_metadata: dict[str, Any] | None = None,
        infer: bool = True,
    ) -> Any:
        decision = self._router.route(route_input)
        if not decision.should_store:
            return {
                "stored": False,
                "route": decision.model_dump(mode="json"),
            }
        return self._store_routed_memory(decision, extra_metadata, infer)

    def _store_routed_memory(
        self,
        decision: MemoryRouteDecision,
        extra_metadata: dict[str, Any] | None = None,
        infer: bool = True,
    ) -> Any:
        if not decision.should_store or decision.normalized_memory is None:
            raise ValueError("MemoryRouteDecision must allow storage and include normalized_memory.")
        memory = decision.normalized_memory
        importance_score = memory.metadata.importance
        enriched_metadata = MemoryEvolutionEngine.normalize_static({
            **memory.to_mem0_metadata(extra_metadata),
            "importance_score": importance_score,
            "retrieval_count": 0,
            "status": "active",
            "archived": False,
            "event_boost": 0.0,
            "route_reason": decision.reason,
            "logged_epoch": int(time.time()),
        })
        messages = [{"role": "user", "content": memory.content}]
        return self._client.add(messages, user_id=memory.user_id, metadata=enriched_metadata, infer=infer)

    def add_sleep_memory(self, sleep: SleepInput) -> dict[str, Any]:
        content = self._format_sleep_content(sleep)
        sleep_duration = float(sleep.sleep_duration or 0.0)
        llm_tag = {
            "type": "sleep",
            "importance": 0.6,
            "decay": 0.0,
            "feedback_weight": 0.0,
            "topic": "sleep",
            "emotion": "neutral",
            "timestamp": sleep.sleep_end.isoformat(),
        }
        route_input = MemoryRouteInput(
            user_id=sleep.user_id,
            agent_id=None,
            message=content,
            should_store=True,
            namespace="user",
            type="sleep",
            llm_tag=llm_tag,
        )
        decision = self._router.route(route_input)
        metadata = {
            "source": sleep.source,
            "role": "system",
            "user_name": sleep.user_name,
            "agent_display_id": sleep.agent_id,
            "agent_name": sleep.agent_name,
            "subject_role": "user",
            "subject_id": sleep.user_id,
            "subject_name": sleep.user_name,
            "source_agent_id": sleep.agent_id,
            "source_agent_name": sleep.agent_name,
            "sleep_start": sleep.sleep_start.isoformat(),
            "sleep_end": sleep.sleep_end.isoformat(),
            "sleep_duration": sleep_duration,
            "deep_sleep_duration": sleep.deep_sleep_duration,
            "awake_count": sleep.awake_count,
            "rem_sleep_duration": sleep.rem_sleep_duration,
            "decay_profile": "low",
            "ingestion": "sleep_api",
        }
        result = self._store_routed_memory(decision, metadata, infer=False)
        return {
            "memory": decision.normalized_memory,
            "result": result,
        }

    @staticmethod
    def _extra_agent_id(extra_metadata: dict[str, Any] | None = None) -> str | None:
        if not extra_metadata:
            return None
        agent_id = extra_metadata.get("agent_id")
        return str(agent_id) if agent_id else None

    @staticmethod
    def _write_decision(extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if not extra_metadata:
            return {}
        decision = extra_metadata.get("memory_decision")
        return decision if isinstance(decision, dict) else {}

    @staticmethod
    def _namespace_kind(
        namespace: Any,
        memory_type: MemoryType,
        agent_id: str | None,
    ) -> MemoryNamespaceKind:
        if namespace in {"user", "agent", "summary"}:
            return namespace
        return namespace_kind_for(memory_type, agent_id)

    def search(self, user_id: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
        active_limit = max(limit * 4, 20)
        summary_limit = max(min(limit * 2, 20), 5)

        active_result = self._client.search(
            query=query,
            top_k=active_limit,
            filters={
                "user_id": user_id,
                "namespace": resolve_memory_namespace(user_id, "chat"),
                "type": {"ne": "summary"},
                "NOT": [{"archived": True}],
            },
        )
        summary_result = self._client.search(
            query=query,
            top_k=summary_limit,
            filters={
                "user_id": user_id,
                "namespace": resolve_memory_namespace(user_id, "summary"),
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
        effective_filters = self._with_default_namespaces(user_id, filters)
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
                "namespace": resolve_memory_namespace(user_id, "chat"),
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
                "namespace": resolve_memory_namespace(user_id, "chat"),
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
                "namespace": resolve_memory_namespace(user_id, "summary"),
                "type": "summary",
            },
            top_k=limit,
        )
        return self._normalize_results(result)

    def get_agent_memories(self, user_id: str, agent_id: str, limit: int = 200) -> list[dict[str, Any]]:
        result = self._client.get_all(
            filters={
                "user_id": user_id,
                "agent_id": agent_id,
                "namespace": resolve_memory_namespace(user_id, "chat", agent_id),
                "NOT": [{"archived": True}],
            },
            top_k=limit,
        )
        return self._normalize_results(result)

    def get_all_memories(self, user_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        result = self._client.get_all(
            filters=self._with_default_namespaces(user_id),
            top_k=limit,
        )
        return self._normalize_results(result)

    def update_memory_metadata(self, memory_id: str, metadata: dict[str, Any]) -> Any:
        return self._client.update(memory_id=memory_id, metadata=self._canonicalize_metadata(metadata))

    def add_summary_memory(self, summary: MemorySummary) -> Any:
        content = self._format_summary_content(summary)
        llm_tag = {
            "emotion": "neutral",
            "type": "summary",
            "importance": 0.9,
            "decay": 0.0,
            "feedback_weight": 0.0,
            "topic": "summary",
            "timestamp": summary.time_range.end,
        }
        route_input = MemoryRouteInput(
            user_id=summary.user_id,
            message=content,
            should_store=True,
            namespace="summary",
            type="summary",
            llm_tag=llm_tag,
        )
        metadata = {
            "time_range": summary.time_range.model_dump(),
            "summary_object": summary.model_dump(),
            "source": "summary",
            "role": "system",
        }
        return self._route_and_store(route_input, metadata, infer=False)

    def add_agent_interaction_summary(
        self,
        user_id: str,
        agent_id: str,
        content: str,
        metadata: dict[str, Any],
    ) -> Any:
        llm_tag = {
            "emotion": metadata.get("emotion", "neutral"),
            "type": "event",
            "importance": metadata.get("importance", 0.8),
            "decay": 0.0,
            "feedback_weight": 0.0,
            "topic": metadata.get("topic", "agent_interaction"),
            "timestamp": metadata.get("timestamp") or datetime.utcnow().isoformat() + "Z",
        }
        route_input = MemoryRouteInput(
            user_id=user_id,
            agent_id=agent_id,
            message=content,
            should_store=True,
            namespace="agent",
            type="event",
            llm_tag=llm_tag,
        )
        summary_metadata = {
            **metadata,
            "source": "agent_interaction_summary",
            "role": "system",
            "summary_kind": "agent_interaction_summary",
            "agent_id": agent_id,
        }
        return self._route_and_store(route_input, summary_metadata, infer=False)

    def add_imported_chat_message(
        self,
        user_id: str,
        agent_id: str,
        content: str,
        metadata: dict[str, Any],
    ) -> Any:
        llm_tag = {
            "emotion": metadata.get("emotion", "neutral"),
            "type": "chat",
            "importance": metadata.get("importance", 0.35),
            "decay": 0.0,
            "feedback_weight": 0.0,
            "topic": metadata.get("topic", "imported_chat"),
            "timestamp": metadata.get("timestamp") or datetime.utcnow().isoformat() + "Z",
        }
        route_input = MemoryRouteInput(
            user_id=user_id,
            agent_id=agent_id,
            message=content,
            should_store=True,
            namespace="agent",
            type="chat",
            llm_tag=llm_tag,
        )
        import_metadata = {
            **metadata,
            "source": metadata.get("source", "local_chat_import"),
            "role": metadata.get("speaker_role", "user"),
            "agent_id": agent_id,
            "imported": True,
        }
        return self._route_and_store(route_input, import_metadata, infer=False)

    def add_imported_user_preference(
        self,
        user_id: str,
        content: str,
        metadata: dict[str, Any],
    ) -> Any:
        llm_tag = {
            "emotion": metadata.get("emotion", "neutral"),
            "type": "preference",
            "importance": metadata.get("importance", 0.75),
            "decay": 0.0,
            "feedback_weight": 0.0,
            "topic": metadata.get("topic", "user_preference"),
            "timestamp": metadata.get("timestamp") or datetime.utcnow().isoformat() + "Z",
        }
        route_input = MemoryRouteInput(
            user_id=user_id,
            message=content,
            should_store=True,
            namespace="user",
            type="preference",
            llm_tag=llm_tag,
        )
        preference_metadata = {
            **metadata,
            "source": metadata.get("source", "local_chat_import_summary"),
            "role": "system",
            "summary_kind": "imported_user_preference",
            "imported": True,
        }
        return self._route_and_store(route_input, preference_metadata, infer=False)

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
            self.update_memory_metadata(str(memory_id), metadata)
            archived_ids.append(str(memory_id))
        return archived_ids

    @staticmethod
    def _canonicalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        canonical = dict(metadata or {})
        user_id = str(canonical.get("user_id") or "")
        if not user_id:
            raise ValueError("Memory metadata must include user_id.")
        memory_type = normalize_memory_type(canonical.get("type"))
        agent_id = canonical.get("agent_id")
        agent_id = str(agent_id) if agent_id else None
        if "importance" not in canonical and "importance_score" in canonical:
            canonical["importance"] = canonical["importance_score"]
        metadata_object = UnifiedMemoryMetadata.model_validate(canonical)
        canonical.update(
            {
                "user_id": user_id,
                "agent_id": agent_id,
                "namespace": resolve_memory_namespace(user_id, memory_type, agent_id),
                "namespace_kind": namespace_kind_for(memory_type, agent_id),
                "type": memory_type,
                "timestamp": metadata_object.timestamp.isoformat(),
                "importance": metadata_object.importance,
                "decay": metadata_object.decay,
                "feedback_weight": metadata_object.feedback_weight,
            }
        )
        return MemoryEvolutionEngine.normalize_static(canonical)

    @staticmethod
    def _with_default_namespaces(user_id: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        effective_filters = {
            "user_id": user_id,
            "namespace": {
                "in": [
                    resolve_memory_namespace(user_id, "chat"),
                    resolve_memory_namespace(user_id, "summary"),
                ]
            },
        }
        effective_filters.update(filters or {})
        return effective_filters

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
        return self._evolution_engine.rank_memories(memories, limit, now_epoch)

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
            f"每日总结：{summary.daily_summary}\n"
            f"情绪趋势：{summary.emotional_trend}\n"
            f"关键事件：\n{key_events}\n"
            f"新增用户偏好：\n{preferences}"
        )

    @staticmethod
    def _format_sleep_content(sleep: SleepInput) -> str:
        lines = [
            f"Sleep from {sleep.sleep_start.strftime('%H:%M')} to {sleep.sleep_end.strftime('%H:%M')}.",
            f"Duration: {float(sleep.sleep_duration or 0.0):.2f}h",
        ]
        if sleep.deep_sleep_duration is not None:
            lines.append(f"Deep sleep: {sleep.deep_sleep_duration:.2f}h")
        if sleep.rem_sleep_duration is not None:
            lines.append(f"REM sleep: {sleep.rem_sleep_duration:.2f}h")
        if sleep.awake_count is not None:
            lines.append(f"Awakenings: {sleep.awake_count}")
        lines.append(f"Source: {sleep.source}")
        return "\n".join(lines)

    @staticmethod
    def _build_config(settings: Settings) -> dict[str, Any]:
        embedder_config: dict[str, Any] = {
            "model": settings.mem0_embedder_model,
            "embedding_dims": settings.mem0_embedder_dims,
        }
        if settings.mem0_embedder_provider == "huggingface":
            model_path = Path(settings.mem0_embedder_model)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Local embedding model directory not found: {model_path}. "
                    "Automatic HuggingFace downloads are disabled."
                )
            embedder_config["model_kwargs"] = {"local_files_only": True}
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
