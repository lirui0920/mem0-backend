import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.services.llm_service import LLMService
from app.services.memory_evolution_engine import MemoryEvolutionEngine
from app.services.memory_service import MemoryService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchestratorEvent:
    type: str
    subtype: str
    user_id: str
    timestamp: str
    confidence: float
    priority: int
    context: dict[str, Any]


class MemoryOrchestrator:
    """Minimal background memory lifecycle scheduler."""

    _MAX_BUFFER_MESSAGES = 50
    _SUMMARY_TURN_THRESHOLD = 15
    _SUMMARY_TOKEN_THRESHOLD = 4000
    _EMOTION_SCORE_THRESHOLD = 0.75
    _EVENT_MEMORY_LIFECYCLE = "MEMORY_LIFECYCLE_EVENT"
    _EVENT_SUMMARY = "SUMMARY_EVENT"
    _EVENT_PROACTIVE = "PROACTIVE_EVENT"
    _SUBTYPE_CONVERSATIONAL_DENSITY = "conversational_density_high"
    _SUBTYPE_EMOTIONAL_SPIKE = "emotional_spike"
    _SUBTYPE_HEALTH_SIGNAL = "health_signal"
    _SUBTYPE_PROACTIVE_MESSAGE_TRIGGER = "proactive_message_trigger"
    _HEALTH_SIGNALS = ("sleep", "tired", "insomnia", "headache", "heart rate")
    _STARVATION_WAIT_SECONDS = 300
    _STARVATION_PRIORITY_BOOST = 20

    def __init__(self, settings: Settings) -> None:
        self._state_path = Path(settings.memory_debug_log_path).parent / "memory_orchestrator_state.json"
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._states = self._load_states()
        self._event_queue: list[OrchestratorEvent] = []
        self._evolution_engine = MemoryEvolutionEngine(settings)
        self._lock = threading.RLock()
        self._queue_condition = threading.Condition(self._lock)
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def update_state(
        self,
        user_id: str,
        message: str,
        intent: Any,
        response: str,
        memories: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        now = self._now()
        with self._lock:
            state = self._states.get(user_id) or self._new_state(user_id)
            token_estimate = self._estimate_tokens(message) + self._estimate_tokens(response)
            emotion = self._intent_value(intent, "emotion", "neutral")
            emotion_score = self._emotion_score(emotion)
            memory_ids = self._memory_ids(memories or [])
            state["turn_count"] = int(state.get("turn_count", 0)) + 1
            state["token_accumulated"] = int(state.get("token_accumulated", 0)) + token_estimate
            state["emotion_score_accumulated"] = round(
                float(state.get("emotion_score_accumulated", 0.0)) + emotion_score,
                4,
            )
            state["last_activity_time"] = now
            state.setdefault("buffer_messages", []).append(
                {
                    "user_message": message,
                    "assistant_response": response,
                    "timestamp": now,
                    "token_estimate": token_estimate,
                    "emotion": emotion,
                    "emotion_score": emotion_score,
                    "intent_type": self._intent_value(intent, "intent_type", "unknown"),
                }
            )
            state["buffer_messages"] = state["buffer_messages"][-self._MAX_BUFFER_MESSAGES :]
            if memory_ids:
                state["recent_memory_ids"] = memory_ids
            self._states[user_id] = state
            self._save_states_locked()
            return dict(state)

    def get_state(self, user_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._states.get(user_id) or self._new_state(user_id)
            self._states[user_id] = state
            return dict(state)

    def start(
        self,
        memory_service: MemoryService,
        llm_service: LLMService,
        interval_seconds: int = 60,
    ) -> None:
        with self._lock:
            if self._worker_thread and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._run_worker,
                args=(memory_service, llm_service, interval_seconds),
                name="memory-orchestrator",
                daemon=True,
            )
            self._worker_thread.start()
            logger.info("MemoryOrchestrator worker started interval_seconds=%s", interval_seconds)

    def stop(self) -> None:
        self._stop_event.set()
        with self._queue_condition:
            self._queue_condition.notify_all()
        thread = self._worker_thread
        if thread and thread.is_alive():
            thread.join(timeout=5)

    def _run_worker(
        self,
        memory_service: MemoryService,
        llm_service: LLMService,
        interval_seconds: int,
    ) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_once(memory_service, llm_service)
            except Exception:
                logger.exception("MemoryOrchestrator worker iteration failed")
            with self._queue_condition:
                if not self._event_queue and not self._stop_event.is_set():
                    self._queue_condition.wait(timeout=interval_seconds)

    def run_once(self, memory_service: MemoryService, llm_service: LLMService) -> list[dict[str, Any]]:
        events = []
        with self._lock:
            while self._event_queue:
                events.append(self._pop_highest_priority_event_locked())

        results = []
        for event in events:
            results.append(self._handle_event(event, memory_service, llm_service))
        return results

    def emit_event(
        self,
        event_type: str,
        subtype: str,
        user_id: str,
        confidence: float = 1.0,
        priority: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> OrchestratorEvent:
        event = OrchestratorEvent(
            type=event_type,
            subtype=subtype,
            user_id=user_id,
            timestamp=self._now(),
            confidence=confidence,
            priority=priority if priority is not None else self._priority_for_event(event_type, subtype, confidence),
            context=context or {},
        )
        with self._queue_condition:
            self._mark_event_pending_locked(event)
            self._enqueue_event_locked(event)
        logger.info("MemoryOrchestrator event emitted: %s", asdict(event))
        return event

    def _handle_event(
        self,
        event: OrchestratorEvent,
        memory_service: MemoryService,
        llm_service: LLMService,
    ) -> dict[str, Any]:
        self._mark_event_processed(event.user_id)
        feedback_result = self.apply_memory_feedback(event, memory_service)
        if event.type == self._EVENT_SUMMARY and event.subtype == self._SUBTYPE_CONVERSATIONAL_DENSITY:
            summary_result = self._trigger_summary(event.user_id, memory_service, llm_service, event)
            summary_result["feedback_result"] = feedback_result
            return summary_result
        if event.type == self._EVENT_PROACTIVE and event.subtype == self._SUBTYPE_EMOTIONAL_SPIKE:
            self._mark_emotional_flag(event.user_id)
            summary_result = self._trigger_summary(event.user_id, memory_service, llm_service, event)
            proactive_result = self._record_proactive_event(event)
            return {
                "user_id": event.user_id,
                "created": bool(summary_result.get("created")),
                "reason": "emotional_spike_handled",
                "summary_result": summary_result,
                "proactive_event": proactive_result,
                "feedback_result": feedback_result,
            }
        if event.type == self._EVENT_MEMORY_LIFECYCLE and event.subtype == self._SUBTYPE_HEALTH_SIGNAL:
            result = self._handle_health_signal(event)
            result["feedback_result"] = feedback_result
            return result
        if event.type == self._EVENT_PROACTIVE and event.subtype == self._SUBTYPE_PROACTIVE_MESSAGE_TRIGGER:
            result = self._record_proactive_event(event)
            result["feedback_result"] = feedback_result
            return result
        return {
            "user_id": event.user_id,
            "created": False,
            "reason": "unknown_event_type",
            "event_type": event.type,
            "event_subtype": event.subtype,
            "feedback_result": feedback_result,
        }

    def apply_memory_feedback(
        self,
        event: OrchestratorEvent,
        memory_service: MemoryService,
    ) -> dict[str, Any]:
        memory_ids = self._event_memory_ids(event)
        if not memory_ids:
            return {"updated_count": 0, "reason": "no_related_memories"}

        memories_by_id = self._memories_by_id(event.user_id, memory_ids, memory_service)
        updated = []
        for memory_id in memory_ids:
            memory = memories_by_id.get(memory_id)
            if not memory:
                continue
            metadata = dict(memory.get("metadata") or {})
            self._apply_feedback_metadata(event, metadata)
            memory_service.update_memory_metadata(memory_id, metadata)
            updated.append(
                {
                    "memory_id": memory_id,
                    "feedback_weight": metadata.get("feedback_weight", 0.0),
                    "importance_score": metadata.get("importance_score"),
                    "reinforcement_count": metadata.get("reinforcement_count", 0),
                    "last_event_type": metadata.get("last_event_type"),
                }
            )

        return {
            "updated_count": len(updated),
            "memory_ids": [item["memory_id"] for item in updated],
            "updates": updated,
        }

    def _apply_feedback_metadata(self, event: OrchestratorEvent, metadata: dict[str, Any]) -> None:
        event_cycle_id = f"{event.subtype}:{event.timestamp}"
        if metadata.get("last_event_cycle_id") == event_cycle_id:
            metadata["event_reinforcement_skipped_reason"] = "event_cycle_already_applied"
            return

        if event.subtype == self._SUBTYPE_EMOTIONAL_SPIKE:
            base_delta = 0.05
            event_boost_delta = 0.1
        elif event.subtype == self._SUBTYPE_HEALTH_SIGNAL:
            base_delta = 0.04
            event_boost_delta = 0.08
            metadata["health_related"] = True
        elif event.subtype == self._SUBTYPE_CONVERSATIONAL_DENSITY:
            base_delta = -0.02
            event_boost_delta = 0.0
        else:
            base_delta = 0.0
            event_boost_delta = 0.0

        normalized, applied = self._evolution_engine.reinforce_metadata(
            metadata,
            session_id=f"event:{event_cycle_id}",
            base_delta=base_delta,
            event_boost_delta=event_boost_delta,
        )
        metadata.update(normalized)
        metadata.update(
            {
                "last_event_type": event.subtype,
                "last_event_timestamp": event.timestamp,
                "last_event_cycle_id": event_cycle_id,
                "event_reinforcement_applied": applied,
            }
        )

    def _trigger_summary(
        self,
        user_id: str,
        memory_service: MemoryService,
        llm_service: LLMService,
        event: OrchestratorEvent | None = None,
    ) -> dict[str, Any]:
        if not self._acquire_summary_lock(user_id):
            return {"user_id": user_id, "created": False, "reason": "summary_locked"}

        try:
            should_run, policy_reason, memories = memory_service.should_summarize(user_id, None, 500)
            if not memories:
                result = {"user_id": user_id, "created": False, "reason": "no_unarchived_memories"}
                self._reset_after_summary(user_id, result)
                return result

            start_epoch, end_epoch = self._memory_time_range(memories)
            summary = llm_service.summarize_memories(user_id, memories, start_epoch, end_epoch)
            add_result = memory_service.add_summary_memory(summary)
            archived_ids = memory_service.archive_memories(memories, summary.time_range.model_dump())
            result = {
                "user_id": user_id,
                "created": True,
                "reason": event.subtype if event else ("orchestrator_turn_count" if not should_run else policy_reason),
                "policy_reason": policy_reason,
                "event": asdict(event) if event else None,
                "source_memory_count": len(memories),
                "archived_memory_ids": archived_ids,
                "result": add_result,
            }
            self._reset_after_summary(user_id, result)
            logger.info(
                "MemoryOrchestrator summary created user_id=%s memories=%s archived=%s",
                user_id,
                len(memories),
                len(archived_ids),
            )
            return result
        except Exception as exc:
            self._release_summary_lock(user_id)
            logger.exception("MemoryOrchestrator summary failed user_id=%s", user_id)
            return {"user_id": user_id, "created": False, "reason": "summary_error", "error": str(exc)}

    def _record_proactive_event(self, event: OrchestratorEvent) -> dict[str, Any]:
        record = {
            "type": "proactive_message_candidate",
            "user_id": event.user_id,
            "timestamp": self._now(),
            "reason": event.subtype,
            "confidence": event.confidence,
            "context": event.context,
            "status": "logged_only",
        }
        logger.info("MemoryOrchestrator proactive event candidate: %s", record)
        return record

    def _log_semantic_event(self, event: OrchestratorEvent) -> dict[str, Any]:
        result = {
            "user_id": event.user_id,
            "created": False,
            "reason": "semantic_event_logged_only",
            "event": asdict(event),
        }
        logger.info("MemoryOrchestrator semantic event logged: %s", result)
        return result

    def _handle_health_signal(self, event: OrchestratorEvent) -> dict[str, Any]:
        with self._lock:
            state = self._states.get(event.user_id) or self._new_state(event.user_id)
            previous_count = int(state.get("health_signal_count", 0))
            state["health_signal_count"] = previous_count + 1
            self._states[event.user_id] = state
            self._save_states_locked()

        result = self._log_semantic_event(event)
        result["health_signal_count"] = previous_count + 1
        result["repeated"] = previous_count > 0
        result["effective_priority"] = event.priority + (10 if previous_count > 0 else 0)
        return result

    def _mark_emotional_flag(self, user_id: str) -> None:
        with self._lock:
            state = self._states.get(user_id) or self._new_state(user_id)
            state["emotional_flag"] = True
            state["last_emotional_event_time"] = self._now()
            self._states[user_id] = state
            self._save_states_locked()

    def _mark_event_processed(self, user_id: str) -> None:
        with self._lock:
            state = self._states.get(user_id) or self._new_state(user_id)
            state["last_event_time"] = self._now()
            state["event_processed_count"] = int(state.get("event_processed_count", 0)) + 1
            self._states[user_id] = state
            self._save_states_locked()

    def _acquire_summary_lock(self, user_id: str) -> bool:
        with self._lock:
            state = self._states.get(user_id) or self._new_state(user_id)
            if state.get("summary_lock"):
                return False
            state["summary_lock"] = True
            self._states[user_id] = state
            self._save_states_locked()
            return True

    def _release_summary_lock(self, user_id: str) -> None:
        with self._lock:
            state = self._states.get(user_id) or self._new_state(user_id)
            state["summary_lock"] = False
            state["summary_event_pending"] = False
            self._states[user_id] = state
            self._save_states_locked()

    def _reset_after_summary(self, user_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            state = self._states.get(user_id) or self._new_state(user_id)
            state["turn_count"] = 0
            state["token_accumulated"] = 0
            state["last_summary_time"] = self._now()
            state["summary_lock"] = False
            state["summary_event_pending"] = False
            state["last_summary_result"] = result
            self._states[user_id] = state
            self._save_states_locked()

    def _enqueue_event_locked(self, event: OrchestratorEvent) -> None:
        self._event_queue.append(event)
        self._queue_condition.notify()
        logger.info("MemoryOrchestrator semantic event queued: %s", asdict(event))

    def _mark_event_pending_locked(self, event: OrchestratorEvent) -> None:
        state = self._states.get(event.user_id) or self._new_state(event.user_id)
        if event.type == self._EVENT_SUMMARY or (
            event.type == self._EVENT_PROACTIVE and event.subtype == self._SUBTYPE_EMOTIONAL_SPIKE
        ):
            state["summary_event_pending"] = True
        self._states[event.user_id] = state
        self._save_states_locked()

    def _pop_highest_priority_event_locked(self) -> OrchestratorEvent:
        if not self._event_queue:
            raise IndexError("event queue is empty")

        now_epoch = time.time()
        best_index = max(
            range(len(self._event_queue)),
            key=lambda index: self._effective_priority(self._event_queue[index], now_epoch),
        )
        return self._event_queue.pop(best_index)

    def _effective_priority(self, event: OrchestratorEvent, now_epoch: float | None = None) -> int:
        now_epoch = now_epoch or time.time()
        waiting_seconds = max(0.0, now_epoch - self._event_epoch(event))
        if waiting_seconds > self._STARVATION_WAIT_SECONDS:
            return event.priority + self._STARVATION_PRIORITY_BOOST
        return event.priority

    def _load_states(self) -> dict[str, dict[str, Any]]:
        if not self._state_path.exists():
            return {}
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Ignoring invalid MemoryOrchestrator state file: %s", self._state_path)
            return {}
        if not isinstance(data, dict):
            return {}
        states = {str(user_id): state for user_id, state in data.items() if isinstance(state, dict)}
        for state in states.values():
            state["summary_lock"] = False
            state["summary_event_pending"] = False
            state.setdefault("last_event_time", None)
            state.setdefault("event_processed_count", 0)
        return states

    def _save_states_locked(self) -> None:
        self._state_path.write_text(
            json.dumps(self._states, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    @classmethod
    def _new_state(cls, user_id: str) -> dict[str, Any]:
        now = cls._now()
        return {
            "user_id": user_id,
            "turn_count": 0,
            "token_accumulated": 0,
            "emotion_score_accumulated": 0.0,
            "last_summary_time": None,
            "last_activity_time": now,
            "buffer_messages": [],
            "summary_lock": False,
            "summary_event_pending": False,
            "last_event_time": None,
            "event_processed_count": 0,
            "recent_memory_ids": [],
            "last_summary_result": None,
        }

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _estimate_tokens(*texts: str) -> int:
        total_chars = sum(len(text or "") for text in texts)
        return max(1, total_chars // 4)

    @staticmethod
    def _intent_value(intent: Any, key: str, default: str) -> str:
        if isinstance(intent, dict):
            return str(intent.get(key, default))
        return str(getattr(intent, key, default))

    @staticmethod
    def _emotion_score(emotion: str) -> float:
        return {
            "happy": 0.25,
            "neutral": 0.0,
            "sad": 0.45,
            "anxious": 0.6,
            "angry": 0.7,
        }.get(emotion, 0.0)

    @classmethod
    def _priority_for_event(cls, event_type: str, subtype: str, confidence: float) -> int:
        base_priority = 10
        if subtype == cls._SUBTYPE_EMOTIONAL_SPIKE:
            base_priority = 100
        elif subtype == cls._SUBTYPE_HEALTH_SIGNAL:
            base_priority = 90
        elif event_type == cls._EVENT_SUMMARY:
            base_priority = 50
        if subtype == cls._SUBTYPE_CONVERSATIONAL_DENSITY:
            base_priority = 40
        return int(base_priority + confidence * 10)

    @staticmethod
    def _event_epoch(event: OrchestratorEvent) -> float:
        try:
            return datetime.fromisoformat(event.timestamp.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return time.time()

    @staticmethod
    def _compact(text: str, limit: int) -> str:
        cleaned = " ".join((text or "").split())
        return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."

    @staticmethod
    def _memory_ids(memories: list[dict[str, Any]]) -> list[str]:
        memory_ids = []
        for memory in memories:
            memory_id = memory.get("id")
            if memory_id:
                memory_ids.append(str(memory_id))
        return memory_ids

    def _event_memory_ids(self, event: OrchestratorEvent) -> list[str]:
        memory_ids = event.context.get("memory_ids")
        if isinstance(memory_ids, list) and memory_ids:
            return [str(memory_id) for memory_id in memory_ids if memory_id]

        with self._lock:
            state = self._states.get(event.user_id) or {}
            recent_memory_ids = state.get("recent_memory_ids") or []
        if isinstance(recent_memory_ids, list):
            return [str(memory_id) for memory_id in recent_memory_ids if memory_id]
        return []

    @staticmethod
    def _memories_by_id(
        user_id: str,
        memory_ids: list[str],
        memory_service: MemoryService,
    ) -> dict[str, dict[str, Any]]:
        target_ids = set(memory_ids)
        memories = memory_service.get_all_memories(user_id, limit=1000)
        return {
            str(memory["id"]): memory
            for memory in memories
            if memory.get("id") and str(memory["id"]) in target_ids
        }

    @staticmethod
    def _metadata_importance_score(metadata: dict[str, Any]) -> float:
        score = metadata.get("importance_score")
        if isinstance(score, int | float):
            return float(score)
        return {
            "low": 0.2,
            "medium": 0.5,
            "high": 0.9,
        }.get(str(metadata.get("importance", "medium")), 0.5)

    @staticmethod
    def _memory_time_range(memories: list[dict[str, Any]]) -> tuple[int, int]:
        epochs = []
        for memory in memories:
            metadata = memory.get("metadata") or {}
            logged_epoch = metadata.get("logged_epoch")
            if isinstance(logged_epoch, int):
                epochs.append(logged_epoch)
                continue

            created_at = memory.get("created_at") or metadata.get("created_at")
            if isinstance(created_at, str):
                try:
                    epochs.append(int(datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()))
                except ValueError:
                    continue

        now_epoch = int(time.time())
        if not epochs:
            return now_epoch, now_epoch
        return min(epochs), max(epochs)
