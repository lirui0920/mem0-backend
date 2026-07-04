import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.config import Settings

if TYPE_CHECKING:
    from app.services.memory_debug import MemoryDebugService
    from app.services.memory_service import MemoryService


class MemoryEvolutionEngine:
    """Bounded memory scoring, decay, reinforcement, and profile maintenance."""

    _IMPORTANCE_FALLBACK = {"low": 0.2, "medium": 0.5, "high": 0.8}
    _SECONDS_PER_DAY = 24 * 60 * 60
    _MIN_IMPORTANCE = 0.0
    _MAX_IMPORTANCE = 1.0
    _MIN_EFFECTIVE_IMPORTANCE = 0.05
    _MIN_FEEDBACK_WEIGHT = -0.5
    _MAX_FEEDBACK_WEIGHT = 0.5
    _MIN_EVENT_BOOST = 0.0
    _MAX_EVENT_BOOST = 0.3
    _MAX_REINFORCEMENT_PER_SESSION = 3
    _BASE_REINFORCEMENT_DELTA = 0.05
    _BASE_DECAY_RATE = 0.08

    def __init__(self, settings: Settings) -> None:
        self._profile_path = Path(settings.memory_profile_path)
        self._profile_path.parent.mkdir(parents=True, exist_ok=True)

    def score_memory(self, memory: dict[str, Any], now_epoch: int | None = None) -> dict[str, Any]:
        now_epoch = now_epoch or int(time.time())
        metadata = self.normalize_metadata(dict(memory.get("metadata") or {}), memory, now_epoch)
        importance_weight = self._importance_score(metadata)
        similarity_score = self._semantic_similarity(memory)
        decay_value = self.decay_value(metadata, memory, now_epoch)
        recency_bonus = self._recency_bonus(decay_value)
        feedback_weight = self.clamp_feedback(metadata.get("feedback_weight", 0.0))
        event_boost = self.clamp_event_boost(metadata.get("event_boost", 0.0))
        final_score = (
            importance_weight * 0.4
            + similarity_score * 0.3
            + recency_bonus * 0.1
            + feedback_weight * 0.1
            + event_boost * 0.1
        )
        return {
            "score": round(final_score, 6),
            "metadata": metadata,
            "components": {
                "importance_weight": round(importance_weight, 6),
                "similarity_score": round(similarity_score, 6),
                "recency_bonus": round(recency_bonus, 6),
                "bounded_feedback_weight": round(feedback_weight, 6),
                "bounded_event_boost": round(event_boost, 6),
                "decay_value": round(decay_value, 6),
            },
        }

    def explain_score(self, memory: dict[str, Any], now_epoch: int | None = None) -> dict[str, Any]:
        now_epoch = now_epoch or int(time.time())
        score = self.score_memory(memory, now_epoch)
        components = score["components"]
        return {
            "memory_id": str(memory.get("id", "")),
            "final_score": score["score"],
            "score_breakdown": {
                "importance": components["importance_weight"],
                "similarity": components["similarity_score"],
                "recency_bonus": components["recency_bonus"],
                "feedback_weight": components["bounded_feedback_weight"],
                "event_boost": components["bounded_event_boost"],
                "decay_penalty": components["decay_value"],
            },
            "intermediate_calculations": {
                "weighted_importance": round(components["importance_weight"] * 0.4, 6),
                "weighted_similarity": round(components["similarity_score"] * 0.3, 6),
                "weighted_recency": round(components["recency_bonus"] * 0.1, 6),
                "weighted_feedback": round(components["bounded_feedback_weight"] * 0.1, 6),
                "weighted_event_boost": round(components["bounded_event_boost"] * 0.1, 6),
                "formula": "importance*0.4 + similarity*0.3 + recency_bonus*0.1 + feedback_weight*0.1 + event_boost*0.1",
            },
        }

    def rank_memories(
        self,
        memories: list[dict[str, Any]],
        limit: int,
        now_epoch: int | None = None,
    ) -> list[dict[str, Any]]:
        now_epoch = now_epoch or int(time.time())
        scored = []
        for memory in memories:
            ranked = dict(memory)
            score = self.score_memory(ranked, now_epoch)
            ranked["metadata"] = score["metadata"]
            ranked["evolution_score"] = score["score"]
            ranked["decay_score"] = score["score"]
            ranked["policy_score"] = score["score"]
            ranked["score_components"] = score["components"]
            ranked["score_explanation"] = self.explain_score(ranked, now_epoch)
            ranked["policy_score_components"] = {
                "importance_weight": score["components"]["importance_weight"],
                "semantic_similarity": score["components"]["similarity_score"],
                "recency_decay": score["components"]["decay_value"],
                "feedback_weight": score["components"]["bounded_feedback_weight"],
                "event_boost": score["components"]["bounded_event_boost"],
            }
            scored.append(ranked)
        scored.sort(key=lambda item: item.get("evolution_score", 0.0), reverse=True)
        return scored[:limit]

    def normalize_metadata(
        self,
        metadata: dict[str, Any],
        memory: dict[str, Any] | None = None,
        now_epoch: int | None = None,
    ) -> dict[str, Any]:
        now_epoch = now_epoch or int(time.time())
        normalized = dict(metadata or {})
        importance = self.clamp_importance(normalized.get("importance", normalized.get("importance_score", 0.5)))
        feedback_weight = self.clamp_feedback(normalized.get("feedback_weight", 0.0))
        event_boost = self.clamp_event_boost(normalized.get("event_boost", 0.0))
        decay = self.decay_value(normalized, memory or {}, now_epoch)
        normalized.update(
            {
                "importance": importance,
                "importance_score": importance,
                "feedback_weight": feedback_weight,
                "event_boost": event_boost,
                "decay": decay,
            }
        )
        return normalized

    @classmethod
    def normalize_static(cls, metadata: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(metadata or {})
        importance = cls.clamp_importance(normalized.get("importance", normalized.get("importance_score", 0.5)))
        normalized.update(
            {
                "importance": importance,
                "importance_score": importance,
                "feedback_weight": cls.clamp_feedback(normalized.get("feedback_weight", 0.0)),
                "event_boost": cls.clamp_event_boost(normalized.get("event_boost", 0.0)),
                "decay": max(0.0, cls._coerce_float(normalized.get("decay", 0.0), 0.0)),
            }
        )
        return normalized

    def reinforce_metadata(
        self,
        metadata: dict[str, Any],
        *,
        session_id: str = "default",
        now_epoch: int | None = None,
        base_delta: float = _BASE_REINFORCEMENT_DELTA,
        event_boost_delta: float = 0.0,
    ) -> tuple[dict[str, Any], bool]:
        now_epoch = now_epoch or int(time.time())
        normalized = self.normalize_metadata(metadata, None, now_epoch)
        session_counts = normalized.get("reinforcement_session_counts")
        if not isinstance(session_counts, dict):
            session_counts = {}
        session_count = int(session_counts.get(session_id, 0))
        if session_count >= self._MAX_REINFORCEMENT_PER_SESSION:
            normalized["reinforcement_skipped_reason"] = "session_reinforcement_limit"
            return normalized, False

        reinforcement_count = int(normalized.get("reinforcement_count") or 0)
        reinforcement_delta = base_delta / (1 + reinforcement_count)
        importance = self.clamp_importance(self._importance_score(normalized) + reinforcement_delta)
        feedback_weight = self.clamp_feedback(float(normalized.get("feedback_weight", 0.0)) + reinforcement_delta)
        event_boost = self.clamp_event_boost(float(normalized.get("event_boost", 0.0)) + event_boost_delta)
        session_counts[session_id] = session_count + 1
        normalized.update(
            {
                "importance": importance,
                "importance_score": importance,
                "feedback_weight": feedback_weight,
                "event_boost": event_boost,
                "reinforcement_count": reinforcement_count + 1,
                "reinforcement_session_counts": session_counts,
                "last_accessed_epoch": now_epoch,
                "last_reinforced_epoch": now_epoch,
                "reinforcement_delta": round(reinforcement_delta, 6),
            }
        )
        return normalized, True

    def decay_value(
        self,
        metadata: dict[str, Any],
        memory: dict[str, Any],
        now_epoch: int,
    ) -> float:
        epoch = self._last_access_epoch(metadata, memory)
        if epoch is None:
            return 0.0
        days_since_last_access = max(0.0, (now_epoch - epoch) / self._SECONDS_PER_DAY)
        return round(self._BASE_DECAY_RATE * math.log1p(days_since_last_access), 6)

    def reinforce_retrieved_memories(
        self,
        memories: list[dict[str, Any]],
        memory_service: "MemoryService",
        session_id: str | None = None,
        debug_service: "MemoryDebugService | None" = None,
    ) -> dict[str, Any]:
        updated = []
        skipped = []
        now = int(time.time())
        session_id = session_id or f"session:{now // 3600}"
        for memory in memories:
            memory_id = memory.get("id")
            if not memory_id:
                continue
            metadata = dict(memory.get("metadata") or {})
            if metadata.get("status") == "archived":
                continue

            normalized, applied = self.reinforce_metadata(metadata, session_id=session_id, now_epoch=now)
            retrieval_count = int(normalized.get("retrieval_count") or 0) + (1 if applied else 0)
            normalized["retrieval_count"] = retrieval_count
            normalized["status"] = self._stable_status(normalized, metadata.get("status", "active"))
            normalized["archived"] = bool(metadata.get("archived", False)) if normalized["status"] != "archived" else True

            if applied:
                memory_service.update_memory_metadata(str(memory_id), normalized)
                if debug_service:
                    debug_service.log_memory_lifecycle(
                        str(memory_id),
                        {
                            "event": "reinforced",
                            "timestamp": self._now_iso(),
                            "delta": normalized.get("reinforcement_delta"),
                            "score": normalized.get("importance_score"),
                            "session_id": session_id,
                        },
                    )
                updated.append(
                    {
                        "memory_id": str(memory_id),
                        "importance_score": normalized["importance_score"],
                        "feedback_weight": normalized["feedback_weight"],
                        "event_boost": normalized["event_boost"],
                        "reinforcement_count": normalized.get("reinforcement_count", 0),
                        "retrieval_count": retrieval_count,
                        "status": normalized.get("status", "active"),
                    }
                )
            else:
                skipped.append({"memory_id": str(memory_id), "reason": normalized.get("reinforcement_skipped_reason")})
        return {"updated": updated, "updated_count": len(updated), "skipped": skipped, "skipped_count": len(skipped)}

    def run_evolution_job(
        self,
        user_id: str,
        memory_service: "MemoryService",
        limit: int = 1000,
        debug_service: "MemoryDebugService | None" = None,
    ) -> dict[str, Any]:
        memories = memory_service.get_all_memories(user_id, limit)
        now = int(time.time())
        updated = []
        promoted = []
        demoted = []

        for memory in memories:
            memory_id = memory.get("id")
            if not memory_id:
                continue
            metadata = dict(memory.get("metadata") or {})
            if metadata.get("type") == "summary":
                continue

            normalized = self.normalize_metadata(metadata, memory, now)
            decay = float(normalized.get("decay", 0.0))
            importance = max(
                self._MIN_EFFECTIVE_IMPORTANCE,
                self.clamp_importance(self._importance_score(normalized) - min(decay, 0.4)),
            )
            normalized["importance"] = importance
            normalized["importance_score"] = importance
            normalized["inactive_days"] = round(self._inactive_days(normalized, memory, now), 2)
            normalized["last_evolved_epoch"] = now
            status = self._stable_status(normalized, metadata.get("status", "active"))
            normalized["status"] = status
            if status == "core_memory":
                promoted.append(str(memory_id))
                normalized["archived"] = False
            elif status == "archived":
                demoted.append(str(memory_id))
                normalized["archived"] = True

            memory_service.update_memory_metadata(str(memory_id), normalized)
            if debug_service:
                debug_service.log_memory_lifecycle(
                    str(memory_id),
                    {
                        "event": "decayed",
                        "timestamp": self._now_iso(),
                        "decay_value": decay,
                        "score": importance,
                        "status": normalized.get("status", "active"),
                    },
                )
            updated.append(str(memory_id))

        profile = self.regenerate_profile(user_id, memory_service, limit)
        return {
            "user_id": user_id,
            "updated_count": len(updated),
            "promoted_count": len(promoted),
            "demoted_count": len(demoted),
            "promoted_memory_ids": promoted,
            "demoted_memory_ids": demoted,
            "personality_profile": profile,
        }

    def regenerate_profile(
        self,
        user_id: str,
        memory_service: "MemoryService",
        limit: int = 1000,
    ) -> dict[str, Any]:
        memories = memory_service.get_all_memories(user_id, limit)
        evolved = [memory for memory in memories if (memory.get("metadata") or {}).get("status") != "archived"]
        emotions = Counter()
        topics = Counter()
        preferences = []

        for memory in evolved:
            metadata = memory.get("metadata") or {}
            if metadata.get("emotion"):
                emotions[str(metadata["emotion"])] += 1
            if metadata.get("topic"):
                topics[str(metadata["topic"])] += 1
            if metadata.get("type") == "preference" and self._importance_score(metadata) >= 0.5:
                preferences.append(str(memory.get("memory") or ""))

        profile = {
            "user_id": user_id,
            "generated_epoch": int(time.time()),
            "dominant_emotion": emotions.most_common(1)[0][0] if emotions else "neutral",
            "stable_preferences": preferences[:10],
            "recurring_topics": [topic for topic, _ in topics.most_common(10)],
            "behavioral_patterns": self._patterns(topics, emotions),
        }
        self._write_profile(user_id, profile)
        return profile

    def get_profile(self, user_id: str) -> dict[str, Any] | None:
        profiles = self._read_profiles()
        return profiles.get(user_id)

    def update_sleep_profile(
        self,
        user_id: str,
        memory_service: "MemoryService",
        limit: int = 90,
    ) -> dict[str, Any]:
        memories = memory_service.get_all_memories(user_id, limit)
        sleep_records = [
            memory
            for memory in memories
            if (memory.get("metadata") or {}).get("type") == "sleep"
            and (memory.get("metadata") or {}).get("sleep_duration") is not None
        ]
        baseline = self._sleep_baseline(sleep_records)
        profiles = self._read_profiles()
        profile = profiles.get(user_id, {"user_id": user_id})
        profile["sleep_profile"] = baseline
        profile["sleep_profile_updated_epoch"] = int(time.time())
        profiles[user_id] = profile
        self._profile_path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
        return baseline

    def _importance_score(self, metadata: dict[str, Any]) -> float:
        score = metadata.get("importance_score", metadata.get("importance"))
        if isinstance(score, int | float):
            return self.clamp_importance(score)
        return self._IMPORTANCE_FALLBACK.get(str(score or "medium"), 0.5)

    @classmethod
    def clamp_importance(cls, value: Any) -> float:
        return round(cls._clamp(cls._coerce_float(value, 0.5), cls._MIN_IMPORTANCE, cls._MAX_IMPORTANCE), 6)

    @classmethod
    def clamp_feedback(cls, value: Any) -> float:
        return round(cls._clamp(cls._coerce_float(value, 0.0), cls._MIN_FEEDBACK_WEIGHT, cls._MAX_FEEDBACK_WEIGHT), 6)

    @classmethod
    def clamp_event_boost(cls, value: Any) -> float:
        return round(cls._clamp(cls._coerce_float(value, 0.0), cls._MIN_EVENT_BOOST, cls._MAX_EVENT_BOOST), 6)

    @staticmethod
    def _semantic_similarity(memory: dict[str, Any]) -> float:
        score = memory.get("score", 0.0)
        if isinstance(score, int | float):
            return MemoryEvolutionEngine._clamp(float(score), 0.0, 1.0)
        return 0.0

    @staticmethod
    def _recency_bonus(decay_value: float) -> float:
        return round(MemoryEvolutionEngine._clamp(1.0 - decay_value, 0.0, 1.0), 6)

    def _stable_status(self, metadata: dict[str, Any], current_status: Any) -> str:
        importance = self._importance_score(metadata)
        retrieval_count = int(metadata.get("retrieval_count") or 0)
        inactive_days = float(metadata.get("inactive_days") or 0.0)
        if importance >= 0.85 and retrieval_count >= 4:
            return "core_memory"
        if importance <= 0.08 and inactive_days >= 90:
            return "archived"
        return str(current_status or "active")

    def _inactive_days(self, metadata: dict[str, Any], memory: dict[str, Any], now_epoch: int) -> float:
        epoch = self._last_access_epoch(metadata, memory)
        if epoch is None:
            return 0.0
        return max(0.0, (now_epoch - epoch) / self._SECONDS_PER_DAY)

    @staticmethod
    def _last_access_epoch(metadata: dict[str, Any], memory: dict[str, Any]) -> int | None:
        candidates = [
            metadata.get("last_accessed_epoch"),
            metadata.get("last_reinforced_epoch"),
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
                    from datetime import datetime

                    return int(datetime.fromisoformat(candidate.replace("Z", "+00:00")).timestamp())
                except ValueError:
                    continue
        return None

    @staticmethod
    def _patterns(topics: Counter, emotions: Counter) -> list[str]:
        patterns = []
        for topic, count in topics.most_common(5):
            if count >= 2:
                patterns.append(f"Recurring topic: {topic}")
        for emotion, count in emotions.most_common(3):
            if count >= 2:
                patterns.append(f"Recurring emotion: {emotion}")
        return patterns

    @classmethod
    def _sleep_baseline(cls, memories: list[dict[str, Any]]) -> dict[str, Any]:
        durations = []
        start_minutes = []
        deep_durations = []
        rem_durations = []
        awake_counts = []
        for memory in memories:
            metadata = memory.get("metadata") or {}
            duration = cls._coerce_float(metadata.get("sleep_duration"), -1.0)
            if duration >= 0:
                durations.append(duration)
            start_minute = cls._sleep_start_minute(metadata.get("sleep_start"))
            if start_minute is not None:
                start_minutes.append(start_minute)
            deep_duration = cls._coerce_float(metadata.get("deep_sleep_duration"), -1.0)
            if deep_duration >= 0:
                deep_durations.append(deep_duration)
            rem_duration = cls._coerce_float(metadata.get("rem_sleep_duration"), -1.0)
            if rem_duration >= 0:
                rem_durations.append(rem_duration)
            awake_count = metadata.get("awake_count")
            if isinstance(awake_count, int | float):
                awake_counts.append(int(awake_count))

        average_duration = cls._average(durations)
        duration_variance = cls._average([abs(duration - average_duration) for duration in durations]) if durations else 0.0
        consistency_score = cls._clamp(1.0 - (duration_variance / 4.0), 0.0, 1.0)
        return {
            "record_count": len(memories),
            "average_duration": round(average_duration, 2),
            "average_sleep_time": cls._minutes_to_hhmm(round(cls._circular_average_minutes(start_minutes))) if start_minutes else None,
            "average_deep_sleep_duration": round(cls._average(deep_durations), 2) if deep_durations else None,
            "average_rem_sleep_duration": round(cls._average(rem_durations), 2) if rem_durations else None,
            "average_awake_count": round(cls._average(awake_counts), 2) if awake_counts else None,
            "sleep_consistency_score": round(consistency_score, 3),
        }

    @staticmethod
    def _average(values: list[float] | list[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _sleep_start_minute(value: Any) -> int | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            from datetime import datetime

            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.hour * 60 + parsed.minute

    @staticmethod
    def _circular_average_minutes(values: list[int]) -> float:
        if not values:
            return 0.0
        radians = [2 * math.pi * (value / 1440) for value in values]
        sin_sum = sum(math.sin(value) for value in radians)
        cos_sum = sum(math.cos(value) for value in radians)
        angle = math.atan2(sin_sum / len(values), cos_sum / len(values))
        if angle < 0:
            angle += 2 * math.pi
        return (angle / (2 * math.pi)) * 1440

    @staticmethod
    def _minutes_to_hhmm(minutes: int) -> str:
        minutes = minutes % 1440
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    def _read_profiles(self) -> dict[str, Any]:
        if not self._profile_path.exists():
            return {}
        try:
            return json.loads(self._profile_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write_profile(self, user_id: str, profile: dict[str, Any]) -> None:
        profiles = self._read_profiles()
        profiles[user_id] = profile
        self._profile_path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _now_iso() -> str:
        from datetime import UTC, datetime

        return datetime.now(UTC).isoformat()

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        if isinstance(value, int | float):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))
