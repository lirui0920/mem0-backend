import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.services.memory_service import MemoryService


class MemoryEvolutionEngine:
    _IMPORTANCE_FALLBACK = {"low": 0.2, "medium": 0.5, "high": 0.8}
    _SECONDS_PER_DAY = 24 * 60 * 60

    def __init__(self, settings: Settings) -> None:
        self._profile_path = Path(settings.memory_profile_path)
        self._profile_path.parent.mkdir(parents=True, exist_ok=True)

    def reinforce_retrieved_memories(
        self,
        memories: list[dict[str, Any]],
        memory_service: MemoryService,
    ) -> dict[str, Any]:
        updated = []
        now = int(time.time())
        for memory in memories:
            memory_id = memory.get("id")
            if not memory_id:
                continue
            metadata = dict(memory.get("metadata") or {})
            if metadata.get("status") == "archived":
                continue

            retrieval_count = int(metadata.get("retrieval_count") or 0) + 1
            importance_score = round(min(1.0, self._importance_score(metadata) + 0.1), 4)
            metadata.update(
                {
                    "retrieval_count": retrieval_count,
                    "last_accessed_epoch": now,
                    "importance_score": importance_score,
                    "status": metadata.get("status", "active"),
                }
            )
            if importance_score >= 0.8 and retrieval_count >= 3:
                metadata["status"] = "core_memory"
                metadata["archived"] = False

            memory_service.update_memory_metadata(str(memory_id), metadata)
            updated.append(
                {
                    "memory_id": str(memory_id),
                    "importance_score": metadata["importance_score"],
                    "retrieval_count": retrieval_count,
                    "status": metadata.get("status", "active"),
                }
            )
        return {"updated": updated, "updated_count": len(updated)}

    def run_evolution_job(
        self,
        user_id: str,
        memory_service: MemoryService,
        limit: int = 1000,
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

            last_accessed = int(metadata.get("last_accessed_epoch") or metadata.get("logged_epoch") or now)
            inactive_days = max(0, (now - last_accessed) / self._SECONDS_PER_DAY)
            importance_score = self._importance_score(metadata)
            importance_score = max(0.1, importance_score - self._decay_amount(inactive_days))

            retrieval_count = int(metadata.get("retrieval_count") or 0)
            status = metadata.get("status", "active")
            if importance_score >= 0.8 and retrieval_count >= 3:
                status = "core_memory"
                metadata["archived"] = False
                promoted.append(str(memory_id))
            elif importance_score < 0.2 and inactive_days >= 30:
                status = "archived"
                metadata["archived"] = True
                demoted.append(str(memory_id))

            metadata.update(
                {
                    "importance_score": round(importance_score, 4),
                    "status": status,
                    "last_evolved_epoch": now,
                    "inactive_days": round(inactive_days, 2),
                }
            )
            memory_service.update_memory_metadata(str(memory_id), metadata)
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
        memory_service: MemoryService,
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

    def _importance_score(self, metadata: dict[str, Any]) -> float:
        score = metadata.get("importance_score")
        if isinstance(score, int | float):
            return float(score)
        return self._IMPORTANCE_FALLBACK.get(str(metadata.get("importance", "medium")), 0.5)

    @staticmethod
    def _decay_amount(inactive_days: float) -> float:
        if inactive_days >= 90:
            return 0.6
        if inactive_days >= 30:
            return 0.3
        if inactive_days >= 7:
            return 0.1
        return 0.0

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
