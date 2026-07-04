import time
from datetime import UTC, datetime
from typing import Any

from app.services.memory_evolution_engine import MemoryEvolutionEngine


class MemoryExplainabilityEngine:
    def __init__(self, evolution_engine: MemoryEvolutionEngine) -> None:
        self._evolution_engine = evolution_engine

    def explain_memory(
        self,
        memory: dict[str, Any] | None,
        lifecycle: list[dict[str, Any]] | None = None,
        now_epoch: int | None = None,
    ) -> dict[str, Any]:
        if not memory:
            return {
                "memory_id": None,
                "found": False,
                "decision_reason": ["memory not found in accessible memory set"],
                "lifecycle": lifecycle or [],
            }
        explanation = self.explain_ranking(memory, now_epoch)
        explanation["found"] = True
        explanation["content"] = memory.get("memory") or memory.get("content")
        explanation["metadata"] = memory.get("metadata") or {}
        explanation["lifecycle"] = lifecycle or self.lifecycle_from_metadata(memory)
        return explanation

    def explain_ranking(self, memory: dict[str, Any], now_epoch: int | None = None) -> dict[str, Any]:
        score = self._evolution_engine.score_memory(memory, now_epoch)
        components = score["components"]
        breakdown = {
            "importance": components["importance_weight"],
            "similarity": components["similarity_score"],
            "recency_bonus": components["recency_bonus"],
            "feedback_weight": components["bounded_feedback_weight"],
            "event_boost": components["bounded_event_boost"],
            "decay_penalty": components["decay_value"],
        }
        return {
            "memory_id": str(memory.get("id", "")),
            "final_score": score["score"],
            "score_breakdown": breakdown,
            "decision_reason": self._decision_reasons(breakdown, memory),
            "intermediate": {
                "formula": "importance*0.4 + similarity*0.3 + recency_bonus*0.1 + feedback_weight*0.1 + event_boost*0.1",
                "raw_components": components,
            },
        }

    def explain_ranking_list(self, memories: list[dict[str, Any]], now_epoch: int | None = None) -> list[dict[str, Any]]:
        ranked = self._evolution_engine.rank_memories(memories, len(memories), now_epoch)
        return [
            {
                **self.explain_ranking(memory, now_epoch),
                "rank": index + 1,
                "content": memory.get("memory") or memory.get("content"),
                "filtered": False,
            }
            for index, memory in enumerate(ranked)
        ]

    def explain_reinforcement(self, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        return {
            "event": "reinforced",
            "timestamp": self._now_iso(),
            "delta": round(
                float(after.get("importance_score", after.get("importance", 0.0)) or 0.0)
                - float(before.get("importance_score", before.get("importance", 0.0)) or 0.0),
                6,
            ),
            "feedback_delta": round(
                float(after.get("feedback_weight", 0.0) or 0.0) - float(before.get("feedback_weight", 0.0) or 0.0),
                6,
            ),
            "reinforcement_count": after.get("reinforcement_count", 0),
        }

    def lifecycle_from_metadata(self, memory: dict[str, Any]) -> list[dict[str, Any]]:
        metadata = memory.get("metadata") or {}
        lifecycle = []
        created_at = metadata.get("logged_epoch") or metadata.get("timestamp") or memory.get("created_at")
        lifecycle.append(
            {
                "event": "created",
                "timestamp": self._timestamp(created_at),
                "score": metadata.get("importance_score", metadata.get("importance", 0.0)),
            }
        )
        if metadata.get("retrieval_count"):
            lifecycle.append(
                {
                    "event": "retrieved",
                    "timestamp": self._timestamp(metadata.get("last_accessed_epoch")),
                    "score": memory.get("policy_score") or memory.get("evolution_score"),
                    "count": metadata.get("retrieval_count"),
                }
            )
        if metadata.get("reinforcement_count"):
            lifecycle.append(
                {
                    "event": "reinforced",
                    "timestamp": self._timestamp(metadata.get("last_reinforced_epoch")),
                    "delta": metadata.get("reinforcement_delta"),
                    "count": metadata.get("reinforcement_count"),
                }
            )
        if metadata.get("decay") is not None:
            lifecycle.append(
                {
                    "event": "decayed",
                    "timestamp": self._now_iso(),
                    "decay_value": metadata.get("decay"),
                }
            )
        if metadata.get("archived") is True:
            lifecycle.append(
                {
                    "event": "archived",
                    "timestamp": self._timestamp(metadata.get("archived_at")),
                    "summary_time_range": metadata.get("summary_time_range"),
                }
            )
        return lifecycle

    @staticmethod
    def _decision_reasons(breakdown: dict[str, float], memory: dict[str, Any]) -> list[str]:
        reasons = []
        if breakdown["similarity"] >= 0.75:
            reasons.append("high semantic similarity")
        elif breakdown["similarity"] <= 0.2:
            reasons.append("low semantic similarity")
        if breakdown["importance"] >= 0.75:
            reasons.append("high importance")
        if breakdown["recency_bonus"] >= 0.8:
            reasons.append("recently accessed")
        if breakdown["feedback_weight"] > 0:
            reasons.append("recently reinforced")
        if breakdown["event_boost"] > 0:
            reasons.append("event boosted")
        if breakdown["decay_penalty"] <= 0.1:
            reasons.append("low decay penalty")
        elif breakdown["decay_penalty"] >= 0.4:
            reasons.append("high decay penalty")
        metadata = memory.get("metadata") or {}
        if metadata.get("status") == "core_memory":
            reasons.append("core memory status")
        return reasons or ["balanced score components"]

    @staticmethod
    def _timestamp(value: Any) -> str:
        if isinstance(value, int | float):
            return datetime.fromtimestamp(float(value), UTC).isoformat()
        if isinstance(value, str) and value:
            return value
        return MemoryExplainabilityEngine._now_iso()

    @staticmethod
    def _now_iso() -> str:
        return datetime.fromtimestamp(time.time(), UTC).isoformat()
