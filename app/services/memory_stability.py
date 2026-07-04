from dataclasses import asdict
from difflib import SequenceMatcher
from itertools import combinations
from statistics import mean
from typing import Any

from app.services.agent_core import AgentCore
from app.services.llm_service import LLMService


class MemoryStabilityTestEngine:
    def run(
        self,
        user_id: str,
        test_cases: list[str],
        repeat: int,
        agent_core: AgentCore,
        llm_service: LLMService,
    ) -> dict[str, Any]:
        per_case_results = [
            self._run_case(user_id, test_case, repeat, agent_core, llm_service)
            for test_case in test_cases
        ]
        summary = self._summary(per_case_results)
        drift = self._drift_analysis(per_case_results, summary)
        return {
            "test_summary": summary,
            "drift_analysis": drift,
            "per_case_results": per_case_results,
        }

    def _run_case(
        self,
        user_id: str,
        test_case: str,
        repeat: int,
        agent_core: AgentCore,
        llm_service: LLMService,
    ) -> dict[str, Any]:
        runs = []
        for index in range(repeat):
            _, memory_context = agent_core.prepare_context(user_id, test_case, 8)
            intent = memory_context.intent
            memories = memory_context.retrieved_memories
            prompt = memory_context.context_messages
            response = llm_service.generate_response(prompt)
            runs.append(
                {
                    "run": index + 1,
                    "intent": asdict(intent),
                    "retrieved_memories": self._memory_snapshot(memories),
                    "memory_scores": [
                        {
                            "memory_id": str(memory.get("id", "")),
                            "score": float(memory.get("policy_score", memory.get("score", 0.0)) or 0.0),
                            "type": (memory.get("metadata") or {}).get("type"),
                        }
                        for memory in memories
                    ],
                    "response": response,
                }
            )

        return {
            "test_case": test_case,
            "runs": runs,
            "scores": {
                "intent_stability": self._intent_stability(runs),
                "memory_stability": self._memory_stability(runs),
                "response_stability": self._response_stability(runs),
                "ranking_stability": self._ranking_stability(runs),
            },
            "drift": self._case_drift(runs),
        }

    @staticmethod
    def _memory_snapshot(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "memory_id": str(memory.get("id", "")),
                "memory": memory.get("memory"),
                "type": (memory.get("metadata") or {}).get("type"),
                "importance": (memory.get("metadata") or {}).get("importance"),
                "policy_score": memory.get("policy_score"),
                "explanation": memory.get("explanation"),
            }
            for memory in memories
        ]

    @staticmethod
    def _intent_stability(runs: list[dict[str, Any]]) -> float:
        if not runs:
            return 1.0
        intents = [run["intent"].get("intent_type") for run in runs]
        most_common = max(intents.count(intent) for intent in set(intents))
        return round(most_common / len(intents), 4)

    @staticmethod
    def _memory_stability(runs: list[dict[str, Any]]) -> float:
        memory_sets = [
            {memory["memory_id"] for memory in run["retrieved_memories"] if memory["memory_id"]}
            for run in runs
        ]
        if len(memory_sets) < 2:
            return 1.0
        scores = []
        for left, right in combinations(memory_sets, 2):
            if not left and not right:
                scores.append(1.0)
            else:
                scores.append(len(left & right) / max(len(left | right), 1))
        return round(mean(scores), 4)

    @staticmethod
    def _response_stability(runs: list[dict[str, Any]]) -> float:
        responses = [run["response"] for run in runs]
        if len(responses) < 2:
            return 1.0
        scores = [
            SequenceMatcher(None, left, right).ratio()
            for left, right in combinations(responses, 2)
        ]
        return round(mean(scores), 4)

    @staticmethod
    def _ranking_stability(runs: list[dict[str, Any]]) -> float:
        rankings = [
            [memory["memory_id"] for memory in run["retrieved_memories"] if memory["memory_id"]]
            for run in runs
        ]
        if len(rankings) < 2:
            return 1.0
        scores = []
        for left, right in combinations(rankings, 2):
            max_len = max(len(left), len(right), 1)
            matches = sum(1 for idx in range(min(len(left), len(right))) if left[idx] == right[idx])
            scores.append(matches / max_len)
        return round(mean(scores), 4)

    def _case_drift(self, runs: list[dict[str, Any]]) -> dict[str, Any]:
        unstable_components = []
        if self._intent_stability(runs) < 1.0:
            unstable_components.append("intent")
        if self._memory_stability(runs) < 0.8:
            unstable_components.append("retrieval")
        if self._ranking_stability(runs) < 0.8:
            unstable_components.append("ranking")
        if self._response_stability(runs) < 0.65:
            unstable_components.append("response")

        type_sets = [
            {memory["type"] for memory in run["retrieved_memories"] if memory.get("type")}
            for run in runs
        ]
        if len({tuple(sorted(types)) for types in type_sets}) > 1:
            unstable_components.append("memory_types")

        drift_level = self._drift_level(unstable_components)
        return {
            "drift_detected": bool(unstable_components),
            "drift_level": drift_level,
            "unstable_components": sorted(set(unstable_components)),
        }

    @staticmethod
    def _drift_level(unstable_components: list[str]) -> str:
        count = len(set(unstable_components))
        if count == 0:
            return "low"
        if count <= 2:
            return "medium"
        return "high"

    @staticmethod
    def _summary(per_case_results: list[dict[str, Any]]) -> dict[str, float]:
        if not per_case_results:
            return {
                "intent_stability": 1.0,
                "memory_stability": 1.0,
                "response_stability": 1.0,
            }
        return {
            "intent_stability": round(mean(case["scores"]["intent_stability"] for case in per_case_results), 4),
            "memory_stability": round(mean(case["scores"]["memory_stability"] for case in per_case_results), 4),
            "response_stability": round(mean(case["scores"]["response_stability"] for case in per_case_results), 4),
        }

    def _drift_analysis(self, per_case_results: list[dict[str, Any]], summary: dict[str, float]) -> dict[str, Any]:
        unstable = []
        for case in per_case_results:
            unstable.extend(case["drift"]["unstable_components"])
        if summary["intent_stability"] < 1.0:
            unstable.append("intent")
        if summary["memory_stability"] < 0.8:
            unstable.append("retrieval")
        if summary["response_stability"] < 0.65:
            unstable.append("response")
        unstable = sorted(set(unstable))
        return {
            "drift_detected": bool(unstable),
            "drift_level": self._drift_level(unstable),
            "unstable_components": unstable,
        }
