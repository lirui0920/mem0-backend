import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Literal

from app.core.config import Settings
from app.schemas import StructuredMemory
from app.services.memory_evolution_engine import MemoryEvolutionEngine
from app.services.llm_service import LLMService
from app.services.memory_service import MemoryService

Emotion = Literal["happy", "sad", "anxious", "neutral", "angry"]
IntentType = Literal[
    "casual_chat",
    "emotional_support",
    "factual_question",
    "relationship_context",
    "memory_recall_request",
]


@dataclass(frozen=True)
class IntentClassification:
    emotion: Emotion
    intent_type: IntentType
    summary: str


@dataclass(frozen=True)
class WriteDecision:
    should_store: bool
    reason: str


class MemoryPolicyLayer:
    _IMPORTANCE_WEIGHTS = {"low": 0.2, "medium": 0.5, "high": 0.9}
    _VALID_RETRIEVAL_TYPES = {"chat", "sleep", "preference", "event", "summary"}
    _SECONDS_PER_DAY = 24 * 60 * 60
    _NOISE = {
        "ok",
        "okay",
        "k",
        "lol",
        "haha",
        "哈哈",
        "哈哈哈",
        "嗯",
        "嗯嗯",
        "哦",
        "噢",
        "啊",
        "好",
        "好的",
        "是的",
        "yes",
        "no",
        "hi",
        "hello",
    }

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._evolution_engine = MemoryEvolutionEngine(settings)
        self._recent_messages: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=12))

    def classify_intent(self, message: str) -> IntentClassification:
        text = message.strip()
        lower = text.lower()
        emotion = self._classify_emotion(lower)

        if self._contains_any(lower, ["remember", "recall", "do you remember", "你还记得", "记得吗", "我之前", "上次"]):
            intent_type: IntentType = "memory_recall_request"
        elif emotion in {"sad", "anxious", "angry"} or self._contains_any(lower, ["崩溃", "难受", "撑不住", "焦虑", "生气"]):
            intent_type = "emotional_support"
        elif self._contains_any(lower, ["relationship", "boyfriend", "girlfriend", "partner", "朋友", "同事", "伴侣", "男朋友", "女朋友", "家人", "妈妈", "爸爸"]):
            intent_type = "relationship_context"
        elif "?" in text or "？" in text or self._contains_any(lower, ["what", "why", "how", "when", "where", "什么", "为什么", "怎么", "如何", "多少"]):
            intent_type = "factual_question"
        else:
            intent_type = "casual_chat"

        return IntentClassification(
            emotion=emotion,
            intent_type=intent_type,
            summary=f"emotion={emotion}; intent={intent_type}",
        )

    def retrieve_memories(
        self,
        user_id: str,
        message: str,
        intent: IntentClassification,
        memory_service: MemoryService,
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        if intent.intent_type == "casual_chat" and not self._looks_memory_relevant(message):
            return []

        filters = self._filters_for_intent(intent)
        candidate_limit = max(top_k * 4, 20)
        candidates = memory_service.search_candidates(user_id, message, candidate_limit, filters)
        candidates = self._dedupe_memories(
            [
                *candidates,
                *self._core_candidates(user_id, message, memory_service, top_k),
            ]
        )
        ranked = memory_service.apply_decay_ranking(candidates, top_k)
        return [self.explain_memory(memory, intent, True) for memory in ranked]

    def debug_retrieve_memories(
        self,
        user_id: str,
        message: str,
        intent: IntentClassification,
        memory_service: MemoryService,
        top_k: int = 8,
    ) -> dict[str, Any]:
        if intent.intent_type == "casual_chat" and not self._looks_memory_relevant(message):
            return {
                "intent": intent,
                "filters": {},
                "selected": [],
                "ranking": [],
                "rejected": [],
                "matched_memory_types": {},
                "excluded_memory_reasons": {"casual_chat_without_memory_signal": 1},
                "skip_reason": "casual_chat_without_memory_signal",
            }

        filters = self._debug_filters_for_intent(intent)
        candidate_limit = max(top_k * 6, 40)
        candidates = memory_service.search_candidates(user_id, message, candidate_limit, filters)
        candidates = self._dedupe_memories(
            [
                *candidates,
                *self._core_candidates(user_id, message, memory_service, top_k),
            ]
        )
        archived_candidates = [
            memory
            for memory in candidates
            if (memory.get("metadata") or {}).get("archived") is True
            or (memory.get("metadata") or {}).get("status") == "archived"
        ]
        active_candidates = [
            memory
            for memory in candidates
            if (memory.get("metadata") or {}).get("archived") is not True
            and (memory.get("metadata") or {}).get("status") != "archived"
        ]
        ranked = memory_service.apply_decay_ranking(active_candidates, len(active_candidates) or candidate_limit)
        selected_ids = {self._memory_key(memory) for memory in ranked[:top_k]}
        explained = [self.explain_memory(memory, intent, self._memory_key(memory) in selected_ids) for memory in ranked]
        rejected = [
            {
                "memory": memory,
                "reason": "below_top_k_after_policy_rerank",
                "explanation": memory.get("explanation", {}),
            }
            for memory in explained[top_k:]
        ]
        rejected.extend(
            {
                "memory": self.explain_memory(memory, intent, False),
                "reason": "archived_memory_debug_only",
                "explanation": self.explain_memory(memory, intent, False).get("explanation", {}),
            }
            for memory in archived_candidates
        )
        return {
            "intent": intent,
            "filters": filters,
            "selected": explained[:top_k],
            "ranking": explained,
            "rejected": rejected,
            "matched_memory_types": self._type_counts(explained),
            "excluded_memory_reasons": self._rejection_counts(rejected),
            "skip_reason": None,
        }

    def rank_memories(
        self,
        memories: list[dict[str, Any]],
        top_k: int,
        now_epoch: int | None = None,
    ) -> list[dict[str, Any]]:
        active = [
            memory
            for memory in memories
            if (memory.get("metadata") or {}).get("archived") is not True
            and (memory.get("metadata") or {}).get("status") != "archived"
        ]
        return self._evolution_engine.rank_memories(active, top_k, now_epoch)

    def explain_memory(
        self,
        memory: dict[str, Any],
        intent: IntentClassification,
        selected: bool,
    ) -> dict[str, Any]:
        item = dict(memory)
        components = item.get("policy_score_components") or {}
        semantic_similarity = float(components.get("semantic_similarity", item.get("score", 0.0) or 0.0))
        importance_weight = float(components.get("importance_weight", 0.0))
        decay_penalty = float(components.get("recency_decay", 0.0))
        evolution_bonus = float(components.get("evolution_bonus", 0.0))
        feedback_weight = float(components.get("feedback_weight", 0.0))
        event_boost = float(components.get("event_boost", 0.0))
        intent_match_bonus = 0.0
        final_score = float(item.get("policy_score", item.get("evolution_score", 0.0)) or 0.0)
        item["explanation"] = {
            "reason": self._selection_reason(item, intent, selected),
            "score_breakdown": {
                "semantic_similarity": semantic_similarity,
                "importance_weight": importance_weight,
                "evolution_bonus": evolution_bonus,
                "feedback_weight": feedback_weight,
                "event_boost": event_boost,
                "decay_penalty": decay_penalty,
                "intent_match_bonus": intent_match_bonus,
            },
            "final_score": final_score,
        }
        return item

    def should_store_message(
        self,
        user_id: str,
        message: str,
        existing_memories: list[dict[str, Any]] | None = None,
    ) -> WriteDecision:
        normalized = self._normalize(message)
        if len(normalized) < 3:
            return WriteDecision(False, "too_short")
        if normalized in self._NOISE:
            return WriteDecision(False, "meaningless_noise")
        if self._is_plain_recall_request(normalized):
            return WriteDecision(False, "plain_recall_request")
        if self._is_non_personal_question(normalized):
            return WriteDecision(False, "non_personal_question")
        if self._is_low_value_chat(normalized):
            return WriteDecision(False, "low_value_conversation")
        if self._is_duplicate(user_id, normalized, existing_memories or []):
            return WriteDecision(False, "duplicate_or_near_duplicate")
        return WriteDecision(True, "candidate")

    def should_store_tagged_memory(self, memory: StructuredMemory) -> WriteDecision:
        if memory.type in {"preference", "event", "sleep", "summary"}:
            return WriteDecision(True, "valuable_memory_type")
        if memory.metadata.importance >= 0.5 and memory.metadata.emotion in {"happy", "sad", "anxious", "angry"}:
            return WriteDecision(True, "emotional_or_important_statement")
        return WriteDecision(False, "tagged_low_value_chat")

    def build_context_messages(
        self,
        user_id: str,
        user_message: str,
        intent: IntentClassification,
        memories: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        recent = self._format_recent_conversation(user_id)
        memory_context = self._format_memories(memories)
        return [
            {
                "role": "system",
                "content": (
                    "你是一个稳定、克制、有长期记忆的 AI 助手。"
                    "只使用提供的记忆上下文，不要编造未出现的用户事实。"
                    "如果记忆不相关，请忽略。回答应自然、简洁，并优先解决当前用户消息。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户意图摘要:\n{intent.summary}\n\n"
                    f"短期对话:\n{recent}\n\n"
                    f"受控长期记忆:\n{memory_context}\n\n"
                    f"当前用户消息:\n{user_message}"
                ),
            },
        ]

    def record_turn(self, user_id: str, user_message: str, assistant_response: str) -> None:
        history = self._recent_messages[user_id]
        history.append(f"User: {self._compact(user_message, 240)}")
        history.append(f"Assistant: {self._compact(assistant_response, 240)}")

    def _filters_for_intent(self, intent: IntentClassification) -> dict[str, Any]:
        if intent.intent_type == "emotional_support":
            return {
                "type": {"in": ["sleep", "preference", "event", "summary"]},
                "NOT": [{"archived": True}, {"status": "archived"}],
            }
        if intent.intent_type == "factual_question":
            return {
                "type": {"in": ["sleep", "preference", "event"]},
                "logged_epoch": {"gte": int(time.time()) - 30 * self._SECONDS_PER_DAY},
                "NOT": [{"archived": True}, {"status": "archived"}],
            }
        if intent.intent_type == "memory_recall_request":
            return {
                "type": {"in": ["summary", "sleep", "event", "preference"]},
                "OR": [{"importance": {"gte": 0.8}}, {"status": "core_memory"}],
                "NOT": [{"archived": True}, {"status": "archived"}],
            }
        if intent.intent_type == "relationship_context":
            return {
                "type": {"in": ["chat", "sleep", "preference", "event", "summary"]},
                "topic": {"in": ["relationship", "family", "friendship", "work"]},
                "NOT": [{"archived": True}, {"status": "archived"}],
            }
        return {
            "NOT": [{"archived": True}, {"status": "archived"}],
            "type": {"in": ["chat", "sleep", "event"]},
            "logged_epoch": {"gte": int(time.time()) - 7 * self._SECONDS_PER_DAY},
        }

    def _debug_filters_for_intent(self, intent: IntentClassification) -> dict[str, Any]:
        filters = dict(self._filters_for_intent(intent))
        filters.pop("NOT", None)
        return filters

    def _core_candidates(
        self,
        user_id: str,
        message: str,
        memory_service: MemoryService,
        top_k: int,
    ) -> list[dict[str, Any]]:
        return memory_service.search_candidates(
            user_id,
            message,
            max(3, min(top_k, 5)),
            {
                "status": "core_memory",
                "NOT": [{"archived": True}],
            },
        )

    def _selection_reason(self, memory: dict[str, Any], intent: IntentClassification, selected: bool) -> str:
        metadata = memory.get("metadata") or {}
        state = "Selected" if selected else "Ranked but not selected"
        return (
            f"{state}: matched {intent.intent_type} retrieval policy; "
            f"type={metadata.get('type', 'unknown')}, "
            f"importance={metadata.get('importance', 'unknown')}, "
            f"emotion={metadata.get('emotion', 'unknown')}."
        )

    def _classify_emotion(self, lower: str) -> Emotion:
        if self._contains_any(lower, ["angry", "mad", "furious", "生气", "愤怒", "烦死"]):
            return "angry"
        if self._contains_any(lower, ["anxious", "worried", "nervous", "焦虑", "担心", "紧张", "害怕"]):
            return "anxious"
        if self._contains_any(lower, ["sad", "down", "depressed", "难过", "伤心", "沮丧", "失落"]):
            return "sad"
        if self._contains_any(lower, ["happy", "great", "excited", "开心", "高兴", "兴奋", "快乐"]):
            return "happy"
        return "neutral"

    @staticmethod
    def _contains_any(text: str, needles: list[str]) -> bool:
        return any(needle in text for needle in needles)

    @staticmethod
    def _memory_key(memory: dict[str, Any]) -> str:
        return str(memory.get("id") or memory.get("memory") or memory.get("content") or memory)

    def _dedupe_memories(self, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen = set()
        deduped = []
        for memory in memories:
            key = self._memory_key(memory)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(memory)
        return deduped

    def _type_counts(self, memories: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for memory in memories:
            memory_type = str((memory.get("metadata") or {}).get("type", "unknown"))
            counts[memory_type] = counts.get(memory_type, 0) + 1
        return counts

    @staticmethod
    def _rejection_counts(rejected: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in rejected:
            reason = str(item.get("reason", "unknown"))
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    @staticmethod
    def _looks_memory_relevant(message: str) -> bool:
        lower = message.lower()
        return any(word in lower for word in ["remember", "记得", "之前", "喜欢", "讨厌", "计划", "important", "重要"])

    @staticmethod
    def _normalize(message: str) -> str:
        return re.sub(r"\s+", " ", message.strip().lower())

    @staticmethod
    def _compact(text: str, limit: int) -> str:
        cleaned = re.sub(r"\s+", " ", text.strip())
        return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."

    def _is_low_value_chat(self, normalized: str) -> bool:
        if len(normalized) <= 8 and not re.search(r"[\u4e00-\u9fff]{3,}|[a-zA-Z]{4,}", normalized):
            return True
        return False

    @staticmethod
    def _is_plain_recall_request(normalized: str) -> bool:
        return any(token in normalized for token in ["你还记得", "记得吗", "do you remember"]) and not any(
            signal in normalized for signal in ["我喜欢", "我讨厌", "my favorite", "i like", "i hate"]
        )

    @staticmethod
    def _is_non_personal_question(normalized: str) -> bool:
        question_like = normalized.endswith("?") or normalized.endswith("？") or any(
            token in normalized for token in ["怎么", "如何", "为什么", "什么是", "what", "why", "how"]
        )
        personal_signal = any(
            token in normalized
            for token in ["我", "我的", "i ", "my ", "喜欢", "讨厌", "计划", "今天", "昨天", "tomorrow", "today"]
        )
        return question_like and not personal_signal

    def _is_duplicate(self, user_id: str, normalized: str, existing_memories: list[dict[str, Any]]) -> bool:
        recent = self._recent_messages[user_id]
        for previous in recent:
            previous_normalized = self._normalize(previous)
            if previous_normalized == f"user: {normalized}":
                continue
            if SequenceMatcher(None, normalized, previous_normalized).ratio() >= 0.9:
                return True
        for memory in existing_memories[:5]:
            content = self._normalize(memory.get("memory") or memory.get("content") or "")
            if content and SequenceMatcher(None, normalized, content).ratio() >= 0.88:
                return True
        return False

    def _format_recent_conversation(self, user_id: str) -> str:
        history = list(self._recent_messages[user_id])[-6:]
        return "\n".join(history) if history else "暂无短期对话。"

    def _format_memories(self, memories: list[dict[str, Any]]) -> str:
        if not memories:
            return "无相关长期记忆。"
        lines = []
        for index, memory in enumerate(memories[:10], start=1):
            metadata = memory.get("metadata") or {}
            content = self._compact(memory.get("memory") or memory.get("content") or "", 260)
            if not content:
                continue
            meta = ", ".join(
                str(part)
                for part in [
                    metadata.get("type"),
                    metadata.get("importance"),
                    metadata.get("topic"),
                    self._identity_label(metadata),
                ]
                if part
            )
            prefix = f"{index}. [{meta}] " if meta else f"{index}. "
            lines.append(prefix + content)
        return "\n".join(lines) if lines else "无相关长期记忆。"

    @staticmethod
    def _identity_label(metadata: dict[str, Any]) -> str | None:
        speaker_role = metadata.get("speaker_role")
        target_role = metadata.get("target_role")
        subject_role = metadata.get("subject_role")
        if speaker_role:
            speaker = metadata.get("speaker_name") or metadata.get("speaker_id") or speaker_role
            if target_role:
                target = metadata.get("target_name") or metadata.get("target_id") or target_role
                return f"speaker={speaker_role}:{speaker}->target={target_role}:{target}"
            return f"speaker={speaker_role}:{speaker}"
        if subject_role:
            subject = metadata.get("subject_name") or metadata.get("subject_id") or subject_role
            return f"subject={subject_role}:{subject}"
        return None
