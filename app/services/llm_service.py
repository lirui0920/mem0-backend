from datetime import UTC, datetime
from json import JSONDecodeError

from openai import OpenAI
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas import MemorySummary, StructuredMemory


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )

    def tag_memory(self, user_id: str, content: str) -> StructuredMemory:
        timestamp = datetime.now(UTC).isoformat()
        messages = [
            {
                "role": "system",
                "content": (
                    "You extract metadata for a long-term AI memory system. "
                    "Return only valid JSON matching this schema: "
                    '{"user_id":"string","content":"string","metadata":'
                    '{"emotion":"happy|sad|angry|anxious|neutral",'
                    '"type":"fact|chat|preference|event",'
                    '"importance":"low|medium|high",'
                    '"topic":"short topic such as health, relationship, daily life, work, finance, study, travel, family, hobby, or other",'
                    '"timestamp":"ISO-8601 timestamp"}}. '
                    "Use the supplied user_id and original message exactly. "
                    "For timestamp, use an explicit date/time from the message when present; otherwise use the supplied current timestamp. "
                    "Classify emotion, type, importance, and topic using your language understanding. "
                    "Importance should consider sentiment strength, urgency, durable user preferences/facts, and strong intent. "
                    "Do not add facts that are not present in the message."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"user_id: {user_id}\n"
                    f"current_timestamp: {timestamp}\n"
                    f"original_message: {content}"
                ),
            },
        ]
        raw = self._complete_json(messages)
        try:
            memory = StructuredMemory.model_validate_json(raw)
        except (ValidationError, JSONDecodeError) as exc:
            raise ValueError(f"LLM returned invalid memory tag JSON: {raw}") from exc
        if memory.user_id != user_id or memory.content != content:
            raise ValueError("LLM returned a memory object that does not match the original input.")
        return memory

    def chat(self, user_message: str, memories: list[dict], user_id: str) -> str:
        memory_context = self._format_memories(memories)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个有长期记忆的 AI 聊天助手。"
                    "请优先利用提供的长期记忆回答用户，但不要编造不存在的记忆。"
                    "如果记忆与当前问题无关，请自然忽略。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户ID: {user_id}\n\n"
                    f"相关长期记忆:\n{memory_context}\n\n"
                    f"用户当前消息:\n{user_message}"
                ),
            },
        ]
        return self._complete(messages)

    def generate_response(self, messages: list[dict[str, str]]) -> str:
        return self._complete(messages)

    def summarize_diary(self, memories: list[dict], user_id: str, timezone: str) -> str:
        memory_context = self._format_memories(memories)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个善于整理个人记录的中文日记助手。"
                    "请基于给定记忆生成一篇自然、克制、真实的日记。"
                    "不要加入未出现的事实。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户ID: {user_id}\n"
                    f"时区: {timezone}\n\n"
                    f"过去24小时记忆:\n{memory_context}\n\n"
                    "请输出一篇中文日记，包含今天发生的事、情绪/想法和可以跟进的事项。"
                ),
            },
        ]
        return self._complete(messages)

    def summarize_memories(
        self,
        user_id: str,
        memories: list[dict],
        start_epoch: int,
        end_epoch: int,
    ) -> MemorySummary:
        start = datetime.fromtimestamp(start_epoch, UTC).isoformat()
        end = datetime.fromtimestamp(end_epoch, UTC).isoformat()
        memory_context = self._format_memories(memories)
        messages = [
            {
                "role": "system",
                "content": (
                    "You generate compact long-term memory summaries. "
                    "Return only valid JSON matching this schema: "
                    '{"user_id":"string","daily_summary":"string","emotional_trend":"string",'
                    '"key_events":["string"],"new_user_preferences":["string"],'
                    '"time_range":{"start":"ISO-8601","end":"ISO-8601","start_epoch":0,"end_epoch":0}}. '
                    "Do not invent details. Summarize only the supplied memories. "
                    "Keep key_events and new_user_preferences concise and deduplicated."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"user_id: {user_id}\n"
                    f"time_range.start: {start}\n"
                    f"time_range.end: {end}\n"
                    f"time_range.start_epoch: {start_epoch}\n"
                    f"time_range.end_epoch: {end_epoch}\n\n"
                    f"memories:\n{memory_context}"
                ),
            },
        ]
        raw = self._complete_json(messages, max_tokens=1200)
        try:
            summary = MemorySummary.model_validate_json(raw)
        except (ValidationError, JSONDecodeError) as exc:
            raise ValueError(f"LLM returned invalid summary JSON: {raw}") from exc
        if summary.user_id != user_id:
            raise ValueError("LLM returned a summary object for the wrong user.")
        return summary

    def _complete(self, messages: list[dict[str, str]]) -> str:
        completion = self._client.chat.completions.create(
            model=self._settings.llm_chat_model,
            messages=messages,
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
        )
        return completion.choices[0].message.content or ""

    def _complete_json(self, messages: list[dict[str, str]], max_tokens: int = 500) -> str:
        completion = self._client.chat.completions.create(
            model=self._settings.llm_chat_model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content or "{}"

    @staticmethod
    def _format_memories(memories: list[dict]) -> str:
        if not memories:
            return "暂无相关记忆。"

        lines = []
        for index, memory in enumerate(memories, start=1):
            content = memory.get("memory") or memory.get("content") or str(memory)
            score = memory.get("score")
            score_text = f" (score={score:.3f})" if isinstance(score, float) else ""
            lines.append(f"{index}. {content}{score_text}")
        return "\n".join(lines)
