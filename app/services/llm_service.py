from datetime import UTC, datetime
from json import JSONDecodeError
from typing import Any
from uuid import uuid4

from openai import OpenAI
from pydantic import ValidationError

from app.core.config import Settings
from app.models.memory import resolve_memory_namespace
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
        memory_id = str(uuid4())
        messages = [
            {
                "role": "system",
                "content": (
                    "You extract metadata for a long-term AI memory system. "
                    "Return only valid JSON matching this schema: "
                    '{"id":"string","user_id":"string","agent_id":null,'
                    '"namespace":"user:{user_id}","type":"chat|sleep|preference|event|summary",'
                    '"content":"string","embedding":null,"metadata":'
                    '{"emotion":"happy|sad|angry|anxious|neutral",'
                    '"importance":0.0,'
                    '"decay":0.0,'
                    '"feedback_weight":0.0,'
                    '"topic":"short topic such as health, relationship, daily life, work, finance, study, travel, family, hobby, or other",'
                    '"timestamp":"ISO-8601 timestamp"}}. '
                    "Use the supplied memory_id, user_id, namespace, and original message exactly. "
                    "For timestamp, use an explicit date/time from the message when present; otherwise use the supplied current timestamp. "
                    "Classify emotion, type, numeric importance from 0.0 to 1.0, and topic using your language understanding. "
                    "Use type sleep for durable sleep-related memory, preference for durable likes/dislikes/settings, event for facts/events/plans, and chat for useful conversational context. "
                    "Do not use type summary for a single user message. "
                    "Importance should consider sentiment strength, urgency, durable user preferences/events, and strong intent. "
                    "Do not add facts that are not present in the message."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"memory_id: {memory_id}\n"
                    f"user_id: {user_id}\n"
                    f"namespace: {resolve_memory_namespace(user_id, 'chat')}\n"
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
                    "你负责为长期记忆系统生成紧凑、准确的中文总结。"
                    "Return only valid JSON matching this schema: "
                    '{"user_id":"string","daily_summary":"string","emotional_trend":"string",'
                    '"key_events":["string"],"new_user_preferences":["string"],'
                    '"time_range":{"start":"ISO-8601","end":"ISO-8601","start_epoch":0,"end_epoch":0}}. '
                    "所有 string 字段必须使用简体中文。"
                    "不要编造细节，只总结提供的记忆。"
                    "key_events 和 new_user_preferences 要简洁、去重。"
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

    def summarize_agent_memories(
        self,
        user_id: str,
        agent_id: str,
        memories: list[dict],
    ) -> dict[str, Any]:
        memory_context = self._format_memories(memories)
        messages = [
            {
                "role": "system",
                "content": (
                    "你负责总结某个用户和某个 AI 角色之间的互动记忆。"
                    "请只基于提供的记忆，输出简体中文 JSON，不要编造。"
                    "Return only valid JSON matching this schema: "
                    '{"user_id":"string","agent_id":"string","interaction_summary":"string",'
                    '"relationship_tone":"string","style_preferences":["string"],'
                    '"recent_events":["string"],"conflicts":["string"],"boundaries":["string"],'
                    '"open_loops":["string"]}. '
                    "重点总结：用户喜欢这个 AI 如何说话、如何调情/安抚/互动，最近发生过什么事件或冲突，以及后续需要注意什么边界。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"user_id: {user_id}\n"
                    f"agent_id: {agent_id}\n\n"
                    f"agent 相关记忆:\n{memory_context}"
                ),
            },
        ]
        raw = self._complete_json(messages, max_tokens=1200)
        try:
            summary = __import__("json").loads(raw)
        except JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid agent summary JSON: {raw}") from exc
        summary["user_id"] = user_id
        summary["agent_id"] = agent_id
        return summary

    def summarize_agent_interaction_events(
        self,
        user_id: str,
        agent_id: str,
        memories: list[dict],
    ) -> dict[str, Any]:
        memory_context = self._format_memories(memories)
        messages = [
            {
                "role": "system",
                "content": (
                    "你负责把某个用户和某个 AI 角色之间的多轮互动记忆，拆分成多条可检索的中文长期事件总结。"
                    "不要把不同主题挤在同一条 summary 里。角色扮演、日常分享、吵架/冲突、调情风格、用户对该 AI 的要求，应尽量分开。"
                    "请只基于提供的记忆，不要编造。"
                    "Return only valid JSON matching this schema: "
                    '{"user_id":"string","agent_id":"string","events":['
                    '{"category":"roleplay|daily_share|conflict|flirting_style|agent_preference|boundary|repair|other",'
                    '"title":"string","summary":"string","preference_update":"string",'
                    '"follow_up":"string","importance":0.0,"start_time":"ISO-8601 or empty",'
                    '"end_time":"ISO-8601 or empty","source_memory_ids":["string"]}'
                    "]}. "
                    "所有 string 字段必须使用简体中文。"
                    "每个 event 只能表达一个主题。"
                    "如果没有足够信息，不要输出空泛总结。"
                    "summary 必须明确这是哪个用户和哪个 AI 之间的互动，并写清楚时间是历史时间，不要暗示一定发生在今天。"
                    "agent_preference 用来记录用户希望这个 AI 做什么或保持什么风格，例如占有欲强、主动、温柔、解释清楚。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"user_id: {user_id}\n"
                    f"agent_id: {agent_id}\n\n"
                    f"agent 相关记忆:\n{memory_context}"
                ),
            },
        ]
        raw = self._complete_json(messages, max_tokens=1800)
        try:
            summary = __import__("json").loads(raw)
        except JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid agent event summary JSON: {raw}") from exc
        summary["user_id"] = user_id
        summary["agent_id"] = agent_id
        events = summary.get("events")
        summary["events"] = events if isinstance(events, list) else []
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
            metadata = memory.get("metadata") or {}
            labels = []
            if memory.get("id"):
                labels.append(f"id={memory['id']}")
            if metadata.get("type"):
                labels.append(f"type={metadata['type']}")
            if metadata.get("timestamp"):
                labels.append(f"time={metadata['timestamp']}")
            elif metadata.get("logged_epoch"):
                labels.append(f"logged_epoch={metadata['logged_epoch']}")
            if metadata.get("speaker_role"):
                speaker = metadata.get("speaker_name") or metadata.get("speaker_id") or metadata["speaker_role"]
                labels.append(f"speaker={metadata['speaker_role']}:{speaker}")
            if metadata.get("target_role"):
                target = metadata.get("target_name") or metadata.get("target_id") or metadata["target_role"]
                labels.append(f"target={metadata['target_role']}:{target}")
            if metadata.get("subject_role"):
                subject = metadata.get("subject_name") or metadata.get("subject_id") or metadata["subject_role"]
                labels.append(f"subject={metadata['subject_role']}:{subject}")
            if metadata.get("user_name"):
                labels.append(f"user_name={metadata['user_name']}")
            if metadata.get("agent_name"):
                labels.append(f"agent_name={metadata['agent_name']}")
            score = memory.get("score")
            score_text = f" (score={score:.3f})" if isinstance(score, float) else ""
            label_text = f" [{' / '.join(labels)}]" if labels else ""
            lines.append(f"{index}. {content}{label_text}{score_text}")
        return "\n".join(lines)
