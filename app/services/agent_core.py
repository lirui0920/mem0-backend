import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from app.core.system_bus import MemoryContext, SystemBus, TriggerDecision
from app.services.llm_service import LLMService
from app.services.memory_orchestrator import MemoryOrchestrator
from app.services.memory_policy import IntentClassification, MemoryPolicyLayer
from app.services.memory_service import MemoryService
from app.services.understanding_service import StructuredUnderstanding, UnderstandingService


@dataclass
class AgentCoreChatResult:
    trace_id: str
    trace: dict[str, Any]
    understanding: StructuredUnderstanding
    memory_context: MemoryContext
    reasoning: dict[str, Any]
    memory_write_decision: dict[str, Any]
    response: str
    orchestrator_state: dict[str, Any]
    trigger_decision: TriggerDecision
    trigger_result: dict[str, Any]
    system_bus: SystemBus
    latency: dict[str, float]


@dataclass
class AgentMemoryWriteResult:
    should_store: bool
    reason: str
    memory: Any | None = None
    result: Any | None = None
    decision: dict[str, Any] | None = None


class AgentCore:
    """Single decision-making center for retrieval, prompts, events, and writes."""

    class MemoryPlanner:
        def __init__(self, memory_policy: MemoryPolicyLayer) -> None:
            self._memory_policy = memory_policy

        def plan(self, user_id: str, message: str, top_k: int = 8) -> dict[str, Any]:
            intent = self._memory_policy.classify_intent(message)
            query_strategy = self._query_strategy(intent.intent_type)
            should_query = len(message.strip()) >= 3
            return {
                "should_query_memory": should_query,
                "query_strategy": query_strategy,
                "filters": {},
                "intent": intent,
                "top_k": top_k,
                "query": message,
                "user_id": user_id,
            }

        @staticmethod
        def _query_strategy(intent_type: str) -> str:
            if intent_type == "casual_chat":
                return "recent"
            if intent_type in {"memory_recall_request", "relationship_context"}:
                return "hybrid"
            return "semantic"

    class ContextBuilder:
        def __init__(self, memory_policy: MemoryPolicyLayer) -> None:
            self._memory_policy = memory_policy

        def build(
            self,
            user_id: str,
            message: str,
            intent: IntentClassification,
            retrieved_memories: list[dict[str, Any]],
            plan: dict[str, Any],
        ) -> MemoryContext:
            context_messages = self._memory_policy.build_context_messages(
                user_id,
                message,
                intent,
                retrieved_memories,
            )
            memory_state = {
                "retrieved_count": len(retrieved_memories),
                "memory_ids": [str(memory.get("id")) for memory in retrieved_memories if memory.get("id")],
                "memory_types": [
                    (memory.get("metadata") or {}).get("type")
                    for memory in retrieved_memories
                    if (memory.get("metadata") or {}).get("type")
                ],
                "query_strategy": plan.get("query_strategy"),
                "filters": plan.get("filters", {}),
                "final_prompt_context": self._final_prompt_context(retrieved_memories),
            }
            return MemoryContext(
                intent=intent,
                retrieved_memories=retrieved_memories,
                context_messages=context_messages,
                memory_state=memory_state,
            )

        @staticmethod
        def _final_prompt_context(memories: list[dict[str, Any]]) -> dict[str, Any]:
            by_namespace = {"user": [], "agent": [], "summary": []}
            for memory in memories:
                metadata = memory.get("metadata") or {}
                namespace_kind = str(metadata.get("namespace_kind") or "user")
                if namespace_kind not in by_namespace:
                    namespace_kind = "user"
                by_namespace[namespace_kind].append(memory)
            return {
                "user_memory": by_namespace["user"],
                "agent_memory": by_namespace["agent"],
                "summary_memory": by_namespace["summary"],
                "chat_history": [],
            }

    class ReasoningEngine:
        def __init__(
            self,
            understanding_service: UnderstandingService,
            memory_policy: MemoryPolicyLayer,
        ) -> None:
            self._understanding_service = understanding_service
            self._memory_policy = memory_policy

        def interpret(
            self,
            message: str,
            planned_intent: IntentClassification | None = None,
        ) -> dict[str, Any]:
            understanding = self._understanding_service.parse(message)
            intent = planned_intent or self._memory_policy.classify_intent(message)
            signals = []
            if understanding.should_trigger_hint:
                signals.append(understanding.should_trigger_hint)
            if understanding.input_type in {"health", "sleep", "emotion"}:
                signals.append(understanding.input_type)
            return {
                "intent": intent.intent_type,
                "intent_object": intent,
                "emotion": intent.emotion,
                "emotion_score": understanding.severity,
                "signals": sorted(set(signals)),
                "understanding": understanding,
            }

    class EventDecider:
        def decide(
            self,
            reasoning: dict[str, Any],
            memory_context: MemoryContext,
            orchestrator_state: dict[str, Any],
        ) -> TriggerDecision:
            understanding = reasoning["understanding"]
            event_context = {
                "memory_ids": memory_context.memory_state.get("memory_ids", []),
                "understanding": understanding.model_dump(),
                "memory_state": memory_context.memory_state,
                "turn_count": int(orchestrator_state.get("turn_count", 0)),
                "token_accumulated": int(orchestrator_state.get("token_accumulated", 0)),
                "emotion_score_accumulated": float(orchestrator_state.get("emotion_score_accumulated", 0.0)),
                "signals": reasoning.get("signals", []),
            }

            if self._should_emit_emotional_event(understanding, orchestrator_state):
                return TriggerDecision(
                    action="emit_event",
                    reason="emotional_signal",
                    event_type="PROACTIVE_EVENT",
                    event_subtype="emotional_spike",
                    confidence=max(0.7, understanding.severity),
                    context=event_context,
                )

            if understanding.input_type in {"health", "sleep"}:
                return TriggerDecision(
                    action="emit_event",
                    reason="health_signal",
                    event_type="MEMORY_LIFECYCLE_EVENT",
                    event_subtype="health_signal",
                    confidence=max(0.65, understanding.severity),
                    context=event_context,
                )

            if self._should_emit_conversational_density(orchestrator_state):
                return TriggerDecision(
                    action="emit_event",
                    reason="conversational_density",
                    event_type="SUMMARY_EVENT",
                    event_subtype="conversational_density_high",
                    confidence=0.95,
                    context={
                        **event_context,
                        "reasons": self._summary_reasons(orchestrator_state),
                    },
                )

            if self._should_emit_proactive_message(understanding, orchestrator_state):
                return TriggerDecision(
                    action="emit_event",
                    reason="proactive_message_trigger",
                    event_type="PROACTIVE_EVENT",
                    event_subtype="proactive_message_trigger",
                    confidence=0.65,
                    context=event_context,
                )

            return TriggerDecision(action="do_nothing", reason="no_trigger")

        @staticmethod
        def _should_emit_emotional_event(
            understanding: StructuredUnderstanding,
            orchestrator_state: dict[str, Any],
        ) -> bool:
            if orchestrator_state.get("summary_lock") or orchestrator_state.get("summary_event_pending"):
                return False
            return (
                understanding.input_type == "emotion"
                or float(orchestrator_state.get("emotion_score_accumulated", 0.0)) >= 0.75
            )

        @staticmethod
        def _should_emit_conversational_density(orchestrator_state: dict[str, Any]) -> bool:
            if orchestrator_state.get("summary_lock") or orchestrator_state.get("summary_event_pending"):
                return False
            return (
                int(orchestrator_state.get("turn_count", 0)) >= 15
                or int(orchestrator_state.get("token_accumulated", 0)) >= 4000
            )

        @staticmethod
        def _should_emit_proactive_message(
            understanding: StructuredUnderstanding,
            orchestrator_state: dict[str, Any],
        ) -> bool:
            return bool(orchestrator_state.get("emotional_flag")) and understanding.input_type == "chat"

        @staticmethod
        def _summary_reasons(orchestrator_state: dict[str, Any]) -> list[str]:
            reasons = []
            if int(orchestrator_state.get("turn_count", 0)) >= 15:
                reasons.append("turn_count")
            if int(orchestrator_state.get("token_accumulated", 0)) >= 4000:
                reasons.append("token_threshold")
            return reasons

    class MemoryDecider:
        def __init__(self, memory_policy: MemoryPolicyLayer) -> None:
            self._memory_policy = memory_policy

        def decide(
            self,
            user_id: str,
            message: str,
            reasoning: dict[str, Any],
            agent_id: str | None = None,
            existing_memories: list[dict[str, Any]] | None = None,
        ) -> dict[str, Any]:
            understanding = reasoning["understanding"]
            policy_decision = self._memory_policy.should_store_message(user_id, message, existing_memories or [])
            if not policy_decision.should_store or not understanding.should_store:
                return {
                    "should_store": False,
                    "reason": policy_decision.reason if not policy_decision.should_store else "understanding_rejected",
                    "namespace": "user",
                    "type": "chat",
                    "importance": 0.0,
                }

            intent = reasoning["intent_object"]
            memory_type = self._memory_type(understanding, intent)
            namespace = self._namespace(message, intent, agent_id, memory_type)
            importance = self._importance(understanding, intent)
            return {
                "should_store": True,
                "reason": "agent_core_memory_decision",
                "namespace": namespace,
                "type": memory_type,
                "importance": importance,
            }

        @staticmethod
        def _memory_type(
            understanding: StructuredUnderstanding,
            intent: IntentClassification,
        ) -> str:
            if understanding.input_type == "sleep":
                return "sleep"
            if understanding.input_type == "health":
                return "event"
            if intent.intent_type == "relationship_context":
                return "event"
            if intent.intent_type == "emotional_support":
                return "chat"
            return "chat"

        @staticmethod
        def _namespace(
            message: str,
            intent: IntentClassification,
            agent_id: str | None,
            memory_type: str,
        ) -> str:
            lower = message.lower()
            agent_signals = (
                "this ai",
                "this assistant",
                "your tone",
                "how you talk",
                "你说话",
                "你的语气",
                "这个助手",
            )
            if agent_id and (
                intent.intent_type == "relationship_context" or any(signal in lower for signal in agent_signals)
            ):
                return "agent"
            if memory_type == "summary":
                return "summary"
            return "user"

        @staticmethod
        def _importance(
            understanding: StructuredUnderstanding,
            intent: IntentClassification,
        ) -> float:
            if understanding.severity >= 0.7:
                return 0.8
            if intent.intent_type in {"relationship_context", "emotional_support"}:
                return 0.6
            return 0.5

    class ResponseGenerator:
        def __init__(self, llm_service: LLMService) -> None:
            self._llm_service = llm_service

        def generate(self, context_messages: list[dict[str, str]]) -> str:
            return self._llm_service.generate_response(context_messages)

    def __init__(
        self,
        understanding_service: UnderstandingService,
        memory_policy: MemoryPolicyLayer,
        memory_service: MemoryService,
        llm_service: LLMService,
        memory_orchestrator: MemoryOrchestrator,
    ) -> None:
        self._understanding_service = understanding_service
        self._memory_policy = memory_policy
        self._memory_service = memory_service
        self._llm_service = llm_service
        self._memory_orchestrator = memory_orchestrator
        self._state: dict[str, dict[str, Any]] = {}
        self._memory_planner = self.MemoryPlanner(memory_policy)
        self._context_builder = self.ContextBuilder(memory_policy)
        self._reasoning_engine = self.ReasoningEngine(understanding_service, memory_policy)
        self._event_decider = self.EventDecider()
        self._memory_decider = self.MemoryDecider(memory_policy)
        self._response_generator = self.ResponseGenerator(llm_service)

    def run_chat(self, user_id: str, message: str, agent_id: str | None = None) -> AgentCoreChatResult:
        trace_id = str(uuid.uuid4())
        bus = SystemBus()
        total_start = time.perf_counter()

        retrieval_start = time.perf_counter()
        plan = self._memory_planner.plan(user_id, message, 8)
        plan["agent_id"] = agent_id
        memories = self._retrieve_with_plan(plan)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        memory_context = self._context_builder.build(user_id, message, plan["intent"], memories, plan)
        reasoning = self._reasoning_engine.interpret(message, plan["intent"])
        understanding = reasoning["understanding"]
        memory_context.memory_state.update(
            {
                "understanding_type": understanding.input_type,
                "understanding_intent": understanding.intent,
                "reasoning": {
                    "intent": reasoning["intent"],
                    "emotion_score": reasoning["emotion_score"],
                    "signals": reasoning["signals"],
                },
            }
        )
        bus.understanding = understanding
        bus.memory_context = memory_context

        orchestrator_state = self._memory_orchestrator.get_state(user_id)
        trigger_decision = self._event_decider.decide(reasoning, memory_context, orchestrator_state)
        bus.trigger_events.append(trigger_decision)

        memory_write_decision = self._memory_decider.decide(
            user_id,
            message,
            reasoning,
            agent_id,
            memories,
        )

        llm_start = time.perf_counter()
        response = self._response_generator.generate(memory_context.context_messages)
        llm_ms = (time.perf_counter() - llm_start) * 1000
        self._memory_policy.record_turn(user_id, message, response)
        self._update_internal_state(user_id, reasoning)

        orchestrator_state = self._memory_orchestrator.update_state(
            user_id,
            message,
            memory_context.intent,
            response,
            memory_context.retrieved_memories,
        )
        trigger_result = self.execute_event(trigger_decision, user_id)
        total_ms = (time.perf_counter() - total_start) * 1000
        trace = self._build_trace(
            trace_id,
            user_id,
            message,
            plan,
            memories,
            memory_context,
            reasoning,
            trigger_decision,
            memory_write_decision,
            response,
        )

        return AgentCoreChatResult(
            trace_id=trace_id,
            trace=trace,
            understanding=understanding,
            memory_context=memory_context,
            reasoning=reasoning,
            memory_write_decision=memory_write_decision,
            response=response,
            orchestrator_state=orchestrator_state,
            trigger_decision=trigger_decision,
            trigger_result=trigger_result,
            system_bus=bus,
            latency={
                "retrieval_ms": retrieval_ms,
                "llm_ms": llm_ms,
                "total_ms": total_ms,
            },
        )

    def build_memory_context(
        self,
        user_id: str,
        message: str,
        understanding: StructuredUnderstanding,
        top_k: int = 8,
    ) -> MemoryContext:
        plan = self._memory_planner.plan(user_id, message, top_k)
        memories = self._retrieve_with_plan(plan)
        memory_context = self._context_builder.build(user_id, message, plan["intent"], memories, plan)
        memory_context.memory_state.update(
            {
                "understanding_type": understanding.input_type,
                "understanding_intent": understanding.intent,
            }
        )
        return memory_context

    def prepare_context(self, user_id: str, message: str, top_k: int = 8) -> tuple[StructuredUnderstanding, MemoryContext]:
        reasoning = self._reasoning_engine.interpret(message)
        understanding = reasoning["understanding"]
        return understanding, self.build_memory_context(user_id, message, understanding, top_k)

    def retrieve_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        debug: bool = False,
    ) -> dict[str, Any]:
        plan = self._memory_planner.plan(user_id, query, limit)
        intent = plan["intent"]
        if debug:
            result = self._memory_policy.debug_retrieve_memories(
                user_id,
                query,
                intent,
                self._memory_service,
                limit,
            )
            return {
                "intent": result["intent"],
                "results": result["selected"],
                "debug_result": result,
            }
        results = self._retrieve_with_plan(plan)
        return {"intent": intent, "results": results, "debug_result": None}

    def store_memory_from_message(
        self,
        user_id: str,
        message: str,
        agent_id: str | None = None,
        source: str = "chat",
        role: str = "user",
        retrieved_memories: list[dict[str, Any]] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> AgentMemoryWriteResult:
        decision = self.decide_memory_write(user_id, message, agent_id, retrieved_memories or [])
        if not decision["should_store"]:
            return AgentMemoryWriteResult(
                should_store=False,
                reason=str(decision["reason"]),
                decision=decision,
            )

        memory = self._llm_service.tag_memory(user_id, message)
        metadata = {
            "source": source,
            "role": role,
            "memory_decision": decision,
            **(extra_metadata or {}),
        }
        if agent_id:
            metadata["agent_id"] = agent_id
        result = self._memory_service.add_structured_memory(memory, metadata)
        return AgentMemoryWriteResult(
            should_store=True,
            reason=str(decision["reason"]),
            memory=memory,
            result=result,
            decision=decision,
        )

    def decide_memory_write(
        self,
        user_id: str,
        message: str,
        agent_id: str | None = None,
        existing_memories: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        reasoning = self._reasoning_engine.interpret(message)
        return self._memory_decider.decide(user_id, message, reasoning, agent_id, existing_memories or [])

    def decide_event(
        self,
        user_id: str,
        understanding: StructuredUnderstanding,
        memory_context: MemoryContext,
        orchestrator_state: dict[str, Any],
    ) -> TriggerDecision:
        reasoning = {
            "understanding": understanding,
            "signals": [understanding.should_trigger_hint] if understanding.should_trigger_hint else [],
        }
        return self._event_decider.decide(reasoning, memory_context, orchestrator_state)

    def execute_event(self, decision: TriggerDecision, user_id: str) -> dict[str, Any]:
        if decision.action != "emit_event" or not decision.event_type or not decision.event_subtype:
            return {"action": decision.action, "reason": decision.reason, "event": None}

        event = self._memory_orchestrator.emit_event(
            decision.event_type,
            decision.event_subtype,
            user_id,
            confidence=decision.confidence,
            priority=decision.priority,
            context=decision.context,
        )
        return {"action": decision.action, "reason": decision.reason, "event": event}

    def _retrieve_with_plan(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        if not plan.get("should_query_memory", True):
            return []
        return self._memory_policy.retrieve_memories(
            str(plan["user_id"]),
            str(plan["query"]),
            plan["intent"],
            self._memory_service,
            int(plan.get("top_k", 8)),
        )

    def _update_internal_state(self, user_id: str, reasoning: dict[str, Any]) -> None:
        state = self._state.get(
            user_id,
            {
                "user_id": user_id,
                "turn_count": 0,
                "session_emotion_score": 0.0,
                "recent_signals": [],
            },
        )
        state["turn_count"] = int(state.get("turn_count", 0)) + 1
        state["session_emotion_score"] = round(
            float(state.get("session_emotion_score", 0.0)) + float(reasoning.get("emotion_score", 0.0)),
            4,
        )
        recent_signals = [*list(state.get("recent_signals", [])), *list(reasoning.get("signals", []))]
        state["recent_signals"] = recent_signals[-20:]
        self._state[user_id] = state

    def _build_trace(
        self,
        trace_id: str,
        user_id: str,
        message: str,
        plan: dict[str, Any],
        memories: list[dict[str, Any]],
        memory_context: MemoryContext,
        reasoning: dict[str, Any],
        trigger_decision: TriggerDecision,
        memory_write_decision: dict[str, Any],
        response: str,
    ) -> dict[str, Any]:
        return {
            "trace_id": trace_id,
            "user_id": user_id,
            "stages": {
                "memory_planner": {
                    "input": {"user_id": user_id, "message": message},
                    "output": self._jsonable_plan(plan),
                    "key_signals_used": ["message_length", "intent_type"],
                },
                "context_builder": {
                    "input": {
                        "memory_count": len(memories),
                        "memory_ids": [str(memory.get("id")) for memory in memories if memory.get("id")],
                    },
                    "output": {
                        "context_message_count": len(memory_context.context_messages),
                        "memory_state": memory_context.memory_state,
                    },
                    "key_signals_used": ["retrieved_memories", "namespace_kind", "summary_memory"],
                },
                "reasoning_engine": {
                    "input": {"message": message},
                    "output": {
                        "intent": reasoning.get("intent"),
                        "emotion": reasoning.get("emotion"),
                        "emotion_score": reasoning.get("emotion_score"),
                        "signals": reasoning.get("signals", []),
                        "understanding": reasoning["understanding"].model_dump(),
                    },
                    "key_signals_used": reasoning.get("signals", []),
                },
                "event_decider": {
                    "input": {
                        "reasoning": {
                            "intent": reasoning.get("intent"),
                            "emotion_score": reasoning.get("emotion_score"),
                            "signals": reasoning.get("signals", []),
                        },
                        "memory_state": memory_context.memory_state,
                    },
                    "output": trigger_decision.model_dump(),
                    "key_signals_used": trigger_decision.context.get("signals", []),
                },
                "memory_decider": {
                    "input": {
                        "message": message,
                        "retrieved_memory_count": len(memories),
                        "agent_id_present": bool(plan.get("agent_id")),
                    },
                    "output": memory_write_decision,
                    "key_signals_used": ["understanding.should_store", "policy_duplicate_check", "importance"],
                },
                "response_generator": {
                    "input": {"context_message_count": len(memory_context.context_messages)},
                    "output": {"response_length": len(response)},
                    "key_signals_used": ["final_prompt_context"],
                },
            },
        }

    @staticmethod
    def _jsonable_plan(plan: dict[str, Any]) -> dict[str, Any]:
        result = dict(plan)
        result["intent"] = AgentCore.intent_dict(result["intent"])
        return result

    @staticmethod
    def intent_dict(intent: Any) -> dict[str, Any]:
        return asdict(intent) if hasattr(intent, "__dataclass_fields__") else dict(intent or {})
