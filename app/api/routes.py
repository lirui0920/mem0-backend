import logging
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from dataclasses import asdict
from functools import lru_cache

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings, get_settings
from app.core.security import verify_api_key
from app.schemas import (
    AgentSummaryRunRequest,
    AgentSummaryRunResponse,
    ChatImportRequest,
    ChatImportResponse,
    ChatRequest,
    ChatResponse,
    DiaryGenerateRequest,
    DiaryGenerateResponse,
    HealthResponse,
    MemoryAddRequest,
    MemoryAddResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryStabilityTestRequest,
    SleepInput,
    SleepResponse,
    SummaryRunRequest,
    SummaryRunResponse,
)
from app.services.agent_core import AgentCore
from app.services.llm_service import LLMService
from app.services.memory_debug import MemoryDebugService
from app.services.memory_evolution import MemoryEvolutionEngine
from app.services.memory_explainability_engine import MemoryExplainabilityEngine
from app.services.memory_orchestrator import MemoryOrchestrator
from app.services.memory_policy import MemoryPolicyLayer
from app.services.memory_stability import MemoryStabilityTestEngine
from app.services.memory_service import MemoryService
from app.services.understanding_service import UnderstandingService

router = APIRouter(dependencies=[Depends(verify_api_key)])
public_router = APIRouter()
logger = logging.getLogger(__name__)


@lru_cache
def _memory_service() -> MemoryService:
    return MemoryService(get_settings())


@lru_cache
def _llm_service() -> LLMService:
    return LLMService(get_settings())


@lru_cache
def _memory_policy() -> MemoryPolicyLayer:
    return MemoryPolicyLayer(get_settings())


@lru_cache
def _memory_debug() -> MemoryDebugService:
    return MemoryDebugService(get_settings())


@lru_cache
def _memory_stability() -> MemoryStabilityTestEngine:
    return MemoryStabilityTestEngine()


@lru_cache
def _memory_evolution() -> MemoryEvolutionEngine:
    return MemoryEvolutionEngine(get_settings())


@lru_cache
def _memory_explainability() -> MemoryExplainabilityEngine:
    return MemoryExplainabilityEngine(_memory_evolution())


@lru_cache
def _memory_orchestrator() -> MemoryOrchestrator:
    return MemoryOrchestrator(get_settings())


@lru_cache
def _understanding_service() -> UnderstandingService:
    return UnderstandingService()


@lru_cache
def _agent_core() -> AgentCore:
    return AgentCore(
        _understanding_service(),
        _memory_policy(),
        _memory_service(),
        _llm_service(),
        _memory_orchestrator(),
    )


def get_memory_service(settings: Settings = Depends(get_settings)) -> MemoryService:
    _ = settings
    return _memory_service()


def get_llm_service(settings: Settings = Depends(get_settings)) -> LLMService:
    _ = settings
    return _llm_service()


def get_memory_policy(settings: Settings = Depends(get_settings)) -> MemoryPolicyLayer:
    _ = settings
    return _memory_policy()


def get_memory_debug(settings: Settings = Depends(get_settings)) -> MemoryDebugService:
    _ = settings
    return _memory_debug()


def get_memory_stability() -> MemoryStabilityTestEngine:
    return _memory_stability()


def get_memory_evolution(settings: Settings = Depends(get_settings)) -> MemoryEvolutionEngine:
    _ = settings
    return _memory_evolution()


def get_memory_explainability(settings: Settings = Depends(get_settings)) -> MemoryExplainabilityEngine:
    _ = settings
    return _memory_explainability()


def get_memory_orchestrator(settings: Settings = Depends(get_settings)) -> MemoryOrchestrator:
    _ = settings
    return _memory_orchestrator()


def get_understanding_service() -> UnderstandingService:
    return _understanding_service()


def get_agent_core() -> AgentCore:
    return _agent_core()


def start_memory_orchestrator_worker() -> None:
    _memory_orchestrator().start(_memory_service(), _llm_service())


def stop_memory_orchestrator_worker() -> None:
    _memory_orchestrator().stop()


@public_router.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(status="ok", app=settings.app_name)


@router.post("/sleep", response_model=SleepResponse)
async def ingest_sleep(
    payload: SleepInput,
    background_tasks: BackgroundTasks,
    memory_service: MemoryService = Depends(get_memory_service),
    llm_service: LLMService = Depends(get_llm_service),
    memory_debug: MemoryDebugService = Depends(get_memory_debug),
    memory_evolution: MemoryEvolutionEngine = Depends(get_memory_evolution),
) -> SleepResponse:
    request_id = str(uuid.uuid4())
    try:
        write_result = await run_in_threadpool(memory_service.add_sleep_memory, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    memory = write_result["memory"]
    memory_id = memory.id
    memory_debug.log_memory_write(
        {
            "request_id": request_id,
            "trace_id": request_id,
            "user_id": payload.user_id,
            "user_name": payload.user_name,
            "agent_id": payload.agent_id,
            "agent_name": payload.agent_name,
            "outcome": "stored",
            "reason": "sleep_ingestion",
            "stage": "sleep_api_mem0_add",
            "input": payload.model_dump(mode="json"),
            "output": {"should_store": True, "memory_id": memory_id, "result": write_result["result"]},
            "memory": memory.model_dump(mode="json"),
            "result": write_result["result"],
        }
    )
    memory_debug.log_memory_lifecycle(
        memory_id,
        {
            "event": "created",
            "timestamp": _now_iso(),
            "score": memory.metadata.importance,
            "source": payload.source,
            "trace_id": request_id,
            "stage": "sleep_api",
            "user_name": payload.user_name,
            "agent_id": payload.agent_id,
            "agent_name": payload.agent_name,
        },
    )
    background_tasks.add_task(
        _update_sleep_profile_background,
        payload.user_id,
        memory_service,
        memory_evolution,
        memory_debug,
        request_id,
    )
    return SleepResponse(
        status="ok",
        memory_id=memory_id,
        profile_updated=True,
        summary={
            "duration": payload.sleep_duration,
            "deep_sleep": payload.deep_sleep_duration,
            "awake_count": payload.awake_count,
            "rem_sleep": payload.rem_sleep_duration,
        },
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    background_tasks: BackgroundTasks,
    memory_service: MemoryService = Depends(get_memory_service),
    memory_debug: MemoryDebugService = Depends(get_memory_debug),
    memory_evolution: MemoryEvolutionEngine = Depends(get_memory_evolution),
    agent_core: AgentCore = Depends(get_agent_core),
) -> ChatResponse:
    request_id = str(uuid.uuid4())
    core_result = await run_in_threadpool(
        agent_core.run_chat,
        payload.user_id,
        payload.message,
        payload.agent_id,
    )
    memory_context = core_result.memory_context
    intent = memory_context.intent
    memories = memory_context.retrieved_memories
    context_messages = memory_context.context_messages
    _log_retrieved_memories(memory_debug, memories, core_result.trace_id, payload.user_id, "chat_memory_retrieval")
    background_tasks.add_task(
        memory_evolution.reinforce_retrieved_memories,
        memories,
        memory_service,
        core_result.trace_id,
        memory_debug,
    )
    background_tasks.add_task(
        _store_memory_background,
        request_id,
        payload.user_id,
        payload.user_name,
        payload.agent_id,
        payload.agent_name,
        payload.message,
        memories,
        agent_core,
        memory_service,
        llm_service,
        memory_debug,
        core_result.trace_id,
    )
    memory_debug.log_chat_trace(
        _build_chat_trace(
            request_id,
            payload.user_id,
            payload.message,
            intent,
            memories,
            context_messages,
            core_result.response,
            core_result.latency["retrieval_ms"],
            core_result.latency["llm_ms"],
            core_result.latency["total_ms"],
            core_result.understanding.model_dump(),
            core_result.trigger_decision.model_dump(),
            core_result.trigger_result,
            core_result.system_bus.model_dump(),
            core_result.trace_id,
        )
    )
    memory_debug.log_agent_trace(core_result.trace)
    memory_debug.log_memory_ranking(
        {
            "trace_id": core_result.trace_id,
            "user_id": payload.user_id,
            "stage": "memory_ranking",
            "input": {
                "message": payload.message,
                "memory_count": len(memories),
                "user_name": payload.user_name,
                "agent_id": payload.agent_id,
                "agent_name": payload.agent_name,
            },
            "output": {"ranked_memory_ids": [str(memory.get("id")) for memory in memories if memory.get("id")]},
            "ranking": memories,
        }
    )
    if core_result.trigger_decision.action == "emit_event":
        memory_debug.log_event_trigger(
            {
                "trace_id": core_result.trace_id,
                "user_id": payload.user_id,
                "user_name": payload.user_name,
                "agent_id": payload.agent_id,
                "agent_name": payload.agent_name,
                "stage": "event_decider",
                "input": core_result.trace["stages"]["event_decider"]["input"],
                "output": core_result.trigger_decision.model_dump(),
                "trigger_result": core_result.trigger_result,
            }
        )
    return ChatResponse(
        request_id=request_id,
        user_id=payload.user_id,
        intent=AgentCore.intent_dict(intent),
        memory=None,
        response=core_result.response,
        memories=memories,
    )


@router.post("/memory/search", response_model=MemorySearchResponse)
async def search_memory(
    payload: MemorySearchRequest,
    memory_service: MemoryService = Depends(get_memory_service),
    memory_debug: MemoryDebugService = Depends(get_memory_debug),
    memory_evolution: MemoryEvolutionEngine = Depends(get_memory_evolution),
    agent_core: AgentCore = Depends(get_agent_core),
) -> MemorySearchResponse:
    retrieval = await run_in_threadpool(agent_core.retrieve_memories, payload.user_id, payload.query, payload.limit)
    results = retrieval["results"]
    trace_id = f"search:{payload.user_id}:{int(time.time())}"
    _log_retrieved_memories(memory_debug, results, trace_id, payload.user_id, "search_memory_retrieval")
    memory_debug.log_memory_ranking(
        {
            "trace_id": trace_id,
            "user_id": payload.user_id,
            "stage": "memory_ranking",
            "input": {"query": payload.query, "memory_count": len(results)},
            "output": {"ranked_memory_ids": [str(memory.get("id")) for memory in results if memory.get("id")]},
            "ranking": results,
        }
    )
    await run_in_threadpool(
        memory_evolution.reinforce_retrieved_memories,
        results,
        memory_service,
        trace_id,
        memory_debug,
    )
    return MemorySearchResponse(results=results)


@router.get("/memory/search")
async def search_memory_get(
    user_id: str = Query(min_length=1, max_length=128),
    query: str = Query(min_length=1, max_length=2000),
    limit: int = Query(default=10, ge=1, le=50),
    debug: bool = False,
    memory_service: MemoryService = Depends(get_memory_service),
    memory_debug: MemoryDebugService = Depends(get_memory_debug),
    memory_evolution: MemoryEvolutionEngine = Depends(get_memory_evolution),
    agent_core: AgentCore = Depends(get_agent_core),
) -> dict:
    retrieval = await run_in_threadpool(agent_core.retrieve_memories, user_id, query, limit, debug)
    intent = retrieval["intent"]
    if debug:
        debug_result = retrieval["debug_result"]
        return {
            "debug": True,
            "intent": AgentCore.intent_dict(debug_result["intent"]),
            "filters": debug_result["filters"],
            "selected": debug_result["selected"],
            "ranking": debug_result["ranking"],
            "rejected": debug_result["rejected"],
            "matched_memory_types": debug_result["matched_memory_types"],
            "excluded_memory_reasons": debug_result["excluded_memory_reasons"],
            "skip_reason": debug_result["skip_reason"],
        }
    results = retrieval["results"]
    trace_id = f"search:{user_id}:{int(time.time())}"
    _log_retrieved_memories(memory_debug, results, trace_id, user_id, "search_memory_retrieval")
    memory_debug.log_memory_ranking(
        {
            "trace_id": trace_id,
            "user_id": user_id,
            "stage": "memory_ranking",
            "input": {"query": query, "memory_count": len(results)},
            "output": {"ranked_memory_ids": [str(memory.get("id")) for memory in results if memory.get("id")]},
            "ranking": results,
        }
    )
    await run_in_threadpool(
        memory_evolution.reinforce_retrieved_memories,
        results,
        memory_service,
        trace_id,
        memory_debug,
    )
    return {"debug": False, "intent": AgentCore.intent_dict(intent), "results": results}


@router.post("/memory/add", response_model=MemoryAddResponse)
async def add_memory(
    payload: MemoryAddRequest,
    agent_core: AgentCore = Depends(get_agent_core),
    memory_debug: MemoryDebugService = Depends(get_memory_debug),
) -> MemoryAddResponse:
    request_id = str(uuid.uuid4())
    try:
        write_result = await run_in_threadpool(
            agent_core.store_memory_from_message,
            payload.user_id,
            payload.content,
            payload.agent_id,
            "manual",
            "user",
            [],
            _identity_metadata(
                payload.user_id,
                payload.user_name,
                payload.agent_id,
                payload.agent_name,
                payload.metadata,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if not write_result.should_store:
        memory_debug.log_memory_write(
            {
                "request_id": request_id,
                "trace_id": request_id,
                "user_id": payload.user_id,
                "user_name": payload.user_name,
                "agent_id": payload.agent_id,
                "agent_name": payload.agent_name,
                "outcome": "rejected",
                "reason": write_result.reason,
                "stage": "manual_agent_core_decision",
                "input": {
                    "content": payload.content,
                    "agent_id": payload.agent_id,
                    "user_name": payload.user_name,
                    "agent_name": payload.agent_name,
                },
                "output": {"should_store": False, "reason": write_result.reason},
            }
        )
        raise HTTPException(status_code=422, detail=f"Memory rejected by AgentCore: {write_result.reason}")

    memory_debug.log_memory_write(
        {
            "request_id": request_id,
            "trace_id": request_id,
            "user_id": payload.user_id,
            "user_name": payload.user_name,
            "agent_id": payload.agent_id,
            "agent_name": payload.agent_name,
            "outcome": "stored",
            "reason": write_result.reason,
            "stage": "manual_agent_core_store",
            "input": {
                "content": payload.content,
                "agent_id": payload.agent_id,
                "user_name": payload.user_name,
                "agent_name": payload.agent_name,
            },
            "output": {"should_store": True, "result": write_result.result},
            "memory": write_result.memory.model_dump() if write_result.memory else None,
            "result": write_result.result,
        }
    )
    if write_result.memory:
        memory_debug.log_memory_lifecycle(
            write_result.memory.id,
            {
                "event": "created",
                "timestamp": _now_iso(),
                "score": write_result.memory.metadata.importance,
                "source": "manual",
                "trace_id": request_id,
                "user_name": payload.user_name,
                "agent_id": payload.agent_id,
                "agent_name": payload.agent_name,
            },
        )
    return MemoryAddResponse(memory=write_result.memory, result=write_result.result)


@router.post("/memory/summary/run", response_model=SummaryRunResponse)
async def run_memory_summary(
    payload: SummaryRunRequest,
    memory_service: MemoryService = Depends(get_memory_service),
    llm_service: LLMService = Depends(get_llm_service),
    memory_debug: MemoryDebugService = Depends(get_memory_debug),
) -> SummaryRunResponse:
    return await _run_summary(payload.user_id, payload.force, payload.limit, memory_service, llm_service, memory_debug)


@router.get("/memory/agent")
async def get_agent_memories(
    user_id: str = Query(min_length=1, max_length=128),
    agent_id: str = Query(min_length=1, max_length=128),
    limit: int = Query(default=100, ge=1, le=1000),
    memory_service: MemoryService = Depends(get_memory_service),
) -> dict:
    memories = await run_in_threadpool(memory_service.get_agent_memories, user_id, agent_id, limit)
    return {
        "user_id": user_id,
        "agent_id": agent_id,
        "namespace": f"agent:{user_id}:{agent_id}",
        "memory_count": len(memories),
        "memories": memories,
    }


@router.get("/memory/agent/summary")
async def get_agent_summary(
    user_id: str = Query(min_length=1, max_length=128),
    agent_id: str = Query(min_length=1, max_length=128),
    limit: int = Query(default=200, ge=1, le=1000),
    memory_service: MemoryService = Depends(get_memory_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> dict:
    memories = await run_in_threadpool(memory_service.get_agent_memories, user_id, agent_id, limit)
    if not memories:
        return {
            "user_id": user_id,
            "agent_id": agent_id,
            "memory_count": 0,
            "summary": None,
            "reason": "no_agent_memories",
        }
    summary = await run_in_threadpool(llm_service.summarize_agent_memories, user_id, agent_id, memories)
    return {
        "user_id": user_id,
        "agent_id": agent_id,
        "memory_count": len(memories),
        "summary": summary,
    }


@router.post("/memory/agent/summary/run", response_model=AgentSummaryRunResponse)
async def run_agent_summary(
    payload: AgentSummaryRunRequest,
    memory_service: MemoryService = Depends(get_memory_service),
    llm_service: LLMService = Depends(get_llm_service),
    memory_debug: MemoryDebugService = Depends(get_memory_debug),
) -> AgentSummaryRunResponse:
    memories = await run_in_threadpool(memory_service.get_agent_memories, payload.user_id, payload.agent_id, payload.limit)
    source_memories = _agent_summary_source_memories(memories, include_summarized=payload.force)
    if not source_memories:
        return AgentSummaryRunResponse(
            created=False,
            reason="no_agent_memories",
            user_id=payload.user_id,
            agent_id=payload.agent_id,
        )
    created_summaries, results = await run_in_threadpool(
        _create_agent_summary_events_sync,
        payload.user_id,
        payload.agent_id,
        source_memories,
        memory_service,
        llm_service,
        memory_debug,
        "manual_agent_summary",
        payload.force,
    )
    if not created_summaries:
        return AgentSummaryRunResponse(
            created=False,
            reason="no_distinct_agent_events",
            user_id=payload.user_id,
            agent_id=payload.agent_id,
            source_memory_count=len(source_memories),
            summaries=[],
        )

    return AgentSummaryRunResponse(
        created=bool(created_summaries),
        reason="agent_interaction_summary_created",
        user_id=payload.user_id,
        agent_id=payload.agent_id,
        source_memory_count=len(source_memories),
        created_count=len(created_summaries),
        summaries=created_summaries,
        results=results,
    )


@router.post("/memory/import/chat", response_model=ChatImportResponse)
async def import_chat_history(
    payload: ChatImportRequest,
    memory_service: MemoryService = Depends(get_memory_service),
    llm_service: LLMService = Depends(get_llm_service),
    memory_debug: MemoryDebugService = Depends(get_memory_debug),
) -> ChatImportResponse:
    import_id = str(uuid.uuid4())
    ordered_messages = sorted(payload.messages, key=lambda item: item.timestamp)
    raw_memory_ids = []
    results = []

    if payload.store_raw:
        for index, message in enumerate(ordered_messages, start=1):
            content = _format_imported_message_content(payload, message)
            metadata = _imported_message_metadata(payload, message, import_id, index)
            result = await run_in_threadpool(
                memory_service.add_imported_chat_message,
                payload.user_id,
                payload.agent_id,
                content,
                metadata,
            )
            results.append(result)
            memory_id = _result_memory_id(result)
            if memory_id:
                raw_memory_ids.append(memory_id)
                memory_debug.log_memory_lifecycle(
                    memory_id,
                    {
                        "event": "created",
                        "timestamp": _now_iso(),
                        "score": float(metadata.get("importance", 0.35)),
                        "source": "local_chat_import",
                        "user_id": payload.user_id,
                        "agent_id": payload.agent_id,
                        "import_id": import_id,
                        "original_timestamp": metadata.get("timestamp"),
                        "speaker_role": metadata.get("speaker_role"),
                    },
                )

    event_summaries: list[dict] = []
    user_preferences: list[dict] = []
    if payload.summarize:
        summary_input = [
            _import_message_for_llm(payload, message, import_id, index)
            for index, message in enumerate(ordered_messages, start=1)
        ]
        imported_summary = await run_in_threadpool(
            llm_service.summarize_imported_chat_batch,
            payload.user_id,
            payload.agent_id,
            summary_input,
        )
        agent_events = _valid_agent_summary_events(imported_summary.get("agent_events", []), force=True)
        for event in agent_events:
            content = _format_agent_summary_event_content(payload.user_id, payload.agent_id, event)
            metadata = _agent_summary_event_metadata(payload.user_id, payload.agent_id, event)
            metadata.update(
                {
                    "import_id": import_id,
                    "source": "local_chat_import_summary",
                    "user_name": payload.user_name,
                    "agent_name": payload.agent_name,
                    "target_name": payload.agent_name,
                    "conversation_user_name": payload.user_name,
                    "conversation_agent_name": payload.agent_name,
                }
            )
            result = await run_in_threadpool(
                memory_service.add_agent_interaction_summary,
                payload.user_id,
                payload.agent_id,
                content,
                metadata,
            )
            results.append(result)
            memory_id = _result_memory_id(result)
            created = {**event, "content": content, "memory_id": memory_id}
            event_summaries.append(created)
            if memory_id:
                memory_debug.log_memory_lifecycle(
                    memory_id,
                    {
                        "event": "created",
                        "timestamp": _now_iso(),
                        "score": float(metadata.get("importance", 0.8)),
                        "source": "local_chat_import_summary",
                        "user_id": payload.user_id,
                        "agent_id": payload.agent_id,
                        "import_id": import_id,
                        "category": metadata.get("interaction_category"),
                        "time_range": metadata.get("time_range"),
                    },
                )

        for preference in _valid_imported_user_preferences(imported_summary.get("user_preferences", [])):
            content = _format_imported_user_preference_content(payload.user_id, preference)
            metadata = _imported_user_preference_metadata(payload, preference, import_id)
            result = await run_in_threadpool(
                memory_service.add_imported_user_preference,
                payload.user_id,
                content,
                metadata,
            )
            results.append(result)
            memory_id = _result_memory_id(result)
            created = {**preference, "content": content, "memory_id": memory_id}
            user_preferences.append(created)
            if memory_id:
                memory_debug.log_memory_lifecycle(
                    memory_id,
                    {
                        "event": "created",
                        "timestamp": _now_iso(),
                        "score": float(metadata.get("importance", 0.75)),
                        "source": "local_chat_import_user_preference",
                        "user_id": payload.user_id,
                        "import_id": import_id,
                        "category": metadata.get("preference_category"),
                        "time_range": metadata.get("time_range"),
                    },
                )

    return ChatImportResponse(
        status="ok",
        import_id=import_id,
        user_id=payload.user_id,
        agent_id=payload.agent_id,
        received_count=len(payload.messages),
        stored_raw_count=len(raw_memory_ids),
        created_event_summary_count=len(event_summaries),
        created_user_preference_count=len(user_preferences),
        raw_memory_ids=raw_memory_ids,
        event_summaries=event_summaries,
        user_preferences=user_preferences,
        results=results,
    )


@router.get("/memory/lifecycle/{user_id}")
async def memory_lifecycle(
    user_id: str,
    limit: int = Query(default=1000, ge=1, le=5000),
    memory_service: MemoryService = Depends(get_memory_service),
    memory_debug: MemoryDebugService = Depends(get_memory_debug),
) -> dict:
    memories = await run_in_threadpool(memory_service.get_all_memories, user_id, limit)
    recalled = memory_debug.recalled_counts()
    now_epoch = int(time.time())
    return _build_lifecycle_report(memories, recalled, now_epoch)


@router.get("/debug/prompt/{request_id}")
async def debug_prompt(
    request_id: str,
    memory_debug: MemoryDebugService = Depends(get_memory_debug),
) -> dict:
    trace = memory_debug.get_prompt_trace(request_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Request trace not found.")
    return trace


@router.get("/debug/stats")
async def debug_stats(memory_debug: MemoryDebugService = Depends(get_memory_debug)) -> dict:
    return memory_debug.stats()


@router.get("/debug/memory/explain/{memory_id}")
async def explain_memory(
    memory_id: str,
    memory_debug: MemoryDebugService = Depends(get_memory_debug),
    explainability: MemoryExplainabilityEngine = Depends(get_memory_explainability),
) -> dict:
    latest_ranking = memory_debug.latest_memory_ranking()
    memory = _find_memory_in_ranking(memory_id, latest_ranking)
    lifecycle = memory_debug.memory_lifecycle(memory_id)
    explanation = explainability.explain_memory(memory, lifecycle)
    explanation["memory_id"] = memory_id
    explanation["retrieval_status"] = "seen_in_latest_ranking" if memory else "not_seen_in_latest_ranking"
    explanation["latest_ranking_trace_id"] = (latest_ranking or {}).get("trace_id")
    return explanation


@router.get("/debug/memory/ranking")
async def debug_memory_ranking(
    user_id: str = Query(min_length=1, max_length=128),
    limit: int = Query(default=100, ge=1, le=1000),
    memory_service: MemoryService = Depends(get_memory_service),
    memory_debug: MemoryDebugService = Depends(get_memory_debug),
    explainability: MemoryExplainabilityEngine = Depends(get_memory_explainability),
) -> dict:
    memories = await run_in_threadpool(memory_service.get_all_memories, user_id, limit)
    ranked = explainability.explain_ranking_list(memories)
    trace_id = f"debug-ranking:{user_id}:{int(time.time())}"
    memory_debug.log_memory_ranking(
        {
            "trace_id": trace_id,
            "user_id": user_id,
            "stage": "debug_memory_ranking",
            "input": {"limit": limit, "memory_count": len(memories)},
            "output": {"ranked_memory_ids": [item["memory_id"] for item in ranked]},
            "ranking": memories,
        }
    )
    return {
        "trace_id": trace_id,
        "user_id": user_id,
        "ranked": ranked,
        "filtering_reasons": _filtering_reasons(memories),
    }


@router.get("/debug/agent/trace/{trace_id}")
async def debug_agent_trace(
    trace_id: str,
    memory_debug: MemoryDebugService = Depends(get_memory_debug),
) -> dict:
    trace = memory_debug.get_agent_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="AgentCore trace not found.")
    return trace


@router.get("/debug/events/{user_id}")
async def debug_events(
    user_id: str,
    memory_debug: MemoryDebugService = Depends(get_memory_debug),
) -> dict:
    events = memory_debug.events_for_user(user_id)
    return {
        "user_id": user_id,
        "event_count": len(events),
        "events": events,
    }


@router.post("/debug/memory/stability-test")
async def memory_stability_test(
    payload: MemoryStabilityTestRequest,
    agent_core: AgentCore = Depends(get_agent_core),
    llm_service: LLMService = Depends(get_llm_service),
    stability_engine: MemoryStabilityTestEngine = Depends(get_memory_stability),
) -> dict:
    return await run_in_threadpool(
        stability_engine.run,
        payload.user_id,
        payload.test_cases,
        payload.repeat,
        agent_core,
        llm_service,
    )


@router.post("/debug/memory/evolution/run")
async def run_memory_evolution(
    user_id: str,
    limit: int = Query(default=1000, ge=1, le=5000),
    memory_service: MemoryService = Depends(get_memory_service),
    memory_evolution: MemoryEvolutionEngine = Depends(get_memory_evolution),
    memory_debug: MemoryDebugService = Depends(get_memory_debug),
) -> dict:
    return await run_in_threadpool(memory_evolution.run_evolution_job, user_id, memory_service, limit, memory_debug)


@router.get("/debug/memory/profile/{user_id}")
async def get_memory_profile(
    user_id: str,
    memory_evolution: MemoryEvolutionEngine = Depends(get_memory_evolution),
) -> dict:
    profile = memory_evolution.get_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Personality profile not found.")
    return profile


@router.post("/diary/generate", response_model=DiaryGenerateResponse)
async def generate_diary(
    payload: DiaryGenerateRequest,
    memory_service: MemoryService = Depends(get_memory_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> DiaryGenerateResponse:
    since_epoch = int(time.time()) - 24 * 60 * 60
    memories = await run_in_threadpool(
        memory_service.memories_since,
        payload.user_id,
        since_epoch,
        payload.limit,
    )
    diary = await run_in_threadpool(llm_service.summarize_diary, memories, payload.user_id, payload.timezone)
    return DiaryGenerateResponse(
        user_id=payload.user_id,
        diary=diary,
        memory_count=len(memories),
        memories=memories,
    )


def _store_memory_background(
    request_id: str,
    user_id: str,
    user_name: str | None,
    agent_id: str | None,
    agent_name: str | None,
    message: str,
    retrieved_memories: list[dict],
    agent_core: AgentCore,
    memory_service: MemoryService,
    llm_service: LLMService,
    memory_debug: MemoryDebugService,
    trace_id: str | None = None,
) -> None:
    trace_id = trace_id or request_id
    try:
        write_result = agent_core.store_memory_from_message(
            user_id,
            message,
            agent_id,
            "chat",
            "user",
            retrieved_memories,
            {
                "context": {
                    "chat_history": [],
                    "retrieved_memories": retrieved_memories,
                },
                **_identity_metadata(user_id, user_name, agent_id, agent_name),
            },
        )
        if not write_result.should_store:
            memory_debug.log_memory_write(
                {
                    "request_id": request_id,
                    "trace_id": trace_id,
                    "user_id": user_id,
                    "user_name": user_name,
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "outcome": "rejected",
                    "reason": write_result.reason,
                    "stage": "agent_core_write_decision",
                    "input": {
                        "message": message,
                        "agent_id": agent_id,
                        "user_name": user_name,
                        "agent_name": agent_name,
                        "retrieved_memory_count": len(retrieved_memories),
                    },
                    "output": {"should_store": False, "reason": write_result.reason},
                    "decision": write_result.decision,
                }
            )
            logger.info("Memory write skipped for user_id=%s reason=%s", user_id, write_result.reason)
            return

        memory_debug.log_memory_write(
            {
                "request_id": request_id,
                "trace_id": trace_id,
                "user_id": user_id,
                "user_name": user_name,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "outcome": "stored",
                "reason": write_result.reason,
                "stage": "agent_core_mem0_add",
                "input": {
                    "message": message,
                    "agent_id": agent_id,
                    "user_name": user_name,
                    "agent_name": agent_name,
                    "retrieved_memory_count": len(retrieved_memories),
                },
                "output": {"should_store": True, "result": write_result.result},
                "memory": write_result.memory.model_dump() if write_result.memory else None,
                "result": write_result.result,
            }
        )
        if write_result.memory:
            memory_debug.log_memory_lifecycle(
                write_result.memory.id,
                {
                    "event": "created",
                    "timestamp": _now_iso(),
                    "score": write_result.memory.metadata.importance,
                    "source": "chat",
                    "trace_id": trace_id,
                    "user_name": user_name,
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                },
            )
            if agent_id:
                _maybe_auto_agent_summary(
                    user_id,
                    agent_id,
                    memory_service,
                    llm_service,
                    memory_debug,
                    trace_id,
                )
    except Exception:
        memory_debug.log_memory_write(
            {
                "request_id": request_id,
                "trace_id": trace_id,
                "user_id": user_id,
                "user_name": user_name,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "outcome": "error",
                "reason": "background_exception",
                "stage": "exception",
                "input": {
                    "message": message,
                    "agent_id": agent_id,
                    "user_name": user_name,
                    "agent_name": agent_name,
                    "retrieved_memory_count": len(retrieved_memories),
                },
                "output": {"error": "background_exception"},
            }
        )
        logger.exception("Background memory processing failed for user_id=%s", user_id)


def _update_sleep_profile_background(
    user_id: str,
    memory_service: MemoryService,
    memory_evolution: MemoryEvolutionEngine,
    memory_debug: MemoryDebugService,
    trace_id: str,
) -> None:
    try:
        profile = memory_evolution.update_sleep_profile(user_id, memory_service)
        memory_debug.log_memory_write(
            {
                "request_id": trace_id,
                "trace_id": trace_id,
                "user_id": user_id,
                "outcome": "updated",
                "reason": "sleep_profile_updated",
                "stage": "sleep_profile_update",
                "input": {"user_id": user_id},
                "output": profile,
            }
        )
    except Exception:
        memory_debug.log_memory_write(
            {
                "request_id": trace_id,
                "trace_id": trace_id,
                "user_id": user_id,
                "outcome": "error",
                "reason": "sleep_profile_update_failed",
                "stage": "sleep_profile_update",
                "input": {"user_id": user_id},
                "output": {"error": "background_exception"},
            }
        )
        logger.exception("Sleep profile update failed for user_id=%s", user_id)


async def _run_summary(
    user_id: str,
    force: bool,
    limit: int,
    memory_service: MemoryService,
    llm_service: LLMService,
    memory_debug: MemoryDebugService,
) -> SummaryRunResponse:
    should_run, reason, memories = await run_in_threadpool(memory_service.should_summarize, user_id, None, limit)
    if not force and not should_run:
        return SummaryRunResponse(created=False, reason=reason, source_memory_count=len(memories))
    if not memories:
        return SummaryRunResponse(created=False, reason="no_unarchived_memories", source_memory_count=0)

    start_epoch, end_epoch = _memory_time_range(memories)
    try:
        summary = await run_in_threadpool(llm_service.summarize_memories, user_id, memories, start_epoch, end_epoch)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    result = await run_in_threadpool(memory_service.add_summary_memory, summary)
    summary_memory_id = _result_memory_id(result)
    if summary_memory_id:
        memory_debug.log_memory_lifecycle(
            summary_memory_id,
            {
                "event": "created",
                "timestamp": _now_iso(),
                "score": 0.9,
                "source": "summary",
                "source_memory_count": len(memories),
            },
        )
    archived_ids = await run_in_threadpool(
        memory_service.archive_memories,
        memories,
        summary.time_range.model_dump(),
    )
    return SummaryRunResponse(
        created=True,
        reason="forced" if force else reason,
        summary=summary,
        source_memory_count=len(memories),
        archived_memory_ids=archived_ids,
        result=result,
    )


def _memory_time_range(memories: list[dict]) -> tuple[int, int]:
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


def _log_retrieved_memories(
    memory_debug: MemoryDebugService,
    memories: list[dict],
    trace_id: str,
    user_id: str,
    stage: str,
) -> None:
    for memory in memories:
        memory_id = memory.get("id")
        if not memory_id:
            continue
        memory_debug.log_memory_lifecycle(
            str(memory_id),
            {
                "event": "retrieved",
                "timestamp": _now_iso(),
                "score": float(memory.get("policy_score", memory.get("score", 0.0)) or 0.0),
                "trace_id": trace_id,
                "user_id": user_id,
                "stage": stage,
            },
        )


def _find_memory_in_ranking(memory_id: str, ranking_record: dict | None) -> dict | None:
    if not ranking_record:
        return None
    for memory in ranking_record.get("ranking", []):
        if str(memory.get("id")) == str(memory_id):
            return memory
    return None


def _filtering_reasons(memories: list[dict]) -> list[dict]:
    reasons = []
    for memory in memories:
        metadata = memory.get("metadata") or {}
        memory_id = str(memory.get("id", ""))
        if metadata.get("archived") is True or metadata.get("status") == "archived":
            reasons.append({"memory_id": memory_id, "filtered": True, "reason": "archived"})
        else:
            reasons.append({"memory_id": memory_id, "filtered": False, "reason": "eligible"})
    return reasons


def _result_memory_id(result) -> str | None:
    if isinstance(result, dict):
        candidate = result.get("id") or result.get("memory_id")
        if candidate:
            return str(candidate)
        results = result.get("results")
        if isinstance(results, list) and results:
            return _result_memory_id(results[0])
    if isinstance(result, list) and result:
        return _result_memory_id(result[0])
    return None


def _identity_metadata(
    user_id: str,
    user_name: str | None,
    agent_id: str | None,
    agent_name: str | None,
    base: dict | None = None,
) -> dict:
    metadata = dict(base or {})
    metadata.setdefault("speaker_role", "user")
    metadata.setdefault("speaker_id", user_id)
    metadata.setdefault("conversation_user_id", user_id)
    if user_name:
        metadata.setdefault("user_name", user_name)
        metadata.setdefault("speaker_name", user_name)
        metadata.setdefault("conversation_user_name", user_name)
    if agent_id:
        metadata.setdefault("target_role", "agent")
        metadata.setdefault("target_id", agent_id)
        metadata.setdefault("conversation_agent_id", agent_id)
    if agent_name:
        metadata.setdefault("agent_name", agent_name)
        metadata.setdefault("target_name", agent_name)
        metadata.setdefault("conversation_agent_name", agent_name)
    return metadata


def _valid_agent_summary_events(events: list[dict], force: bool = False) -> list[dict]:
    valid = []
    for event in events:
        if not isinstance(event, dict):
            continue
        summary = str(event.get("summary") or "").strip()
        title = str(event.get("title") or "").strip()
        category = str(event.get("category") or "other").strip() or "other"
        if not summary:
            continue
        if not force and len(summary) < 12 and not title:
            continue
        event["category"] = category
        event["title"] = title or category
        event["summary"] = summary
        valid.append(event)
    return valid


def _agent_summary_source_memories(memories: list[dict], include_summarized: bool = False) -> list[dict]:
    source = []
    for memory in memories:
        metadata = memory.get("metadata") or {}
        if metadata.get("summary_kind") == "agent_interaction_summary":
            continue
        if not include_summarized and metadata.get("agent_summary_batch_id"):
            continue
        source.append(memory)
    return source


def _should_auto_agent_summary(memories: list[dict]) -> tuple[bool, str]:
    if not memories:
        return False, "no_unsummarized_agent_memories"
    char_count = sum(len(str(memory.get("memory") or memory.get("content") or "")) for memory in memories)
    text = "\n".join(str(memory.get("memory") or memory.get("content") or "") for memory in memories)
    relationship_signals = (
        "吵架",
        "冷淡",
        "委屈",
        "生气",
        "调情",
        "暧昧",
        "拉扯",
        "哄我",
        "占有欲",
        "角色扮演",
        "主动一点",
        "你刚才",
        "我们刚才",
        "道歉",
        "解释",
        "误会",
        "flirt",
        "roleplay",
        "possessive",
    )
    has_signal = any(signal.lower() in text.lower() for signal in relationship_signals)
    if len(memories) >= 80:
        return True, "agent_memory_count_threshold"
    if char_count >= 3000:
        return True, "agent_char_count_threshold"
    if has_signal and char_count >= 800:
        return True, "agent_relationship_signal_threshold"
    return False, "threshold_not_met"


def _maybe_auto_agent_summary(
    user_id: str,
    agent_id: str,
    memory_service: MemoryService,
    llm_service: LLMService,
    memory_debug: MemoryDebugService,
    trace_id: str,
) -> None:
    memories = memory_service.get_agent_memories(user_id, agent_id, 1000)
    source_memories = _agent_summary_source_memories(memories)
    should_run, reason = _should_auto_agent_summary(source_memories)
    if not should_run:
        memory_debug.log_memory_write(
            {
                "request_id": trace_id,
                "trace_id": trace_id,
                "user_id": user_id,
                "agent_id": agent_id,
                "outcome": "skipped",
                "reason": reason,
                "stage": "auto_agent_summary_check",
                "input": {"source_memory_count": len(source_memories)},
                "output": {"should_summarize": False},
            }
        )
        return
    created_summaries, _ = _create_agent_summary_events_sync(
        user_id,
        agent_id,
        source_memories,
        memory_service,
        llm_service,
        memory_debug,
        reason,
        False,
    )
    memory_debug.log_memory_write(
        {
            "request_id": trace_id,
            "trace_id": trace_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "outcome": "created" if created_summaries else "skipped",
            "reason": reason if created_summaries else "no_distinct_agent_events",
            "stage": "auto_agent_summary",
            "input": {"source_memory_count": len(source_memories)},
            "output": {"created_count": len(created_summaries)},
        }
    )


def _create_agent_summary_events_sync(
    user_id: str,
    agent_id: str,
    source_memories: list[dict],
    memory_service: MemoryService,
    llm_service: LLMService,
    memory_debug: MemoryDebugService,
    reason: str,
    force: bool = False,
) -> tuple[list[dict], list]:
    summary = llm_service.summarize_agent_interaction_events(user_id, agent_id, source_memories)
    events = _valid_agent_summary_events(summary.get("events", []), force)
    if not events:
        return [], []

    batch_id = str(uuid.uuid4())
    results = []
    created_summaries = []
    for event in events:
        event = _ensure_agent_event_time_range(event, source_memories)
        content = _format_agent_summary_event_content(user_id, agent_id, event)
        metadata = _agent_summary_event_metadata(user_id, agent_id, event)
        metadata["agent_summary_batch_id"] = batch_id
        metadata["agent_summary_reason"] = reason
        result = memory_service.add_agent_interaction_summary(user_id, agent_id, content, metadata)
        results.append(result)
        memory_id = _result_memory_id(result)
        created = {**event, "content": content, "memory_id": memory_id}
        created_summaries.append(created)
        if memory_id:
            memory_debug.log_memory_lifecycle(
                memory_id,
                {
                    "event": "created",
                    "timestamp": _now_iso(),
                    "score": float(metadata.get("importance", 0.8)),
                    "source": "agent_interaction_summary",
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "category": metadata.get("interaction_category"),
                    "time_range": metadata.get("time_range"),
                    "agent_summary_batch_id": batch_id,
                    "reason": reason,
                },
            )
    _mark_agent_source_memories_summarized(memory_service, source_memories, batch_id, created_summaries)
    return created_summaries, results


def _ensure_agent_event_time_range(event: dict, source_memories: list[dict]) -> dict:
    if event.get("start_time") and event.get("end_time"):
        return event
    epochs = []
    iso_values = []
    source_ids = {str(item) for item in (event.get("source_memory_ids") or event.get("source_message_ids") or [])}
    for memory in source_memories:
        metadata = memory.get("metadata") or {}
        memory_id = str(memory.get("id") or metadata.get("id") or "")
        if source_ids and memory_id not in source_ids and str(metadata.get("import_message_id") or "") not in source_ids:
            continue
        timestamp = metadata.get("timestamp") or metadata.get("original_timestamp")
        if isinstance(timestamp, str) and timestamp:
            iso_values.append(timestamp)
            try:
                epochs.append(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp())
            except ValueError:
                continue
    if not iso_values and source_ids:
        return event
    if not iso_values:
        for memory in source_memories:
            metadata = memory.get("metadata") or {}
            timestamp = metadata.get("timestamp") or metadata.get("original_timestamp")
            if isinstance(timestamp, str) and timestamp:
                iso_values.append(timestamp)
                try:
                    epochs.append(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp())
                except ValueError:
                    continue
    if not iso_values:
        return event
    if epochs:
        start = datetime.fromtimestamp(min(epochs), timezone.utc).isoformat()
        end = datetime.fromtimestamp(max(epochs), timezone.utc).isoformat()
    else:
        start = iso_values[0]
        end = iso_values[-1]
    event.setdefault("start_time", start)
    event.setdefault("end_time", end)
    return event


def _mark_agent_source_memories_summarized(
    memory_service: MemoryService,
    source_memories: list[dict],
    batch_id: str,
    created_summaries: list[dict],
) -> None:
    summary_ids = [summary.get("memory_id") for summary in created_summaries if summary.get("memory_id")]
    now_epoch = int(time.time())
    for memory in source_memories:
        memory_id = memory.get("id")
        if not memory_id:
            continue
        metadata = dict(memory.get("metadata") or {})
        metadata["agent_summary_batch_id"] = batch_id
        metadata["agent_summary_created_at"] = now_epoch
        metadata["agent_summary_memory_ids"] = summary_ids
        memory_service.update_memory_metadata(str(memory_id), metadata)


def _format_agent_summary_event_content(user_id: str, agent_id: str, event: dict) -> str:
    category = str(event.get("category") or "other")
    title = str(event.get("title") or category)
    start_time = str(event.get("start_time") or "未知开始时间")
    end_time = str(event.get("end_time") or "未知结束时间")
    summary = str(event.get("summary") or "")
    preference_update = str(event.get("preference_update") or "").strip()
    follow_up = str(event.get("follow_up") or "").strip()
    parts = [
        f"AI 角色互动事件总结：{title}",
        f"用户 ID：{user_id}",
        f"AI 角色 ID：{agent_id}",
        f"事件分类：{category}",
        f"发生时间范围：{start_time} 至 {end_time}",
        f"事件总结：{summary}",
    ]
    if preference_update:
        parts.append(f"偏好更新：{preference_update}")
    if follow_up:
        parts.append(f"后续互动建议：{follow_up}")
    return "\n".join(parts)


def _agent_summary_event_metadata(user_id: str, agent_id: str, event: dict) -> dict:
    now = _now_iso()
    start_time = str(event.get("start_time") or "")
    end_time = str(event.get("end_time") or "")
    source_memory_ids = event.get("source_memory_ids") or event.get("source_message_ids")
    if not isinstance(source_memory_ids, list):
        source_memory_ids = []
    try:
        importance = float(event.get("importance", 0.8))
    except (TypeError, ValueError):
        importance = 0.8
    importance = max(0.0, min(1.0, importance))
    return {
        "timestamp": end_time or start_time or now,
        "importance": importance,
        "emotion": "neutral",
        "topic": "agent_interaction",
        "agent_id": agent_id,
        "speaker_role": "system",
        "speaker_id": "agent_interaction_summary",
        "target_role": "agent",
        "target_id": agent_id,
        "conversation_user_id": user_id,
        "conversation_agent_id": agent_id,
        "interaction_category": str(event.get("category") or "other"),
        "interaction_title": str(event.get("title") or ""),
        "time_range": {
            "start": start_time,
            "end": end_time,
        },
        "source_memory_ids": [str(memory_id) for memory_id in source_memory_ids],
        "preference_update": str(event.get("preference_update") or ""),
        "follow_up": str(event.get("follow_up") or ""),
    }


def _format_imported_message_content(payload: ChatImportRequest, message) -> str:
    sender_name = message.sender_name or _sender_default_name(payload, message.sender_role)
    timestamp = message.timestamp.isoformat()
    return "\n".join(
        [
            "导入的历史聊天原文",
            f"时间：{timestamp}",
            f"发言者：{message.sender_role}:{sender_name}",
            f"用户：{payload.user_name or payload.user_id}",
            f"AI 角色：{payload.agent_name or payload.agent_id}",
            f"内容：{message.content}",
        ]
    )


def _imported_message_metadata(payload: ChatImportRequest, message, import_id: str, index: int) -> dict:
    message_id = message.message_id or f"{import_id}:{index}"
    timestamp = message.timestamp.isoformat()
    sender_id = message.sender_id or (payload.user_id if message.sender_role == "user" else payload.agent_id)
    sender_name = message.sender_name or _sender_default_name(payload, message.sender_role)
    metadata = {
        "timestamp": timestamp,
        "importance": 0.35,
        "emotion": "neutral",
        "topic": "imported_chat",
        "source": payload.source,
        "import_id": import_id,
        "import_message_id": message_id,
        "import_index": index,
        "original_timestamp": timestamp,
        "user_name": payload.user_name,
        "agent_name": payload.agent_name,
        "conversation_user_id": payload.user_id,
        "conversation_user_name": payload.user_name,
        "conversation_agent_id": payload.agent_id,
        "conversation_agent_name": payload.agent_name,
    }
    if message.sender_role == "agent":
        metadata.update(
            {
                "speaker_role": "agent",
                "speaker_id": sender_id,
                "speaker_name": sender_name,
                "target_role": "user",
                "target_id": payload.user_id,
                "target_name": payload.user_name,
            }
        )
    elif message.sender_role == "system":
        metadata.update(
            {
                "speaker_role": "system",
                "speaker_id": sender_id or "system",
                "speaker_name": sender_name or "system",
                "target_role": "agent",
                "target_id": payload.agent_id,
                "target_name": payload.agent_name,
            }
        )
    else:
        metadata.update(
            {
                "speaker_role": "user",
                "speaker_id": sender_id,
                "speaker_name": sender_name,
                "target_role": "agent",
                "target_id": payload.agent_id,
                "target_name": payload.agent_name,
            }
        )
    return metadata


def _import_message_for_llm(payload: ChatImportRequest, message, import_id: str, index: int) -> dict:
    metadata = _imported_message_metadata(payload, message, import_id, index)
    return {
        "message_id": metadata["import_message_id"],
        "timestamp": metadata["timestamp"],
        "sender_role": metadata["speaker_role"],
        "sender_id": metadata["speaker_id"],
        "sender_name": metadata.get("speaker_name"),
        "content": message.content,
    }


def _sender_default_name(payload: ChatImportRequest, sender_role: str) -> str | None:
    if sender_role == "user":
        return payload.user_name
    if sender_role == "agent":
        return payload.agent_name
    return "system"


def _valid_imported_user_preferences(preferences: list[dict]) -> list[dict]:
    valid = []
    for preference in preferences:
        if not isinstance(preference, dict):
            continue
        summary = str(preference.get("summary") or "").strip()
        if not summary:
            continue
        preference["summary"] = summary
        preference["category"] = str(preference.get("category") or "other")
        valid.append(preference)
    return valid


def _format_imported_user_preference_content(user_id: str, preference: dict) -> str:
    category = str(preference.get("category") or "other")
    start_time = str(preference.get("start_time") or "未知开始时间")
    end_time = str(preference.get("end_time") or "未知结束时间")
    summary = str(preference.get("summary") or "")
    return "\n".join(
        [
            "导入聊天提取的用户偏好",
            f"用户 ID：{user_id}",
            f"偏好分类：{category}",
            f"证据时间范围：{start_time} 至 {end_time}",
            f"偏好总结：{summary}",
        ]
    )


def _imported_user_preference_metadata(payload: ChatImportRequest, preference: dict, import_id: str) -> dict:
    now = _now_iso()
    start_time = str(preference.get("start_time") or "")
    end_time = str(preference.get("end_time") or "")
    source_message_ids = preference.get("source_message_ids")
    if not isinstance(source_message_ids, list):
        source_message_ids = []
    try:
        importance = float(preference.get("importance", 0.75))
    except (TypeError, ValueError):
        importance = 0.75
    importance = max(0.0, min(1.0, importance))
    return {
        "timestamp": end_time or start_time or now,
        "importance": importance,
        "emotion": "neutral",
        "topic": "user_preference",
        "import_id": import_id,
        "user_name": payload.user_name,
        "agent_name": payload.agent_name,
        "speaker_role": "system",
        "speaker_id": "imported_chat_preference_summary",
        "subject_role": "user",
        "subject_id": payload.user_id,
        "subject_name": payload.user_name,
        "conversation_agent_id": payload.agent_id,
        "conversation_agent_name": payload.agent_name,
        "preference_category": str(preference.get("category") or "other"),
        "time_range": {
            "start": start_time,
            "end": end_time,
        },
        "source_message_ids": [str(message_id) for message_id in source_message_ids],
    }


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _build_chat_trace(
    request_id: str,
    user_id: str,
    user_input: str,
    intent,
    memories: list[dict],
    context_messages: list[dict[str, str]],
    llm_output: str,
    retrieval_ms: float,
    llm_ms: float,
    total_ms: float,
    understanding: dict | None = None,
    trigger_decision: dict | None = None,
    trigger_result: dict | None = None,
    system_bus: dict | None = None,
    trace_id: str | None = None,
) -> dict:
    prompt_context = _prompt_context_from_messages(context_messages, memories)
    return {
        "request_id": request_id,
        "trace_id": trace_id,
        "user_id": user_id,
        "user_input": user_input,
        "understanding": understanding or {},
        "trigger_decision": trigger_decision or {},
        "trigger_result": trigger_result or {},
        "system_bus": system_bus or {},
        "intent": asdict(intent),
        "retrieved_memories": [
            {
                "memory_id": str(memory.get("id", "")),
                "score": float(memory.get("policy_score", memory.get("score", 0.0)) or 0.0),
                "reason": (memory.get("explanation") or {}).get("reason", "selected_by_memory_policy"),
            }
            for memory in memories
        ],
        "prompt_context": prompt_context,
        "filtered_memories": [],
        "final_prompt": context_messages,
        "llm_output": llm_output,
        "latency": {
            "retrieval_ms": round(retrieval_ms, 3),
            "llm_ms": round(llm_ms, 3),
            "total_ms": round(total_ms, 3),
        },
    }


def _prompt_context_from_messages(context_messages: list[dict[str, str]], memories: list[dict]) -> dict:
    system_prompt = next((message["content"] for message in context_messages if message.get("role") == "system"), "")
    user_prompt = next((message["content"] for message in context_messages if message.get("role") == "user"), "")
    return {
        "system_prompt": system_prompt,
        "short_term": _extract_prompt_section(user_prompt, "短期对话:", "受控长期记忆:"),
        "long_term": [
            {
                "memory_id": memory.get("id"),
                "content": memory.get("memory") or memory.get("content"),
                "explanation": memory.get("explanation"),
            }
            for memory in memories
        ],
        "intent_summary": _extract_prompt_section(user_prompt, "用户意图摘要:", "短期对话:"),
    }


def _extract_prompt_section(prompt: str, start_marker: str, end_marker: str) -> list[str]:
    if start_marker not in prompt:
        return []
    section = prompt.split(start_marker, 1)[1]
    if end_marker in section:
        section = section.split(end_marker, 1)[0]
    lines = [line.strip() for line in section.strip().splitlines() if line.strip()]
    return lines


def _build_lifecycle_report(memories: list[dict], recalled: Counter, now_epoch: int) -> dict:
    by_type = Counter()
    by_importance = Counter()
    by_emotion = Counter()
    decay_distribution = Counter()
    archived = []

    for memory in memories:
        metadata = memory.get("metadata") or {}
        by_type[str(metadata.get("type", "unknown"))] += 1
        by_importance[str(metadata.get("importance", "unknown"))] += 1
        by_emotion[str(metadata.get("emotion", "unknown"))] += 1
        decay_distribution[_decay_bucket(metadata, memory, now_epoch)] += 1
        if metadata.get("archived") is True:
            archived.append(memory)

    memory_by_id = {str(memory.get("id")): memory for memory in memories if memory.get("id")}
    most_recalled = []
    for memory_id, count in recalled.most_common(10):
        memory = memory_by_id.get(memory_id)
        if memory:
            most_recalled.append(
                {
                    "memory_id": memory_id,
                    "recall_count": count,
                    "memory": memory.get("memory"),
                    "metadata": memory.get("metadata") or {},
                }
            )

    return {
        "total_memories": len(memories),
        "distribution": {
            "type": dict(by_type),
            "importance": dict(by_importance),
            "emotion": dict(by_emotion),
        },
        "decay_distribution": dict(decay_distribution),
        "most_frequently_recalled_memories": most_recalled,
        "forgotten_or_archived_memories": [
            {
                "memory_id": memory.get("id"),
                "memory": memory.get("memory"),
                "metadata": memory.get("metadata") or {},
            }
            for memory in archived[:50]
        ],
    }


def _decay_bucket(metadata: dict, memory: dict, now_epoch: int) -> str:
    epoch = MemoryService._memory_epoch(metadata, memory)
    if epoch is None:
        return "unknown"
    age_days = max(0, (now_epoch - epoch) / (24 * 60 * 60))
    if age_days < 7:
        return "fresh_lt_7d"
    if age_days <= 30:
        return "decayed_7_30d"
    return "strongly_decayed_gt_30d"
