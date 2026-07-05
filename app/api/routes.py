import logging
import time
import uuid
from collections import Counter
from datetime import datetime
from dataclasses import asdict
from functools import lru_cache

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings, get_settings
from app.core.security import verify_api_key
from app.schemas import (
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
