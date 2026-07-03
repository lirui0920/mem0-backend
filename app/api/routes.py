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
    SummaryRunRequest,
    SummaryRunResponse,
)
from app.services.llm_service import LLMService
from app.services.memory_debug import MemoryDebugService
from app.services.memory_evolution import MemoryEvolutionEngine
from app.services.memory_policy import MemoryPolicyLayer
from app.services.memory_stability import MemoryStabilityTestEngine
from app.services.memory_service import MemoryService

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


@public_router.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(status="ok", app=settings.app_name)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    background_tasks: BackgroundTasks,
    memory_service: MemoryService = Depends(get_memory_service),
    llm_service: LLMService = Depends(get_llm_service),
    memory_policy: MemoryPolicyLayer = Depends(get_memory_policy),
    memory_debug: MemoryDebugService = Depends(get_memory_debug),
    memory_evolution: MemoryEvolutionEngine = Depends(get_memory_evolution),
) -> ChatResponse:
    request_id = str(uuid.uuid4())
    total_start = time.perf_counter()
    intent = memory_policy.classify_intent(payload.message)
    retrieval_start = time.perf_counter()
    memories = await run_in_threadpool(
        memory_policy.retrieve_memories,
        payload.user_id,
        payload.message,
        intent,
        memory_service,
        8,
    )
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
    context_messages = memory_policy.build_context_messages(payload.user_id, payload.message, intent, memories)
    llm_start = time.perf_counter()
    response = await run_in_threadpool(llm_service.generate_response, context_messages)
    llm_ms = (time.perf_counter() - llm_start) * 1000
    memory_policy.record_turn(payload.user_id, payload.message, response)
    background_tasks.add_task(
        memory_evolution.reinforce_retrieved_memories,
        memories,
        memory_service,
    )
    background_tasks.add_task(
        _store_memory_background,
        request_id,
        payload.user_id,
        payload.message,
        memory_service,
        llm_service,
        memory_policy,
        memory_debug,
    )
    total_ms = (time.perf_counter() - total_start) * 1000
    memory_debug.log_chat_trace(
        _build_chat_trace(
            request_id,
            payload.user_id,
            payload.message,
            intent,
            memories,
            context_messages,
            response,
            retrieval_ms,
            llm_ms,
            total_ms,
        )
    )
    return ChatResponse(
        request_id=request_id,
        user_id=payload.user_id,
        intent=asdict(intent),
        memory=None,
        response=response,
        memories=memories,
    )


@router.post("/memory/search", response_model=MemorySearchResponse)
async def search_memory(
    payload: MemorySearchRequest,
    memory_service: MemoryService = Depends(get_memory_service),
    memory_policy: MemoryPolicyLayer = Depends(get_memory_policy),
    memory_evolution: MemoryEvolutionEngine = Depends(get_memory_evolution),
) -> MemorySearchResponse:
    intent = memory_policy.classify_intent(payload.query)
    results = await run_in_threadpool(
        memory_policy.retrieve_memories,
        payload.user_id,
        payload.query,
        intent,
        memory_service,
        payload.limit,
    )
    await run_in_threadpool(memory_evolution.reinforce_retrieved_memories, results, memory_service)
    return MemorySearchResponse(results=results)


@router.get("/memory/search")
async def search_memory_get(
    user_id: str = Query(min_length=1, max_length=128),
    query: str = Query(min_length=1, max_length=2000),
    limit: int = Query(default=10, ge=1, le=50),
    debug: bool = False,
    memory_service: MemoryService = Depends(get_memory_service),
    memory_policy: MemoryPolicyLayer = Depends(get_memory_policy),
    memory_evolution: MemoryEvolutionEngine = Depends(get_memory_evolution),
) -> dict:
    intent = memory_policy.classify_intent(query)
    if debug:
        debug_result = await run_in_threadpool(
            memory_policy.debug_retrieve_memories,
            user_id,
            query,
            intent,
            memory_service,
            limit,
        )
        return {
            "debug": True,
            "intent": asdict(debug_result["intent"]),
            "filters": debug_result["filters"],
            "selected": debug_result["selected"],
            "ranking": debug_result["ranking"],
            "rejected": debug_result["rejected"],
            "matched_memory_types": debug_result["matched_memory_types"],
            "excluded_memory_reasons": debug_result["excluded_memory_reasons"],
            "skip_reason": debug_result["skip_reason"],
        }
    results = await run_in_threadpool(
        memory_policy.retrieve_memories,
        user_id,
        query,
        intent,
        memory_service,
        limit,
    )
    await run_in_threadpool(memory_evolution.reinforce_retrieved_memories, results, memory_service)
    return {"debug": False, "intent": asdict(intent), "results": results}


@router.post("/memory/add", response_model=MemoryAddResponse)
async def add_memory(
    payload: MemoryAddRequest,
    memory_service: MemoryService = Depends(get_memory_service),
    llm_service: LLMService = Depends(get_llm_service),
    memory_policy: MemoryPolicyLayer = Depends(get_memory_policy),
    memory_debug: MemoryDebugService = Depends(get_memory_debug),
) -> MemoryAddResponse:
    request_id = str(uuid.uuid4())
    decision = memory_policy.should_store_message(payload.user_id, payload.content)
    if not decision.should_store:
        memory_debug.log_memory_write(
            {
                "request_id": request_id,
                "user_id": payload.user_id,
                "outcome": "rejected",
                "reason": decision.reason,
                "stage": "manual_pre_filter",
            }
        )
        raise HTTPException(status_code=422, detail=f"Memory rejected by policy: {decision.reason}")
    try:
        memory = await run_in_threadpool(llm_service.tag_memory, payload.user_id, payload.content)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    tagged_decision = memory_policy.should_store_tagged_memory(memory)
    if not tagged_decision.should_store:
        memory_debug.log_memory_write(
            {
                "request_id": request_id,
                "user_id": payload.user_id,
                "outcome": "rejected",
                "reason": tagged_decision.reason,
                "stage": "manual_tagged_filter",
                "memory": memory.model_dump(),
            }
        )
        raise HTTPException(status_code=422, detail=f"Memory rejected by policy: {tagged_decision.reason}")

    result = await run_in_threadpool(
        memory_service.add_structured_memory,
        memory,
        {"source": "manual", **payload.metadata},
    )
    memory_debug.log_memory_write(
        {
            "request_id": request_id,
            "user_id": payload.user_id,
            "outcome": "stored",
            "reason": tagged_decision.reason,
            "stage": "manual_mem0_add",
            "memory": memory.model_dump(),
            "result": result,
        }
    )
    await _try_auto_summary(payload.user_id, memory_service, llm_service)
    return MemoryAddResponse(memory=memory, result=result)


@router.post("/memory/summary/run", response_model=SummaryRunResponse)
async def run_memory_summary(
    payload: SummaryRunRequest,
    memory_service: MemoryService = Depends(get_memory_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> SummaryRunResponse:
    return await _run_summary(payload.user_id, payload.force, payload.limit, memory_service, llm_service)


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


@router.post("/debug/memory/stability-test")
async def memory_stability_test(
    payload: MemoryStabilityTestRequest,
    memory_service: MemoryService = Depends(get_memory_service),
    memory_policy: MemoryPolicyLayer = Depends(get_memory_policy),
    llm_service: LLMService = Depends(get_llm_service),
    stability_engine: MemoryStabilityTestEngine = Depends(get_memory_stability),
) -> dict:
    return await run_in_threadpool(
        stability_engine.run,
        payload.user_id,
        payload.test_cases,
        payload.repeat,
        memory_service,
        memory_policy,
        llm_service,
    )


@router.post("/debug/memory/evolution/run")
async def run_memory_evolution(
    user_id: str,
    limit: int = Query(default=1000, ge=1, le=5000),
    memory_service: MemoryService = Depends(get_memory_service),
    memory_evolution: MemoryEvolutionEngine = Depends(get_memory_evolution),
) -> dict:
    return await run_in_threadpool(memory_evolution.run_evolution_job, user_id, memory_service, limit)


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


async def _try_auto_summary(
    user_id: str,
    memory_service: MemoryService,
    llm_service: LLMService,
) -> None:
    try:
        await _run_summary(user_id, False, 500, memory_service, llm_service)
    except Exception:
        logger.exception("Automatic memory summary failed for user_id=%s", user_id)


def _store_memory_background(
    request_id: str,
    user_id: str,
    message: str,
    memory_service: MemoryService,
    llm_service: LLMService,
    memory_policy: MemoryPolicyLayer,
    memory_debug: MemoryDebugService,
) -> None:
    try:
        decision = memory_policy.should_store_message(user_id, message)
        if not decision.should_store:
            memory_debug.log_memory_write(
                {
                    "request_id": request_id,
                    "user_id": user_id,
                    "outcome": "rejected",
                    "reason": decision.reason,
                    "stage": "pre_filter",
                }
            )
            logger.info("Memory write skipped for user_id=%s reason=%s", user_id, decision.reason)
            return

        existing = memory_service.search_candidates(
            user_id,
            message,
            5,
            {"NOT": [{"archived": True}]},
        )
        decision = memory_policy.should_store_message(user_id, message, existing)
        if not decision.should_store:
            memory_debug.log_memory_write(
                {
                    "request_id": request_id,
                    "user_id": user_id,
                    "outcome": "rejected",
                    "reason": decision.reason,
                    "stage": "duplicate_filter",
                }
            )
            logger.info("Memory write skipped for user_id=%s reason=%s", user_id, decision.reason)
            return

        memory = llm_service.tag_memory(user_id, message)
        tagged_decision = memory_policy.should_store_tagged_memory(memory)
        if not tagged_decision.should_store:
            memory_debug.log_memory_write(
                {
                    "request_id": request_id,
                    "user_id": user_id,
                    "outcome": "rejected",
                    "reason": tagged_decision.reason,
                    "stage": "tagged_filter",
                    "memory": memory.model_dump(),
                }
            )
            logger.info("Tagged memory skipped for user_id=%s reason=%s", user_id, tagged_decision.reason)
            return

        result = memory_service.add_structured_memory(memory, {"source": "chat", "role": "user"})
        memory_debug.log_memory_write(
            {
                "request_id": request_id,
                "user_id": user_id,
                "outcome": "stored",
                "reason": tagged_decision.reason,
                "stage": "mem0_add",
                "memory": memory.model_dump(),
                "result": result,
            }
        )
        _run_summary_sync(user_id, False, 500, memory_service, llm_service)
    except Exception:
        memory_debug.log_memory_write(
            {
                "request_id": request_id,
                "user_id": user_id,
                "outcome": "error",
                "reason": "background_exception",
                "stage": "exception",
            }
        )
        logger.exception("Background memory processing failed for user_id=%s", user_id)


async def _run_summary(
    user_id: str,
    force: bool,
    limit: int,
    memory_service: MemoryService,
    llm_service: LLMService,
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


def _run_summary_sync(
    user_id: str,
    force: bool,
    limit: int,
    memory_service: MemoryService,
    llm_service: LLMService,
) -> SummaryRunResponse:
    should_run, reason, memories = memory_service.should_summarize(user_id, None, limit)
    if not force and not should_run:
        return SummaryRunResponse(created=False, reason=reason, source_memory_count=len(memories))
    if not memories:
        return SummaryRunResponse(created=False, reason="no_unarchived_memories", source_memory_count=0)

    start_epoch, end_epoch = _memory_time_range(memories)
    summary = llm_service.summarize_memories(user_id, memories, start_epoch, end_epoch)
    result = memory_service.add_summary_memory(summary)
    archived_ids = memory_service.archive_memories(memories, summary.time_range.model_dump())
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
) -> dict:
    prompt_context = _prompt_context_from_messages(context_messages, memories)
    return {
        "request_id": request_id,
        "user_id": user_id,
        "user_input": user_input,
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
