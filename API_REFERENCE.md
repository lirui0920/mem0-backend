# AI Memory Backend API Reference

版本：1.0  
框架：FastAPI + mem0 + Qdrant  
适用对象：前端开发者、iOS Shortcut 集成、外部 API 消费者、生产部署维护者

---

## 1. 系统概览

这是一个 AI 记忆后端系统，用于为 AI 助手提供长期记忆、上下文召回、事件触发、睡眠/健康信号识别、记忆演化和可观测性能力。

系统核心组件：

- **FastAPI**：HTTP API 服务层。
- **mem0**：长期记忆写入、检索与管理框架。
- **Qdrant**：向量检索存储，作为 mem0 的向量数据库后端。
- **AgentCore**：系统唯一的智能决策中心，负责记忆检索决策、上下文构造、意图理解、事件判断、记忆写入判断和回复生成。
- **MemoryRouter**：确定性格式化组件，只负责统一记忆 schema、namespace 解析和类型校验。
- **MemoryEvolutionEngine**：负责记忆评分、衰减、强化、排序稳定性。
- **MemoryDebug / Explainability**：提供 trace、记忆生命周期、评分解释和事件调试能力。

典型能力：

- 长期用户记忆
- agent 专属关系记忆
- summary 压缩记忆
- sleep / health / emotion 信号识别
- proactive event 触发
- 记忆强化与衰减
- Debug API 与可解释性追踪

---

## 2. 基础信息

### Base URL

本地默认：

```text
http://localhost:8000
```

生产环境请替换为实际部署域名。

### Content-Type

所有 JSON 请求建议使用：

```http
Content-Type: application/json
```

### OpenAPI 文档

非生产环境下，FastAPI 自动文档默认可用：

```text
GET /docs
GET /redoc
```

生产环境中如果 `APP_ENV=production`，`/docs` 和 `/redoc` 会被关闭。

---

## 3. 认证

除公开健康检查接口外，业务接口使用可选 API Key 认证。

如果服务端配置了环境变量：

```text
SERVICE_API_KEY=your-secret-key
```

则请求必须带：

```http
X-API-Key: your-secret-key
```

如果未配置 `SERVICE_API_KEY`，则当前实现不会强制校验 API Key。

认证失败响应：

```json
{
  "detail": "Invalid or missing API key."
}
```

HTTP 状态码：

```text
401 Unauthorized
```

---

## 4. 核心概念

### 4.1 AgentCore

AgentCore 是系统唯一的智能决策中心。

它负责：

- 决定是否检索记忆
- 决定记忆查询策略
- 构造最终 prompt context
- 理解用户输入
- 判断情绪、健康、睡眠等信号
- 决定是否触发事件
- 决定是否写入记忆
- 生成最终回复

AgentCore 内部按固定 pipeline 执行：

```text
MemoryPlanner
→ Memory Retrieval
→ ContextBuilder
→ ReasoningEngine
→ EventDecider
→ MemoryDecider
→ ResponseGenerator
```

### 4.2 Memory System

系统使用统一记忆模型，并将记忆按 namespace 路由。

Namespace 规则：

```text
user memory    → user:{user_id}
agent memory   → agent:{user_id}:{agent_id}
summary memory → summary:{user_id}
```

允许的记忆类型：

```text
chat | sleep | preference | event | summary
```

### 4.3 Event System

事件系统只执行 AgentCore 已经决定的事件，不做智能判断。

支持事件语义：

- `emotional_spike`
- `health_signal`
- `conversational_density`
- `proactive_message_trigger`

### 4.4 Memory Evolution System

MemoryEvolutionEngine 负责稳定的记忆生命周期管理：

- importance clamp
- feedback weight clamp
- event boost clamp
- decay 计算
- retrieval reinforcement
- ranking score 计算

最终排序分数由多项因素组成：

```text
importance * 0.4
+ similarity * 0.3
+ recency_bonus * 0.1
+ feedback_weight * 0.1
+ event_boost * 0.1
```

### 4.5 Observability System

系统会记录：

- `request_id`
- AgentCore `trace_id`
- memory ranking
- memory lifecycle
- event trigger logs
- prompt trace
- score breakdown

注意：当前 `/chat` HTTP 响应返回 `request_id`，AgentCore 内部会生成 `trace_id` 并写入 debug log。当前响应 schema 尚未直接返回 `trace_id`。

---

## 5. 数据模型

### 5.1 UnifiedMemory

```json
{
  "id": "string",
  "user_id": "string",
  "agent_id": "string | null",
  "namespace": "user:{user_id} | agent:{user_id}:{agent_id} | summary:{user_id}",
  "type": "chat | sleep | preference | event | summary",
  "content": "string",
  "embedding": ["number"] ,
  "metadata": {
    "timestamp": "ISO-8601 datetime",
    "importance": 0.5,
    "decay": 0.0,
    "feedback_weight": 0.0,
    "event_boost": 0.0
  }
}
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 记忆 ID，系统生成或由上游指定 |
| `user_id` | string | 必填，所有记忆都必须属于某个用户 |
| `agent_id` | string/null | agent 专属记忆使用；用户全局记忆为 null |
| `namespace` | string | 路由命名空间 |
| `type` | string | 记忆类型 |
| `content` | string | 记忆正文 |
| `embedding` | array/null | 通常由 mem0 / embedder 处理 |
| `metadata` | object | 评分、衰减、时间、调试等元数据 |

### 5.2 Metadata

```json
{
  "timestamp": "2026-07-04T10:00:00Z",
  "importance": 0.7,
  "decay": 0.0,
  "feedback_weight": 0.0,
  "event_boost": 0.0
}
```

约束：

| 字段 | 范围 | 说明 |
|---|---:|---|
| `importance` | `0.0 ~ 1.0` | 记忆重要性 |
| `decay` | `>= 0.0` | 衰减值 |
| `feedback_weight` | `-0.5 ~ 0.5` | 反馈强化权重 |
| `event_boost` | `0.0 ~ 0.3` | 事件增强权重 |

---

## 6. API Endpoints

### 6.1 Health Check

```http
GET /health
```

公开接口，不需要 API Key。

#### Response

```json
{
  "status": "ok",
  "app": "memory-chat-service"
}
```

---

## 7. Chat API

### POST `/chat`

主聊天接口。前端、iOS Shortcut 或外部消费者通常应优先使用此接口。

#### Request

```json
{
  "user_id": "user_123",
  "agent_id": "assistant_a",
  "message": "我最近总是凌晨两点才睡，白天很累。"
}
```

字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `user_id` | string | 是 | 用户 ID |
| `agent_id` | string/null | 否 | agent ID；用于 agent 专属记忆 |
| `message` | string | 是 | 用户输入，最长 8000 字符 |

#### Response

当前实现：

```json
{
  "request_id": "6a01d2c1-0d1e-4b7d-b3ef-3f0dd2a1a111",
  "user_id": "user_123",
  "intent": {
    "intent_type": "sleep",
    "emotion": "tired",
    "query": "我最近总是凌晨两点才睡，白天很累。"
  },
  "memory": null,
  "response": "听起来你最近睡眠节律被推得很晚，白天疲惫也很正常。我们可以先记录这个趋势，再一起看看有没有可以调整的点。",
  "memories": []
}
```

说明：

- `request_id`：HTTP 请求追踪 ID，可用于 `/debug/prompt/{request_id}`。
- `response`：AI 回复文本。
- `memories`：本轮召回并用于上下文的记忆。
- AgentCore 内部会生成 `trace_id` 并写入 debug log；当前 HTTP 响应未直接返回 `trace_id`。

#### 内部行为

```text
POST /chat
→ AgentCore.MemoryPlanner 决定是否检索记忆
→ MemoryService / mem0 / Qdrant 执行检索
→ ContextBuilder 合并 user / agent / summary memory
→ ReasoningEngine 理解意图和信号
→ EventDecider 判断是否触发事件
→ MemoryDecider 判断是否写入记忆
→ ResponseGenerator 调用 LLM 生成回复
→ MemoryEvolutionEngine 后台强化被召回记忆
→ DebugService 写入 trace / ranking / lifecycle
```

---

## 8. Sleep API

### POST `/sleep`

独立睡眠数据摄入接口，适合 Apple Shortcuts、Apple Watch、可穿戴设备或手动记录工具调用。

#### Request

```json
{
  "user_id": "user_123",
  "agent_id": "assistant_a",
  "sleep_start": "2026-07-04T00:30:00+08:00",
  "sleep_end": "2026-07-04T08:10:00+08:00",
  "sleep_duration": 7.67,
  "deep_sleep_duration": 2.1,
  "awake_count": 2,
  "rem_sleep_duration": 1.4,
  "source": "apple_watch"
}
```

字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `user_id` | string | 是 | 用户 ID |
| `agent_id` | string/null | 否 | 调用来源相关 agent ID；当前睡眠记忆仍强制写入 user namespace |
| `sleep_start` | datetime | 是 | 入睡开始时间，ISO-8601 |
| `sleep_end` | datetime | 是 | 醒来/睡眠结束时间，必须晚于 `sleep_start` |
| `sleep_duration` | float/null | 否 | 睡眠时长，单位小时；缺省时由 start/end 自动计算 |
| `deep_sleep_duration` | float/null | 否 | 深睡时长，单位小时 |
| `awake_count` | int/null | 否 | 醒来次数 |
| `rem_sleep_duration` | float/null | 否 | REM 睡眠时长，单位小时 |
| `source` | string | 是 | `apple_shortcuts`、`apple_watch` 或 `manual` |

#### Response

```json
{
  "status": "ok",
  "memory_id": "memory_uuid",
  "profile_updated": true,
  "summary": {
    "duration": 7.67,
    "deep_sleep": 2.1,
    "awake_count": 2,
    "rem_sleep": 1.4
  }
}
```

#### 内部行为

```text
POST /sleep
→ validate timestamps
→ compute sleep_duration if omitted
→ format sleep summary content
→ MemoryRouter route(namespace=user, type=sleep)
→ MemoryService store via mem0/Qdrant
→ write memory lifecycle/debug logs
→ background update sleep profile
```

生成的记忆内容类似：

```text
Sleep from 00:30 to 08:10.
Duration: 7.67h
Deep sleep: 2.10h
REM sleep: 1.40h
Awakenings: 2
Source: apple_watch
```

关键规则：

- `/sleep` 不调用 AgentCore。
- `/sleep` 不触发事件判断。
- `/sleep` 不写入 agent namespace，即使请求带了 `agent_id`。
- `/sleep` 强制写入 `user:{user_id}`。
- 记忆类型固定为 `sleep`。
- baseline importance 为 `0.6`。
- profile 更新为后台任务，不阻塞响应。

#### Sleep Profile

写入成功后，后台会更新用户 sleep profile：

- average sleep time
- average duration
- average deep sleep duration
- average REM sleep duration
- average awake count
- sleep consistency score

Profile 可通过以下接口读取：

```http
GET /debug/memory/profile/{user_id}
```

#### 通过 `/chat` 记录睡眠

自然语言睡眠描述仍然可以通过 `/chat` 进入系统：

```json
{
  "user_id": "user_123",
  "agent_id": "assistant_a",
  "message": "我昨晚 23:30 睡，早上 7:20 醒，大概睡了 7 小时 50 分钟。"
}
```

但 Apple Shortcuts / wearable 集成建议优先使用结构化 `/sleep`。

---

## 9. Memory APIs

### POST `/memory/add`

手动写入一条记忆。当前实现仍会经过 AgentCore 的写入判断，然后由 MemoryRouter 格式化为统一 schema。

#### Request

```json
{
  "user_id": "user_123",
  "agent_id": "assistant_a",
  "content": "I prefer gentle and concise replies from this assistant.",
  "metadata": {
    "type": "preference",
    "importance": 0.8,
    "source": "manual"
  }
}
```

字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `user_id` | string | 是 | 用户 ID |
| `agent_id` | string/null | 否 | agent ID |
| `content` | string | 是 | 记忆内容 |
| `metadata` | object | 否 | 附加元数据 |

#### Response

```json
{
  "memory": {
    "id": "memory_uuid",
    "user_id": "user_123",
    "agent_id": "assistant_a",
    "namespace": "agent:user_123:assistant_a",
    "type": "preference",
    "content": "I prefer gentle and concise replies from this assistant.",
    "embedding": null,
    "metadata": {
      "timestamp": "2026-07-04T10:00:00Z",
      "importance": 0.8,
      "decay": 0.0,
      "feedback_weight": 0.0,
      "event_boost": 0.0,
      "emotion": "neutral",
      "topic": "assistant_preference"
    }
  },
  "result": {
    "id": "mem0_memory_id"
  }
}
```

#### 可能错误

如果 AgentCore 判断不应写入：

```json
{
  "detail": "Memory rejected by AgentCore: duplicate_or_low_value"
}
```

HTTP 状态码：

```text
422 Unprocessable Entity
```

---

### POST `/memory/search`

按 query 搜索用户记忆。

#### Request

```json
{
  "user_id": "user_123",
  "query": "我的睡眠习惯是什么？",
  "limit": 10
}
```

#### Response

```json
{
  "results": [
    {
      "id": "memory_id",
      "memory": "User usually sleeps around 23:30.",
      "score": 0.82,
      "metadata": {
        "user_id": "user_123",
        "namespace": "user:user_123",
        "type": "sleep",
        "importance": 0.7
      },
      "policy_score": 0.74,
      "score_components": {
        "importance_weight": 0.7,
        "similarity_score": 0.82,
        "recency_bonus": 0.91,
        "bounded_feedback_weight": 0.1,
        "bounded_event_boost": 0.0,
        "decay_value": 0.09
      }
    }
  ]
}
```

---

### GET `/memory/search`

Query 参数版本，适合浏览器、iOS Shortcut 或简单集成。

#### Request

```http
GET /memory/search?user_id=user_123&query=我的睡眠习惯是什么&limit=10
```

可选 debug：

```http
GET /memory/search?user_id=user_123&query=我的睡眠习惯是什么&limit=10&debug=true
```

#### Response

普通模式：

```json
{
  "debug": false,
  "intent": {
    "intent_type": "memory_recall_request",
    "emotion": "neutral"
  },
  "results": []
}
```

Debug 模式：

```json
{
  "debug": true,
  "intent": {},
  "filters": {},
  "selected": [],
  "ranking": [],
  "rejected": [],
  "matched_memory_types": [],
  "excluded_memory_reasons": {},
  "skip_reason": null
}
```

---

### POST `/memory/summary/run`

手动触发 summary 记忆生成。

#### Request

```json
{
  "user_id": "user_123",
  "force": false,
  "limit": 500
}
```

#### Response

```json
{
  "created": true,
  "reason": "memory_count_threshold",
  "summary": {
    "user_id": "user_123",
    "daily_summary": "User recently discussed sleep and fatigue.",
    "emotional_trend": "tired but stable",
    "key_events": ["Mentioned late sleep several times"],
    "new_user_preferences": [],
    "time_range": {
      "start": "2026-07-03T00:00:00Z",
      "end": "2026-07-04T00:00:00Z",
      "start_epoch": 1783008000,
      "end_epoch": 1783094400
    }
  },
  "source_memory_count": 120,
  "archived_memory_ids": ["memory_1", "memory_2"],
  "result": {}
}
```

---

### GET `/memory/lifecycle/{user_id}`

查看某个用户的记忆生命周期统计。

#### Request

```http
GET /memory/lifecycle/user_123?limit=1000
```

#### Response

```json
{
  "total_memories": 42,
  "distribution": {
    "type": {
      "chat": 20,
      "sleep": 5,
      "preference": 8,
      "event": 7,
      "summary": 2
    },
    "importance": {},
    "emotion": {}
  },
  "decay_distribution": {
    "fresh_lt_7d": 30,
    "decayed_7_30d": 10,
    "strongly_decayed_gt_30d": 2
  },
  "most_frequently_recalled_memories": [],
  "forgotten_or_archived_memories": []
}
```

---

## 10. Debug / Observability APIs

Debug API 用于解释：

- 为什么某条记忆被选中
- 为什么某条记忆排名高或低
- 某条记忆经历了哪些生命周期事件
- AgentCore 每个 stage 的输入输出是什么
- 为什么触发了某个事件

生产环境建议限制访问权限。

---

### GET `/debug/memory/explain/{memory_id}`

解释某条记忆的评分、生命周期和最近一次检索状态。

#### Request

```http
GET /debug/memory/explain/memory_123
```

#### Response

```json
{
  "memory_id": "memory_123",
  "found": true,
  "final_score": 0.72,
  "score_breakdown": {
    "importance": 0.8,
    "similarity": 0.7,
    "recency_bonus": 0.9,
    "feedback_weight": 0.1,
    "event_boost": 0.0,
    "decay_penalty": 0.1
  },
  "decision_reason": [
    "high importance",
    "recently accessed",
    "low decay penalty"
  ],
  "content": "User usually sleeps late.",
  "metadata": {},
  "lifecycle": [
    {
      "event": "created",
      "timestamp": "2026-07-04T10:00:00Z",
      "score": 0.8
    },
    {
      "event": "retrieved",
      "timestamp": "2026-07-04T11:00:00Z",
      "score": 0.72
    }
  ],
  "retrieval_status": "seen_in_latest_ranking",
  "latest_ranking_trace_id": "trace_uuid"
}
```

---

### GET `/debug/memory/ranking`

查看某用户当前记忆排序与评分解释。

#### Request

```http
GET /debug/memory/ranking?user_id=user_123&limit=100
```

#### Response

```json
{
  "trace_id": "debug-ranking:user_123:1783094400",
  "user_id": "user_123",
  "ranked": [
    {
      "memory_id": "memory_123",
      "rank": 1,
      "final_score": 0.72,
      "score_breakdown": {
        "importance": 0.8,
        "similarity": 0.7,
        "recency_bonus": 0.9,
        "feedback_weight": 0.1,
        "event_boost": 0.0,
        "decay_penalty": 0.1
      },
      "decision_reason": ["high importance", "recently accessed"],
      "content": "User usually sleeps late.",
      "filtered": false
    }
  ],
  "filtering_reasons": [
    {
      "memory_id": "memory_123",
      "filtered": false,
      "reason": "eligible"
    }
  ]
}
```

---

### GET `/debug/agent/trace/{trace_id}`

查看 AgentCore 的完整 pipeline trace。

#### Request

```http
GET /debug/agent/trace/trace_uuid
```

#### Response

```json
{
  "event": "agent_trace",
  "trace_id": "trace_uuid",
  "user_id": "user_123",
  "stages": {
    "memory_planner": {
      "input": {},
      "output": {},
      "key_signals_used": ["message_length", "intent_type"]
    },
    "context_builder": {
      "input": {},
      "output": {},
      "key_signals_used": ["retrieved_memories", "namespace_kind", "summary_memory"]
    },
    "reasoning_engine": {
      "input": {},
      "output": {},
      "key_signals_used": ["sleep", "health_signal"]
    },
    "event_decider": {
      "input": {},
      "output": {},
      "key_signals_used": ["health_signal"]
    },
    "memory_decider": {
      "input": {},
      "output": {},
      "key_signals_used": ["understanding.should_store", "policy_duplicate_check", "importance"]
    },
    "response_generator": {
      "input": {},
      "output": {},
      "key_signals_used": ["final_prompt_context"]
    }
  }
}
```

如果找不到：

```json
{
  "detail": "AgentCore trace not found."
}
```

HTTP 状态码：

```text
404 Not Found
```

---

### GET `/debug/events/{user_id}`

查看某用户已触发事件。

#### Request

```http
GET /debug/events/user_123
```

#### Response

```json
{
  "user_id": "user_123",
  "event_count": 1,
  "events": [
    {
      "event": "event_trigger",
      "trace_id": "trace_uuid",
      "user_id": "user_123",
      "stage": "event_decider",
      "input": {},
      "output": {
        "action": "emit_event",
        "reason": "health_signal",
        "event_subtype": "health_signal",
        "confidence": 0.8
      },
      "trigger_result": {}
    }
  ]
}
```

---

### GET `/debug/prompt/{request_id}`

根据 `/chat` 返回的 `request_id` 查看 prompt 相关 trace。

#### Request

```http
GET /debug/prompt/6a01d2c1-0d1e-4b7d-b3ef-3f0dd2a1a111
```

#### Response

```json
{
  "request_id": "6a01d2c1-0d1e-4b7d-b3ef-3f0dd2a1a111",
  "trace_id": "trace_uuid",
  "prompt_context": {},
  "final_prompt": [],
  "retrieved_memories": [],
  "filtered_memories": []
}
```

---

### GET `/debug/stats`

查看 debug 日志聚合统计。

#### Response

```json
{
  "average_retrieval_time_ms": 12.4,
  "memory_write_count": 20,
  "memory_rejection_rate": 0.15,
  "most_common_intents": [["chat", 10]],
  "memory_growth_over_time": {
    "2026-07-04": 20
  },
  "chat_trace_count": 30,
  "memory_write_event_count": 25
}
```

---

### POST `/debug/memory/stability-test`

运行记忆稳定性测试。

#### Request

```json
{
  "user_id": "user_123",
  "test_cases": [
    "我喜欢辣的食物",
    "我最近睡得很晚"
  ],
  "repeat": 5
}
```

---

### POST `/debug/memory/evolution/run`

手动运行记忆演化任务。

#### Request

```http
POST /debug/memory/evolution/run?user_id=user_123&limit=1000
```

#### Response

```json
{
  "user_id": "user_123",
  "updated_count": 42,
  "promoted_count": 2,
  "demoted_count": 1,
  "promoted_memory_ids": [],
  "demoted_memory_ids": [],
  "personality_profile": {}
}
```

---

### GET `/debug/memory/profile/{user_id}`

读取 MemoryEvolutionEngine 生成的用户 profile。

#### Request

```http
GET /debug/memory/profile/user_123
```

#### Response

```json
{
  "user_id": "user_123",
  "generated_epoch": 1783094400,
  "dominant_emotion": "neutral",
  "stable_preferences": [],
  "recurring_topics": [],
  "behavioral_patterns": []
}
```

---

## 11. Diary API

### POST `/diary/generate`

根据最近记忆生成日记文本。

#### Request

```json
{
  "user_id": "user_123",
  "timezone": "Asia/Shanghai",
  "limit": 100
}
```

#### Response

```json
{
  "user_id": "user_123",
  "diary": "今天你多次提到了睡眠和疲惫，也表达了希望调整作息的想法。",
  "memory_count": 12,
  "memories": []
}
```

---

## 12. Event System 行为

事件由 AgentCore 判断，由 MemoryOrchestrator / EventSystem 执行。

### emotional_spike

当用户输入出现较强情绪信号时触发。

示例：

```text
我真的快撑不住了。
```

### health_signal

当用户输入涉及睡眠、健康、疲劳、失眠等信号时触发。

示例：

```text
我连续三天只睡了四个小时。
```

### conversational_density

当会话密度过高时触发，用于提示系统可能需要整理、总结或主动处理上下文。

### proactive_message_trigger

当系统判断存在主动关怀或提醒价值时触发。

事件调试：

```http
GET /debug/events/{user_id}
GET /debug/agent/trace/{trace_id}
```

---

## 13. Memory Lifecycle

记忆生命周期：

```text
created
→ retrieved
→ reinforced
→ decayed
→ summarized / archived
```

### creation

记忆由 `/chat` 后台写入、`/memory/add` 手动写入或 summary 任务生成。

### retrieval

用户请求触发记忆检索，系统按 namespace、type、语义相似度和策略过滤候选记忆。

### reinforcement

被召回的记忆会被强化，但有上限，避免无限放大：

```text
max reinforcement per memory per session = 3
```

### decay

长期未访问的记忆会产生衰减，但不会让 importance 变成负值或无限下降。

### summary conversion

当记忆数量或时间阈值满足条件时，系统可生成 summary memory，并归档旧记忆。

---

## 14. 内部架构数据流

```text
User / Frontend / iOS Shortcut
        ↓
      FastAPI
        ↓
    AgentCore
        ↓
 ┌───────────────┬─────────────────┐
 │               │                 │
MemoryRouter  EventSystem   ResponseGenerator
 │               │                 │
mem0 / Qdrant    │                 │
 │               ↓                 ↓
MemoryEvolutionEngine      Assistant Response
 │
BackgroundWorker / Summary / Decay
 │
Debug & Explainability Logs
```

更详细的聊天流程：

```text
POST /chat
↓
AgentCore.MemoryPlanner
↓
MemoryService.search → mem0 → Qdrant
↓
AgentCore.ContextBuilder
↓
AgentCore.ReasoningEngine
↓
AgentCore.EventDecider
↓
AgentCore.MemoryDecider
↓
AgentCore.ResponseGenerator
↓
HTTP Response
↓
BackgroundTasks:
  - reinforce retrieved memories
  - store new memory if approved
  - write debug lifecycle logs
```

---

## 15. 错误处理

### 401 Unauthorized

API Key 缺失或错误。

```json
{
  "detail": "Invalid or missing API key."
}
```

### 404 Not Found

资源不存在，例如 trace 不存在。

```json
{
  "detail": "AgentCore trace not found."
}
```

### 422 Unprocessable Entity

请求字段不合法，或记忆被 AgentCore 拒绝写入。

```json
{
  "detail": "Memory rejected by AgentCore: duplicate_or_low_value"
}
```

### 500 Internal Server Error

未捕获服务端异常。

```json
{
  "detail": "Internal server error."
}
```

---

## 16. iOS Shortcut 集成建议

### Chat 快捷指令

请求：

```http
POST /chat
Content-Type: application/json
X-API-Key: your-secret-key
```

Body：

```json
{
  "user_id": "ios_user",
  "agent_id": "shortcut_agent",
  "message": "我昨晚 1 点睡，今天很困。"
}
```

读取响应字段：

```text
response
```

### 睡眠记录快捷指令

建议直接调用 `/sleep`：

```json
{
  "user_id": "ios_user",
  "agent_id": "shortcut_agent",
  "sleep_start": "2026-07-04T01:00:00+08:00",
  "sleep_end": "2026-07-04T08:00:00+08:00",
  "sleep_duration": 7.0,
  "deep_sleep_duration": null,
  "awake_count": null,
  "rem_sleep_duration": null,
  "source": "apple_shortcuts"
}
```

---

## 17. 生产部署注意事项

建议：

- 设置 `SERVICE_API_KEY`
- 限制 Debug API 访问
- 持久化 `storage/qdrant`
- 持久化 `storage/memory_debug_logs.jsonl`
- 持久化 `storage/personality_profiles.json`
- 监控 LLM API 错误和延迟
- 对 `/chat`、`/memory/add` 增加速率限制
- 后续如需外部直接使用 AgentCore trace，建议将 `trace_id` 加入 `ChatResponse`

---

## 18. 当前实现与需求差异说明

以下是为了避免集成误用而明确记录的当前状态：

| 需求项 | 当前状态 |
|---|---|
| `POST /chat` | 已实现 |
| `POST /sleep` | 已实现 |
| `POST /memory/add` | 已实现 |
| `GET /memory/search` | 已实现 |
| `POST /memory/search` | 已实现 |
| Debug memory explain | 已实现 |
| Debug memory ranking | 已实现 |
| Debug AgentCore trace | 已实现 |
| Debug events | 已实现 |
| `/chat` 返回 `trace_id` | 当前未直接返回；内部已记录 |
