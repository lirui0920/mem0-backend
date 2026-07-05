# AI 长期记忆聊天后端服务

这是一个基于 FastAPI + mem0 的 AI 聊天后端。服务会把用户消息写入 mem0 长期记忆，检索相关记忆后拼接上下文调用 LLM，并提供记忆检索、手动添加记忆和最近 24 小时日记生成接口。

## 技术栈

- Python 3.11
- FastAPI
- mem0ai
- Qdrant 本地持久化向量库
- DeepSeek / OpenAI 兼容 Chat Completions API
- Docker / Gunicorn / Uvicorn

## API

### `GET /dashboard`

内置轻量后端可视化界面。服务启动后可直接访问：

```text
http://127.0.0.1:8000/dashboard
```

服务器部署后替换为实际 IP 或域名，例如：

```text
http://your-server:8000/dashboard
```

Dashboard 支持查看记忆总览、搜索记忆、查看 agent 专属记忆、生成 agent 总结、上传睡眠记录，以及查看 debug ranking/events/profile。如果配置了 `SERVICE_API_KEY`，需要在页面左侧填写 `X-API-Key`。

### `POST /chat`

请求：

```json
{
  "user_id": "u_001",
  "message": "我最近在准备去杭州旅行"
}
```

流程：

1. 使用 Memory Policy Layer 对用户消息做 intent classification。
2. 按 intent 受控检索长期记忆，并构造紧凑上下文。
3. 立即调用 LLM 生成回复并返回。
4. 在后台异步执行 memory write filtering、tagging 和 mem0 写入。

后台写入时生成的 `memory` 示例：

```json
{
  "id": "8b7f5fd8-2d11-4e88-9a26-d53fd0898b1d",
  "user_id": "u_001",
  "agent_id": null,
  "namespace": "user:u_001",
  "type": "event",
  "content": "我最近在准备去杭州旅行",
  "embedding": null,
  "metadata": {
    "emotion": "happy",
    "importance": 0.5,
    "decay": 0.0,
    "feedback_weight": 0.0,
    "topic": "travel",
    "timestamp": "2026-07-03T12:00:00+00:00"
  }
}
```

### `POST /memory/search`

```json
{
  "user_id": "u_001",
  "query": "旅行计划",
  "limit": 10
}
```

### `POST /memory/add`

```json
{
  "user_id": "u_001",
  "content": "用户喜欢靠窗的位置",
  "metadata": {
    "tag": "preference"
  }
}
```

### `POST /memory/summary/run`

手动触发 memory summary，适合在 ECS 上用 cron 定时调用。

```json
{
  "user_id": "u_001",
  "force": false,
  "limit": 500
}
```

当 `force=false` 时，服务会在满足以下任一条件时生成 summary：

- 距离上一次 summary 已超过 `SUMMARY_INTERVAL_SECONDS`，默认 24 小时
- 未归档普通 memory 数量达到 `SUMMARY_MEMORY_BATCH_SIZE`，默认 100 条

### `POST /diary/generate`

```json
{
  "user_id": "u_001",
  "timezone": "Asia/Shanghai",
  "limit": 100
}
```

服务会读取写入时带有 `logged_epoch` 元数据且属于最近 24 小时的记忆，并调用 LLM 生成中文日记。

## Memory Tagging

`/chat` 和 `/memory/add` 在写入 mem0 前都会先调用 LLM 生成结构化标签，不使用硬编码规则。写入 mem0 的 metadata 包含：

- `emotion`: `happy` / `sad` / `angry` / `anxious` / `neutral`
- `type`: `chat` / `sleep` / `preference` / `event` / `summary`
- `namespace`: `user:{user_id}` / `agent:{user_id}:{agent_id}` / `summary:{user_id}`
- `importance`: `0.0` 到 `1.0`
- `decay`: 衰减权重，默认 `0.0`
- `feedback_weight`: 反馈权重，默认 `0.0`
- `topic`: 例如 `health`、`relationship`、`daily life`、`work` 等
- `timestamp`: 消息中明确出现的时间，或当前写入时间
- `memory_object`: 完整结构化 memory JSON
- `logged_epoch`: 服务端写入时的 Unix 时间，用于最近 24 小时日记筛选

迁移说明：旧的 `fact` 类型会在写入/更新时归一化为 `event`；旧的 `low` / `medium` / `high` importance 会归一化为 `0.2` / `0.5` / `0.9`。新写入的 memory 必须包含 `user_id`，并由服务层自动解析 `namespace`。

## Memory Control Layer

`app/services/memory_policy.py` 是 mem0 上方的控制层，不修改 mem0 core。它负责：

- Intent classification: `casual_chat`、`emotional_support`、`factual_question`、`relationship_context`、`memory_recall_request`
- 写入过滤：丢弃过短消息、无意义内容、非个人事实问题、纯回忆请求、重复/近重复和低价值闲聊
- 受控检索：不同 intent 使用不同 mem0 filters 和 top-k
- 重排：`importance_weight + semantic_similarity - recency_decay`
- 上下文构造：短期对话、受控长期记忆、用户意图摘要和 system prompt 分区拼接

`/chat` 的 memory 写入是后台任务，不阻塞用户响应。后台流程为：

```text
policy pre-filter -> duplicate check -> LLM tagging -> tagged-memory filter -> mem0.add
```

## Memory Observability

`app/services/memory_debug.py` 提供轻量 JSONL 调试日志，默认写入：

```bash
./storage/memory_debug_logs.jsonl
```

每次 `/chat` 都会生成 `request_id`，并记录：

- 用户输入
- intent classification
- 被选中的 memory、分数和原因
- 实际发送给 LLM 的 prompt
- LLM 输出
- retrieval / LLM / total latency

调试接口：

```bash
GET /memory/search?user_id=u_001&query=旅行&limit=10&debug=true
GET /memory/lifecycle/u_001
GET /debug/prompt/{request_id}
GET /debug/stats
POST /debug/memory/stability-test
POST /debug/memory/evolution/run?user_id=u_001
GET /debug/memory/profile/u_001
```

`/memory/search?debug=true` 会返回完整候选 ranking、top-k selected、top-k 外 rejected 以及 score breakdown：

```json
{
  "semantic_similarity": 0.72,
  "importance_weight": 0.9,
  "decay_penalty": 0.35,
  "intent_match_bonus": 0.0
}
```

`/memory/lifecycle/{user_id}` 可查看 memory 总量、type/importance/emotion 分布、decay 分布、最常被召回 memory 和 archived memory。

`/debug/memory/stability-test` 是开发诊断接口，不会走 `/chat` 的后台写入流程，因此不会污染记忆。它会重复执行 intent classification、controlled retrieval、prompt construction 和 LLM response，用于检测 intent、retrieval、ranking、response 是否稳定。

请求示例：

```json
{
  "user_id": "u_001",
  "test_cases": [
    "你还记得我喜欢什么吗？",
    "我最近状态怎么样？"
  ],
  "repeat": 5
}
```

返回包含 `test_summary`、`drift_analysis` 和每个 case 的逐次运行结果。

### Memory Evolution

`app/services/memory_evolution.py` 负责长期记忆动态，不修改 mem0 core：

- 检索命中后 reinforcement：`importance_score += 0.1`，最高 `1.0`
- 记录 `retrieval_count` 和 `last_accessed_epoch`
- `importance_score >= 0.8` 且 `retrieval_count >= 3` 时标记 `status=core_memory`
- 长时间未访问会按 7/30/90 天衰减 importance
- `importance_score < 0.2` 且 30 天未访问时标记 `status=archived`
- archived memory 从普通检索排除，只在 debug 模式可见
- 根据 evolved metadata 生成 personality profile

手动运行 evolution job：

```bash
curl -X POST "http://127.0.0.1:8000/debug/memory/evolution/run?user_id=u_001&limit=1000"
```

查看用户 profile：

```bash
curl http://127.0.0.1:8000/debug/memory/profile/u_001
```

## Memory Summary

服务会把一段时间内的未归档普通记忆压缩成一条 summary memory。summary 由 LLM 生成，包含：

- `daily_summary`
- `emotional_trend`
- `key_events`
- `new_user_preferences`
- `time_range.start` / `time_range.end`

summary 写入 mem0 时会带以下 metadata：

- `type=summary`
- `importance=high`
- `time_range`
- `summary_object`
- `archived=false`

原始 memory 不会删除，只会通过 mem0 `update` 标记：

```json
{
  "archived": true,
  "archived_at": 1783070400,
  "summary_time_range": {
    "start": "2026-07-02T12:00:00+00:00",
    "end": "2026-07-03T12:00:00+00:00"
  }
}
```

长期上下文检索会优先混入相关 summary memory，同时检索未归档普通 memory，避免旧原文重复挤占上下文。

## Memory Decay

检索时会先让 mem0 返回候选记忆，然后在发送给 LLM 前执行 decay 重排：

```text
final_score = importance_weight + relevance_score - time_decay_penalty
```

时间衰减规则：

- `< 7 天`: 不扣分
- `7-30 天`: 扣 `DECAY_MEDIUM_PENALTY`
- `> 30 天`: 扣 `DECAY_STRONG_PENALTY`

importance 权重：

- `low`: `0.2`
- `medium`: `0.5`
- `high`: `0.9`

summary memory 会额外获得 `SUMMARY_RETENTION_BOOST`，默认 `0.6`，因此长期摘要比同等相关度的旧普通 memory 更容易保留在上下文中。接口返回的 memory 会包含 `decay_score` 和 `score_components`，便于调试检索排序。

ECS cron 示例，每小时检查一次：

```bash
0 * * * * curl -s -X POST http://127.0.0.1:8000/memory/summary/run \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me" \
  -d '{"user_id":"u_001","force":false,"limit":500}' >/dev/null
```

## 环境变量

复制环境变量模板：

```bash
cp .env.example .env
```

至少需要配置：

```bash
DEEPSEEK_API_KEY=sk-your-deepseek-key
LLM_BASE_URL=https://api.deepseek.com
LLM_CHAT_MODEL=deepseek-v4-flash
MEM0_LLM_PROVIDER=openai
MEM0_LLM_MODEL=deepseek-v4-flash
MEM0_EMBEDDER_PROVIDER=huggingface
MEM0_EMBEDDER_MODEL=/root/bge-model
MEM0_EMBEDDER_DIMS=384
SERVICE_API_KEY=change-me
SUMMARY_INTERVAL_SECONDS=86400
SUMMARY_MEMORY_BATCH_SIZE=100
DECAY_MEDIUM_PENALTY=0.35
DECAY_STRONG_PENALTY=0.75
SUMMARY_RETENTION_BOOST=0.6
```

DeepSeek 使用 OpenAI-compatible 协议，所以代码仍复用 OpenAI SDK。embedding 默认使用本地 SentenceTransformer 模型目录 `/root/bge-model`，并以离线模式加载；如果目录不存在，服务会直接报错，不会从 HuggingFace 自动下载模型。

如果你的模型服务兼容 OpenAI API，例如通义千问百炼兼容模式，也可以把 `LLM_BASE_URL` 改成对应地址，并调整模型名。

## 本地运行

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

如果设置了 `SERVICE_API_KEY`，业务接口需要带请求头：

```bash
curl -X POST http://127.0.0.1:8000/memory/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me" \
  -d '{"user_id":"u_001","query":"旅行","limit":5}'
```

## Docker 运行

```bash
cp .env.example .env
docker compose up -d --build
docker compose logs -f
```

数据会持久化到宿主机 `./storage` 目录。

## 阿里云 ECS 部署步骤

以下以 Alibaba Cloud Linux / Ubuntu 为例：

1. 安装 Docker 和 Compose。

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```

2. 拉取代码并配置环境变量。

```bash
git clone <your-repo-url> memory-chat-service
cd memory-chat-service
cp .env.example .env
vim .env
```

3. 启动服务。

```bash
docker compose up -d --build
docker compose ps
```

4. 在 ECS 安全组放行 `8000` 端口，或使用 Nginx 反向代理到内网端口。

Nginx 示例：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

5. 建议生产环境开启 HTTPS，并设置强 `SERVICE_API_KEY`。如果服务暴露到公网，不要把 `/docs` 暴露在 `APP_ENV=production` 环境中，本项目默认已关闭生产环境文档页。

## 生产建议

- 使用云盘或 NAS 持久化 `storage` 目录，避免实例重建丢失记忆数据。
- 使用 systemd 或 Docker restart policy 保证进程自动恢复，本项目 docker-compose 已设置 `restart: unless-stopped`。
- 通过 SLB / Nginx 终止 HTTPS。
- 把 `.env` 放在服务器本地，不要提交到 Git。
- 根据并发量调整 Dockerfile 中 Gunicorn `--workers` 数量。
