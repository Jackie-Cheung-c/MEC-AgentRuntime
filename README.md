# MEC-AgentRuntime

MEC-AgentRuntime 是一个面向通用任务的可恢复、可审计、可扩展 Agent 运行时。

它将对话编排、工具调用、Skill、业务状态、双层记忆、附件证据、质量审查和持久化任务队列统一在同一条执行链路中。平台核心不绑定具体行业，可用于客服支持、办公自动化、知识工作、业务办理、研究分析等需要多轮交互和外部工具的任务。
## 项目结构
```
app/
├── api/                    # FastAPI 入口、认证、能力目录和领域路由
├── web/                    # Agent 工作台静态资源
├── cli.py                  # 本地交互入口
└── worker.py               # 持久化任务 worker 入口

config/
└── settings.py             # 环境变量与运行配置

src/
├── agent/                  # 通用编排、依赖组装、状态注入和工具选择
├── attachments/            # 上传、VLM 提取与证据构建
├── auth/                   # 注册、登录、密码哈希与 Bearer Token
├── domain/                 # 请求、结果、轨迹、证据和状态契约
├── memory/                 # 长期记忆、FTS5 与 Chroma 向量索引
├── policies/               # 回答质量审查与重试策略
├── prompts/                # 系统提示词与领域 Prompt 资源
├── session_memory/         # 会话记忆、混合检索与上下文压缩
├── skills/                 # Skill 发现、加载和 UI 能力元数据
├── state_adapters/         # 业务状态读写适配器
├── storage/                # SQLite 仓储、任务队列和 MinIO 对象存储
├── tools/                  # 工具注册、MCP 与内置工具实现
├── tracing/                # LangChain 执行轨迹采集
├── travel/                 # 内置旅行领域示例
└── worker.py               # 任务领取、重试和恢复逻辑

data/                       # 默认 SQLite 与运行数据目录
```

## 统一入口位置：

- HTTP：`app/api/main.py`
- CLI：`app/cli.py`


## 注册示例：

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/auth/register `
  -H "Content-Type: application/json" `
  -d "{\"email\":\"alice@example.com\",\"password\":\"correct-horse-battery\",\"display_name\":\"Alice\"}"
```

登录后使用返回的 `access_token`：

```powershell
curl.exe http://127.0.0.1:8000/api/memories `
  -H "Authorization: Bearer <access_token>"
```

认证配置：

```dotenv
AUTH_SECRET_KEY=
AUTH_TOKEN_TTL_SECONDS=604800
AUTH_ISSUER=travelops
AUTH_REQUIRED=true
```
## 至少配置
```dotenv
DEEPSEEK_API_KEY=<model-api-key>
DEEPSEEK_BASE_URL=<openai-compatible-base-url>
AUTH_SECRET_KEY=<random-secret>
AUTH_REQUIRED=true
```

## 如需图片附件提取
```dotenv
MINIO_ENDPOINT=127.0.0.1:9000
MINIO_ACCESS_KEY=<minio-access-key>
MINIO_SECRET_KEY=<minio-secret-key>
MINIO_BUCKET=<>
MINIO_SECURE=false

VLM_MODEL=<vision-model>
VLM_BASE_URL=<vision-openai-compatible-base-url>
VLM_API_KEY=<vision-api-key>
```

## 本地启动MinIO
```powershell
docker compose up -d minio
```

## 运行

安装依赖并配置所需环境变量后，在仓库根目录运行：

```powershell
python -m app.cli
```

启动 Web 工作台：

```powershell
uvicorn app.api.main:app --reload
```

启动独立 Agent worker（建议至少与 Web 服务分开一个进程运行）：

```powershell
python -m app.worker
```
默认地址为 `http://127.0.0.1:8000`

## 扩展 Agent 
### 新增工具
1. 在 src/tools/ 中实现 LangChain Tool。
2. 在 src/tools/catalog.py 中注册稳定的 capability_id、标签和所需环境变量。
3. 在目标 Skill 的 tools 元数据中声明工具名称。
### 新增 Skill
在 src/skills/<skill-name>/SKILL.md 中添加元数据和领域
### 新增业务状态
实现 src/state_adapters/base.py 中的 StateAdapter，在运行前加载业务上下文，在运行后从执行轨迹中收集状态变化、事件、产物和待确认操作，然后在 src/agent/composition.py 中注册。
接入 MCP
在 src/tools/mcp/servers.py 中声明 MCP Server，通过 src/tools/mcp/manager.py 延迟加载工具。缺少依赖或连接失败时，能力目录会标记该能力不可用。