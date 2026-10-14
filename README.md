# TravelOps

TravelOps 是具备持久化执行、幂等、证据链、审计轨迹的单机旅游 Agent 基础平台。 

## 项目结构

```text
app/
├── api/                    # FastAPI 接口
├── web/                    # 用户工作台静态资源
└── cli.py                  # 本地交互入口
scripts/                    # 数据检查和维护脚本
src/
├── agent/                  # 唯一的 TravelOps Agent 编排
├── auth/                   # 账号、密码哈希与 Token 校验
├── attachments/            # 图片上传、解析与证据提取
├── domain/                 # 附件、证据、行程状态和统一运行契约
├── planning/               # 行程规划与重规划
├── tools/                  # 天气、地图、费用和 MCP 工具
├── policies/               # 规则与模型辅助质量检查
├── tracing/                # 执行轨迹
├── storage/                # SQLite 生产存储与可自动清理的测试存储
├── evals/                  # 数据集、断言和故障注入
├── memory/                 # 跨会话长期记忆、FTS5 与 Chroma 索引
├── session_memory/         # 当前 user_id + thread_id 的持久化会话记忆
├── prompts/                # Prompt 资源
└── skills/                 # Agent Skill 资源
tests/                      # 单元测试
```

## 统一入口位置：

- HTTP：`app/api/main.py`
- CLI：`app/cli.py`

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
conda activate myvenv
默认地址为 `http://127.0.0.1:8000`


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
