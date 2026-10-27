from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from config import settings
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.agent.state_context import StateContextMiddleware
from src.prompts.prompt_manager import prompt_manager
from src.tools.catalog import load_enabled_tools
from src.agent.tool_selection import DynamicToolSelectionMiddleware

model = init_chat_model(
    model = "deepseek-v4-flash",
    model_provider="openai",
    api_key = settings.DEEPSEEK_API_KEY,
    base_url = settings.DEEPSEEK_BASE_URL,
    disabled_params={"prompt_cache_retention": None},
)

# 上下文压缩模型
compress_model = init_chat_model(
    model = "deepseek-v4-flash",
    model_provider="openai",
    api_key = settings.DEEPSEEK_API_KEY,
    base_url = settings.DEEPSEEK_BASE_URL,
)

# 审查模型
critic_model = init_chat_model(
    model = "deepseek-v4-flash",
    model_provider="openai",
    api_key = settings.DEEPSEEK_API_KEY,
    base_url = settings.DEEPSEEK_BASE_URL,
    disabled_params={"prompt_cache_retention": None},
)

_checkpointer: AsyncSqliteSaver | None = None
_checkpointer_connection: aiosqlite.Connection | None = None


async def get_default_checkpointer() -> AsyncSqliteSaver:
    """Return one process-wide durable SQLite checkpointer."""
    global _checkpointer, _checkpointer_connection
    if _checkpointer is None:
        _checkpointer_connection = await aiosqlite.connect(
            settings.CHECKPOINT_DB_PATH
        )
        _checkpointer = AsyncSqliteSaver(_checkpointer_connection)
        await _checkpointer.setup()
    return _checkpointer


async def close_default_checkpointer() -> None:
    global _checkpointer, _checkpointer_connection
    if _checkpointer_connection is not None:
        await _checkpointer_connection.close()
    _checkpointer = None
    _checkpointer_connection = None

# 读取系统提示词
SYSTEM_PROMPT = prompt_manager.get("System_prompt")

async def build_agent(
    *,
    primary_model=None,
    tools=None,
    checkpointer_instance=None,
):
    """Build the single universal Agent after the event loop is available."""
    if tools is None:
        tools = await load_enabled_tools()
        
    active_checkpointer = (
        checkpointer_instance
        if checkpointer_instance is not None
        else await get_default_checkpointer()
    )
    
    return create_agent(
        model=primary_model if primary_model is not None else model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            StateContextMiddleware(),
            DynamicToolSelectionMiddleware()],
        checkpointer=active_checkpointer,
    )

