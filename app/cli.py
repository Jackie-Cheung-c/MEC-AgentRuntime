import asyncio
from src.agent.service import TravelOpsService
from src.domain import AgentRunRequest, RunStatus
import argparse

# USER_ID = "local_user"
# THREAD_ID = "1"


async def run_agent(user_id:str,thread_id:str) -> None:
    """Run the CLI through the same application service as the HTTP API."""
    service = TravelOpsService.with_defaults()
    print("TravelOps 智能旅行助手。输入 exit 或 quit 结束对话。")

    while True:
        try:
            user_input = input(">👨 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n对话结束。")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("对话结束。")
            break
        if not user_input:
            continue
        print("正在处理，请稍候...")
        result = await service.run(
            AgentRunRequest(
                user_id=user_id,
                thread_id=thread_id,
                query=user_input,
            )
        )

        if result.status == RunStatus.FAILED:
            print(f"🤖 Agent error: {result.error}")
        else:
            print(f"🤖 Agent: {result.answer}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="local_user", help="用户ID")
    parser.add_argument("--thread", default="1", help="会话ID")
    args = parser.parse_args()

    asyncio.run(run_agent(user_id=args.user, thread_id=args.thread))
