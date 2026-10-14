import os
from pathlib import Path

from dotenv import load_dotenv


# Load the project-level environment for every entry point (API, CLI, and evals).
# The API module also loads dotenv for backwards compatibility, but standalone
# commands import this module directly and must not depend on the API startup.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")


VLM_MODEL = os.getenv("VLM_MODEL")
VLM_BASE_URL = os.getenv("VLM_BASE_URL", OPENAI_BASE_URL)

CRITIC_MODEL = os.getenv("CRITIC_MODEL", "deepseek-v4-flash")
CRITIC_BASE_URL = os.getenv("CRITIC_BASE_URL", DEEPSEEK_BASE_URL)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "travelops-attachments")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() in {"1", "true", "yes"}
MAX_IMAGE_UPLOAD_BYTES = int(os.getenv("MAX_IMAGE_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "50000000"))
 
MODEL_CONTEXT_WINDOW_TOKENS = 521_000
CONTEXT_COMPRESSION_RATIO = 0.8
CONTEXT_COMPRESSION_THRESHOLD_TOKENS = int(
    MODEL_CONTEXT_WINDOW_TOKENS * CONTEXT_COMPRESSION_RATIO
)

LONG_TERM_MEMORY_BATCH_TURNS = int(
    os.getenv("LONG_TERM_MEMORY_BATCH_TURNS", "3")
)
LONG_TERM_MEMORY_CONTEXT_TURNS = int(
    os.getenv("LONG_TERM_MEMORY_CONTEXT_TURNS", "12")
)
LONG_TERM_MEMORY_RETRIEVAL_LIMIT = int(
    os.getenv("LONG_TERM_MEMORY_RETRIEVAL_LIMIT", "12")
)
LONG_TERM_MEMORY_CHAR_BUDGET = int(
    os.getenv("LONG_TERM_MEMORY_CHAR_BUDGET", "2400")
)

AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "")
AUTH_TOKEN_TTL_SECONDS = int(
    os.getenv("AUTH_TOKEN_TTL_SECONDS", str(7 * 24 * 60 * 60))
)
AUTH_ISSUER = os.getenv("AUTH_ISSUER", "travelops")
AUTH_REQUIRED = os.getenv("AUTH_REQUIRED", "true").lower() in {
    "1",
    "true",
    "yes",
}

CHECKPOINT_DB_PATH = (
    os.getenv("CHECKPOINT_DB_PATH")
    or str(PROJECT_ROOT / "data" / "checkpoints.db")
)

AGENT_TASK_POLL_INTERVAL_SECONDS = float(
    os.getenv("AGENT_TASK_POLL_INTERVAL_SECONDS", "1.0")
)
AGENT_TASK_LEASE_SECONDS = int(os.getenv("AGENT_TASK_LEASE_SECONDS", "10"))
AGENT_TASK_MAX_ATTEMPTS = int(os.getenv("AGENT_TASK_MAX_ATTEMPTS", "3"))
AGENT_TASK_RETRY_DELAY_SECONDS = float(
    os.getenv("AGENT_TASK_RETRY_DELAY_SECONDS", "5")
)
AGENT_TASK_RECOVERY_INTERVAL_SECONDS = float(
    os.getenv("AGENT_TASK_RECOVERY_INTERVAL_SECONDS", "10")
)
AGENT_TASK_EVENT_POLL_INTERVAL_SECONDS = float(
    os.getenv("AGENT_TASK_EVENT_POLL_INTERVAL_SECONDS", "0.25")
)
AGENT_TASK_RESULT_WAIT_TIMEOUT_SECONDS = float(
    os.getenv("AGENT_TASK_RESULT_WAIT_TIMEOUT_SECONDS", "600")
)
