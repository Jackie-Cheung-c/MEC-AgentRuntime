import os
from pathlib import Path

from dotenv import load_dotenv


# Load the project-level environment for every entry point (API, CLI, and evals).
# The API module also loads dotenv for backwards compatibility, but standalone
# commands import this module directly and must not depend on the API startup.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


LEGACY_ENV_PREFIX = "TRAVELOPS_"


def _getenv(name: str, default: str | None = None) -> str | None:
    """Prefer unprefixed settings, accepting legacy prefixed keys."""
    value = os.getenv(name)
    if value is not None:
        return value
    return os.getenv(f"{LEGACY_ENV_PREFIX}{name}", default)


DEEPSEEK_API_KEY = _getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = _getenv("DEEPSEEK_BASE_URL")

OPENAI_BASE_URL = _getenv("OPENAI_BASE_URL")
EMBEDDING_BASE_URL = _getenv("EMBEDDING_BASE_URL")


VLM_MODEL = _getenv("VLM_MODEL")
VLM_BASE_URL = _getenv("VLM_BASE_URL", OPENAI_BASE_URL)

CRITIC_MODEL = _getenv("CRITIC_MODEL", "deepseek-v4-flash")
CRITIC_BASE_URL = _getenv("CRITIC_BASE_URL", DEEPSEEK_BASE_URL)

MINIO_ENDPOINT = _getenv("MINIO_ENDPOINT", "127.0.0.1:9000")
MINIO_ACCESS_KEY = _getenv("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = _getenv("MINIO_SECRET_KEY", "")
# Existing object paths use this legacy bucket name; changing it would orphan data.
MINIO_BUCKET = _getenv("MINIO_BUCKET", "travelops-attachments")
MINIO_SECURE = (_getenv("MINIO_SECURE", "false") or "false").lower() in {
    "1",
    "true",
    "yes",
}
MAX_IMAGE_UPLOAD_BYTES = int(
    _getenv("MAX_IMAGE_UPLOAD_BYTES", str(10 * 1024 * 1024))
)
MAX_IMAGE_PIXELS = int(_getenv("MAX_IMAGE_PIXELS", "50000000"))
 
MODEL_CONTEXT_WINDOW_TOKENS = 521_000
CONTEXT_COMPRESSION_RATIO = 0.8
CONTEXT_COMPRESSION_THRESHOLD_TOKENS = int(
    MODEL_CONTEXT_WINDOW_TOKENS * CONTEXT_COMPRESSION_RATIO
)

LONG_TERM_MEMORY_BATCH_TURNS = int(
    _getenv("LONG_TERM_MEMORY_BATCH_TURNS", "3")
)
LONG_TERM_MEMORY_CONTEXT_TURNS = int(
    _getenv("LONG_TERM_MEMORY_CONTEXT_TURNS", "12")
)
LONG_TERM_MEMORY_RETRIEVAL_LIMIT = int(
    _getenv("LONG_TERM_MEMORY_RETRIEVAL_LIMIT", "12")
)
LONG_TERM_MEMORY_CHAR_BUDGET = int(
    _getenv("LONG_TERM_MEMORY_CHAR_BUDGET", "2400")
)

AUTH_SECRET_KEY = _getenv("AUTH_SECRET_KEY", "")
AUTH_TOKEN_TTL_SECONDS = int(
    _getenv("AUTH_TOKEN_TTL_SECONDS", str(7 * 24 * 60 * 60))
)
# Existing bearer tokens were issued under this legacy issuer value.
AUTH_ISSUER = _getenv("AUTH_ISSUER", "travelops")
AUTH_REQUIRED = (_getenv("AUTH_REQUIRED", "true") or "true").lower() in {
    "1",
    "true",
    "yes",
}

CHECKPOINT_DB_PATH = (
    _getenv("CHECKPOINT_DB_PATH")
    or str(PROJECT_ROOT / "data" / "checkpoints.db")
)

AGENT_TASK_POLL_INTERVAL_SECONDS = float(
    _getenv("AGENT_TASK_POLL_INTERVAL_SECONDS", "1.0")
)
AGENT_TASK_LEASE_SECONDS = int(_getenv("AGENT_TASK_LEASE_SECONDS", "10"))
AGENT_TASK_MAX_ATTEMPTS = int(_getenv("AGENT_TASK_MAX_ATTEMPTS", "3"))
AGENT_TASK_RETRY_DELAY_SECONDS = float(
    _getenv("AGENT_TASK_RETRY_DELAY_SECONDS", "5")
)
AGENT_TASK_RECOVERY_INTERVAL_SECONDS = float(
    _getenv("AGENT_TASK_RECOVERY_INTERVAL_SECONDS", "10")
)
AGENT_TASK_EVENT_POLL_INTERVAL_SECONDS = float(
    _getenv("AGENT_TASK_EVENT_POLL_INTERVAL_SECONDS", "0.25")
)
AGENT_TASK_RESULT_WAIT_TIMEOUT_SECONDS = float(
    _getenv("AGENT_TASK_RESULT_WAIT_TIMEOUT_SECONDS", "600")
)
