"""Constants and enums for E2E test infrastructure."""

from enum import Enum, auto


class ConnectionMode(str, Enum):
    """Worker connection protocol."""

    HTTP = "http"
    GRPC = "grpc"


class WorkerType(str, Enum):
    """Worker specialization type."""

    REGULAR = "regular"
    PREFILL = "prefill"
    DECODE = "decode"


class Runtime(str, Enum):
    """Inference runtime/backend."""

    SGLANG = "sglang"
    VLLM = "vllm"
    OPENAI = "openai"
    XAI = "xai"
    GEMINI = "gemini"


# Convenience sets
LOCAL_MODES = frozenset({ConnectionMode.HTTP, ConnectionMode.GRPC})
LOCAL_RUNTIMES = frozenset({Runtime.SGLANG, Runtime.VLLM})
CLOUD_RUNTIMES = frozenset({Runtime.OPENAI, Runtime.XAI, Runtime.GEMINI})

# Fixture parameter names (used in @pytest.mark.parametrize)
PARAM_SETUP_BACKEND = "setup_backend"
PARAM_BACKEND_ROUTER = "backend_router"
PARAM_MODEL = "model"

# Default model
DEFAULT_MODEL = "llama-8b"

# Environment variable names
ENV_MODELS = "E2E_MODELS"
ENV_BACKENDS = "E2E_BACKENDS"
ENV_MODEL = "E2E_MODEL"
ENV_STARTUP_TIMEOUT = "E2E_STARTUP_TIMEOUT"
ENV_SKIP_MODEL_POOL = "SKIP_MODEL_POOL"
ENV_SKIP_BACKEND_SETUP = "SKIP_BACKEND_SETUP"
ENV_SHOW_ROUTER_LOGS = "SHOW_ROUTER_LOGS"
ENV_SHOW_WORKER_LOGS = "SHOW_WORKER_LOGS"

# Network
DEFAULT_HOST = "127.0.0.1"

# Timeouts (seconds)
DEFAULT_STARTUP_TIMEOUT = 300
DEFAULT_ROUTER_TIMEOUT = 60
HEALTH_CHECK_INTERVAL = 5
