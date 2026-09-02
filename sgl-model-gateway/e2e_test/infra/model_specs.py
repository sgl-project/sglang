"""Model specifications for E2E tests.

Each model spec defines:
- model: HuggingFace model path or local path
- memory_gb: Estimated GPU memory required
- tp: Tensor parallelism size (number of GPUs needed)
- features: List of features this model supports (for test filtering)
"""

from __future__ import annotations

import os

# Environment variable for local model paths (CI uses local copies for speed)
ROUTER_LOCAL_MODEL_PATH = os.environ.get("ROUTER_LOCAL_MODEL_PATH", "")


def _resolve_model_path(hf_path: str) -> str:
    """Resolve model path, preferring local path if available."""
    if ROUTER_LOCAL_MODEL_PATH:
        local_path = os.path.join(ROUTER_LOCAL_MODEL_PATH, hf_path)
        if os.path.exists(local_path):
            return local_path
    return hf_path


MODEL_SPECS: dict[str, dict] = {
    # Primary chat model - used for most tests
    "llama-8b": {
        "model": _resolve_model_path("meta-llama/Llama-3.1-8B-Instruct"),
        "memory_gb": 16,
        "tp": 1,
        "features": ["chat", "streaming", "function_calling"],
    },
    # Small model for quick tests
    "llama-1b": {
        "model": _resolve_model_path("meta-llama/Llama-3.2-1B-Instruct"),
        "memory_gb": 4,
        "tp": 1,
        "features": ["chat", "streaming", "tool_choice"],
    },
    # Function calling specialist
    "qwen-7b": {
        "model": _resolve_model_path("Qwen/Qwen2.5-7B-Instruct"),
        "memory_gb": 14,
        "tp": 1,
        "features": ["chat", "streaming", "function_calling", "pythonic_tools"],
    },
    # Function calling specialist (larger, for Response API tests)
    "qwen-14b": {
        "model": _resolve_model_path("Qwen/Qwen2.5-14B-Instruct"),
        "memory_gb": 28,
        "tp": 2,
        "features": ["chat", "streaming", "function_calling", "pythonic_tools"],
        "worker_args": [
            "--context-length=1000"
        ],  # Faster startup, prevents memory issues
    },
    # Reasoning model
    "deepseek-7b": {
        "model": _resolve_model_path("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        "memory_gb": 14,
        "tp": 1,
        "features": ["chat", "streaming", "reasoning"],
    },
    # Thinking/reasoning model (larger)
    "qwen-30b": {
        "model": _resolve_model_path("Qwen/Qwen3-30B-A3B"),
        "memory_gb": 60,
        "tp": 4,
        "features": ["chat", "streaming", "thinking", "reasoning"],
    },
    # Mistral for function calling
    "mistral-7b": {
        "model": _resolve_model_path("mistralai/Mistral-7B-Instruct-v0.3"),
        "memory_gb": 14,
        "tp": 1,
        "features": ["chat", "streaming", "function_calling"],
    },
    # Embedding model
    "embedding": {
        "model": _resolve_model_path("intfloat/e5-mistral-7b-instruct"),
        "memory_gb": 14,
        "tp": 1,
        "features": ["embedding"],
    },
    # GPT-OSS model (Harmony)
    "gpt-oss": {
        "model": _resolve_model_path("openai/gpt-oss-20b"),
        "memory_gb": 40,
        "tp": 2,
        "features": ["chat", "streaming", "reasoning", "harmony"],
    },
}


def get_models_with_feature(feature: str) -> list[str]:
    """Get list of model IDs that support a specific feature."""
    return [
        model_id
        for model_id, spec in MODEL_SPECS.items()
        if feature in spec.get("features", [])
    ]


def get_model_spec(model_id: str) -> dict:
    """Get spec for a specific model, raising KeyError if not found."""
    if model_id not in MODEL_SPECS:
        raise KeyError(
            f"Unknown model: {model_id}. Available: {list(MODEL_SPECS.keys())}"
        )
    return MODEL_SPECS[model_id]


# Convenience groupings for test parametrization
CHAT_MODELS = get_models_with_feature("chat")
EMBEDDING_MODELS = get_models_with_feature("embedding")
REASONING_MODELS = get_models_with_feature("reasoning")
FUNCTION_CALLING_MODELS = get_models_with_feature("function_calling")


# =============================================================================
# Default model path constants (for backward compatibility with existing tests)
# =============================================================================

DEFAULT_MODEL_PATH = MODEL_SPECS["llama-8b"]["model"]
DEFAULT_SMALL_MODEL_PATH = MODEL_SPECS["llama-1b"]["model"]
DEFAULT_REASONING_MODEL_PATH = MODEL_SPECS["deepseek-7b"]["model"]
DEFAULT_ENABLE_THINKING_MODEL_PATH = MODEL_SPECS["qwen-30b"]["model"]
DEFAULT_QWEN_FUNCTION_CALLING_MODEL_PATH = MODEL_SPECS["qwen-7b"]["model"]
DEFAULT_MISTRAL_FUNCTION_CALLING_MODEL_PATH = MODEL_SPECS["mistral-7b"]["model"]
DEFAULT_GPT_OSS_MODEL_PATH = MODEL_SPECS["gpt-oss"]["model"]
DEFAULT_EMBEDDING_MODEL_PATH = MODEL_SPECS["embedding"]["model"]


# =============================================================================
# Third-party model configurations (cloud APIs)
# =============================================================================

THIRD_PARTY_MODELS: dict[str, dict] = {
    "openai": {
        "description": "OpenAI API",
        "model": "gpt-5-nano",
        "api_key_env": "OPENAI_API_KEY",
    },
    "xai": {
        "description": "xAI API",
        "model": "grok-4-fast",
        "api_key_env": "XAI_API_KEY",
    },
}
