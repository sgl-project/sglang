"""Model specifications for sgl-router e2e tests.

Adapted from SMG's e2e_test/infra/model_specs.py. The same dict-of-dicts
shape (so test code reads the same) but the entries are narrower —
sgl-router tests today target small/medium models only; the larger
function-calling / reasoning models from SMG are out of scope.

Each entry:
    - model: HuggingFace path or local path (env-resolved)
    - memory_gb: estimated single-GPU footprint
    - tp: tensor-parallel size (= GPUs needed)
    - features: feature tags for filtering
    - worker_args: optional extra `sglang.launch_server` flags
"""

from __future__ import annotations

import os

# Local-cache root for CI / cluster nodes that pre-download HF weights.
# Mirrors the SMG `ROUTER_LOCAL_MODEL_PATH` env var.
ROUTER_LOCAL_MODEL_PATH = os.environ.get("ROUTER_LOCAL_MODEL_PATH", "")


def _resolve_model_path(hf_path: str) -> str:
    """Prefer a local copy of the model when one exists under
    ``ROUTER_LOCAL_MODEL_PATH``; otherwise fall back to the HuggingFace ID.
    """
    if ROUTER_LOCAL_MODEL_PATH:
        local_path = os.path.join(ROUTER_LOCAL_MODEL_PATH, hf_path)
        if os.path.exists(local_path):
            return local_path
    return hf_path


MODEL_SPECS: dict[str, dict] = {
    # Fast-start tiny model for convergence / decode-affinity / stale-request
    # tests. Single GPU, ~2 GB weights, sub-30s start on a warm cache.
    "qwen3-0.6b": {
        "model": _resolve_model_path("Qwen/Qwen3-0.6B"),
        "memory_gb": 4,
        "tp": 1,
        "features": ["chat", "streaming"],
    },
    # Standard small chat model — matches SMG's `llama-1b` entry.
    "llama-1b": {
        "model": _resolve_model_path("meta-llama/Llama-3.2-1B-Instruct"),
        "memory_gb": 4,
        "tp": 1,
        "features": ["chat", "streaming"],
    },
    # Primary 8B chat model — matches SMG's `llama-8b`.
    "llama-8b": {
        "model": _resolve_model_path("meta-llama/Llama-3.1-8B-Instruct"),
        "memory_gb": 16,
        "tp": 1,
        "features": ["chat", "streaming"],
    },
}


def get_model_spec(model_id: str) -> dict:
    """Return the spec dict for ``model_id``; KeyError if absent."""
    if model_id not in MODEL_SPECS:
        raise KeyError(
            f"Unknown model: {model_id}. Available: {list(MODEL_SPECS.keys())}"
        )
    return MODEL_SPECS[model_id]


def get_models_with_feature(feature: str) -> list[str]:
    """Filter model IDs by feature tag (e.g. ``streaming``, ``chat``)."""
    return [
        model_id
        for model_id, spec in MODEL_SPECS.items()
        if feature in spec.get("features", [])
    ]
