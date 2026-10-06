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


# Every worker in this suite exists to answer a handful of 8–1024-token router
# assertions, so the engine's serving defaults are actively wrong here. At
# ``--mem-fraction-static 0.83`` the KV pool claims whatever the card has minus
# a slack of ``free_memory_at_dist_init * (1 - mem_fraction_static)`` — on an
# idle 80 GB H100 that is ~60 GB of KV (556K tokens for a 0.6B model) against
# ~13 GB of slack, and the activation working set for the default
# ``chunked_prefill_size=8192`` already accounts for nearly all of it. The
# prefill CUDA graph then has nothing left to capture its 58 num-token buckets
# into, dies part-way through with ``CUDA error: out of memory`` inside
# ``graph.capture_end()``, and the worker exits during startup — surfacing as a
# router e2e failure that has nothing to do with the router. Bounding the pool
# and skipping the prefill graph removes the whole class; it also cuts the
# per-spawn capture time, which this suite pays once per test.
CI_WORKER_ARGS: list[str] = [
    "--mem-fraction-static=0.6",
    "--cuda-graph-backend-prefill=disabled",
]


MODEL_SPECS: dict[str, dict] = {
    # Fast-start tiny model for convergence / decode-affinity / stale-request
    # tests. Single GPU, ~2 GB weights, sub-30s start on a warm cache.
    "qwen3-0.6b": {
        "model": _resolve_model_path("Qwen/Qwen3-0.6B"),
        "memory_gb": 4,
        "tp": 1,
        "features": ["chat", "streaming"],
        "worker_args": CI_WORKER_ARGS,
    },
    # Standard small chat model — matches SMG's `llama-1b` entry.
    "llama-1b": {
        "model": _resolve_model_path("meta-llama/Llama-3.2-1B-Instruct"),
        "memory_gb": 4,
        "tp": 1,
        "features": ["chat", "streaming"],
        "worker_args": CI_WORKER_ARGS,
    },
    # Primary 8B chat model — matches SMG's `llama-8b`.
    "llama-8b": {
        "model": _resolve_model_path("meta-llama/Llama-3.1-8B-Instruct"),
        "memory_gb": 16,
        "tp": 1,
        "features": ["chat", "streaming"],
        "worker_args": CI_WORKER_ARGS,
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
