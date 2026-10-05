"""Opt-in route switch for the Cake kernel forwarding layer.

``SGLANG_CAKE_ROUTES`` selects which engine call sites may take a Cake
KernelSpec instead of the engine's default kernel.  The value is a
comma-separated list of route names (below) or ``all``.  A selected route is
still subject to the adapter's ``supports_<op>()`` admission at call time and
falls back to the default kernel when that admission fails, so enabling a route
never changes behaviour on devices or shapes the Cake kernel does not cover.

Importing this module loads no FlashInfer / CUDA / JIT code.
"""

from __future__ import annotations

import functools
import os

# Route name -> short description of the engine call site it switches.
ROUTES = {
    "gdn_prefill": "Gated Delta Net chunked prefill (attention.gdn_chunk_gated_delta_rule)",
    "gdn_decode": "Gated Delta Net decode on the state pool (attention.gdn_decode_pretranspose)",
    "kda_decode": "KDA recurrent decode (attention.kda_recurrent)",
    "moe_fp8_grouped": "FP8 block-scaled contiguous grouped expert GEMM (gemm.prepare_group_gemm_fp8_nt_groupwise_contiguous[_silu_quant])",
    "moe_nvfp4_warp_decode": "NVFP4 small-token MoE warp-decode runner (moe.warp_decode_*)",
    "kimi_k3_fp8_projection": "Kimi-K3 FP8 dense projection GEMMs (gemm.kimi_k3_fp8_projection)",
    "kimi_k3_mla": "Kimi-K3 FP8 MLA paged decode (attention.kimi_k3_mla_fp8_paged_attention)",
    # P1/P2 model routes (wiring + CPU route tests; model-level validation pending weights on a reachable route).
    "dsv3_grouped_routing": "DeepSeek-V3/R1/V3.2 grouped (node-limited) top-k routing (moe.fused_topk_deepseek)",
    "dsa_indexer": "DeepSeek-V3.2 lightning indexer logits + top-k (attention.sparse_mqa_logits / dsa_indexer_topk)",
    "dsv4_sparse_mla_decode": "DeepSeek-V4/Flash sparse MLA decode (attention.trtllm_batch_decode_sparse_mla_dsv4 / SM120 NVFP4 variants)",
    "msa_nvfp4_sparse_decode": "MiniMax-M3 NVFP4 sparse MSA decode (attention.msa_nvfp4_sparse_decode)",
    "mamba_ssu": "Mamba2 / Nemotron-H / granite selective state update decode (mamba.selective_state_update)",
    "mamba_ssd_prefill": "Mamba2 / Nemotron-H / granite SSD combined prefill (mamba.ssd_combined)",
    "sp_all_gather_matmul": "Sequence-parallel all-gather + matmul (communication.all_gather_matmul, Llama-3.1-70B SP)",
    "minimax_h3_diffusion": "MiniMax-H3 diffusion attention / pre-attention / projection stages (diffusion.minimax_h3_*)",
}

ENV_VAR = "SGLANG_CAKE_ROUTES"


@functools.lru_cache(maxsize=1)
def _selected() -> frozenset[str]:
    raw = os.environ.get(ENV_VAR, "")
    names = {part.strip() for part in raw.split(",") if part.strip()}
    if "all" in names:
        return frozenset(ROUTES)
    unknown = names - set(ROUTES)
    if unknown:
        raise ValueError(
            f"{ENV_VAR} names unknown Cake route(s) {sorted(unknown)}; "
            f"known routes: {sorted(ROUTES)} or 'all'"
        )
    return frozenset(names)


def cake_route_enabled(name: str) -> bool:
    """True when ``name`` is selected by ``SGLANG_CAKE_ROUTES``.

    Raises ``KeyError`` for a name that is not in ``ROUTES`` so a typo at a call
    site is caught by the unit tests rather than silently disabling the route.
    """
    if name not in ROUTES:
        raise KeyError(f"unknown Cake route {name!r}; known: {sorted(ROUTES)}")
    return name in _selected()


# Process-wide environment a selected route needs before torch.distributed or
# any symmetric-memory allocation runs in the engine's worker processes.
# ``sp_all_gather_matmul``: FlashInfer's Cake all-gather matmul allocates its
# symmetric scratch through torch's NVSHMEM symmetric-memory backend, and torch
# fixes the backend process-wide at the first symmetric allocation (the engine's
# custom all-reduce), so the backend must be chosen before the workers start.
ROUTE_PROCESS_ENV: dict[str, dict[str, str]] = {
    "sp_all_gather_matmul": {"TORCH_SYMMMEM": "NVSHMEM"},
}


def apply_route_process_env(environ=None) -> dict[str, str]:
    """Export the process environment the selected routes need.

    Called by the engine launcher before the scheduler processes are spawned so
    they inherit the values. Variables the user already set are left alone (the
    route then falls back with a logged reason if the value is incompatible).
    Returns the variables this call set.
    """
    environ = os.environ if environ is None else environ
    applied: dict[str, str] = {}
    for route, variables in ROUTE_PROCESS_ENV.items():
        if route not in _selected():
            continue
        for key, value in variables.items():
            if key in environ:
                continue
            environ[key] = value
            applied[key] = value
    return applied


def reset_cache_for_tests() -> None:
    _selected.cache_clear()
