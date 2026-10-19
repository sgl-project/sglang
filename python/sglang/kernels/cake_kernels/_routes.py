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


def reset_cache_for_tests() -> None:
    _selected.cache_clear()
