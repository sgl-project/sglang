"""One-launch Kimi-K3 routing + MXFP8 quant with packed top-k words, for layers whose routed experts run on
DeepGEMM's MegaMoE (srt/layers/moe/k3_deepgemm_routed.py).

Stock device code (route_radix_block + per-token group quant, as in kimi_k3_fused_routing's first launch) with the
packed top-k output and padded rows masked; no trtllm-gen routing-metadata launch. The caller passes the output
tensors (the MegaMoE buffer's x / x_sf / top-k regions), so nothing is copied afterwards.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

# The next launch (MegaMoE) may start once this one has passed its own PDL wait
_EARLY_TRIGGER = 1


@cache_once
def _jit_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "k3_route_quant_packed",
        *args,
        cuda_files=["moe/k3_route_quant_packed.cuh"],
        cuda_wrappers=[("run", f"K3RouteQuantPackedKernel<{args}>::run")],
        # no fast-math: routing must stay bit-identical to route_radix
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def available() -> bool:
    import logging

    try:
        _jit_module()
        return True
    except Exception as e:  # pragma: no cover - toolchain dependent
        logging.getLogger(__name__).warning(
            f"Failed to load the JIT K3 packed route+quant kernel: {e}"
        )
        return False


def route_quant(
    scores: torch.Tensor,
    bias: torch.Tensor,
    x: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_scale: bool,
    out_packed: torch.Tensor,
    out_q: torch.Tensor,
    out_s: torch.Tensor,
    num_token_non_padded: Optional[torch.Tensor],
    out_wbuf: Optional[torch.Tensor] = None,
):
    """Returns ``(weights [M, topk] fp32, ids [M, topk] int32)``; the packed words,
    x_q and x_s are written into the given tensors. With ``out_wbuf`` (fp32
    ``[rows >= M, topk]``) the words carry bit 30 instead of bf16 weight bits and
    the fp32 weights go to ``out_wbuf``."""
    M = scores.shape[0]
    out_w = torch.empty((M, topk), dtype=torch.float32, device=scores.device)
    out_i = torch.empty((M, topk), dtype=torch.int32, device=scores.device)
    _jit_module().run(
        scores,
        bias,
        out_w,
        out_i,
        out_packed,
        x,
        out_q,
        out_s,
        num_token_non_padded,
        topk,
        float(routed_scaling_factor),
        bool(renormalize),
        bool(apply_scale),
        _EARLY_TRIGGER,
        out_wbuf,
    )
    return out_w, out_i
