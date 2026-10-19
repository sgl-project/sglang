"""Fused small-M NVFP4 MoE expert phase for Qwen3.5 decode verify on the trtllm-gen weight layout.

One launch computes the DynBlock-exact top-10 from the router logits, FC1, SwiGLU, the NVFP4 requantization of the
intermediate activation and FC2 for M <= 7 tokens (512 experts, hidden 4096, intermediate 128 per rank at TP8 or
256 at TP4; at TP4 a second engine takes 8..16 tokens and a third 17..21), and writes the trtllm deferred-finalize rows:
row ``t * 10 + k`` is expert ``k`` of token ``t``, identity permutation.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.jit.utils.compile.spec import resolve_sources

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

HIDDEN = 4096
NUM_EXPERTS = 512
TOP_K = 10
# (intermediate size per rank, largest token count of the engine) -> (JIT module name, launcher struct)
_ENGINES = {
    (128, 7): ("qwen_mkm_moe", "MkmMoe"),
    (256, 7): ("qwen_mkm_moe_tp4", "MkmMoeTp4"),
    (256, 16): ("qwen_mkm_moe_tp4m16", "MkmMoeTp4M16"),
    (256, 21): ("qwen_mkm_moe_tp4m21", "MkmMoeTp4M21"),
}
INTERMEDIATES = (128, 256)
MAX_TOKENS = 7
MAX_TOKENS_M16 = 16
MAX_TOKENS_M21 = 21


@cache_once
def _jit_module(intermediate: int, max_tokens: int) -> Module:
    name, launcher = _ENGINES[(intermediate, max_tokens)]
    module = load_jit(
        name,
        cuda_files=[f"moe/{name}.cuh"],
        cuda_wrappers=[
            ("run", f"{launcher}::run"),
            ("source_md5_lo", f"{launcher}::source_md5_lo"),
        ],
        # No fast-math: the routing softmax pins its approximate instructions explicitly.
        extra_cuda_cflags=["-O3"],
    )
    # The loaded module must come from the header on disk, not from a cached build of another tree.
    (header,) = resolve_sources([f"moe/{name}.cuh"])
    with open(header) as f:
        md5 = re.search(r"source md5 ([0-9a-f]{32})", f.read()).group(1)
    loaded = int(module.source_md5_lo())
    if loaded != int(md5[:15], 16):
        raise RuntimeError(
            f"{name}: loaded module was built from source {loaded:015x}, header {header} has {md5}"
        )
    logger.info("%s JIT module loaded: device code md5 %s (%s)", name, md5, header)
    return module


def mkm_moe(
    logits: torch.Tensor,
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    w13: torch.Tensor,
    w13_sf: torch.Tensor,
    w2: torch.Tensor,
    w2_sf: torch.Tensor,
    g1_alpha: torch.Tensor,
    g1_scale_c: torch.Tensor,
    g2_alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (gemm2_out bf16 [M * 10, 4096], expert_weights bf16 [M, 10]).

    logits [M, 512] bf16; x_fp4 [M, 2048] uint8 and x_sf [M, 256] uint8 (linear e4m3) as router_gemm_quant writes
    them; w13 [512, 2 I, 2048] / w2 [512, 4096, I / 2] uint8 (I = 128 or 256 per rank) and their e4m3 scales viewed
    as uint8, in the layout prepare_static_weights_for_trtllm_fp4_moe produces; g1_alpha / g1_scale_c / g2_alpha
    [512] fp32.
    """
    m = logits.shape[0]
    device = logits.device
    gemm2_out = torch.empty((m * TOP_K, HIDDEN), dtype=torch.bfloat16, device=device)
    expert_weights = torch.empty((m, TOP_K), dtype=torch.bfloat16, device=device)
    topk_ids = torch.empty((m, TOP_K), dtype=torch.int32, device=device)
    max_tokens = next(n for n in (MAX_TOKENS, MAX_TOKENS_M16, MAX_TOKENS_M21) if m <= n)
    _jit_module(w2.shape[2] * 2, max_tokens).run(
        logits,
        expert_weights,
        topk_ids,
        x_fp4,
        x_sf,
        w13,
        w13_sf,
        w2,
        w2_sf,
        g1_alpha,
        g1_scale_c,
        g2_alpha,
        gemm2_out,
    )
    return gemm2_out, expert_weights
