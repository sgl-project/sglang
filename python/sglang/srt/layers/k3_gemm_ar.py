"""K3 fused o_proj GEMM+AR dispatch (``SGLANG_K3_GEMM_AR``).

Glue between the model and ``kernels.ops.kimi_k3.gemm_ar``: lazily allocates
the P2P comm region on the TP group and swaps the o_proj RowParallelLinear
forward for the single fused GEMM+all-reduce kernel at decode shapes
(falling through to the regular GEMM + AR path whenever the input doesn't
fit — prefill, capture, non-2D input). Eager-mode only; see
kernels/ops/kimi_k3/GEMM_AR_README.md.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.linear import RowParallelLinear

logger = logging.getLogger(__name__)

_INITIALIZED = False
_ENABLED = False


def _init() -> bool:
    global _INITIALIZED, _ENABLED
    if _INITIALIZED:
        return _ENABLED
    _INITIALIZED = True
    if not envs.SGLANG_K3_GEMM_AR.get():
        return False
    from sglang.srt.runtime_context import get_parallel

    world_size = get_parallel().tp_size
    if not (2 <= world_size <= 8):
        logger.warning("SGLANG_K3_GEMM_AR requires 2 <= TP <= 8; disabled.")
        return False
    if torch.cuda.get_device_capability() < (10, 0):
        logger.warning("SGLANG_K3_GEMM_AR requires SM100+; disabled.")
        return False
    _ENABLED = True
    logger.info("K3 fused o_proj GEMM+AR enabled (world_size=%d)", world_size)
    return True


def maybe_wrap_o_proj(o_proj: RowParallelLinear) -> None:
    """SGLANG_K3_GEMM_AR: route decode-shaped o_proj calls through the fused
    GEMM+all-reduce kernel; everything else falls through to the original
    forward. The comm region + JIT compile happen HERE (model build, weight
    shape known, well before CUDA-graph capture) — capture must only see the
    ready-to-launch path."""
    if not _init():
        return
    from sglang.kernels.ops.kimi_k3 import gemm_ar as mod
    from sglang.srt.distributed.parallel_state import get_tp_group
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    world_size = parallel.tp_size
    if not o_proj.reduce_results or o_proj.tp_size != world_size:
        return
    weight = o_proj.weight
    if not (
        isinstance(weight, torch.Tensor)
        and weight.shape[0] == mod.N
        and weight.shape[1] % 128 == 0
    ):
        return

    mod.init(
        world_size=world_size,
        rank=parallel.tp_rank,
        group=get_tp_group().cpu_group,
        k=weight.shape[1],
    )
    # per-K compile + base-address stash, pre-capture
    mod._module_with_bases(weight.shape[1], world_size)

    inner = o_proj.forward

    def _gemm_ar_forward(x, *args, **kwargs):
        weight = o_proj.weight
        if (
            isinstance(weight, torch.Tensor)
            and weight.dtype == torch.bfloat16
            and mod.fits(x)
        ):
            return mod.o_proj_gemm_ar(x, weight), None
        return inner(x, *args, **kwargs)

    o_proj.forward = _gemm_ar_forward
