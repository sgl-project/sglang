"""Cake MXFP8 MegaMoE EP16 (NVSHMEM dispatch + 2 fused kernels + combine) via FlashInfer.

FlashInfer entries: ``flashinfer.moe_ep.preprocess_cake_mxfp8_megamoe_ep16_weights(w13, w2)
-> CakeMxfp8MegaMoeEp16Weights`` and the session factory
``flashinfer.moe_ep.CakeMxfp8MegaMoeEp16(weights, topk_ids, *, process_group=None,
backend="cuda")`` (implementation ``flashinfer.experimental.cake_mxfp8_megamoe_ep16``,
JIT ``...cake_mxfp8_megamoe_ep16.jit`` with the manifest-driven CUDA module;
``backend="cute_dsl"`` is the CuTe-DSL flavour). Contract at FlashInfer
``46340689a5ab``: exact CC 10.3 (sm_103a); EXACTLY 16 EP ranks (``torch.distributed``
must be initialized; the group's world size must be 16; all ranks must pick the
same ``backend``, checked collectively); 512 global experts (32 local), top-k
8, hidden 3072, intermediate 5120, tokens per rank in {16, 32, 64}; rank-local
BF16 ``w13 [32, 10240, 3072]`` (gate rows then up rows) and ``w2 [32, 3072,
5120]`` are preprocessed into FP8 e4m3 + packed UE8M0 scales; ``topk_ids [T, 8]``
int64 is IMMUTABLE for the session (same ``data_ptr`` on every ``run``; at most
64 routes to any expert across ranks); ``session.run(hidden_states [T, 3072]
BF16, topk_ids, topk_weights [T, 8] f32, *, out=session.workspace_output)`` ->
BF16 ``[T, 3072]``. NVSHMEM symmetric memory is required.

Process-group requirement: this adapter does NOT own process groups. The
caller (``sglang.srt`` runtime integration) initializes ``torch.distributed``
with the 16-rank EP group and passes it as ``process_group``; construction
allocates all symmetric + scratch memory, runs the TMA setup kernel, then
``torch.cuda.synchronize`` + ``dist.barrier`` collectively.

CUDA graphs: NOT supported (``run()`` raises under capture). ``run()`` is
allocation-free (exactly 2 kernels); serialize on one stream; recreate the
session after 14,913,080 forwards (grid-counter epoch).

Not supported here: other EP sizes / geometries, mutable routing, SM100,
graph capture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from sglang.kernels.cake_kernels._support import SM103, cuda_tensor_on
from sglang.kernels.cake_kernels.moe_common import (
    contiguous_cuda,
    cuda_device_in,
    current_cuda_index,
    modules_available,
)

if TYPE_CHECKING:
    import torch
    import torch.distributed as dist

FI_MODULE = "flashinfer.moe_ep.cake_mxfp8_megamoe_ep16"
FI_JIT_MODULE = "flashinfer.experimental.cake_mxfp8_megamoe_ep16.jit"
ARCHS = (SM103,)
WORLD_SIZE = 16
LOCAL_EXPERTS = 32
NUM_EXPERTS = WORLD_SIZE * LOCAL_EXPERTS
TOP_K = 8
HIDDEN = 3072
INTERMEDIATE = 5120
SUPPORTED_TOKENS = (16, 32, 64)
MAX_ROUTES_PER_EXPERT = 64
MAX_LAUNCH_EPOCH = 14_913_080


def supports_mxfp8_megamoe_ep16_weights(w13: torch.Tensor, w2: torch.Tensor) -> bool:
    """Weight-preprocessing admission (rank-local BF16 shapes on an SM103 device); never raises."""
    import torch

    try:
        return (
            modules_available(FI_MODULE, FI_JIT_MODULE)
            and cuda_tensor_on(w13, ARCHS)
            and contiguous_cuda(
                w13,
                shape=(LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
                dtype=torch.bfloat16,
            )
            and contiguous_cuda(
                w2, shape=(LOCAL_EXPERTS, HIDDEN, INTERMEDIATE), dtype=torch.bfloat16
            )
            and w2.device == w13.device
        )
    except Exception:
        return False


def supports_mxfp8_megamoe_ep16(
    topk_ids: torch.Tensor,
    *,
    process_group: Optional[dist.ProcessGroup] = None,
) -> bool:
    """Session admission: SM103 device, module present, initialized 16-rank group, routed ``topk_ids``.

    Returns ``False`` (never raises) when ``torch.distributed`` is not
    initialized or the group is not exactly 16 ranks.
    """
    import torch
    import torch.distributed as dist

    try:
        if not modules_available(FI_MODULE, FI_JIT_MODULE):
            return False
        if not dist.is_available() or not dist.is_initialized():
            return False
        group = dist.group.WORLD if process_group is None else process_group
        if int(dist.get_world_size(group)) != WORLD_SIZE:
            return False
        if not cuda_device_in(current_cuda_index(), ARCHS):
            return False
        return (
            contiguous_cuda(topk_ids, dtype=torch.int64, ndim=2)
            and int(topk_ids.shape[0]) in SUPPORTED_TOKENS
            and int(topk_ids.shape[1]) == TOP_K
        )
    except Exception:
        return False


def preprocess_mxfp8_megamoe_ep16_weights(w13: torch.Tensor, w2: torch.Tensor):
    """Forward to FlashInfer; returns ``CakeMxfp8MegaMoeEp16Weights`` (setup path, torch ops)."""
    from flashinfer.moe_ep import preprocess_cake_mxfp8_megamoe_ep16_weights

    return preprocess_cake_mxfp8_megamoe_ep16_weights(w13, w2)


def create_mxfp8_megamoe_ep16_session(
    weights: Any,
    topk_ids: torch.Tensor,
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    backend: str = "cuda",
) -> Any:
    """Collective session factory (every rank of the 16-rank group must call it).

    Raises ``RuntimeError("torch.distributed must be initialized")`` without a
    process group; ``backend in {"cuda", "cute_dsl"}``.
    """
    from flashinfer.moe_ep import CakeMxfp8MegaMoeEp16

    return CakeMxfp8MegaMoeEp16(
        weights, topk_ids, process_group=process_group, backend=backend
    )


def get_mxfp8_megamoe_ep16_weights_class():
    from flashinfer.moe_ep import CakeMxfp8MegaMoeEp16Weights

    return CakeMxfp8MegaMoeEp16Weights
