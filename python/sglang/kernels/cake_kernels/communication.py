"""Cake single-node tensor-parallel communication kernels via FlashInfer.

Four kernel families, all built for sm_100a / sm_103a and all multi-rank:

* **All-gather matmul** (``flashinfer.comm.all_gather_matmul(backend="cake")``
  -> ``flashinfer.comm.all_gather_matmul.cake_all_gather_matmul``). Push-wait
  all-gather of a local contiguous BF16/FP16 ``inp [M, 8192]`` (any ``M >
  0``; the kernel pads its tiles to 128 rows and masks the tail) fused with
  ``@ w`` into ``out [M * world_size, N]``. ``w`` is the logical ``[8192, N]``
  weight, ``N % 256 == 0``, either contiguous or the ``weight.t()`` view of
  the engine's contiguous ``[N, 8192]`` parameter (each layout has its own
  kernel; no transposed copy). Needs an initialised NCCL process group with
  ``world_size in {2, 4, 8}`` and the NVSHMEM
  ``torch.distributed._symmetric_memory`` backend for FlashInfer's internal
  symmetric scratch/flags (cached per device, group and dtype; the scratch
  grows to the largest ``M`` seen, and that growth and the first use
  synchronise; a failed collective poisons the state). The host side is
  nvcc-built from FlashInfer's ``csrc/cake_all_gather_matmul`` at first use.
  ``prepare_all_gather_matmul(backend="cake", max_rows=)`` binds the weight,
  the group and a row capacity in one collective and returns a replayable
  launcher that serves any contiguous input with up to ``max_rows`` rows; the
  callable re-validates dtype/device/rows, group identity and the weight
  fingerprint on every call and raises instead of falling back.
* **Fused norm combine** (``flashinfer.comm.cake_fused_norm_combine``, JIT
  ``flashinfer.jit.cake_fused_norm_combine``). Residual add + two-track RMSNorm
  + eight-peer BF16 combine in one launch per rank: ``x`` / ``residual`` /
  ``norm_out`` / ``residual_out`` BF16 ``[T, 2, 2560]``, ``weight`` BF16
  ``[2, 2560]``, ``collective_out`` BF16 ``[T, 2560]``, ``T <= max_tokens``.
  Exactly eight ranks on one node with CUDA IPC. The peer-mapped
  ``CakeFusedNormCombineWorkspace`` is created and destroyed collectively
  (same ``max_tokens`` and call order on every rank, outside CUDA-graph
  capture) and carries one ordered launch sequence.
* **MoE all-reduce union** (``flashinfer.comm.trtllm_moe_allreduce_fusion(
  backend="cake")``, JIT ``flashinfer.jit.cake_trtllm_moe_allreduce_union``).
  Expert-scaled reduction + token input + one-shot Lamport all-reduce +
  residual + RMSNorm. FP16/BF16 contiguous tensors, ``hidden_dim == 7168``,
  ``world_size in {2, 4, 8}``, ``token_num >= 1``, ``residual_out`` and
  ``norm_out`` required, no quantisation (``quant_out`` / ``scale_out`` /
  ``layout_code`` must be ``None``). ``workspace_ptrs`` is the int64 pointer
  table of a TRT-LLM IPC workspace
  (``trtllm_create_ipc_workspace_for_all_reduce_fusion``) with at least
  ``3 * world_size + 1`` entries.
* **MoE finalize + all-reduce** (``flashinfer.comm.
  trtllm_moe_finalize_allreduce_fusion(backend="cake")``, JIT
  ``flashinfer.jit.cake_moe_finalize_comm``). Top-k combine of the permuted
  expert output + all-reduce + residual + RMSNorm, optional NVFP4 ``quant_out``
  (``token_num * 7168 / 2`` bytes) with E4M3 ``scale_out`` in SWIZZLED_128x4
  layout. Same dtype / hidden size / world-size / workspace contract as the
  union; ``norm_out``, ``residual_out`` and ``expert_scale_factor`` required.

Process groups, NVSHMEM / CUDA-IPC workspaces and launch ordering across ranks
are owned by the runtime (``sglang.srt.layers.communication``); this module
only forwards FlashInfer's prepare / factory / launch functions. The
``supports_*`` checks cover what one process can verify: device architecture,
dtype, shapes, hidden size, the ``world_size`` argument and FlashInfer module
availability. They do not verify the symmetric-memory backend, peer access,
or that every rank selected the same backend.

Not supported here (keep the existing SGLang path): other hidden sizes,
world sizes outside the sets above, FP8 inputs, ``allreduce_fusion(
moe_finalize_backend="cake")`` on a non-TRT-LLM workspace, and the private
``_all_gather_matmul_cake_packed_qkv_sm103_tp4`` experiment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from sglang.kernels.cake_kernels._support import (
    BLACKWELL_DATACENTER,
    cuda_tensor_on,
    device_capability,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch
    import torch.distributed as dist

ARCHS = BLACKWELL_DATACENTER

# All-gather matmul (A38 / A39 / E2-24 / E2-25).
FI_AG_MODULE = "flashinfer.comm.all_gather_matmul.cake_all_gather_matmul"
FI_AG_DISPATCH_MODULE = "flashinfer.comm.all_gather_matmul.all_gather_matmul"
AG_K = 8192
# Output tile width: the kernel admits any N that is a multiple of it.
AG_BLOCK_N = 256
# Row tile: M is padded to it inside the kernel (informational for callers).
AG_BLOCK_M = 128
AG_WORLD_SIZES = (2, 4, 8)

# Fused norm combine (A40 / A41 / E2-32).
FI_NC_MODULE = "flashinfer.comm.cake_fused_norm_combine"
FI_NC_JIT_MODULE = "flashinfer.jit.cake_fused_norm_combine"
NC_WORLD_SIZE = 8
NC_HIDDEN = 2560
NC_TRACKS = 2

# MoE all-reduce union (E2-27) and MoE finalize + all-reduce (E2-28).
FI_AR_MODULE = "flashinfer.comm.trtllm_ar"
FI_MOE_AR_JIT_MODULE = "flashinfer.jit.cake_trtllm_moe_allreduce_union"
FI_MOE_FINALIZE_JIT_MODULE = "flashinfer.jit.cake_moe_finalize_comm"
MOE_AR_HIDDEN = 7168
MOE_AR_WORLD_SIZES = (2, 4, 8)


def _same_cuda_pair(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        cuda_tensor_on(a, ARCHS)
        and b.is_cuda
        and b.device == a.device
        and b.dtype == a.dtype
        and a.is_contiguous()
        and b.is_contiguous()
    )


# --------------------------------------------------------------------------
# All-gather matmul
# --------------------------------------------------------------------------


def _ag_weight_admitted(w: torch.Tensor) -> bool:
    """``w`` is a ``[8192, N]`` view with ``N % 256 == 0`` in one of the two
    kernel layouts: contiguous ``[K, N]`` or the transposed view of a
    contiguous ``[N, K]`` parameter. Other strides would need a copy."""
    if w.ndim != 2 or w.shape[0] != AG_K:
        return False
    n = int(w.shape[1])
    if n <= 0 or n % AG_BLOCK_N:
        return False
    stride_k, stride_n = (int(stride) for stride in w.stride())
    return (stride_n == 1 and stride_k == n) or (stride_k == 1 and stride_n == AG_K)


def _ag_operands_admitted(inp: torch.Tensor, w: torch.Tensor, *, world_size: int) -> bool:
    import torch

    return (
        flashinfer_module_available(FI_AG_MODULE, FI_AG_DISPATCH_MODULE)
        and cuda_tensor_on(inp, ARCHS)
        and w.is_cuda
        and w.device == inp.device
        and w.dtype == inp.dtype
        and inp.dtype in (torch.bfloat16, torch.float16)
        and inp.ndim == 2
        and inp.is_contiguous()
        and inp.shape[0] > 0
        and inp.shape[1] == AG_K
        and _ag_weight_admitted(w)
        and world_size in AG_WORLD_SIZES
    )


def supports_all_gather_matmul(
    inp: torch.Tensor, w: torch.Tensor, *, world_size: int
) -> bool:
    """Admission check for ``all_gather_matmul(backend="cake")``; never raises."""
    return _ag_operands_admitted(inp, w, world_size=world_size)


def supports_prepare_all_gather_matmul(
    inp: torch.Tensor,
    w: torch.Tensor,
    *,
    world_size: int,
    max_rows: Optional[int] = None,
) -> bool:
    """Admission check for the capacity-bound prepared launcher; never raises.

    ``max_rows`` (default ``inp.shape[0]``) is the capacity the launcher is
    prepared for; the sample input must fit in it.
    """
    return _ag_operands_admitted(inp, w, world_size=world_size) and (
        max_rows is None or int(max_rows) >= int(inp.shape[0])
    )


def all_gather_matmul(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    verbose: bool = False,
) -> torch.Tensor:
    """Forward to FlashInfer; returns ``out [M * world_size, N]``.

    ``inp`` is local-only (it need not be a symmetric-memory tensor); the
    backend's own scratch and flags use NVSHMEM symmetric memory. ``w`` is
    passed as given (the engine's ``weight.t()`` view is a supported layout).
    """
    from flashinfer.comm.all_gather_matmul.all_gather_matmul import all_gather_matmul

    return all_gather_matmul(inp, w, group, backend="cake", verbose=verbose)


def prepare_all_gather_matmul(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    max_rows: Optional[int] = None,
    verbose: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Forward to FlashInfer; returns the launcher bound to ``w``, ``group`` and
    a capacity of ``max_rows`` rows (default ``inp.shape[0]``).

    Prepare outside the hot path (it synchronises and sizes the symmetric
    scratch collectively); the returned callable accepts any contiguous input
    with the dtype, device and ``K`` of ``inp`` and at most ``max_rows`` rows
    without a further collective, and is CUDA-graph replayable.
    """
    from flashinfer.comm.all_gather_matmul.all_gather_matmul import (
        prepare_all_gather_matmul,
    )

    return prepare_all_gather_matmul(
        inp, w, group, backend="cake", max_rows=max_rows, verbose=verbose
    )


# --------------------------------------------------------------------------
# Fused norm combine
# --------------------------------------------------------------------------


def supports_fused_norm_combine(
    x: torch.Tensor, *, world_size: int, max_tokens: Optional[int] = None
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises.

    ``max_tokens`` is the workspace capacity when a workspace already exists.
    """
    import torch

    return (
        flashinfer_module_available(FI_NC_MODULE, FI_NC_JIT_MODULE)
        and cuda_tensor_on(x, ARCHS)
        and x.dtype == torch.bfloat16
        and x.ndim == 3
        and x.is_contiguous()
        and x.shape[0] > 0
        and x.shape[1] == NC_TRACKS
        and x.shape[2] == NC_HIDDEN
        and world_size == NC_WORLD_SIZE
        and (max_tokens is None or x.shape[0] <= max_tokens)
    )


def fused_norm_combine_create_workspace(
    *,
    rank: int,
    world_size: int,
    max_tokens: int,
    hidden: int = NC_HIDDEN,
    group: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
) -> Any:
    """Forward to FlashInfer; collectively allocates this rank's IPC workspace.

    Every rank of ``group`` (default WORLD) calls this in the same order with
    the same ``max_tokens``. ``torch.distributed`` must be initialised and
    ``rank`` / ``world_size`` must match the group.
    """
    from flashinfer.comm.cake_fused_norm_combine import (
        cake_fused_norm_combine_create_workspace,
    )

    return cake_fused_norm_combine_create_workspace(
        rank=rank,
        world_size=world_size,
        max_tokens=max_tokens,
        hidden=hidden,
        group=group,
        device=device,
    )


def fused_norm_combine_destroy_workspace(workspace: Any) -> None:
    """Forward to FlashInfer; destroy collectively after the last launch completed."""
    from flashinfer.comm.cake_fused_norm_combine import (
        cake_fused_norm_combine_destroy_workspace,
    )

    cake_fused_norm_combine_destroy_workspace(workspace)


def fused_norm_combine(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    norm_out: torch.Tensor,
    residual_out: torch.Tensor,
    collective_out: torch.Tensor,
    workspace: Any,
    epsilon: float,
) -> None:
    """Forward to FlashInfer; one launch per rank, outputs written in place.

    Every rank launches with the same token count in the same order; a
    workspace carries one ordered launch sequence.
    """
    from flashinfer.comm.cake_fused_norm_combine import cake_fused_norm_combine

    cake_fused_norm_combine(
        x,
        residual,
        weight,
        norm_out=norm_out,
        residual_out=residual_out,
        collective_out=collective_out,
        workspace=workspace,
        epsilon=epsilon,
        backend="cake",
    )


# --------------------------------------------------------------------------
# MoE all-reduce union / MoE finalize + all-reduce
# --------------------------------------------------------------------------


def _supports_moe_allreduce_tensor(t: torch.Tensor, *, world_size: int) -> bool:
    import torch

    return (
        cuda_tensor_on(t, ARCHS)
        and t.dtype in (torch.float16, torch.bfloat16)
        and t.is_contiguous()
        and world_size in MOE_AR_WORLD_SIZES
    )


def supports_trtllm_moe_allreduce_fusion(
    moe_reduction_token_input: torch.Tensor,
    *,
    world_size: int,
    hidden_dim: int,
) -> bool:
    """Admission check for ``trtllm_moe_allreduce_fusion(backend="cake")``; never raises."""
    return (
        flashinfer_module_available(FI_AR_MODULE, FI_MOE_AR_JIT_MODULE)
        and _supports_moe_allreduce_tensor(
            moe_reduction_token_input, world_size=world_size
        )
        and hidden_dim == MOE_AR_HIDDEN
        and moe_reduction_token_input.ndim == 2
        and moe_reduction_token_input.shape[0] >= 1
        and moe_reduction_token_input.shape[1] == MOE_AR_HIDDEN
    )


def supports_trtllm_moe_finalize_allreduce_fusion(
    residual_in: torch.Tensor, *, world_size: int
) -> bool:
    """Admission check for ``trtllm_moe_finalize_allreduce_fusion(backend="cake")``; never raises."""
    return (
        flashinfer_module_available(FI_AR_MODULE, FI_MOE_FINALIZE_JIT_MODULE)
        and _supports_moe_allreduce_tensor(residual_in, world_size=world_size)
        and residual_in.ndim == 2
        and residual_in.shape[0] >= 1
        and residual_in.shape[1] == MOE_AR_HIDDEN
    )


def trtllm_moe_allreduce_fusion(
    world_size: int,
    world_rank: int,
    token_num: int,
    hidden_dim: int,
    workspace_ptrs: torch.Tensor,
    launch_with_pdl: bool,
    residual_in: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    scale_factor: float,
    moe_reduction_device_num_experts: int,
    moe_reduction_scale_input: torch.Tensor,
    moe_reduction_active_experts_token_input: torch.Tensor,
    moe_reduction_token_input: torch.Tensor,
    layout_code: Optional[Any],
    moe_allreduce_out: Optional[torch.Tensor],
    residual_out: Optional[torch.Tensor],
    norm_out: Optional[torch.Tensor],
    quant_out: Optional[torch.Tensor],
    scale_out: Optional[torch.Tensor],
    weight_bias: Optional[float] = None,
) -> None:
    """Forward to FlashInfer with ``backend="cake"``; outputs written in place.

    ``moe_allreduce_out=None`` is served from loader-owned scratch. The
    pointer table is read on the device only, so the launch is
    CUDA-graph capturable once the JIT module is built.
    """
    from flashinfer.comm.trtllm_ar import trtllm_moe_allreduce_fusion

    trtllm_moe_allreduce_fusion(
        world_size,
        world_rank,
        token_num,
        hidden_dim,
        workspace_ptrs,
        launch_with_pdl,
        residual_in,
        rms_gamma,
        rms_eps,
        scale_factor,
        moe_reduction_device_num_experts,
        moe_reduction_scale_input,
        moe_reduction_active_experts_token_input,
        moe_reduction_token_input,
        layout_code,
        moe_allreduce_out,
        residual_out,
        norm_out,
        quant_out,
        scale_out,
        weight_bias,
        backend="cake",
    )


def trtllm_moe_finalize_allreduce_fusion(
    allreduce_in: torch.Tensor,
    residual_in: torch.Tensor,
    norm_weight: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    norm_out: Optional[torch.Tensor],
    residual_out: Optional[torch.Tensor],
    quant_out: Optional[torch.Tensor],
    scale_out: Optional[torch.Tensor],
    workspace_ptrs: torch.Tensor,
    launch_with_pdl: bool,
    world_rank: int,
    world_size: int,
    eps: float,
    shared_expert_output: Optional[torch.Tensor],
    expert_scale_factor: Optional[torch.Tensor],
    routed_scaling_factor: Optional[float],
    weight_bias: Optional[float] = None,
) -> None:
    """Forward to FlashInfer with ``backend="cake"``; outputs written in place."""
    from flashinfer.comm.trtllm_ar import trtllm_moe_finalize_allreduce_fusion

    trtllm_moe_finalize_allreduce_fusion(
        allreduce_in,
        residual_in,
        norm_weight,
        expanded_idx_to_permuted_idx,
        norm_out,
        residual_out,
        quant_out,
        scale_out,
        workspace_ptrs,
        launch_with_pdl,
        world_rank,
        world_size,
        eps,
        shared_expert_output,
        expert_scale_factor,
        routed_scaling_factor,
        weight_bias,
        backend="cake",
    )
