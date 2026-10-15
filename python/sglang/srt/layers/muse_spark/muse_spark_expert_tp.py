"""Expert-TP plumbing shared by Muse Spark's stock fallback and megakernels."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import torch

from sglang.kernels.fused_op import get_fused_op_backend, set_fused_op_backend
from sglang.kernels.spec import KernelBackend
from sglang.srt.distributed import moe_tensor_model_parallel_all_reduce
from sglang.srt.runtime_context import get_parallel, get_platform, get_spec


def helion_enabled() -> bool:
    return get_fused_op_backend() is KernelBackend.HELION


# Spec-decode verify widths (draft tokens per request) the megakernels serve.
VERIFY_TOKENS = (8, 16)


def helion_verify_enabled() -> bool:
    """Whether spec-decode verify runs the Helion verify megakernels."""
    parallel = get_parallel()
    return (
        helion_enabled()
        # GB300 only: the bank holds a second W2 next to FlashInfer's (~36 GB
        # per rank), which leaves a B200 too little memory for the KV cache.
        and get_platform().device_sm == 103
        and parallel.tp_size == 4
        and parallel.moe_tp_size == 4
        and parallel.moe_ep_size == 1
        and get_spec().speculative_num_draft_tokens in VERIFY_TOKENS
    )


@contextlib.contextmanager
def default_fused_ops_for_single_request(forward_batch: Any) -> Iterator[None]:
    """For a single request (its prefill and every spec-decode verify) the Helion
    verify megakernels stay on, but every other fused op runs on its default backend:
    their forced Helion versions are slower (prefill ~20 ms; in verify, e.g. the
    all-gather + norm splits in two)."""
    if not (helion_verify_enabled() and forward_batch.batch_size == 1):
        yield
        return
    forced = get_fused_op_backend()
    set_fused_op_backend(None)
    try:
        yield
    finally:
        set_fused_op_backend(forced)


def validate(int4_checkpoint: bool) -> None:
    parallel = get_parallel()
    if parallel.moe_tp_size == 1:
        return
    if parallel.moe_ep_size != 1:
        raise ValueError("Muse experts support either expert TP or EP, not both")
    if not int4_checkpoint and not helion_enabled():
        raise ValueError(
            "Muse BF16/NVFP4 expert TP requires "
            "SGLANG_FORCE_FUSED_OP_BACKEND=helion; otherwise use EP equal to TP"
        )


def partition(intermediate_size: int) -> tuple[int, bool]:
    moe_tp_size = get_parallel().moe_tp_size
    if intermediate_size % moe_tp_size:
        raise ValueError(
            f"Muse expert intermediate size {intermediate_size} is not divisible "
            f"by expert TP size {moe_tp_size}"
        )
    return intermediate_size // moe_tp_size, moe_tp_size > 1


def reduce_partials(
    *, gemm2_out: torch.Tensor, expanded_idx_to_permuted_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce K-split GEMM2 rows before Muse's per-route post-expert norm."""
    flat_map = expanded_idx_to_permuted_idx.reshape(-1)
    rows = moe_tensor_model_parallel_all_reduce(
        gemm2_out.index_select(0, flat_map.clamp(min=0))
    )
    identity = torch.arange(flat_map.numel(), dtype=torch.int32, device=flat_map.device)
    identity = torch.where(flat_map >= 0, identity, -1)
    return rows, identity.view(expanded_idx_to_permuted_idx.shape)


def reduce_deferred(deferred_output):
    gemm2_out, expanded_idx_to_permuted_idx = reduce_partials(
        gemm2_out=deferred_output.gemm2_out,
        expanded_idx_to_permuted_idx=deferred_output.expanded_idx_to_permuted_idx,
    )
    return type(deferred_output)(
        gemm2_out=gemm2_out,
        expert_weights=deferred_output.expert_weights,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        top_k=deferred_output.top_k,
    )


def copy_shard(param: torch.Tensor, loaded_weight: torch.Tensor, name: str) -> bool:
    """Copy this rank's intermediate-dimension slice from a stacked checkpoint."""
    parallel = get_parallel()
    if parallel.moe_tp_size == 1:
        return False
    if name.endswith((".gate_up_proj", ".gate_up_proj_weight_scale")):
        local = param.shape[1] // 2
        start = parallel.moe_tp_rank * local
        gate, up = loaded_weight.chunk(2, dim=1)
        param.data[:, :local].copy_(gate.narrow(1, start, local))
        param.data[:, local:].copy_(up.narrow(1, start, local))
        return True
    if not name.endswith((".down_proj", ".down_proj_weight_scale")):
        return False
    local = param.shape[-1]
    loaded_weight = loaded_weight.narrow(-1, parallel.moe_tp_rank * local, local)
    if param.shape != loaded_weight.shape:
        raise RuntimeError(
            f"Expert weight shape mismatch for {name}: "
            f"parameter={tuple(param.shape)}, checkpoint={tuple(loaded_weight.shape)}"
        )
    param.data.copy_(loaded_weight)
    return True


__all__ = [
    "VERIFY_TOKENS",
    "copy_shard",
    "default_fused_ops_for_single_request",
    "helion_enabled",
    "helion_verify_enabled",
    "partition",
    "reduce_deferred",
    "reduce_partials",
    "validate",
]
