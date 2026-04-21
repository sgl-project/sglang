import contextlib
from typing import Iterable, Optional

import torch
import torch.distributed as dist

_TP_INVARIANT_MODE = False
_TORCH_LIBRARY_HANDLES = []
_MATMUL_K_BLOCK = 128


def is_tp_invariant_mode_enabled() -> bool:
    return _TP_INVARIANT_MODE


def enable_tp_invariant_mode() -> None:
    global _TP_INVARIANT_MODE
    _TP_INVARIANT_MODE = True


def disable_tp_invariant_mode() -> None:
    global _TP_INVARIANT_MODE
    _TP_INVARIANT_MODE = False


@contextlib.contextmanager
def set_tp_invariant_mode(enabled: bool = True):
    global _TP_INVARIANT_MODE
    old_state = _TP_INVARIANT_MODE
    if enabled:
        enable_tp_invariant_mode()
    else:
        disable_tp_invariant_mode()
    try:
        yield
    finally:
        _TP_INVARIANT_MODE = old_state


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _fixed_tree_sum_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Sum tensors in a fixed binary-tree order.

    This helper intentionally does not call `sum()` or stack-and-reduce. Those
    APIs can choose different reduction orders across backends, while
    true-on-policy comparison needs a stable order that can be mirrored by
    Megatron.
    """
    partials = list(tensors)
    if not partials:
        raise ValueError("at least one tensor is required")

    while len(partials) > 1:
        next_partials = []
        for idx in range(0, len(partials), 2):
            if idx + 1 < len(partials):
                next_partials.append(partials[idx] + partials[idx + 1])
            else:
                next_partials.append(partials[idx])
        partials = next_partials

    return partials[0]


def matmul_tp_persistent(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    fp32_accum: bool = False,
) -> torch.Tensor:
    """Matrix multiply with a stable K-block reduction order.

    PR 1 keeps this path dormant and correctness-first. Later PRs can replace
    the block implementation with architecture-specific Triton kernels behind
    the same public API without changing SGLang/Miles call sites.
    """
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError(
            f"expected 2D tensors, got A.dim={A.dim()} and B.dim={B.dim()}"
        )
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"matmul dimension mismatch: A={tuple(A.shape)}, B={tuple(B.shape)}"
        )

    out_dtype = A.dtype
    compute_A = A.float() if fp32_accum else A
    compute_B = B.float() if fp32_accum else B
    k_size = A.shape[1]

    partials = []
    for start in range(0, k_size, _MATMUL_K_BLOCK):
        end = min(start + _MATMUL_K_BLOCK, k_size)
        partials.append(compute_A[:, start:end] @ compute_B[start:end, :])

    output = _fixed_tree_sum_tensors(partials).to(out_dtype)
    if bias is not None:
        output = output + bias
    return output


def matmul_tp_inv(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return matmul_tp_persistent(A, B, bias=bias)


def tree_all_reduce_sum(
    x: torch.Tensor,
    device_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """All-reduce by all-gather plus a fixed binary-tree local sum."""
    if not dist.is_available() or not dist.is_initialized():
        return x.clone()

    world_size = dist.get_world_size(group=device_group)
    if not _is_power_of_two(world_size):
        raise ValueError(
            f"tree_all_reduce_sum requires a power-of-two world size, got {world_size}"
        )
    if world_size == 1:
        return x.clone()

    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x, group=device_group)
    return _fixed_tree_sum_tensors(gathered)


def _validate_moe_tree_reduce_inputs(
    input: torch.Tensor,
    output: torch.Tensor,
    curr_topk_ids: torch.Tensor,
    E: int,
) -> tuple[int, int, int]:
    if input.dim() != 3:
        raise ValueError(
            "input must have shape [tokens, topk, hidden], " f"got {tuple(input.shape)}"
        )
    if curr_topk_ids.dim() != 2:
        raise ValueError(
            "curr_topk_ids must have shape [tokens, topk], "
            f"got {tuple(curr_topk_ids.shape)}"
        )
    if not _is_power_of_two(E):
        raise ValueError(f"E must be a power of two, got {E}")

    token_num, topk, hidden_size = input.shape
    if curr_topk_ids.shape != (token_num, topk):
        raise ValueError(
            "curr_topk_ids shape must match input token/topk dimensions: "
            f"ids={tuple(curr_topk_ids.shape)}, input={tuple(input.shape)}"
        )
    if output.shape != (token_num, hidden_size):
        raise ValueError(
            f"output must have shape {(token_num, hidden_size)}, "
            f"got {tuple(output.shape)}"
        )
    return token_num, topk, hidden_size


def moe_sum_tree_reduce(
    input: torch.Tensor,
    output: torch.Tensor,
    curr_topk_ids: torch.Tensor,
    routed_scaling_factor: float,
    E: int,
) -> torch.Tensor:
    """Reduce local MoE expert outputs in expert-id binary-tree order.

    `curr_topk_ids == -1` marks remote experts and contributes zero. The
    reference implementation is deliberately straightforward and deterministic;
    optimized kernels can specialize this later without changing the contract.
    """
    _, topk, _ = _validate_moe_tree_reduce_inputs(input, output, curr_topk_ids, E)
    ids = curr_topk_ids.to(device=input.device)

    expert_partials = []
    zeros = torch.zeros_like(input[:, 0, :])
    for expert_id in range(E):
        slot_partials = []
        for slot in range(topk):
            slot_partials.append(
                torch.where(
                    (ids[:, slot] == expert_id).unsqueeze(-1),
                    input[:, slot, :],
                    zeros,
                )
            )
        expert_partials.append(_fixed_tree_sum_tensors(slot_partials))

    result = _fixed_tree_sum_tensors(expert_partials) * routed_scaling_factor
    output.copy_(result.to(output.dtype))
    return output


def _register_torch_ops() -> None:
    try:
        def_lib = torch.library.Library("tp_inv_ops", "DEF")
        def_lib.define("matmul_tp_inv(Tensor a, Tensor b, Tensor? bias=None) -> Tensor")
        _TORCH_LIBRARY_HANDLES.append(def_lib)
    except RuntimeError as exc:
        if "already" not in str(exc).lower():
            raise

    try:
        impl_lib = torch.library.Library("tp_inv_ops", "IMPL")
        impl_lib.impl("matmul_tp_inv", matmul_tp_inv, "CompositeExplicitAutograd")
        _TORCH_LIBRARY_HANDLES.append(impl_lib)
    except RuntimeError as exc:
        if "already" not in str(exc).lower():
            raise


_register_torch_ops()
