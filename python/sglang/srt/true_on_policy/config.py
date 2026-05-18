from __future__ import annotations

import contextlib
from typing import Any, Iterator, Optional

import torch

from sglang.srt.true_on_policy.contracts import resolve_true_on_policy_runtime_policy

ROW_LINEAR_INV_BLOCK_K = 128


def _get_server_args() -> Any:
    from sglang.srt.runtime_context import get_server_args

    return get_server_args()


def get_rl_on_policy_target() -> Optional[str]:
    return getattr(_get_server_args(), "rl_on_policy_target", None)


def is_true_on_policy_enabled() -> bool:
    return resolve_true_on_policy_runtime_policy(_get_server_args()).enabled


def is_tp_invariant_target() -> bool:
    return resolve_true_on_policy_runtime_policy(
        _get_server_args()
    ).tp_invariant_row_linear


def should_disable_reduce_scatter_for_on_policy() -> bool:
    return resolve_true_on_policy_runtime_policy(
        _get_server_args()
    ).disable_reduce_scatter


def should_disable_mlp_allreduce_fusion_for_on_policy() -> bool:
    return resolve_true_on_policy_runtime_policy(
        _get_server_args()
    ).disable_mlp_allreduce_fusion


def should_disable_flashinfer_allreduce_fusion() -> bool:
    return resolve_true_on_policy_runtime_policy(
        _get_server_args()
    ).disable_flashinfer_allreduce_fusion


def should_force_bfloat16_dense_tensor_math() -> bool:
    return resolve_true_on_policy_runtime_policy(
        _get_server_args()
    ).force_bfloat16_dense_tensor_math


def should_force_bfloat16_lm_head(
    *,
    use_fp32_lm_head: bool = False,
) -> bool:
    return (
        resolve_true_on_policy_runtime_policy(
            _get_server_args()
        ).force_bfloat16_lm_head
        and not use_fp32_lm_head
    )


def should_disable_fused_qk_norm_mrope() -> bool:
    return resolve_true_on_policy_runtime_policy(
        _get_server_args()
    ).disable_fused_qk_norm_mrope


def get_on_policy_rms_norm_kwargs(
    *,
    weight_dtype: Optional[torch.dtype] = None,
    override_orig_dtype: Optional[torch.dtype] = None,
    fp32_residual: bool = False,
) -> dict[str, Any]:
    if not is_true_on_policy_enabled():
        return {}

    kwargs: dict[str, Any] = {
        "cast_x_before_out_mul": True,
        "fp32_residual": fp32_residual,
    }
    if weight_dtype is not None:
        kwargs["weight_dtype"] = weight_dtype
    if override_orig_dtype is not None:
        kwargs["override_orig_dtype"] = override_orig_dtype
    return kwargs


def should_use_tp_invariant_row_linear(
    k_size: int,
    row_linear_enable_inv: Optional[bool] = None,
) -> bool:
    policy_enabled = resolve_true_on_policy_runtime_policy(
        _get_server_args()
    ).tp_invariant_row_linear
    if row_linear_enable_inv is not None:
        policy_enabled = policy_enabled and row_linear_enable_inv

    return (
        policy_enabled
        and k_size >= ROW_LINEAR_INV_BLOCK_K
        and k_size % ROW_LINEAR_INV_BLOCK_K == 0
    )


def should_use_tp_invariant_tree_all_reduce(
    accl_binary_tree_enabled: Optional[bool] = None,
) -> bool:
    policy_enabled = resolve_true_on_policy_runtime_policy(
        _get_server_args()
    ).deterministic_tree_all_reduce
    if accl_binary_tree_enabled is not None:
        policy_enabled = policy_enabled and not accl_binary_tree_enabled

    return policy_enabled


@contextlib.contextmanager
def patch_prefill_only_deterministic_inference_for_cuda_graph(
    server_args: Any,
    *,
    attn_backend: Optional[Any] = None,
    dvr_target_verify_cuda_graph: bool = False,
) -> Iterator[bool]:
    enabled = (
        getattr(server_args, "enable_prefill_only_deterministic_inference", False)
        and not dvr_target_verify_cuda_graph
    )
    if not enabled:
        yield False
        return

    saved_num_splits = None
    if attn_backend is not None and hasattr(attn_backend, "num_splits"):
        saved_num_splits = attn_backend.num_splits

    try:
        if attn_backend is not None and hasattr(attn_backend, "num_splits"):
            attn_backend.num_splits = 0

        yield True
    finally:
        if attn_backend is not None and hasattr(attn_backend, "num_splits"):
            attn_backend.num_splits = saved_num_splits
