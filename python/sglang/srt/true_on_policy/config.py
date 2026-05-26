from __future__ import annotations

from typing import Any, Optional

import torch

from sglang.srt.true_on_policy.contracts import resolve_true_on_policy_runtime_policy

ROW_LINEAR_INV_BLOCK_K = 128


def _get_server_args(server_args: Optional[Any] = None) -> Any:
    if server_args is not None:
        return server_args

    from sglang.srt.server_args import get_global_server_args

    return get_global_server_args()


def is_true_on_policy_enabled(server_args: Optional[Any] = None) -> bool:
    return resolve_true_on_policy_runtime_policy(_get_server_args(server_args)).enabled


def is_tp_invariant_target(server_args: Optional[Any] = None) -> bool:
    return resolve_true_on_policy_runtime_policy(
        _get_server_args(server_args)
    ).tp_invariant_row_linear


def should_disable_reduce_scatter_for_on_policy(
    server_args: Optional[Any] = None,
) -> bool:
    return resolve_true_on_policy_runtime_policy(
        _get_server_args(server_args)
    ).disable_reduce_scatter


def should_disable_mlp_allreduce_fusion_for_on_policy(
    server_args: Optional[Any] = None,
) -> bool:
    return resolve_true_on_policy_runtime_policy(
        _get_server_args(server_args)
    ).disable_mlp_allreduce_fusion


def should_disable_flashinfer_allreduce_fusion(
    server_args: Optional[Any] = None,
) -> bool:
    return resolve_true_on_policy_runtime_policy(
        _get_server_args(server_args)
    ).disable_flashinfer_allreduce_fusion


def should_force_bfloat16_dense_tensor_math(
    server_args: Optional[Any] = None,
) -> bool:
    return resolve_true_on_policy_runtime_policy(
        _get_server_args(server_args)
    ).force_bfloat16_dense_tensor_math


def should_force_bfloat16_lm_head(
    *,
    server_args: Optional[Any] = None,
    use_fp32_lm_head: bool = False,
) -> bool:
    return (
        resolve_true_on_policy_runtime_policy(
            _get_server_args(server_args)
        ).force_bfloat16_lm_head
        and not use_fp32_lm_head
    )


def should_disable_fused_qk_norm_mrope(
    server_args: Optional[Any] = None,
) -> bool:
    return resolve_true_on_policy_runtime_policy(
        _get_server_args(server_args)
    ).disable_fused_qk_norm_mrope


def should_use_deterministic_moe_routing(server_args: Optional[Any] = None) -> bool:
    return resolve_true_on_policy_runtime_policy(
        _get_server_args(server_args)
    ).deterministic_moe_routing


def should_use_deterministic_moe_combine(server_args: Optional[Any] = None) -> bool:
    return resolve_true_on_policy_runtime_policy(
        _get_server_args(server_args)
    ).deterministic_moe_combine


def get_moe_topk_tiebreak(server_args: Optional[Any] = None) -> Optional[str]:
    return resolve_true_on_policy_runtime_policy(
        _get_server_args(server_args)
    ).moe_topk_tiebreak


def get_on_policy_rms_norm_kwargs(
    server_args: Optional[Any] = None,
    *,
    weight_dtype: Optional[torch.dtype] = None,
    override_orig_dtype: Optional[torch.dtype] = None,
    fp32_residual: bool = False,
) -> dict[str, Any]:
    if not is_true_on_policy_enabled(server_args):
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
    server_args: Optional[Any] = None,
    row_linear_enable_inv: Optional[bool] = None,
) -> bool:
    policy_enabled = resolve_true_on_policy_runtime_policy(
        _get_server_args(server_args)
    ).tp_invariant_row_linear
    if row_linear_enable_inv is not None:
        policy_enabled = policy_enabled and row_linear_enable_inv

    return (
        policy_enabled
        and k_size >= ROW_LINEAR_INV_BLOCK_K
        and k_size % ROW_LINEAR_INV_BLOCK_K == 0
    )


def should_use_tp_invariant_tree_all_reduce(
    server_args: Optional[Any] = None,
    accl_binary_tree_enabled: Optional[bool] = None,
) -> bool:
    policy_enabled = resolve_true_on_policy_runtime_policy(
        _get_server_args(server_args)
    ).deterministic_tree_all_reduce
    if accl_binary_tree_enabled is not None:
        policy_enabled = policy_enabled and not accl_binary_tree_enabled

    return policy_enabled
