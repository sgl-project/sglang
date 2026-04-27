from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator, Optional

import torch

from sglang.srt.true_on_policy.contracts import resolve_true_on_policy_runtime_policy

ROW_LINEAR_INV_BLOCK_K = 128


def _get_server_args(server_args: Optional[Any] = None) -> Any:
    if server_args is not None:
        return server_args

    from sglang.srt.server_args import get_global_server_args

    return get_global_server_args()


def get_rl_on_policy_target(server_args: Optional[Any] = None) -> Optional[str]:
    return getattr(_get_server_args(server_args), "rl_on_policy_target", None)


def is_true_on_policy_enabled(server_args: Optional[Any] = None) -> bool:
    return get_rl_on_policy_target(server_args) is not None


def is_tp_invariant_target(server_args: Optional[Any] = None) -> bool:
    return get_rl_on_policy_target(server_args) == "fsdp_tp"


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
    if row_linear_enable_inv is None:
        row_linear_enable_inv = os.environ.get("ROW_LINEAR_ENABLE_INV", "0") == "1"

    return (
        resolve_true_on_policy_runtime_policy(
            _get_server_args(server_args)
        ).tp_invariant_row_linear
        and row_linear_enable_inv
        and k_size >= ROW_LINEAR_INV_BLOCK_K
        and k_size % ROW_LINEAR_INV_BLOCK_K == 0
    )


def should_use_tp_invariant_tree_all_reduce(
    server_args: Optional[Any] = None,
    accl_binary_tree_enabled: Optional[bool] = None,
) -> bool:
    if accl_binary_tree_enabled is None:
        accl_binary_tree_enabled = os.environ.get("ACCL_BINARY_TREE_ENABLE") == "1"

    return (
        resolve_true_on_policy_runtime_policy(
            _get_server_args(server_args)
        ).deterministic_tree_all_reduce
        and not accl_binary_tree_enabled
    )


@contextlib.contextmanager
def patch_prefill_only_deterministic_inference_for_cuda_graph(
    server_args: Any,
    *,
    attn_backend: Optional[Any] = None,
    global_server_args: Optional[Any] = None,
    dvr_target_verify_cuda_graph: bool = False,
) -> Iterator[bool]:
    enabled = (
        getattr(server_args, "enable_prefill_only_deterministic_inference", False)
        and not dvr_target_verify_cuda_graph
    )
    if not enabled:
        yield False
        return

    saved_server_state = {
        "enable_deterministic_inference": getattr(
            server_args, "enable_deterministic_inference", False
        ),
        "enable_flashinfer_allreduce_fusion": getattr(
            server_args, "enable_flashinfer_allreduce_fusion", False
        ),
        "rl_on_policy_target": getattr(server_args, "rl_on_policy_target", None),
        "true_on_policy_contract": getattr(
            server_args, "true_on_policy_contract", None
        ),
        "disable_custom_all_reduce": getattr(
            server_args, "disable_custom_all_reduce", False
        ),
    }
    saved_global_state = None
    if global_server_args is not None:
        saved_global_state = {
            "enable_deterministic_inference": getattr(
                global_server_args, "enable_deterministic_inference", False
            ),
            "enable_flashinfer_allreduce_fusion": getattr(
                global_server_args, "enable_flashinfer_allreduce_fusion", False
            ),
            "rl_on_policy_target": getattr(
                global_server_args, "rl_on_policy_target", None
            ),
            "true_on_policy_contract": getattr(
                global_server_args, "true_on_policy_contract", None
            ),
            "disable_custom_all_reduce": getattr(
                global_server_args, "disable_custom_all_reduce", False
            ),
        }

    saved_num_splits = None
    if attn_backend is not None and hasattr(attn_backend, "num_splits"):
        saved_num_splits = attn_backend.num_splits

    env_keys = [
        "SGLANG_ENABLE_DETERMINISTIC_INFERENCE",
        "SGLANG_DISABLE_CUSTOM_ALL_REDUCE",
        "NCCL_ALGO",
        "ACCL_BINARY_TREE_ENABLE",
    ]
    saved_env = {key: os.environ.get(key) for key in env_keys}

    from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
        disable_batch_invariant_mode,
        enable_batch_invariant_mode,
    )
    from sglang.srt.tp_invariant_ops import (
        disable_tp_invariant_mode,
        enable_tp_invariant_mode,
    )

    def _apply_mutation(obj: Any) -> None:
        obj.enable_deterministic_inference = False
        obj.enable_flashinfer_allreduce_fusion = True
        obj.rl_on_policy_target = None
        obj.true_on_policy_contract = None
        obj.disable_custom_all_reduce = False

    def _restore(obj: Any, state: dict[str, Any]) -> None:
        for key, value in state.items():
            setattr(obj, key, value)

    try:
        _apply_mutation(server_args)
        if global_server_args is not None:
            _apply_mutation(global_server_args)

        os.environ["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"] = "0"
        os.environ["SGLANG_DISABLE_CUSTOM_ALL_REDUCE"] = "0"
        os.environ.pop("NCCL_ALGO", None)
        if os.environ.get("ACCL_BINARY_TREE_ENABLE") == "1":
            os.environ["ACCL_BINARY_TREE_ENABLE"] = "0"

        if attn_backend is not None and hasattr(attn_backend, "num_splits"):
            attn_backend.num_splits = 0

        disable_batch_invariant_mode()
        disable_tp_invariant_mode()
        yield True
    finally:
        _restore(server_args, saved_server_state)
        if global_server_args is not None and saved_global_state is not None:
            _restore(global_server_args, saved_global_state)

        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        if attn_backend is not None and hasattr(attn_backend, "num_splits"):
            attn_backend.num_splits = saved_num_splits

        if saved_server_state["enable_deterministic_inference"]:
            enable_batch_invariant_mode()
        if saved_server_state["rl_on_policy_target"] == "fsdp_tp":
            enable_tp_invariant_mode()
