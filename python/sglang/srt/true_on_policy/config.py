from __future__ import annotations

import contextlib
from typing import Any, Iterator, Optional

import torch

from sglang.srt.true_on_policy.contracts import resolve_true_on_policy_runtime_policy


def _get_global_server_args() -> Any:
    from sglang.srt.runtime_context import get_server_args

    return get_server_args()


def get_rl_on_policy_target() -> Optional[str]:
    return getattr(_get_global_server_args(), "rl_on_policy_target", None)


def is_true_on_policy_enabled() -> bool:
    return resolve_true_on_policy_runtime_policy(_get_global_server_args()).enabled


def should_disable_reduce_scatter_for_on_policy() -> bool:
    return resolve_true_on_policy_runtime_policy(
        _get_global_server_args()
    ).disable_reduce_scatter


def should_disable_mlp_allreduce_fusion_for_on_policy() -> bool:
    return resolve_true_on_policy_runtime_policy(
        _get_global_server_args()
    ).disable_mlp_allreduce_fusion


def should_disable_flashinfer_allreduce_fusion() -> bool:
    return resolve_true_on_policy_runtime_policy(
        _get_global_server_args()
    ).disable_flashinfer_allreduce_fusion


def should_force_bfloat16_dense_tensor_math() -> bool:
    return resolve_true_on_policy_runtime_policy(
        _get_global_server_args()
    ).force_bfloat16_dense_tensor_math


def should_force_bfloat16_lm_head(
    *,
    use_fp32_lm_head: bool = False,
) -> bool:
    return (
        resolve_true_on_policy_runtime_policy(
            _get_global_server_args()
        ).force_bfloat16_lm_head
        and not use_fp32_lm_head
    )


def should_disable_fused_qk_norm_mrope() -> bool:
    return resolve_true_on_policy_runtime_policy(
        _get_global_server_args()
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
        "fp32_residual": fp32_residual,
    }
    if weight_dtype is not None:
        kwargs["weight_dtype"] = weight_dtype
    if override_orig_dtype is not None:
        kwargs["override_orig_dtype"] = override_orig_dtype
    return kwargs


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


def should_use_matched_tp1_row_linear() -> bool:
    return resolve_true_on_policy_runtime_policy(
        _get_global_server_args()
    ).matched_tp1_row_linear
