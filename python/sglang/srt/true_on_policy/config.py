from __future__ import annotations

import contextlib
from typing import Any, Iterator, Optional

import torch

from sglang.srt.true_on_policy.contracts import (
    override_true_on_policy_runtime_policy_enabled,
    resolve_true_on_policy_runtime_policy,
)

ROW_LINEAR_INV_BLOCK_K = 128

_ATTENTION_BACKEND_CHILD_ATTRS = (
    "attn_backend",
    "decode_backend",
    "full_attn_backend",
    "linear_attn_backend",
    "prefill_backend",
    "primary",
)
_ATTENTION_BACKEND_CHILD_LIST_ATTRS = (
    "attn_backend_list",
    "attn_backends",
)


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
    return resolve_true_on_policy_runtime_policy(_get_server_args(server_args)).moe_topk_tiebreak


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


def _iter_attention_backend_tree(attn_backend: Any) -> Iterator[Any]:
    seen: set[int] = set()
    stack = [attn_backend]
    while stack:
        backend = stack.pop()
        if backend is None:
            continue
        backend_id = id(backend)
        if backend_id in seen:
            continue
        seen.add(backend_id)
        yield backend

        for attr in _ATTENTION_BACKEND_CHILD_ATTRS:
            child = getattr(backend, attr, None)
            if child is not None:
                stack.append(child)
        for attr in _ATTENTION_BACKEND_CHILD_LIST_ATTRS:
            children = getattr(backend, attr, None)
            if children is None:
                continue
            stack.extend(child for child in children if child is not None)


@contextlib.contextmanager
def patch_prefill_only_deterministic_attention_backend(
    attn_backend: Optional[Any],
) -> Iterator[None]:
    """Temporarily match full deterministic attention metadata for prefill scoring."""
    saved_num_splits: list[tuple[Any, Any]] = []
    if attn_backend is None:
        yield
        return

    try:
        for backend in _iter_attention_backend_tree(attn_backend):
            if hasattr(backend, "num_splits"):
                saved_num_splits.append((backend, backend.num_splits))
                backend.num_splits = 1
        yield
    finally:
        for backend, num_splits in reversed(saved_num_splits):
            backend.num_splits = num_splits


@contextlib.contextmanager
def DeterministicInferenceScope(
    server_args: Any,
    *,
    attn_backend: Optional[Any] = None,
) -> Iterator[None]:
    """Bundle every save/restore needed to enter deterministic inference for one forward.

    Owns the full toggle set so callers can't drift them out of sync:
      * ``server_args.enable_deterministic_inference``
      * batch-invariant mode
      * TP-invariant mode (entered only if the contract requests TP-invariant rollout)
      * deterministic attention metadata (``patch_prefill_only_deterministic_attention_backend``)
      * runtime-policy override flag
      * ``SGLANG_ENABLE_DETERMINISTIC_INFERENCE`` env override

    Anything previously enabled stays enabled across the scope; only modes this
    scope flipped are flipped back on exit.
    """
    from sglang.srt.batch_invariant_ops import (
        disable_batch_invariant_mode,
        enable_batch_invariant_mode,
        is_batch_invariant_mode_enabled,
    )
    from sglang.srt.environ import envs
    from sglang.srt.tp_invariant_ops import (
        disable_tp_invariant_mode,
        enable_tp_invariant_mode,
        is_tp_invariant_mode_enabled,
    )

    saved_deterministic_flag = server_args.enable_deterministic_inference
    batch_invariant_was_enabled = is_batch_invariant_mode_enabled()
    tp_invariant_was_enabled = is_tp_invariant_mode_enabled()
    enabled_batch_invariant_here = False
    enabled_tp_invariant_here = False

    try:
        server_args.enable_deterministic_inference = True
        if not batch_invariant_was_enabled:
            enable_batch_invariant_mode()
            enabled_batch_invariant_here = True
        if is_tp_invariant_target(server_args) and not tp_invariant_was_enabled:
            enable_tp_invariant_mode()
            enabled_tp_invariant_here = True
        with (
            patch_prefill_only_deterministic_attention_backend(attn_backend),
            override_true_on_policy_runtime_policy_enabled(True),
            envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(True),
        ):
            yield
    finally:
        server_args.enable_deterministic_inference = saved_deterministic_flag
        if enabled_batch_invariant_here:
            disable_batch_invariant_mode()
        if enabled_tp_invariant_here:
            disable_tp_invariant_mode()


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

    saved_num_splits = None
    if attn_backend is not None and hasattr(attn_backend, "num_splits"):
        saved_num_splits = attn_backend.num_splits

    try:
        if attn_backend is not None and hasattr(attn_backend, "num_splits"):
            attn_backend.num_splits = 0

        with override_true_on_policy_runtime_policy_enabled(False):
            yield True
    finally:
        if attn_backend is not None and hasattr(attn_backend, "num_splits"):
            attn_backend.num_splits = saved_num_splits
