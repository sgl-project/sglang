# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import addict

KERNEL_COMPILE_POLICY_ENV = "SGLANG_DIFFUSION_KERNEL_COMPILE_POLICY"
KERNEL_COMPILE_OPS_ENV = "SGLANG_DIFFUSION_KERNEL_COMPILE_OPS"
KERNEL_COMPILE_MODE_ENV = "SGLANG_DIFFUSION_KERNEL_COMPILE_MODE"

SUPPORTED_KERNEL_COMPILE_OPS = frozenset(
    {
        "mul_add",
        "fuse_layernorm_scale_shift_gate_select01",
        "fuse_residual_layernorm_scale_shift_gate_select01",
        "LayerNormScaleShift",
        "RMSNormScaleShift",
        "ScaleResidualLayerNormScaleShift",
        "ScaleResidualRMSNormScaleShift",
    }
)


def _normalize_policy(value: Any, default: str = "off") -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return "auto" if value else "off"
    return str(value).strip().lower().replace("-", "_")


def _get_nested(config: Mapping[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def configure_acceleration_policy(config: Mapping[str, Any]) -> None:
    kernel_policy = None
    for candidate in (
        config.get("kernel_compile_policy"),
        _get_nested(config, "kernel", "compile_policy"),
        _get_nested(config, "kernel_compile", "policy"),
    ):
        if candidate is not None:
            kernel_policy = candidate
            break
    if kernel_policy is not None:
        os.environ[KERNEL_COMPILE_POLICY_ENV] = _normalize_policy(kernel_policy)

    kernel_ops = None
    for candidate in (
        config.get("kernel_compile_ops"),
        _get_nested(config, "kernel", "compile_ops"),
        _get_nested(config, "kernel_compile", "ops"),
    ):
        if candidate is not None:
            kernel_ops = candidate
            break
    if isinstance(kernel_ops, str):
        os.environ[KERNEL_COMPILE_OPS_ENV] = kernel_ops
    elif kernel_ops is not None:
        os.environ[KERNEL_COMPILE_OPS_ENV] = ",".join(str(op) for op in kernel_ops)


def should_torch_compile_custom_op(op_name: str | None, class_name: str) -> bool:
    policy = _normalize_policy(os.environ.get(KERNEL_COMPILE_POLICY_ENV))
    if policy in {"off", "false", "0", "none", "force_fused"}:
        return False
    if policy == "force_torch_compile":
        return True
    if policy != "auto":
        return False

    configured_ops = os.environ.get(KERNEL_COMPILE_OPS_ENV)
    if configured_ops:
        allowed_ops = {
            op.strip()
            for op in configured_ops.split(",")
            if op.strip() in SUPPORTED_KERNEL_COMPILE_OPS
        }
    else:
        allowed_ops = set()
    return class_name in allowed_ops or (op_name is not None and op_name in allowed_ops)


def kernel_compile_kwargs() -> dict[str, Any]:
    mode = os.environ.get(
        KERNEL_COMPILE_MODE_ENV,
        os.environ.get("SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"),
    )
    return {"fullgraph": False, "dynamic": None, "mode": mode}


def attention_allows_cudnn_sdp(extra_impl_args: Mapping[str, Any]) -> bool:
    if "allow_cudnn_sdp" in extra_impl_args:
        return bool(extra_impl_args["allow_cudnn_sdp"])

    from sglang.multimodal_gen.runtime.server_args import get_global_server_args

    server_args = get_global_server_args()
    attention_cfg = server_args.attention_backend_config or addict.Dict()
    acceleration_cfg = server_args.acceleration_config or addict.Dict()
    policy = None
    for candidate in (
        attention_cfg.get("allow_cudnn_sdp"),
        attention_cfg.get("cudnn_sdp"),
        acceleration_cfg.get("allow_cudnn_sdp"),
        acceleration_cfg.get("cudnn_sdp"),
        _get_nested(acceleration_cfg, "attention", "allow_cudnn_sdp"),
        _get_nested(acceleration_cfg, "attention", "cudnn_sdp"),
    ):
        if candidate is not None:
            policy = candidate
            break
    policy = _normalize_policy(policy)
    return policy in {"auto", "on", "true", "1", "force"}
