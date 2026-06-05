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


def _get_server_policy_configs() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    from sglang.multimodal_gen.runtime.server_args import get_global_server_args

    try:
        server_args = get_global_server_args()
    except ValueError:
        return addict.Dict(), addict.Dict()
    return (
        server_args.attention_backend_config or addict.Dict(),
        server_args.acceleration_config or addict.Dict(),
    )


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

    attention_cfg, acceleration_cfg = _get_server_policy_configs()
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
    policy = _normalize_policy(policy, default="auto")
    return policy in {"auto", "on", "true", "1", "force"}


def attention_autotune_config(
    extra_impl_args: Mapping[str, Any],
) -> tuple[bool, int, int, float]:
    attention_cfg: Mapping[str, Any] = addict.Dict()
    acceleration_cfg: Mapping[str, Any] = addict.Dict()
    if "attention_autotune" in extra_impl_args:
        enabled = bool(extra_impl_args["attention_autotune"])
    else:
        attention_cfg, acceleration_cfg = _get_server_policy_configs()
        enabled = True
        for candidate in (
            attention_cfg.get("attention_autotune"),
            attention_cfg.get("autotune_attention"),
            acceleration_cfg.get("attention_autotune"),
            acceleration_cfg.get("autotune_attention"),
            _get_nested(acceleration_cfg, "attention", "autotune"),
        ):
            if candidate is not None:
                enabled = _normalize_policy(candidate) in {
                    "auto",
                    "on",
                    "true",
                    "1",
                    "force",
                }
                break

    warmup = int(
        extra_impl_args.get(
            "attention_autotune_warmup",
            attention_cfg.get(
                "attention_autotune_warmup",
                acceleration_cfg.get(
                    "attention_autotune_warmup",
                    _get_nested(acceleration_cfg, "attention", "autotune_warmup")
                    or 3,
                ),
            ),
        )
    )
    iters = int(
        extra_impl_args.get(
            "attention_autotune_iters",
            attention_cfg.get(
                "attention_autotune_iters",
                acceleration_cfg.get(
                    "attention_autotune_iters",
                    _get_nested(acceleration_cfg, "attention", "autotune_iters")
                    or 10,
                ),
            ),
        )
    )
    min_speedup = float(
        extra_impl_args.get(
            "attention_autotune_min_speedup",
            attention_cfg.get(
                "attention_autotune_min_speedup",
                acceleration_cfg.get(
                    "attention_autotune_min_speedup",
                    _get_nested(acceleration_cfg, "attention", "autotune_min_speedup")
                    or 1.02,
                ),
            ),
        )
    )
    return enabled, warmup, iters, min_speedup
