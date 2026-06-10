# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import addict

KERNEL_COMPILE_POLICY_ENV = "SGLANG_DIFFUSION_KERNEL_COMPILE_POLICY"
KERNEL_COMPILE_OPS_ENV = "SGLANG_DIFFUSION_KERNEL_COMPILE_OPS"
KERNEL_COMPILE_MODE_ENV = "SGLANG_DIFFUSION_KERNEL_COMPILE_MODE"
KERNEL_COMPILE_WARMUP_ENV = "SGLANG_DIFFUSION_KERNEL_COMPILE_WARMUP"
KERNEL_COMPILE_ITERS_ENV = "SGLANG_DIFFUSION_KERNEL_COMPILE_ITERS"
KERNEL_COMPILE_MIN_SPEEDUP_ENV = "SGLANG_DIFFUSION_KERNEL_COMPILE_MIN_SPEEDUP"
KERNEL_COMPILE_LIVE_MISS_ENV = "SGLANG_DIFFUSION_KERNEL_COMPILE_LIVE_MISS"
TORCH_COMPILE_MODE_ENV = "SGLANG_TORCH_COMPILE_MODE"
_KERNEL_COMPILE_SUPPRESSION_DEPTH = 0

# keep defaults narrow: shape-sensitive scale-shift kernels remain opt-in
DEFAULT_KERNEL_COMPILE_OPS = frozenset(
    {
        "rotary_embedding",
        "RotaryEmbedding",
        "gelu_and_mul",
        "GeluAndMul",
    }
)

SUPPORTED_KERNEL_COMPILE_OPS = frozenset(
    {
        *DEFAULT_KERNEL_COMPILE_OPS,
        "mul_add",
        "fuse_layernorm_scale_shift_gate_select01",
        "fuse_residual_layernorm_scale_shift_gate_select01",
        "LayerNormScaleShift",
        "RMSNormScaleShift",
        "ScaleResidualLayerNormScaleShift",
        "ScaleResidualRMSNormScaleShift",
        "silu_and_mul",
        "SiluAndMul",
        "LayerNormTanhMulAdd",
        "RMSNormTanhMulAdd",
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


@dataclass(frozen=True)
class TorchCompileAutotuneConfig:
    policy: str
    warmup: int
    iters: int
    min_speedup: float
    live_miss: bool


@dataclass(frozen=True)
class KernelCompileAutotuneConfig:
    policy: str
    warmup: int
    iters: int
    min_speedup: float
    live_miss: bool


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

    kernel_compile_warmup = _first_present(
        config,
        "kernel_compile_warmup",
        ("kernel", "compile_warmup"),
        ("kernel_compile", "warmup"),
    )
    if kernel_compile_warmup is not None:
        os.environ[KERNEL_COMPILE_WARMUP_ENV] = str(kernel_compile_warmup)

    kernel_compile_iters = _first_present(
        config,
        "kernel_compile_iters",
        ("kernel", "compile_iters"),
        ("kernel_compile", "iters"),
    )
    if kernel_compile_iters is not None:
        os.environ[KERNEL_COMPILE_ITERS_ENV] = str(kernel_compile_iters)

    kernel_compile_min_speedup = _first_present(
        config,
        "kernel_compile_min_speedup",
        ("kernel", "compile_min_speedup"),
        ("kernel_compile", "min_speedup"),
    )
    if kernel_compile_min_speedup is not None:
        os.environ[KERNEL_COMPILE_MIN_SPEEDUP_ENV] = str(kernel_compile_min_speedup)

    kernel_compile_live_miss = _first_present(
        config,
        "kernel_compile_live_miss",
        ("kernel", "compile_live_miss"),
        ("kernel_compile", "live_miss"),
    )
    if kernel_compile_live_miss is not None:
        os.environ[KERNEL_COMPILE_LIVE_MISS_ENV] = str(kernel_compile_live_miss)

    torch_compile_mode = None
    for candidate in (
        config.get("torch_compile_mode"),
        _get_nested(config, "torch_compile", "mode"),
    ):
        if candidate is not None:
            torch_compile_mode = candidate
            break
    if torch_compile_mode is not None:
        os.environ[TORCH_COMPILE_MODE_ENV] = str(torch_compile_mode)


def _first_present(config: Mapping[str, Any], *keys: str | tuple[str, ...]) -> Any:
    for key in keys:
        if isinstance(key, tuple):
            value = _get_nested(config, *key)
        else:
            value = config.get(key)
        if value is not None:
            return value
    return None


def _env_or_config(
    env_name: str, acceleration_cfg: Mapping[str, Any], *keys: str | tuple[str, ...]
) -> Any:
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return env_value
    return _first_present(acceleration_cfg, *keys)


def _is_kernel_compile_op_allowed(op_name: str | None, class_name: str) -> bool:
    configured_ops = os.environ.get(KERNEL_COMPILE_OPS_ENV)
    allowed_ops = (
        {
            op.strip()
            for op in configured_ops.split(",")
            if op.strip() in SUPPORTED_KERNEL_COMPILE_OPS
        }
        if configured_ops
        else set(DEFAULT_KERNEL_COMPILE_OPS)
    )
    return class_name in allowed_ops or (op_name is not None and op_name in allowed_ops)


@contextmanager
def suppress_kernel_compile_autotune():
    global _KERNEL_COMPILE_SUPPRESSION_DEPTH
    _KERNEL_COMPILE_SUPPRESSION_DEPTH += 1
    try:
        yield
    finally:
        _KERNEL_COMPILE_SUPPRESSION_DEPTH -= 1


def kernel_compile_autotune_suppressed() -> bool:
    return _KERNEL_COMPILE_SUPPRESSION_DEPTH > 0


def kernel_compile_autotune_config(
    acceleration_cfg: Mapping[str, Any] | None = None,
) -> KernelCompileAutotuneConfig:
    if acceleration_cfg is None:
        _, acceleration_cfg = _get_server_policy_configs()
    else:
        acceleration_cfg = acceleration_cfg or addict.Dict()

    policy = _normalize_policy(
        _env_or_config(
            KERNEL_COMPILE_POLICY_ENV,
            acceleration_cfg,
            "kernel_compile_policy",
            ("kernel", "compile_policy"),
            ("kernel_compile", "policy"),
        ),
        default="auto",
    )
    if policy in {"true", "on", "1"}:
        policy = "auto"
    elif policy in {"false", "0", "none", "off", "eager", "native"}:
        policy = "force_fused"
    elif policy in {"force", "compile", "compiled"}:
        policy = "force_torch_compile"
    elif policy not in {"auto", "force_torch_compile", "force_fused"}:
        policy = "auto"

    warmup = int(
        _env_or_config(
            KERNEL_COMPILE_WARMUP_ENV,
            acceleration_cfg,
            "kernel_compile_warmup",
            ("kernel", "compile_warmup"),
            ("kernel_compile", "warmup"),
        )
        or 3
    )
    iters = int(
        _env_or_config(
            KERNEL_COMPILE_ITERS_ENV,
            acceleration_cfg,
            "kernel_compile_iters",
            ("kernel", "compile_iters"),
            ("kernel_compile", "iters"),
        )
        or 10
    )
    min_speedup = float(
        _env_or_config(
            KERNEL_COMPILE_MIN_SPEEDUP_ENV,
            acceleration_cfg,
            "kernel_compile_min_speedup",
            ("kernel", "compile_min_speedup"),
            ("kernel_compile", "min_speedup"),
        )
        or 1.03
    )
    live_miss_policy = _env_or_config(
        KERNEL_COMPILE_LIVE_MISS_ENV,
        acceleration_cfg,
        "kernel_compile_live_miss",
        ("kernel", "compile_live_miss"),
        ("kernel_compile", "live_miss"),
    )
    live_miss = _normalize_policy(live_miss_policy) in {
        "auto",
        "on",
        "true",
        "1",
        "force",
    }
    return KernelCompileAutotuneConfig(
        policy=policy,
        warmup=max(warmup, 0),
        iters=max(iters, 1),
        min_speedup=max(min_speedup, 1.0),
        live_miss=live_miss,
    )


def custom_op_kernel_compile_policy(op_name: str | None, class_name: str) -> str:
    if kernel_compile_autotune_suppressed():
        return "force_fused"
    config = kernel_compile_autotune_config()
    if config.policy != "auto":
        return config.policy
    return "auto" if _is_kernel_compile_op_allowed(op_name, class_name) else "force_fused"


def should_torch_compile_custom_op(op_name: str | None, class_name: str) -> bool:
    return custom_op_kernel_compile_policy(op_name, class_name) == "force_torch_compile"


def kernel_compile_kwargs() -> dict[str, Any]:
    mode = os.environ.get(
        KERNEL_COMPILE_MODE_ENV,
        os.environ.get(TORCH_COMPILE_MODE_ENV, "max-autotune-no-cudagraphs"),
    )
    return {"fullgraph": False, "dynamic": None, "mode": mode}


def torch_compile_kwargs() -> dict[str, Any]:
    mode = os.environ.get(TORCH_COMPILE_MODE_ENV, "max-autotune-no-cudagraphs")
    return {"fullgraph": False, "dynamic": None, "mode": mode}


def torch_compile_autotune_config(
    acceleration_cfg: Mapping[str, Any] | None = None,
) -> TorchCompileAutotuneConfig:
    if acceleration_cfg is None:
        _, acceleration_cfg = _get_server_policy_configs()
    else:
        acceleration_cfg = acceleration_cfg or addict.Dict()

    policy_value = None
    for candidate in (
        acceleration_cfg.get("torch_compile_policy"),
        acceleration_cfg.get("torch_compile_autotune"),
        acceleration_cfg.get("compile_autotune"),
        _get_nested(acceleration_cfg, "torch_compile", "policy"),
    ):
        if candidate is not None:
            policy_value = candidate
            break

    policy = _normalize_policy(policy_value, default="auto")
    if policy in {"true", "on", "1"}:
        policy = "auto"
    elif policy in {"false", "0", "none"}:
        policy = "off"
    elif policy in {"force", "compile", "compiled", "force_torch_compile"}:
        policy = "force_compile"
    elif policy in {"eager", "native"}:
        policy = "force_eager"
    elif policy not in {"auto", "off", "force_compile", "force_eager"}:
        policy = "auto"

    warmup = int(
        acceleration_cfg.get(
            "torch_compile_warmup",
            _get_nested(acceleration_cfg, "torch_compile", "warmup") or 1,
        )
    )
    iters = int(
        acceleration_cfg.get(
            "torch_compile_iters",
            _get_nested(acceleration_cfg, "torch_compile", "iters") or 3,
        )
    )
    min_speedup = float(
        acceleration_cfg.get(
            "torch_compile_min_speedup",
            _get_nested(acceleration_cfg, "torch_compile", "min_speedup") or 1.05,
        )
    )
    live_miss_policy = acceleration_cfg.get(
        "torch_compile_live_miss",
        _get_nested(acceleration_cfg, "torch_compile", "live_miss"),
    )
    live_miss = _normalize_policy(live_miss_policy) in {
        "auto",
        "on",
        "true",
        "1",
        "force",
    }
    return TorchCompileAutotuneConfig(
        policy=policy,
        warmup=max(warmup, 0),
        iters=max(iters, 1),
        min_speedup=max(min_speedup, 1.0),
        live_miss=live_miss,
    )


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
) -> tuple[bool, int, int, float, bool]:
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
                    _get_nested(acceleration_cfg, "attention", "autotune_warmup") or 5,
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
                    _get_nested(acceleration_cfg, "attention", "autotune_iters") or 20,
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
                    or 1.03,
                ),
            ),
        )
    )
    live_miss_policy = extra_impl_args.get(
        "attention_autotune_live_miss",
        attention_cfg.get(
            "attention_autotune_live_miss",
            acceleration_cfg.get(
                "attention_autotune_live_miss",
                _get_nested(acceleration_cfg, "attention", "autotune_live_miss"),
            ),
        ),
    )
    live_miss = _normalize_policy(live_miss_policy) in {
        "auto",
        "on",
        "true",
        "1",
        "force",
    }
    return enabled, warmup, iters, min_speedup, live_miss
