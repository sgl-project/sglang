# SPDX-License-Identifier: Apache-2.0
"""Kernel-wise fused-vs-compiled policy for diffusion CustomOp modules.

This module only decides whether a CustomOp is selected for torch.compile
autotune. Eligible ops declare selector metadata with
`CustomOp.kernel_compile_autotune`; the actual benchmarking and commit logic
lives in CustomOp itself so kernel selection stays close to the fused/native
implementations being timed.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import addict

from sglang.multimodal_gen.runtime.optimization.common import (
    TORCH_COMPILE_MODE_ENV,
    env_or_config,
    get_server_policy_configs,
    normalize_policy,
)

KERNEL_COMPILE_POLICY_ENV = "SGLANG_DIFFUSION_KERNEL_COMPILE_POLICY"
KERNEL_COMPILE_OPS_ENV = "SGLANG_DIFFUSION_KERNEL_COMPILE_OPS"
KERNEL_COMPILE_MODE_ENV = "SGLANG_DIFFUSION_KERNEL_COMPILE_MODE"
KERNEL_COMPILE_WARMUP_ENV = "SGLANG_DIFFUSION_KERNEL_COMPILE_WARMUP"
KERNEL_COMPILE_ITERS_ENV = "SGLANG_DIFFUSION_KERNEL_COMPILE_ITERS"
KERNEL_COMPILE_MIN_SPEEDUP_ENV = "SGLANG_DIFFUSION_KERNEL_COMPILE_MIN_SPEEDUP"
KERNEL_COMPILE_LIVE_MISS_ENV = "SGLANG_DIFFUSION_KERNEL_COMPILE_LIVE_MISS"

_KERNEL_COMPILE_SUPPRESSION_DEPTH = 0

# kernel-wise compile stays opt-in until a full-model e2e win is validated
DEFAULT_KERNEL_COMPILE_OPS = frozenset()

KERNEL_COMPILE_OP_GROUPS = frozenset(
    {
        "activation",
        "rotary",
        "elementwise",
        "norm",
        "scale_shift",
        "fused_gate",
        "all",
    }
)


@dataclass(frozen=True)
class KernelCompileAutotuneConfig:
    policy: str
    warmup: int
    iters: int
    min_speedup: float
    live_miss: bool


def _configured_kernel_compile_selectors(configured_ops: str | None) -> set[str]:
    if not configured_ops:
        return set(DEFAULT_KERNEL_COMPILE_OPS)

    return {
        selector.strip()
        for selector in configured_ops.split(",")
        if selector.strip()
    }


def _is_kernel_compile_op_allowed(op_name: str | None, op_cls: type | str) -> bool:
    if isinstance(op_cls, str):
        return False
    if not getattr(op_cls, "kernel_compile_autotune_enabled", False):
        return False

    configured_selectors = _configured_kernel_compile_selectors(
        os.environ.get(KERNEL_COMPILE_OPS_ENV)
    )
    if "all" in configured_selectors:
        return True

    op_selectors = {op_cls.__name__}
    if op_name is not None:
        op_selectors.add(op_name)
    op_selectors.update(getattr(op_cls, "kernel_compile_names", ()))
    op_groups = set(getattr(op_cls, "kernel_compile_groups", ()))
    return bool(
        op_selectors & configured_selectors or op_groups & configured_selectors
    )


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
        _, acceleration_cfg = get_server_policy_configs()
    else:
        acceleration_cfg = acceleration_cfg or addict.Dict()

    policy = normalize_policy(
        env_or_config(
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
        env_or_config(
            KERNEL_COMPILE_WARMUP_ENV,
            acceleration_cfg,
            "kernel_compile_warmup",
            ("kernel", "compile_warmup"),
            ("kernel_compile", "warmup"),
        )
        or 3
    )
    iters = int(
        env_or_config(
            KERNEL_COMPILE_ITERS_ENV,
            acceleration_cfg,
            "kernel_compile_iters",
            ("kernel", "compile_iters"),
            ("kernel_compile", "iters"),
        )
        or 10
    )
    min_speedup = float(
        env_or_config(
            KERNEL_COMPILE_MIN_SPEEDUP_ENV,
            acceleration_cfg,
            "kernel_compile_min_speedup",
            ("kernel", "compile_min_speedup"),
            ("kernel_compile", "min_speedup"),
        )
        or 1.03
    )
    live_miss_policy = env_or_config(
        KERNEL_COMPILE_LIVE_MISS_ENV,
        acceleration_cfg,
        "kernel_compile_live_miss",
        ("kernel", "compile_live_miss"),
        ("kernel_compile", "live_miss"),
    )
    live_miss = normalize_policy(live_miss_policy) in {
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


def custom_op_kernel_compile_policy(op_name: str | None, op_cls: type | str) -> str:
    if kernel_compile_autotune_suppressed():
        return "force_fused"
    config = kernel_compile_autotune_config()
    if config.policy != "auto":
        return config.policy
    if _is_kernel_compile_op_allowed(op_name, op_cls):
        return "auto"
    return "force_fused"


def should_torch_compile_custom_op(op_name: str | None, op_cls: type | str) -> bool:
    return custom_op_kernel_compile_policy(op_name, op_cls) == "force_torch_compile"


def kernel_compile_kwargs() -> dict[str, Any]:
    mode = os.environ.get(
        KERNEL_COMPILE_MODE_ENV,
        os.environ.get(TORCH_COMPILE_MODE_ENV, "max-autotune-no-cudagraphs"),
    )
    return {"fullgraph": False, "dynamic": None, "mode": mode}
