# SPDX-License-Identifier: Apache-2.0
"""Apply user-facing acceleration config to process-wide runtime knobs."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from sglang.multimodal_gen.runtime.optimization.common import (
    TORCH_COMPILE_MODE_ENV,
    first_present,
    get_nested,
    normalize_policy,
)
from sglang.multimodal_gen.runtime.optimization.kernel_policy import (
    KERNEL_COMPILE_ITERS_ENV,
    KERNEL_COMPILE_LIVE_MISS_ENV,
    KERNEL_COMPILE_MIN_SPEEDUP_ENV,
    KERNEL_COMPILE_OPS_ENV,
    KERNEL_COMPILE_POLICY_ENV,
    KERNEL_COMPILE_WARMUP_ENV,
)

def configure_acceleration_policy(config: Mapping[str, Any]) -> None:
    kernel_policy = None
    for candidate in (
        config.get("kernel_compile_policy"),
        get_nested(config, "kernel", "compile_policy"),
        get_nested(config, "kernel_compile", "policy"),
    ):
        if candidate is not None:
            kernel_policy = candidate
            break
    if kernel_policy is not None:
        os.environ[KERNEL_COMPILE_POLICY_ENV] = normalize_policy(kernel_policy)

    kernel_ops = None
    for candidate in (
        config.get("kernel_compile_ops"),
        get_nested(config, "kernel", "compile_ops"),
        get_nested(config, "kernel_compile", "ops"),
    ):
        if candidate is not None:
            kernel_ops = candidate
            break
    if isinstance(kernel_ops, str):
        os.environ[KERNEL_COMPILE_OPS_ENV] = kernel_ops
    elif kernel_ops is not None:
        os.environ[KERNEL_COMPILE_OPS_ENV] = ",".join(str(op) for op in kernel_ops)

    kernel_compile_warmup = first_present(
        config,
        "kernel_compile_warmup",
        ("kernel", "compile_warmup"),
        ("kernel_compile", "warmup"),
    )
    if kernel_compile_warmup is not None:
        os.environ[KERNEL_COMPILE_WARMUP_ENV] = str(kernel_compile_warmup)

    kernel_compile_iters = first_present(
        config,
        "kernel_compile_iters",
        ("kernel", "compile_iters"),
        ("kernel_compile", "iters"),
    )
    if kernel_compile_iters is not None:
        os.environ[KERNEL_COMPILE_ITERS_ENV] = str(kernel_compile_iters)

    kernel_compile_min_speedup = first_present(
        config,
        "kernel_compile_min_speedup",
        ("kernel", "compile_min_speedup"),
        ("kernel_compile", "min_speedup"),
    )
    if kernel_compile_min_speedup is not None:
        os.environ[KERNEL_COMPILE_MIN_SPEEDUP_ENV] = str(kernel_compile_min_speedup)

    kernel_compile_live_miss = first_present(
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
        get_nested(config, "torch_compile", "mode"),
    ):
        if candidate is not None:
            torch_compile_mode = candidate
            break
    if torch_compile_mode is not None:
        os.environ[TORCH_COMPILE_MODE_ENV] = str(torch_compile_mode)
