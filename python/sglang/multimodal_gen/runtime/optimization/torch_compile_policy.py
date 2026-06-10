# SPDX-License-Identifier: Apache-2.0
"""Policy parsing for full DiT torch.compile autotune."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import addict

from sglang.multimodal_gen.runtime.optimization.common import (
    TORCH_COMPILE_MODE_ENV,
    get_nested,
    get_server_policy_configs,
    normalize_policy,
)


@dataclass(frozen=True)
class TorchCompileAutotuneConfig:
    policy: str
    warmup: int
    iters: int
    min_speedup: float
    live_miss: bool


def torch_compile_kwargs() -> dict[str, Any]:
    mode = os.environ.get(TORCH_COMPILE_MODE_ENV, "max-autotune-no-cudagraphs")
    return {"fullgraph": False, "dynamic": None, "mode": mode}


def torch_compile_autotune_config(
    acceleration_cfg: Mapping[str, Any] | None = None,
) -> TorchCompileAutotuneConfig:
    if acceleration_cfg is None:
        _, acceleration_cfg = get_server_policy_configs()
    else:
        acceleration_cfg = acceleration_cfg or addict.Dict()

    policy_value = None
    for candidate in (
        acceleration_cfg.get("torch_compile_policy"),
        acceleration_cfg.get("torch_compile_autotune"),
        acceleration_cfg.get("compile_autotune"),
        get_nested(acceleration_cfg, "torch_compile", "policy"),
    ):
        if candidate is not None:
            policy_value = candidate
            break

    policy = normalize_policy(policy_value, default="auto")
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
            get_nested(acceleration_cfg, "torch_compile", "warmup") or 1,
        )
    )
    iters = int(
        acceleration_cfg.get(
            "torch_compile_iters",
            get_nested(acceleration_cfg, "torch_compile", "iters") or 3,
        )
    )
    min_speedup = float(
        acceleration_cfg.get(
            "torch_compile_min_speedup",
            get_nested(acceleration_cfg, "torch_compile", "min_speedup") or 1.05,
        )
    )
    live_miss_policy = acceleration_cfg.get(
        "torch_compile_live_miss",
        get_nested(acceleration_cfg, "torch_compile", "live_miss"),
    )
    live_miss = normalize_policy(live_miss_policy) in {
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
