# SPDX-License-Identifier: Apache-2.0
"""Attention backend autotune policy for dense diffusion attention."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import addict

from sglang.multimodal_gen.runtime.optimization.common import (
    get_nested,
    get_server_policy_configs,
    normalize_policy,
)


def attention_allows_cudnn_sdp(extra_impl_args: Mapping[str, Any]) -> bool:
    if "allow_cudnn_sdp" in extra_impl_args:
        return bool(extra_impl_args["allow_cudnn_sdp"])

    attention_cfg, acceleration_cfg = get_server_policy_configs()
    policy = None
    for candidate in (
        attention_cfg.get("allow_cudnn_sdp"),
        attention_cfg.get("cudnn_sdp"),
        acceleration_cfg.get("allow_cudnn_sdp"),
        acceleration_cfg.get("cudnn_sdp"),
        get_nested(acceleration_cfg, "attention", "allow_cudnn_sdp"),
        get_nested(acceleration_cfg, "attention", "cudnn_sdp"),
    ):
        if candidate is not None:
            policy = candidate
            break
    policy = normalize_policy(policy, default="auto")
    return policy in {"auto", "on", "true", "1", "force"}


def attention_autotune_config(
    extra_impl_args: Mapping[str, Any],
) -> tuple[bool, int, int, float, bool]:
    attention_cfg: Mapping[str, Any] = addict.Dict()
    acceleration_cfg: Mapping[str, Any] = addict.Dict()
    if "attention_autotune" in extra_impl_args:
        enabled = bool(extra_impl_args["attention_autotune"])
    else:
        attention_cfg, acceleration_cfg = get_server_policy_configs()
        enabled = True
        for candidate in (
            attention_cfg.get("attention_autotune"),
            attention_cfg.get("autotune_attention"),
            acceleration_cfg.get("attention_autotune"),
            acceleration_cfg.get("autotune_attention"),
            get_nested(acceleration_cfg, "attention", "autotune"),
        ):
            if candidate is not None:
                enabled = normalize_policy(candidate) in {
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
                    get_nested(acceleration_cfg, "attention", "autotune_warmup") or 5,
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
                    get_nested(acceleration_cfg, "attention", "autotune_iters") or 20,
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
                    get_nested(acceleration_cfg, "attention", "autotune_min_speedup")
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
                get_nested(acceleration_cfg, "attention", "autotune_live_miss"),
            ),
        ),
    )
    live_miss = normalize_policy(live_miss_policy) in {
        "auto",
        "on",
        "true",
        "1",
        "force",
    }
    return enabled, warmup, iters, min_speedup, live_miss
