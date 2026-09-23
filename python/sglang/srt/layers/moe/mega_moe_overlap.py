# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""CPU-safe overlap gate for Aiter MegaMoE shared/routed streams."""

from __future__ import annotations

from typing import Any

from sglang.srt.environ import envs


def should_overlap_shared_and_routed(moe: Any, num_tokens: int) -> bool:
    if envs.SGLANG_AITER_MEGA_RANK_SYNC.get():
        return False
    from sglang.srt.model_executor.runner import get_is_capture_mode

    return (
        moe.alt_stream is not None
        and moe.num_fused_shared_experts == 0
        and num_tokens > 0
        and get_is_capture_mode()
    )
