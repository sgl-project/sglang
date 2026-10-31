# SPDX-License-Identifier: Apache-2.0
"""CMO helpers — stream state lives in stream_management."""

from sglang.srt.hardware_backend.npu.stream_management import (
    get_cmo_stream,
    get_share_stream,
    prepare_weight_cache,
    set_cmo_stream,
    set_share_stream,
    shared_expert_on_independent_stream,
    wait_cmo_stream,
    wait_share_stream,
)

__all__ = [
    "get_cmo_stream",
    "set_cmo_stream",
    "prepare_weight_cache",
    "wait_cmo_stream",
    "get_share_stream",
    "set_share_stream",
    "wait_share_stream",
    "shared_expert_on_independent_stream",
]
