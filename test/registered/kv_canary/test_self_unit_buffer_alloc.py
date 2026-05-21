from __future__ import annotations

import sys

import torch

from sglang.jit_kernel.kv_canary.consts import RealKvHashMode
from sglang.srt.kv_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_canary.pool_patch.buffer_alloc import (
    make_packed_source,
    make_row_source,
    resolve_read_bytes,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="extra-a", runner_config="1-gpu-large")


def _config(mode: RealKvHashMode) -> CanaryConfig:
    return CanaryConfig(mode=CanaryMode.RAISE, real_kv_hash_mode=mode)


def test_resolve_read_bytes_off_returns_zero() -> None:
    """Verify OFF mode disables real KV byte reads."""
    assert resolve_read_bytes(_config(RealKvHashMode.OFF)) == 0


def test_resolve_read_bytes_partial_returns_16() -> None:
    """Verify PARTIAL mode reads the fixed byte prefix."""
    assert resolve_read_bytes(_config(RealKvHashMode.PARTIAL)) == 16


def test_resolve_read_bytes_all_returns_sentinel_so_full_stride_used() -> None:
    """Verify ALL mode requests the full token stride."""
    assert resolve_read_bytes(_config(RealKvHashMode.ALL)) == sys.maxsize


def test_make_row_source_partial_large_stride_clips_to_32() -> None:
    """Verify row sources cap partial reads on large strides."""
    num_slots = 4
    bytes_per_token = 128
    layer_buf = torch.zeros(num_slots, bytes_per_token, dtype=torch.uint8)
    sources = make_row_source(layer_buffer=layer_buf, read_bytes=32)
    assert len(sources) == 1
    assert sources[0].read_bytes == 32
    assert sources[0].num_bytes_per_token == bytes_per_token


def test_make_row_source_all_large_stride_uses_full_stride() -> None:
    """Verify row sources use the full stride for ALL mode."""
    num_slots = 4
    bytes_per_token = 128
    layer_buf = torch.zeros(num_slots, bytes_per_token, dtype=torch.uint8)
    sources = make_row_source(layer_buffer=layer_buf, read_bytes=sys.maxsize)
    assert len(sources) == 1
    assert sources[0].read_bytes == bytes_per_token
    assert sources[0].num_bytes_per_token == bytes_per_token


def test_make_row_source_partial_small_stride_clips_to_stride() -> None:
    """Verify row sources cap partial reads at a small stride."""
    num_slots = 4
    bytes_per_token = 8
    layer_buf = torch.zeros(num_slots, bytes_per_token, dtype=torch.uint8)
    sources = make_row_source(layer_buffer=layer_buf, read_bytes=32)
    assert len(sources) == 1
    assert sources[0].read_bytes == bytes_per_token


def test_make_row_source_all_small_stride_clips_to_stride() -> None:
    """Verify row sources cap ALL mode at a small stride."""
    num_slots = 4
    bytes_per_token = 8
    layer_buf = torch.zeros(num_slots, bytes_per_token, dtype=torch.uint8)
    sources = make_row_source(layer_buffer=layer_buf, read_bytes=sys.maxsize)
    assert len(sources) == 1
    assert sources[0].read_bytes == bytes_per_token


def test_make_packed_source_partial_large_stride_clips_to_32() -> None:
    """Verify packed sources cap partial reads on large strides."""
    bytes_per_token = 128
    page_size = 2
    page_buffer = torch.zeros(4, bytes_per_token * page_size, dtype=torch.uint8)
    sources = make_packed_source(
        page_buffer=page_buffer,
        page_size=page_size,
        bytes_per_token=bytes_per_token,
        read_bytes=32,
    )
    assert len(sources) == 1
    assert sources[0].read_bytes == 32
    assert sources[0].num_bytes_per_token == bytes_per_token


def test_make_packed_source_all_large_stride_uses_full_stride() -> None:
    """Verify packed sources use the full token stride for ALL mode."""
    bytes_per_token = 128
    page_size = 2
    page_buffer = torch.zeros(4, bytes_per_token * page_size, dtype=torch.uint8)
    sources = make_packed_source(
        page_buffer=page_buffer,
        page_size=page_size,
        bytes_per_token=bytes_per_token,
        read_bytes=sys.maxsize,
    )
    assert len(sources) == 1
    assert sources[0].read_bytes == bytes_per_token
    assert sources[0].num_bytes_per_token == bytes_per_token


def test_make_packed_source_partial_small_stride_clips_to_stride() -> None:
    """Verify packed sources cap partial reads at a small stride."""
    bytes_per_token = 8
    page_size = 1
    page_buffer = torch.zeros(4, bytes_per_token, dtype=torch.uint8)
    sources = make_packed_source(
        page_buffer=page_buffer,
        page_size=page_size,
        bytes_per_token=bytes_per_token,
        read_bytes=32,
    )
    assert len(sources) == 1
    assert sources[0].read_bytes == bytes_per_token


def test_make_packed_source_all_small_stride_clips_to_stride() -> None:
    """Verify packed sources cap ALL mode at a small stride."""
    bytes_per_token = 8
    page_size = 1
    page_buffer = torch.zeros(4, bytes_per_token, dtype=torch.uint8)
    sources = make_packed_source(
        page_buffer=page_buffer,
        page_size=page_size,
        bytes_per_token=bytes_per_token,
        read_bytes=sys.maxsize,
    )
    assert len(sources) == 1
    assert sources[0].read_bytes == bytes_per_token
