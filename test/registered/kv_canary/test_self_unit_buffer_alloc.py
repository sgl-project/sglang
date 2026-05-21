from __future__ import annotations

import sys
import unittest

import torch

from sglang.jit_kernel.kv_canary.consts import RealKvHashMode
from sglang.srt.kv_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_canary.pool_patch.buffer_alloc import (
    make_packed_source,
    make_row_source,
    resolve_real_kv_read_bytes,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="extra-a", runner_config="1-gpu-large")


def _config(mode: RealKvHashMode) -> CanaryConfig:
    return CanaryConfig(
        mode=CanaryMode.RAISE,
        ring_capacity=1024,
        sweep_interval=0,
        real_kv_hash_mode=mode,
        input_check_mode=False,
        stats_print_every_n_steps=100,
    )


class TestBufferAlloc(CustomTestCase):
    def test_resolve_real_kv_read_bytes_off_returns_zero(self) -> None:
        """Verify OFF mode disables real KV byte reads."""
        self.assertEqual(resolve_real_kv_read_bytes(_config(RealKvHashMode.OFF)), 0)

    def test_resolve_real_kv_read_bytes_partial_returns_16(self) -> None:
        """Verify PARTIAL mode reads the fixed byte prefix."""
        self.assertEqual(
            resolve_real_kv_read_bytes(_config(RealKvHashMode.PARTIAL)), 16
        )

    def test_resolve_real_kv_read_bytes_all_returns_sentinel_so_full_stride_used(
        self,
    ) -> None:
        """Verify ALL mode requests the full token stride."""
        self.assertEqual(
            resolve_real_kv_read_bytes(_config(RealKvHashMode.ALL)), sys.maxsize
        )

    def test_make_row_source_partial_large_stride_clips_to_32(self) -> None:
        """Verify row sources cap partial reads on large strides."""
        num_slots = 4
        bytes_per_token = 128
        layer_buf = torch.zeros(num_slots, bytes_per_token, dtype=torch.uint8)
        sources = make_row_source(layer_buffer=layer_buf, read_bytes=32)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].read_bytes, 32)
        self.assertEqual(sources[0].num_bytes_per_token, bytes_per_token)

    def test_make_row_source_all_large_stride_uses_full_stride(self) -> None:
        """Verify row sources use the full stride for ALL mode."""
        num_slots = 4
        bytes_per_token = 128
        layer_buf = torch.zeros(num_slots, bytes_per_token, dtype=torch.uint8)
        sources = make_row_source(layer_buffer=layer_buf, read_bytes=sys.maxsize)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].read_bytes, bytes_per_token)
        self.assertEqual(sources[0].num_bytes_per_token, bytes_per_token)

    def test_make_row_source_partial_small_stride_raises(self) -> None:
        """Verify row sources reject strides that cannot satisfy 16-byte aligned loads."""
        num_slots = 4
        bytes_per_token = 8
        layer_buf = torch.zeros(num_slots, bytes_per_token, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "num_bytes_per_token"):
            make_row_source(layer_buffer=layer_buf, read_bytes=32)

    def test_make_row_source_all_small_stride_raises(self) -> None:
        """Verify ALL mode rejects strides that cannot satisfy 16-byte aligned loads."""
        num_slots = 4
        bytes_per_token = 8
        layer_buf = torch.zeros(num_slots, bytes_per_token, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "num_bytes_per_token"):
            make_row_source(layer_buffer=layer_buf, read_bytes=sys.maxsize)

    def test_make_packed_source_partial_large_stride_clips_to_32(self) -> None:
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
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].read_bytes, 32)
        self.assertEqual(sources[0].num_bytes_per_token, bytes_per_token)

    def test_make_packed_source_all_large_stride_uses_full_stride(self) -> None:
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
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].read_bytes, bytes_per_token)
        self.assertEqual(sources[0].num_bytes_per_token, bytes_per_token)

    def test_make_packed_source_partial_small_stride_raises(self) -> None:
        """Verify packed sources reject strides that cannot satisfy 16-byte aligned loads."""
        bytes_per_token = 8
        page_size = 1
        page_buffer = torch.zeros(4, bytes_per_token, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "num_bytes_per_token"):
            make_packed_source(
                page_buffer=page_buffer,
                page_size=page_size,
                bytes_per_token=bytes_per_token,
                read_bytes=32,
            )

    def test_make_packed_source_all_small_stride_raises(self) -> None:
        """Verify ALL mode rejects strides that cannot satisfy 16-byte aligned loads."""
        bytes_per_token = 8
        page_size = 1
        page_buffer = torch.zeros(4, bytes_per_token, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "num_bytes_per_token"):
            make_packed_source(
                page_buffer=page_buffer,
                page_size=page_size,
                bytes_per_token=bytes_per_token,
                read_bytes=sys.maxsize,
            )

    def test_make_packed_source_unaligned_read_bytes_raises(self) -> None:
        """Verify packed sources reject unaligned explicit reads."""
        bytes_per_token = 128
        page_size = 1
        page_buffer = torch.zeros(4, bytes_per_token, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            make_packed_source(
                page_buffer=page_buffer,
                page_size=page_size,
                bytes_per_token=bytes_per_token,
                read_bytes=24,
            )

    def test_make_packed_source_oversized_read_bytes_raises(self) -> None:
        """Verify packed sources reject oversized explicit reads."""
        bytes_per_token = 128
        page_size = 1
        page_buffer = torch.zeros(4, bytes_per_token, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "<= num_bytes_per_token"):
            make_packed_source(
                page_buffer=page_buffer,
                page_size=page_size,
                bytes_per_token=bytes_per_token,
                read_bytes=256,
            )


if __name__ == "__main__":
    unittest.main()
