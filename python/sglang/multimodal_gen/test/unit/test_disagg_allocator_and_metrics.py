# SPDX-License-Identifier: Apache-2.0
"""Unit tests for BuddyAllocator, TransferTensorBuffer, and DisaggMetrics."""

from collections import deque
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.disaggregation.metrics import DisaggMetrics
from sglang.multimodal_gen.runtime.disaggregation.transport.allocator import (
    BuddyAllocator,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.buffer import (
    TransferTensorBuffer,
)


def test_buddy_allocator_fifo_order_and_coalescing():
    """Verify O(1) ordered-set free lists preserve FIFO allocation and full coalescing."""
    allocator = BuddyAllocator(pool_size=1 << 20, min_block_size=1 << 16)
    offsets = [allocator.allocate(1 << 16, request_id=f"req-{i}") for i in range(16)]
    assert offsets == [i * (1 << 16) for i in range(16)]
    assert allocator.allocate(1 << 16) is None

    # Free alternating blocks first, then their buddies to trigger multi-level coalescing
    for idx in range(0, 16, 2):
        assert allocator.free(offsets[idx]) is True
    for idx in range(1, 16, 2):
        assert allocator.free(offsets[idx]) is True

    stats = allocator.get_stats()
    assert stats["allocated_bytes"] == 0
    assert stats["num_allocations"] == 0
    assert stats["free_blocks_by_size"] == {1 << 20: 1}


def test_buddy_allocator_fifo_reuse_within_order():
    """Freed blocks at the same order are reused in FIFO order when not coalesced."""
    allocator = BuddyAllocator(pool_size=4 * 1024, min_block_size=1024)
    offsets = [allocator.allocate(1024) for _ in range(4)]
    assert offsets == [0, 1024, 2048, 3072]

    # Free non-buddy blocks (offset 0 and offset 2048) so neither coalesces
    assert allocator.free(2048) is True
    assert allocator.free(0) is True
    assert allocator.allocate(1024) == 2048
    assert allocator.allocate(1024) == 0


def test_transfer_tensor_buffer_read_write_roundtrip():
    """TransferTensorBuffer round-trips tensors using dtype.itemsize."""
    buf = TransferTensorBuffer(pool_size=1 << 16, min_block_size=1 << 12, device="cpu")
    handle = buf.allocate(size=1024, request_id="r1")
    assert handle is not None

    src = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    written = buf.write_tensor(handle, src, byte_offset=16)
    assert written == 96

    out = buf.read_tensor(handle, shape=[4, 6], dtype=torch.float32, byte_offset=16)
    assert torch.equal(out, src)
    assert buf.free(handle) is True


def test_disagg_metrics_sliding_window_deque_pruning():
    """DisaggMetrics uses a deque for O(1) sliding-window completion pruning."""
    metrics = DisaggMetrics(role="denoiser")
    assert isinstance(metrics._completion_times, deque)

    clock = [100.0]
    with patch(
        "sglang.multimodal_gen.runtime.disaggregation.metrics.time.monotonic",
        side_effect=lambda: clock[0],
    ):
        for i in range(5):
            clock[0] = 100.0 + i * 10.0
            metrics.record_request_start(f"req-{i}")
            clock[0] += 1.0
            metrics.record_request_complete(f"req-{i}")

        # At t=175.0, cutoff is 115.0, so completions at 101.0 and 111.0 are pruned
        clock[0] = 175.0
        snap = metrics.snapshot()

    assert snap.requests_completed == 5
    assert len(metrics._completion_times) == 3
    assert list(metrics._completion_times) == [121.0, 131.0, 141.0]
    assert abs(snap.throughput_rps - (3 / 60.0)) < 1e-9
