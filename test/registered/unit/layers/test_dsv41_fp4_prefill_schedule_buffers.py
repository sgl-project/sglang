"""PrefillScheduleBuffers keeps its row regions and the [safe, total_splits] pair 16-byte
aligned for every row count, so AITER's alignment-specialized cta_info kernel compiles no
new variant per rows % 4 while serving."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer_schedule_hip import (
    MAX_FUSED_ROWS,
    PrefillScheduleBuffers,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_ROW_COUNTS = (*range(1, 34), 1001, 4095, MAX_FUSED_ROWS, MAX_FUSED_ROWS + 1, 6003)


def _offset(buffers: PrefillScheduleBuffers, region: torch.Tensor) -> int:
    return region.storage_offset() - buffers.storage.storage_offset()


class TestPrefillScheduleBuffersAlignment(CustomTestCase):
    def test_regions_start_16_byte_aligned_and_do_not_overlap(self):
        for rows in _ROW_COUNTS:
            buffers = PrefillScheduleBuffers(rows=rows, device=torch.device("cpu"))
            if buffers.fused:
                regions = (
                    buffers.chunks,
                    buffers.incl,
                    buffers.excl,
                    buffers.row_to_batch,
                    buffers.local_starts,
                )
            else:
                regions = (buffers.row_to_batch, buffers.local_starts)
            offsets = [_offset(buffers, region) for region in regions]
            for region, offset in zip(regions, offsets):
                self.assertEqual(region.numel(), rows)
                self.assertEqual(offset % 4, 0, f"rows={rows}: region at {offset}")
            for prev, nxt in zip(offsets, offsets[1:]):
                self.assertGreaterEqual(nxt, prev + rows, f"rows={rows}: overlap")
            if buffers.fused:
                # The prep kernel writes [safe, total_splits] through one pointer.
                safe = _offset(buffers, buffers.safe)
                self.assertEqual(safe % 4, 0, f"rows={rows}: safe at {safe}")
                self.assertGreaterEqual(safe, offsets[-1] + rows)
                self.assertEqual(_offset(buffers, buffers.total_splits), safe + 1)


if __name__ == "__main__":
    unittest.main()
