"""record_stream_each defers block reuse until the recorded stream catches up.

The host-entry cases that run anywhere live in
test/registered/unit/spec/test_spec_utils_record_stream.py.
"""

from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=10, suite="stage-b-test-1-gpu-xpu")

import unittest

import torch

from sglang.srt.speculative.spec_utils import record_stream_each
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.xpu.is_available(), "Intel XPU not available")
class TestRecordStreamEachOnDeviceStreams(CustomTestCase):
    def test_block_is_not_reused_while_recorded_stream_is_busy(self):
        """A recorded block must not be handed out again on its allocating
        stream while the recorded stream still has queued work."""
        alloc_stream, read_stream = torch.xpu.Stream(), torch.xpu.Stream()
        self.assertNotEqual(alloc_stream, torch.xpu.current_stream())
        # Drain read_stream before the process moves on, pass or fail; the
        # recorded blocks' barriers complete with it.
        self.addCleanup(read_stream.synchronize)

        # Sizes from torch-xpu-ops' record_stream test (about 2 GiB); read_stream
        # must still be busy at the reallocation, or the reuse below is legitimate.
        with torch.xpu.stream(read_stream):
            x = torch.randn(256, 1024, 1024, device="xpu")
            y = torch.randn_like(x)
        torch.xpu.synchronize()

        with torch.xpu.stream(alloc_stream):
            tensor = torch.empty(4, device="xpu")
        with torch.xpu.stream(read_stream):
            for _ in range(30):
                x.add_(y)
        record_stream_each([tensor], read_stream)
        freed_ptr = tensor.data_ptr()
        del tensor

        # If read_stream drained early the reallocation below says nothing.
        self.assertFalse(
            read_stream.query(), "read_stream drained before the reallocation"
        )
        with torch.xpu.stream(alloc_stream):
            realloc = torch.empty(4, device="xpu")
        self.assertNotEqual(realloc.data_ptr(), freed_ptr)


if __name__ == "__main__":
    unittest.main()
