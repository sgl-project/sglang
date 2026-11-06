"""record_stream_each must skip host tensors instead of raising.

The case that needs a real device stream lives in
test/registered/xpu/test_record_stream_each_xpu.py.
"""

import unittest

import torch

from sglang.srt.speculative.spec_utils import record_stream_each
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestRecordStreamEachHostEntries(CustomTestCase):
    def test_host_tensors_and_non_tensors_are_skipped(self):
        """Host tensors have no record_stream kernel; skip them, do not raise.

        CPU spec decoding (non-overlap EAGLE-family V2, n-gram) passes host
        tensors here, so an always-true filter would crash it.
        """
        record_stream_each([torch.empty(2), None, 3], torch.Stream("cpu"))


if __name__ == "__main__":
    unittest.main()
