import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.utils.common import get_num_new_pages
from sglang.test.ci.ci_register import register_cpu_ci, register_xpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_xpu_ci(est_time=5, suite="stage-a-test-1-gpu-xpu")


class _OperationRecorder(TorchDispatchMode):
    def __init__(self):
        self.operations = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.operations.add(str(func))
        return func(*args, **(kwargs or {}))


class TestGetNumNewPages(unittest.TestCase):
    def test_decode_counts_page_boundaries_without_dtype_conversion(self):
        # Given: four requests, two of which start a new KV page on decode.
        seq_lens = torch.tensor([1, 2, 65, 130], dtype=torch.int64)

        # When: the host-side decode allocator counts new pages.
        recorder = _OperationRecorder()
        with recorder:
            result = get_num_new_pages(seq_lens, page_size=64, decode=True)

        # Then: the count is correct and the boolean mask is not copied to int32.
        self.assertEqual(result, 2)
        self.assertNotIn("aten._to_copy.default", recorder.operations)


if __name__ == "__main__":
    unittest.main()
