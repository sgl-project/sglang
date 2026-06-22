import unittest

import torch

from sglang.srt.models.deepseek_common.pp_proxy import PPProxyTopKBuffer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPPProxyTopKBuffer(unittest.TestCase):
    def test_copy_uses_distinct_reusable_storage(self):
        buffer = PPProxyTopKBuffer()
        source = torch.arange(6, dtype=torch.int32).view(2, 3)

        first = buffer.copy(source)

        self.assertTrue(torch.equal(first, source))
        self.assertNotEqual(first.data_ptr(), source.data_ptr())
        self.assertEqual(first.data_ptr(), buffer.buffer.data_ptr())

        source.fill_(-1)
        self.assertTrue(
            torch.equal(first, torch.arange(6, dtype=torch.int32).view(2, 3))
        )

        second_source = torch.full((2, 3), 7, dtype=torch.int32)
        second = buffer.copy(second_source)

        self.assertEqual(second.data_ptr(), first.data_ptr())
        self.assertTrue(torch.equal(second, second_source))

    def test_copy_grows_buffer_when_needed(self):
        buffer = PPProxyTopKBuffer()
        small = buffer.copy(torch.ones((2, 3), dtype=torch.int32))
        small_ptr = small.data_ptr()

        large_source = torch.arange(12, dtype=torch.int32).view(3, 4)
        large = buffer.copy(large_source)

        self.assertNotEqual(large.data_ptr(), small_ptr)
        self.assertGreaterEqual(buffer.buffer.numel(), large_source.numel())
        self.assertTrue(torch.equal(large, large_source))

    def test_empty_tensor_does_not_allocate_buffer(self):
        buffer = PPProxyTopKBuffer()
        empty = torch.empty((0, 2048), dtype=torch.int32)

        copied = buffer.copy(empty)

        self.assertIs(copied, empty)
        self.assertIsNone(buffer.buffer)


if __name__ == "__main__":
    unittest.main()
