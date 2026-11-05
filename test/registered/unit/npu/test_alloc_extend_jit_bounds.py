"""CPU launch-contract tests for the Ascend paged KV allocator."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestAllocExtendJitBounds(unittest.TestCase):
    def test_one_block_extends_share_a_key_and_larger_extends_keep_their_coverage(self):
        def fill_indices(
            pre_lens,
            seq_lens,
            last_loc,
            free_pages,
            page_size,
            out_indices,
            max_num_extend_tokens,
        ):
            out_indices.zero_()

        operator = MagicMock(side_effect=fill_indices)
        public_ops = SimpleNamespace(alloc_extend_indices=operator)
        with patch.dict(
            sys.modules, {"sgl_kernel_npu.operators.mem_cache": public_ops}
        ):
            for length, expected_bound in (
                (0, 2048),
                (1, 2048),
                (17, 2048),
                (128, 2048),
                (1025, 2048),
                (2047, 2048),
                (2048, 2048),
                (2049, 4096),
                (4097, 8192),
            ):
                with self.subTest(length=length):
                    allocator = SimpleNamespace(
                        device="cpu",
                        page_size=128,
                        roundup=127,
                        debug_mode=False,
                        free_pages=torch.arange(256),
                    )
                    prefix = torch.tensor([0])
                    seq = torch.tensor([length])
                    pages = (length + 127) // 128
                    out = NPUPagedTokenToKVPoolAllocator.alloc_extend(
                        allocator,
                        prefix,
                        prefix,
                        seq,
                        seq,
                        torch.tensor([-1]),
                        length,
                        num_new_pages=pages,
                    )
                    call = operator.call_args
                    self.assertEqual(call.args[-1], expected_bound)
                    self.assertEqual(call.args[4], allocator.page_size)
                    self.assertEqual(call.args[5].dtype, torch.int64)
                    self.assertEqual(out.numel(), length)
                    self.assertEqual(out.dtype, torch.int32)
                    self.assertEqual(allocator.free_pages[0].item(), pages)


if __name__ == "__main__":
    unittest.main()
