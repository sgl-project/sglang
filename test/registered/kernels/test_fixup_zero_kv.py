import unittest

import torch

from sglang.kernels.ops.attention.fixup_zero_kv import fixup_zero_kv_rows
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _cuda_available() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is required"
    return True, ""


_SUPPORTED, _SKIP_REASON = _cuda_available()


@unittest.skipIf(not _SUPPORTED, _SKIP_REASON)
class TestFixupZeroKVRows(CustomTestCase):
    def _run_case(self, out: torch.Tensor, lse: torch.Tensor) -> None:
        total_tokens = out.shape[0]
        kv_lens = torch.tensor([0, 2, 0], dtype=torch.int32, device="cuda")
        cum_seq_lens = torch.tensor(
            [0, 2, 3, total_tokens], dtype=torch.int32, device="cuda"
        )

        out.fill_(7)
        lse.fill_(9)
        fixup_zero_kv_rows(out, lse, kv_lens, cum_seq_lens, max_seq_len=2)
        torch.cuda.synchronize()

        zero_tokens = torch.tensor([0, 1, 3, 4], device="cuda")
        keep_tokens = torch.tensor([2], device="cuda")

        self.assertTrue(torch.all(out[zero_tokens] == 0).item())
        self.assertTrue(torch.all(torch.isneginf(lse[zero_tokens])).item())
        self.assertTrue(torch.all(out[keep_tokens] == 7).item())
        self.assertTrue(torch.all(lse[keep_tokens] == 9).item())

    def test_contiguous_nonzero_storage_offset_uses_scalar_path(self):
        total_tokens, num_heads, v_head_dim = 5, 4, 8
        out_base = torch.empty(
            total_tokens * num_heads * v_head_dim + 1,
            dtype=torch.float16,
            device="cuda",
        )
        lse_base = torch.empty(
            total_tokens * num_heads + 1,
            dtype=torch.float32,
            device="cuda",
        )
        out = out_base[1:].view(total_tokens, num_heads, v_head_dim)
        lse = lse_base[1:].view(total_tokens, num_heads)

        self.assertTrue(out.is_contiguous())
        self.assertTrue(lse.is_contiguous())
        self.assertEqual(out.storage_offset(), 1)
        self.assertEqual(lse.storage_offset(), 1)
        self.assertNotEqual(out.data_ptr() % 16, 0)
        self.assertNotEqual(lse.data_ptr() % 16, 0)

        self._run_case(out, lse)

    def test_mixed_alignment_keeps_independent_store_modes(self):
        total_tokens, num_heads, v_head_dim = 5, 3, 128
        out = torch.empty(
            (total_tokens, num_heads, v_head_dim),
            dtype=torch.float16,
            device="cuda",
        )
        lse = torch.empty((total_tokens, num_heads), dtype=torch.float32, device="cuda")

        self.assertEqual(out.data_ptr() % 16, 0)
        self.assertEqual((num_heads * v_head_dim * out.element_size()) % 16, 0)
        self.assertEqual(lse.data_ptr() % 16, 0)
        self.assertNotEqual((num_heads * lse.element_size()) % 16, 0)

        self._run_case(out, lse)


if __name__ == "__main__":
    unittest.main()
