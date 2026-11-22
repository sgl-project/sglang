import unittest

import torch

from sglang.kernels.ops.attention.dsa import hy4_flash_indexpool as indexpool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE = 64
HEADS = 16
GROUPS_PER_ROW = 511
TOPK = 2048


def _index_cache(num_pages: int, device) -> torch.Tensor:
    """Random pooled FP8 keys (no NaN encodings) with positive scales."""
    buf = torch.randint(
        0, 255, (num_pages, PAGE * 132), dtype=torch.uint8, device=device
    )
    keys = buf[:, : PAGE * 128]
    keys[(keys & 0x7F) == 0x7F] = 0
    buf[:, PAGE * 128 :].view(torch.float32)[:] = (
        torch.rand(num_pages, PAGE, device=device) * 0.1 + 0.01
    )
    return buf


class TestHy4FlashIndexPoolDecodeTopK(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        torch.manual_seed(0)

    def test_length_aware_selection_matches_full_width(self):
        """Selecting each row's top groups within its own length picks the same
        tokens as a top-k over the -inf-padded full width, including rows with
        fewer groups than the budget and rows whose position runs past the
        scored width."""
        from flashinfer import top_k
        from flashinfer.topk import top_k_ragged_transform

        device = "cuda"
        rows, num_pages = 8, 48
        num_groups = num_pages * PAGE // 4
        pages = torch.randperm(rows * num_pages, device=device).view(rows, num_pages)
        pages = (pages % 96).int()
        buf = _index_cache(96, device)
        positions = torch.randint(0, num_pages * PAGE, (rows,), device=device)
        positions[0] = 1000  # 250 groups: fewer than the budget
        positions[1] = num_pages * PAGE + 4096  # past the scored width
        row_to_batch = torch.arange(rows, device=device)
        q = (torch.randn(rows, HEADS, 128, device=device) * 20).clamp(-448, 448)
        q = q.to(torch.float8_e4m3fn)
        w = torch.rand(rows, HEADS, device=device)

        full = indexpool.indexpool_logits(
            q, w, pages, row_to_batch, positions, buf, PAGE, num_groups
        )
        values, picked = top_k(full, GROUPS_PER_ROW)
        expected = indexpool.indexpool_expand(
            picked, values, positions, pages, row_to_batch, None, TOPK
        )

        lengths = torch.empty(rows, dtype=torch.int32, device=device)
        ragged = indexpool.indexpool_logits(
            q, w, pages, row_to_batch, positions, buf, PAGE, num_groups, lengths
        )
        picked = top_k_ragged_transform(
            ragged, torch.zeros_like(lengths), lengths, GROUPS_PER_ROW
        )
        actual = indexpool.indexpool_expand(
            picked, None, positions, pages, row_to_batch, None, TOPK
        )

        self.assertLessEqual(int(lengths.max()), num_groups)
        for r in range(rows):
            want, got = expected[r], actual[r]
            count = int((got >= 0).sum())
            self.assertEqual(count, int((want >= 0).sum()))
            self.assertTrue(bool((got[:count] >= 0).all()), "indices not packed")
            self.assertEqual(set(got[:count].tolist()), set(want[want >= 0].tolist()))

    def test_empty_width_leaves_no_stale_lengths(self):
        """A zero-width selection (decode graphs before any page is mapped)
        reports zero groups for every row instead of leaving the buffer as is."""
        device = "cuda"
        rows = 4
        lengths = torch.full((rows,), 0x01010101, dtype=torch.int32, device=device)
        q = torch.zeros(rows, HEADS, 128, device=device).to(torch.float8_e4m3fn)
        indexpool.indexpool_logits(
            q,
            torch.zeros(rows, HEADS, device=device),
            torch.zeros(rows, 1, dtype=torch.int32, device=device),
            torch.arange(rows, device=device),
            torch.full((rows,), 100, device=device),
            _index_cache(1, device),
            PAGE,
            0,
            lengths,
        )
        self.assertEqual(lengths.tolist(), [0] * rows)


if __name__ == "__main__":
    unittest.main()
