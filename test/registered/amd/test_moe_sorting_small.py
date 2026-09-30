"""Small-batch MoE sort (sglang.kernels.ops.moe.moe_sorting_small, no-quant path) vs a torch reference
of the aiter moe_sorting layout, over every kernel variant (P = M * topk <= 256)."""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")


def _reference(ids, w, num_experts, block_size):
    m, topk = ids.shape
    flat = ids.flatten().tolist()
    sorted_ids, sorted_w, expert_ids = [], [], []
    for e in range(num_experts):
        pairs = [p for p, x in enumerate(flat) if x == e]
        if not pairs:
            continue
        n = -(-len(pairs) // block_size) * block_size
        sorted_ids += [((p % topk) << 24) | (p // topk) for p in pairs]
        sorted_ids += [(topk << 24) | m] * (n - len(pairs))
        sorted_w += [w.flatten()[p].item() for p in pairs] + [0.0] * (n - len(pairs))
        expert_ids += [e] * (n // block_size)
    return sorted_ids, sorted_w, expert_ids


@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip, "ROCm only")
class TestMoeSortingSmall(CustomTestCase):
    def _check(self, m, topk, num_experts, block_size, hot):
        from sglang.kernels.ops.moe.moe_sorting_small import _run_small_sort

        g = torch.Generator().manual_seed(m * 1000 + topk)
        ids = torch.stack(
            [torch.randperm(hot, generator=g)[:topk] for _ in range(m)]
        ).to(torch.int32)
        ids = (ids + num_experts - hot).cuda()  # hot = the top `hot` expert ids
        w = torch.rand(m, topk, generator=g).cuda()
        max_padded = m * topk + num_experts * block_size - topk
        sorted_ids = torch.full((max_padded,), -7, dtype=torch.int32, device="cuda")
        sorted_w = torch.full((max_padded,), -7.0, device="cuda")
        expert_ids = torch.full(
            (-(-max_padded // block_size),), -7, dtype=torch.int32, device="cuda"
        )
        num_valid = torch.full((2,), -7, dtype=torch.int32, device="cuda")
        moe_buf = torch.ones(m, 7168, dtype=torch.bfloat16, device="cuda")
        args = (sorted_ids, sorted_w, expert_ids, num_valid, moe_buf, block_size)
        _run_small_sort(ids, w, *args, None, num_experts)
        ref_ids, ref_w, ref_e = _reference(ids.cpu(), w.cpu(), num_experts, block_size)
        n = len(ref_ids)
        self.assertEqual(num_valid.tolist(), [n, m])
        self.assertEqual(sorted_ids[:n].tolist(), ref_ids)
        self.assertEqual(sorted_w[:n].tolist(), ref_w)
        self.assertEqual(expert_ids[: n // block_size].tolist(), ref_e)
        self.assertTrue(bool((moe_buf == 0).all()))

    def test_no_quant(self):
        for num_experts, topk, block_size in ((385, 7, 32), (257, 9, 32), (129, 8, 16)):
            for m in range(1, 256 // topk + 1):
                for hot in (num_experts, 12):  # uniform and skewed routing
                    with self.subTest(e=num_experts, topk=topk, m=m, hot=hot):
                        self._check(m, topk, num_experts, block_size, hot)


if __name__ == "__main__":
    unittest.main()
