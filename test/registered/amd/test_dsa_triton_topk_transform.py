"""The ROCm single-pass DSA top-k selector (triton_topk_transform.py) against a
torch reference: an exact top-k (ties at the boundary may be broken either way),
mapped through the page table, -1 padded; including rows whose candidate buffer
overflows and must take the exact whole-row path."""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci

PAGE, TOPK = 64, 2048


def _skip_reason():
    if not torch.cuda.is_available() or torch.version.hip is None:
        return "ROCm GPU required"
    try:
        from sglang.kernels.ops.attention.dsa import triton_topk_transform  # noqa: F401
    except ImportError as exc:  # pragma: no cover - build-environment dependent
        return f"unavailable: {exc}"
    return None


SKIP = _skip_reason()
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipIf(SKIP is not None, SKIP or "")
class TestTritonTopkTransform(unittest.TestCase):
    def setUp(self):
        from sglang.kernels.ops.attention.dsa.triton_topk_transform import (
            topk_transform_paged_triton,
        )

        self.fn = topk_transform_paged_triton
        self.dev = torch.device("cuda", 0)

    def _run(self, logits, lens):
        rows, width = logits.shape
        pages = -(-width // PAGE)
        g = torch.Generator(device="cpu").manual_seed(rows + width)
        pt = torch.stack(
            [torch.randperm(1 << 20, generator=g)[:pages] for _ in range(rows)]
        ).int()
        out = self.fn(logits, lens.to(self.dev), pt.to(self.dev), PAGE, TOPK)
        torch.cuda.synchronize()
        return out.cpu(), pt

    def _check(self, logits, lens):
        out, pt = self._run(logits, lens)
        lg = logits.cpu()
        for r in range(lg.shape[0]):
            n = int(lens[r])
            k = min(n, TOPK)
            got = out[r]
            sel = got[got >= 0]
            self.assertEqual(sel.numel(), k, f"row {r}")
            self.assertEqual(int((got == -1).sum()), TOPK - k, f"row {r}")
            self.assertEqual(sel.unique().numel(), k, f"row {r}: duplicate slots")
            # slot -> position through the (unique-per-row) page table
            npg = -(-n // PAGE)
            page_idx = {int(p): i for i, p in enumerate(pt[r, :npg].tolist())}
            pos = torch.tensor(
                [page_idx[int(s) // PAGE] * PAGE + int(s) % PAGE for s in sel]
            )
            self.assertTrue(bool((pos < n).all()), f"row {r}: slot past length")
            if k == n:
                self.assertEqual(sorted(pos.tolist()), list(range(n)), f"row {r}")
                continue
            vals = lg[r, :n]
            kth = torch.topk(vals, k).values[-1]
            self.assertTrue(bool((vals[pos] >= kth).all()), f"row {r}: below k-th")
            self.assertEqual(
                int((vals[pos] > kth).sum()), int((vals > kth).sum()), f"row {r}"
            )

    def test_random_rows(self):
        g = torch.Generator(device=self.dev).manual_seed(0)
        for rows, width in ((1, 4096), (6, 200_000), (48, 300_000), (128, 65_536)):
            with self.subTest(rows=rows, width=width):
                logits = torch.randn(rows, width, device=self.dev, generator=g)
                lens = torch.randint(1, width + 1, (rows,), generator=None).int()
                lens[0] = width
                self._check(logits, lens)

    def test_short_rows_select_everything(self):
        logits = torch.randn(8, 4096, device=self.dev)
        self._check(logits, torch.tensor([1, 100, 2047, 2048, 7, 0 + 5, 30, 1000]))

    def test_candidate_overflow_takes_exact_path(self):
        """A row whose sampled probes see only zeros while 5000 ones sit between
        them: the sampled threshold admits the whole row (> CAND candidates), and
        only the exact path returns ones."""
        from sglang.kernels.ops.attention.dsa import triton_topk_transform as m

        width = 1 << 20
        logits = torch.zeros(2, width, device=self.dev)
        span = width // m._SMP_NCH
        probe = (torch.arange(width) % span) < m._SMP_CH
        free = torch.nonzero(~probe).flatten()
        logits[0, free[torch.randperm(free.numel())[:5000]]] = 1.0
        logits[1] = torch.randn(width, device=self.dev)
        self._check(logits, torch.tensor([width, width]))

    def test_all_equal_row(self):
        logits = torch.full((3, 100_000), 0.5, device=self.dev)
        self._check(logits, torch.tensor([100_000, 50_000, 3000]))


if __name__ == "__main__":
    unittest.main()
