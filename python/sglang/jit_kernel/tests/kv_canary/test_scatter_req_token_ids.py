from __future__ import annotations

import random
import unittest

import torch

from sglang.jit_kernel.kv_canary.scatter_req_token_ids import (
    launch_scatter_req_token_ids_kernel,
    scatter_req_token_ids_torch_reference,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, suite="base-b-kernel-unit-1-gpu-large")


_DEVICE = torch.device("cuda")


def _build_pool(*, max_reqs: int, max_context_len: int) -> torch.Tensor:
    return torch.zeros((max_reqs, max_context_len), dtype=torch.int32, device=_DEVICE)


def _build_offsets(lens: list[int]) -> torch.Tensor:
    cumsum = [0]
    for n in lens:
        cumsum.append(cumsum[-1] + n)
    return torch.tensor(cumsum, dtype=torch.int64, device=_DEVICE)


def _build_flat(seqs: list[list[int]]) -> torch.Tensor:
    flat: list[int] = []
    for s in seqs:
        flat.extend(s)
    return torch.tensor(flat, dtype=torch.int64, device=_DEVICE)


class TestScatterReqTokenIds(CustomTestCase):
    def test_scatter_byte_equal_basic(self) -> None:
        """Triton output matches the PyTorch reference for a small mixed batch."""
        seqs = [[10, 20, 30], [40, 50], [60, 70, 80, 90]]
        lens = [len(s) for s in seqs]
        rp = [3, 1, 5]

        flat = _build_flat(seqs)
        offsets = _build_offsets(lens)
        req_pool_indices = torch.tensor(rp, dtype=torch.int64, device=_DEVICE)
        triton_pool = _build_pool(max_reqs=8, max_context_len=16)
        ref_pool = _build_pool(max_reqs=8, max_context_len=16)

        launch_scatter_req_token_ids_kernel(
            flat_in=flat,
            offsets=offsets,
            req_pool_indices=req_pool_indices,
            pool_out=triton_pool,
        )
        scatter_req_token_ids_torch_reference(
            flat_in=flat,
            offsets=offsets,
            req_pool_indices=req_pool_indices,
            pool_out=ref_pool,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(triton_pool, ref_pool))
        # Spot-check: req in slot 3 holds [10,20,30,0,0,...] etc.
        self.assertEqual(triton_pool[3, :3].tolist(), [10, 20, 30])
        self.assertEqual(triton_pool[1, :2].tolist(), [40, 50])
        self.assertEqual(triton_pool[5, :4].tolist(), [60, 70, 80, 90])

    def test_scatter_empty_batch_no_op(self) -> None:
        """Empty input (num_tokens == 0) returns without touching the pool."""
        flat = torch.empty(0, dtype=torch.int64, device=_DEVICE)
        offsets = torch.zeros(1, dtype=torch.int64, device=_DEVICE)
        req_pool_indices = torch.empty(0, dtype=torch.int64, device=_DEVICE)
        pool = _build_pool(max_reqs=8, max_context_len=16)
        pool_before = pool.clone()

        launch_scatter_req_token_ids_kernel(
            flat_in=flat,
            offsets=offsets,
            req_pool_indices=req_pool_indices,
            pool_out=pool,
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(pool, pool_before))

    def test_scatter_single_req(self) -> None:
        """One req with a full row of tokens writes byte-equal to the reference."""
        seqs = [list(range(10))]
        lens = [10]
        rp = [2]

        flat = _build_flat(seqs)
        offsets = _build_offsets(lens)
        req_pool_indices = torch.tensor(rp, dtype=torch.int64, device=_DEVICE)
        triton_pool = _build_pool(max_reqs=4, max_context_len=16)
        ref_pool = _build_pool(max_reqs=4, max_context_len=16)

        launch_scatter_req_token_ids_kernel(
            flat_in=flat,
            offsets=offsets,
            req_pool_indices=req_pool_indices,
            pool_out=triton_pool,
        )
        scatter_req_token_ids_torch_reference(
            flat_in=flat,
            offsets=offsets,
            req_pool_indices=req_pool_indices,
            pool_out=ref_pool,
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(triton_pool, ref_pool))

    def test_scatter_truncates_at_max_context_len(self) -> None:
        """Tokens past the pool's max_context_len are silently dropped (no row spill)."""
        # Two reqs; first req longer than max_context_len. Second req must remain
        # uncorrupted.
        seqs = [list(range(20)), [777, 888, 999]]
        lens = [len(s) for s in seqs]
        rp = [1, 2]
        max_context_len = 8

        flat = _build_flat(seqs)
        offsets = _build_offsets(lens)
        req_pool_indices = torch.tensor(rp, dtype=torch.int64, device=_DEVICE)
        pool = _build_pool(max_reqs=4, max_context_len=max_context_len)

        launch_scatter_req_token_ids_kernel(
            flat_in=flat,
            offsets=offsets,
            req_pool_indices=req_pool_indices,
            pool_out=pool,
        )
        torch.cuda.synchronize()

        self.assertEqual(
            pool[1, :max_context_len].tolist(), list(range(max_context_len))
        )
        self.assertEqual(pool[2, :3].tolist(), [777, 888, 999])

    def test_scatter_random_byte_equal(self) -> None:
        """Randomized fuzz across bs, seq lengths, and req pool indices."""
        rng = random.Random(0)
        max_reqs = 64
        max_context_len = 32

        for _ in range(8):
            bs = rng.randint(1, 16)
            lens = [rng.randint(0, max_context_len) for _ in range(bs)]
            # All distinct req pool indices in [1, max_reqs)
            rp = rng.sample(range(1, max_reqs), k=bs)
            seqs = [[rng.randint(0, 1 << 30) for _ in range(n)] for n in lens]

            flat = _build_flat(seqs)
            offsets = _build_offsets(lens)
            req_pool_indices = torch.tensor(rp, dtype=torch.int64, device=_DEVICE)
            triton_pool = _build_pool(
                max_reqs=max_reqs, max_context_len=max_context_len
            )
            ref_pool = _build_pool(max_reqs=max_reqs, max_context_len=max_context_len)

            launch_scatter_req_token_ids_kernel(
                flat_in=flat,
                offsets=offsets,
                req_pool_indices=req_pool_indices,
                pool_out=triton_pool,
            )
            scatter_req_token_ids_torch_reference(
                flat_in=flat,
                offsets=offsets,
                req_pool_indices=req_pool_indices,
                pool_out=ref_pool,
            )
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(triton_pool, ref_pool))


if __name__ == "__main__":
    unittest.main()
