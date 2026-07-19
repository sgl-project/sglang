from __future__ import annotations

import random
import unittest

import torch

from sglang.jit_kernel.kv_canary.scatter_req_token_ids import (
    launch_scatter_req_token_ids_kernel,
    scatter_req_token_ids_torch_reference,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")


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

    def test_scatter_mixed_empty_and_nonempty_reqs(self) -> None:
        """Middle req has length 0 between two non-empty reqs: pool rows are byte-equal and untouched rows stay zero."""
        seqs = [[1, 2], [], [3, 4, 5]]
        lens = [len(s) for s in seqs]
        rp = [2, 4, 6]

        flat = _build_flat(seqs)
        offsets = _build_offsets(lens)
        req_pool_indices = torch.tensor(rp, dtype=torch.int64, device=_DEVICE)
        triton_pool = _build_pool(max_reqs=8, max_context_len=8)
        ref_pool = _build_pool(max_reqs=8, max_context_len=8)

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
        # Middle req contributes nothing; its pool row stays zero.
        zero_row = torch.zeros(8, dtype=torch.int32, device=_DEVICE)
        self.assertTrue(torch.equal(triton_pool[4], zero_row))
        # First and third reqs are written to their respective rows.
        self.assertEqual(triton_pool[2, :2].tolist(), [1, 2])
        self.assertEqual(triton_pool[6, :3].tolist(), [3, 4, 5])

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

    def test_scatter_large_batch_non_uniform(self) -> None:
        """A large, non-uniform batch (bs=1024, past any per-program batch cap) is byte-equal.

        The (request, column-block) grid scales directly with bs, so there is no
        upper bound on the number of requests. Segment lengths vary widely and
        some are empty, so this also fuzzes the offsets handling at scale.
        """
        rng = random.Random(1)
        bs = 1024
        max_reqs = bs + 1
        max_context_len = 40

        lens = [rng.randint(0, max_context_len) for _ in range(bs)]
        rp = rng.sample(range(1, max_reqs), k=bs)
        seqs = [[rng.randint(0, 1 << 30) for _ in range(n)] for n in lens]

        flat = _build_flat(seqs)
        offsets = _build_offsets(lens)
        req_pool_indices = torch.tensor(rp, dtype=torch.int64, device=_DEVICE)
        triton_pool = _build_pool(max_reqs=max_reqs, max_context_len=max_context_len)
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

    def test_scatter_multi_column_block_and_truncation(self) -> None:
        """Requests longer than one column tile (and than max_context_len) stay byte-equal.

        Mixes a very long request (spanning several column blocks and truncated at
        max_context_len) with short ones, at randomized non-uniform offsets, so the
        cblk>0 tiles and the per-request truncation boundary are both exercised.
        """
        rng = random.Random(2)
        max_context_len = 2048  # forces >1 column block (COL_BLOCK=1024)
        # One request far longer than max_context_len, plus assorted shorter ones.
        lens = [3000, 0, 1, 1025, 500, 2048, 2049, 7]
        bs = len(lens)
        max_reqs = 32
        rp = rng.sample(range(1, max_reqs), k=bs)
        seqs = [[rng.randint(0, 1 << 30) for _ in range(n)] for n in lens]

        flat = _build_flat(seqs)
        offsets = _build_offsets(lens)
        req_pool_indices = torch.tensor(rp, dtype=torch.int64, device=_DEVICE)
        triton_pool = _build_pool(max_reqs=max_reqs, max_context_len=max_context_len)
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


class TestScatterInputValidation(CustomTestCase):
    """Cover the strict input checks in launch_scatter_req_token_ids_kernel."""

    def test_raises_on_2d_flat_in(self) -> None:
        """A 2-D flat_in tensor triggers a ValueError before any kernel launch."""
        flat = torch.zeros((2, 2), dtype=torch.int64, device=_DEVICE)
        offsets = torch.tensor([0, 1, 2], dtype=torch.int64, device=_DEVICE)
        req_pool_indices = torch.tensor([1, 2], dtype=torch.int64, device=_DEVICE)
        pool = _build_pool(max_reqs=4, max_context_len=4)
        with self.assertRaises(ValueError):
            launch_scatter_req_token_ids_kernel(
                flat_in=flat,
                offsets=offsets,
                req_pool_indices=req_pool_indices,
                pool_out=pool,
            )

    def test_raises_on_wrong_dtype_pool(self) -> None:
        """A pool_out with non-int32 dtype triggers a TypeError."""
        flat = torch.tensor([10, 20], dtype=torch.int64, device=_DEVICE)
        offsets = torch.tensor([0, 2], dtype=torch.int64, device=_DEVICE)
        req_pool_indices = torch.tensor([1], dtype=torch.int64, device=_DEVICE)
        pool = torch.zeros((4, 4), dtype=torch.int64, device=_DEVICE)
        with self.assertRaises(TypeError):
            launch_scatter_req_token_ids_kernel(
                flat_in=flat,
                offsets=offsets,
                req_pool_indices=req_pool_indices,
                pool_out=pool,
            )

    def test_raises_on_offsets_len_mismatch(self) -> None:
        """offsets.shape[0] must equal bs + 1; mismatch triggers a ValueError."""
        flat = torch.tensor([10, 20], dtype=torch.int64, device=_DEVICE)
        # bs = 2 but offsets has length 2 instead of 3.
        offsets = torch.tensor([0, 2], dtype=torch.int64, device=_DEVICE)
        req_pool_indices = torch.tensor([1, 2], dtype=torch.int64, device=_DEVICE)
        pool = _build_pool(max_reqs=4, max_context_len=4)
        with self.assertRaises(ValueError):
            launch_scatter_req_token_ids_kernel(
                flat_in=flat,
                offsets=offsets,
                req_pool_indices=req_pool_indices,
                pool_out=pool,
            )

    def test_raises_on_wrong_dtype_flat_in(self) -> None:
        """A flat_in with non-int64 dtype triggers a TypeError."""
        flat = torch.tensor([10, 20], dtype=torch.int32, device=_DEVICE)
        offsets = torch.tensor([0, 2], dtype=torch.int64, device=_DEVICE)
        req_pool_indices = torch.tensor([1], dtype=torch.int64, device=_DEVICE)
        pool = _build_pool(max_reqs=4, max_context_len=4)
        with self.assertRaises(TypeError):
            launch_scatter_req_token_ids_kernel(
                flat_in=flat,
                offsets=offsets,
                req_pool_indices=req_pool_indices,
                pool_out=pool,
            )


if __name__ == "__main__":
    unittest.main()
