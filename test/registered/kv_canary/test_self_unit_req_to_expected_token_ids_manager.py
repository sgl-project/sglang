from __future__ import annotations

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.kv_canary.req_to_expected_token_ids_manager import (
    compute_req_all_ids_info,
    populate_req_to_expected_token_ids,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kv_canary.fixtures import DEFAULT_DEVICE, make_forward_batch
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=15, suite="extra-a-test-1-gpu-small-amd")


def _make_req(*, origin: list[int], output: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        origin_input_ids=array("q", origin),
        output_ids=array("q", output),
    )


class TestComputeReqAllIdsInfo(CustomTestCase):
    def test_single_req_concatenates_origin_then_output(self) -> None:
        """One req's flat is origin_input_ids followed by output_ids in that order."""
        req = _make_req(origin=[10, 20, 30], output=[40, 50])
        flat, lens = compute_req_all_ids_info([req])
        self.assertEqual(flat.tolist(), [10, 20, 30, 40, 50])
        self.assertEqual(lens.tolist(), [5])

    def test_multi_req_flat_is_concat_across_reqs(self) -> None:
        """Multi-req flat is per-req (origin+output) concatenated in req order."""
        reqs = [
            _make_req(origin=[1, 2], output=[3]),
            _make_req(origin=[100], output=[]),
            _make_req(origin=[7, 8, 9], output=[10, 11]),
        ]
        flat, lens = compute_req_all_ids_info(reqs)
        self.assertEqual(flat.tolist(), [1, 2, 3, 100, 7, 8, 9, 10, 11])
        self.assertEqual(lens.tolist(), [3, 1, 5])

    def test_returned_cpu_tensors_are_pinned(self) -> None:
        """Snapshot tensors live on pinned CPU memory so the manager's async H2D actually overlaps."""
        req = _make_req(origin=[1, 2, 3], output=[4])
        flat, lens = compute_req_all_ids_info([req])
        self.assertTrue(flat.is_pinned())
        self.assertTrue(lens.is_pinned())
        self.assertEqual(flat.device, torch.device("cpu"))
        self.assertEqual(lens.device, torch.device("cpu"))


class TestPopulateReqToExpectedTokenIds(CustomTestCase):
    def setUp(self) -> None:
        self.device = DEFAULT_DEVICE

    def _make_pool(self, *, max_reqs: int, max_context_len: int) -> torch.Tensor:
        return torch.full(
            (max_reqs, max_context_len),
            -999,
            dtype=torch.int32,
            device=self.device,
        )

    def _fb_with_snapshot(
        self,
        *,
        req_pool_indices: list[int],
        lens: list[int],
        flat: list[int],
    ) -> SimpleNamespace:
        fb = make_forward_batch(
            self.device,
            bs=len(req_pool_indices),
            req_pool_indices=torch.tensor(
                req_pool_indices, dtype=torch.int64, device=self.device
            ),
        )
        fb.req_all_ids_flat = torch.tensor(flat, dtype=torch.int64, pin_memory=True)
        fb.req_all_ids_lens = torch.tensor(lens, dtype=torch.int64, pin_memory=True)
        return fb

    def test_no_op_when_snapshot_is_none(self) -> None:
        """Cuda-graph capture's dry-run leaves snapshot fields as None; manager must early-return."""
        fb = make_forward_batch(self.device, bs=2)
        pool = self._make_pool(max_reqs=4, max_context_len=8)
        original = pool.clone()
        populate_req_to_expected_token_ids(
            forward_batch=fb, req_to_verify_expected_tokens=pool
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(pool, original))

    def test_no_op_when_pool_is_none(self) -> None:
        """When the validator is off the device pool is None; manager must early-return without touching anything."""
        fb = self._fb_with_snapshot(req_pool_indices=[1], lens=[3], flat=[10, 20, 30])
        populate_req_to_expected_token_ids(
            forward_batch=fb, req_to_verify_expected_tokens=None
        )

    def test_no_op_when_bs_zero(self) -> None:
        """Empty batch (bs == 0) must early-return; no kernel launch."""
        fb = make_forward_batch(
            self.device,
            bs=0,
            req_pool_indices=torch.zeros(0, dtype=torch.int64, device=self.device),
            seq_lens=torch.zeros(0, dtype=torch.int32, device=self.device),
        )
        fb.req_all_ids_flat = torch.zeros(0, dtype=torch.int64, pin_memory=True)
        fb.req_all_ids_lens = torch.zeros(0, dtype=torch.int64, pin_memory=True)
        pool = self._make_pool(max_reqs=4, max_context_len=8)
        original = pool.clone()
        populate_req_to_expected_token_ids(
            forward_batch=fb, req_to_verify_expected_tokens=pool
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(pool, original))

    def test_raises_when_lens_length_mismatches_batch_size(self) -> None:
        """req_all_ids_lens length must equal forward_batch batch size; mismatch indicates a corrupted snapshot."""
        fb = self._fb_with_snapshot(
            req_pool_indices=[1, 2], lens=[3], flat=[10, 20, 30]
        )
        pool = self._make_pool(max_reqs=4, max_context_len=8)
        with self.assertRaisesRegex(RuntimeError, "req_all_ids_lens length"):
            populate_req_to_expected_token_ids(
                forward_batch=fb, req_to_verify_expected_tokens=pool
            )

    def test_raises_when_cumsum_does_not_match_flat_numel(self) -> None:
        """cumsum(lens) must equal flat.numel(); inconsistent snapshot raises."""
        fb = self._fb_with_snapshot(
            req_pool_indices=[1, 2], lens=[3, 4], flat=[10, 20, 30, 40]
        )
        pool = self._make_pool(max_reqs=4, max_context_len=8)
        with self.assertRaisesRegex(RuntimeError, "snapshot inconsistent"):
            populate_req_to_expected_token_ids(
                forward_batch=fb, req_to_verify_expected_tokens=pool
            )

    def test_happy_path_scatters_each_req_into_its_pool_row(self) -> None:
        """Scatter populates pool[rp, :len_r] = req_r's flattened tokens; other rows untouched."""
        fb = self._fb_with_snapshot(
            req_pool_indices=[1, 3],
            lens=[3, 2],
            flat=[10, 20, 30, 40, 50],
        )
        pool = self._make_pool(max_reqs=5, max_context_len=8)
        original = pool.clone()
        populate_req_to_expected_token_ids(
            forward_batch=fb, req_to_verify_expected_tokens=pool
        )
        torch.cuda.synchronize()

        pool_cpu = pool.cpu()
        self.assertEqual(pool_cpu[1, :3].tolist(), [10, 20, 30])
        self.assertEqual(pool_cpu[3, :2].tolist(), [40, 50])
        # Other rows must be left at their pre-scatter sentinel.
        for untouched in (0, 2, 4):
            self.assertTrue(
                torch.equal(pool_cpu[untouched], original[untouched].cpu()),
                f"row {untouched} was unexpectedly modified",
            )


if __name__ == "__main__":
    unittest.main()
