"""CPU regressions for the V4.1 Torch indexer with interleaved CP rows."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsv4.v41_indexer.full_topk import FullTopKIndexer
from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import prefill_k_slots
from sglang.srt.layers.attention.dsv4.v41_indexer.types import PrefillInputs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDSV41FullTopKCP(CustomTestCase):
    def setUp(self):
        self.req_to_token = torch.arange(8 * 64).view(8, 64) * 2
        self.pool = SimpleNamespace(
            get_low_ratio_index_k_dequant=lambda layer, slots: slots.float()[:, None]
        )
        with patch(
            "sglang.srt.layers.attention.dsv4.v41_indexer.full_topk.is_deep_select_supported",
            return_value=False,
        ):
            self.backend = FullTopKIndexer(
                token_to_kv_pool=self.pool,
                req_to_token=self.req_to_token,
                use_deep_gemm_prefill=False,
                use_deep_gemm_decode=False,
            )

    def _inputs(self, req, pos, counts, ratio, *, explicit=False, device_counts=True):
        rows = pos.numel()
        indexer = SimpleNamespace(
            index_topk=3,
            queries=lambda q, freqs: q[:, None, :],
            head_weights=lambda x: x.new_ones((x.shape[0], 1)),
            scores=lambda q, k, weights: q[:, 0, 0, None] * k[None, :, 0],
        )
        req_pool_indices = torch.tensor([5, 2, 7])
        seq_lens_cpu = [20, 0, 20]
        return PrefillInputs(
            indexer=indexer,
            layer_id=0,
            compress_ratio=ratio,
            freqs_cis=torch.zeros(64, 1),
            x=torch.ones(rows, 1),
            q_lora=torch.ones(rows, 1),
            positions=pos,
            req_rows=req if explicit else None,
            req_pool_indices=req_pool_indices,
            k_slots=prefill_k_slots(
                req_to_token=self.req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens_cpu=seq_lens_cpu,
                ratio=ratio,
            ),
            index_plans=None,
            kv_page_table=torch.zeros(rows, 1, dtype=torch.int32),
            seq_lens_cpu=seq_lens_cpu,
            rows_per_request=counts,
            rows_per_request_device=(
                torch.tensor(counts, dtype=torch.int32) if device_counts else None
            ),
            out_raw_indices=torch.full((rows + 2, 3), 99, dtype=torch.int32),
            out_page_indices=torch.full((rows + 2, 3), 99, dtype=torch.int32),
        )

    def _check(self, inputs, req):
        self.backend.topk_prefill(inputs)
        expected_raw = torch.full_like(inputs.out_raw_indices, -1)
        expected_page = torch.full_like(inputs.out_page_indices, -1)
        for row, (request, position) in enumerate(
            zip(req.tolist(), inputs.positions.tolist())
        ):
            visible = (position + 1) // inputs.compress_ratio
            picks = torch.arange(max(0, visible - 3), visible)
            expected_raw[row, : len(picks)] = picks.to(torch.int32)
            slots = (
                self.req_to_token[request, picks * inputs.compress_ratio]
                // inputs.compress_ratio
            )
            expected_page[row, : len(picks)] = slots.to(torch.int32)
        torch.testing.assert_close(inputs.out_raw_indices, expected_raw)
        torch.testing.assert_close(inputs.out_page_indices, expected_page)

    def test_interleaved_cp_topk_with_prefix_and_empty_requests(self):
        lengths = [3, 0, 9]
        req = torch.repeat_interleave(torch.tensor([5, 2, 7]), torch.tensor(lengths))
        pos = torch.cat([torch.arange(7, 10), torch.arange(11, 20)])
        for ratio in (1, 2):
            for cp in (2, 8):
                for rank in range(cp):
                    with self.subTest(ratio=ratio, cp=cp, rank=rank):
                        counts = [int((req[rank::cp] == r).sum()) for r in (5, 2, 7)]
                        inputs = self._inputs(
                            req[rank::cp], pos[rank::cp], counts, ratio
                        )
                        self._check(inputs, req[rank::cp])

    def test_small_prompt_has_empty_ranks_and_no_visible_compressed_keys(self):
        req = torch.tensor([5, 7, 7])
        pos = torch.tensor([0, 0, 1])
        for rank in range(8):
            with self.subTest(rank=rank):
                counts = [int((req[rank::8] == r).sum()) for r in (5, 2, 7)]
                self._check(
                    self._inputs(req[rank::8], pos[rank::8], counts, 2), req[rank::8]
                )

    def test_cp_supports_host_row_counts(self):
        req = torch.tensor([5, 7, 7])
        pos = torch.tensor([8, 13, 15])
        self._check(self._inputs(req, pos, [1, 0, 2], 2, device_counts=False), req)

    def test_explicit_request_rows_keep_existing_torch_behavior(self):
        req = torch.tensor([7, 5, 7])
        pos = torch.tensor([12, 8, 14])
        self._check(self._inputs(req, pos, [1, 0, 2], 2, explicit=True), req)


if __name__ == "__main__":
    unittest.main()
