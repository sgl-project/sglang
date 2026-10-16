import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention.dsv4 import low_ratio_backend_hip as m
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=3, suite="stage-b-test-1-gpu-small-amd-mi35x")

TOPK = 8


def _run_extend(rows_per_request, rows_per_chunk):
    """Run the HIP low-ratio extend indexer with the scoring and top-k kernels
    replaced; returns the row counts scored and the published candidate masks."""
    num_tokens = sum(rows_per_request)
    num_reqs = len(rows_per_request)
    indexer_metadata = SimpleNamespace(
        page_table=torch.zeros((num_tokens, 4), dtype=torch.int32),
        compressed_seq_lens=torch.ones(num_tokens, dtype=torch.int32),
        compressed_page_size=64,
    )
    page_indices = torch.zeros((num_tokens, TOPK), dtype=torch.int32)
    backend = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(
            get_index_k_fp4_payload_buffer=lambda layer_id: None,
            get_index_k_fp4_scale_buffer=lambda layer_id: None,
        ),
        forward_metadata=SimpleNamespace(
            core_metadata=SimpleNamespace(
                sparse_page_indices=lambda ratio: page_indices,
                sparse_raw_indices=lambda ratio: None,
            ),
            late_layer_tail=None,
            low_ratio_indexer_metadata=lambda ratio: indexer_metadata,
            fp4_low_ratio_prefill_workspaces={},
        ),
        low_ratio_identity_skip=False,
        candidate_masks=None,
    )
    layer = SimpleNamespace(
        compress_ratio=1,
        layer_id=0,
        indexer=SimpleNamespace(
            is_candidate_source=True, uses_candidates=False, index_topk=TOPK
        ),
    )
    forward_batch = SimpleNamespace(seq_lens_cpu=[1000] * num_reqs)
    scored_rows = []

    def fake_logits(**kw):
        scored_rows.append(kw["q_fp4"].shape[0])
        return torch.zeros((kw["q_fp4"].shape[0], 16))

    def fake_select(**kw):
        kw["publish"].append(("mask", kw["page_indices"].shape[0]))

    inputs = (
        torch.zeros((num_tokens, 64, 64), dtype=torch.uint8),
        torch.zeros((num_tokens, 1, 4, 16, 4), dtype=torch.uint8),
        torch.zeros((num_tokens, 64)),
    )
    with (
        mock.patch.object(m, "_indexer_inputs", return_value=inputs),
        mock.patch.object(m, "logits_rows_per_chunk", return_value=rows_per_chunk),
        mock.patch.object(m, "aiter_fp4_paged_mqa_logits", side_effect=fake_logits),
        mock.patch.object(m, "_select_topk_extend_hip", side_effect=fake_select),
        mock.patch.object(
            m, "cat_candidate_blocks", side_effect=lambda pieces: ("cat", pieces)
        ),
    ):
        m.low_ratio_index_topk_hip_extend(
            backend,
            layer,
            x=None,
            q_lora=None,
            pos=torch.arange(990, 990 + num_tokens),
            forward_batch=forward_batch,
            rows_per_request=rows_per_request,
        )
    return scored_rows, backend.candidate_masks


class TestLowRatioZeroRowGroup(CustomTestCase):
    def test_zero_row_group_is_not_scored(self):
        """Prefill CP: a one-token extend owned by the other rank, next to a request
        wider than the logits block, formed a group with no rows here; scoring it
        launched the fp4 prefill logits kernel on 0 rows (GPU fault at address 0)."""
        self.assertEqual(m._request_groups([0, 3], 2), [(0, 1, 0, 0), (1, 2, 0, 3)])
        scored_rows, masks = _run_extend([0, 3], rows_per_chunk=2)
        self.assertEqual(scored_rows, [2, 1])
        self.assertEqual(masks, [None, ("cat", [("mask", 2), ("mask", 1)])])

    def test_zero_row_request_sharing_a_group_is_scored_with_it(self):
        self.assertEqual(m._request_groups([0, 2, 1], 4), [(0, 3, 0, 3)])
        scored_rows, masks = _run_extend([0, 2, 1], rows_per_chunk=4)
        self.assertEqual(scored_rows, [3])
        self.assertEqual(masks, [("mask", 3)])


if __name__ == "__main__":
    unittest.main()
