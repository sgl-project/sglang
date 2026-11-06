"""H20 Torch fallback consumes rank-local CP request ownership."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.dsv4.v41_indexer.full_topk import FullTopKIndexer
from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import prefill_k_slots
from sglang.srt.layers.attention.dsv4.v41_indexer.types import PrefillInputs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestTorchCPIndexer(CustomTestCase):
    def test_unequal_local_rows_empty_request_causality_and_pool_slots(self):
        for ratio in (1, 2):
            with self.subTest(compress_ratio=ratio):
                # Request 3 has no queries on this CP rank; pool IDs are sparse.
                reqs = torch.tensor([1, 3, 5])
                positions = torch.tensor([0, 4, 2])
                table = torch.arange(6 * 16).reshape(6, 16) * ratio
                raw = torch.full((3, 6), 999, dtype=torch.int32)
                pages = raw.clone()
                seq_lens = [5, 7, 3]

                def keys(layer_id, slots):
                    return torch.arange(len(slots)).float().view(-1, 1)

                indexer = SimpleNamespace(
                    index_topk=6,
                    queries=lambda q, freqs: q,
                    head_weights=lambda x: x,
                    scores=lambda q, k, weights: k.T.expand(len(q), -1),
                )
                inputs = PrefillInputs(
                    indexer=indexer,
                    layer_id=0,
                    compress_ratio=ratio,
                    freqs_cis=torch.ones(16, 1),
                    x=torch.ones(3, 1),
                    q_lora=torch.ones(3, 1),
                    positions=positions,
                    req_rows=None,
                    req_pool_indices=reqs,
                    kv_page_table=torch.empty(0),
                    k_slots=prefill_k_slots(
                        req_to_token=table,
                        req_pool_indices=reqs,
                        seq_lens_cpu=seq_lens,
                        ratio=ratio,
                    ),
                    index_plans=None,
                    seq_lens_cpu=seq_lens,
                    rows_per_request=[2, 0, 1],
                    rows_per_request_device=torch.tensor([2, 0, 1]),
                    out_raw_indices=raw,
                    out_page_indices=pages,
                )
                backend = object.__new__(FullTopKIndexer)
                backend.token_to_kv_pool = SimpleNamespace(
                    get_low_ratio_index_k_dequant=keys
                )
                backend.req_to_token = table
                backend._torch_prefill(inputs)
                for row, req in enumerate((1, 1, 5)):
                    visible = (int(positions[row]) + 1) // ratio
                    chosen = set(raw[row].tolist()) - {-1}
                    self.assertEqual(chosen, set(range(visible)))
                    expected_slots = {
                        int(table[req, i * ratio]) // ratio for i in chosen
                    }
                    self.assertEqual(set(pages[row].tolist()) - {-1}, expected_slots)
                    self.assertEqual(int((raw[row] == -1).sum()), 6 - visible)


if __name__ == "__main__":
    unittest.main()
