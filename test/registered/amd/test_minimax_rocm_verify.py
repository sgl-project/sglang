"""Causal verify metadata and routing with no ordinary prefill metadata."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(torch.version.hip, "ROCm verify integration")
class TestMiniMaxROCmVerify(unittest.TestCase):
    def setUp(self):
        from sglang.srt.layers.attention.minimax_sparse_backend import (
            MiniMaxSparseAttnBackend,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        self.backend = MiniMaxSparseAttnBackend.__new__(MiniMaxSparseAttnBackend)
        self.backend.is_hip = True
        self.backend.speculative_num_draft_tokens = 4
        self.backend._linear_verify_meta = None
        self.batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            seq_lens=torch.tensor([127, 2047], dtype=torch.int32, device="cuda"),
            req_pool_indices=torch.tensor([3, 1], dtype=torch.int64, device="cuda"),
            out_cache_loc=torch.arange(8, dtype=torch.int64, device="cuda"),
            extend_seq_lens=None,
            extend_prefix_lens=None,
        )
        self.q = torch.zeros(8, 1, 128, dtype=torch.bfloat16, device="cuda")
        self.seen = []

        def decode(q, k, v, layer, batch, save_kv_cache, **kwargs):
            self.seen.append(batch)
            # Expose the metadata received by the unchanged sparse decoder.
            return None, torch.stack((batch.req_pool_indices, batch.seq_lens), -1)

        self.backend.forward_decode = decode

    def run_verify(self):
        self.backend._init_rocm_linear_verify_metadata(self.batch)
        return self.backend.forward_extend(
            self.q,
            self.q,
            self.q,
            None,
            self.batch,
            idx_q=self.q,
            idx_k=self.q,
            idx_v=None,
        )[1]

    def assert_causal_rows(self, output, expected):
        torch.testing.assert_close(
            output,
            torch.tensor(expected, dtype=torch.int64, device="cuda"),
            rtol=0,
            atol=0,
        )
        self.assertIsNot(self.seen[-1], self.batch)
        self.assertIs(self.seen[-1].out_cache_loc, self.batch.out_cache_loc)
        self.assertIsNone(self.batch.extend_seq_lens)
        self.assertEqual(self.batch.seq_lens.numel(), 2)

    def test_verify_keeps_original_batch_and_cache_locations(self):
        output = self.run_verify()
        self.assert_causal_rows(
            output,
            [
                [3, 128],
                [3, 129],
                [3, 130],
                [3, 131],
                [1, 2048],
                [1, 2049],
                [1, 2050],
                [1, 2051],
            ],
        )
        self.assertEqual(self.batch.seq_lens.tolist(), [127, 2047])

    def test_graph_replay_reads_new_prefixes_and_requests(self):
        self.run_verify()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.run_verify()
        self.batch.seq_lens.copy_(
            torch.tensor([4095, 255], dtype=torch.int32, device="cuda")
        )
        self.batch.req_pool_indices.copy_(
            torch.tensor([2, 0], dtype=torch.int64, device="cuda")
        )
        graph.replay()
        self.assert_causal_rows(
            output,
            [
                [2, 4096],
                [2, 4097],
                [2, 4098],
                [2, 4099],
                [0, 256],
                [0, 257],
                [0, 258],
                [0, 259],
            ],
        )
        self.assertEqual(self.batch.seq_lens.tolist(), [4095, 255])

    def test_missing_metadata_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Missing MiniMax-M3 linear verify"):
            self.backend.forward_extend(
                self.q,
                self.q,
                self.q,
                None,
                self.batch,
                idx_q=self.q,
                idx_k=self.q,
                idx_v=None,
            )


if __name__ == "__main__":
    unittest.main()
