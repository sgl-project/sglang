"""ROCm MiniMax sparse attention: verify metadata and routing."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd")
register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(torch.version.hip, "ROCm verify integration")
class TestMiniMaxROCmVerify(unittest.TestCase):
    def setUp(self):
        from sglang.srt.layers.attention.minimax_sparse_backend import (
            MiniMaxSparseAttnBackend,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        self.backend = MiniMaxSparseAttnBackend.__new__(MiniMaxSparseAttnBackend)
        self.backend._linear_verify = True
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
        self.backend._init_linear_verify_metadata(self.batch)
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


class TestMiniMaxROCmVerifyIndexerGate(unittest.TestCase):
    def test_m3_verify_takes_grouped_indexer(self):
        """An M3 target (Quark MXFP4, EAGLE3) on gfx950 must score verify rows with the
        grouped indexer."""
        from sglang.srt.layers.attention import minimax_sparse_backend_hip as _hip
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        def init(spec="EAGLE3", score_type="max"):
            backend = SimpleNamespace(
                kv_pool=None, score_type=score_type, speculative_num_draft_tokens=4
            )
            with patch.object(_hip, "is_gfx95_supported", lambda: True):
                _hip.init_sparse_backend(
                    backend, SimpleNamespace(speculative_algorithm=spec)
                )
            return backend

        self.assertTrue(init()._use_grouped_verify_indexer)
        self.assertFalse(init(spec=None)._use_grouped_verify_indexer)
        self.assertFalse(init(score_type="lse")._use_grouped_verify_indexer)

        def num_draft(backend, mode):
            return _hip.verify_num_draft(backend, SimpleNamespace(forward_mode=mode))

        self.assertEqual(num_draft(init(), ForwardMode.TARGET_VERIFY), 4)
        self.assertIsNone(num_draft(init(), ForwardMode.DECODE))
        self.assertIsNone(num_draft(init(spec=None), ForwardMode.TARGET_VERIFY))


@unittest.skipUnless(
    torch.version.hip, "the deferred seq-lens pull is ROCm DSpark glue"
)
class TestDeferredSeqLensPull(unittest.TestCase):
    def test_host_lengths_are_unknown_until_the_pull(self):
        """A decode step resolves its GPU lengths before the worker pulls the host ones;
        until then the previous step's host lengths must not pass for this step's."""
        from sglang.srt.managers.overlap_utils_hip import DeferredSeqLensPull

        lens = torch.tensor([5, 9, 7, 0], dtype=torch.int64, device="cuda")
        publish_ready = torch.cuda.Event()
        future_map = SimpleNamespace(
            new_seq_lens_buf=lens,
            publish_ready=publish_ready,
            req_pool_size=lens.numel(),
            device=lens.device,
            _publish_fresh=True,
        )
        pull = DeferredSeqLensPull(future_map)
        pull.on_publish(lens)
        publish_ready.record()
        batch = SimpleNamespace(
            spec_info=SimpleNamespace(
                future_indices=torch.tensor([2, 1], device="cuda")
            ),
            req_pool_indices_cpu=torch.tensor([2, 1]),
            seq_lens=None,
            seq_lens_cpu=torch.tensor([6, 8]),
            seq_lens_sum=14,
        )
        pull.resolve(batch)
        self.assertEqual(batch.seq_lens.tolist(), [7, 9])
        self.assertIsNone(batch.seq_lens_cpu)
        self.assertIsNone(batch.seq_lens_sum)
        pull.apply(batch)
        self.assertEqual(batch.seq_lens_cpu.tolist(), [7, 9])
        self.assertEqual(batch.seq_lens_sum, 16)


if __name__ == "__main__":
    unittest.main()
