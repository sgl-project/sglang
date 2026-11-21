"""KPool's complete history includes an unpooled tail and physical page remapping."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsa.kpool_short_decode import (
    kpool_short_decode_indices,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention import dsa_backend
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    topk_from_pooled_history_logits,
)
from sglang.srt.layers.attention.graph_variants import (
    KPOOL_SCORED_DECODE,
    KPOOL_SHORT_DECODE,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")


class TestKPoolShortDecode(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "Test requires CUDA")
    def test_paged_indices_survive_replay_and_request_reordering(self):
        target = 2048
        kpool = 4
        width = target + kpool - 1
        bs = 7
        device = "cuda"
        lens = torch.ones(bs, dtype=torch.int32, device=device)
        # Preserve a nontrivial row stride and physical (not logical) slots.
        table = torch.empty((bs, 2304), device=device, dtype=torch.int32)[:, :2176]
        scores = torch.randn((bs, 544), device=device, dtype=torch.float32)
        kpool_short_decode_indices(table, lens, bs + 1, width)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = kpool_short_decode_indices(table, lens, bs + 1, width)

        for lengths in ([0, 1, 2, 3, 4, 2047, 2048], [2048, 176, 513, 1023, 16, 4, 1]):
            lens.copy_(lens.new_tensor(lengths))
            table.copy_(torch.randperm(table.numel(), device=device).view_as(table))
            scores.normal_()
            expected = topk_from_pooled_history_logits(
                logits=scores,
                group_lengths=lens // kpool,
                kpool=kpool,
                topk=target,
                page_table=table,
                seq_lens=lens,
                out_rows=bs + 1,
            )
            graph.replay()
            torch.testing.assert_close(out, expected, atol=0, rtol=0)
            for i, n in enumerate(lengths):
                torch.testing.assert_close(out[i, :n], table[i, :n], atol=0, rtol=0)
                self.assertTrue((out[i, n:] == -1).all())
            self.assertTrue((out[-1] == -1).all())

    def test_longest_live_sequence_selects_graph(self):
        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        backend.device_sm_major = 9
        backend.qk_rope_head_dim = 0
        backend.dsa_index_kpool = 4
        backend.dsa_index_topk = 2048
        backend.physical_page_size = 64
        backend.attn_dcp_size = 1
        backend.hisparse_coordinator = None
        backend.dsa_decode_impl = "tilelang"
        backend.kv_cache_dtype = torch.bfloat16
        backend.use_fused_topk = True
        self.enterContext(patch.object(dsa_backend, "is_cuda", return_value=True))
        self.enterContext(get_parallel().override(attn_dp_size=1))
        spec = SimpleNamespace(speculative_algorithm=None)
        self.enterContext(patch.object(dsa_backend, "get_spec", return_value=spec))
        switch = envs.SGLANG_ENABLE_DSA_KPOOL_SHORT_DECODE
        self.enterContext(switch.override(True))
        policy = backend.get_decode_graph_variants(ForwardMode.DECODE, 1)
        for lens, expected in [
            ([1, 2048], KPOOL_SHORT_DECODE),
            ([2049, 1], KPOOL_SCORED_DECODE),
            ([], KPOOL_SCORED_DECODE),
            (None, KPOOL_SCORED_DECODE),
        ]:
            batch = SimpleNamespace(
                seq_lens_cpu=None if lens is None else torch.tensor(lens)
            )
            self.assertEqual(policy.select(batch), expected)
        # Unfused attention expects logical indices; verification has more
        # than one causal row per request. Neither can use this page table.
        with patch.object(backend, "use_fused_topk", False):
            self.assertIsNone(backend.get_decode_graph_variants(ForwardMode.DECODE, 1))
        with patch.object(backend, "hisparse_coordinator", object()):
            self.assertIsNone(backend.get_decode_graph_variants(ForwardMode.DECODE, 1))
        with patch.object(backend, "qk_rope_head_dim", 64):
            self.assertIsNone(backend.get_decode_graph_variants(ForwardMode.DECODE, 1))
        self.assertIsNone(
            backend.get_decode_graph_variants(ForwardMode.TARGET_VERIFY, 6)
        )
        with switch.override(False):
            self.assertIsNone(backend.get_decode_graph_variants(ForwardMode.DECODE, 1))


if __name__ == "__main__":
    unittest.main()
