"""A FlashInfer verify graph captured for one ragged token tier replays live
ragged layouts: planned over the captured slots with the tier's full token
count, the real rows match an eager plan of the live layout."""

import unittest

import flashinfer
import torch

from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    build_capture_verify_lens,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")

HEADS, KV_HEADS, DIM = 32, 8, 128
SLOTS, TIER, WIDTH = 8, 16, 8
GRID = [8, 16, 24, 32]
MAX_LEN = 512


def _reference(q, k, v):
    """Causal attention with the queries aligned to the end of the keys."""
    k = k.repeat_interleave(HEADS // KV_HEADS, dim=1).float()
    v = v.repeat_interleave(HEADS // KV_HEADS, dim=1).float()
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) / DIM**0.5
    offset = k.shape[0] - q.shape[0]
    rows = torch.arange(q.shape[0], device=q.device)[:, None]
    cols = torch.arange(k.shape[0], device=q.device)[None, :]
    scores = scores.masked_fill(cols > rows + offset, float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFlashInferRaggedVerifyReplay(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.kv_cache = torch.randn(
            SLOTS * MAX_LEN, 2, 1, KV_HEADS, DIM, device="cuda", dtype=torch.bfloat16
        )
        # Row r of the request pool owns pages [r * MAX_LEN, (r + 1) * MAX_LEN).
        self.req_to_token = torch.arange(
            SLOTS * MAX_LEN, device="cuda", dtype=torch.int32
        ).view(SLOTS, MAX_LEN)
        self.workspace = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
        self.qo_indptr_buf = torch.zeros(SLOTS + 1, dtype=torch.int32, device="cuda")
        self.kv_indptr_buf = torch.zeros(SLOTS + 1, dtype=torch.int32, device="cuda")
        self.kv_indices_buf = torch.zeros(
            SLOTS * MAX_LEN, dtype=torch.int32, device="cuda"
        )
        self.last_page_len = torch.ones(SLOTS, dtype=torch.int32, device="cuda")

    def _plan(self, wrapper, layout, prefix_lens):
        """The CSR args DFlashVerifyInput.generate_attn_arg_prefill builds."""
        bs = layout.bs
        kv_lens = prefix_lens[:bs] + layout.verify_lens
        kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
        kv_indptr[1:] = torch.cumsum(kv_lens, 0)
        kv_indices = torch.cat(
            [self.req_to_token[i, : int(kv_lens[i])] for i in range(bs)]
        )
        wrapper.plan(
            layout.qo_indptr_device,
            kv_indptr,
            kv_indices,
            self.last_page_len[:bs],
            HEADS,
            KV_HEADS,
            DIM,
            1,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )

    def _live(self, verify_lens):
        return RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=verify_lens, device=torch.device("cuda"), grid=GRID
        )

    def test_replay_matches_eager_plan_of_live_layout(self):
        prefix_lens = torch.tensor(
            [37, 5, 300, 64, 129, 1, 250, 17], dtype=torch.int32, device="cuda"
        )
        graph_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace,
            "NHD",
            use_cuda_graph=True,
            backend="fa2",
            qo_indptr_buf=self.qo_indptr_buf,
            paged_kv_indptr_buf=self.kv_indptr_buf,
            paged_kv_indices_buf=self.kv_indices_buf,
            paged_kv_last_page_len_buf=self.last_page_len,
        )
        capture = self._live(
            build_capture_verify_lens(
                num_tokens=TIER, num_slots=SLOTS, num_draft_tokens=WIDTH
            )
        )
        self.assertEqual(capture.graph_num_tokens, TIER)
        self._plan(graph_wrapper, capture, prefix_lens)
        q_buf = torch.zeros(TIER, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
        graph_wrapper.run(q_buf, self.kv_cache)  # warm up outside capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_buf = graph_wrapper.run(q_buf, self.kv_cache)

        eager_workspace = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
        cases = (
            [1, 8, 3],  # fewer live rows than slots: padding rows take the rest
            [2, 1, 1, 2, 1, 1, 1, 1],  # every slot live, below the tier
            [8, 1, 7],  # rows of mixed width
        )
        for verify_lens in cases:
            with self.subTest(verify_lens=verify_lens):
                live = self._live(verify_lens)
                self.assertEqual(live.graph_num_tokens, TIER)
                total = sum(verify_lens)
                q = torch.randn(total, HEADS, DIM, device="cuda", dtype=torch.bfloat16)

                padded = live.padded_to_bucket(padded_bs=SLOTS)
                self.assertEqual(int(padded.qo_indptr_device[-1]), TIER)
                self._plan(graph_wrapper, padded, prefix_lens)
                q_buf.zero_()
                q_buf[:total].copy_(q)
                graph.replay()
                replayed = out_buf[:total].clone()

                eager = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    eager_workspace, "NHD", backend="fa2"
                )
                self._plan(eager, live, prefix_lens)
                expected = eager.run(q, self.kv_cache)
                torch.testing.assert_close(replayed, expected, atol=2e-2, rtol=2e-2)

                start = 0
                for row, width in enumerate(verify_lens):
                    kv_len = int(prefix_lens[row]) + width
                    pages = self.req_to_token[row, :kv_len].long()
                    ref = _reference(
                        q[start : start + width],
                        self.kv_cache[pages, 0, 0],
                        self.kv_cache[pages, 1, 0],
                    )
                    torch.testing.assert_close(
                        replayed[start : start + width].float(),
                        ref,
                        atol=2e-2,
                        rtol=2e-2,
                    )
                    start += width


if __name__ == "__main__":
    unittest.main()
