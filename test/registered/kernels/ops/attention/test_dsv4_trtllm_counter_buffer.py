"""The DSV4 trtllm-gen backend's persistent multi-CTA KV counter buffer.

One zeroed buffer serves every launch only if each launch leaves its counters at
zero, so outputs must equal FlashInfer's per-call counters bit for bit across
layers and CUDA-graph replays that share it.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

# c32 TP4 DSV4.1-Flash decode: 36 rows of 16 heads with a 512-wide tail split KV
# across CTAs, so the kernel takes and must release the counters.
ROWS, HEADS, TAIL, LAYERS, SWA = 36, 16, 512, 4, 128
SCALE = 512**-0.5


def _fp8(*shape):
    return torch.randn(*shape, device="cuda").to(torch.float8_e4m3fn)


def _indices(*shape):
    return torch.randint(0, 256 * 64, shape, dtype=torch.int32, device="cuda")


def _int32(*shape):
    return torch.empty(*shape, dtype=torch.int32, device="cuda")


def _counter_buffer(rows, heads):
    """The backend's own allocation, with fresh buffers and stubbed sizing inputs."""
    from sglang.srt.layers.attention import deepseek_v4_trtllm_backend as trtllm
    from sglang.srt.runtime_context import get_context

    runner = SimpleNamespace(model_config=SimpleNamespace(num_attention_heads=heads))
    with (
        get_context().resources.override(buffers={}),
        mock.patch.object(trtllm, "_trtllm_query_row_capacity", return_value=rows),
    ):
        return trtllm._get_trtllm_counter_buffer(runner, torch.device("cuda"))


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "trtllm-gen DSV4 sparse MLA requires SM10x",
)
class TestDSV4TrtllmCounterBuffer(CustomTestCase):
    def setUp(self):
        from sglang.srt.layers.attention import deepseek_v4_trtllm_backend as trtllm

        torch.manual_seed(1020)
        self.kv = (_fp8(256, 1, 64, 512), _fp8(256, 1, 64, 512))
        self.sink = torch.randn(HEADS, device="cuda")
        self.layer = SimpleNamespace(layer_id=0, k_scale_float=None)
        b = object.__new__(trtllm.DeepseekV4TrtllmAttnBackend)
        b.softmax_scale = SCALE
        b._tail_refresh_layers = None  # every indexed layer refreshes the tail
        b.trtllm_workspace_buffer = torch.zeros(
            128 << 20, dtype=torch.uint8, device="cuda"
        )
        b.trtllm_counter_buffer = _counter_buffer(64, 64)
        b._trtllm_kv_cache_views = lambda layer_id, compress_ratio: self.kv
        self.backend = b
        # The reference lets FlashInfer zero fresh counters on every call.
        self.ref_workspace = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")

    def _reference(self, q, table, lens, seq_lens, **varseq):
        from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

        out = trtllm_batch_decode_sparse_mla_dsv4(
            query=q if varseq else q.view(q.shape[0], 1, HEADS, 512),
            swa_kv_cache=self.kv[0],
            workspace_buffer=self.ref_workspace,
            sparse_indices=table,
            compressed_kv_cache=self.kv[1],
            sparse_topk_lens=lens,
            seq_lens=seq_lens,
            bmm1_scale=SCALE,
            sinks=self.sink,
            backend="trtllm-gen",
            **varseq,
        )
        return out.view(-1, HEADS, 512)

    def _assert_counters_zero(self):
        torch.cuda.synchronize()
        self.assertFalse(self.backend.trtllm_counter_buffer.any().item())

    def _assert_call_uses_backend_buffer(self, forward):
        # FlashInfer validates a caller buffer; without one it allocates its own.
        saved = self.backend.trtllm_counter_buffer
        self.backend.trtllm_counter_buffer = torch.zeros(
            16, dtype=torch.uint8, device="cuda"
        )
        with self.assertRaisesRegex(ValueError, "too small"):
            forward()
        self.backend.trtllm_counter_buffer = saved

    def test_buffer_starts_zeroed_and_covers_rows_and_heads(self):
        from flashinfer.utils import (
            get_device_sm_count,
            get_trtllm_gen_multi_ctas_kv_counter_bytes,
        )

        # Above the helper's 8192-row and 64-head floors.
        rows, heads = 16384, 128
        need = get_trtllm_gen_multi_ctas_kv_counter_bytes(
            rows, heads, get_device_sm_count(torch.device("cuda"))
        )
        # The private pool's only free block is filled with 0xFF and goes to the
        # allocation; the kernel spins on counters that do not start at zero.
        with torch.cuda.use_mem_pool(torch.cuda.MemPool()):
            poison = torch.full((need,), 0xFF, dtype=torch.uint8, device="cuda")
            poison_ptr = poison.data_ptr()
            del poison
            buffer = _counter_buffer(rows, heads)
            self.assertGreaterEqual(buffer.numel(), need)
            self.assertEqual(buffer.data_ptr(), poison_ptr)
            torch.cuda.synchronize()
            self.assertFalse(buffer.any().item())
            del buffer

    def test_decode_layers_and_graph_replays_share_counters(self):
        # One token per request runs dense decode; an MTP verify of 4 runs VarSeq.
        for q_len in (1, 4):
            with self.subTest(q_len=q_len):
                self._check_decode(q_len)

    def _check_decode(self, q_len):
        from sglang.kernels.ops.attention.dsv4.trtllm_metadata import (
            pack_sparse_tail,
        )

        n_req = ROWS // q_len
        # Tile-padded tables as TrtllmSparseTablePool hands them out.
        table = torch.full((64, SWA + TAIL), -1, dtype=torch.int32, device="cuda")
        lens = torch.full((64,), SWA, dtype=torch.int32, device="cuda")
        seq_lens_req = _int32(n_req)
        offsets = torch.arange(q_len, dtype=torch.int32, device="cuda").repeat(n_req)
        cum = torch.arange(0, ROWS + 1, q_len, dtype=torch.int32, device="cuda")
        tail_lens = _int32(ROWS)
        core = SimpleNamespace(
            seq_lens_casual=_int32(ROWS),
            swa_page_indices=table[:ROWS, :SWA],
            trtllm_c4_indices=table[:ROWS],
            trtllm_c4_lens=lens[:ROWS],
            trtllm_seq_lens_req=seq_lens_req if q_len > 1 else None,
            trtllm_cum_seq_lens_q=cum if q_len > 1 else None,
            _c4_topk_writes_table=lambda: False,
            sparse_topk_lengths=lambda compress_ratio: tail_lens,
        )
        q = [_fp8(ROWS, HEADS, 512) for _ in range(LAYERS)]
        tail = [_indices(ROWS, TAIL) for _ in range(LAYERS)]

        def refresh():
            table[:ROWS, :SWA].copy_(_indices(ROWS, SWA))
            seq_lens_req.random_(SWA + q_len, 1 << 16)
            # Each request's per-token causal lengths end at its KV length.
            core.seq_lens_casual.copy_(
                seq_lens_req.repeat_interleave(q_len) - (q_len - 1) + offsets
            )
            tail_lens.random_(1, TAIL + 1)
            for i in range(LAYERS):
                q[i].copy_(_fp8(ROWS, HEADS, 512))
                tail[i].copy_(_indices(ROWS, TAIL))

        def forward():
            return [
                self.backend._forward_trtllm_decode(
                    q=q[i],
                    layer=self.layer,
                    compress_ratio=4,
                    core_attn_metadata=core,
                    attn_sink=self.sink,
                    extra_indices=tail[i],
                )
                for i in range(LAYERS)
            ]

        def check(outs):
            varseq = dict(cum_seq_lens_q=cum, max_q_len=q_len) if q_len > 1 else {}
            seq_lens = seq_lens_req if q_len > 1 else core.seq_lens_casual
            for i, out in enumerate(outs):
                pack_sparse_tail(tail[i], tail_lens, table[:ROWS], lens[:ROWS])
                ref = self._reference(
                    q[i], table[:ROWS], lens[:ROWS], seq_lens, **varseq
                )
                self.assertTrue(torch.equal(out, ref))
            self._assert_counters_zero()

        refresh()
        check(forward())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outs = forward()
        for _ in range(3):
            refresh()
            graph.replay()
            check(graph_outs)
        self._assert_call_uses_backend_buffer(forward)

    def test_prefill_shares_counters(self):
        # Requests of 3 and 9 tokens, one dense row per token: the c4 layer packs
        # its top-k into the shared table, then attends.
        sum_q = 12
        seq_lens = torch.cat([torch.arange(4097, 4100), torch.arange(8193, 8202)]).int()
        seq_lens = seq_lens.cuda()
        table = torch.full((64, SWA + TAIL), -1, dtype=torch.int32, device="cuda")
        lens = torch.full((64,), SWA, dtype=torch.int32, device="cuda")
        tail_lens = _int32(sum_q)
        core = SimpleNamespace(
            trtllm_prefill_qmeta=(sum_q, seq_lens),
            trtllm_prefill_swa_indices=table[:sum_q, :SWA],
            trtllm_prefill_c4_indices=table[:sum_q],
            trtllm_prefill_c4_lens=lens[:sum_q],
            trtllm_tail_needs_pack=False,
            _c4_topk_writes_table=lambda: False,
            sparse_topk_lengths=lambda compress_ratio: tail_lens,
        )
        self.backend.forward_metadata = SimpleNamespace(core_attn_metadata=core)
        self.q = _fp8(sum_q, HEADS, 512)
        self.extra = _indices(sum_q, TAIL)

        def forward():
            return self.backend._forward_trtllm_prefill(
                q=self.q,
                layer=self.layer,
                compress_ratio=4,
                forward_batch=None,
                attn_sink=self.sink,
                extra_indices=self.extra,
            )

        for _ in range(2):
            table[:sum_q, :SWA].copy_(_indices(sum_q, SWA))
            tail_lens.random_(1, TAIL + 1)
            self.q.copy_(_fp8(sum_q, HEADS, 512))
            self.extra.copy_(_indices(sum_q, TAIL))
            # The table and lens attention must read, built without the backend.
            ref_table = table[:sum_q].clone()
            ref_table[:, SWA:] = self.extra
            out = forward()
            ref = self._reference(self.q, ref_table, tail_lens + SWA, seq_lens)
            self.assertTrue(torch.equal(out, ref))
            self._assert_counters_zero()
        self._assert_call_uses_backend_buffer(forward)


if __name__ == "__main__":
    unittest.main()
