"""Persistent multi-CTA KV counters in the DSV4 trtllm-gen backend.

Without a caller-owned counter buffer FlashInfer zeroes a fresh one per call,
an extra node before every attention kernel in a captured decode graph. The
backend passes one persistent buffer instead and relies on the kernel resetting
its counters at the end of each launch, so outputs must match FlashInfer's
per-call buffer bit for bit across launches and graph replays that reuse it.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

# c32 TP4 DSV4.1-Flash decode: 36 query rows of 16 heads, a 512-wide c4 tail.
# These shapes split KV across CTAs; the kernel spins on counters left non-zero.
ROWS, HEADS, TAIL, LAYERS = 36, 16, 512, 4
SWA = 128
POOL_PAGES, PAGE = 256, 64
SCALE = 512**-0.5
CAPACITY_ROWS = 4096


def _fp8(*shape):
    return torch.randn(*shape, device="cuda").to(torch.float8_e4m3fn)


def _indices(*shape):
    return torch.randint(0, POOL_PAGES * PAGE, shape, dtype=torch.int32, device="cuda")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "trtllm-gen DSV4 sparse MLA requires SM10x",
)
class TestDSV4TrtllmCounterBuffer(CustomTestCase):
    def setUp(self):
        from sglang.srt.layers.attention.deepseek_v4_trtllm_backend import (
            DeepseekV4TrtllmAttnBackend,
            _trtllm_counter_bytes,
        )

        torch.manual_seed(1020)
        self.swa_cache = _fp8(POOL_PAGES, 1, PAGE, 512)
        self.compressed_cache = _fp8(POOL_PAGES, 1, PAGE, 512)
        self.sink = torch.randn(HEADS, device="cuda")
        self.layer = SimpleNamespace(layer_id=0, k_scale_float=None)
        backend = object.__new__(DeepseekV4TrtllmAttnBackend)
        backend.softmax_scale = SCALE
        backend.cuda_int32_kwargs = {"device": "cuda", "dtype": torch.int32}
        backend.trtllm_workspace_buffer = torch.zeros(
            128 << 20, dtype=torch.uint8, device="cuda"
        )
        backend.trtllm_counter_buffer = torch.zeros(
            _trtllm_counter_bytes(torch.device("cuda"), CAPACITY_ROWS, HEADS),
            dtype=torch.uint8,
            device="cuda",
        )
        backend._trtllm_kv_cache_views = lambda layer_id, compress_ratio: (
            self.swa_cache,
            self.compressed_cache,
        )
        self.backend = backend
        # The reference gets FlashInfer's per-call counters and its own workspace.
        self.ref_workspace = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")

    def _reference(self, q, table, lens, seq_lens, **varseq):
        from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

        if not varseq:
            q = q.view(q.shape[0], 1, HEADS, 512)
        out = trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=self.swa_cache,
            workspace_buffer=self.ref_workspace,
            sparse_indices=table,
            compressed_kv_cache=self.compressed_cache,
            sparse_topk_lens=lens,
            seq_lens=seq_lens,
            bmm1_scale=SCALE,
            bmm2_scale=1.0,
            sinks=self.sink,
            kv_layout="HND",
            backend="trtllm-gen",
            **varseq,
        )
        return out.view(-1, HEADS, 512)

    def _counters_are_zero(self):
        torch.cuda.synchronize()
        return not self.backend.trtllm_counter_buffer.any().item()

    def test_decode_graph_replays_reuse_counters(self):
        from sglang.kernels.ops.attention.dsv4.trtllm_metadata import (
            pack_sparse_tail,
        )

        # Tile-padded tables as init_trtllm_sparse_buffers builds them.
        table = torch.full((64, SWA + TAIL), -1, dtype=torch.int32, device="cuda")
        lens = torch.full((64,), SWA, dtype=torch.int32, device="cuda")
        core = SimpleNamespace(
            seq_lens_casual=torch.empty(ROWS, dtype=torch.int32, device="cuda"),
            trtllm_c4_indices=table[:ROWS],
            trtllm_c4_lens=lens[:ROWS],
        )
        swa_page_indices = torch.empty(ROWS, SWA, dtype=torch.int32, device="cuda")
        q = [_fp8(ROWS, HEADS, 512) for _ in range(LAYERS)]
        tail = [_indices(ROWS, TAIL) for _ in range(LAYERS)]
        tail_lens = [
            torch.empty(ROWS, dtype=torch.int32, device="cuda") for _ in range(LAYERS)
        ]

        def refresh():
            table[:ROWS, :SWA].copy_(_indices(ROWS, SWA))
            core.seq_lens_casual.random_(SWA, 1 << 16)
            for layer in range(LAYERS):
                q[layer].copy_(_fp8(ROWS, HEADS, 512))
                tail[layer].copy_(_indices(ROWS, TAIL))
                tail_lens[layer].random_(1, TAIL + 1)

        def forward():
            return [
                self.backend._forward_trtllm_decode(
                    q=q[layer],
                    layer=self.layer,
                    compress_ratio=4,
                    core_attn_metadata=core,
                    attn_sink=self.sink,
                    swa_page_indices=swa_page_indices,
                    extra_indices=tail[layer],
                    extra_topk_lengths=tail_lens[layer],
                )
                for layer in range(LAYERS)
            ]

        def reference():
            outs = []
            for layer in range(LAYERS):
                pack_sparse_tail(
                    tail[layer], tail_lens[layer], table[:ROWS], lens[:ROWS]
                )
                outs.append(
                    self._reference(
                        q[layer], table[:ROWS], lens[:ROWS], core.seq_lens_casual
                    )
                )
            return outs

        refresh()
        for out, ref in zip(forward(), reference()):
            self.assertTrue(torch.equal(out, ref))
        self.assertTrue(self._counters_are_zero())

        graph = torch.cuda.CUDAGraph()
        graph.enable_debug_mode()
        with torch.cuda.graph(graph):
            graph_outs = forward()
        ref_graph = torch.cuda.CUDAGraph()
        ref_graph.enable_debug_mode()
        with torch.cuda.graph(ref_graph):
            reference()
        nodes = {}
        with tempfile.TemporaryDirectory() as tmp:
            for name, g in (("backend", graph), ("reference", ref_graph)):
                path = os.path.join(tmp, f"{name}.dot")
                g.debug_dump(path)
                with open(path) as f:
                    dot = f.read()
                # Torch zeroes the per-call buffer with a fill kernel or a memset.
                nodes[name] = dot.count("KERNEL") + dot.count("MEMSET")
        self.assertEqual(nodes["reference"] - nodes["backend"], LAYERS)

        for _ in range(3):
            refresh()
            graph.replay()
            expected = reference()
            for out, ref in zip(graph_outs, expected):
                self.assertTrue(torch.equal(out, ref))
            self.assertTrue(self._counters_are_zero())

    def test_varseq_prefill_reuses_counters(self):
        import sglang.srt.layers.attention.deepseek_v4_trtllm_backend as trtllm

        # Two requests of 3 and 9 query rows, a 512-wide c128 tail.
        q_lens = (3, 9)
        sum_q, width = sum(q_lens), 512
        cum = torch.tensor([0, 3, 12], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([3 + 4096, 9 + 8192], dtype=torch.int32, device="cuda")
        for _ in range(2):
            core = SimpleNamespace(
                trtllm_prefill_qmeta=(cum, max(q_lens), sum_q, seq_lens),
                trtllm_prefill_c128=None,
            )
            self.backend.forward_metadata = SimpleNamespace(core_attn_metadata=core)
            q = _fp8(sum_q, HEADS, 512)
            # Prefill is not graph-captured, and a dropped buffer changes neither
            # the output nor these counters (FlashInfer zeroes its own), so check
            # that the call passes the backend's buffer.
            with mock.patch.object(
                trtllm,
                "trtllm_batch_decode_sparse_mla_dsv4",
                wraps=trtllm.trtllm_batch_decode_sparse_mla_dsv4,
            ) as fi_call:
                out = self.backend._forward_trtllm_prefill(
                    q=q,
                    layer=self.layer,
                    compress_ratio=128,
                    forward_batch=None,
                    attn_sink=self.sink,
                    swa_page_indices=_indices(sum_q, SWA),
                    extra_indices=_indices(sum_q, width),
                    extra_topk_lengths=torch.randint(
                        1, width + 1, (sum_q,), dtype=torch.int32, device="cuda"
                    ),
                )
            self.assertIs(
                fi_call.call_args.kwargs.get("multi_ctas_kv_counter_buffer"),
                self.backend.trtllm_counter_buffer,
            )
            table, lens = core.trtllm_prefill_c128
            ref = self._reference(
                q, table, lens, seq_lens, cum_seq_lens_q=cum, max_q_len=max(q_lens)
            )
            self.assertTrue(torch.equal(out, ref))
            self.assertTrue(self._counters_are_zero())


if __name__ == "__main__":
    unittest.main()
