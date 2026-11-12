# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""DFlash draft L2 prefetch (SGLANG_OPT_DFLASH_DRAFT_L2_PREFETCH): the chunk table covers each tensor's bytes, the
collector takes the draft layers' small tensors and split-K TMA maps but no linear weight, and the capture hooks add one
prefetch kernel on a forked branch of the graph without changing what the graph computes."""

import json
import os
import tempfile
import unittest

import torch
import torch.nn as nn

from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")

DEV = "cuda"
HIDDEN, Q_LORA, BLOCK, TAPS, GROUP, HEADS = 7168, 1536, 8, 2, 16, 6


class TestDraftL2PrefetchChunks(unittest.TestCase):
    def test_chunks_cover_each_span_widened_to_16_bytes(self):
        from sglang.kernels.ops.speculative.l2_prefetch import l2_prefetch_chunks

        spans = [(0x1000, 4096), (0x1008, 10), (0x2000, 10000), (0x5004, 0), (0x6001, 4096)]
        # Hand-derived: [addr & ~15, ceil16(addr + bytes)) cut into <= 4096 B pieces; an empty span adds nothing.
        want = [
            (0x1000, 4096),
            (0x1000, 32),
            (0x2000, 4096),
            (0x3000, 4096),
            (0x4000, 1808),
            (0x6000, 4096),
            (0x7000, 16),
        ]
        self.assertEqual(l2_prefetch_chunks(spans, chunk_bytes=4096), want)


class TestDraftL2PrefetchCollect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    @classmethod
    def tearDownClass(cls):
        reset_context()

    def test_collects_small_tensors_and_maps_but_no_linear_weight(self):
        from sglang.srt.layers.layernorm import RMSNorm
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.models.dflash import DFlashGroupedConv, DFlashMLAAttention
        from sglang.srt.speculative.dflash_draft_splitk import (
            draft_l2_prefetch_tensors,
            install_draft_splitk,
        )

        gen = torch.Generator(device=DEV).manual_seed(11)

        def linear(n, k):
            m = ReplicatedLinear(k, n, bias=False, params_dtype=torch.bfloat16).to(DEV)
            with torch.no_grad():
                m.weight.copy_((torch.randn(n, k, device=DEV, generator=gen) * 0.02).to(torch.bfloat16))
            return m

        def conv():
            c = DFlashGroupedConv(HIDDEN, BLOCK, TAPS, GROUP).to(DEV, torch.bfloat16)
            with torch.no_grad():
                c.kernel_projection.weight.normal_(0, 0.02, generator=gen)
            return c

        attn = DFlashMLAAttention.__new__(DFlashMLAAttention)
        nn.Module.__init__(attn)
        attn.q_lora_rank, attn.num_local_heads, attn.qk_head_dim = Q_LORA, HEADS, 192
        attn.fused_qkv_a_proj_with_mqa = linear(2112, HIDDEN)
        attn.q_b_proj = linear(1152, Q_LORA)
        attn.o_proj = linear(HIDDEN, 768)
        attn.has_fused_proj, attn.is_packed_weight, attn.fused_a_gemm_backend = True, False, "auto"
        attn._q_b_proj_verified_shape = True
        attn._use_min_latency_fused_a_gemm = None
        attn._use_min_latency_q_b_gemm = None
        attn.w_kc = torch.randn(HEADS, 128, 512, device=DEV, generator=gen).to(torch.bfloat16)
        attn.w_vc = torch.randn(HEADS, 512, 128, device=DEV, generator=gen).to(torch.bfloat16)
        mlp = nn.Module()
        mlp.gate_up_proj, mlp.down_proj = linear(1792, HIDDEN), linear(HIDDEN, 896)
        mlp.split_k_gate_up_act = None
        layer = nn.Module()
        layer.self_attn, layer.mlp, layer.attention_conv, layer.mlp_conv = attn, mlp, conv(), conv()
        layer.input_layernorm = RMSNorm(HIDDEN, eps=1e-5).to(DEV, torch.bfloat16)
        layer.q_a_layernorm = RMSNorm(Q_LORA, eps=1e-5).to(DEV, torch.bfloat16)
        model = nn.Module()
        model.layers = nn.ModuleList([layer])
        model.norm = RMSNorm(HIDDEN, eps=1e-5).to(DEV, torch.bfloat16)

        self.assertEqual(install_draft_splitk(model), 7)
        got = {t.data_ptr() for t in draft_l2_prefetch_tensors(model)}

        # Independent of the collector: what the layer re-reads besides GEMM weights, and the plugin's own map tensors.
        linears = [attn.fused_qkv_a_proj_with_mqa, attn.q_b_proj, attn.o_proj, mlp.gate_up_proj, mlp.down_proj]
        projections = [lin.quant_method.projection for lin in linears]
        projections += [layer.attention_conv.split_k_projection, layer.mlp_conv.split_k_projection]
        maps = [m for p in projections for m in p._prepared.maps.values()]
        self.assertEqual(len(maps), 7)
        small = [attn.w_kc, attn.w_vc, layer.input_layernorm.weight, layer.q_a_layernorm.weight, model.norm.weight]
        small += [layer.attention_conv.base_kernel, layer.mlp_conv.base_kernel]
        for t in small + maps:
            self.assertIn(t.data_ptr(), got)
        weights = [lin.weight for lin in linears]
        weights += [layer.attention_conv.kernel_projection.weight, layer.mlp_conv.kernel_projection.weight]
        for w in weights:
            self.assertNotIn(w.data_ptr(), got)


def _kernel_streams(trace_path, name):
    events = json.load(open(trace_path))["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    hit = [e["args"]["stream"] for e in kernels if name in e["name"]]
    rest = {e["args"]["stream"] for e in kernels if name not in e["name"]}
    return hit, rest


class TestDraftL2PrefetchHooks(unittest.TestCase):
    def test_hooks_fork_one_prefetch_and_keep_outputs(self):
        from sglang.srt.speculative.dflash_draft_splitk import DraftL2Prefetch

        gen = torch.Generator(device=DEV).manual_seed(5)
        w1 = torch.randn(2048, 1024, device=DEV, generator=gen).to(torch.bfloat16)
        w2 = torch.randn(1024, 2048, device=DEV, generator=gen).to(torch.bfloat16)
        x = torch.randn(8, 1024, device=DEV, generator=gen).to(torch.bfloat16)
        prefetch = DraftL2Prefetch([w1[:16], torch.randn(3000, device=DEV, generator=gen), x])

        def body():
            return torch.relu(x @ w1.T) @ w2.T

        def capture(hooks):
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                body()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                if hooks:
                    prefetch.capture_head_hook(None, None)
                out = body()
                if hooks:
                    prefetch.capture_tail_hook(None, out, None, x.shape[0])
            return graph, out

        g0, out0 = capture(False)
        g1, out1 = capture(True)
        g0.replay()
        g1.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out0, out1))

        with tempfile.TemporaryDirectory() as tmp:
            traces = {}
            for tag, graph in (("off", g0), ("on", g1)):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    graph.replay()
                    torch.cuda.synchronize()
                traces[tag] = os.path.join(tmp, f"{tag}.json")
                prof.export_chrome_trace(traces[tag])
            hit_off, _ = _kernel_streams(traces["off"], "l2_prefetch_kernel")
            hit_on, rest_on = _kernel_streams(traces["on"], "l2_prefetch_kernel")
        self.assertEqual(hit_off, [])
        self.assertEqual(len(hit_on), 1)
        self.assertTrue(rest_on)
        # Fork form: the prefetch runs on its own branch, not on the stream the chain's kernels run on.
        self.assertNotIn(hit_on[0], rest_on)


if __name__ == "__main__":
    unittest.main()
