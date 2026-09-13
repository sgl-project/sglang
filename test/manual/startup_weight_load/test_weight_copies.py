"""GPU graph checks for refreshed weight copies; no server or distributed run.

DCP uses a concatenating two-shard surrogate, not a two-rank collective.
HPC-Ops checks captured scale consumption, not the optional fused MoE kernel.
"""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.model_executor.model_runner_components.replicated_q_proj import (
    prepare_replicated_q_proj,
)
from sglang.srt.model_executor.model_runner_components.startup_weight_load import (
    ModelStorageManifest,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.test_utils import CustomTestCase


def _capture(forward):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()
    graph.replay()
    return graph, output, output.clone()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestStartupWeightCopies(CustomTestCase):
    @torch.no_grad()
    def test_dcp_refreshed_copies_feed_existing_graph(self):
        def make_attention(has_q_b_proj):
            attention = object.__new__(DeepseekV2AttentionMLA)
            nn.Module.__init__(attention)
            attention.w_kc = torch.full(
                (2, 4, 8), 0.01, dtype=torch.float16, device="cuda"
            )
            attention.w_kc_qrep = None
            attention.q_b_proj_qrep_weight = None
            attention.has_q_b_proj = has_q_b_proj
            q_proj = nn.Linear(8, 8, bias=False, dtype=torch.float16, device="cuda")
            q_proj.weight.requires_grad_(False)
            q_proj.weight.fill_(0.02)
            q_proj.quant_method = UnquantizedLinearMethod()
            setattr(attention, "q_b_proj" if has_q_b_proj else "q_proj", q_proj)
            return attention, q_proj

        def forward(attention, hidden):
            query = hidden @ attention.q_b_proj_qrep_weight.t()
            query = query.reshape(3, 4, 4).transpose(0, 1)
            return torch.bmm(query, attention.w_kc_qrep)

        group = SimpleNamespace(
            world_size=2,
            all_gather=lambda tensor, dim: torch.cat((tensor, tensor), dim=dim),
        )
        for has_q_b_proj in (False, True):
            with self.subTest(has_q_b_proj=has_q_b_proj):
                candidate, q_proj = make_attention(has_q_b_proj)
                prepare_replicated_q_proj(model=candidate, dcp_group=group)
                manifest = ModelStorageManifest.capture(candidate)
                pointers = (
                    candidate.w_kc_qrep.data_ptr(),
                    candidate.q_b_proj_qrep_weight.data_ptr(),
                )
                hidden = torch.randn((3, 8), device="cuda", dtype=torch.float16)
                graph, captured, sentinel_output = _capture(
                    lambda: forward(candidate, hidden)
                )

                serial, serial_q_proj = make_attention(has_q_b_proj)
                candidate.w_kc.copy_(torch.randn_like(candidate.w_kc))
                q_proj.weight.copy_(torch.randn_like(q_proj.weight))
                serial.w_kc.copy_(candidate.w_kc)
                serial_q_proj.weight.copy_(q_proj.weight)
                prepare_replicated_q_proj(model=candidate, dcp_group=group)
                prepare_replicated_q_proj(model=serial, dcp_group=group)
                self.assertEqual(manifest.changed_names(candidate), ())
                self.assertEqual(
                    pointers,
                    (
                        candidate.w_kc_qrep.data_ptr(),
                        candidate.q_b_proj_qrep_weight.data_ptr(),
                    ),
                )
                graph.replay()
                torch.testing.assert_close(
                    captured, forward(serial, hidden), rtol=0, atol=0
                )
                self.assertFalse(torch.equal(captured, sentinel_output))

    @torch.no_grad()
    def test_eagle_mapped_head_refresh_feeds_existing_graph(self):
        head = nn.Parameter(torch.zeros((64, 32), device="cuda"), requires_grad=False)
        embed = nn.Parameter(torch.ones_like(head), requires_grad=False)
        draft_model = SimpleNamespace()

        def set_embed_and_head(shared_embed, mapped_head):
            draft_model.embed = shared_embed
            draft_model.head = mapped_head

        draft_model.set_embed_and_head = set_embed_and_head
        worker = object.__new__(EagleDraftWorker)
        worker.speculative_algorithm = SpeculativeAlgorithm.EAGLE
        worker.hot_token_id = torch.arange(0, 64, 4)
        worker.target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(get_embed_and_head=lambda: (embed, head))
            )
        )
        worker.draft_runner = SimpleNamespace(model=draft_model)
        worker.init_lm_head()
        self.assertIs(draft_model.embed, embed)
        mapped_pointer = draft_model.head.data_ptr()
        self.assertNotEqual(mapped_pointer, head.data_ptr())
        hidden = torch.randn((2, 32), device="cuda")
        graph, captured, sentinel_output = _capture(
            lambda: hidden @ draft_model.head.t()
        )

        head.copy_(torch.randn_like(head))
        wrapper = object.__new__(EAGLEWorkerV2)
        wrapper._draft_worker = worker
        wrapper.refresh_startup_weight_load()
        self.assertEqual(draft_model.head.data_ptr(), mapped_pointer)
        graph.replay()
        reference = hidden @ head[worker.hot_token_id].t()
        torch.testing.assert_close(captured, reference, rtol=0, atol=0)
        self.assertFalse(torch.equal(captured, sentinel_output))

    @torch.no_grad()
    def test_hpc_padded_scales_refresh_feeds_existing_graph(self):
        method = object.__new__(Fp8MoEMethod)
        method.block_quant = True
        layer = nn.Module()
        for prefix in ("w13", "w2"):
            layer.register_parameter(
                f"{prefix}_weight_scale_inv",
                nn.Parameter(torch.ones((2, 2, 3), device="cuda"), requires_grad=False),
            )
        method._prepare_hpc_ops_weights(layer)
        manifest = ModelStorageManifest.capture(layer)
        pointers = tuple(buffer.data_ptr() for buffer in layer.buffers())
        hidden = torch.randn((2, 3, 2), device="cuda")
        graph, captured, sentinel_output = _capture(
            lambda: (
                torch.bmm(hidden, layer.hpc_ops_w13_weight_scale)
                + torch.bmm(hidden, layer.hpc_ops_w2_weight_scale)
            )
        )

        for prefix in ("w13", "w2"):
            scale = getattr(layer, f"{prefix}_weight_scale_inv")
            scale.copy_(torch.rand_like(scale) + 2)
        method._prepare_hpc_ops_weights(layer)
        self.assertEqual(manifest.changed_names(layer), ())
        self.assertEqual(
            pointers, tuple(buffer.data_ptr() for buffer in layer.buffers())
        )
        graph.replay()
        reference = sum(
            torch.bmm(
                hidden,
                torch.nn.functional.pad(
                    getattr(layer, f"{prefix}_weight_scale_inv"), (0, 1)
                ),
            )
            for prefix in ("w13", "w2")
        )
        torch.testing.assert_close(captured, reference, rtol=0, atol=0)
        self.assertFalse(torch.equal(captured, sentinel_output))


if __name__ == "__main__":
    unittest.main()
