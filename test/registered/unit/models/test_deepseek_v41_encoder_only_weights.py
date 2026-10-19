import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import deepseek_v4, deepseek_v4_boundary
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4Model,
    DeepseekV41EncoderBoundaryLayer,
    dsv41_encoder_only_prefill_weight_needed,
)
from sglang.srt.models.deepseek_v4_mhc import HcConfig, HcNormed, HcState
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDsv41EncoderOnlyWeights(CustomTestCase):
    def test_weight_filter_keeps_only_boundary_producers(self):
        boundary = 20
        kept = (
            "model.embed_tokens.weight",
            "model.layers.19.mlp.experts.w2_weight",
            "model.layers.20.input_layernorm.weight",
            "model.layers.20.self_attn.compressor.wkv.weight",
            "model.layers.20.self_attn.indexer.wk.weight",
            "model.layers.20.self_attn.indexer.k_norm.weight",
        )
        skipped = (
            "model.layers.20.post_attention_layernorm.weight",
            "model.layers.20.hc_attn_fn",
            "model.layers.20.self_attn.wq_a.weight",
            "model.layers.20.self_attn.indexer.wq_b.weight",
            "model.layers.20.self_attn.indexer.weights_proj.weight",
            "model.layers.20.mlp.experts.w2_weight",
            "model.layers.21.self_attn.wkv.weight",
            "model.norm.weight",
            "model.hc_head_fn",
            "lm_head.weight",
            "mtp.0.ffn.shared_experts.w1.weight",
            "model.mtp.layers.0.ffn.shared_experts.w1.weight",
        )
        self.assertTrue(
            all(
                dsv41_encoder_only_prefill_weight_needed(name, boundary)
                for name in kept
            )
        )
        self.assertTrue(
            all(
                not dsv41_encoder_only_prefill_weight_needed(name, boundary)
                for name in skipped
            )
        )

    @patch("sglang.srt.models.deepseek_v4.get_attn_backend")
    def test_cache_only_boundary_consumes_current_mhc_state(self, backend):
        layer = object.__new__(DeepseekV41EncoderBoundaryLayer)
        nn.Module.__init__(layer)
        layer.hc_cfg = HcConfig(4, 20, 1e-6, 1e-5, 16, True, False)
        layer.input_layernorm = nn.Identity()
        layer.self_attn = SimpleNamespace(
            compress_ratio=1,
            compressor=object(),
            indexer=SimpleNamespace(owns_k=True),
        )
        residual = torch.arange(64, dtype=torch.float32).reshape(1, 4, 16)
        normalized = torch.full((1, 16), 7.0)
        for state, expected in (
            (HcState(residual), residual[:, 0, :]),
            (HcState(residual, input=HcNormed(normalized)), normalized),
        ):
            with self.subTest(precomputed=state.input is not None):
                layer.write_global_cache_only(
                    torch.zeros(1, dtype=torch.long), state, SimpleNamespace()
                )
                call = backend.return_value.forward_low_ratio_sources.call_args
                self.assertTrue(torch.equal(call.kwargs["x"], expected))
                self.assertIsNone(call.kwargs["q_lora"])
                self.assertTrue(call.kwargs["run_compressor"])
                self.assertFalse(call.kwargs["run_indexer"])

        # The model loop must finish the previous layer's deferred MoE update,
        # rather than hand the embedding-time HcState to the cache producer.
        class PreviousLayer(nn.Module):
            engram = None

            def forward(self, hidden_states, forward_batch, **kwargs):
                stream = residual_batch.stream_of(forward_batch)
                stream.check(hidden_states)
                update = deepseek_v4_boundary.HcUpdate(self, False)
                update.post = lambda value, state, **kw: HcState(state.residual + value)
                return stream.record(torch.full_like(residual, 3.0), update)

        layer.input_layernorm = RMSNorm(16)
        model = object.__new__(DeepseekV4Model)
        nn.Module.__init__(model)
        model.pp_group = SimpleNamespace(world_size=1)
        model.layers = nn.ModuleList([PreviousLayer(), layer])
        model.start_layer, model.end_layer = 0, 2
        model.encoder_only_boundary_layer = 1
        model.engram_hasher = model.late_layer_start = None
        positions = torch.zeros(1, dtype=torch.long)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            input_ids=positions,
            positions=positions,
            residual_stream=None,
        )
        recorder = SimpleNamespace(with_current_layer=lambda _: nullcontext())
        self.addCleanup(residual_batch.clear, batch)
        with (
            patch.object(deepseek_v4, "_is_cuda", True),
            patch.object(deepseek_v4, "is_cp_active", return_value=False),
            patch.object(deepseek_v4, "check_cuda_graph_backend", return_value=False),
            patch.object(
                deepseek_v4,
                "get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
        ):
            output = model._forward_layers_hc_pre_from_prev(
                positions, residual, batch, positions, positions, False, []
            )
        self.assertEqual(output, (None, None, None))
        actual = backend.return_value.forward_low_ratio_sources.call_args.kwargs["x"]
        torch.testing.assert_close(actual, layer.input_layernorm((residual + 3)[:, 0]))


if __name__ == "__main__":
    unittest.main()
