import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    resolve_layer_indices,
)
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
    DeepseekV41EncoderBoundaryLayer,
    dsv41_encoder_only_prefill_weight_needed,
)
from sglang.srt.models.deepseek_v4_mhc import HcConfig, HcNormed, HcState
from sglang.srt.runtime_context import get_context
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

    @patch("sglang.srt.models.deepseek_v4.EngramLayout.from_config", return_value=None)
    @patch("sglang.srt.models.deepseek_v4.VocabParallelEmbedding")
    @patch("sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer")
    @patch("sglang.srt.models.deepseek_v4.DeepseekV41EncoderBoundaryLayer")
    @patch("sglang.srt.models.deepseek_v4._is_npu", False)
    @patch("sglang.srt.models.deepseek_v4._is_hip", False)
    @patch("sglang.srt.models.deepseek_v4._is_cuda", False)
    @patch("sglang.srt.models.deepseek_v4.is_dp_attention_enabled", return_value=False)
    @patch("sglang.srt.models.deepseek_v4.get_parallel")
    @patch("sglang.srt.utils.common.get_parallel")
    def test_model_materializes_only_encoder_layers_and_boundary(
        self,
        get_parallel_common,
        get_parallel,
        _is_dp_attention_enabled,
        boundary_layer,
        decoder_layer,
        embed,
        _engram_layout,
    ):
        get_parallel.return_value.pp_group = SimpleNamespace(
            is_first_rank=True,
            is_last_rank=True,
            rank_in_group=0,
            world_size=1,
        )
        get_parallel_common.return_value = SimpleNamespace(pp_rank=0, pp_size=1)
        embed.side_effect = lambda *_args, **_kwargs: nn.Identity()
        decoder_layer.side_effect = lambda **kw: nn.Identity()
        boundary_layer.side_effect = lambda **kw: nn.Identity()
        config = SimpleNamespace(
            hidden_size=16,
            q_lora_rank=4,
            vocab_size=32,
            rms_norm_eps=1e-5,
            num_hidden_layers=40,
            kv_source_layer_ids=[2, 8, 14, 20],
            hc_eps=1e-6,
            hc_mult=4,
            hc_pre_from_prev_sublayer=True,
            model_type="deepseek_v41",
            vision_n_layers=0,
        )

        with get_context().override_server_args(
            dsv41_encoder_only_prefill=True,
            disaggregation_mode="prefill",
        ):
            model = DeepseekV4Model(config)
            with (
                patch(
                    "sglang.srt.models.deepseek_v4.DeepseekV4Model", return_value=model
                ),
                patch.object(
                    DeepseekV4ForCausalLM, "determine_num_fused_shared_experts"
                ),
                patch("sglang.srt.models.deepseek_v4.LogitsProcessor"),
                patch("sglang.srt.models.deepseek_v4.get_attn_tp_context"),
            ):
                wrapper = DeepseekV4ForCausalLM(config)
            model_config = SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
                num_hidden_layers=40,
                num_attention_layers=40,
                num_nextn_predict_layers=3,
                is_hybrid_swa=False,
                swa_attention_layer_ids=None,
            )
            layer_info = resolve_layer_indices(
                model=wrapper, model_config=model_config, is_draft_worker=False
            )
            # KV configurators consume this runner range, not the original
            # 40-layer checkpoint config or the placeholder ModuleList length.
            self.assertEqual((layer_info.start_layer, layer_info.end_layer), (0, 21))
            self.assertEqual(layer_info.num_effective_layers, 21)

        self.assertEqual(len(model.layers), 40)
        self.assertEqual(model.start_layer, 0)
        self.assertEqual(model.end_layer, 21)
        self.assertEqual(decoder_layer.call_count, 20)
        boundary_layer.assert_called_once()
        self.assertIsInstance(model.layers[21], PPMissingLayer)
        self.assertIsInstance(model.norm, PPMissingLayer)
        self.assertIsNone(model.hc_head_fn)

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


if __name__ == "__main__":
    unittest.main()
