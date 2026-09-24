import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Indexer
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
    dsv41_encoder_only_prefill_weight_needed,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDsv41EncoderOnlyWeights(unittest.TestCase):
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

    def test_key_only_indexer_refuses_decode_projections(self):
        indexer = object.__new__(DeepseekV41Indexer)
        nn.Module.__init__(indexer)
        indexer.key_only = True
        with self.assertRaisesRegex(RuntimeError, "key-only indexer"):
            indexer.queries(torch.empty(1), torch.empty(1))
        with self.assertRaisesRegex(RuntimeError, "key-only indexer"):
            indexer.head_weights_raw(torch.empty(1))

    @patch("sglang.srt.models.deepseek_v4.EngramLayout.from_config", return_value=None)
    @patch("sglang.srt.models.deepseek_v4.VocabParallelEmbedding")
    @patch("sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer")
    @patch("sglang.srt.models.deepseek_v4.DeepseekV41EncoderBoundaryLayer")
    @patch("sglang.srt.models.deepseek_v4._is_npu", False)
    @patch("sglang.srt.models.deepseek_v4._is_hip", False)
    @patch("sglang.srt.models.deepseek_v4._is_cuda", False)
    @patch("sglang.srt.models.deepseek_v4.is_dp_attention_enabled", return_value=False)
    @patch("sglang.srt.models.deepseek_v4.get_parallel")
    def test_model_materializes_only_encoder_layers_and_boundary(
        self,
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
        embed.side_effect = lambda *_args, **_kwargs: nn.Identity()
        decoder_layer.side_effect = lambda **kw: nn.Identity()
        boundary_layer.side_effect = lambda **kw: nn.Identity()
        config = SimpleNamespace(
            hidden_size=16,
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

        self.assertEqual(len(model.layers), 40)
        self.assertEqual(model.start_layer, 0)
        self.assertEqual(model.end_layer, 21)
        self.assertEqual(decoder_layer.call_count, 20)
        boundary_layer.assert_called_once()
        self.assertIsInstance(model.layers[21], PPMissingLayer)
        self.assertIsInstance(model.norm, PPMissingLayer)
        self.assertIsNone(model.hc_head_fn)

    def test_embed_head_access_fails_for_encoder_only_prefill(self):
        model = object.__new__(DeepseekV4ForCausalLM)
        nn.Module.__init__(model)
        model.encoder_only_prefill = True
        with self.assertRaisesRegex(AttributeError, "encoder-only Prefill"):
            model.get_embed_and_head()
        with self.assertRaisesRegex(AttributeError, "encoder-only Prefill"):
            model.set_embed_and_head(torch.empty(1), torch.empty(1))


if __name__ == "__main__":
    unittest.main()
