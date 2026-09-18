"""Unit tests for DeepSeek-V4 layer-level RoPE selection."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _ModuleStub(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _RMSNormStub(_ModuleStub):
    def __init__(self, hidden_size, *args, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))


class _RoPEConsumerStub(_ModuleStub):
    def __init__(self, *args, freqs_cis, rotary_emb=None, **kwargs):
        super().__init__()
        self.freqs_cis = freqs_cis
        self.rotary_emb = rotary_emb


class _RotaryEmbeddingStub(_ModuleStub):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.rope_scaling = kwargs["rope_scaling"]


class TestDeepseekV4RoPEPolicy(CustomTestCase):
    @staticmethod
    def _config(compress_ratio):
        return SimpleNamespace(
            hidden_size=16,
            qk_rope_head_dim=64,
            qk_nope_head_dim=64,
            head_dim=128,
            num_attention_heads=2,
            num_key_value_heads=1,
            o_groups=2,
            q_lora_rank=8,
            o_lora_rank=8,
            rms_norm_eps=1e-6,
            compress_ratios=[compress_ratio],
            rope_theta=10_000,
            compress_rope_theta=160_000,
            max_position_embeddings=128,
            rope_scaling={
                "original_max_position_embeddings": 65_536,
                "factor": 16.0,
                "beta_fast": 32,
                "beta_slow": 1,
                "type": "yarn",
            },
        )

    def _make_layer(self, compress_ratio):
        parallel = SimpleNamespace(attn_tp_rank=0, attn_tp_size=1, tp_size=1)
        device = SimpleNamespace(device=torch.device("cpu"))
        with (
            envs.SGLANG_OPT_FUSE_WQA_WKV.override(False),
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.override(False),
            patch.object(deepseek_v4, "_FP8_WO_A_GEMM", False),
            patch.object(deepseek_v4, "_is_hip", False),
            patch.object(deepseek_v4, "_is_npu", False),
            patch.object(deepseek_v4, "is_dsa_enable_prefill_cp", return_value=False),
            patch.object(deepseek_v4, "get_parallel", return_value=parallel),
            patch.object(deepseek_v4, "get_device", return_value=device),
            patch.object(deepseek_v4, "ReplicatedLinear", _ModuleStub),
            patch.object(deepseek_v4, "ColumnParallelLinear", _ModuleStub),
            patch.object(deepseek_v4, "RowParallelLinear", _ModuleStub),
            patch.object(deepseek_v4, "RMSNorm", _RMSNormStub),
            patch.object(
                deepseek_v4,
                "get_rope_wrapper",
                side_effect=lambda *args, **kwargs: _RotaryEmbeddingStub(
                    *args, **kwargs
                ),
            ),
            patch.object(deepseek_v4, "Compressor", _RoPEConsumerStub),
            patch.object(deepseek_v4, "C4Indexer", _RoPEConsumerStub),
            patch.object(deepseek_v4, "RadixAttention", _ModuleStub),
        ):
            return deepseek_v4.MQALayer(
                config=self._config(compress_ratio),
                layer_id=0,
            )

    def test_pure_swa_layer_uses_unscaled_main_rope(self):
        layer = self._make_layer(0)
        expected = precompute_freqs_cis(
            dim=64,
            seqlen=128,
            original_seq_len=0,
            base=10_000,
            factor=16.0,
            beta_fast=32,
            beta_slow=1,
        )

        torch.testing.assert_close(layer.freqs_cis, expected)
        self.assertIsNone(layer.rotary_emb.rope_scaling)
        self.assertIsNone(layer.compressor)
        self.assertIsNone(layer.indexer)

    def test_c4_and_c128_layers_share_yarn_compress_rope(self):
        for compress_ratio in (4, 128):
            with self.subTest(compress_ratio=compress_ratio):
                layer = self._make_layer(compress_ratio)
                expected_compressed = precompute_freqs_cis(
                    dim=64,
                    seqlen=128,
                    original_seq_len=65_536,
                    base=160_000,
                    factor=16.0,
                    beta_fast=32,
                    beta_slow=1,
                )

                torch.testing.assert_close(layer.freqs_cis, expected_compressed)
                self.assertIs(layer.compressor.freqs_cis, layer.freqs_cis)
                self.assertIs(layer.compressor.rotary_emb, layer.rotary_emb)
                self.assertEqual(
                    layer.rotary_emb.rope_scaling["rope_type"], "deepseek_yarn"
                )
                if compress_ratio == 4:
                    self.assertIs(layer.indexer.freqs_cis, layer.compressor.freqs_cis)
                    self.assertIs(layer.indexer.rotary_emb, layer.compressor.rotary_emb)


if __name__ == "__main__":
    unittest.main()
