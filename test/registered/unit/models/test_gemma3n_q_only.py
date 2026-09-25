"""Checkpoint and projection-layout contracts for Gemma3n shared KV layers."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.models.gemma3n_causal import Gemma3nAttention, Gemma3nQOnlyLinear
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestGemma3nQOnly(CustomTestCase):
    def test_checkpoint_query_matches_packed_projection_on_each_tp_rank(self):
        """Dropping K/V must preserve Q rows, including replicated-KV TP ranks."""
        weight = torch.arange(48 * 16, dtype=torch.float32).reshape(48, 16) / 256
        bias = torch.arange(48, dtype=torch.float32) / 16
        hidden = torch.arange(3 * 16, dtype=torch.float32).reshape(3, 16) / 32

        for tp_size in (1, 2, 4):
            for tp_rank in range(tp_size):
                for with_bias in (False, True):
                    for fused in (False, True):
                        with (
                            self.subTest(
                                tp_size=tp_size,
                                tp_rank=tp_rank,
                                bias=with_bias,
                                fused=fused,
                            ),
                            get_context().override_server_args(),
                            get_parallel().override(
                                tp_size=tp_size,
                                tp_rank=tp_rank,
                                attn_tp_size=tp_size,
                                attn_tp_rank=tp_rank,
                                attn_dp_size=1,
                                attn_dp_rank=0,
                                attn_cp_size=1,
                                attn_cp_rank=0,
                                moe_tp_size=tp_size,
                                moe_tp_rank=tp_rank,
                                moe_ep_size=1,
                                moe_ep_rank=0,
                                moe_dp_size=1,
                                moe_dp_rank=0,
                            ),
                        ):
                            query = Gemma3nQOnlyLinear(
                                input_size=16,
                                output_size=32,
                                bias=with_bias,
                                params_dtype=torch.float32,
                            )
                            packed = QKVParallelLinear(
                                hidden_size=16,
                                head_size=4,
                                total_num_heads=8,
                                total_num_kv_heads=2,
                                bias=with_bias,
                                params_dtype=torch.float32,
                            )
                            for layer in (query, packed):
                                self._load_projection(layer, weight, bias, fused)

                            q_size = 32 // tp_size
                            start = tp_rank * q_size
                            expected_weight = weight[start : start + q_size]
                            expected_bias = (
                                bias[start : start + q_size] if with_bias else None
                            )
                            torch.testing.assert_close(query.weight, expected_weight)
                            if with_bias:
                                torch.testing.assert_close(query.bias, expected_bias)
                            query_output, _ = query(hidden)
                            packed_output, _ = packed(hidden)
                            torch.testing.assert_close(
                                query_output,
                                F.linear(hidden, expected_weight, expected_bias),
                            )
                            torch.testing.assert_close(
                                query_output, packed_output[:, :q_size]
                            )

    @staticmethod
    def _load_projection(layer, weight, bias, fused):
        for name, source in (("weight", weight), ("bias", bias)):
            param = getattr(layer, name)
            if param is None:
                continue
            if fused:
                param.weight_loader(param, source)
            else:
                for shard_id, shard in zip(("q", "k", "v"), source.split((32, 8, 8))):
                    param.weight_loader(param, shard, shard_id)

    @staticmethod
    def _config():
        return SimpleNamespace(
            hidden_size=16,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=4,
            attention_bias=True,
            num_hidden_layers=4,
            num_kv_shared_layers=2,
            layer_types=["sliding_attention", "full_attention"] * 2,
            max_position_embeddings=32,
            sliding_window=16,
            rope_parameters={},
            rms_norm_eps=1e-6,
        )

    def test_consumers_preserve_lora_and_sharded_checkpoint_layouts(self):
        """LoRA buffers and direct state copies require the packed QKV layout."""
        weight = torch.arange(48 * 16, dtype=torch.float32).reshape(48, 16) / 256
        bias = torch.arange(48, dtype=torch.float32) / 16
        cases = (
            (0, False, None, "auto", 48),
            (2, False, None, "auto", 32),
            (2, True, None, "auto", 48),
            (2, False, ["adapter"], "auto", 48),
            (2, False, None, "sharded_state", 48),
            (2, False, None, "remote", 48),
            (2, False, None, "remote_instance", 48),
        )
        for layer_id, enable_lora, lora_paths, load_format, output_size in cases:
            with (
                self.subTest(
                    layer_id=layer_id,
                    enable_lora=enable_lora,
                    lora_paths=lora_paths,
                    load_format=load_format,
                ),
                get_context().override_server_args(
                    enable_lora=enable_lora,
                    lora_paths=lora_paths,
                    load_format=load_format,
                ),
                get_parallel().override(tp_size=1, tp_rank=0),
            ):
                attention = Gemma3nAttention(
                    layer_id=layer_id, config=self._config(), max_position_embeddings=32
                )
                if load_format in ("sharded_state", "remote", "remote_instance"):
                    attention.qkv_proj.load_state_dict({"weight": weight, "bias": bias})
                else:
                    self._load_projection(attention.qkv_proj, weight, bias, fused=False)
                torch.testing.assert_close(
                    attention.qkv_proj.weight, weight[:output_size]
                )
                torch.testing.assert_close(attention.qkv_proj.bias, bias[:output_size])

    def test_quantized_consumers_preserve_all_checkpoint_scales(self):
        """Packed FP8 input scales include K/V even though attention reuses KV."""
        with (
            get_context().override_server_args(),
            get_parallel().override(tp_size=1, tp_rank=0),
        ):
            quant_config = Fp8Config(
                is_checkpoint_fp8_serialized=True, activation_scheme="static"
            )
            attention = Gemma3nAttention(
                layer_id=2,
                config=self._config(),
                max_position_embeddings=32,
                quant_config=quant_config,
            )
            for name in ("weight_scale", "input_scale"):
                param = getattr(attention.qkv_proj, name)
                for shard_id, scale in zip(("q", "k", "v"), (0.5, 1.0, 2.0)):
                    param.weight_loader(param, torch.tensor(scale), shard_id)
                torch.testing.assert_close(param, torch.tensor([0.5, 1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
