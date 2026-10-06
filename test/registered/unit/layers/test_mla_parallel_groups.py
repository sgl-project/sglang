"""MLA projections retain their attention partition across reload scopes."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


def build_mla(model, variant, reduce_results=False, width=8, quant_config=None):
    heads, head_dim = 8, width
    rope_scaling = {
        "type": "yarn",
        "rope_type": "deepseek_yarn",
        "factor": 2.0,
        "original_max_position_embeddings": 16,
    }
    if model == "deepseek_v4":
        from sglang.srt.models.deepseek_v4 import MqaAttentionBase

        config = SimpleNamespace(
            hidden_size=width,
            head_dim=head_dim,
            qk_rope_head_dim=4,
            num_attention_heads=heads,
            num_key_value_heads=1,
            o_groups=4,
            q_lora_rank=width,
            o_lora_rank=width,
            rms_norm_eps=1e-6,
            q_head_norm=True,
            compress_ratios=[0],
            rope_theta=10000,
            max_position_embeddings=16,
            rope_scaling=None,
        )
        return MqaAttentionBase(
            config,
            0,
            quant_config,
            "",
            fuse_wqa_wkv=variant,
            wo_b_reduce_results=reduce_results,
        )
    if model == "dots3":
        from sglang.srt.configs.dots3 import Dots3Config
        from sglang.srt.models.dots3_common.modeling import Dots3AttentionMLA

        config = Dots3Config(
            hidden_size=width,
            num_hidden_layers=1,
            max_position_embeddings=16,
            layer_types=[variant],
            rope_scaling=rope_scaling,
            num_attention_heads=heads,
            num_key_value_heads=heads,
            q_lora_rank=width,
            kv_lora_rank=width,
            qk_nope_head_dim=head_dim - 4,
            qk_rope_head_dim=4,
            v_head_dim=head_dim,
            swa_num_attention_heads=heads,
            swa_num_key_value_heads=heads,
            swa_q_lora_rank=width,
            swa_kv_lora_rank=width,
            swa_qk_nope_head_dim=head_dim - 4,
            swa_qk_rope_head_dim=4,
            swa_v_head_dim=head_dim,
        )
        return Dots3AttentionMLA(
            config, quant_config=quant_config, reduce_results=reduce_results, layer_id=0
        )
    config = SimpleNamespace(
        rms_norm_eps=1e-6,
        qk_nope_head_dim=head_dim - 4,
        qk_rope_head_dim=4,
        v_head_dim=head_dim,
        q_lora_rank=variant,
        kv_lora_rank=width,
    )
    if model == "deepseek_v2":
        from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

        return DeepseekV2AttentionMLA(
            config,
            width,
            heads,
            head_dim - 4,
            4,
            head_dim,
            variant,
            width,
            rope_scaling=rope_scaling,
            layer_id=0,
            max_position_embeddings=16,
            quant_config=quant_config,
            reduce_results=reduce_results,
        )
    from sglang.srt.models.sarvam_moe import SarvamMoEMLAAttention

    return SarvamMoEMLAAttention(
        config,
        width,
        heads,
        layer_id=0,
        max_position_embeddings=16,
        quant_config=quant_config,
    )


class TestMLAParallelGroups(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=4, attn_dp_size=2),
            role="test",
            ranks=SpawnRanks(world_rank=3),
        )

    def test_model_projections_reload_and_keep_row_policy(self):
        variants = {
            "deepseek_v2": (None, 8),
            "sarvam": (None, 8),
            "dots3": ("full_attention", "sliding_attention"),
            "deepseek_v4": (False, True),
        }
        tp = Mock()
        tp.all_reduce.side_effect = lambda x: x * 2
        for model, choices in variants.items():
            for variant in choices:
                for reduce in (False, True):
                    with self.subTest(model=model, variant=variant, reduce=reduce):
                        attention = build_mla(model, variant, reduce)
                        names = (
                            ("wq_b", "wo_a", "wo_b")
                            if model == "deepseek_v4"
                            else (
                                "q_b_proj"
                                if variant is not None or model == "dots3"
                                else "q_proj",
                                "kv_b_proj",
                                "o_proj",
                            )
                        )
                        for name in names:
                            layer = getattr(attention, name)
                            is_row = name in ("o_proj", "wo_b")
                            dtype = layer.weight.dtype
                            weight = (
                                torch.arange(
                                    layer.output_size * layer.input_size,
                                    dtype=torch.float32,
                                ).reshape(layer.output_size, layer.input_size)
                                % 17
                                / 32
                            ).to(dtype)
                            with get_parallel().override(
                                tp_rank=0, attn_dp_rank=0, attn_tp_rank=0
                            ):
                                layer.weight.weight_loader(layer.weight, weight)
                            shard = weight.chunk(2, dim=1 if is_row else 0)[1]
                            torch.testing.assert_close(layer.weight, shard)
                            x = (
                                torch.arange(2 * shard.shape[1], dtype=torch.float32)
                                .reshape(2, -1)
                                .to(dtype)
                                / 16
                            )
                            expected = F.linear(x, shard)
                            if is_row and layer.reduce_results:
                                expected *= 2
                            tp.all_reduce.reset_mock()
                            with (
                                get_parallel().override(tp_group=tp),
                                patch(
                                    "sglang.srt.layers.linear.is_allocation_symmetric",
                                    return_value=True,
                                ),
                                patch(
                                    "sglang.srt.layers.linear.use_symmetric_memory",
                                    return_value=nullcontext(),
                                ) as allocator,
                            ):
                                torch.testing.assert_close(layer(x)[0], expected)
                                if is_row:
                                    allocator.assert_called_once_with(
                                        tp, disabled=False
                                    )
                                    self.assertEqual(
                                        tp.all_reduce.call_count,
                                        int(layer.reduce_results),
                                    )
                                else:
                                    allocator.assert_not_called()
                            self.assertEqual((layer.tp_rank, layer.tp_size), (1, 2))
                            if is_row:
                                self.assertFalse(layer.use_dp_attention_reduce)
                                self.assertEqual(
                                    layer.reduce_results,
                                    reduce if model != "sarvam" else False,
                                )
                        if model == "deepseek_v4":
                            self.assertEqual(
                                attention.wo_a.weight.dtype, torch.bfloat16
                            )
                            self.assertEqual(attention.n_local_groups, 2)


if __name__ == "__main__":
    unittest.main()
