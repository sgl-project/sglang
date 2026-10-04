"""Attention projections preserve checkpoint placement and collective policy."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.dp_attention import initialize_dp_attention_flags
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

ATTENTION_CASES = (
    ("bailing", "head_wise"),
    ("bailing", "element_wise"),
    ("bailing", None),
    ("gigachat", False),
    ("gigachat", True),
    ("hunyuan", None),
    ("hunyuan", "elementwise"),
    ("exaone4", False),
    ("exaone4", True),
    ("mimo", False),
    ("mimo", True),
    ("minimax", "dense"),
    ("minimax", "index_replicated"),
    ("minimax", "index_sharded"),
    ("minimax", "index_no_value"),
    ("mtp", "attention"),
    ("mtp", "moe"),
    ("sdar", False),
    ("sdar", True),
    ("step3", False),
    ("step3", True),
    ("xllm", "gated"),
    ("xllm", "mova"),
)


def build_attention(
    model, variant, width=32, head_dim=8, quant_config=None, start=True
):
    config = SimpleNamespace(
        hidden_size=width,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=head_dim,
        rms_norm_eps=1e-6,
        layer_norm_epsilon=1e-6,
        max_position_embeddings=16,
        rope_theta=10000,
        rope_scaling=None,
        qk_nope_head_dim=head_dim - 4,
        qk_rope_head_dim=4,
        v_head_dim=head_dim,
        q_lora_rank=width,
        kv_lora_rank=width,
        attention_bias=True,
        rope_parameters={
            "rope_theta": 10000,
            "rope_type": "deepseek_yarn",
            "factor": 1.0,
            "original_max_position_embeddings": 16,
        },
        gated_attention_proj_granularity_type=variant,
        gated_attention=bool(variant),
        gating_type=variant,
        use_qk_norm=True,
        qk_norm_type="per_head",
        use_gemma_norm=variant == "index_sharded",
        dtype=torch.get_default_dtype(),
        num_values=4,
        num_values_per_tok=2,
        sparse_attention_config={
            "sparse_num_index_heads": 8 if variant == "index_sharded" else 1,
            "sparse_index_dim": head_dim,
        },
    )
    if model in ("bailing", "gigachat", "hunyuan"):
        if model == "hunyuan":
            from sglang.srt.models.hunyuan_v4 import HYV4Attention

            module = HYV4Attention(config, 0, quant_config=quant_config)
            names = ("linear_gate",)
        else:
            if model == "bailing":
                from sglang.srt.models.bailing_moe_v3 import DsV3MLA

                constructor, name = DsV3MLA, "g_proj"
            else:
                from sglang.srt.models.gigachat35 import GigaChat35AttentionMLA

                constructor, name = GigaChat35AttentionMLA, "attn_gate"
            module = constructor(
                config,
                width,
                8,
                head_dim - 4,
                4,
                head_dim,
                None if model == "gigachat" and variant else width,
                width,
                rope_scaling=config.rope_parameters,
                max_position_embeddings=16,
                quant_config=quant_config,
                reduce_results=False,
                layer_id=0,
            )
            names = (name,) if getattr(module, name, None) is not None else ()
    elif model == "exaone4":
        from sglang.srt.models.exaone4 import Exaone4Attention

        module = Exaone4Attention(
            config,
            width,
            8,
            1 if variant else 4,
            head_dim=head_dim,
            max_position_embeddings=16,
            quant_config=quant_config,
            bias=True,
            bias_o_proj=True,
        )
        # Preserve the existing full-TP head counts independently of projection placement.
        assert module.num_heads == 8 // get_parallel().tp_size
        names = ("qkv_proj", "o_proj")
    elif model == "mimo":
        from sglang.srt.models.mimo_v2 import MiMoV2Attention

        module = MiMoV2Attention(
            width,
            8,
            1 if variant else 4,
            head_dim=head_dim,
            v_head_dim=2 * head_dim if variant else head_dim,
            attention_bias=True,
            max_position_embeddings=16,
            quant_config=quant_config,
        )
        names = ("qkv_proj", "o_proj")
    elif model == "minimax":
        from sglang.srt.models.minimax_m3 import MiniMaxM3Attention

        module = MiniMaxM3Attention(
            config,
            quant_config=quant_config,
            is_sparse_attention_layer=variant != "dense",
            disable_index_value=variant == "index_no_value",
        )
        names = ("qkv_proj", "o_proj")
    elif model == "mtp":
        from sglang.srt.models.nemotron_h import (
            NemotronHAttentionDecoderLayer,
            NemotronHMoEDecoderLayer,
        )
        from sglang.srt.models.nemotron_h_mtp import (
            NemotronHMTPAttentionDecoderLayer,
            NemotronHMTPMoEDecoderLayer,
        )

        constructor, parent = (
            (NemotronHMTPAttentionDecoderLayer, NemotronHAttentionDecoderLayer)
            if variant == "attention"
            else (NemotronHMTPMoEDecoderLayer, NemotronHMoEDecoderLayer)
        )
        # Exercise the MTP constructor, leaving the unrelated parent mixer outside this fixture.
        with patch.object(
            parent, "__init__", lambda self, **kwargs: torch.nn.Module.__init__(self)
        ):
            module = constructor(
                config,
                0,
                quant_config=quant_config,
                has_start_projections=start,
                has_end_norm=True,
            )
        names = ("eh_proj",) if start else ()
    elif model == "sdar":
        from sglang.srt.models.sdar_moe import SDARMoeAttention

        module = SDARMoeAttention(
            config, 0, quant_config=quant_config, reduce_results=variant
        )
        names = ("o_proj",)
    elif model == "step3":
        from sglang.srt.models.step3_vl import Step3TextAttention

        module = Step3TextAttention(
            width,
            8,
            4,
            head_dim,
            width // 2 if variant else 0,
            rms_norm_eps=1e-6,
            max_position_embeddings=16,
            quant_config=quant_config,
        )
        names = ("qkv_proj", "wq", "o_proj")
    else:
        from sglang.srt.models.xllm import XllmGatedAttention, XllmMoVAAttention

        constructor = XllmGatedAttention if variant == "gated" else XllmMoVAAttention
        module = constructor(config, 0, quant_config=quant_config)
        names = ("q_proj", "k_proj", "gate_proj", "o_proj")
        if variant == "gated":
            names += ("v_proj",)
    return module, names


def values(rows, columns, layer, offset=0):
    return (
        (torch.arange(rows * columns, device=layer.weight.device) + offset) % 23 - 11
    ).reshape(rows, columns).to(layer.weight.dtype) / 128


def load_projection(layer):
    """Load at rank zero and return the checkpoint shard owned at construction."""
    rank, size = layer.tp_rank, layer.tp_size
    row = isinstance(layer, RowParallelLinear)
    bias_shard = None
    with get_parallel().override(tp_rank=0, attn_tp_rank=0, attn_dp_rank=0):
        if isinstance(layer, QKVParallelLinear):
            weights, biases = [], []
            for i, shard_id in enumerate(("q", "k", "v")):
                heads = (
                    layer.total_num_heads
                    if shard_id == "q"
                    else layer.total_num_kv_heads
                )
                dim = layer.v_head_size if shard_id == "v" else layer.head_size
                weight = values(heads * dim, layer.input_size, layer, i)
                partitions = size if shard_id == "q" else min(heads, size)
                index = rank if shard_id == "q" else rank // max(size // heads, 1)
                layer.weight.weight_loader(layer.weight, weight, shard_id)
                weights.append(weight.chunk(partitions)[index])
                if layer.bias is not None:
                    bias = values(heads * dim, 1, layer, i + 4).flatten()
                    layer.bias.weight_loader(layer.bias, bias, shard_id)
                    biases.append(bias.chunk(partitions)[index])
            shard = torch.cat(weights)
            if biases:
                bias_shard = torch.cat(biases)
        else:
            weight = values(layer.output_size, layer.input_size, layer)
            layer.weight.weight_loader(layer.weight, weight)
            pieces = weight.split(getattr(layer, "output_sizes", [layer.output_size]))
            shard = (
                weight.chunk(size, dim=1)[rank]
                if row
                else torch.cat([p.chunk(size)[rank] for p in pieces])
            )
            if layer.bias is not None:
                bias = values(layer.output_size, 1, layer, 5).flatten()
                layer.bias.weight_loader(layer.bias, bias)
                bias_shard = bias if row else bias.chunk(size)[rank]
    return shard, bias_shard


class TestAttentionProjectionParallelGroups(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def test_projection_reload_math_and_collective_policy(self):
        for dp_size in (1, 2):
            for rank in (0, 3):
                reset_context()
                server = ServerArgs(
                    model_path="dummy", device="cpu", tp_size=4, attn_dp_size=dp_size
                )
                publish(server, role="test", ranks=SpawnRanks(world_rank=rank))
                initialize_dp_attention_flags(server)
                tp, attn = Mock(), Mock()
                tp.all_reduce.side_effect = lambda x: x * 4
                attn.all_reduce.side_effect = lambda x: x * (4 // dp_size)
                tp.all_gather.side_effect = lambda x, dim: torch.cat([x] * 4, dim=dim)
                attn.all_gather.side_effect = lambda x, dim: torch.cat(
                    [x] * (4 // dp_size), dim=dim
                )
                for model, variant in ATTENTION_CASES:
                    with self.subTest(
                        dp=dp_size, rank=rank, model=model, variant=variant
                    ):
                        module, names = build_attention(model, variant)
                        for name in names:
                            layer = getattr(module, name)
                            shard, bias = load_projection(layer)
                            torch.testing.assert_close(layer.weight, shard)
                            if bias is not None:
                                torch.testing.assert_close(layer.bias, bias)
                            row = isinstance(layer, RowParallelLinear)
                            size = (
                                1
                                if model == "step3" and name == "qkv_proj"
                                else 4 // dp_size
                            )
                            if model == "mtp" and dp_size == 1:
                                size = 4
                            self.assertEqual(
                                (layer.tp_rank, layer.tp_size), (rank % size, size)
                            )
                            inputs = values(2, shard.shape[1], layer, 7)
                            used_bias = bias if not row or layer.tp_rank == 0 else None
                            expected = F.linear(inputs, shard, used_bias)
                            if row and layer.reduce_results:
                                expected *= 4
                            if not row and layer.gather_output:
                                expected = torch.cat([expected] * 4, dim=-1)
                            for group in (tp, attn):
                                group.all_reduce.reset_mock()
                                group.all_gather.reset_mock()
                            with (
                                get_parallel().override(
                                    tp_group=tp, attn_tp_group=attn
                                ),
                                patch(
                                    "sglang.srt.layers.linear.is_allocation_symmetric",
                                    return_value=True,
                                ),
                                patch(
                                    "sglang.srt.layers.linear.use_symmetric_memory",
                                    return_value=nullcontext(),
                                ) as allocator,
                            ):
                                actual = layer(inputs)[0]
                                torch.testing.assert_close(actual, expected)
                            if row:
                                allocator.assert_called_once_with(tp, disabled=False)
                                self.assertFalse(layer.use_dp_attention_reduce)
                                self.assertEqual(
                                    layer.reduce_results, model == "sdar" and variant
                                )
                            else:
                                allocator.assert_not_called()
                            self.assertEqual(
                                tp.all_reduce.call_count,
                                int(row and layer.reduce_results),
                            )
                            self.assertEqual(
                                tp.all_gather.call_count,
                                int(not row and layer.gather_output),
                            )
                            self.assertEqual(
                                attn.all_reduce.call_count + attn.all_gather.call_count,
                                0,
                            )

    def test_disabled_mtp_start_projections(self):
        publish(ServerArgs(model_path="dummy", device="cpu"), role="test")
        for variant in ("attention", "moe"):
            module, names = build_attention("mtp", variant, start=False)
            self.assertFalse(names)
            self.assertFalse(hasattr(module, "eh_proj"))
            self.assertTrue(hasattr(module, "final_layernorm"))

    def test_xllm_keeps_quantization_rejection(self):
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        publish(ServerArgs(model_path="dummy", device="cpu"), role="test")
        for variant in ("gated", "mova"):
            with self.assertRaisesRegex(ValueError, "unquantized bf16/fp16"):
                build_attention("xllm", variant, quant_config=Fp8Config())


if __name__ == "__main__":
    unittest.main()
