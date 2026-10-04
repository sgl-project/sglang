"""Kimi attention projections retain checkpoint layout and reduction policy."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.dp_attention import initialize_dp_attention_flags
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.models.kimi_k3 import KimiK3DeltaAttention, KimiK3MLAAttention
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

KIMI_CASES = tuple((kind, gate) for kind in ("delta", "mla") for gate in (False, True))


def build_kimi(
    kind,
    gate,
    fusion=False,
    fallback=False,
    width=32,
    head_dim=8,
    heads=8,
    quant_config=None,
):
    config = SimpleNamespace(
        hidden_size=width,
        num_attention_heads=heads,
        qk_nope_head_dim=head_dim - 4,
        qk_rope_head_dim=4,
        v_head_dim=head_dim,
        q_lora_rank=width,
        kv_lora_rank=width,
        rms_norm_eps=1e-6,
        max_position_embeddings=16,
        dtype=torch.get_default_dtype(),
        mla_use_output_gate=gate,
        linear_attn_config={
            "head_dim": head_dim,
            "num_heads": heads,
            "short_conv_kernel_size": 3,
            "use_full_rank_gate": gate,
        },
    )
    with (
        patch("sglang.srt.models.kimi_k3._o_proj_takes_output", return_value=False)
        if fallback
        else nullcontext()
    ):
        if kind == "mla":
            module = KimiK3MLAAttention(
                config, 0, quant_config=quant_config, all_reduce_fusion=fusion
            )
            names = ("g_proj",) if gate else ()
        else:
            module = KimiK3DeltaAttention(
                0, width, config, quant_config=quant_config, all_reduce_fusion=fusion
            )
            if gate:
                names = (
                    "fused_qkvg_proj",
                    "b_proj",
                    "f_b_proj",
                    "qkv_conv1d",
                    "o_proj",
                )
            elif module.do_fuse_qkvbfg:
                # The full-TP repeated/batched helpers keep their existing interface.
                names = ("qkv_conv1d", "o_proj")
            else:
                names = (
                    "qkv_proj",
                    "b_proj",
                    "f_b_proj",
                    "g_b_proj",
                    "qkv_conv1d",
                    "o_proj",
                )
    return module, names


def values(rows, columns, layer, offset=0):
    return (
        (torch.arange(rows * columns, device=layer.weight.device) + offset) % 23 - 11
    ).reshape(rows, columns).to(layer.weight.dtype) / 128


def load_projection(layer):
    rank, size = layer.tp_rank, layer.tp_size
    row = isinstance(layer, RowParallelLinear)
    with get_parallel().override(tp_rank=0, attn_tp_rank=0, attn_dp_rank=0):
        if isinstance(layer, QKVParallelLinear):
            shards = []
            for i, shard_id in enumerate(("q", "k", "v")):
                heads = (
                    layer.total_num_heads
                    if shard_id == "q"
                    else layer.total_num_kv_heads
                )
                dim = layer.v_head_size if shard_id == "v" else layer.head_size
                weight = values(heads * dim, layer.input_size, layer, i)
                layer.weight.weight_loader(layer.weight, weight, shard_id)
                shards.append(weight.chunk(size)[rank])
            shard = torch.cat(shards)
        else:
            weight = values(layer.output_size, layer.input_size, layer)
            if layer.weight.ndim == 3:
                weight = weight.unsqueeze(1)
            layer.weight.weight_loader(layer.weight, weight)
            pieces = weight.split(getattr(layer, "output_sizes", [layer.output_size]))
            shard = (
                weight.chunk(size, dim=1)[rank]
                if row
                else torch.cat([p.chunk(size)[rank] for p in pieces])
            )
    return shard


class TestKimiAttentionParallelGroups(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def test_projection_reload_math_and_policy(self):
        for dp in (1, 2):
            for rank in (0, 3):
                reset_context()
                server = ServerArgs(
                    model_path="dummy", device="cpu", tp_size=4, attn_dp_size=dp
                )
                publish(server, role="test", ranks=SpawnRanks(world_rank=rank))
                initialize_dp_attention_flags(server)
                tp, attn = Mock(), Mock()
                tp.all_reduce.side_effect = lambda x: x * 4
                attn.all_reduce.side_effect = lambda x: x * (4 // dp)
                for kind, gate in KIMI_CASES:
                    for fusion in (False, True):
                        for fallback in (False, True) if fusion else (False,):
                            with self.subTest(
                                dp=dp,
                                rank=rank,
                                kind=kind,
                                gate=gate,
                                fusion=fusion,
                                fallback=fallback,
                            ):
                                module, names = build_kimi(kind, gate, fusion, fallback)
                                active_fusion = fusion and not fallback
                                self.assertEqual(
                                    module.all_reduce_fusion, active_fusion
                                )
                                self.assertEqual(
                                    module.o_proj.reduce_results, not active_fusion
                                )
                                self.assertEqual(
                                    module.o_proj.use_dp_attention_reduce,
                                    not active_fusion,
                                )
                                for name in names:
                                    layer = getattr(module, name)
                                    shard = load_projection(layer)
                                    torch.testing.assert_close(layer.weight, shard)
                                    self.assertEqual(
                                        (layer.tp_rank, layer.tp_size),
                                        (rank % (4 // dp), 4 // dp),
                                    )
                                    if name == "qkv_conv1d":
                                        inputs = values(
                                            2 * shard.shape[0], 6, layer, 7
                                        ).reshape(2, shard.shape[0], 6)
                                        actual = F.conv1d(
                                            inputs, layer.weight, groups=shard.shape[0]
                                        )
                                        expected = (
                                            inputs.unfold(-1, 3, 1)
                                            * shard.squeeze(1)[None, :, None, :]
                                        ).sum(-1)
                                        torch.testing.assert_close(actual, expected)
                                        continue
                                    row = isinstance(layer, RowParallelLinear)
                                    inputs = values(2, shard.shape[1], layer, 7)
                                    expected = F.linear(inputs, shard)
                                    if row and layer.reduce_results:
                                        expected *= 4 // dp
                                    for group in (tp, attn):
                                        group.all_reduce.reset_mock()
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
                                        ) as allocation,
                                    ):
                                        actual = layer(inputs)[0]
                                        torch.testing.assert_close(actual, expected)
                                    if row:
                                        if layer.use_dp_attention_reduce:
                                            allocation.assert_called_once_with(attn)
                                        else:
                                            allocation.assert_called_once_with(
                                                tp, disabled=False
                                            )
                                    else:
                                        allocation.assert_not_called()
                                    self.assertEqual(tp.all_reduce.call_count, 0)
                                    self.assertEqual(
                                        attn.all_reduce.call_count,
                                        int(row and layer.reduce_results),
                                    )


if __name__ == "__main__":
    unittest.main()
