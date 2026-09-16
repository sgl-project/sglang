"""The residual dense branch must remain distinct from Grok's FFN entry."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.models import grok
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class DenseBranch(nn.Module):
    def __init__(self, events):
        super().__init__()
        self.down_proj = nn.Linear(2, 2, bias=False)
        self.down_proj.weight.data.copy_(torch.eye(2) * 2)
        self.events = events

    def forward(self, x):
        self.events.append("dense")
        return self.down_proj(x)


class MoeBranch(nn.Module):
    def __init__(self, events):
        super().__init__()
        self.gate = nn.Linear(2, 2, bias=False)
        self.gate.weight.data.copy_(torch.eye(2) * 3)
        self.events = events

    def forward(self, x):
        self.events.append("moe")
        # Model the in-place input update that requires dense execution first.
        return x.copy_(self.gate(x))


class TestGrokResidualFFN(unittest.TestCase):
    def make_layer(self, *, residual_moe, tp_size, events):
        config = SimpleNamespace(
            num_local_experts=2,
            hidden_size=2,
            residual_moe=residual_moe,
            rope_theta=10000,
            num_attention_heads=1,
            num_key_value_heads=1,
            max_position_embeddings=16,
            num_experts_per_tok=1,
            intermediate_size=2,
            rms_norm_eps=1e-6,
        )
        with (
            patch.object(grok, "Grok1Attention", return_value=nn.Identity()),
            patch.object(grok, "Grok1MLP", side_effect=lambda **_: DenseBranch(events)),
            patch.object(grok, "Grok1MoE", side_effect=lambda **_: MoeBranch(events)),
            patch.object(grok, "RMSNorm", side_effect=lambda *_, **__: nn.Identity()),
            patch.object(
                grok, "get_parallel", return_value=SimpleNamespace(tp_size=tp_size)
            ),
        ):
            layer = grok.Grok1DecoderLayer(config, alt_stream=object())
        return config, layer

    def test_residual_branches_are_distinct_and_execute_once(self):
        for tp_size in (1, 2):
            with self.subTest(tp_size=tp_size):
                events = []
                _, layer = self.make_layer(
                    residual_moe=True, tp_size=tp_size, events=events
                )
                self.assertEqual(
                    set(layer._modules),
                    {
                        "self_attn",
                        "mlp",
                        "block_sparse_moe",
                        "pre_attn_norm",
                        "post_attn_norm",
                        "pre_moe_norm",
                        "post_moe_norm",
                    },
                )
                x = torch.tensor([[1.0, 2.0]])
                with (
                    torch.no_grad(),
                    patch.object(grok, "get_is_capture_mode", return_value=False),
                    patch.object(
                        grok,
                        "tensor_model_parallel_all_reduce",
                        side_effect=lambda value: value * tp_size,
                    ) as reduce,
                ):
                    output = layer.ffn(x.clone())
                torch.testing.assert_close(output, x * (5 / math.sqrt(2)) * tp_size)
                self.assertEqual(events, ["dense", "moe"])
                self.assertEqual(reduce.call_count, int(tp_size > 1))

    def test_moe_only_entry(self):
        events = []
        _, layer = self.make_layer(residual_moe=False, tp_size=1, events=events)
        self.assertNotIn("mlp", layer._modules)
        self.assertIs(layer.ffn, layer.block_sparse_moe)
        with torch.no_grad():
            torch.testing.assert_close(
                layer.ffn(torch.ones(1, 2)), torch.full((1, 2), 3.0)
            )
        self.assertEqual(events, ["moe"])

    def test_original_dense_and_moe_checkpoint_paths_load_separately(self):
        config, layer = self.make_layer(residual_moe=True, tp_size=1, events=[])
        model = grok.Grok1ForCausalLM.__new__(grok.Grok1ForCausalLM)
        nn.Module.__init__(model)
        model.config = config
        model.loaded_param_names = set()
        model.model = nn.Module()
        model.model.layers = nn.ModuleList([layer])
        dense = torch.full((2, 2), 7.0)
        moe = torch.full((2, 2), 11.0)
        names = [
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.block_sparse_moe.gate.weight",
        ]
        loaded = model.load_weights(list(zip(names, (dense, moe))))
        self.assertEqual(loaded, set(names))
        self.assertEqual(model.loaded_param_names, set(names))
        torch.testing.assert_close(layer.mlp.down_proj.weight, dense)
        torch.testing.assert_close(layer.block_sparse_moe.gate.weight, moe)


if __name__ == "__main__":
    unittest.main()
