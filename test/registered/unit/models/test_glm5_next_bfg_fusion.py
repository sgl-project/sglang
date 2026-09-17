import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models import glm5_next
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PREFIX = "model.layers.0.self_attn"
QKV = ("q_proj", "k_proj", "v_proj")
BFG = ("b_proj", "f_a_proj", "g_a_proj", "f_b_proj", "g_b_proj")


class MockQuantizedLinearMethod:
    """Keep dense storage so the test isolates routing and checkpoint loading."""

    create_weights = UnquantizedLinearMethod.create_weights

    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight, bias)


class MockFp8Config:
    def __init__(self, ignored):
        self.ignored_layers = {f"{PREFIX}.{name}" for name in ignored}

    def get_name(self):
        return "fp8"

    def get_quant_method(self, layer, prefix):
        names = (
            [prefix.replace("qkv_proj", name) for name in QKV]
            if prefix.endswith(".qkv_proj")
            else [prefix]
        )
        if all(name in self.ignored_layers for name in names):
            return UnquantizedLinearMethod()
        return MockQuantizedLinearMethod()


class TestGlm5NextBfgFusion(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_default_dtype, torch.get_default_dtype())
        torch.set_default_dtype(torch.float32)
        override = get_context().override_server_args(
            device="cpu", enable_lora=False, lora_paths=None
        )
        override.install()
        self.addCleanup(override.restore)
        patcher = patch.object(
            UnquantizedLinearMethod,
            "apply",
            MockQuantizedLinearMethod.apply,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @torch.no_grad()
    def test_projection_loading_matches_unfused_reference(self):
        torch.manual_seed(42)
        hidden, heads, dim = 16, 4, 8
        shapes = {name: (heads * dim, hidden) for name in QKV}
        shapes.update(
            b_proj=(heads, hidden),
            f_a_proj=(dim, hidden),
            g_a_proj=(dim, hidden),
            f_b_proj=(heads * dim, dim),
            g_b_proj=(heads * dim, dim),
        )
        weights = {name: torch.randn(shape) for name, shape in shapes.items()}
        x = torch.randn(7, hidden)
        for ignored, expected_route in (
            (QKV + BFG, (True, False)),
            (BFG, (False, True)),
            ((), (False, False)),
        ):
            for attn_tp, rank in ((1, 0), (2, 0), (2, 1)):
                with (
                    self.subTest(route=expected_route, attn_tp=attn_tp, rank=rank),
                    get_parallel().override(
                        tp_size=4, tp_rank=3, attn_tp_size=attn_tp, attn_tp_rank=rank
                    ),
                ):
                    quant = MockFp8Config(ignored)
                    attention = glm5_next.Glm5NextLinearAttention(
                        layer_idx=0,
                        hidden_size=hidden,
                        config=SimpleNamespace(
                            linear_attn_config={
                                "head_dim": dim,
                                "num_heads": heads,
                                "short_conv_kernel_size": 4,
                            }
                        ),
                        quant_config=quant,
                        prefix=PREFIX,
                    )
                    self.assertEqual(
                        (attention.do_fuse_qkvbfg, attention.fuse_bfg), expected_route
                    )
                    for parameter in attention.parameters():
                        parameter.fill_(torch.nan)
                    model = SimpleNamespace(
                        config=SimpleNamespace(n_routed_experts=0),
                        num_fused_shared_experts=0,
                        quant_config=quant,
                        named_parameters=lambda: (
                            (f"{PREFIX}.{name}", param)
                            for name, param in attention.named_parameters()
                        ),
                    )
                    with patch.object(
                        glm5_next.DeepseekV2WeightLoaderMixin, "post_load_weights"
                    ):
                        glm5_next.Glm5NextForConditionalGeneration.load_weights(
                            model,
                            [
                                (f"{PREFIX}.{name}.weight", w)
                                for name, w in weights.items()
                            ],
                        )

                    def linear(value, name):
                        weight = weights[name]
                        if name not in ("f_a_proj", "g_a_proj"):
                            weight = weight.chunk(attn_tp, dim=0)[rank]
                        return F.linear(value, weight)

                    expected = (
                        torch.cat([linear(x, name) for name in QKV], dim=-1),
                        linear(x, "b_proj"),
                        linear(linear(x, "f_a_proj"), "f_b_proj"),
                        linear(linear(x, "g_a_proj"), "g_b_proj"),
                    )
                    forward = (
                        attention.forward_qkvbfg_fused
                        if attention.do_fuse_qkvbfg
                        else attention.forward_qkvbfg
                    )
                    for actual, reference in zip(forward(x, None), expected):
                        torch.testing.assert_close(
                            actual, reference, atol=1e-5, rtol=1e-5
                        )

    def test_each_quantized_gate_projection_disables_fusion(self):
        for quantized in BFG:
            quant = MockFp8Config(name for name in QKV + BFG if name != quantized)
            for packed in ("fused_qkvbfg_a_proj", "fused_bfg_a_proj"):
                with self.subTest(quantized=quantized, packed=packed):
                    self.assertFalse(
                        glm5_next.Glm5NextLinearAttention._can_fuse_proj(
                            quant, PREFIX, packed, "fused_fg_b_proj"
                        )
                    )

    def test_lora_disables_full_and_bfg_fusion(self):
        for enable_lora, paths in ((True, None), (False, ["adapter"])):
            with patch.object(
                glm5_next,
                "get_lora",
                return_value=SimpleNamespace(enable_lora=enable_lora, lora_paths=paths),
            ):
                for quant in (None, MockFp8Config(QKV + BFG)):
                    for packed in ("fused_qkvbfg_a_proj", "fused_bfg_a_proj"):
                        with self.subTest(
                            enabled=enable_lora, paths=paths, packed=packed
                        ):
                            self.assertFalse(
                                glm5_next.Glm5NextLinearAttention._can_fuse_proj(
                                    quant, PREFIX, packed, "fused_fg_b_proj"
                                )
                            )


if __name__ == "__main__":
    unittest.main()
