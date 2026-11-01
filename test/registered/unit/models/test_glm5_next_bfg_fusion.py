import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.kernels.ops.gemm import _prefer_torch_rowwise_fp8
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models import glm5_next
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PREFIX = "model.layers.0.self_attn"
QKV = ("q_proj", "k_proj", "v_proj")
BFG = ("b_proj", "f_a_proj", "g_a_proj", "f_b_proj", "g_b_proj")
KDA_FP8_INPUT_MODULES = ("qkv_proj", "f_a_proj", "g_a_proj")
KDA_FP8_ALL_MODULES = ",".join(KDA_FP8_INPUT_MODULES + ("o_proj",))


class MockQuantizedLinearMethod:
    """Keep dense storage so the test isolates routing and checkpoint loading."""

    create_weights = UnquantizedLinearMethod.create_weights

    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight, bias)


class MockFp8Config:
    def __init__(self, ignored):
        self.ignored_layers = {f"{PREFIX}.{name}" for name in ignored}

    is_linear_unquantized = QuantizationConfig.is_linear_unquantized

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


def quark_mxfp4_config(excluded):
    """A Quark MXFP4 export that keeps `excluded` in BF16, named as the HF
    checkpoint names them and mapped the way the loader maps them."""
    from sglang.srt.layers.quantization.quark.quark import QuarkConfig

    mxfp4 = {"dtype": "fp4", "qscheme": "per_group", "group_size": 32}
    model = glm5_next.Glm5NextForConditionalGeneration
    quant = QuarkConfig.from_config(
        {
            "quant_method": "quark",
            "export": {"kv_cache_group": [], "pack_method": "reorder"},
            "global_quant_config": {
                "weight": {**mxfp4, "is_dynamic": False, "scale_format": "e8m0"},
                "input_tensors": {**mxfp4, "is_dynamic": True, "scale_format": "e8m0"},
                "output_tensors": None,
                "bias": None,
            },
            "layer_quant_config": {},
            "layer_type_quant_config": {},
            "exclude": [
                f"model.language_model.layers.0.self_attn.{name}" for name in excluded
            ],
            "packed_modules_mapping": dict(model.packed_modules_mapping),
        }
    )
    quant.apply_weight_name_mapper(model.hf_to_sglang_mapper)
    return quant


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
                        tp_size=4,
                        tp_rank=rank,
                        attn_tp_size=attn_tp,
                        attn_tp_rank=rank,
                        attn_dp_size=4 // attn_tp,
                        attn_dp_rank=0,
                        attn_cp_size=1,
                        attn_cp_rank=0,
                        moe_tp_size=4,
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

    def test_quark_fuses_only_projections_the_checkpoint_keeps_in_bf16(self):
        for packed in ("fused_qkvbfg_a_proj", "fused_bfg_a_proj"):
            with self.subTest(packed=packed):
                self.assertTrue(
                    glm5_next.Glm5NextLinearAttention._can_fuse_proj(
                        quark_mxfp4_config(QKV + BFG),
                        PREFIX,
                        packed,
                        "fused_fg_b_proj",
                    )
                )
        for quantized in QKV + BFG:
            quant = quark_mxfp4_config(
                [name for name in QKV + BFG if name != quantized]
            )
            with self.subTest(quantized=quantized):
                self.assertFalse(
                    glm5_next.Glm5NextLinearAttention._can_fuse_proj(
                        quant, PREFIX, "fused_qkvbfg_a_proj", "fused_fg_b_proj"
                    )
                )
                # The probe must not register a quantized layer the model never builds.
                self.assertEqual(quant.quantized_layers, ([], 0))

    def test_quark_answers_from_its_exclude_list(self):
        quant = quark_mxfp4_config(["b_proj"])
        self.assertTrue(quant.is_linear_unquantized(f"{PREFIX}.b_proj"))
        self.assertFalse(quant.is_linear_unquantized(f"{PREFIX}.q_proj"))
        self.assertEqual(quant.quantized_layers, ([], 0))
        # Excluded layers that a mixed-precision export keeps in FP8 are quantized.
        with patch.object(quant, "excluded_fp8_config", SimpleNamespace()):
            self.assertFalse(quant.is_linear_unquantized(f"{PREFIX}.b_proj"))

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


def kda_fp8_modules(value, fp8_gemm=True):
    with (
        envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override(value),
        patch.object(glm5_next, "cutlass_fp8_supported", return_value=fp8_gemm),
    ):
        return glm5_next._kda_fp8_modules()


class TestGlm5NextKdaFp8Switch(CustomTestCase):
    """SGLANG_OPT_GLM53_KDA_PTPC_MODULES parsing and wiring; the FP8 numerics run on
    GPU in test/registered/kernels/ops/gemm/test_glm53_kda_fp8_projection.py."""

    def setUp(self):
        # EnvField.override skips its restore when the body raises (the rejection cases);
        # this outer override exits normally and restores the variable.
        switch = envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override("")
        switch.__enter__()
        self.addCleanup(switch.__exit__, None, None, None)
        override = get_context().override_server_args(
            device="cpu", enable_lora=False, lora_paths=None
        )
        override.install()
        self.addCleanup(override.restore)
        parallel = get_parallel().override(
            tp_size=1,
            tp_rank=0,
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_dp_size=1,
            attn_dp_rank=0,
            attn_cp_size=1,
            attn_cp_rank=0,
            moe_tp_size=1,
        )
        parallel.__enter__()
        self.addCleanup(parallel.__exit__, None, None, None)

    def build(self, modules, quant_config=None):
        with (
            envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override(modules),
            patch.object(glm5_next, "cutlass_fp8_supported", return_value=True),
        ):
            return glm5_next.Glm5NextLinearAttention(
                layer_idx=0,
                hidden_size=16,
                config=SimpleNamespace(
                    linear_attn_config={
                        "head_dim": 8,
                        "num_heads": 4,
                        "short_conv_kernel_size": 4,
                    }
                ),
                quant_config=quant_config,
                prefix=PREFIX,
            )

    def test_input_projections_switch_together(self):
        # The fused input projection is one GEMM, so its parts cannot differ in dtype.
        self.assertEqual(
            kda_fp8_modules(KDA_FP8_ALL_MODULES),
            set(KDA_FP8_INPUT_MODULES) | {"o_proj"},
        )
        self.assertEqual(kda_fp8_modules("o_proj"), {"o_proj"})
        for partial in ("qkv_proj", "qkv_proj,f_a_proj", "g_a_proj,o_proj"):
            with (
                self.subTest(partial=partial),
                self.assertRaisesRegex(ValueError, "together"),
            ):
                kda_fp8_modules(partial)

    def test_rejects_unknown_module_and_missing_fp8_gemm(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            kda_fp8_modules("in_proj")
        with self.assertRaisesRegex(ValueError, "SM89"):
            kda_fp8_modules("o_proj", fp8_gemm=False)
        with (
            envs.SGLANG_FORCE_FP8_MARLIN.override(True),
            self.assertRaisesRegex(ValueError, "SGLANG_FORCE_FP8_MARLIN"),
        ):
            kda_fp8_modules("o_proj")
        self.assertEqual(kda_fp8_modules("", fp8_gemm=False), frozenset())

    def test_switch_selects_the_fp8_method_per_projection(self):
        for modules, in_fp8, o_fp8 in (
            ("", False, False),
            ("o_proj", False, True),
            (",".join(KDA_FP8_INPUT_MODULES), True, False),
            (KDA_FP8_ALL_MODULES, True, True),
        ):
            with self.subTest(modules=modules):
                attention = self.build(modules)
                for layer, fp8 in (
                    (attention.fused_qkvbfg_a_proj, in_fp8),
                    (attention.o_proj, o_fp8),
                ):
                    expected = (
                        glm5_next._KdaFp8LinearMethod
                        if fp8
                        else UnquantizedLinearMethod
                    )
                    self.assertIs(type(layer.quant_method), expected)

    def test_fp8_checkpoint_keeps_the_bf16_projections_for_the_switch(self):
        # GLM-5.3-Flash's block-FP8 checkpoint lists the KDA projections as unquantized.
        attention = self.build(
            KDA_FP8_ALL_MODULES, MockFp8Config(QKV + BFG + ("o_proj",))
        )
        for layer in (attention.fused_qkvbfg_a_proj, attention.o_proj):
            self.assertIs(type(layer.quant_method), glm5_next._KdaFp8LinearMethod)
        with self.assertRaisesRegex(ValueError, "needs a BF16 o_proj"):
            self.build("o_proj", MockFp8Config(QKV + BFG))

    def test_unsupported_loads_and_runtime_modes_are_rejected(self):
        # These load formats copy processed tensors into parameters built from the
        # checkpoint; a transfer-engine seed serves its FP8 tensors to such peers.
        cases = [
            ({"load_format": fmt}, fmt)
            for fmt in ("remote_instance", "sharded_state", "presharded", "remote")
        ]
        cases.append(
            (
                {"remote_instance_weight_loader_start_seed_via_transfer_engine": True},
                "transfer-engine",
            )
        )
        # Elastic EP reloads lost experts from disk and ignores the update's refusal.
        cases.append(({"elastic_ep_backend": "mooncake"}, "elastic-ep-backend"))
        # The FP8 path picks its kernel by token count.
        cases.append(
            ({"enable_deterministic_inference": True}, "deterministic-inference")
        )
        for args, error in cases:
            with self.subTest(**args):
                override = get_context().override_server_args(
                    device="cpu", enable_lora=False, lora_paths=None, **args
                )
                override.install()
                self.addCleanup(override.restore)
                with self.assertRaisesRegex(ValueError, error):
                    kda_fp8_modules("o_proj")
                self.assertEqual(kda_fp8_modules(""), frozenset())

    def test_unfused_input_projection_is_rejected(self):
        # LoRA turns the projection fusion off.
        override = get_context().override_server_args(enable_lora=True)
        override.install()
        self.addCleanup(override.restore)
        self.assertIsInstance(self.build("o_proj").o_proj.quant_method, Fp8LinearMethod)
        with self.assertRaisesRegex(ValueError, "fused"):
            self.build(KDA_FP8_ALL_MODULES)

    def test_fp8_scaled_mm_keeps_the_aot_kernel_below_8192_rows(self):
        # _apply_dynamic_per_token calls the AOT kernel directly below 8192 rows, where
        # fp8_scaled_mm never prefers torch._scaled_mm.
        cuda = torch.device("cuda", 0)

        def tensor(shape, stride, dtype):
            # The predicate reads only metadata, so CUDA tensors are described, not allocated.
            return SimpleNamespace(
                shape=shape,
                ndim=len(shape),
                stride=lambda dim: stride[dim],
                dtype=dtype,
                device=cuda,
                is_contiguous=lambda: stride[-1] == 1,
                numel=lambda: shape[0] * shape[1],
            )

        def prefers_torch(m, n=24896, k=4096):
            return _prefer_torch_rowwise_fp8(
                tensor((m, k), (k, 1), torch.float8_e4m3fn),
                tensor((k, n), (1, k), torch.float8_e4m3fn),
                tensor((m, 1), (1, 1), torch.float32),
                tensor((n, 1), (1, 1), torch.float32),
                torch.bfloat16,
                None,
            )

        self.assertTrue(prefers_torch(8192))
        for n, k in ((24896, 4096), (6416, 4096), (3344, 4096), (4096, 8192)):
            with self.subTest(n=n, k=k):
                self.assertFalse(prefers_torch(8191, n, k))


if __name__ == "__main__":
    unittest.main()
