import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
    _model_load_weights_direct,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.models import glm5_next
from sglang.srt.models.glm5_next_nextn import Glm5NextForConditionalGenerationNextN
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

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


class TestGlm5NextLinearAttentionOnormHandoff(CustomTestCase):
    def setUp(self):
        self.addCleanup(torch.set_default_dtype, torch.get_default_dtype())
        override = get_context().override_server_args(
            device="cpu", enable_lora=False, lora_paths=None
        )
        override.install()
        self.addCleanup(override.restore)

    def make_attention(self, raw, gate):
        _, tokens, heads, head_dim = raw.shape
        attention = glm5_next.Glm5NextLinearAttention(
            layer_idx=0,
            hidden_size=heads * head_dim,
            config=SimpleNamespace(
                linear_attn_config={
                    "head_dim": head_dim,
                    "num_heads": heads,
                    "short_conv_kernel_size": 4,
                }
            ),
            prefix=PREFIX,
        )
        attention.do_fuse_qkvbfg = False
        attention.forward_qkvbfg = lambda *_: (
            torch.zeros(tokens, 3 * heads * head_dim),
            torch.zeros(tokens, heads),
            torch.zeros(tokens, heads * head_dim),
            gate,
        )
        attention.o_proj.forward = lambda value: (value, None)
        backend = Mock(return_value=raw)
        attention.attn.forward = backend
        attention.o_norm.forward = lambda value, gate: (
            F.rms_norm(
                value, (head_dim,), attention.o_norm.weight, attention.o_norm.eps
            )
            * torch.sigmoid(gate)
        )
        return attention, backend

    @torch.no_grad()
    def test_weight_reload_updates_fused_norm_without_replacing_storage(self):
        """Verify and fallback must use the same norm weights after a reload."""
        tokens, heads, head_dim = 3, 2, 4
        raw = (
            torch.arange(1, tokens * heads * head_dim + 1)
            .float()
            .reshape(1, tokens, heads, head_dim)
        )
        gate = torch.linspace(-1, 1, raw.numel()).reshape(tokens, heads * head_dim)
        hidden = torch.zeros(tokens, heads * head_dim)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            torch.set_default_dtype(dtype)
            for route in (
                "checkpoint",
                "nextn",
                "direct",
                "custom",
                "post_load",
                "quantized_rl",
            ):
                with self.subTest(dtype=dtype, route=route):
                    attention, backend_mock = self.make_attention(raw, gate)
                    self.assertEqual(attention.o_norm.weight.dtype, torch.float32)
                    attention.o_norm.weight.fill_(1)

                    model_class = (
                        Glm5NextForConditionalGenerationNextN
                        if route == "nextn"
                        else glm5_next.Glm5NextForConditionalGeneration
                    )
                    model = model_class.__new__(model_class)
                    torch.nn.Module.__init__(model)
                    model.config = SimpleNamespace(
                        n_routed_experts=0,
                        num_hidden_layers=2,
                        num_nextn_predict_layers=1,
                    )
                    model.num_fused_shared_experts = 0
                    model.quant_config = None
                    model.encoder_only = False
                    model.model = torch.nn.Module()
                    layer = torch.nn.Module()
                    layer.self_attn = attention
                    if route == "nextn":
                        model.model.decoder = layer
                    else:
                        model.model.layers = torch.nn.ModuleList([layer])
                        model.model.start_layer = 0
                        model.model.end_layer = 1

                    def run_backend(batch, **kwargs):
                        if batch.forward_mode.is_target_verify():
                            attn = attention.attn
                            attn._k3_onorm_consumed = True
                            weight, eps = attn._k3_onorm_args
                            return F.rms_norm(
                                raw, (head_dim,), weight, eps
                            ) * torch.sigmoid(
                                attn._k3_onorm_gate.unflatten(-1, (-1, head_dim))
                            )
                        return raw

                    backend_mock.side_effect = run_backend

                    def forward(mode):
                        return attention(
                            hidden,
                            SimpleNamespace(forward_mode=mode, attn_cp_metadata=None),
                        )

                    original = forward(ForwardMode.TARGET_VERIFY)
                    source_storage = attention.o_norm.weight.data_ptr()
                    prefix = "model.layers.2.self_attn" if route == "nextn" else PREFIX
                    weights = [
                        (f"{prefix}.o_norm.weight", torch.full((head_dim,), 2.0))
                    ]
                    with torch.enable_grad():
                        if route in ("checkpoint", "nextn"):
                            model.load_weights(weights)
                        elif route == "custom":
                            updater = WeightUpdater(
                                tp_rank=0,
                                device="cpu",
                                gpu_id=0,
                                model_config=None,
                                custom_weight_loaders={"test_loader": None},
                                get_model=lambda: model,
                                update_model_fields=None,
                                recapture_cuda_graph=None,
                                get_model_runner=None,
                            )
                            with (
                                get_parallel().override(tp_rank=0),
                                patch(
                                    "sglang.srt.model_executor.model_runner_components.weight_updater.dynamic_import",
                                    return_value=_model_load_weights_direct,
                                ),
                            ):
                                success, _ = updater.update_weights_from_tensor(
                                    weights, load_format="test_loader"
                                )
                            self.assertTrue(success)
                        else:
                            _model_load_weights_direct(model, weights)
                            if route == "post_load":
                                model.post_load_weights()
                            else:
                                if route == "quantized_rl":
                                    model.original_weights_rebuild_keys = {}
                                    model.recorded_loader = {}
                                    model.flash_rl_initial_load_complete = True
                                DefaultModelLoader.postprocess_weights(
                                    model, torch.device("cpu")
                                )
                    self.assertEqual(attention.o_norm.weight.data_ptr(), source_storage)
                    fallback = forward(ForwardMode.EXTEND)
                    torch.testing.assert_close(fallback, original * 2)
                    torch.testing.assert_close(
                        forward(ForwardMode.TARGET_VERIFY), fallback
                    )

    def test_output_norm_handoff_and_fallback(self):
        torch.set_default_dtype(torch.float32)
        tokens, heads, head_dim = 3, 2, 4
        raw = torch.arange(tokens * heads * head_dim, dtype=torch.float32).reshape(
            1, tokens, heads, head_dim
        )
        gate = torch.linspace(-1, 1, raw.numel()).reshape(tokens, heads * head_dim)
        attention, backend = self.make_attention(raw, gate)
        attention.o_norm.forward = Mock(side_effect=lambda value, gate: value + 3)
        hidden = torch.zeros(tokens, heads * head_dim)
        for mode, consume in (
            (ForwardMode.TARGET_VERIFY, True),
            (ForwardMode.TARGET_VERIFY, False),
            (ForwardMode.DECODE, False),
            (ForwardMode.EXTEND, False),
        ):
            with self.subTest(mode=mode, consume=consume):
                norm = attention.o_norm.forward
                norm.reset_mock()
                verify = mode == ForwardMode.TARGET_VERIFY

                def run_backend(batch, **kwargs):
                    attn = attention.attn
                    if verify:
                        self.assertIs(attn._k3_onorm_gate, gate)
                        weight, eps = attn._k3_onorm_args
                        self.assertIs(weight, attention.o_norm.weight)
                        self.assertEqual(eps, attention.o_norm.eps)
                        self.assertFalse(attn._k3_onorm_consumed)
                        attn._k3_onorm_consumed = consume
                    else:
                        self.assertIsNone(attn._k3_onorm_gate)
                        self.assertIsNone(attn._k3_onorm_args)
                    return raw + 11 if consume else raw

                backend.side_effect = run_backend
                batch = SimpleNamespace(forward_mode=mode, attn_cp_metadata=None)
                actual = attention(hidden, batch)
                expected = (raw + (11 if consume else 3)).squeeze(0).flatten(-2)
                torch.testing.assert_close(actual, expected)
                if not consume:
                    torch.testing.assert_close(
                        norm.call_args.args[1], gate.unflatten(-1, (-1, head_dim))
                    )
                self.assertIsNone(attention.attn._k3_onorm_gate)
                self.assertIsNone(attention.attn._k3_onorm_args)

        backend.side_effect = RuntimeError("verify failed")
        with self.assertRaisesRegex(RuntimeError, "verify failed"):
            attention(
                hidden,
                SimpleNamespace(
                    forward_mode=ForwardMode.TARGET_VERIFY, attn_cp_metadata=None
                ),
            )
        self.assertIsNone(attention.attn._k3_onorm_gate)
        self.assertIsNone(attention.attn._k3_onorm_args)

    def test_verify_adapter_accepts_compact_and_k3_norm_arguments(self):
        from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend

        tokens, heads, head_dim = 3, 2, 4
        raw = (
            torch.arange(1, tokens * heads * head_dim + 1)
            .float()
            .reshape(1, tokens, heads, head_dim)
        )
        gate = torch.linspace(-1, 1, raw.numel()).reshape(tokens, heads * head_dim)
        attention, _ = self.make_attention(raw, gate)
        layer = attention.attn
        layer.lower_bound = -5.0
        norm_weight = torch.full((head_dim,), 1.25, dtype=torch.float32)
        args = (norm_weight, attention.o_norm.eps)

        def kernel(*, onorm_gate, onorm_weight, onorm_eps, **kwargs):
            if onorm_weight is None:
                return raw
            return F.rms_norm(
                raw, (head_dim,), onorm_weight, onorm_eps
            ) * torch.sigmoid(onorm_gate)

        with patch.dict(
            "sys.modules",
            {
                "sglang.kernels.ops.attention.kda_decode_mtp": SimpleNamespace(
                    fused_kda_decode_mtp_dspark=kernel
                )
            },
        ):
            for handoff in ("compact", "k3", "none"):
                with self.subTest(handoff=handoff):
                    layer._k3_onorm_gate = gate
                    layer._k3_onorm_args = args if handoff == "compact" else None
                    layer._k3_fused_decode_args = (
                        (torch.empty(0),) * 5 + args if handoff == "k3" else None
                    )
                    layer._k3_onorm_consumed = False
                    output = KDAAttnBackend._run_dspark_cutedsl_mtp(
                        layer=layer,
                        mixed_qkv=torch.zeros(tokens, 3 * heads * head_dim),
                        a=torch.zeros(1, tokens, heads * head_dim),
                        b=torch.zeros(1, tokens, heads),
                        conv_states=torch.zeros(1, 3, 3 * heads * head_dim),
                        ssm_states=torch.zeros(1, heads, head_dim, head_dim),
                        intermediate_state_cache=torch.zeros(
                            1, tokens, heads, head_dim, head_dim
                        ),
                        intermediate_conv_window_cache=torch.zeros(
                            1, tokens, 3, 3 * heads * head_dim
                        ),
                        intermediate_state_indices=torch.zeros(1, dtype=torch.int32),
                        cache_indices=torch.zeros(1, dtype=torch.int32),
                        query_start_loc=torch.tensor([0, tokens], dtype=torch.int32),
                    )
                    expected = raw
                    if handoff != "none":
                        expected = kernel(
                            onorm_gate=gate.unflatten(-1, (-1, head_dim)).unsqueeze(0),
                            onorm_weight=norm_weight,
                            onorm_eps=attention.o_norm.eps,
                        )
                    torch.testing.assert_close(output, expected)
                    self.assertEqual(layer._k3_onorm_consumed, handoff != "none")


if __name__ == "__main__":
    unittest.main()
