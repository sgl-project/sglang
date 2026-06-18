import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ModuleNotFoundError:
    register_cpu_ci = None

if register_cpu_ci is not None:
    register_cpu_ci(est_time=2, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]
MODELSLIM_PATH = (
    REPO_ROOT / "python/sglang/srt/layers/quantization/modelslim/modelslim.py"
)


class _FakeLinearBase:
    pass


class _FakeFusedMoE:
    pass


class _FakeScheme:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _FakeMethodBase:
    pass


class _FakeQuantizationConfig:
    pass


class _FakeUnquantizedLinearMethod:
    pass


class _FakeParameter:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _stub_modules():
    package_names = [
        "sglang",
        "sglang.srt",
        "sglang.srt.hardware_backend",
        "sglang.srt.hardware_backend.npu",
        "sglang.srt.hardware_backend.npu.quantization",
        "sglang.srt.layers",
        "sglang.srt.layers.moe",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.modelslim",
    ]
    modules = {}
    for name in package_names:
        package = _module(name)
        package.__path__ = []
        modules[name] = package

    torch_nn = _module("torch.nn", Module=object, Parameter=_FakeParameter)
    modules["torch"] = _module(
        "torch",
        int8=object(),
        float16=object(),
        bfloat16=object(),
        nn=torch_nn,
    )
    modules["torch.nn"] = torch_nn
    modules.update(
        {
            "sglang.srt.hardware_backend.npu.quantization.linear_method_npu": _module(
                "sglang.srt.hardware_backend.npu.quantization.linear_method_npu",
                _NPULinearMethodBase=_FakeMethodBase,
            ),
            "sglang.srt.layers.linear": _module(
                "sglang.srt.layers.linear",
                LinearBase=_FakeLinearBase,
            ),
            "sglang.srt.layers.moe.fused_moe_triton": _module(
                "sglang.srt.layers.moe.fused_moe_triton",
                FusedMoE=_FakeFusedMoE,
            ),
            "sglang.srt.layers.quantization.base_config": _module(
                "sglang.srt.layers.quantization.base_config",
                FusedMoEMethodBase=_FakeMethodBase,
                QuantizationConfig=_FakeQuantizationConfig,
            ),
            "sglang.srt.layers.quantization.modelslim.schemes": _module(
                "sglang.srt.layers.quantization.modelslim.schemes",
                ModelSlimMXFP8Scheme=_FakeScheme,
                ModelSlimW4A4Int4=_FakeScheme,
                ModelSlimW4A4Int4MoE=_FakeScheme,
                ModelSlimW4A8Int8MoE=_FakeScheme,
                ModelSlimW8A8Int8=_FakeScheme,
                ModelSlimW8A8Int8MoE=_FakeScheme,
            ),
            "sglang.srt.layers.quantization.unquant": _module(
                "sglang.srt.layers.quantization.unquant",
                UnquantizedLinearMethod=_FakeUnquantizedLinearMethod,
            ),
            "sglang.srt.utils": _module(
                "sglang.srt.utils",
                apply_module_patch=lambda *args, **kwargs: None,
            ),
        }
    )
    return modules


class _ModelSlimLoader:
    def __enter__(self):
        self.patcher = patch.dict(sys.modules, _stub_modules())
        self.patcher.start()
        spec = importlib.util.spec_from_file_location(
            "modelslim_under_test", MODELSLIM_PATH
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        module.logger.info_once = lambda *args, **kwargs: None
        return module

    def __exit__(self, exc_type, exc_value, traceback):
        self.patcher.stop()


def _load_modelslim_module():
    return _ModelSlimLoader()


class TestModelSlimMiniMaxM3Mapping(unittest.TestCase):
    def test_minimax_m3_fused_linear_prefixes_are_mapped_to_quant_keys(self):
        with _load_modelslim_module() as modelslim:
            quant_config = modelslim.ModelSlimConfig(
                {
                    "language_model.model.layers.0.self_attn.q_proj.weight": "W8A8_DYNAMIC",
                    "language_model.model.layers.0.self_attn.index_q_proj.weight": "FLOAT",
                    "language_model.model.layers.0.self_attn.index_k_proj.weight": "FLOAT",
                    "language_model.model.layers.1.mlp.gate_proj.weight": "W8A8_DYNAMIC",
                    "language_model.model.layers.2.block_sparse_moe.shared_experts.gate_proj.weight": "W8A8_DYNAMIC",
                    "language_model.model.layers.2.block_sparse_moe.shared_experts.down_proj.weight": "FLOAT",
                    "model.layers.0.self_attn.q_proj.weight": "W8A8",
                    "model.layers.0.self_attn.index_q_proj.weight": "W8A8",
                    "model.layers.1.mlp.gate_proj.weight": "W8A8",
                    "model.layers.2.block_sparse_moe.shared_experts.gate_proj.weight": "W8A8",
                }
            )

            cases = [
                (
                    "model.layers.0.self_attn.qkv_proj",
                    "model.layers.0.self_attn.q_proj",
                ),
                (
                    "model.layers.0.self_attn.index_qkv_proj",
                    "model.layers.0.self_attn.index_q_proj",
                ),
                ("model.layers.1.mlp.gate_up_proj", "model.layers.1.mlp.gate_proj"),
                (
                    "model.layers.2.mlp.shared_experts.gate_up_proj",
                    "model.layers.2.block_sparse_moe.shared_experts.gate_proj",
                ),
                (
                    "language_model.model.layers.0.self_attn.qkv_proj",
                    "language_model.model.layers.0.self_attn.q_proj",
                ),
                (
                    "language_model.model.layers.1.mlp.gate_up_proj",
                    "language_model.model.layers.1.mlp.gate_proj",
                ),
                (
                    "language_model.model.layers.2.mlp.shared_experts.gate_up_proj",
                    "language_model.model.layers.2.block_sparse_moe.shared_experts.gate_proj",
                ),
            ]
            for layer_prefix, expected_quant_prefix in cases:
                with self.subTest(layer_prefix=layer_prefix):
                    layer = _FakeLinearBase()

                    quant_config.get_quant_method(layer, layer_prefix)

                    self.assertIsNotNone(layer.scheme)
                    self.assertEqual(
                        layer.scheme.kwargs["prefix"], expected_quant_prefix
                    )

            method = quant_config.get_quant_method(
                _FakeLinearBase(),
                "language_model.model.layers.0.self_attn.index_qkv_proj",
            )
            self.assertIsInstance(method, _FakeUnquantizedLinearMethod)

            method = quant_config.get_quant_method(
                _FakeLinearBase(),
                "language_model.model.layers.2.mlp.shared_experts.down_proj",
            )
            self.assertIsInstance(method, _FakeUnquantizedLinearMethod)

    def test_minimax_m3_moe_prefix_accepts_block_sparse_moe_w_names(self):
        with _load_modelslim_module() as modelslim:
            quant_config = modelslim.ModelSlimConfig(
                {
                    "model.layers.0.block_sparse_moe.experts.0.w1.weight": "W8A8_DYNAMIC",
                }
            )

            scheme = quant_config.get_moe_scheme(
                _FakeFusedMoE(), "model.layers.0.mlp.experts"
            )

            self.assertIsInstance(scheme, modelslim.ModelSlimW8A8Int8MoE)

    def test_missing_scheme_raises_actionable_error_before_create_weights(self):
        with _load_modelslim_module() as modelslim:
            quant_config = modelslim.ModelSlimConfig({})

            linear_layer = _FakeLinearBase()
            linear_layer.scheme = None
            with self.assertRaisesRegex(ValueError, "ModelSlim Linear"):
                modelslim.ModelSlimLinearMethod(quant_config).create_weights(
                    linear_layer,
                    input_size_per_partition=1,
                    output_partition_sizes=[1],
                    input_size=1,
                    output_size=1,
                    params_dtype=object(),
                )

            moe_layer = _FakeFusedMoE()
            moe_layer.scheme = None
            with self.assertRaisesRegex(ValueError, "ModelSlim MoE"):
                modelslim.ModelSlimFusedMoEMethod(quant_config).create_weights(
                    moe_layer,
                    num_experts=1,
                    hidden_size=1,
                    intermediate_size_per_partition=1,
                    params_dtype=object(),
                )


if __name__ == "__main__":
    unittest.main()
