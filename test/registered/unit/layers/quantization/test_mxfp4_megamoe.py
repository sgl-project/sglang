"""CPU contracts for compressed-tensors MXFP4 on SM90 MegaMoE."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# ``sglang.__init__`` installs this stub only on macOS/MPS, while CPU CI also
# needs it before importing the SGLang package. Load the repository-provided
# implementation directly to avoid importing ``sglang`` before the stub is in
# place (and avoid maintaining a second test-only Triton mock).
if importlib.util.find_spec("triton") is None:
    _stub_path = Path(__file__).parents[5] / "python/sglang/_triton_stub.py"
    _stub_spec = importlib.util.spec_from_file_location(
        "_sglang_cpu_triton_stub", _stub_path
    )
    if _stub_spec is None or _stub_spec.loader is None:
        raise ImportError(f"Unable to load CPU Triton stub from {_stub_path}")
    _triton_stub = importlib.util.module_from_spec(_stub_spec)
    sys.modules[_stub_spec.name] = _triton_stub
    _stub_spec.loader.exec_module(_triton_stub)
    _triton_stub.install()

# This CPU contract does not exercise image decoding. Some minimal CI images
# intentionally omit torchvision, while development CPU torch wheels can also
# lack torchvision's compiled NMS operator. Stub only the import used by
# sglang.srt.utils.common so test collection reaches the quantization code.
try:
    from torchvision.io import decode_jpeg as _decode_jpeg  # noqa: F401
except (ImportError, RuntimeError):
    _torchvision = types.ModuleType("torchvision")
    _torchvision_io = types.ModuleType("torchvision.io")
    _torchvision.__spec__ = importlib.util.spec_from_loader("torchvision", loader=None)
    _torchvision_io.__spec__ = importlib.util.spec_from_loader("torchvision.io", loader=None)
    _torchvision_io.decode_jpeg = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("torchvision decode_jpeg is unavailable in CPU contract tests")
    )
    _torchvision.io = _torchvision_io
    sys.modules["torchvision"] = _torchvision
    sys.modules["torchvision.io"] = _torchvision_io

import torch

from sglang.srt.layers.moe.mega_moe import (
    _MEGA_MOE_SYMM_BUFFER,
    _get_mega_moe_symm_buffer,
)
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.mega_moe_sm90 import (
    _resolve_sm90_fp4_symm_buffer_constructor,
    _resolve_sm90_fp4_weight_transform,
    _transform_weights_for_mega_moe_sm90_fp4_compat,
    build_sm90_fp4_mega_moe_experts_weights,
    is_sm90_fp4_mega_moe_available,
    run_sm90_mega_routed,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsFusedMoEMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A8Mxfp4MoE,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


MXFP4_GROUP = {
    "format": "mxfp4-pack-quantized",
    "targets": ["Linear"],
    "weights": {
        "num_bits": 4,
        "type": "float",
        "symmetric": True,
        "strategy": "group",
        "group_size": 32,
        "dynamic": False,
    },
    "input_activations": {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "strategy": "token",
        "dynamic": True,
    },
}


def _config(group=MXFP4_GROUP, *, top_format="mxfp4-pack-quantized"):
    return CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": top_format,
            "config_groups": {"group_0": group},
            "ignore": [],
        }
    )


def _scheme(config=None):
    config = config or _config()
    scheme_dict = config.target_scheme_map["Linear"]
    with (
        mock.patch(
            "sglang.srt.layers.quantization.compressed_tensors.schemes."
            "compressed_tensors_w4a8_mxfp4_moe.is_sm90_supported",
            return_value=True,
        ),
        mock.patch(
            "sglang.srt.layers.quantization.compressed_tensors.schemes."
            "compressed_tensors_w4a8_mxfp4_moe.is_sm100_supported",
            return_value=False,
        ),
    ):
        return CompressedTensorsW4A8Mxfp4MoE(
            config,
            scheme_dict["weights"],
            scheme_dict["input_activations"],
            scheme_dict["format"],
        )


class _DummyFusedMoE(torch.nn.Module):
    pass


class TestMxfp4SchemeSelection(CustomTestCase):
    def test_glm52_config_group_linear_and_null_scale_dtype(self):
        group = dict(MXFP4_GROUP)
        group["scale_dtype"] = None
        config = _config(group)
        self.assertEqual(config.target_scheme_map["Linear"]["format"], "mxfp4-pack-quantized")
        self.assertIsNone(config.target_scheme_map["Linear"]["weights"].scale_dtype)
        self.assertIsNotNone(config.target_scheme_map["Linear"]["input_activations"])

    def test_deepseek_packed_mapping_contract(self):
        source = (
            Path(__file__).resolve().parents[5]
            / "python/sglang/srt/models/deepseek_v2.py"
        ).read_text()
        self.assertIn('"gate_up_proj": ["gate_proj", "up_proj"]', source)

    def test_mxfp4_format_preserves_dynamic_fp8_activations(self):
        scheme_dict = _config().target_scheme_map["Linear"]
        self.assertEqual(scheme_dict["format"], "mxfp4-pack-quantized")
        self.assertIsNotNone(scheme_dict["input_activations"])
        self.assertEqual(scheme_dict["input_activations"].num_bits, 8)
        self.assertTrue(scheme_dict["input_activations"].dynamic)

    def test_exact_glm_scheme_bypasses_generic_mxfp4_method(self):
        config = _config()
        layer = _DummyFusedMoE()
        with (
            mock.patch(
                "sglang.srt.layers.moe.fused_moe_triton.FusedMoE",
                _DummyFusedMoE,
            ),
            mock.patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes."
                "compressed_tensors_w4a8_mxfp4_moe.is_sm90_supported",
                return_value=True,
            ),
            mock.patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes."
                "compressed_tensors_w4a8_mxfp4_moe.is_sm100_supported",
                return_value=False,
            ),
        ):
            method = config.get_quant_method(layer, "model.layers.0.mlp.experts")

        self.assertIsInstance(method, CompressedTensorsFusedMoEMethod)
        self.assertIsInstance(layer.scheme, CompressedTensorsW4A8Mxfp4MoE)

    def test_other_mxfp4_shape_keeps_generic_method(self):
        other_group = dict(MXFP4_GROUP)
        other_group["input_activations"] = None
        config = _config(other_group)
        layer = _DummyFusedMoE()
        sentinel = object()
        with (
            mock.patch(
                "sglang.srt.layers.moe.fused_moe_triton.FusedMoE",
                _DummyFusedMoE,
            ),
            mock.patch(
                "sglang.srt.layers.quantization.mxfp4.Mxfp4MoEMethod",
                return_value=sentinel,
            ) as generic_method,
        ):
            method = config.get_quant_method(layer, "model.layers.0.mlp.experts")

        self.assertIs(method, sentinel)
        generic_method.assert_called_once_with(prefix="model.layers.0.mlp.experts")

    def test_megamoe_scheme_rejects_sm100_without_sm90_fp4_runtime(self):
        config = _config()
        scheme_dict = config.target_scheme_map["Linear"]
        backend = SimpleNamespace(is_marlin=lambda: False, value="deep_gemm")
        with (
            mock.patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes."
                "compressed_tensors_w4a8_mxfp4_moe.get_moe_runner_backend",
                return_value=backend,
            ),
            mock.patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes."
                "compressed_tensors_w4a8_mxfp4_moe.is_sm90_supported",
                return_value=False,
            ),
            mock.patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes."
                "compressed_tensors_w4a8_mxfp4_moe.is_sm100_supported",
                return_value=True,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "requires SM90"):
                CompressedTensorsW4A8Mxfp4MoE(
                    config,
                    scheme_dict["weights"],
                    scheme_dict["input_activations"],
                    scheme_dict["format"],
                )

    def test_marlin_scheme_keeps_sm100_support(self):
        config = _config()
        scheme_dict = config.target_scheme_map["Linear"]
        backend = SimpleNamespace(is_marlin=lambda: True, value="marlin")
        with (
            mock.patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes."
                "compressed_tensors_w4a8_mxfp4_moe.get_moe_runner_backend",
                return_value=backend,
            ),
            mock.patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes."
                "compressed_tensors_w4a8_mxfp4_moe.is_sm90_supported",
                return_value=False,
            ),
            mock.patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes."
                "compressed_tensors_w4a8_mxfp4_moe.is_sm100_supported",
                return_value=True,
            ),
        ):
            scheme = CompressedTensorsW4A8Mxfp4MoE(
                config,
                scheme_dict["weights"],
                scheme_dict["input_activations"],
                scheme_dict["format"],
            )
        self.assertIsInstance(scheme, CompressedTensorsW4A8Mxfp4MoE)


class TestMxfp4PackedLoaderContract(CustomTestCase):
    def test_loader_mapping_hits_registered_gate_and_down_params(self):
        layer = torch.nn.Module()
        scheme = _scheme()
        scheme.create_weights(
            layer,
            num_experts=2,
            hidden_size=64,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
        )
        registered = dict(layer.named_parameters())

        mappings = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=2,
        )
        mapped = set()
        for param_prefix, checkpoint_prefix, _, _ in mappings:
            for suffix in ("weight_packed", "weight_scale"):
                mapped.add((param_prefix + suffix).removeprefix("experts."))
                self.assertTrue(checkpoint_prefix.startswith("experts."))

        self.assertIn("w13_weight_packed", mapped)
        self.assertIn("w2_weight_packed", mapped)
        self.assertIn("w13_weight_packed", registered)
        self.assertIn("w2_weight_packed", registered)
        self.assertNotIn("w13_weight", registered)
        self.assertNotIn("w2_weight", registered)

    def test_post_load_decodes_and_renames_without_generic_upcast(self):
        layer = torch.nn.Module()
        scheme = _scheme()
        scheme.create_weights(
            layer,
            num_experts=1,
            hidden_size=64,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
        )
        layer.w13_weight_scale.data.fill_(127)
        layer.w2_weight_scale.data.fill_(127)

        with mock.patch.object(scheme, "_build_mega_moe_weights") as build:
            scheme.process_weights_after_loading(layer)

        build.assert_called_once_with(layer)
        self.assertEqual(layer.w13_weight.dtype, torch.int8)
        self.assertEqual(layer.w2_weight.dtype, torch.int8)
        self.assertEqual(layer.w13_weight_scale_inv.dtype, torch.float32)
        self.assertEqual(layer.w2_weight_scale_inv.dtype, torch.float32)
        self.assertTrue(torch.all(layer.w13_weight_scale_inv == 1.0))
        self.assertTrue(torch.all(layer.w2_weight_scale_inv == 1.0))
        self.assertFalse(hasattr(layer, "w13_weight_packed"))
        self.assertFalse(hasattr(layer, "w2_weight_packed"))
        self.assertTrue(layer.is_mxfp4_converted)

    def test_real_weight_loader_places_each_expert_and_projection(self):
        layer = torch.nn.Module()
        scheme = _scheme()
        scheme.create_weights(
            layer,
            num_experts=2,
            hidden_size=64,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
        )

        loader = object.__new__(FusedMoE)
        torch.nn.Module.__init__(loader)
        loader.quant_config = _config()
        loader.quant_method = SimpleNamespace(load_up_proj_weight_first=False)
        loader.scheme = scheme
        loader.moe_runner_config = SimpleNamespace(is_gated=True)
        loader.moe_tp_rank = 0
        loader.moe_tp_size = 1
        loader.moe_ep_rank = 0
        loader.moe_ep_size = 1
        loader._expert_storage_rank = 0
        loader._num_local_routed = 2
        loader._num_global_routed = 2
        loader.num_local_experts = 2
        loader._has_fused_shared = False
        loader.num_fused_shared_experts = 0
        loader.use_presharded_weights = True
        loader.use_flashinfer_trtllm_moe = False
        loader.use_triton_kernels = False
        loader.__dict__["use_padded_loading"] = False

        values = {
            (0, "w1"): 11,
            (0, "w3"): 12,
            (0, "w2"): 13,
            (1, "w1"): 21,
            (1, "w3"): 22,
            (1, "w2"): 23,
        }
        params = dict(layer.named_parameters())
        with mock.patch(
            "sglang.srt.layers.moe.fused_moe_triton.layer."
            "get_global_expert_location_metadata",
            return_value=None,
        ):
            for expert_id in range(2):
                for shard_id in ("w1", "w3", "w2"):
                    param_prefix = "w13" if shard_id in ("w1", "w3") else "w2"
                    rows = 64
                    packed_cols = 32
                    scale_cols = 2
                    value = values[(expert_id, shard_id)]
                    loader.weight_loader(
                        params[f"{param_prefix}_weight_packed"],
                        torch.full((rows, packed_cols), value, dtype=torch.uint8),
                        f"experts.{param_prefix}_weight_packed",
                        shard_id,
                        expert_id,
                    )
                    loader.weight_loader(
                        params[f"{param_prefix}_weight_scale"],
                        torch.full((rows, scale_cols), value + 100, dtype=torch.uint8),
                        f"experts.{param_prefix}_weight_scale",
                        shard_id,
                        expert_id,
                    )

        for expert_id in range(2):
            self.assertTrue(
                torch.all(
                    layer.w13_weight_packed[expert_id, :64]
                    == values[(expert_id, "w1")]
                )
            )
            self.assertTrue(
                torch.all(
                    layer.w13_weight_packed[expert_id, 64:]
                    == values[(expert_id, "w3")]
                )
            )
            self.assertTrue(
                torch.all(
                    layer.w13_weight_scale[expert_id, :64]
                    == values[(expert_id, "w1")] + 100
                )
            )
            self.assertTrue(
                torch.all(
                    layer.w13_weight_scale[expert_id, 64:]
                    == values[(expert_id, "w3")] + 100
                )
            )
            self.assertTrue(
                torch.all(
                    layer.w2_weight_packed[expert_id]
                    == values[(expert_id, "w2")]
                )
            )
            self.assertTrue(
                torch.all(
                    layer.w2_weight_scale[expert_id]
                    == values[(expert_id, "w2")] + 100
                )
            )

    def test_megamoe_backend_fails_closed_without_sm90(self):
        scheme = _scheme()
        layer = torch.nn.Module()
        backend = SimpleNamespace(is_megamoe=lambda: True)

        with (
            mock.patch(
                "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
                return_value=backend,
            ),
            mock.patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes."
                "compressed_tensors_w4a8_mxfp4_moe.is_sm90_supported",
                return_value=False,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires the SM90 FP4"):
                scheme._build_mega_moe_weights(layer)


class TestSm90Fp4MegaMoEContract(CustomTestCase):
    @staticmethod
    def _experts():
        experts = torch.nn.Module()
        experts.register_parameter(
            "w13_weight",
            torch.nn.Parameter(
                torch.arange(2 * 128 * 64, dtype=torch.int32)
                .to(torch.uint8)
                .reshape(2, 128, 64)
                .view(torch.int8),
                requires_grad=False,
            ),
        )
        experts.register_parameter(
            "w2_weight",
            torch.nn.Parameter(
                torch.zeros((2, 64, 64), dtype=torch.int8), requires_grad=False
            ),
        )
        experts.register_parameter(
            "w13_weight_scale_inv",
            torch.nn.Parameter(
                torch.ones((2, 128, 4), dtype=torch.float32), requires_grad=False
            ),
        )
        experts.register_parameter(
            "w2_weight_scale_inv",
            torch.nn.Parameter(
                torch.ones((2, 64, 4), dtype=torch.float32), requires_grad=False
            ),
        )
        return experts

    def test_weight_builder_uses_sm90_fp4_transform(self):
        experts = self._experts()
        l1 = (
            torch.ones_like(experts.w13_weight),
            torch.ones((2, 128, 1), dtype=torch.int32),
        )
        l2 = (
            torch.ones_like(experts.w2_weight),
            torch.ones((2, 64, 1), dtype=torch.int32),
        )
        transform = mock.Mock(return_value=(l1, l2))
        deep_gemm = SimpleNamespace(
            fp8_fp4_mega_moe=mock.Mock(),
            mega_moe_pre_dispatch_sm90=mock.Mock(),
            transform_weights_for_mega_moe_sm90_fp4=transform,
        )

        with (
            mock.patch.dict("sys.modules", {"deep_gemm": deep_gemm}),
            mock.patch(
                "sglang.srt.layers.moe.mega_moe_sm90._env_bool", return_value=False
            ),
        ):
            build_sm90_fp4_mega_moe_experts_weights(experts)

        transform.assert_called_once()
        self.assertIs(experts.mega_l1_weights[0], l1[0])
        self.assertIs(experts.mega_l1_weights[1], l1[1])
        self.assertIs(experts.mega_l2_weights[0], l2[0])
        self.assertIs(experts.mega_l2_weights[1], l2[1])
        self.assertTrue(experts._mega_moe_sm90_fp4_weights)
        self.assertTrue(experts._mega_moe_weights_built)

    def test_weight_builder_uses_compat_transform_when_helper_is_absent(self):
        experts = self._experts()
        original_l1 = experts.w13_weight.detach().clone()
        deep_gemm = SimpleNamespace(
            fp8_fp4_mega_moe=mock.Mock(),
            mega_moe_pre_dispatch_sm90=mock.Mock(),
        )

        with (
            mock.patch.dict("sys.modules", {"deep_gemm": deep_gemm}),
            mock.patch(
                "sglang.srt.layers.moe.mega_moe_sm90._env_bool", return_value=False
            ),
        ):
            build_sm90_fp4_mega_moe_experts_weights(experts)

        expected_l1 = torch.stack(
            [
                original_l1[:, :64].reshape(2, 8, 8, 64),
                original_l1[:, 64:].reshape(2, 8, 8, 64),
            ],
            dim=2,
        ).reshape_as(original_l1)
        self.assertTrue(torch.equal(experts.mega_l1_weights[0], expected_l1))
        self.assertEqual(experts.mega_l1_weights[1].dtype, torch.int32)
        self.assertEqual(experts.mega_l1_weights[1].shape, (2, 128, 1))
        self.assertTrue(
            torch.all(experts.mega_l1_weights[1] == int.from_bytes(b"\x7f" * 4, "little"))
        )

    def test_compat_transform_rejects_non_ue8m0_scale(self):
        experts = self._experts()
        experts.w13_weight_scale_inv.data.fill_(1.5)
        with self.assertRaisesRegex(ValueError, "non-zero fp32 mantissa"):
            _transform_weights_for_mega_moe_sm90_fp4_compat(
                (experts.w13_weight, experts.w13_weight_scale_inv),
                (experts.w2_weight, experts.w2_weight_scale_inv),
            )

    def test_transform_resolution_fails_closed_without_public_kernel(self):
        deep_gemm = SimpleNamespace(mega_moe_pre_dispatch_sm90=mock.Mock())
        with self.assertRaisesRegex(RuntimeError, "fp8_fp4_mega_moe"):
            _resolve_sm90_fp4_weight_transform(deep_gemm)

    def test_availability_requires_symmetric_buffer_constructor(self):
        experts = SimpleNamespace(_mega_moe_sm90_fp4_weights=True)
        deep_gemm = SimpleNamespace(
            fp8_fp4_mega_moe=mock.Mock(),
            mega_moe_pre_dispatch_sm90=mock.Mock(),
        )
        with (
            mock.patch.dict("sys.modules", {"deep_gemm": deep_gemm}),
            mock.patch(
                "sglang.srt.layers.moe.mega_moe_sm90._device_sm", 90
            ),
        ):
            self.assertFalse(is_sm90_fp4_mega_moe_available(experts))
            deep_gemm.mega = SimpleNamespace(
                get_symm_buffer_for_mega_moe=mock.Mock()
            )
            self.assertTrue(is_sm90_fp4_mega_moe_available(experts))

    def test_fp4_buffer_constructor_uses_ring_buffer_abi(self):
        legacy_constructor = mock.Mock()
        ring_constructor = mock.Mock()
        deep_gemm = SimpleNamespace(
            get_symm_buffer_for_mega_moe=legacy_constructor,
            mega=SimpleNamespace(
                get_symm_buffer_for_mega_moe=ring_constructor,
            ),
        )

        self.assertIs(
            _resolve_sm90_fp4_symm_buffer_constructor(deep_gemm), ring_constructor
        )
        legacy_constructor.assert_not_called()

    def test_fp4_buffer_constructor_rejects_legacy_only_abi(self):
        deep_gemm = SimpleNamespace(get_symm_buffer_for_mega_moe=mock.Mock())
        with self.assertRaisesRegex(RuntimeError, "num_ring_tokens|ring-buffer"):
            _resolve_sm90_fp4_symm_buffer_constructor(deep_gemm)

    def test_fp4_buffer_factory_calls_ring_buffer_abi(self):
        legacy_constructor = mock.Mock()
        ring_buffer = SimpleNamespace(num_ring_tokens=256)
        ring_constructor = mock.Mock(return_value=ring_buffer)
        deep_gemm = SimpleNamespace(
            get_symm_buffer_for_mega_moe=legacy_constructor,
            mega=SimpleNamespace(
                get_symm_buffer_for_mega_moe=ring_constructor,
            ),
        )
        group = object()

        with mock.patch.dict("sys.modules", {"deep_gemm": deep_gemm}):
            result = _get_mega_moe_symm_buffer(
                group, 256, 8192, 8, 6144, 2048,
                use_sm90_fp4_ring_buffer=True,
            )

        self.assertIs(result, ring_buffer)
        ring_constructor.assert_called_once_with(
            group,
            256,
            8192,
            8,
            6144,
            2048,
            activation="swiglu",
            mma_type="fp8xfp4",
        )
        legacy_constructor.assert_not_called()
        _MEGA_MOE_SYMM_BUFFER.clear()

    def test_native_transform_output_contract_is_checked(self):
        experts = self._experts()
        deep_gemm = SimpleNamespace(
            fp8_fp4_mega_moe=mock.Mock(),
            mega_moe_pre_dispatch_sm90=mock.Mock(),
            transform_weights_for_mega_moe_sm90_fp4=mock.Mock(
                return_value=((torch.ones(1), torch.ones(1)), (torch.ones(1), torch.ones(1)))
            ),
        )
        with (
            mock.patch.dict("sys.modules", {"deep_gemm": deep_gemm}),
            self.assertRaisesRegex(TypeError, "weight must be int8"),
        ):
            build_sm90_fp4_mega_moe_experts_weights(experts)

    def test_dispatch_uses_fp8_fp4_kernel_not_fp8_kernel(self):
        pre_dispatch = mock.Mock()
        fp8_fp4 = mock.Mock()
        fp8 = mock.Mock()
        deep_gemm = SimpleNamespace(
            mega_moe_pre_dispatch_sm90=pre_dispatch,
            fp8_fp4_mega_moe=fp8_fp4,
            fp8_mega_moe=fp8,
        )
        experts = SimpleNamespace(
            _mega_moe_sm90_fp4_weights=True,
            should_fuse_routed_scaling_factor_in_topk=True,
            mega_l1_weights=(torch.empty(0), torch.empty(0)),
            mega_l2_weights=(torch.empty(0), torch.empty(0)),
        )
        moe = SimpleNamespace(
            experts=experts,
            config=SimpleNamespace(hidden_size=4, swiglu_limit=None),
            routed_scaling_factor=1.0,
        )
        buf = SimpleNamespace(
            x=torch.empty((1, 4)),
            x_sf=torch.empty((1, 1)),
            topk_idx=torch.empty((1, 1), dtype=torch.int32),
            topk_weights=torch.empty((1, 1)),
        )

        with (
            mock.patch.dict("sys.modules", {"deep_gemm": deep_gemm}),
            mock.patch(
                "sglang.srt.layers.moe.mega_moe_sm90._env_bool", return_value=False
            ),
        ):
            output = run_sm90_mega_routed(
                moe,
                torch.ones((1, 4), dtype=torch.bfloat16),
                torch.zeros((1, 1), dtype=torch.int32),
                torch.ones((1, 1), dtype=torch.float32),
                buf,
                1,
            )

        self.assertEqual(output.shape, (1, 4))
        pre_dispatch.assert_called_once()
        fp8_fp4.assert_called_once()
        fp8.assert_not_called()


if __name__ == "__main__":
    import unittest

    unittest.main()
