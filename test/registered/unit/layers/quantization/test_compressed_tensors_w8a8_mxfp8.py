"""CPU coverage for the compressed-tensors MXFP8 W8A8 scheme.

The GEMM itself needs Blackwell or XPU, so CI has no lane that executes it. What
is checkable on CPU is everything that decides *whether* the scheme is picked,
*which* MXFP8 dense kernel serves it, and *what layout* its operands end up in.
"""

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

maybe_stub_sgl_kernel()

import unittest  # noqa: E402
from contextlib import ExitStack  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest import mock  # noqa: E402

import torch  # noqa: E402
from compressed_tensors.quantization import QuantizationArgs  # noqa: E402

from sglang.srt.arg_groups.choices import FP8_GEMM_RUNNER_BACKEND_CHOICES  # noqa: E402
from sglang.srt.layers.quantization import fp8_utils  # noqa: E402
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E402
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (  # noqa: E402
    CompressedTensorsW8A8Fp8,
    CompressedTensorsW8A8MXFp8,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (  # noqa: E402
    compressed_tensors_w8a8_mxfp8 as schemes_mxfp8,
)
from sglang.srt.runtime_context import override_platform  # noqa: E402

LINEAR_LAYER = "Linear"
LINEAR_TARGET = "Linear"

MXFP8_GROUP_SIZE = 32

# OCP microscaling FP8: e4m3 elements, block-32 UE8M0 scales along K, on both
# operands. Activations are quantized per forward, so they are dynamic.
MXFP8_WEIGHTS = {
    "num_bits": 8,
    "type": "float",
    "symmetric": True,
    "strategy": "group",
    "group_size": MXFP8_GROUP_SIZE,
    "dynamic": False,
    "scale_dtype": "torch.uint8",
}
MXFP8_INPUT_ACTIVATIONS = {**MXFP8_WEIGHTS, "dynamic": True}

MXFP8_WEIGHT_QUANT = QuantizationArgs.model_validate(MXFP8_WEIGHTS)
MXFP8_INPUT_QUANT = QuantizationArgs.model_validate(MXFP8_INPUT_ACTIVATIONS)

# Plain per-channel/per-token FP8, the scheme the MXFP8 branch is checked ahead of.
FP8_WEIGHTS = {
    "num_bits": 8,
    "type": "float",
    "symmetric": True,
    "strategy": "channel",
    "dynamic": False,
}
FP8_INPUT_ACTIVATIONS = {
    "num_bits": 8,
    "type": "float",
    "symmetric": True,
    "strategy": "token",
    "dynamic": True,
}


def _config(quant_format, weights, input_activations):
    return {
        "quant_method": "compressed-tensors",
        "format": quant_format,
        "config_groups": {
            "group_0": {
                "targets": [LINEAR_TARGET],
                "weights": weights,
                "input_activations": input_activations,
            }
        },
        "ignore": [],
    }


def _pin_scheme_to_xpu_backend(test_case):
    # The scheme resolves the host's MXFP8 backend at construction and refuses any
    # but XPU, so a CUDA / ROCm host would otherwise fail every case that builds it.
    for name, value in (
        (
            "resolve_mxfp8_dense_gemm_backend",
            fp8_utils.Mxfp8DenseGemmBackend.XPU_SCALED_MM,
        ),
        ("dispatch_w8a8_mxfp8_linear", fp8_utils.xpu_mxfp8_scaled_mm_linear),
    ):
        patcher = mock.patch.object(schemes_mxfp8, name, return_value=value)
        patcher.start()
        test_case.addCleanup(patcher.stop)


def _resolve_scheme(quant_format, weights, input_activations):
    quant_config = CompressedTensorsConfig.from_config(
        _config(quant_format, weights, input_activations)
    )
    # Capability is a property of the machine, not of the routing logic under test.
    with mock.patch.object(
        CompressedTensorsConfig, "_check_scheme_supported", return_value=True
    ):
        return quant_config.get_linear_scheme(
            torch.nn.Module(), layer_name=LINEAR_LAYER
        )


class TestMXFp8Routing(CustomTestCase):
    def setUp(self):
        _pin_scheme_to_xpu_backend(self)

    def test_mxfp8_quantized_checkpoint_resolves_to_mxfp8_scheme(self):
        # Two independent gates have to agree for this to pass: "mxfp8-quantized"
        # must be an activation-quantization format (otherwise input_activations
        # is dropped during parsing and the whole activation block is skipped),
        # and _is_mxfp8_w8a8 must accept the MXFP8 preset's storage semantics.
        scheme = _resolve_scheme(
            "mxfp8-quantized", MXFP8_WEIGHTS, MXFP8_INPUT_ACTIVATIONS
        )
        self.assertIsInstance(scheme, CompressedTensorsW8A8MXFp8)
        self.assertEqual(scheme.group_size, MXFP8_GROUP_SIZE)

    def test_mxfp8_config_without_scale_dtype_still_resolves(self):
        # Configs written before compressed-tensors recorded scale_dtype omit it;
        # the mxfp8-quantized format alone then records the UE8M0 encoding.
        legacy = {
            name: {k: v for k, v in args.items() if k != "scale_dtype"}
            for name, args in (
                ("weights", MXFP8_WEIGHTS),
                ("input_activations", MXFP8_INPUT_ACTIVATIONS),
            )
        }
        scheme = _resolve_scheme(
            "mxfp8-quantized", legacy["weights"], legacy["input_activations"]
        )
        self.assertIsInstance(scheme, CompressedTensorsW8A8MXFp8)

    def test_grouped_fp8_with_float_scales_is_not_mxfp8(self):
        """An ordinary group-32 FP8 checkpoint has the same type/strategy/group
        size as MXFP8 but floating-point scales; serving it as MXFP8 would read
        those scale bytes as UE8M0 exponents and silently corrupt every output."""
        for scale_dtype in (None, "torch.float32", "torch.bfloat16"):
            with self.subTest(scale_dtype=scale_dtype):
                with self.assertRaises(NotImplementedError):
                    _resolve_scheme(
                        "float-quantized",
                        {**MXFP8_WEIGHTS, "scale_dtype": scale_dtype},
                        {**MXFP8_INPUT_ACTIVATIONS, "scale_dtype": scale_dtype},
                    )

    def test_channelwise_fp8_still_resolves_to_fp8_scheme(self):
        # The MXFP8 branch runs before the FP8 one; it must not swallow plain FP8.
        scheme = _resolve_scheme("float-quantized", FP8_WEIGHTS, FP8_INPUT_ACTIVATIONS)
        self.assertIsInstance(scheme, CompressedTensorsW8A8Fp8)
        self.assertNotIsInstance(scheme, CompressedTensorsW8A8MXFp8)

    def test_non_mxfp8_group_sizes_and_dtypes_are_not_claimed(self):
        # MXFP8 is block-32 float on both operands. Anything else must fall
        # through rather than reach a kernel that only implements BlockWise1x32.
        rejected = {
            "group_size_16": ({"group_size": 16}, {"group_size": 16}),
            "int_weights": ({"type": "int"}, {}),
            "int_activations": ({}, {"type": "int"}),
            "asymmetric_weights": ({"symmetric": False}, {}),
            "4bit_activations": ({}, {"num_bits": 4}),
            "float32_scales": (
                {"scale_dtype": "torch.float32"},
                {"scale_dtype": "torch.float32"},
            ),
        }
        for name, (weight_override, input_override) in rejected.items():
            with self.subTest(name=name):
                with self.assertRaises(NotImplementedError):
                    _resolve_scheme(
                        "mxfp8-quantized",
                        {**MXFP8_WEIGHTS, **weight_override},
                        {**MXFP8_INPUT_ACTIVATIONS, **input_override},
                    )

    def test_static_activation_scales_are_rejected_not_ignored(self):
        # apply_weights re-quantizes activations on every forward and never reads
        # a checkpoint input_scale, so a static-activation checkpoint has to fail
        # loudly rather than load and serve with the scale silently dropped.
        with self.assertRaises(NotImplementedError):
            _resolve_scheme(
                "mxfp8-quantized",
                MXFP8_WEIGHTS,
                {**MXFP8_INPUT_ACTIVATIONS, "dynamic": False},
            )


class TestMXFp8WeightLayout(CustomTestCase):
    """The operand layout ``scaled_mm``'s BlockWise1x32 probe expects."""

    def setUp(self):
        _pin_scheme_to_xpu_backend(self)

    K = 128
    OUTPUT_PARTITIONS = [64, 32]  # a fused gate_up_proj, so N = 96

    def _loaded_layer(self):
        scheme = CompressedTensorsW8A8MXFp8(
            weight_quant=MXFP8_WEIGHT_QUANT, input_quant=MXFP8_INPUT_QUANT
        )
        layer = torch.nn.Module()
        scheme.create_weights(
            layer=layer,
            input_size_per_partition=self.K,
            output_partition_sizes=self.OUTPUT_PARTITIONS,
            input_size=self.K,
            output_size=sum(self.OUTPUT_PARTITIONS),
            params_dtype=torch.bfloat16,
            weight_loader=lambda *args, **kwargs: None,
        )
        return scheme, layer

    def test_checkpoint_shapes_and_dtypes(self):
        _, layer = self._loaded_layer()
        n = sum(self.OUTPUT_PARTITIONS)

        # Rows span every shard of a fused projection, not just the first.
        self.assertEqual(tuple(layer.weight.shape), (n, self.K))
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        # One UE8M0 byte per block of 32 elements along K.
        self.assertEqual(
            tuple(layer.weight_scale.shape), (n, self.K // MXFP8_GROUP_SIZE)
        )
        self.assertEqual(layer.weight_scale.dtype, torch.uint8)

    def test_k_must_be_a_whole_number_of_blocks(self):
        scheme = CompressedTensorsW8A8MXFp8(
            weight_quant=MXFP8_WEIGHT_QUANT, input_quant=MXFP8_INPUT_QUANT
        )
        with self.assertRaises(AssertionError):
            scheme.create_weights(
                layer=torch.nn.Module(),
                input_size_per_partition=self.K + 1,
                output_partition_sizes=[64],
                input_size=self.K + 1,
                output_size=64,
                params_dtype=torch.bfloat16,
                weight_loader=lambda *args, **kwargs: None,
            )

    def test_post_load_keeps_checkpoint_layout_for_dispatch(self):
        scheme, layer = self._loaded_layer()
        n = sum(self.OUTPUT_PARTITIONS)

        scale_bytes = torch.arange(
            n * (self.K // MXFP8_GROUP_SIZE), dtype=torch.uint8
        ).reshape(n, self.K // MXFP8_GROUP_SIZE)
        layer.weight_scale.data.copy_(scale_bytes)

        scheme.process_weights_after_loading(layer)

        # Every dispatch_w8a8_mxfp8_linear kernel takes weight as [N, K] and
        # transposes at the call; a load-time transpose would feed them B twice flipped.
        self.assertEqual(tuple(layer.weight.shape), (n, self.K))
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        # The scale stays raw ue8m0 bytes; the XPU kernel reinterprets them per call.
        self.assertEqual(layer.weight_scale.dtype, torch.uint8)
        self.assertTrue(torch.equal(layer.weight_scale.data, scale_bytes))


class TestMxfp8DenseBackendResolution(CustomTestCase):
    def _resolve(self, *, is_xpu, deep_gemm_enabled):
        with (
            mock.patch.object(fp8_utils, "_is_xpu", is_xpu),
            mock.patch.object(fp8_utils, "_is_hip", False),
            mock.patch.object(
                fp8_utils.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", deep_gemm_enabled
            ),
            # Blackwell with FlashInfer outranks DeepGEMM; keep the host out of it.
            override_platform(is_blackwell=False),
        ):
            return (
                fp8_utils.resolve_mxfp8_dense_gemm_backend(),
                fp8_utils.dispatch_w8a8_mxfp8_linear(),
            )

    def test_xpu_resolves_to_scaled_mm_kernel(self):
        backend, linear = self._resolve(is_xpu=True, deep_gemm_enabled=False)
        self.assertTrue(backend.is_xpu_scaled_mm())
        self.assertIs(linear, fp8_utils.xpu_mxfp8_scaled_mm_linear)

    def test_scheme_rejects_backends_needing_scale_relayout(self):
        # FlashInfer / DeepGEMM read a re-laid-out scale this scheme never builds;
        # handing them the raw checkpoint scale would compute garbage silently.
        with mock.patch.object(
            schemes_mxfp8,
            "resolve_mxfp8_dense_gemm_backend",
            return_value=fp8_utils.Mxfp8DenseGemmBackend.FLASHINFER_CUTLASS,
        ):
            with self.assertRaises(NotImplementedError):
                CompressedTensorsW8A8MXFp8(
                    weight_quant=MXFP8_WEIGHT_QUANT, input_quant=MXFP8_INPUT_QUANT
                )

    def test_xpu_branch_does_not_claim_cuda(self):
        # On CUDA the XPU branch must stay inert so FlashInfer / DeepGEMM still win.
        backend, linear = self._resolve(is_xpu=False, deep_gemm_enabled=True)
        self.assertTrue(backend.is_deep_gemm())
        self.assertIs(linear, fp8_utils.deepgemm_w8a8_mxfp8_linear_with_fallback)


class TestFp8GemmBackendChoices(CustomTestCase):
    def test_every_cli_choice_is_a_runner_backend(self):
        # initialize_fp8_gemm_config does Fp8GemmRunnerBackend(choice), so a choice
        # argparse accepts but the enum lacks fails at startup instead of at parse.
        backends = {backend.value for backend in fp8_utils.Fp8GemmRunnerBackend}
        for choice in FP8_GEMM_RUNNER_BACKEND_CHOICES:
            with self.subTest(choice=choice):
                self.assertIn(choice, backends)


class TestExplicitXpuScaledMmBackend(CustomTestCase):
    """--fp8-gemm-backend=xpu_scaled_mm, from config init through both dispatchers."""

    def _initialize(self, *, is_xpu, flag="xpu_scaled_mm"):
        exec_cfg = SimpleNamespace(kernel=SimpleNamespace(fp8_gemm_runner_backend=flag))
        stack = ExitStack()
        stack.enter_context(mock.patch.object(fp8_utils, "_is_xpu", is_xpu))
        # SM120 / SM110 rewrite "auto" before the XPU rule; keep the host out of them.
        stack.enter_context(
            override_platform(is_xpu=is_xpu, is_sm120=False, is_sm110=False)
        )
        stack.enter_context(mock.patch.object(fp8_utils, "_is_hip", False))
        stack.enter_context(
            mock.patch.object(fp8_utils, "FP8_GEMM_RUNNER_BACKEND", None)
        )
        stack.enter_context(
            mock.patch.object(fp8_utils, "get_exec", return_value=exec_cfg)
        )
        self.addCleanup(stack.close)
        fp8_utils.initialize_fp8_gemm_config()

    def test_config_init_accepts_the_cli_choice(self):
        # Startup used to raise ValueError: the choice parsed but had no enum member.
        self._initialize(is_xpu=True)
        self.assertTrue(fp8_utils.get_fp8_gemm_runner_backend().is_xpu_scaled_mm())

    def test_auto_resolves_to_xpu_scaled_mm_only_on_xpu(self):
        # The default flag is "auto"; on XPU startup pins it to the XPU backend,
        # and the rule must not reach any other platform.
        self._initialize(is_xpu=True, flag="auto")
        self.assertTrue(fp8_utils.get_fp8_gemm_runner_backend().is_xpu_scaled_mm())
        self._initialize(is_xpu=False, flag="auto")
        self.assertTrue(fp8_utils.get_fp8_gemm_runner_backend().is_auto())

    def test_xpu_routes_block_fp8_and_mxfp8_to_scaled_mm(self):
        self._initialize(is_xpu=True)
        # Block-FP8 128-wide layers must not reach the "Unknown FP8 GEMM backend" else.
        self.assertIs(
            fp8_utils.dispatch_w8a8_block_fp8_linear(weight_block_size=[128, 128]),
            fp8_utils.torch_w8a8_block_fp8_linear,
        )
        self.assertTrue(fp8_utils.resolve_mxfp8_dense_gemm_backend().is_xpu_scaled_mm())
        self.assertIs(
            fp8_utils.dispatch_w8a8_mxfp8_linear(),
            fp8_utils.xpu_mxfp8_scaled_mm_linear,
        )

    def test_non_xpu_rejects_the_choice_for_both_formats(self):
        self._initialize(is_xpu=False)
        with self.assertRaises(RuntimeError):
            fp8_utils.dispatch_w8a8_block_fp8_linear(weight_block_size=[128, 128])
        with self.assertRaises(RuntimeError):
            fp8_utils.resolve_mxfp8_dense_gemm_backend()


class TestXpuMxfp8ScaledMmOperands(CustomTestCase):
    """Operand layout handed to scaled_mm; CPU torch has no BlockWise1x32 kernel."""

    M, K, N = 3, 64, 16

    def _capture_scaled_mm(self, **kwargs):
        captured = {}

        def fake_scaled_mm(a, b, scale_a, recipe_a, scale_b, recipe_b, **kw):
            captured.update(
                a=a,
                b=b,
                scale_a=scale_a,
                recipe_a=recipe_a,
                scale_b=scale_b,
                recipe_b=recipe_b,
                **kw,
            )
            return torch.zeros(a.shape[0], b.shape[1], dtype=kw["output_dtype"])

        with mock.patch.object(torch.nn.functional, "scaled_mm", fake_scaled_mm):
            out = fp8_utils.xpu_mxfp8_scaled_mm_linear(**kwargs)
        return out, captured

    def test_weight_and_scale_operand_layout(self):
        weight = torch.zeros(self.N, self.K, dtype=torch.float8_e4m3fn)
        scale_bytes = torch.arange(
            self.N * self.K // MXFP8_GROUP_SIZE, dtype=torch.uint8
        ).reshape(self.N, self.K // MXFP8_GROUP_SIZE)
        x_q = torch.zeros(2, self.M, self.K, dtype=torch.float8_e4m3fn)
        x_scale = torch.zeros(2 * self.M, self.K // MXFP8_GROUP_SIZE, dtype=torch.uint8)

        out, got = self._capture_scaled_mm(
            input=x_q, weight=weight, weight_scale=scale_bytes, input_scale=x_scale
        )

        # B is [K, N] column-major: a transposed view of the [N, K] weight, no copy.
        self.assertEqual(tuple(got["b"].shape), (self.K, self.N))
        self.assertEqual(got["b"].data_ptr(), weight.data_ptr())
        self.assertEqual(got["b"].stride(), (1, self.K))
        # Scales are reinterpreted as e8m0, never value-converted: the bytes survive.
        # A .to(float8_e8m0fnu) would exponent-encode each byte, scaling by 2**garbage.
        self.assertEqual(got["scale_b"].dtype, torch.float8_e8m0fnu)
        self.assertTrue(torch.equal(got["scale_b"].view(torch.uint8), scale_bytes))
        self.assertTrue(got["scale_b"].is_contiguous())
        self.assertEqual(got["scale_a"].dtype, torch.float8_e8m0fnu)
        for recipe in (got["recipe_a"], got["recipe_b"]):
            self.assertEqual(recipe, torch.nn.functional.ScalingType.BlockWise1x32)
        # Leading dims are flattened for the GEMM and restored on the output.
        self.assertEqual(tuple(got["a"].shape), (2 * self.M, self.K))
        self.assertEqual(tuple(out.shape), (2, self.M, self.N))

    def test_dynamic_activation_scale_is_row_major_e8m0(self):
        # The sgl-kernel-xpu quant kernel writes row-major [M, K/32] e8m0, the only
        # activation layout the BlockWise1x32 probe accepts.
        quant_calls = {}

        def fake_quant(**kw):
            quant_calls.update(kw)

        weight = torch.zeros(self.N, self.K, dtype=torch.float8_e4m3fn)
        scale = torch.zeros(self.N, self.K // MXFP8_GROUP_SIZE, dtype=torch.uint8)
        x = torch.randn(self.M, self.K, dtype=torch.bfloat16)
        with mock.patch.dict(
            "sys.modules",
            {"sgl_kernel": mock.MagicMock(sgl_per_token_group_quant_8bit=fake_quant)},
        ):
            out, got = self._capture_scaled_mm(
                input=x, weight=weight, weight_scale=scale
            )

        self.assertIs(got["scale_a"], quant_calls["output_s"])
        self.assertEqual(
            tuple(got["scale_a"].shape), (self.M, self.K // MXFP8_GROUP_SIZE)
        )
        self.assertTrue(got["scale_a"].is_contiguous())
        self.assertEqual(got["scale_a"].dtype, torch.float8_e8m0fnu)
        self.assertTrue(quant_calls["scale_ue8m0"])
        self.assertEqual(quant_calls["group_size"], MXFP8_GROUP_SIZE)
        self.assertEqual(out.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
