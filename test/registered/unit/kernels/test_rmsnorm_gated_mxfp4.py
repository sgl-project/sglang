"""CPU-only policy tests for fused gated RMSNorm MXFP4."""

import ast
import inspect
import math
import struct
import unittest

import torch

from sglang.kernels.ops.attention.fla import (
    rmsnorm_gated_mxfp4 as rmsnorm_gated_mxfp4_module,
)
from sglang.kernels.ops.attention.fla.rmsnorm_gated_mxfp4 import (
    MXFP4_BLOCK_SIZE,
    MXFP4_ROUND_EVEN,
    MXFP4_ROUND_UP,
    _is_mxfp4_shuffled_width_supported,
    _is_rmsnorm_gated_mxfp4_layout_supported,
    can_use_rmsnorm_gated_mxfp4,
    mxfp4_scale_shape,
    rmsnorm_gated_mxfp4_quant,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _float32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _bits_float32(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value))[0]


def _reference_scale(values: list[float], round_mode: int) -> tuple[int, float]:
    amax = max(abs(value) for value in values)
    if round_mode == MXFP4_ROUND_UP:
        amax = max(amax, 1.0e-10)
        bits = _float32_bits(amax * (1.0 / 6.0))
        exponent = (bits >> 23) & 0xFF
        if bits & 0x7FFFFF and exponent < 0xFF:
            exponent += 1
        return exponent, _bits_float32(exponent << 23)

    bits = (_float32_bits(amax) + 0x200000) & 0xFF800000
    rounded = _bits_float32(bits)
    unbiased = -127 if rounded == 0 else math.floor(math.log2(rounded)) - 2
    unbiased = min(max(unbiased, -127), 127)
    return unbiased + 127, math.ldexp(1.0, unbiased)


def _reference_pack(values: list[float], scale: float) -> list[int]:
    levels = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    nibbles = []
    for value in values:
        scaled_magnitude = abs(value / scale)
        magnitude = min(
            range(len(levels)), key=lambda index: abs(levels[index] - scaled_magnitude)
        )
        sign = 0x8 if math.copysign(1.0, value) < 0 else 0
        nibbles.append(magnitude | sign)
    return [
        nibbles[index] | (nibbles[index + 1] << 4)
        for index in range(0, len(nibbles), 2)
    ]


class TestRMSNormGatedMXFP4Policy(unittest.TestCase):
    @staticmethod
    def _source_tree() -> ast.Module:
        return ast.parse(inspect.getsource(rmsnorm_gated_mxfp4_module))

    @classmethod
    def _jit_functions(cls) -> dict[str, ast.FunctionDef]:
        functions = {}
        for node in cls._source_tree().body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if any(
                isinstance(decorator, ast.Attribute)
                and isinstance(decorator.value, ast.Name)
                and decorator.value.id == "triton"
                and decorator.attr == "jit"
                for decorator in node.decorator_list
            ):
                functions[node.name] = node
        return functions

    def test_scale_shapes_match_both_linear_consumers(self):
        self.assertEqual(mxfp4_scale_shape(4, 192, shuffle=False), (4, 192))
        self.assertEqual(mxfp4_scale_shape(4, 192, shuffle=True), (256, 192))
        self.assertEqual(mxfp4_scale_shape(257, 193, shuffle=True), (512, 200))

    def test_shuffled_activation_width_alignment(self):
        self.assertTrue(_is_mxfp4_shuffled_width_supported(6144))
        self.assertTrue(_is_mxfp4_shuffled_width_supported(768))
        self.assertFalse(_is_mxfp4_shuffled_width_supported(288))
        self.assertFalse(_is_mxfp4_shuffled_width_supported(128))

    def test_public_mxfp4_constants_remain_stable(self):
        self.assertEqual(MXFP4_BLOCK_SIZE, 32)
        self.assertEqual(MXFP4_ROUND_UP, 1)
        self.assertEqual(MXFP4_ROUND_EVEN, 2)

    def test_triton_jit_functions_capture_no_ordinary_module_globals(self):
        tree = self._source_tree()
        jit_functions = self._jit_functions()
        module_globals = {
            name.id
            for node in tree.body
            for name in ast.walk(node)
            if not isinstance(node, ast.FunctionDef)
            and isinstance(name, ast.Name)
            and isinstance(name.ctx, ast.Store)
        }
        module_globals.update(
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        )
        module_globals.update(
            alias.asname or alias.name.split(".", 1)[0]
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        )

        for function in jit_functions.values():
            body = ast.Module(body=function.body, type_ignores=[])
            local_names = {
                argument.arg
                for argument in (
                    *function.args.posonlyargs,
                    *function.args.args,
                    *function.args.kwonlyargs,
                )
            }
            local_names.update(
                name.id
                for name in ast.walk(body)
                if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)
            )
            loaded_names = {
                name.id
                for name in ast.walk(body)
                if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Load)
            }
            captures = loaded_names & module_globals - local_names - {"tl"} - set(
                jit_functions
            )
            self.assertEqual(
                captures,
                set(),
                f"{function.name} captures unsupported globals: {sorted(captures)}",
            )

    def test_mxfp4_compile_constants_are_explicitly_forwarded(self):
        jit_functions = self._jit_functions()
        for function_name in (
            "_mxfp4_quantize",
            "_rmsnorm_gated_mxfp4_kernel",
        ):
            annotations = {
                argument.arg: ast.unparse(argument.annotation)
                for argument in jit_functions[function_name].args.args
                if argument.annotation is not None
            }
            self.assertEqual(annotations["FP4_BLOCK_SIZE"], "tl.constexpr")
            self.assertEqual(annotations["ROUND_UP_MODE"], "tl.constexpr")

        kernel_source = ast.unparse(jit_functions["_rmsnorm_gated_mxfp4_kernel"])
        self.assertIn("FP4_BLOCK_SIZE=FP4_BLOCK_SIZE", kernel_source)
        self.assertIn("ROUND_UP_MODE=ROUND_UP_MODE", kernel_source)
        self.assertNotIn(
            "TRITON_ALLOW_NON_CONSTEXPR_GLOBALS",
            inspect.getsource(rmsnorm_gated_mxfp4_module),
        )

    def test_e2m1_sign_is_preserved_for_zero_magnitudes(self):
        helper_source = ast.unparse(self._jit_functions()["_mxfp4_quantize"])
        self.assertIn("e2m1 = e2m1 | (sign >> 28).to(tl.uint8)", helper_source)
        self.assertNotIn("tl.where(e2m1 != 0", helper_source)

        magnitudes = torch.tensor([0x0, 0x0, 0x1, 0x7], dtype=torch.uint8)
        signs = torch.tensor([0x0, 0x8, 0x8, 0x8], dtype=torch.uint8)
        self.assertEqual((magnitudes | signs).tolist(), [0x0, 0x8, 0x9, 0xF])

    def test_aiter_public_quantizer_signed_zero_vectors(self):
        mixed = (
            torch.tensor(
                [-0.0, 0.0, -1e-8, 1e-8, -0.24, 0.24, -0.26, 0.26] + [0.0] * 24,
                dtype=torch.bfloat16,
            )
            .float()
            .tolist()
        )
        negative_zero = torch.zeros(32, dtype=torch.bfloat16)
        negative_zero[0] = -0.0
        negative_zero = negative_zero.float().tolist()

        for round_mode, expected_zero_scale in (
            (MXFP4_ROUND_EVEN, 0),
            (MXFP4_ROUND_UP, 92),
        ):
            with self.subTest(round_mode=round_mode):
                scale_byte, scale = _reference_scale(mixed, round_mode)
                self.assertEqual(scale_byte, 123)
                self.assertEqual(_reference_pack(mixed, scale)[:4], [8, 8, 110, 110])

                zero_scale_byte, zero_scale = _reference_scale(
                    negative_zero, round_mode
                )
                self.assertEqual(zero_scale_byte, expected_zero_scale)
                self.assertEqual(_reference_pack(negative_zero, zero_scale)[0], 8)

    def test_round_up_floor_remains_exact(self):
        helper_source = ast.unparse(self._jit_functions()["_mxfp4_quantize"])
        self.assertIn("amax = tl.maximum(amax, 1e-10)", helper_source)
        scale_byte, _ = _reference_scale([-0.0] + [0.0] * 31, MXFP4_ROUND_UP)
        self.assertEqual(scale_byte, 92)

    def test_qwen38_strided_prefill_layout_is_supported(self):
        num_tokens, num_heads, head_dim = 4, 48, 128
        x_storage = torch.empty(
            (num_tokens * num_heads, head_dim + 32), dtype=torch.bfloat16
        )
        z_storage = torch.empty(
            (num_tokens, num_heads, head_dim * 2), dtype=torch.bfloat16
        )
        x = x_storage[:, :head_dim]
        z = z_storage[..., head_dim:]
        weight = torch.empty(head_dim, dtype=torch.bfloat16)

        self.assertTrue(
            _is_rmsnorm_gated_mxfp4_layout_supported(
                x,
                z,
                weight,
                num_heads=num_heads,
                activation="swish",
                shuffle_scales=True,
            )
        )

    def test_unaligned_width_only_rejects_shuffled_layout(self):
        x = torch.empty((3, 96), dtype=torch.bfloat16)
        z = torch.empty((1, 3, 96), dtype=torch.bfloat16)
        weight = torch.empty(96, dtype=torch.bfloat16)

        self.assertTrue(
            _is_rmsnorm_gated_mxfp4_layout_supported(
                x,
                z,
                weight,
                num_heads=3,
                activation="swish",
                shuffle_scales=False,
            )
        )
        self.assertFalse(
            _is_rmsnorm_gated_mxfp4_layout_supported(
                x,
                z,
                weight,
                num_heads=3,
                activation="swish",
                shuffle_scales=True,
            )
        )

    def test_incompatible_layouts_use_fallback(self):
        x = torch.empty((8, 128), dtype=torch.bfloat16)
        z = torch.empty((2, 4, 128), dtype=torch.bfloat16)
        weight = torch.empty(128, dtype=torch.bfloat16)

        cases = (
            (x[:0], z[:0], weight, 4, "swish"),
            (x, z, weight, 3, "swish"),
            (x, z[..., ::2], weight, 4, "swish"),
            (x, z, weight, 4, "gelu"),
            (x.to(torch.float32), z.to(torch.float32), weight.float(), 4, "swish"),
        )
        for case in cases:
            with self.subTest(case=case[3:]):
                self.assertFalse(
                    _is_rmsnorm_gated_mxfp4_layout_supported(
                        case[0],
                        case[1],
                        case[2],
                        num_heads=case[3],
                        activation=case[4],
                    )
                )

    def test_cpu_tensor_is_ineligible_without_launching_triton(self):
        x = torch.empty((4, 128), dtype=torch.bfloat16)
        z = torch.empty_like(x)
        weight = torch.empty(128, dtype=torch.bfloat16)

        self.assertFalse(
            can_use_rmsnorm_gated_mxfp4(x, z, weight, num_heads=4, activation="silu")
        )
        with self.assertRaisesRegex(ValueError, "device, shape, or layout"):
            rmsnorm_gated_mxfp4_quant(
                x,
                z,
                weight,
                1e-6,
                num_heads=4,
                activation="silu",
                round_mode=MXFP4_ROUND_EVEN,
                shuffle_scales=False,
                use_native_dtypes=False,
            )


if __name__ == "__main__":
    unittest.main()
