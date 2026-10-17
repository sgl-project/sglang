"""SM80 block-FP8 Marlin regression through the real linear-method lifecycle.

Run directly with unittest; no model files, server, or distributed runtime are
required. The large shapes below are synthetic coverage, not model-config or
checkpoint validation. Only group32 and the existing group128 path are tested.
"""

import ast
import os
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

M_SIZES = (1, 8, 9, 16, 17, 32, 48, 64, 128)
GROUP_SIZES = (32, 128)


def _is_sm80():
    return (
        torch.version.hip is None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (8, 0)
    )


def _checkpoint(n, k, group_size, *, seed, exact=False):
    """Build checkpoint FP8 payload and E8M0 scales entirely on the CPU."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if exact:
        # Normal E4M3 values, exactly representable after Marlin's exponent-bias
        # conversion. Tiny/subnormal FP8 values can round during that conversion
        # and are deliberately not part of this zero-tolerance routing oracle.
        values = torch.tensor([-3.0, -1.5, -0.5, 0.75, 1.0, 2.0])
        rows = torch.arange(n).view(n, 1)
        cols = torch.arange(k).view(1, k)
        weight = values[(rows + 3 * cols) % values.numel()]
        scale_rows = torch.arange(n // group_size).view(-1, 1)
        scale_cols = torch.arange(k // group_size).view(1, -1)
        exponents = (124 + (scale_rows + 2 * scale_cols) % 5).to(torch.uint8)
    else:
        weight = torch.randn(n, k, generator=generator, device="cpu") * 0.25
        exponents = torch.randint(
            124,
            129,
            (n // group_size, k // group_size),
            generator=generator,
            dtype=torch.uint8,
            device="cpu",
        )
    weight = weight.to(torch.float8_e4m3fn).contiguous()
    # This is the serialized UE8M0 scale representation. The block-FP8 loader
    # destination is FP32, unlike the uint8 destination of the MXFP8 path.
    scales = exponents.contiguous().view(torch.float8_e8m0fnu).float()
    expanded = scales.repeat_interleave(group_size, 0).repeat_interleave(group_size, 1)
    reference_weight = (weight.float() * expanded).contiguous()
    return weight, scales, reference_weight


def _load_marlin_functions():
    """Exercise Python metadata/byte padding on CPU without CUDA SRT imports."""
    source = (
        Path(__file__).resolve().parents[5]
        / "python/sglang/srt/layers/quantization/marlin_utils_fp8.py"
    )
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for node in functions:
        node.decorator_list = []
    namespace = {
        "torch": torch,
        "envs": runpy.run_path(str(source.parents[2] / "environ.py"))["envs"],
        "Optional": __import__("typing").Optional,
        "USE_FP32_REDUCE_DEFAULT": True,
        "logger": SimpleNamespace(warning_once=lambda *args: None),
    }
    exec(  # noqa: S102 - execute only local function ASTs with controlled globals
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace


class TestFp8MarlinKPaddingCPU(unittest.TestCase):
    def test_raw_bytes_and_neutral_scales_both_layouts(self):
        functions = _load_marlin_functions()
        prepare = functions["prepare_fp8_layer_for_marlin"]
        for size_k_first in (False, True):
            with self.subTest(size_k_first=size_k_first):
                weight, scales, _ = _checkpoint(5120, 288, 32, seed=17, exact=True)
                layer = torch.nn.Module()
                layer.input_size_per_partition = 288
                layer.output_size_per_partition = 5120
                layer.weight_block_size = [32, 32]
                layer.orig_dtype = torch.bfloat16
                raw_weight = weight.T.contiguous() if size_k_first else weight
                layer.weight = torch.nn.Parameter(raw_weight, False)
                layer.weight_scale_inv = torch.nn.Parameter(
                    scales.T.contiguous() if size_k_first else scales, False
                )
                captured = {}

                def repack(captured=captured, **kwargs):
                    captured["repack"] = kwargs
                    return torch.empty(20, 5120 * 4, dtype=torch.int32)

                def permute(captured=captured, **kwargs):
                    captured["scales"] = kwargs["s"].clone()
                    return kwargs["s"]

                functions.update(
                    _is_experimental_sm80_fp8_marlin=lambda device: True,
                    marlin_make_workspace=lambda device: torch.empty(0),
                    gptq_marlin_repack=repack,
                    marlin_permute_scales=permute,
                )
                prepare(layer, size_k_first)
                packed = captured["repack"]["b_q_weight"]
                raw = packed.T.contiguous().view(torch.uint8)
                torch.testing.assert_close(
                    raw[:, :288], weight.view(torch.uint8), rtol=0, atol=0
                )
                self.assertEqual(torch.count_nonzero(raw[:, 288:]).item(), 0)
                self.assertEqual(captured["repack"]["size_k"], 320)
                self.assertEqual(
                    (layer.input_size_per_partition, layer.output_size_per_partition),
                    (288, 5120),
                )
                scale_grid = captured["scales"]
                expected = scales.T.to(torch.bfloat16).repeat_interleave(32, 1)
                torch.testing.assert_close(scale_grid[:9], expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    scale_grid[9],
                    torch.ones(5120, dtype=torch.bfloat16),
                    rtol=0,
                    atol=0,
                )

    def test_gate_is_exact_and_opt_in(self):
        functions = _load_marlin_functions()
        gate = functions["_is_experimental_sm80_fp8_marlin"]
        for flag, cuda, hip, device, capability, expected in (
            ("1", "12.8", None, "cuda", (8, 0), True),
            ("0", "12.8", None, "cuda", (8, 0), False),
            ("true", "12.8", None, "cuda", (8, 0), False),
            ("1", None, None, "cuda", (8, 0), False),
            ("1", "12.8", "6.3", "cuda", (8, 0), False),
            ("1", "12.8", None, "cpu", (8, 0), False),
            ("1", "12.8", None, "cuda", (8, 6), False),
            ("1", "12.8", None, "cuda", (9, 0), False),
        ):
            with (
                self.subTest(flag=flag, device=device, capability=capability),
                patch.dict(os.environ, {"SGLANG_DSV41_SM80_EXPERIMENTAL": flag}),
                patch.object(torch.version, "cuda", cuda),
                patch.object(torch.version, "hip", hip),
                patch.object(
                    torch.cuda, "get_device_capability", return_value=capability
                ),
            ):
                self.assertEqual(gate(torch.device(device)), expected)

    def test_default_group128_channelwise_and_non_sm80_do_not_pad(self):
        for block_size, dtype, enabled in (
            ([32, 32], torch.bfloat16, False),
            ([32, 32], torch.float16, True),
            ([128, 128], torch.bfloat16, True),
            (None, torch.bfloat16, True),
        ):
            with self.subTest(block_size=block_size, dtype=dtype, enabled=enabled):
                functions = _load_marlin_functions()
                layer = torch.nn.Module()
                layer.input_size_per_partition = 288
                layer.output_size_per_partition = 5120
                layer.weight_block_size = block_size
                layer.orig_dtype = dtype
                layer.weight = torch.nn.Parameter(
                    torch.zeros(5120, 288).to(torch.float8_e4m3fn), False
                )
                sentinel = RuntimeError(
                    "stop after observing unchanged logical repack K"
                )

                def repack(sentinel=sentinel, **kwargs):
                    self.assertEqual(kwargs["size_k"], 288)
                    raise sentinel

                functions.update(
                    _is_experimental_sm80_fp8_marlin=lambda device, enabled=enabled: (
                        enabled
                    ),
                    _pad_fp8_marlin_k=lambda *args: self.fail("unexpected K padding"),
                    marlin_make_workspace=lambda device: torch.empty(0),
                    gptq_marlin_repack=repack,
                )
                with self.assertRaisesRegex(RuntimeError, "unchanged logical repack K"):
                    functions["prepare_fp8_layer_for_marlin"](layer, False)

    def test_apply_pads_only_physical_k_difference(self):
        for physical_k in (288, 320):
            with self.subTest(physical_k=physical_k):
                functions = _load_marlin_functions()
                x = torch.arange(2 * 6 * 288).reshape(2, 6, 288).to(torch.bfloat16)
                bias = torch.arange(5120).to(torch.bfloat16)

                def gemm(physical_k=physical_k, x=x, **kwargs):
                    self.assertEqual(kwargs["size_k"], physical_k)
                    a = kwargs["a"]
                    self.assertTrue(a.is_contiguous())
                    self.assertEqual(tuple(a.shape), (12, physical_k))
                    torch.testing.assert_close(
                        a[:, :288], x.reshape(12, 288), rtol=0, atol=0
                    )
                    self.assertEqual(torch.count_nonzero(a[:, 288:]).item(), 0)
                    return torch.zeros(12, 5120, dtype=torch.bfloat16)

                functions.update(
                    scalar_types=SimpleNamespace(float8_e4m3fn=object()),
                    should_use_atomic_add_reduce=lambda **kwargs: False,
                    gptq_marlin_gemm=gemm,
                )
                result = functions["apply_fp8_marlin_linear"](
                    x,
                    torch.empty(physical_k // 16, 5120 * 4, dtype=torch.int32),
                    torch.empty(10, 5120),
                    torch.empty(0),
                    5120,
                    288,
                    bias,
                )
                torch.testing.assert_close(
                    result, bias.expand(2, 6, -1), rtol=0, atol=0
                )

    def test_fake_output_remains_logical(self):
        fake = _load_marlin_functions()["fake_apply_fp8_marlin_linear"]
        result = fake(
            torch.empty(2, 6, 288, dtype=torch.bfloat16),
            torch.empty(20, 5120 * 4, dtype=torch.int32),
            torch.empty(10, 5120),
            torch.empty(0),
            5120,
            288,
            None,
        )
        self.assertEqual(tuple(result.shape), (2, 6, 5120))
        self.assertEqual(result.dtype, torch.bfloat16)


@unittest.skipUnless(_is_sm80(), "requires CUDA SM80 (8, 0)")
class TestFp8MarlinGroup32SM80(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the actual runtime only after the architecture gate; CPU hosts
        # can collect/skip this class without importing CUDA-only SRT providers.
        from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

        cls.fp8_config = Fp8Config
        cls.fp8_method = Fp8LinearMethod
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        cls.addClassCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", old_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False

    def _prepare(
        self, n, k, group_size, *, seed=17, exact=False, bias=False, size_k_first=False
    ):
        config = self.fp8_config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[group_size, group_size],
            scale_fmt="ue8m0",
        )
        method = self.fp8_method(config)
        self.assertTrue(method.use_marlin)
        self.assertFalse(method.use_mxfp8)
        self.assertTrue(method.block_quant)
        layer = torch.nn.Module()
        layer.tp_group = None
        with torch.device("cuda"):
            method.create_weights(layer, k, [n], k, n, torch.bfloat16)
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.weight_scale_inv.dtype, torch.float32)
        self.assertEqual(tuple(layer.weight.shape), (n, k))
        self.assertEqual(
            tuple(layer.weight_scale_inv.shape), (n // group_size, k // group_size)
        )
        self.assertTrue(layer.weight.is_cuda)
        self.assertTrue(layer.weight.is_contiguous())
        weight, scales, reference_weight = _checkpoint(
            n, k, group_size, seed=seed, exact=exact
        )
        with torch.no_grad():
            # Mirror checkpoint-loader copies into create_weights parameters;
            # never quantize FP8 payloads or construct E8M0 scales on SM80.
            layer.weight.copy_(weight.to(device="cuda"))
            layer.weight_scale_inv.copy_(scales.to(device="cuda"))
        if bias:
            layer.register_parameter(
                "bias",
                torch.nn.Parameter(
                    ((torch.arange(n, device="cuda") % 17 - 8) / 16).to(torch.bfloat16),
                    requires_grad=False,
                ),
            )
            expected_bias = layer.bias.detach().clone()
        if size_k_first:
            from sglang.srt.layers.quantization.marlin_utils_fp8 import (
                prepare_fp8_layer_for_marlin,
            )

            layer.weight = torch.nn.Parameter(layer.weight.T.contiguous(), False)
            layer.weight_scale_inv = torch.nn.Parameter(
                layer.weight_scale_inv.T.contiguous(), False
            )
            layer.weight_block_size = [group_size, group_size]
            prepare_fp8_layer_for_marlin(layer, size_k_first=True)
            del layer.input_scale
        else:
            method.process_weights_after_loading(layer)
        self.assertEqual(layer.input_size_per_partition, k)
        self.assertEqual(layer.output_size_per_partition, n)
        physical_k = layer.weight.shape[0] * 16
        self.assertEqual(layer.weight.dtype, torch.int32)
        self.assertTrue(layer.weight.is_cuda)
        self.assertTrue(layer.weight.is_contiguous())
        self.assertEqual(layer.weight_scale.dtype, torch.bfloat16)
        self.assertEqual(tuple(layer.weight_scale.shape), (physical_k // group_size, n))
        self.assertTrue(layer.weight_scale.is_cuda)
        self.assertTrue(layer.weight_scale.is_contiguous())
        self.assertFalse(hasattr(layer, "weight_scale_inv"))
        self.assertFalse(hasattr(layer, "input_scale"))
        self.assertIsNone(getattr(layer, "_block_fp8_bf16_weight", None))
        if bias:
            torch.testing.assert_close(layer.bias, expected_bias, rtol=0, atol=0)
        return method, layer, reference_weight.to(device="cuda")

    def _assert_random(self, method, layer, reference_weight, m, seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(m, reference_weight.shape[1], generator=generator).to(
            device="cuda", dtype=torch.bfloat16
        )
        actual = method.apply(layer, x)
        expected = (x.float() @ reference_weight.T).to(torch.bfloat16)
        torch.cuda.synchronize()
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(tuple(actual.shape), (m, reference_weight.shape[0]))
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.125)

    def test_grouped_random_reference(self):
        # M<=8 and M>=9 select different Marlin M-block branches; the remaining
        # sizes exercise 16/32/64 tile edges and multi-tile prefill.
        for group_size in GROUP_SIZES:
            method, layer, reference_weight = self._prepare(256, 512, group_size)
            for m in M_SIZES:
                with self.subTest(group_size=group_size, m=m):
                    self._assert_random(method, layer, reference_weight, m, 100 + m)

    def test_grouped_single_hot_exact_and_bias(self):
        # Include K boundaries around both grouping schemes and multiple tiles.
        columns = torch.tensor([0, 31, 32, 63, 64, 127, 128, 255, 256, 511])
        for group_size in GROUP_SIZES:
            method, layer, reference_weight = self._prepare(
                256, 512, group_size, exact=True, bias=True
            )
            for m in M_SIZES:
                indices = columns[torch.arange(m) % columns.numel()].to("cuda")
                x = torch.zeros(m, 512, dtype=torch.bfloat16, device="cuda")
                x[torch.arange(m, device="cuda"), indices] = 1
                expected = reference_weight[:, indices].T.to(torch.bfloat16)
                for with_bias in (False, True):
                    with self.subTest(group_size=group_size, m=m, with_bias=with_bias):
                        actual = method.apply(
                            layer, x, bias=layer.bias if with_bias else None
                        )
                        target = expected + layer.bias if with_bias else expected
                        torch.cuda.synchronize()
                        torch.testing.assert_close(actual, target, rtol=0, atol=0)

    def test_shared_expert_k288_tail_padding(self):
        # Real TP8 shared-expert down-projection dimensions; logical checkpoint
        # K has nine group32 scales while the physical Marlin K needs ten.
        from sglang.srt.layers.quantization.marlin_utils_fp8 import (
            fp8_fused_exponent_bias_into_scales,
        )

        with patch.dict(os.environ, {"SGLANG_DSV41_SM80_EXPERIMENTAL": "1"}):
            for size_k_first in (False, True):
                method, layer, reference_weight = self._prepare(
                    5120, 288, 32, exact=True, bias=True, size_k_first=size_k_first
                )
                self.assertEqual(layer.weight.shape[0] * 16, 320)
                self.assertEqual(tuple(layer.weight_scale.shape), (10, 5120))
                neutral = fp8_fused_exponent_bias_into_scales(
                    torch.ones_like(layer.weight_scale[-1])
                )
                torch.testing.assert_close(
                    layer.weight_scale[-1], neutral, rtol=0, atol=0
                )
                for m in (1, 6, 32, 128):
                    with self.subTest(m=m, size_k_first=size_k_first, input="random"):
                        self._assert_random(method, layer, reference_weight, m, 300 + m)
                    x = torch.zeros(m, 288, dtype=torch.bfloat16, device="cuda")
                    x[:, 287] = 1
                    expected = reference_weight[:, 287].expand(m, -1).to(torch.bfloat16)
                    for with_bias in (False, True):
                        with self.subTest(
                            m=m,
                            size_k_first=size_k_first,
                            input="last-K single-hot",
                            with_bias=with_bias,
                        ):
                            actual = method.apply(
                                layer, x, bias=layer.bias if with_bias else None
                            )
                            target = expected + layer.bias if with_bias else expected
                            torch.cuda.synchronize()
                            self.assertEqual(tuple(actual.shape), (m, 5120))
                            torch.testing.assert_close(actual, target, rtol=0, atol=0)

    def test_large_synthetic_shapes(self):
        # Portable synthetic shapes only: do not assume a /model checkpoint or
        # claim these are read from an actual model config. Keep just two shapes.
        for n, k in ((5120, 3072), (4096, 1536)):
            method, layer, reference_weight = self._prepare(n, k, 32)
            for m in (1, 128):
                with self.subTest(n=n, k=k, m=m):
                    self._assert_random(method, layer, reference_weight, m, 200 + m)


if __name__ == "__main__":
    unittest.main()
