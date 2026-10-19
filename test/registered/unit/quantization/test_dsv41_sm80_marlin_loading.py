"""CPU-only tests of the actual SM80 Marlin allocation/loading/preparation code.

AST isolation removes imports, not implementation: no SGLang runtime, CUDA JIT,
model, or server is loaded. Repack/scale-permutation kernels are replaced with
identity copies so the tensors entering those kernels can be inspected.
"""

from __future__ import annotations

import ast
import logging
import os
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[4]
# The marker is a runtime no-op. Load it without importing sglang.__init__,
# which otherwise pulls in the public API and optional runtime dependencies.
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=5, suite="base-a-test-cpu")
QUANT = ROOT / "python/sglang/srt/layers/quantization"
MOE = ROOT / "python/sglang/srt/layers/moe/fused_moe_triton/layer.py"
ENV = "SGLANG_DSV41_SM80_EXPERIMENTAL"


class _RemoveImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


def _load_definitions(path, names, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name is not None:
        nodes = next(
            node.body
            for node in nodes
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    selected = [
        node
        for node in nodes
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in selected} != set(names):
        raise AssertionError(f"Missing definitions in {path}: {names}")
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + [_RemoveImports().visit(node) for node in selected],
        type_ignores=[],
    )
    # Only repository-owned definitions selected above are executed.
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)  # noqa: S102


def _set_attrs(tensor, attrs):
    for name, value in attrs.items():
        setattr(tensor, name, value)


def _isolated_code(capability=(8, 0)):
    namespace = {
        "torch": torch,
        "envs": runpy.run_path(str(ROOT / "python/sglang/srt/environ.py"))["envs"],
        "logger": logging.getLogger(__name__),
        "get_device_capability": Mock(return_value=capability),
        "get_platform": lambda: SimpleNamespace(
            is_sm90=capability == (9, 0), is_sm120=capability == (12, 0)
        ),
        "round_up": lambda value, multiple: (
            (value + multiple - 1) // multiple * multiple
        ),
        "set_weight_attrs": _set_attrs,
        "FusedMoeWeightScaleSupported": SimpleNamespace(
            BLOCK=SimpleNamespace(value="block")
        ),
        "log_info_on_rank0": Mock(),
        "deinterleave_moe_mxfp4_w13_for_marlin": Mock(),
        "marlin_make_workspace": Mock(return_value=torch.zeros(1, dtype=torch.int32)),
        "_repack_moe_fp4_weight_for_marlin": lambda tensor, **kwargs: tensor.clone(),
        "_permute_moe_fp4_scales_for_marlin": lambda tensor, **kwargs: tensor.clone(),
        "_maybe_copy_weight_view_before_h2d": lambda tensor: tensor,
        "_is_cpu": False,
    }
    _load_definitions(
        QUANT / "mxfp4_marlin_moe.py", ["_is_experimental_sm80_marlin"], namespace
    )
    _load_definitions(
        QUANT / "mxfp4_marlin_moe.py",
        ["create_weights", "process_weights_after_loading"],
        namespace,
        "Mxfp4MarlinMoEMethod",
    )
    _load_definitions(
        QUANT / "marlin_utils.py", ["check_moe_marlin_supports_layer"], namespace
    )
    _load_definitions(
        QUANT / "marlin_utils_fp4.py",
        [
            "_normalize_scale_tensor",
            "_get_optional_param",
            "prepare_moe_mxfp4_layer_for_marlin",
        ],
        namespace,
    )
    _load_definitions(MOE, ["_load_w13", "_load_w2"], namespace, "FusedMoE")
    return namespace


def _layer(hidden_size=5120):
    layer = torch.nn.Module()
    layer.hidden_size = hidden_size
    layer.intermediate_size_per_partition = 288
    layer.moe_runner_config = SimpleNamespace(
        is_gated=True, activation="silu", apply_router_weight_on_input=False
    )
    return layer


def _method():
    return SimpleNamespace(
        _fp8=SimpleNamespace(process_weights_after_loading=Mock()),
        prefix="test.experts",
        runner=SimpleNamespace(config=SimpleNamespace(gemm1_alpha=None)),
    )


def _allocate(code, dtype=torch.bfloat16, hidden_size=5120):
    layer, method = _layer(hidden_size), _method()
    code["create_weights"](method, layer, 2, hidden_size, 288, dtype, loader_tag="test")
    return layer, method


class TestDsv41Sm80MarlinLoading(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {ENV: "1"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.code = _isolated_code()

    def assertZero(self, tensor):
        self.assertEqual(torch.count_nonzero(tensor).item(), 0)

    def test_logical_shapes_neutral_scales_and_initialized_weights(self):
        layer, _ = _allocate(self.code)
        self.assertEqual(layer.w13_weight.shape, (2, 576, 2560))
        self.assertEqual(layer.w2_weight.shape, (2, 5120, 144))
        self.assertEqual(layer.w13_weight_scale_inv.shape, (2, 576, 160))
        self.assertEqual(layer.w2_weight_scale_inv.shape, (2, 5120, 9))
        self.assertEqual(layer.orig_dtype, torch.bfloat16)
        for name in ("w13_weight", "w2_weight"):
            tensor = getattr(layer, name)
            self.assertZero(tensor)
            self.assertFalse(tensor.requires_grad)
            self.assertEqual(tensor.loader_tag, "test")
        for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
            scale = getattr(layer, name)
            self.assertEqual(scale.dtype, torch.float8_e8m0fnu)
            self.assertTrue(torch.all(scale.view(torch.uint8) == 127))
            self.assertEqual(scale.quant_method, "block")
            self.assertFalse(scale.format_ue8m0)

    def test_global_and_presharded_tp8_gate_up_and_down_loading(self):
        # Exercise actual GPU loader branches on CPU tensors. In global mode
        # each checkpoint half has 2304 rows; rank 7 must load rows 2016:2304.
        for presharded in (False, True):
            with self.subTest(presharded=presharded):
                layer, method = _allocate(self.code)
                loader = SimpleNamespace(
                    quant_method=method,
                    quant_config=None,
                    moe_runner_config=layer.moe_runner_config,
                    moe_tp_size=8,
                    moe_tp_rank=7,
                    use_presharded_weights=presharded,
                    use_padded_loading=False,
                    use_triton_kernels=False,
                )
                ranks = [7] if presharded else range(8)
                for expert in range(2):
                    for shard, base in (("w1", 10), ("w3", 30)):
                        source = torch.cat(
                            [
                                torch.full(
                                    (288, 2560), base + rank + expert, dtype=torch.int8
                                )
                                for rank in ranks
                            ]
                        )
                        self.code["_load_w13"](
                            loader, layer.w13_weight.data[expert], 0, shard, source
                        )
                    source = torch.cat(
                        [
                            torch.full(
                                (5120, 144), 50 + rank + expert, dtype=torch.int8
                            )
                            for rank in ranks
                        ],
                        dim=1,
                    )
                    self.code["_load_w2"](
                        loader, layer.w2_weight.data[expert], 1, "w2", source
                    )
                    self.assertTrue(
                        torch.all(layer.w13_weight[expert, :288] == 17 + expert)
                    )
                    self.assertTrue(
                        torch.all(layer.w13_weight[expert, 288:] == 37 + expert)
                    )
                    self.assertTrue(torch.all(layer.w2_weight[expert] == 57 + expert))
                    # Run the same real group-scale loader with raw E8M0
                    # containers; gate/up scales must share the weight boundary.
                    for shard, base in (("w1", 90), ("w3", 110)):
                        source = torch.cat(
                            [
                                torch.full(
                                    (288, 160), base + rank + expert, dtype=torch.uint8
                                )
                                for rank in ranks
                            ]
                        ).view(torch.float8_e8m0fnu)
                        self.code["_load_w13"](
                            loader,
                            layer.w13_weight_scale_inv.data[expert],
                            0,
                            shard,
                            source,
                        )
                    source = torch.cat(
                        [
                            torch.full(
                                (5120, 9), 130 + rank + expert, dtype=torch.uint8
                            )
                            for rank in ranks
                        ],
                        dim=1,
                    ).view(torch.float8_e8m0fnu)
                    self.code["_load_w2"](
                        loader, layer.w2_weight_scale_inv.data[expert], 1, "w2", source
                    )
                    raw = layer.w13_weight_scale_inv[expert].view(torch.uint8)
                    self.assertTrue(torch.all(raw[:288] == 97 + expert))
                    self.assertTrue(torch.all(raw[288:] == 117 + expert))
                    self.assertTrue(
                        torch.all(
                            layer.w2_weight_scale_inv[expert].view(torch.uint8)
                            == 137 + expert
                        )
                    )

    def test_partial_load_leaves_each_half_and_down_tail_zero(self):
        layer, method = _allocate(self.code)
        loader = SimpleNamespace(
            quant_method=method,
            quant_config=None,
            moe_runner_config=layer.moe_runner_config,
            use_presharded_weights=True,
            use_padded_loading=False,
            use_triton_kernels=False,
        )
        for shard, start in (("w1", 0), ("w3", 288)):
            self.code["_load_w13"](
                loader,
                layer.w13_weight.data[0],
                0,
                shard,
                torch.full((256, 2560), 17, dtype=torch.int8),
            )
            self.assertTrue(torch.all(layer.w13_weight[0, start : start + 256] == 17))
            self.assertZero(layer.w13_weight[0, start + 256 : start + 288])
        self.code["_load_w2"](
            loader,
            layer.w2_weight.data[0],
            1,
            "w2",
            torch.full((5120, 128), 17, dtype=torch.int8),
        )
        self.assertZero(layer.w2_weight[0, :, 128:])
        self.assertZero(layer.w13_weight[1])
        self.assertZero(layer.w2_weight[1])
        self.assertTrue(torch.all(layer.w13_weight_scale_inv.view(torch.uint8) == 127))

    def test_process_uses_bf16_and_halves_aware_288_to_320_padding(self):
        layer, method = _allocate(self.code)
        layer.w13_weight.data[:, :288].fill_(17)
        layer.w13_weight.data[:, 288:].fill_(34)
        layer.w2_weight.data.fill_(51)
        # Finite normal E8M0 values outside FP16 range; not a claim about
        # E8M0 special values or the GPU kernel's complete exponent domain.
        layer.w13_weight_scale_inv.data.view(torch.uint8)[:, :288].fill_(7)
        layer.w13_weight_scale_inv.data.view(torch.uint8)[:, 288:].fill_(247)
        self.code["process_weights_after_loading"](method, layer)
        method._fp8.process_weights_after_loading.assert_called_once_with(layer)
        self.assertEqual(layer._dsv4_mxfp4_backend, "marlin")
        self.assertEqual(layer.w13_weight.shape, (2, 640, 2560))
        self.assertEqual(layer.w2_weight.shape, (2, 5120, 160))
        self.assertTrue(torch.all(layer.w13_weight[:, :288] == 17))
        self.assertZero(layer.w13_weight[:, 288:320])
        self.assertTrue(torch.all(layer.w13_weight[:, 320:608] == 34))
        self.assertZero(layer.w13_weight[:, 608:640])
        self.assertZero(layer.w2_weight[:, :, 144:])
        scales = layer.w13_weight_scale
        self.assertEqual(scales.dtype, torch.bfloat16)
        self.assertTrue(torch.all(torch.isfinite(scales)))
        self.assertTrue(torch.all(scales[:, :288] == 2.0**-120))
        self.assertTrue(torch.all(scales[:, 320:608] == 2.0**120))
        self.assertZero(scales[:, 288:320])
        self.assertZero(scales[:, 608:640])
        self.assertZero(layer.w2_weight_scale[:, :, 9:])
        self.assertFalse(hasattr(layer, "w13_weight_scale_inv"))
        self.code["deinterleave_moe_mxfp4_w13_for_marlin"].assert_not_called()

    def test_hidden_padding_is_initialized(self):
        layer, method = _allocate(self.code, hidden_size=5056)
        self.assertEqual(method.hidden_pad, 64)
        self.assertZero(layer.w13_weight[:, :, 2528:])
        self.assertZero(layer.w2_weight[:, 5056:])

    def test_non_bf16_allocation_and_preparation_are_rejected(self):
        for dtype in (torch.float16, torch.float32):
            with (
                self.subTest(dtype=dtype),
                self.assertRaisesRegex(RuntimeError, "BF16 params_dtype"),
            ):
                _allocate(self.code, dtype)
        for dtype in (None, torch.float16, torch.float32):
            with self.subTest(orig_dtype=dtype):
                layer, method = _allocate(self.code)
                if dtype is None:
                    del layer.orig_dtype
                else:
                    layer.orig_dtype = dtype
                with self.assertRaisesRegex(RuntimeError, "BF16 orig_dtype"):
                    self.code["process_weights_after_loading"](method, layer)

    def test_opt_in_is_exact_and_sm80_only(self):
        for value in (None, "0", "true", "01", "1"):
            for capability in (
                (8, 0),
                (8, 6),
                (8, 9),
                (9, 0),
                (10, 0),
                (12, 0),
                (None, None),
            ):
                with self.subTest(value=value, capability=capability):
                    if value is None:
                        os.environ.pop(ENV, None)
                    else:
                        os.environ[ENV] = value
                    code = _isolated_code(capability)
                    self.assertEqual(
                        code["_is_experimental_sm80_marlin"](),
                        value == "1" and capability == (8, 0),
                    )

    def test_existing_architecture_policy_is_unchanged(self):
        for value in (None, "1"):
            for capability in ((8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)):
                with self.subTest(value=value, capability=capability):
                    if value is None:
                        os.environ.pop(ENV, None)
                    else:
                        os.environ[ENV] = value
                    code = _isolated_code(capability)
                    layer, method = _allocate(code)
                    expected_i = (
                        288
                        if capability == (9, 0)
                        or (capability == (8, 0) and value == "1")
                        else 384
                    )
                    self.assertEqual(layer.w13_weight.shape[1], 2 * expected_i)
                    allowed = capability in ((9, 0), (12, 0)) or (
                        capability == (8, 0) and value == "1"
                    )
                    if allowed:
                        code["process_weights_after_loading"](method, layer)
                        self.assertEqual(layer._dsv4_mxfp4_backend, "marlin")
                    else:
                        with self.assertRaisesRegex(
                            RuntimeError, "requires SM90 or SM120"
                        ):
                            code["process_weights_after_loading"](method, layer)

    def test_existing_layer_constraints_still_apply(self):
        layer, method = _allocate(self.code)
        layer.moe_runner_config.apply_router_weight_on_input = True
        with self.assertRaisesRegex(RuntimeError, "Marlin constraints"):
            self.code["process_weights_after_loading"](method, layer)


if __name__ == "__main__":
    unittest.main()
