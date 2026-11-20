"""CPU contracts for grouped-GEMM kernel selection and FP8 prepare arguments."""

import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="weekly", runner_config="cpu")

_API = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/kernels/ops/lora/moe/cutedsl/api.py"
)
# The kernel module and class each (family, SM major) must select.
_KERNELS = {
    ("bf16", 9): ("kernel_sm90_bf16", "GroupedGemmKernelSm90Bf16"),
    ("bf16", 10): ("kernel_sm100_bf16", "GroupedGemmKernelSm100Bf16"),
    ("fp8", 9): ("kernel_sm90_fp8", "GroupedGemmKernelSm90Fp8"),
    ("fp8", 10): ("kernel_sm100_fp8", "GroupedGemmKernelSm100Fp8"),
    ("fp8_rowwise", 9): ("kernel_sm90_fp8", "GroupedGemmKernelSm90Fp8Rowwise"),
    ("fp8_rowwise", 10): ("kernel_sm100_fp8", "GroupedGemmKernelSm100Fp8Rowwise"),
}


@contextmanager
def _api_module():
    names = (
        "cuda",
        "cuda.bindings",
        "cuda.bindings.driver",
        "cutlass",
        "cutlass.cute",
        "cutlass.cute.testing",
        "cutlass.cute.runtime",
        "cutlass.utils",
    )
    modules = {name: types.ModuleType(name) for name in names}
    modules["cuda.bindings.driver"].CUstream = int
    modules["cutlass"].Constexpr = object
    for name, width in (("Float8E4M3FN", 8), ("Float8E8M0FNU", 8), ("Float4E2M1FN", 4)):
        setattr(modules["cutlass"], name, type(name, (), {"width": width}))
    modules["cutlass.cute"].Tensor = object
    modules["cutlass.cute"].jit = lambda function: function

    def from_dlpack(tensor, *, assumed_align):
        class View:
            def mark_layout_dynamic(self, *, leading_dim):
                return SimpleNamespace(tensor=tensor, leading_dim=leading_dim)

        return View()

    modules["cutlass.cute.runtime"].from_dlpack = from_dlpack
    classes = {}
    for key, (module, name) in _KERNELS.items():
        # A module may hold several kernel classes (the FP8 scale layouts).
        qualified = f"sglang.kernels.ops.lora.moe.cutedsl.{module}"
        stub = modules.setdefault(qualified, types.ModuleType(qualified))
        classes[key] = type(name, (), {})
        setattr(stub, name, classes[key])
    spec = importlib.util.spec_from_file_location("_cutedsl_api_contract", _API)
    api = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(api)
        yield api, classes


@contextmanager
def _recorded_compiles():
    with (
        _api_module() as (api, _),
        mock.patch.object(api, "_kernel_class_for", return_value=object) as select,
        mock.patch.object(
            api,
            "_compile_prepared",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    ):
        yield api, select


class TestGroupedGemmKernelSelection(CustomTestCase):
    def test_selects_from_operand_device_instead_of_current_device(self):
        """Kernel selection asks the operand's device, not the current one."""
        device = torch.device("cuda:1")
        with _api_module() as (api, classes):
            for family, major in _KERNELS:
                capability = (major, 3 if (family, major) == ("fp8_rowwise", 10) else 0)
                with (
                    self.subTest(family=family, capability=capability),
                    mock.patch.object(
                        torch.cuda,
                        "get_device_capability",
                        side_effect=lambda target=None, capability=capability: (
                            capability if target == device else (12, 0)
                        ),
                    ),
                ):
                    expected = classes.get((family, major))
                    if expected is None:
                        with self.assertRaises(NotImplementedError):
                            api._kernel_class_for(family, device)
                    else:
                        self.assertIs(api._kernel_class_for(family, device), expected)


# Each FP8 prepare: its fixed (rowwise, contiguous) mode and group keyword.
_FP8_PREPARES = {
    "prepare_masked_fp8": (False, False, "masked_m"),
    "prepare_contiguous_fp8": (False, True, "seg_offsets"),
    "prepare_masked_fp8_rowwise": (True, False, "masked_m"),
    "prepare_contiguous_fp8_rowwise": (True, True, "seg_offsets"),
}


class TestFp8Prepares(CustomTestCase):
    def test_calls_forward_operands_options_and_their_fixed_mode(self):
        inputs = dict(
            a=mock.sentinel.a,
            b=mock.sentinel.b,
            sf_tokens=mock.sentinel.sf_tokens,
            sf_weights=mock.sentinel.sf_weights,
            c=mock.sentinel.c,
            group=mock.sentinel.group,
            config=mock.sentinel.config,
            direct_schedule=mock.sentinel.direct_schedule,
            schedule_tiles=mock.sentinel.schedule_tiles,
        )
        with _api_module() as (api, _):
            for name, (rowwise, contiguous, group) in _FP8_PREPARES.items():
                kwargs = dict(inputs)
                kwargs[group] = kwargs.pop("group")
                prepare = getattr(api, name)
                with (
                    self.subTest(name=name),
                    mock.patch.object(api, "_prepare_fp8", autospec=True) as shared,
                ):
                    self.assertIs(prepare(**kwargs), shared.return_value)
                    shared.assert_called_once_with(
                        **inputs, rowwise=rowwise, contiguous=contiguous
                    )
                    for override in ("rowwise", "contiguous"):
                        with self.assertRaises(TypeError, msg=override):
                            prepare(**kwargs, **{override: True})

        with _recorded_compiles() as (api, select):
            weight = torch.empty((2, 128, 128), dtype=torch.float8_e4m3fn)
            options = dict(
                config=api.GroupedGemmConfig(mma_tiler_mn=(128, 64)),
                direct_schedule=torch.zeros(8, dtype=torch.int64),
                schedule_tiles=torch.zeros(4, dtype=torch.int32),
            )
            for rowwise, scale_shapes, axis in (
                (False, ((2, 128, 1), (2, 1, 1)), 2),
                (True, ((1, 128), (2, 128)), 1),
            ):
                a = weight[0] if rowwise else weight
                scales = tuple(torch.ones(shape) for shape in scale_shapes)
                groups = torch.zeros(2 + int(rowwise), dtype=torch.int32)
                result = api._prepare_fp8(
                    a,
                    weight,
                    *scales,
                    torch.empty_like(a, dtype=torch.bfloat16),
                    groups,
                    rowwise=rowwise,
                    contiguous=rowwise,
                    **options,
                )
                self.assertIs(result.group_m_arg.tensor, groups)
                for key, value in options.items():
                    self.assertIs(getattr(result, key), value)
                select.assert_called_with("fp8_rowwise" if rowwise else "fp8", a.device)
                self.assertIs(result.wrapper, api._grouped_gemm_fp8_swap_ab)
                self.assertEqual(result.contiguous_segments, rowwise)
                for view, scale in zip(result.operand_args[2:4], scales):
                    self.assertIs(view.tensor, scale)
                    self.assertEqual(view.leading_dim, axis)


if __name__ == "__main__":
    unittest.main()
