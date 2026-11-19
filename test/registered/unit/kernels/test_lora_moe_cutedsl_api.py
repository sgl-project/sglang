"""CPU contracts for grouped-GEMM kernel selection and block-scaled operands."""

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
    ("blockscaled", 10): (
        "kernel_sm100_blockscaled",
        "GroupedGemmKernelSm100BlockScaled",
    ),
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


def _case(api, *, contiguous, family, fused, experts=2, n=256):
    rows, k = 128, 256
    packed = family != "mxfp8"
    planes = 1 if contiguous else experts
    lead = (rows,) if contiguous else (experts, rows)
    a = torch.empty(
        (*lead, k // (2 if packed else 1)),
        dtype=torch.uint8 if packed else torch.float8_e4m3fn,
    )
    vec = 16 if family.startswith("nvfp4") else 32
    epilogue = dict(
        alpha=torch.ones((experts, 2)) if vec == 16 else None,
        tok_scale=torch.ones((planes, rows)) if family == "nvfp4_token" else None,
        lora_w=None,
        lora_planes=None,
        lora_slots=None,
    )
    if fused:
        epilogue.update(
            lora_w=torch.empty((2 * experts, n, 8), dtype=torch.bfloat16),
            lora_planes=torch.empty((2 * planes, rows, 16), dtype=torch.bfloat16),
            lora_slots=torch.tensor([2, 0, 1], dtype=torch.int32),
        )
    groups = torch.zeros(experts + int(contiguous), dtype=torch.int32)
    # Masked slabs use live row counts; contiguous segments also need offsets.
    group_rows = torch.zeros(experts, dtype=torch.int32) if contiguous else groups
    operands = (
        a,
        torch.empty((experts, n, a.shape[-1]), dtype=a.dtype),
        torch.empty(api.scale_atom_bytes(rows, k, vec, planes), dtype=torch.uint8),
        torch.empty(api.scale_atom_bytes(n, k, vec, experts), dtype=torch.uint8),
        torch.empty((*lead, n), dtype=torch.bfloat16),
        groups,
        *((group_rows,) if contiguous else ()),
    )
    return SimpleNamespace(
        operands=operands,
        epilogue=epilogue,
        vec=vec,
        planes=planes,
        k=k,
        group_rows=group_rows,
    )


class TestGroupedGemmKernelSelection(CustomTestCase):
    def test_selects_from_operand_device_instead_of_current_device(self):
        """Kernel selection asks the operand's device, not the current one."""
        device = torch.device("cuda:1")
        with _api_module() as (api, classes):
            for family, major in (*_KERNELS, ("blockscaled", 9), ("blockscaled", 12)):
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


class TestBlockScaledPreparation(CustomTestCase):
    def test_layouts_preserve_operand_identity_and_scale_contracts(self):
        """Flat rows add a view dimension, not expert copies or scale planes."""
        experts, rows, n = 2, 128, 256
        schedule = torch.zeros(4, dtype=torch.int64)
        tiles = torch.zeros(4, dtype=torch.int32)
        with _recorded_compiles() as (api, select):
            config = api.GroupedGemmConfig(mma_tiler_mn=(128, 64))
            for contiguous, family, fused in (
                (False, "mxfp8", True),
                (True, "mxfp4", False),
                (True, "nvfp4_static", True),
                (False, "nvfp4_token", True),
            ):
                with self.subTest(contiguous=contiguous, family=family, fused=fused):
                    case = _case(api, contiguous=contiguous, family=family, fused=fused)
                    prepare = (
                        api.prepare_contiguous_blockscaled
                        if contiguous
                        else api.prepare_masked_blockscaled
                    )
                    result = prepare(
                        *case.operands,
                        config=config,
                        sf_vec_size=case.vec,
                        alpha_rows_per_half=128,
                        direct_schedule=schedule,
                        schedule_tiles=tiles,
                        **case.epilogue,
                    )
                    a = case.operands[0]
                    select.assert_called_with("blockscaled", a.device)
                    self.assertEqual(result.problem_shape, (n, rows, case.k, experts))
                    self.assertEqual(result.contiguous_segments, contiguous)
                    self.assertIs(result.group_m_arg.tensor, case.operands[5])
                    self.assertIs(result.direct_schedule, schedule)
                    self.assertIs(result.schedule_tiles, tiles)

                    views = result.operand_args
                    self.assertEqual(
                        [view.tensor.data_ptr() for view in views[:5]],
                        [tensor.data_ptr() for tensor in case.operands[:5]],
                    )
                    self.assertEqual(
                        tuple(views[0].tensor.shape), (case.planes, rows, a.shape[-1])
                    )
                    self.assertEqual(
                        tuple(views[4].tensor.shape), (case.planes, rows, n)
                    )
                    self.assertEqual(
                        tuple(view.leading_dim for view in views),
                        (2, 2, 0, 0, 2, 1, 1, 2, 2, 0, 0),
                    )
                    self.assertIs(views[10].tensor, case.group_rows)
                    for view, owner, given in zip(
                        views[5:10], result.owners, case.epilogue.values(), strict=True
                    ):
                        self.assertIs(view.tensor, owner)
                        if given is not None:
                            self.assertIs(owner, given)

                    alpha = case.epilogue["alpha"] is not None
                    token = case.epilogue["tok_scale"] is not None
                    ab_dtype, sf_dtype, *scaling = result.wrapper_constexprs
                    self.assertEqual(
                        ab_dtype.__name__,
                        "Float8E4M3FN" if family == "mxfp8" else "Float4E2M1FN",
                    )
                    self.assertEqual(
                        sf_dtype.__name__,
                        "Float8E4M3FN" if alpha else "Float8E8M0FNU",
                    )
                    self.assertIs(result.ab_dtype, ab_dtype)
                    self.assertEqual(scaling, [alpha, 128, token, experts])
                    self.assertEqual(
                        result.kernel_kwargs,
                        dict(
                            sf_vec_size=case.vec,
                            sf_dtype=sf_dtype,
                            lora_k=16 if fused else 0,
                            lora_acc="separate" if fused and alpha else "shared",
                        ),
                    )

    def test_alpha_split_must_align_with_output_tiles(self):
        """Reject alpha that changes within a tile or lacks a column."""
        with _recorded_compiles() as (api, select):
            case = _case(
                api,
                contiguous=True,
                family="nvfp4_static",
                fused=False,
                experts=1,
                n=384,
            )
            # At 192 the alpha boundary splits a tile; at 128 it needs three columns.
            for rows_per_half in (192, 128):
                with (
                    self.subTest(rows_per_half=rows_per_half),
                    self.assertRaisesRegex(ValueError, "alpha changes every"),
                ):
                    api.prepare_contiguous_blockscaled(
                        *case.operands,
                        config=api.GroupedGemmConfig(mma_tiler_mn=(128, 128)),
                        sf_vec_size=case.vec,
                        alpha=case.epilogue["alpha"],
                        alpha_rows_per_half=rows_per_half,
                    )
            select.assert_not_called()
            api._compile_prepared.assert_not_called()

    def test_epilogue_lora_with_token_scales_needs_alpha(self):
        """Alpha selects a separate accumulator so token scales leave LoRA alone."""
        with _recorded_compiles() as (api, _):
            case = _case(api, contiguous=False, family="nvfp4_token", fused=True)
            with self.assertRaisesRegex(ValueError, "need alpha"):
                api.prepare_masked_blockscaled(
                    *case.operands,
                    config=api.GroupedGemmConfig(),
                    sf_vec_size=case.vec,
                    **dict(case.epilogue, alpha=None),
                )


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
