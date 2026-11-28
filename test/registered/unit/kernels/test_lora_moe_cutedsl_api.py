"""CPU contracts for grouped-GEMM kernel selection and block-scaled operands."""

import ast
import importlib.util
import inspect
import sys
import types
import unittest
from contextlib import ExitStack, contextmanager
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
                    # Only the contiguous entry point can drop the token-scale check.
                    full_tiles = {"tok_scale_full_tiles": True} if contiguous else {}
                    result = prepare(
                        *case.operands,
                        config=config,
                        sf_vec_size=case.vec,
                        alpha_rows_per_half=128,
                        direct_schedule=schedule,
                        schedule_tiles=tiles,
                        **case.epilogue,
                        **full_tiles,
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
                            tok_scale_full_tiles=contiguous,
                        ),
                    )

    def test_alpha_defaults_to_one_scale_per_expert_and_checks_partitions(self):
        with _recorded_compiles() as (api, select):
            case = _case(
                api,
                contiguous=True,
                family="nvfp4_static",
                fused=False,
                experts=1,
                n=384,
            )
            config = api.GroupedGemmConfig(mma_tiler_mn=(128, 128))
            # At 192 the alpha boundary splits a tile; at 128 it needs three columns.
            for rows_per_half in (192, 128):
                with (
                    self.subTest(rows_per_half=rows_per_half),
                    self.assertRaisesRegex(ValueError, "alpha changes every"),
                ):
                    api.prepare_contiguous_blockscaled(
                        *case.operands,
                        config=config,
                        sf_vec_size=case.vec,
                        alpha=case.epilogue["alpha"],
                        alpha_rows_per_half=rows_per_half,
                    )
            select.assert_not_called()
            api._compile_prepared.assert_not_called()
            result = api.prepare_contiguous_blockscaled(
                *case.operands,
                config=config,
                sf_vec_size=case.vec,
                alpha=torch.ones((1, 1)),
            )
            self.assertEqual(result.wrapper_constexprs[2:4], (True, 384))

    def test_scale_atom_bytes_round_partial_groups_up(self):
        with _api_module() as (api, _):
            self.assertEqual(api.scale_atom_bytes(128, 128, 32), 512)
            self.assertEqual(api.scale_atom_bytes(128, 144, 32), 1024)

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


class TestBlockScaledQuantArgs(CustomTestCase):
    def test_quantized_launch_arguments_match_kernel_signatures(self):
        from sglang.kernels.ops.lora.moe.quant_blockscaled import (
            BlockScaledRows,
            act_quant_args,
            quantize_rows,
        )

        codes = torch.empty((128, 64), dtype=torch.uint8)
        storage = dict(
            data=codes,
            scales=torch.empty(1024, dtype=torch.uint8),
            sf_vec_size=16,
            rows_per_slab=128,
        )
        rows = BlockScaledRows(
            **storage,
            row_scales=torch.ones(128),
            row_denominator=1536.0,
            use_4over6=True,
            err_mode="MSE",
        )
        static_rows = BlockScaledRows(**storage, global_scale=0.25)
        # An empty row map returns before any launch.
        quantize_rows(
            torch.empty(128, 128, dtype=torch.bfloat16),
            rows,
            torch.zeros(0, dtype=torch.int32),
        )
        quant_args = rows.kernel_args()
        consumers = (
            ("quant_blockscaled", "_quant_rows_kernel", (quant_args,)),
            (
                "dispatch_contiguous",
                "_fill_rows_contiguous_blockscaled_kernel",
                (quant_args,),
            ),
            (
                "dispatch_masked",
                "_dispatch_fill_masked_blockscaled_kernel",
                (quant_args,),
            ),
            (
                "activation_delta",
                "_act_delta_kernel",
                tuple(
                    act_quant_args(target, fallback=codes) for target in (rows, None)
                ),
            ),
            (
                "fused_act",
                "_b_act_kernel",
                tuple(act_quant_args(target) for target in (static_rows, None)),
            ),
        )
        sources = _API.parents[1]
        launch_options = {"num_warps", "num_stages", "num_ctas", "maxnreg"}
        for module, kernel, variants in consumers:
            tree = ast.parse((sources / f"{module}.py").read_text())
            signature = next(
                node.args
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == kernel
            )
            positional = [arg.arg for arg in (*signature.posonlyargs, *signature.args)]
            allowed = set(positional) | {arg.arg for arg in signature.kwonlyargs}
            required = set(positional[: len(positional) - len(signature.defaults)])
            required.update(
                arg.arg
                for arg, default in zip(signature.kwonlyargs, signature.kw_defaults)
                if default is None
            )
            (launch,) = (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Subscript)
                and getattr(node.func.value, "id", None) == kernel
            )
            self.assertEqual(sum(arg.arg is None for arg in launch.keywords), 1, kernel)
            explicit = positional[: len(launch.args)] + [
                arg.arg
                for arg in launch.keywords
                if arg.arg and arg.arg not in launch_options
            ]
            for quant in variants:
                with self.subTest(kernel=kernel, keys=set(quant)):
                    supplied = explicit + list(quant)
                    # Optional parameters may be absent; each supplied key occurs once.
                    self.assertCountEqual(
                        supplied, required | (allowed & set(supplied))
                    )

    def test_unsafe_arguments_are_refused_before_any_launch(self):
        from msgspec.structs import replace

        from sglang.kernels.ops.lora.common import lora_a
        from sglang.kernels.ops.lora.moe import (
            dispatch_contiguous,
            fused_act,
            quant_blockscaled,
        )

        # One row scale for two data rows: the row-scale prepass would write
        # past it, so the target must be refused before anything launches.
        rows = quant_blockscaled.BlockScaledRows(
            data=torch.empty((2, 64), dtype=torch.uint8),
            scales=torch.empty(1024, dtype=torch.uint8),
            sf_vec_size=16,
            rows_per_slab=2,
            row_scales=torch.ones(1),
            row_denominator=1536.0,
        )
        per_token = replace(rows, row_scales=torch.ones(2))
        x = torch.empty(2, 128, dtype=torch.bfloat16)
        pair_to_row = torch.arange(2, dtype=torch.int32)
        routing = SimpleNamespace(num_rows=2)
        # fused_b_act reads its other arguments only after the check.
        b_act_args = dict.fromkeys(inspect.signature(fused_act.fused_b_act).parameters)
        b_act_args.update(activation="silu", routing=routing, act_quant=per_token)
        kernels = (
            (quant_blockscaled, "_row_scale_kernel"),
            (quant_blockscaled, "_quant_rows_kernel"),
            (dispatch_contiguous, "_token_scale_kernel"),
            (dispatch_contiguous, "_fill_rows_contiguous_blockscaled_kernel"),
            (lora_a, "_grouped_lora_a_kernel"),
            (fused_act, "_b_act_kernel"),
        )
        with ExitStack() as patches:
            launches = [patches.enter_context(mock.patch.object(*k)) for k in kernels]
            with self.assertRaises(AssertionError):
                quant_blockscaled.quantize_rows(x, rows, pair_to_row)
            for rows_out, token_scales, error, message in (
                (rows, torch.empty(2), AssertionError, ""),
                (per_token, torch.empty(1), ValueError, "token_scales"),
                (per_token, torch.empty(1).expand(2), ValueError, "token_scales"),
            ):
                with self.assertRaisesRegex(error, message):
                    dispatch_contiguous.dispatch_fill_rows_contiguous_blockscaled(
                        x,
                        pair_to_row.view(2, 1),
                        pair_to_row,
                        rows_out=rows_out,
                        token_scales=token_scales,
                    )
            # place_rows stores at mapped rows, up to the last input row.
            for row_map, output_rows in ((None, 2), (pair_to_row, 1)):
                with self.assertRaisesRegex(ValueError, "place_rows"):
                    lora_a.grouped_lora_a(
                        x,
                        None,
                        torch.empty(output_rows, 16),
                        routing,
                        config={},
                        pair_to_row=row_map,
                        place_rows=True,
                    )
            with self.assertRaisesRegex(ValueError, "per-token"):
                fused_act.fused_b_act(**b_act_args)
        for launch in launches:
            launch.__getitem__.assert_not_called()


class TestCompileCache(CustomTestCase):
    def test_stage_constants_separate_compiled_code(self):
        from sglang.srt.lora.moe.base_gemm_provider import cutedsl_common

        weight = torch.zeros((2, 256, 128), dtype=torch.uint8)
        build = mock.Mock(
            side_effect=lambda *_: SimpleNamespace(
                launch=mock.Mock(), compiled_fn=object()
            )
        )
        provider = SimpleNamespace(
            _ROW_MODE="masked",
            _DTYPE_TAG="nvfp4",
            quant_info=SimpleNamespace(num_local_experts=2),
            _compiled={64: {}},
            _stage_weight=lambda _: weight,
            _stage_scale=lambda _: None,
            _stage_constexprs=lambda stage: (
                16,
                True,
                128 if stage == "gemm1" else 1 << 30,
            ),
            _prepare_dummy=build,
            _as_dynamic_cute_tensor=lambda tensor, **_: tensor,
        )
        with mock.patch.dict(cutedsl_common._COMPILE_CACHE, clear=True):
            for stage in ("gemm1", "gemm2", "gemm1"):
                cutedsl_common.CuteDslTileMixin._compile_stage(
                    provider, 64, stage, "tile"
                )
        calls = provider._compiled[64]
        self.assertIsNot(calls["gemm1"].compiled_fn, calls["gemm2"].compiled_fn)
        self.assertEqual(build.call_count, 2)


if __name__ == "__main__":
    unittest.main()
