"""CPU checks for DeepGEMM warmup buffers and scale conversion workspace."""

import sys
import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.deep_gemm_wrapper import compile_utils, configurer, entrypoint
from sglang.srt.layers.deep_gemm_wrapper.compile_utils import (
    DeepGemmKernelType,
    _BaseWarmupExecutor,
)

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _allocate_on_meta(factory):
    def allocate(*args, **kwargs):
        kwargs["device"] = "meta"
        return factory(*args, **kwargs)

    return allocate


def _storage_bytes(*tensors):
    return sum(tensor.untyped_storage().nbytes() for tensor in tensors)


def _pack_scales_on_meta(scales):
    scales = scales.unsqueeze(0) if scales.ndim == 2 else scales
    groups, rows, cols = scales.shape
    aligned_rows = (rows + 3) // 4 * 4
    packed_cols = (cols + 3) // 4
    initial = torch.empty_strided(
        (groups, rows, packed_cols),
        (aligned_rows * packed_cols, 1, aligned_rows),
        dtype=torch.int32,
        device="meta",
    )
    peak = _storage_bytes(initial)
    if groups > 1 and rows * cols % 4:
        shifted = scales.view(torch.int32) >> 23
        unpacked = shifted.to(torch.uint8)
        peak = max(peak, _storage_bytes(initial, shifted, unpacked))
        padded = torch.zeros(
            (groups, aligned_rows, packed_cols * 4), dtype=torch.uint8, device="meta"
        )
        packed = torch.empty_strided(
            (groups, aligned_rows, packed_cols),
            (aligned_rows * packed_cols, 1, aligned_rows),
            dtype=torch.int32,
            device="meta",
        )
        peak = max(peak, _storage_bytes(initial, unpacked, padded, packed))
        return packed, peak
    return initial, peak


def _scale_workspace_on_meta(executor, architecture):
    if not hasattr(executor, "lhs_s") or architecture == "musa":
        return 0
    scales = executor.lhs_s
    if architecture == "hopper":
        scales = scales.unsqueeze(0) if scales.ndim == 2 else scales
        _, rows, cols = scales.shape
        aligned_rows = (rows + 3) // 4 * 4
        workspace = torch.empty_strided(
            scales.shape,
            (aligned_rows * cols, 1, aligned_rows),
            dtype=scales.dtype,
            device="meta",
        )
        return _storage_bytes(workspace)
    lhs_packed, lhs_peak = _pack_scales_on_meta(scales)
    rows = executor.rhs_q.shape[-2]
    indices = torch.arange(rows, device="meta") // 128
    broadcast = executor.rhs_s.index_select(-2, indices)
    _, rhs_peak = _pack_scales_on_meta(broadcast)
    return max(
        lhs_peak,
        _storage_bytes(lhs_packed, broadcast) + max(_storage_bytes(indices), rhs_peak),
    )


class TestWarmupMemory(CustomTestCase):
    def test_packed_block32_uses_fp8_fp4_entrypoint(self):
        lhs = (
            torch.empty((2, 32), dtype=torch.float8_e4m3fn),
            torch.zeros((2, 1), dtype=torch.int32),
        )
        rhs = (
            torch.empty((8, 32), dtype=torch.float8_e4m3fn),
            torch.zeros((8, 1), dtype=torch.int32),
        )
        out = torch.empty((2, 8), dtype=torch.bfloat16)
        deep_gemm = SimpleNamespace(
            fp8_fp4_gemm_nt=Mock(
                side_effect=lambda lhs, rhs, out, **kwargs: out.fill_(7)
            ),
            fp8_gemm_nt=Mock(),
        )

        with (
            patch.object(entrypoint, "deep_gemm", deep_gemm, create=True),
            patch.object(
                entrypoint.compile_utils,
                "deep_gemm_execution_hook",
                return_value=nullcontext(),
            ) as execution_hook,
        ):
            entrypoint.gemm_nt_f8f8bf16(lhs, rhs, out, recipe=(1, 32, 32))

        execution_hook.assert_called_once_with(
            2,
            8,
            32,
            1,
            DeepGemmKernelType.GEMM_NT_F8F8BF16_BLOCK32,
        )
        deep_gemm.fp8_fp4_gemm_nt.assert_called_once_with(
            lhs,
            rhs,
            out,
            recipe_a=(1, 32),
            recipe_b=(1, 32),
            disable_ue8m0_cast=True,
        )
        deep_gemm.fp8_gemm_nt.assert_not_called()
        torch.testing.assert_close(out, torch.full_like(out, 7), rtol=0, atol=0)

    def test_unpacked_scales_keep_legacy_entrypoint(self):
        for recipe in (None, (1, 32, 32)):
            with self.subTest(recipe=recipe):
                lhs = (
                    torch.empty((2, 32), dtype=torch.float8_e4m3fn),
                    torch.ones((2, 1), dtype=torch.float32),
                )
                rhs = (
                    torch.empty((8, 32), dtype=torch.float8_e4m3fn),
                    torch.ones((1, 1), dtype=torch.float32),
                )
                out = torch.empty((2, 8), dtype=torch.bfloat16)
                deep_gemm = SimpleNamespace(
                    fp8_fp4_gemm_nt=Mock(),
                    fp8_gemm_nt=Mock(
                        side_effect=lambda lhs, rhs, out, **kwargs: out.fill_(3)
                    ),
                )
                with (
                    patch.object(entrypoint, "deep_gemm", deep_gemm, create=True),
                    patch.object(
                        compile_utils,
                        "deep_gemm_execution_hook",
                        return_value=nullcontext(),
                    ),
                ):
                    entrypoint.gemm_nt_f8f8bf16(lhs, rhs, out, recipe=recipe)
                deep_gemm.fp8_gemm_nt.assert_called_once_with(
                    lhs, rhs, out, **({"recipe": recipe} if recipe is not None else {})
                )
                deep_gemm.fp8_fp4_gemm_nt.assert_not_called()
                torch.testing.assert_close(out, torch.full_like(out, 3), rtol=0, atol=0)

    def test_sm120_warmup_uses_packed_block32_call(self):
        executor = compile_utils._NormalWarmupExecutor.__new__(
            compile_utils._NormalWarmupExecutor
        )
        executor.recipe = (1, 32, 32)
        executor.packed_scales = True
        executor.lhs_q = torch.empty((4, 32), dtype=torch.float8_e4m3fn)
        executor.lhs_s = torch.ones((4, 1), dtype=torch.float32)
        executor.rhs_q = torch.empty((8, 32), dtype=torch.float8_e4m3fn)
        executor.rhs_s = torch.ones((1, 1), dtype=torch.float32)
        executor.out = torch.zeros((4, 8), dtype=torch.bfloat16)
        packed_lhs = torch.zeros((2, 1), dtype=torch.int32)
        packed_rhs = torch.zeros((8, 1), dtype=torch.int32)
        deep_gemm = SimpleNamespace(
            transform_sf_into_required_layout=Mock(
                side_effect=(packed_lhs, packed_rhs)
            ),
            fp8_fp4_gemm_nt=Mock(
                side_effect=lambda lhs, rhs, out, **kwargs: out.fill_(5)
            ),
            fp8_gemm_nt=Mock(),
        )

        with patch.object(compile_utils, "deep_gemm", deep_gemm, create=True):
            executor.execute(m=2)

        lhs_call, rhs_call = deep_gemm.transform_sf_into_required_layout.call_args_list
        self.assertTrue(torch.equal(lhs_call.args[0], executor.lhs_s[:2]))
        self.assertEqual(
            lhs_call.kwargs,
            {
                "mn": 2,
                "k": 32,
                "recipe": (1, 32, 32),
                "is_sfa": True,
            },
        )
        self.assertTrue(torch.equal(rhs_call.args[0], executor.rhs_s))
        self.assertEqual(
            rhs_call.kwargs,
            {
                "mn": 8,
                "k": 32,
                "recipe": (1, 32, 32),
                "is_sfa": False,
            },
        )
        args, kwargs = deep_gemm.fp8_fp4_gemm_nt.call_args
        self.assertIs(args[0][1], packed_lhs)
        self.assertIs(args[1][1], packed_rhs)
        self.assertEqual(
            kwargs,
            {
                "recipe_a": (1, 32),
                "recipe_b": (1, 32),
                "disable_ue8m0_cast": True,
            },
        )
        deep_gemm.fp8_gemm_nt.assert_not_called()
        torch.testing.assert_close(
            executor.out[:2], torch.full_like(executor.out[:2], 5), rtol=0, atol=0
        )
        torch.testing.assert_close(
            executor.out[2:], torch.zeros_like(executor.out[2:]), rtol=0, atol=0
        )

    def test_budget_covers_executor_buffers(self):
        for architecture in ("hopper", "blackwell", "sm120", "musa"):
            for max_m, n, k, num_groups in (
                (9, 131, 259, 3),  # Ceil division and grouped packing fallback.
                (16384, 4096, 7168, 32),  # V3 DP8/EP8 startup OOM shape.
            ):
                for kernel_type in DeepGemmKernelType:
                    if (
                        kernel_type == DeepGemmKernelType.GEMM_NT_F8F8BF16_BLOCK32
                        and architecture == "musa"
                    ):
                        continue  # Native block32 is dispatched on SM90/SM100/SM120.
                    with self.subTest(
                        architecture=architecture, kernel_type=kernel_type, max_m=max_m
                    ):
                        with ExitStack() as stack:
                            stack.enter_context(
                                patch.object(
                                    compile_utils,
                                    "DEEPGEMM_NEED_TMA_ALIGNED_SCALES",
                                    architecture == "hopper",
                                )
                            )
                            stack.enter_context(
                                patch.object(
                                    compile_utils,
                                    "DEEPGEMM_SCALE_UE8M0",
                                    architecture in ("blackwell", "sm120"),
                                )
                            )
                            stack.enter_context(
                                patch.object(
                                    compile_utils,
                                    "DEEPGEMM_PACKED_BLOCK32",
                                    architecture == "sm120",
                                )
                            )
                            for name in ("empty", "ones", "zeros"):
                                factory = getattr(torch, name)
                                stack.enter_context(
                                    patch.object(
                                        torch,
                                        name,
                                        side_effect=_allocate_on_meta(factory),
                                    )
                                )
                            executor = _BaseWarmupExecutor.create(
                                kernel_type,
                                max_m=max_m,
                                n=n,
                                k=k,
                                num_groups=num_groups,
                            )
                            budget_gib = _BaseWarmupExecutor.get_memory_requirement(
                                kernel_type,
                                max_m=max_m,
                                n=n,
                                k=k,
                                num_groups=num_groups,
                            )
                        if hasattr(executor, "packed_scales"):
                            self.assertEqual(
                                executor.packed_scales,
                                architecture == "sm120"
                                and kernel_type
                                == DeepGemmKernelType.GEMM_NT_F8F8BF16_BLOCK32,
                            )
                        allocated_bytes = _storage_bytes(
                            *[
                                tensor
                                for tensor in vars(executor).values()
                                if isinstance(tensor, torch.Tensor)
                            ]
                        )
                        peak_bytes = allocated_bytes + _scale_workspace_on_meta(
                            executor, architecture
                        )
                        self.assertGreaterEqual(budget_gib * (1 << 30), peak_bytes)
                        # Upper bounds include the last row's TMA padding.
                        self.assertLess(budget_gib * (1 << 30) - peak_bytes, 64)


class TestSm120DeepGemmAvailability(CustomTestCase):
    def test_incomplete_api_disables_sm120_deepgemm(self):
        names = (
            "fp8_einsum",
            "fp8_fp4_gemm_nt",
            "m_grouped_fp8_fp4_gemm_nt_contiguous",
            "transform_sf_into_required_layout",
        )
        for sm_version in (120, 121):
            for missing in (None, *names):
                with self.subTest(sm_version=sm_version, missing=missing):
                    module = SimpleNamespace(**{name: Mock() for name in names})
                    if missing is not None:
                        setattr(module, missing, None)
                    with (
                        patch.dict(sys.modules, {"deep_gemm": module}),
                        patch.object(configurer, "_is_cuda", True),
                        patch.object(configurer, "_is_musa", False),
                        patch.object(
                            configurer, "get_device_sm", return_value=sm_version
                        ),
                        patch.object(
                            configurer.envs.SGLANG_ENABLE_JIT_DEEPGEMM,
                            "get",
                            return_value=True,
                        ),
                    ):
                        self.assertEqual(
                            configurer._compute_enable_deep_gemm(), missing is None
                        )


if __name__ == "__main__":
    unittest.main()
