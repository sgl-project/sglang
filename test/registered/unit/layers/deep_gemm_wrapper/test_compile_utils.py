"""Tests for DeepGEMM warmup memory-budget selection."""

import unittest
from contextlib import ExitStack
from typing import Optional
from unittest.mock import MagicMock, call, patch

from sglang.srt.layers.deep_gemm_wrapper import compile_utils
from sglang.srt.layers.deep_gemm_wrapper.compile_utils import (
    DeepGemmKernelType,
    _BaseWarmupExecutor,
    _compile_deep_gemm_one_type_all,
    _select_max_m_within_budget,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_M_LIST = list(range(1, 16 * 1024 + 1))
_KERNEL_TYPE = DeepGemmKernelType.GEMM_NT_F8F8BF16


def _select(memory_budget: float) -> Optional[int]:
    return _select_max_m_within_budget(
        _KERNEL_TYPE,
        m_list=_M_LIST,
        n=4096,
        k=7168,
        num_groups=1,
        memory_budget=memory_budget,
    )


class TestSelectMaxMWithinBudget(CustomTestCase):
    def _patch_memory_requirement(self, fn):
        return patch.object(
            _BaseWarmupExecutor, "get_memory_requirement", staticmethod(fn)
        )

    def test_original_max_m_fits(self):
        with self._patch_memory_requirement(lambda *a, **kw: 1.0):
            self.assertEqual(_select(memory_budget=8.0), max(_M_LIST))

    def test_reduces_below_2048(self):
        with self._patch_memory_requirement(lambda *a, max_m, **kw: max_m / 1024):
            self.assertEqual(_select(memory_budget=1.5), 1024)

    def test_returns_none_when_min_m_does_not_fit(self):
        with self._patch_memory_requirement(lambda *a, **kw: 100.0):
            self.assertIsNone(_select(memory_budget=8.0))


class TestCompileDeepGemmOneTypeAll(CustomTestCase):
    """Covers how the caller reacts to the max_m selection outcome."""

    def _patch_common(self, stack, available_memory: float, memory_requirement):
        stack.enter_context(
            patch.object(
                _BaseWarmupExecutor,
                "get_memory_requirement",
                staticmethod(memory_requirement),
            )
        )
        saved_context = object()
        stack.enter_context(
            patch.object(
                compile_utils,
                "disable_symmetric_memory_context",
                return_value=saved_context,
            )
        )
        restore = stack.enter_context(
            patch.object(compile_utils, "restore_symmetric_memory_context")
        )
        stack.enter_context(
            patch.object(
                compile_utils, "get_available_gpu_memory", return_value=available_memory
            )
        )
        stack.enter_context(patch("torch.cuda.current_device", return_value=0))
        return saved_context, restore

    def test_skips_warmup_when_min_m_does_not_fit(self):
        with ExitStack() as stack:
            saved_context, restore = self._patch_common(
                stack, available_memory=1.0, memory_requirement=lambda *a, **kw: 100.0
            )
            create = stack.enter_context(
                patch.object(_BaseWarmupExecutor, "create", MagicMock())
            )
            with self.assertLogs(compile_utils.logger, level="WARNING") as logs:
                _compile_deep_gemm_one_type_all(
                    kernel_type=_KERNEL_TYPE,
                    n=4096,
                    k=7168,
                    num_groups=1,
                    m_list=[1024, 2048, 4096],
                )

        create.assert_not_called()
        self.assertIn("skipping warmup", "\n".join(logs.output))
        restore.assert_called_once_with(saved_context)

    def test_warms_up_reduced_m_list(self):
        with ExitStack() as stack:
            saved_context, restore = self._patch_common(
                stack,
                available_memory=2.5,
                memory_requirement=lambda *a, max_m, **kw: max_m / 1024,
            )
            executor = MagicMock()
            create = stack.enter_context(
                patch.object(_BaseWarmupExecutor, "create", MagicMock())
            )
            create.return_value = executor
            stack.enter_context(
                patch.object(compile_utils, "deep_gemm", MagicMock(), create=True)
            )
            stack.enter_context(patch("torch.cuda.current_stream"))
            stack.enter_context(patch("torch.cuda.empty_cache"))

            _compile_deep_gemm_one_type_all(
                kernel_type=_KERNEL_TYPE,
                n=4096,
                k=7168,
                num_groups=1,
                m_list=[1024, 2048, 4096],
            )

        create.assert_called_once_with(
            _KERNEL_TYPE, max_m=2048, n=4096, k=7168, num_groups=1
        )
        self.assertEqual(executor.execute.call_args_list, [call(m=1024), call(m=2048)])
        restore.assert_called_once_with(saved_context)


if __name__ == "__main__":
    unittest.main()
