"""CPU unit tests for DeepGEMM precompile cross-rank locking and synchronization."""

import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.deep_gemm_wrapper import compile_utils
from sglang.srt.layers.deep_gemm_wrapper.compile_utils import DeepGemmKernelType

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPrecompileLockSync(CustomTestCase):
    def setUp(self):
        compile_utils._INITIALIZATION_DICT.clear()

    def tearDown(self):
        compile_utils._INITIALIZATION_DICT.clear()

    def test_first_rank_acquires_lock_and_compiles(self):
        lock_entered = []
        compiled_called = []

        class MockLock:
            def __enter__(self):
                lock_entered.append(True)
                return self

            def __exit__(self, *args):
                pass

        with (
            patch.object(compile_utils, "_ENABLE_JIT_DEEPGEMM_PRECOMPILE", True),
            patch.object(compile_utils, "_IS_FIRST_RANK_ON_NODE", True),
            patch.object(
                compile_utils, "_local_rank_compile_lock", return_value=MockLock()
            ),
            patch.object(
                compile_utils,
                "_compile_deep_gemm_one_type_all",
                side_effect=lambda **kw: compiled_called.append(kw),
            ),
        ):
            compile_utils._maybe_compile_deep_gemm_one_type_all(
                DeepGemmKernelType.GEMM_NT_F8F8BF16, n=128, k=128, num_groups=1
            )

        self.assertEqual(len(lock_entered), 1)
        self.assertEqual(len(compiled_called), 1)
        self.assertEqual(compiled_called[0]["n"], 128)

    def test_non_first_rank_acquires_lock_but_skips_compile(self):
        lock_entered = []
        compiled_called = []

        class MockLock:
            def __enter__(self):
                lock_entered.append(True)
                return self

            def __exit__(self, *args):
                pass

        with (
            patch.object(compile_utils, "_ENABLE_JIT_DEEPGEMM_PRECOMPILE", True),
            patch.object(compile_utils, "_IS_FIRST_RANK_ON_NODE", False),
            patch.object(
                compile_utils, "_local_rank_compile_lock", return_value=MockLock()
            ),
            patch.object(
                compile_utils,
                "_compile_deep_gemm_one_type_all",
                side_effect=lambda **kw: compiled_called.append(kw),
            ),
        ):
            compile_utils._maybe_compile_deep_gemm_one_type_all(
                DeepGemmKernelType.GEMM_NT_F8F8BF16, n=128, k=128, num_groups=1
            )

        # The non-first rank MUST enter the lock to synchronize with rank 0,
        # but skip the expensive redundant compilation
        self.assertEqual(len(lock_entered), 1)
        self.assertEqual(len(compiled_called), 0)

    def test_barrier_called_when_tp_size_greater_than_one(self):
        mock_parallel = MagicMock()
        mock_parallel.tp_size = 8
        mock_parallel.tp_group = MagicMock()

        class MockLock:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        with (
            patch.object(compile_utils, "_ENABLE_JIT_DEEPGEMM_PRECOMPILE", True),
            patch.object(compile_utils, "_IS_FIRST_RANK_ON_NODE", False),
            patch.object(
                compile_utils, "_local_rank_compile_lock", return_value=MockLock()
            ),
            patch.object(compile_utils, "get_parallel", return_value=mock_parallel),
        ):
            compile_utils._maybe_compile_deep_gemm_one_type_all(
                DeepGemmKernelType.GEMM_NT_F8F8BF16, n=128, k=128, num_groups=1
            )

        mock_parallel.tp_group.barrier.assert_called_once()


if __name__ == "__main__":
    unittest.main()
