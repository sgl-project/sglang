"""CPU checks for exception-safe DeepGEMM warmup state."""

import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.deep_gemm_wrapper import compile_utils

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestWarmupState(CustomTestCase):
    def setUp(self):
        super().setUp()
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.backend = SimpleNamespace(mode=7)
        self.backend.get_compile_mode = MagicMock(side_effect=lambda: self.backend.mode)
        self.backend.set_compile_mode = MagicMock(
            side_effect=lambda mode: setattr(self.backend, "mode", mode)
        )
        self.executor = MagicMock()
        self.factory = SimpleNamespace(
            create=MagicMock(return_value=self.executor),
            get_memory_requirement=MagicMock(return_value=1),
        )
        self.cuda = SimpleNamespace(
            current_device=MagicMock(return_value=0),
            current_stream=MagicMock(return_value=MagicMock()),
            empty_cache=MagicMock(),
        )
        self.saved_context = object()
        self.restore = MagicMock()
        for name, value in {
            "deep_gemm": self.backend,
            "torch": SimpleNamespace(cuda=self.cuda),
            "_BaseWarmupExecutor": self.factory,
            "get_available_gpu_memory": MagicMock(return_value=8),
            "disable_symmetric_memory_context": MagicMock(
                return_value=self.saved_context
            ),
            "restore_symmetric_memory_context": self.restore,
            "tqdm": lambda values, **kwargs: values,
            "_local_rank_compile_lock": MagicMock(side_effect=lambda *a: nullcontext()),
            "_ENABLE_JIT_DEEPGEMM_PRECOMPILE": True,
            "_DO_COMPILE_ALL": True,
            "_IN_PRECOMPILE_STAGE": True,
            "_INITIALIZATION_DICT": {},
            "_BUILTIN_M_LIST": [1, 8],
        }.items():
            stack.enter_context(patch.object(compile_utils, name, value, create=True))
        self.kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16
        self.key = (self.kernel_type, 128, 128, 1)

    def warmup(self):
        compile_utils._maybe_compile_deep_gemm_one_type_all(*self.key)

    def test_success_restores_state_and_skips_completed_group(self):
        self.warmup()
        self.assertEqual(self.backend.mode, 7)
        self.assertEqual(self.executor.execute.call_args_list, [call(m=1), call(m=8)])
        self.backend.set_compile_mode.assert_has_calls([call(1), call(7)])
        self.restore.assert_called_once_with(self.saved_context)
        self.cuda.current_stream.return_value.synchronize.assert_called_once_with()
        self.cuda.empty_cache.assert_called_once_with()
        self.assertTrue(compile_utils._INITIALIZATION_DICT[self.key])
        self.warmup()
        self.factory.create.assert_called_once()

    def test_failed_execution_restores_state_and_allows_retry(self):
        for error in (RuntimeError("compile failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                compile_utils._INITIALIZATION_DICT.clear()
                self.restore.reset_mock()
                self.executor.execute.side_effect = error
                with self.assertRaises(type(error)) as raised:
                    self.warmup()
                self.assertIs(raised.exception, error)
                self.assertEqual(self.backend.mode, 7)
                self.restore.assert_called_once_with(self.saved_context)
                self.assertNotIn(self.key, compile_utils._INITIALIZATION_DICT)
                self.executor.execute.side_effect = None
                self.warmup()
                self.assertTrue(compile_utils._INITIALIZATION_DICT[self.key])

    def test_legacy_backend_without_complete_compile_mode_api(self):
        for missing in (
            ("get_compile_mode",),
            ("set_compile_mode",),
            ("get_compile_mode", "set_compile_mode"),
        ):
            with self.subTest(missing=missing):
                backend = SimpleNamespace(
                    get_compile_mode=MagicMock(), set_compile_mode=MagicMock()
                )
                for name in missing:
                    delattr(backend, name)
                with patch.object(compile_utils, "deep_gemm", backend):
                    compile_utils._INITIALIZATION_DICT.clear()
                    self.warmup()
                    for name in ("get_compile_mode", "set_compile_mode"):
                        if hasattr(backend, name):
                            getattr(backend, name).assert_not_called()

    def test_executor_creation_failure_restores_context(self):
        error = RuntimeError("allocation failed")
        self.factory.create.side_effect = error
        with self.assertRaises(RuntimeError) as raised:
            self.warmup()
        self.assertIs(raised.exception, error)
        self.backend.set_compile_mode.assert_not_called()
        self.restore.assert_called_once_with(self.saved_context)
        self.assertNotIn(self.key, compile_utils._INITIALIZATION_DICT)

    def test_compile_mode_entry_failure_restores_previous_mode(self):
        error = RuntimeError("enter compile mode failed")

        def set_mode(mode):
            self.backend.mode = mode
            if mode == 1:
                raise error

        self.backend.set_compile_mode.side_effect = set_mode
        with self.assertRaises(RuntimeError) as raised:
            self.warmup()
        self.assertIs(raised.exception, error)
        self.assertEqual(self.backend.mode, 7)
        self.executor.execute.assert_not_called()
        self.restore.assert_called_once_with(self.saved_context)
        self.assertNotIn(self.key, compile_utils._INITIALIZATION_DICT)

    def test_synchronize_failure_does_not_mark_group_complete(self):
        error = RuntimeError("asynchronous warmup failure")
        self.cuda.current_stream.return_value.synchronize.side_effect = error
        with self.assertRaises(RuntimeError) as raised:
            self.warmup()
        self.assertIs(raised.exception, error)
        self.assertEqual(self.backend.mode, 7)
        self.restore.assert_called_once_with(self.saved_context)
        self.assertNotIn(self.key, compile_utils._INITIALIZATION_DICT)

    def test_lock_failure_does_not_remove_other_completed_groups(self):
        other_key = (self.kernel_type, 256, 128, 1)
        compile_utils._INITIALIZATION_DICT[other_key] = True
        error = OSError("lock unavailable")
        with patch.object(compile_utils, "_local_rank_compile_lock", side_effect=error):
            with self.assertRaises(OSError) as raised:
                self.warmup()
        self.assertIs(raised.exception, error)
        self.assertEqual(compile_utils._INITIALIZATION_DICT, {other_key: True})
        self.factory.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
