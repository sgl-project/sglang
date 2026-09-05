"""CPU-only tests for decode graph backend resolution."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.runner_backend import utils as backend_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_graph_runner(*, device="custom"):
    return SimpleNamespace(model_runner=SimpleNamespace(device=device))


def _make_exec_config(
    *,
    backend=Backend.FULL,
    enable_memory_saver=False,
    debug_cuda_graph=False,
):
    return SimpleNamespace(
        graph=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(backend=backend)),
            debug_cuda_graph=debug_cuda_graph,
        ),
        features=SimpleNamespace(
            enable_memory_saver=enable_memory_saver,
        ),
    )


def _make_platform(*, is_out_of_tree, full_backend_cls=None):
    return SimpleNamespace(
        is_out_of_tree=mock.Mock(return_value=is_out_of_tree),
        get_full_graph_backend_cls=mock.Mock(return_value=full_backend_cls),
    )


class TestResolveDecodeBackend(CustomTestCase):
    def test_out_of_tree_platform_provides_full_backend(self):
        runner = _make_graph_runner()
        exec_config = _make_exec_config(enable_memory_saver=True)
        expected_backend = object()
        backend_cls = mock.Mock(return_value=expected_backend)
        platform = _make_platform(
            is_out_of_tree=True,
            full_backend_cls=backend_cls,
        )

        with (
            mock.patch.object(
                backend_utils,
                "get_exec",
                return_value=exec_config,
            ),
            mock.patch.object(
                backend_utils,
                "current_platform",
                platform,
            ),
            mock.patch.object(
                backend_utils,
                "FullCudaGraphBackend",
            ) as default_backend_cls,
        ):
            actual_backend = backend_utils.resolve_decode_backend(runner)

        self.assertIs(actual_backend, expected_backend)
        platform.get_full_graph_backend_cls.assert_called_once_with()
        backend_cls.assert_called_once_with(
            runner,
            enable_memory_saver=True,
        )
        default_backend_cls.assert_not_called()

    def test_in_tree_platform_uses_default_full_backend(self):
        runner = _make_graph_runner(device="cuda")
        exec_config = _make_exec_config()
        expected_backend = object()
        platform = _make_platform(is_out_of_tree=False)

        with (
            mock.patch.object(
                backend_utils,
                "get_exec",
                return_value=exec_config,
            ),
            mock.patch.object(
                backend_utils,
                "current_platform",
                platform,
            ),
            mock.patch.object(
                backend_utils,
                "FullCudaGraphBackend",
                return_value=expected_backend,
            ) as default_backend_cls,
        ):
            actual_backend = backend_utils.resolve_decode_backend(runner)

        self.assertIs(actual_backend, expected_backend)
        platform.get_full_graph_backend_cls.assert_not_called()
        default_backend_cls.assert_called_once_with(
            runner,
            enable_memory_saver=False,
        )

    def test_breakable_backend_does_not_consult_full_backend_factory(self):
        runner = _make_graph_runner()
        exec_config = _make_exec_config(
            backend=Backend.BREAKABLE,
            enable_memory_saver=True,
            debug_cuda_graph=True,
        )
        expected_backend = object()
        platform = _make_platform(is_out_of_tree=True)

        with (
            mock.patch.object(
                backend_utils,
                "get_exec",
                return_value=exec_config,
            ),
            mock.patch.object(
                backend_utils,
                "current_platform",
                platform,
            ),
            mock.patch.object(
                backend_utils,
                "BreakableCudaGraphBackend",
                return_value=expected_backend,
            ) as breakable_backend_cls,
        ):
            actual_backend = backend_utils.resolve_decode_backend(runner)

        self.assertIs(actual_backend, expected_backend)
        platform.is_out_of_tree.assert_not_called()
        platform.get_full_graph_backend_cls.assert_not_called()
        breakable_backend_cls.assert_called_once_with(
            runner,
            enable_memory_saver=True,
            debug_eager=True,
        )


if __name__ == "__main__":
    unittest.main()
