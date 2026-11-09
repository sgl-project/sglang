"""CPU-only tests for decode and prefill graph backend resolution."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.runner_backend import utils as backend_utils
from sglang.srt.model_executor.runner_backend.base_breakable_graph_backend import (
    BaseBreakableGraphBackend,
)
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_graph_runner(*, device="custom"):
    return SimpleNamespace(model_runner=SimpleNamespace(device=device))


def _make_exec_config(
    *,
    backend=Backend.FULL,
    prefill_backend=Backend.BREAKABLE,
    enable_memory_saver=False,
    debug_cuda_graph=False,
):
    return SimpleNamespace(
        graph=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend=backend),
                prefill=SimpleNamespace(backend=prefill_backend),
            ),
            debug_cuda_graph=debug_cuda_graph,
        ),
        features=SimpleNamespace(
            enable_memory_saver=enable_memory_saver,
        ),
    )


def _make_platform(
    *, is_out_of_tree, full_backend_cls=None, breakable_backend_cls=None
):
    return SimpleNamespace(
        is_out_of_tree=mock.Mock(return_value=is_out_of_tree),
        get_full_graph_backend_cls=mock.Mock(return_value=full_backend_cls),
        get_breakable_graph_backend_cls=mock.Mock(return_value=breakable_backend_cls),
    )


class _OutOfTreePlatformWithoutFullGraphBackend(SRTPlatform):
    _enum = PlatformEnum.OOT


class _FakeBreakableBackend(BaseBreakableGraphBackend):
    def __init__(self, runner, **options):
        self.runner = runner
        self.options = options

    def capture_session(self, stream):
        raise NotImplementedError

    def capture_one(
        self,
        shape_key,
        forward_fn,
        capture_inputs=None,
        post_warmup_hook=None,
    ):
        raise NotImplementedError

    def can_run(self, forward_batch, shape_key):
        return False

    def replay_session(self):
        raise NotImplementedError

    def replay(self, shape_key, static_forward_batch, **kwargs):
        raise NotImplementedError

    def cleanup(self):
        pass


class TestGraphBackendTypes(CustomTestCase):
    def test_cuda_breakable_backend_implements_neutral_contract(self):
        self.assertTrue(
            issubclass(BreakableCudaGraphBackend, BaseBreakableGraphBackend)
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

    def test_out_of_tree_platform_without_full_backend_uses_default(self):
        runner = _make_graph_runner()
        exec_config = _make_exec_config()
        expected_backend = object()
        platform = _OutOfTreePlatformWithoutFullGraphBackend()

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
        default_backend_cls.assert_called_once_with(
            runner,
            enable_memory_saver=False,
        )

    def test_retired_backend_is_not_replaced_with_full(self):
        runner = _make_graph_runner(device="cuda")
        exec_config = _make_exec_config(backend=Backend.TC_PIECEWISE)
        with (
            mock.patch.object(backend_utils, "get_exec", return_value=exec_config),
            mock.patch.object(backend_utils, "FullCudaGraphBackend") as full_backend,
        ):
            with self.assertRaisesRegex(ValueError, "Unsupported decode graph backend"):
                backend_utils.resolve_decode_backend(runner)
            full_backend.assert_not_called()

    def test_out_of_tree_platform_provides_breakable_backend(self):
        runner = _make_graph_runner()
        exec_config = _make_exec_config(
            backend=Backend.BREAKABLE,
            enable_memory_saver=True,
            debug_cuda_graph=True,
        )
        platform = _make_platform(
            is_out_of_tree=True,
            breakable_backend_cls=_FakeBreakableBackend,
        )

        with (
            mock.patch.object(backend_utils, "get_exec", return_value=exec_config),
            mock.patch.object(backend_utils, "current_platform", platform),
        ):
            actual_backend = backend_utils.resolve_decode_backend(runner)

        self.assertIsInstance(actual_backend, _FakeBreakableBackend)
        self.assertIs(actual_backend.runner, runner)
        self.assertEqual(
            actual_backend.options,
            {"enable_memory_saver": True, "debug_eager": True},
        )
        platform.get_breakable_graph_backend_cls.assert_called_once_with("decode")
        platform.get_full_graph_backend_cls.assert_not_called()

    def test_breakable_none_preserves_decode_default(self):
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
        platform.is_out_of_tree.assert_called_once_with()
        platform.get_breakable_graph_backend_cls.assert_called_once_with("decode")
        platform.get_full_graph_backend_cls.assert_not_called()
        breakable_backend_cls.assert_called_once_with(
            runner,
            enable_memory_saver=True,
            debug_eager=True,
        )

    def test_in_tree_breakable_does_not_consult_platform_factory(self):
        runner = _make_graph_runner(device="cuda")
        exec_config = _make_exec_config(backend=Backend.BREAKABLE)
        expected_backend = object()
        platform = _make_platform(is_out_of_tree=False)

        with (
            mock.patch.object(backend_utils, "get_exec", return_value=exec_config),
            mock.patch.object(backend_utils, "current_platform", platform),
            mock.patch.object(
                backend_utils,
                "BreakableCudaGraphBackend",
                return_value=expected_backend,
            ),
        ):
            actual_backend = backend_utils.resolve_decode_backend(runner)

        self.assertIs(actual_backend, expected_backend)
        platform.get_breakable_graph_backend_cls.assert_not_called()

    def test_breakable_factory_rejects_non_breakable_class(self):
        runner = _make_graph_runner()
        exec_config = _make_exec_config(backend=Backend.BREAKABLE)
        platform = _make_platform(
            is_out_of_tree=True,
            breakable_backend_cls=object,
        )

        with (
            mock.patch.object(backend_utils, "get_exec", return_value=exec_config),
            mock.patch.object(backend_utils, "current_platform", platform),
        ):
            with self.assertRaisesRegex(TypeError, "BaseBreakableGraphBackend"):
                backend_utils.resolve_decode_backend(runner)


class TestResolvePrefillBackend(CustomTestCase):
    def test_out_of_tree_platform_provides_breakable_backend(self):
        runner = _make_graph_runner()
        exec_config = _make_exec_config(
            prefill_backend=Backend.BREAKABLE,
            enable_memory_saver=True,
            debug_cuda_graph=True,
        )
        platform = _make_platform(
            is_out_of_tree=True,
            breakable_backend_cls=_FakeBreakableBackend,
        )

        with (
            mock.patch.object(backend_utils, "get_exec", return_value=exec_config),
            mock.patch.object(backend_utils, "current_platform", platform),
            mock.patch.object(
                backend_utils,
                "_needs_hicache_event_capture",
                return_value=True,
            ),
        ):
            actual_backend = backend_utils.resolve_prefill_backend(runner)

        self.assertIsInstance(actual_backend, _FakeBreakableBackend)
        self.assertIs(actual_backend.runner, runner)
        self.assertEqual(
            actual_backend.options,
            {
                "enable_memory_saver": True,
                "debug_eager": True,
                "enable_hicache_event_capture": True,
            },
        )
        platform.get_breakable_graph_backend_cls.assert_called_once_with("prefill")

    def test_breakable_none_preserves_prefill_default(self):
        runner = _make_graph_runner()
        exec_config = _make_exec_config(prefill_backend=Backend.BREAKABLE)
        expected_backend = object()
        platform = _make_platform(is_out_of_tree=True)

        with (
            mock.patch.object(backend_utils, "get_exec", return_value=exec_config),
            mock.patch.object(backend_utils, "current_platform", platform),
            mock.patch.object(
                backend_utils,
                "BreakableCudaGraphBackend",
                return_value=expected_backend,
            ) as breakable_backend_cls,
            mock.patch.object(
                backend_utils,
                "_needs_hicache_event_capture",
                return_value=False,
            ),
        ):
            actual_backend = backend_utils.resolve_prefill_backend(runner)

        self.assertIs(actual_backend, expected_backend)
        platform.get_breakable_graph_backend_cls.assert_called_once_with("prefill")
        breakable_backend_cls.assert_called_once_with(
            runner,
            enable_memory_saver=False,
            debug_eager=False,
            enable_hicache_event_capture=False,
        )


if __name__ == "__main__":
    unittest.main()
