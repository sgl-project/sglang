"""CPU test: the offline Engine freezes GC once its subprocesses are up, and the
TokenizerManager freeze path it relies on.

The engine-level check with real subprocesses is in
``test/registered/core/test_engine_child_pids.py`` (CUDA).
"""

import asyncio
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints import engine as engine_module
from sglang.srt.managers import tokenizer_manager as tm_module
from sglang.srt.managers.io_struct import FreezeGCReq
from sglang.srt.managers.multi_tokenizer_mixin import MultiTokenizerRouter
from sglang.srt.managers.tokenizer_manager import TokenizerManager

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestEngineFreezesGcAfterStartup(CustomTestCase):
    def _init_engine(self, tokenizer_manager, worker=None, node_rank=0):
        launched = (
            tokenizer_manager,
            None,  # template_manager
            SimpleNamespace(rpc_ipc_name="ipc://unused"),
            SimpleNamespace(all_child_pids=[], scheduler_infos=[{}]),
            None,  # subprocess_watchdog
            [],  # weight_cache_daemon_procs
        )
        patches = (
            mock.patch.object(engine_module, "load_plugins"),
            mock.patch.object(engine_module, "_server_args_for_logging"),
            mock.patch.object(engine_module.atexit, "register"),
            mock.patch.object(
                engine_module.Engine, "_launch_subprocesses", return_value=launched
            ),
            mock.patch.object(engine_module.zmq, "Context"),
            mock.patch.object(engine_module, "get_zmq_socket"),
            mock.patch.object(
                engine_module,
                "get_parallel",
                return_value=SimpleNamespace(node_rank=node_rank),
            ),
            mock.patch.object(
                engine_module,
                "get_observability",
                return_value=SimpleNamespace(enable_trace=False),
            ),
        )
        if worker is not None:
            patches += (
                mock.patch.object(engine_module, "write_data_for_multi_tokenizer"),
                mock.patch.object(
                    engine_module.Engine,
                    "_init_tokenizer_worker",
                    return_value=(worker, None),
                ),
            )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return engine_module.Engine(server_args=SimpleNamespace())

    def test_collects_then_freezes_through_the_tokenizer_manager(self):
        order = mock.MagicMock()
        with mock.patch.object(engine_module.gc, "collect", order.collect):
            engine = self._init_engine(order.tokenizer_manager)
        self.assertEqual(
            order.mock_calls,
            [mock.call.collect(), mock.call.tokenizer_manager.freeze_gc_sync()],
        )
        self.assertTrue(engine._froze_gc)

    def test_opt_out(self):
        tokenizer_manager = mock.MagicMock()
        with (
            engine_module.envs.SGLANG_ENGINE_FREEZE_GC_AFTER_STARTUP.override(False),
            mock.patch.object(engine_module.gc, "collect") as collect,
        ):
            engine = self._init_engine(tokenizer_manager)
        tokenizer_manager.freeze_gc_sync.assert_not_called()
        collect.assert_not_called()
        self.assertFalse(engine._froze_gc)

    def test_no_tokenizer_manager_on_non_zero_rank(self):
        with mock.patch.object(engine_module.gc, "collect") as collect:
            engine = self._init_engine(None, node_rank=1)
        self.assertIsNone(engine.tokenizer_manager)
        collect.assert_not_called()
        self.assertFalse(engine._froze_gc)

    def test_multi_tokenizer_freezes_through_the_worker_not_the_router(self):
        router = MultiTokenizerRouter.__new__(MultiTokenizerRouter)
        router.startup_time = {}
        router.freeze_gc_sync = mock.MagicMock()
        worker = mock.MagicMock()
        engine = self._init_engine(router, worker=worker)
        self.assertIs(engine.tokenizer_manager, worker)
        worker.freeze_gc_sync.assert_called_once_with()
        router.freeze_gc_sync.assert_not_called()


class TestEngineShutdownUnfreezesGc(CustomTestCase):
    def _shutdown(self, froze_gc):
        engine = engine_module.Engine.__new__(engine_module.Engine)
        engine.tokenizer_manager = None
        engine._weight_cache_daemon_procs = []
        engine._froze_gc = froze_gc
        with mock.patch.object(engine_module, "kill_process_tree"):
            engine.shutdown()
        return engine

    def test_unfreezes_what_it_froze(self):
        with mock.patch.object(engine_module.gc, "unfreeze") as unfreeze:
            engine = self._shutdown(froze_gc=True)
        unfreeze.assert_called_once_with()
        self.assertFalse(engine._froze_gc)

    def test_unfreezes_even_if_teardown_raises(self):
        engine = engine_module.Engine.__new__(engine_module.Engine)
        engine.tokenizer_manager = None
        engine._weight_cache_daemon_procs = []
        engine._froze_gc = True
        with (
            mock.patch.object(
                engine_module, "kill_process_tree", side_effect=RuntimeError
            ),
            mock.patch.object(engine_module.gc, "unfreeze") as unfreeze,
        ):
            with self.assertRaises(RuntimeError):
                engine.shutdown()
        unfreeze.assert_called_once_with()

    def test_leaves_gc_alone_when_it_did_not_freeze(self):
        with mock.patch.object(engine_module.gc, "unfreeze") as unfreeze:
            self._shutdown(froze_gc=False)
        unfreeze.assert_not_called()


class TestEngineFreezeGcMethod(CustomTestCase):
    def test_works_inside_a_running_event_loop(self):
        """Engine.freeze_gc() must not raise when the caller is already in an
        event loop (e.g. an async web handler)."""
        engine = engine_module.Engine.__new__(engine_module.Engine)
        engine.tokenizer_manager = mock.MagicMock()
        engine.loop = asyncio.new_event_loop()
        self.addCleanup(engine.loop.close)

        async def call_from_loop():
            engine.freeze_gc()

        asyncio.run(call_from_loop())
        engine.tokenizer_manager.freeze_gc_sync.assert_called_once_with()


class TestTokenizerManagerFreezeGc(CustomTestCase):
    def _fake_manager(self):
        manager = SimpleNamespace(_dispatch_to_scheduler=mock.MagicMock())
        manager.freeze_gc_sync = types.MethodType(
            TokenizerManager.freeze_gc_sync, manager
        )
        return manager

    def _patched(self, gc_threshold):
        return (
            mock.patch.object(
                tm_module,
                "get_device",
                return_value=SimpleNamespace(gc_threshold=gc_threshold),
            ),
            mock.patch.object(tm_module, "freeze_gc"),
        )

    def _freeze_via(self, entry_point, manager, gc_threshold):
        p_device, p_freeze = self._patched(gc_threshold)
        with p_device, p_freeze as freeze:
            if entry_point == "async":
                asyncio.run(TokenizerManager.freeze_gc(manager))
            else:
                TokenizerManager.freeze_gc_sync(manager)
        return freeze

    def test_dispatches_to_scheduler_and_freezes_locally(self):
        for entry_point in ("async", "sync"):
            with self.subTest(entry_point=entry_point):
                manager = self._fake_manager()
                freeze = self._freeze_via(entry_point, manager, [700, 10, 1000])
                (req,), _ = manager._dispatch_to_scheduler.call_args
                self.assertIsInstance(req, FreezeGCReq)
                self.assertEqual(req.gc_threshold, [700, 10, 1000])
                freeze.assert_called_once_with("Tokenizer Manager", [700, 10, 1000])


if __name__ == "__main__":
    unittest.main()
