import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEngineGracefulShutdown(CustomTestCase):
    def _engine(self):
        from sglang.srt.entrypoints.engine import Engine
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        manager = object.__new__(TokenizerManager)
        manager._subprocess_watchdog = MagicMock()
        manager.cuda_vmm_feature_transport = MagicMock()
        manager.mm_processor = None
        engine = object.__new__(Engine)
        engine.tokenizer_manager = manager
        return engine, manager

    def test_shutdown_waits_for_schedulers_before_killing(self):
        from sglang.srt.entrypoints import engine as engine_module
        from sglang.srt.managers import tokenizer_manager as tm_module
        from sglang.srt.managers.io_struct import ShutdownReq

        engine, manager = self._engine()
        events = []
        manager._dispatch_to_scheduler = lambda obj: events.append(type(obj))

        def schedulers():
            # Alive before ShutdownReq and for two polls after it, then gone.
            if ShutdownReq not in events or events.count("poll after") < 2:
                events.append("poll after" if ShutdownReq in events else "poll")
                return [MagicMock()]
            events.append("gone")
            return []

        with (
            patch.object(engine_module, "collect_scheduler_processes", schedulers),
            patch.object(tm_module, "collect_scheduler_processes", schedulers),
            patch.object(tm_module.time, "sleep"),
            patch.object(
                engine_module,
                "kill_process_tree",
                side_effect=lambda *a, **k: events.append("kill"),
            ),
        ):
            engine.shutdown()

        sent = events.index(ShutdownReq)
        self.assertLess(sent, events.index("gone"))
        self.assertLess(events.index("gone"), events.index("kill"))
        manager._subprocess_watchdog.stop.assert_called_once()

    def test_shutdown_still_kills_when_graceful_stop_fails(self):
        from sglang.srt.entrypoints import engine as engine_module
        from sglang.srt.managers import tokenizer_manager as tm_module

        engine, manager = self._engine()
        manager._dispatch_to_scheduler = MagicMock(side_effect=OSError("closed"))
        kill = MagicMock()
        alive = MagicMock(return_value=[MagicMock()])
        with (
            patch.object(engine_module, "collect_scheduler_processes", alive),
            patch.object(tm_module, "collect_scheduler_processes", alive),
            patch.object(engine_module, "kill_process_tree", kill),
        ):
            engine.shutdown()

        kill.assert_called_once()

    def test_repeated_shutdown_sends_nothing_once_schedulers_are_gone(self):
        from sglang.srt.entrypoints import engine as engine_module

        engine, manager = self._engine()
        manager._dispatch_to_scheduler = MagicMock()
        with (
            patch.object(engine_module, "collect_scheduler_processes", return_value=[]),
            patch.object(engine_module, "kill_process_tree"),
        ):
            engine.shutdown()

        manager._dispatch_to_scheduler.assert_not_called()


if __name__ == "__main__":
    unittest.main()
