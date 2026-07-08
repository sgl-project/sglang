"""Unit tests for runtime plugin loading in subprocess entrypoints."""

import sys
import types
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    import torch

    try:
        torch.xpu.get_device_name = lambda device=None: ""
    except Exception:
        pass
except Exception:
    pass


def _maybe_stub_sgl_kernel():
    try:
        import sgl_kernel  # noqa: F401

        return
    except (ImportError, OSError):
        pass

    stub = types.ModuleType("sgl_kernel")
    stub.__getattr__ = lambda name: MagicMock(name=f"sgl_kernel.{name}")
    sys.modules["sgl_kernel"] = stub


_maybe_stub_sgl_kernel()

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSubprocessRuntimePlugins(unittest.TestCase):
    def _patch_common_process_setup(self, module, events):
        stack = ExitStack()
        stack.enter_context(
            patch.object(
                module,
                "load_plugins",
                side_effect=lambda: events.append("load_plugins"),
            )
        )
        stack.enter_context(patch.object(module, "kill_itself_when_parent_died"))
        stack.enter_context(patch.object(module.setproctitle, "setproctitle"))
        stack.enter_context(patch.object(module, "configure_logger"))
        stack.enter_context(
            patch.object(
                module.psutil,
                "Process",
                return_value=SimpleNamespace(parent=lambda: MagicMock()),
            )
        )
        return stack

    def test_detokenizer_loads_plugins_before_manager_init(self):
        from sglang.srt.managers import detokenizer_manager as module

        events = []

        class DummyDetokenizerManager:
            def __init__(self, server_args, port_args):
                events.append("manager_init")

            def event_loop(self):
                events.append("event_loop")

        server_args = SimpleNamespace(tokenizer_worker_num=1)
        port_args = SimpleNamespace()

        with self._patch_common_process_setup(module, events):
            module.run_detokenizer_process(
                server_args,
                port_args,
                detokenizer_manager_class=DummyDetokenizerManager,
            )

        self.assertEqual(events, ["load_plugins", "manager_init", "event_loop"])

    def test_multi_detokenizer_router_loads_plugins_before_router_init(self):
        from sglang.srt.managers import multi_tokenizer_mixin as module

        events = []

        class DummyRouter:
            def __init__(self, ipc_name_list, port_args):
                events.append(("router_init", ipc_name_list))
                self.socket_mapping = SimpleNamespace(clear_all_sockets=lambda: None)

            def event_loop(self):
                events.append("event_loop")

        with self._patch_common_process_setup(module, events):
            with patch.object(module, "MultiDetokenizerRouter", DummyRouter):
                module.run_multi_detokenizer_router_process(
                    ["ipc://worker0"],
                    SimpleNamespace(),
                    SimpleNamespace(),
                )

        self.assertEqual(
            events,
            ["load_plugins", ("router_init", ["ipc://worker0"]), "event_loop"],
        )

    def test_data_parallel_controller_loads_plugins_before_controller_init(self):
        from sglang.srt.managers import data_parallel_controller as module

        events = []

        class DummyDataParallelController:
            def __init__(self, server_args, port_args, run_scheduler_process_func):
                events.append(("controller_init", run_scheduler_process_func))
                self.scheduler_procs = []
                self.max_total_num_tokens = 1024
                self.max_req_input_len = 256

        server_args = SimpleNamespace(
            enable_trace=False,
            node_rank=1,
        )
        pipe_writer = MagicMock()

        with self._patch_common_process_setup(module, events):
            with patch.object(module.faulthandler, "enable"):
                with patch.object(
                    module,
                    "DataParallelController",
                    DummyDataParallelController,
                ):
                    module.run_data_parallel_controller_process(
                        server_args,
                        SimpleNamespace(),
                        pipe_writer,
                        run_scheduler_process_func="scheduler-target",
                    )

        self.assertEqual(
            events, ["load_plugins", ("controller_init", "scheduler-target")]
        )
        pipe_writer.send.assert_called_once()
        self.assertEqual(
            pipe_writer.send.call_args.args[0],
            {
                "status": "ready",
                "max_total_num_tokens": 1024,
                "max_req_input_len": 256,
                module.SCHEDULER_PIDS_ARG: [],
            },
        )


if __name__ == "__main__":
    unittest.main()
