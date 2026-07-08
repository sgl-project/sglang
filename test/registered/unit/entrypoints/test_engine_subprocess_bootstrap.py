"""Unit tests for Engine subprocess bootstrap wiring."""

import copy
import sys
import types
import unittest
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
    stub.__path__ = []
    stub.__getattr__ = lambda name: MagicMock(name=f"sgl_kernel.{name}")
    sys.modules["sgl_kernel"] = stub


_maybe_stub_sgl_kernel()

from sglang.srt.entrypoints import engine
from sglang.srt.utils.subprocess_bootstrap import (
    DEFAULT_DATA_PARALLEL_CONTROLLER_TARGET,
    DEFAULT_DETOKENIZER_TARGET,
    DEFAULT_MULTI_DETOKENIZER_ROUTER_TARGET,
    DEFAULT_SCHEDULER_TARGET,
    run_subprocess_target,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class DummyProcess:
    created = []
    _next_pid = 1000

    def __init__(self, target, args=(), kwargs=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.pid = None
        self.started = False
        self.exitcode = None
        DummyProcess.created.append(self)

    def start(self):
        self.started = True
        self.pid = DummyProcess._next_pid
        DummyProcess._next_pid += 1
        self.started_args = tuple(
            copy.deepcopy(arg) if isinstance(arg, SimpleNamespace) else arg
            for arg in self.args
        )
        self.started_kwargs = copy.deepcopy(self.kwargs)

    def join(self):
        return None


class TestEngineSubprocessBootstrap(unittest.TestCase):
    def setUp(self):
        DummyProcess.created = []
        DummyProcess._next_pid = 1000

    def test_dp_controller_process_uses_bootstrap_target(self):
        server_args = SimpleNamespace(dp_size=2)
        port_args = SimpleNamespace()

        with patch.object(engine.mp, "Process", DummyProcess):
            scheduler_init_result, scheduler_procs = Engine._launch_scheduler_processes(
                server_args,
                port_args,
                DEFAULT_SCHEDULER_TARGET,
            )

        self.assertEqual(scheduler_init_result.all_child_pids, [1000])
        self.assertEqual(scheduler_procs, DummyProcess.created)
        proc = DummyProcess.created[0]
        self.assertIs(proc.target, run_subprocess_target)
        self.assertEqual(proc.started_args[0], DEFAULT_DATA_PARALLEL_CONTROLLER_TARGET)
        self.assertEqual(proc.started_args[1].dp_size, 2)
        self.assertEqual(proc.started_args[2], port_args)
        self.assertEqual(proc.started_args[4], DEFAULT_SCHEDULER_TARGET)

    def test_single_detokenizer_process_uses_bootstrap_target(self):
        server_args = SimpleNamespace(detokenizer_worker_num=1)
        port_args = SimpleNamespace(detokenizer_ipc_name="ipc://detokenizer")

        with patch.object(engine.mp, "Process", DummyProcess):
            processes, names = Engine._launch_detokenizer_subprocesses(
                server_args,
                port_args,
                DEFAULT_DETOKENIZER_TARGET,
            )

        self.assertEqual(processes, DummyProcess.created)
        self.assertEqual(names, ["detokenizer"])
        proc = DummyProcess.created[0]
        self.assertIs(proc.target, run_subprocess_target)
        self.assertEqual(proc.started_args[0], DEFAULT_DETOKENIZER_TARGET)
        self.assertEqual(proc.started_args[1].detokenizer_worker_num, 1)
        self.assertEqual(proc.started_args[2].detokenizer_ipc_name, "ipc://detokenizer")

    def test_multi_detokenizer_router_process_uses_bootstrap_target(self):
        server_args = SimpleNamespace(detokenizer_worker_num=2)
        port_args = SimpleNamespace(detokenizer_ipc_name="ipc://router")

        with patch.object(engine.mp, "Process", DummyProcess):
            processes, names = Engine._launch_detokenizer_subprocesses(
                server_args,
                port_args,
                DEFAULT_DETOKENIZER_TARGET,
            )

        self.assertEqual(processes, DummyProcess.created)
        self.assertEqual(
            names, ["detokenizer_0", "detokenizer_1", "detokenizer_router"]
        )

        worker_procs = DummyProcess.created[:2]
        for proc in worker_procs:
            self.assertIs(proc.target, run_subprocess_target)
            self.assertEqual(proc.started_args[0], DEFAULT_DETOKENIZER_TARGET)
            self.assertNotEqual(
                proc.started_args[2].detokenizer_ipc_name, "ipc://router"
            )

        router_proc = DummyProcess.created[2]
        self.assertIs(router_proc.target, run_subprocess_target)
        self.assertEqual(
            router_proc.started_args[0], DEFAULT_MULTI_DETOKENIZER_ROUTER_TARGET
        )
        self.assertEqual(len(router_proc.started_args[1]), 2)
        self.assertEqual(router_proc.started_args[2].detokenizer_worker_num, 2)
        self.assertEqual(
            router_proc.started_args[3].detokenizer_ipc_name, "ipc://router"
        )

    def test_callable_detokenizer_override_is_preserved(self):
        def custom_detokenizer(server_args, port_args):
            return None

        server_args = SimpleNamespace(detokenizer_worker_num=1)
        port_args = SimpleNamespace(detokenizer_ipc_name="ipc://detokenizer")

        with patch.object(engine.mp, "Process", DummyProcess):
            Engine._launch_detokenizer_subprocesses(
                server_args,
                port_args,
                custom_detokenizer,
            )

        proc = DummyProcess.created[0]
        self.assertIs(proc.target, custom_detokenizer)
        self.assertEqual(proc.started_args[0].detokenizer_worker_num, 1)
        self.assertEqual(proc.started_args[1].detokenizer_ipc_name, "ipc://detokenizer")


Engine = engine.Engine


if __name__ == "__main__":
    unittest.main()
