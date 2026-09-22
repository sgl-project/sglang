"""Tests for `entrypoints/early_forkserver.py`.

The pure parts are tested in-process. The start method itself is exercised
end to end in a subprocess on Linux: a real forkserver preloading only this
module (PRELOAD is narrowed in the launcher script), a worker forked from it
and a worker forked from that worker.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from sglang.srt.entrypoints import early_forkserver
from sglang.srt.entrypoints.early_forkserver import merge_env
from sglang.srt.environ import envs
from sglang.test.test_utils import CustomTestCase


class TestMergeEnv(CustomTestCase):
    """A forked child starts with the forkserver's environment; merge_env must
    turn it into what a spawned child would have inherited from the launcher,
    without undoing what the preload imports set."""

    BASE = {"HOME": "/h", "USER_VAR": "user", "DROPPED": "x", "SHARED": "base"}

    def test_a_launcher_write_since_cli_entry_wins(self):
        merged = merge_env(
            current={**self.BASE, "SHARED": "preload"},
            launcher={**self.BASE, "SHARED": "launcher", "NEW": "1"},
            base=self.BASE,
        )
        self.assertEqual(merged["SHARED"], "launcher")
        self.assertEqual(merged["NEW"], "1")

    def test_a_launcher_unset_since_cli_entry_removes_the_variable(self):
        launcher = dict(self.BASE)
        del launcher["DROPPED"]
        merged = merge_env(current=dict(self.BASE), launcher=launcher, base=self.BASE)
        self.assertNotIn("DROPPED", merged)

    def test_what_the_preload_imports_set_stays(self):
        # Neither side of the launcher's history knows CACHE_DIR or the new
        # SHARED value, so both come from the forkserver.
        merged = merge_env(
            current={**self.BASE, "CACHE_DIR": "/tmp/k", "SHARED": "preload"},
            launcher=dict(self.BASE),
            base=self.BASE,
        )
        self.assertEqual(merged["CACHE_DIR"], "/tmp/k")
        self.assertEqual(merged["SHARED"], "preload")


class TestStartMethod(CustomTestCase):
    def test_spawn_unless_start_early_ran_in_this_process(self):
        """`python -m sglang.launch_server` never calls start_early(); choosing
        the forkserver there, with none running, made the first start() hang."""
        with mock.patch.object(early_forkserver, "_started", False):
            self.assertEqual(
                early_forkserver.start_method(enable_memory_saver=False), "spawn"
            )

    def test_memory_saver_falls_back_to_spawn_and_stops_the_server(self):
        with (
            mock.patch.object(early_forkserver, "_started", True),
            mock.patch.object(early_forkserver, "stop") as stop,
            envs.SGLANG_NUMA_BIND_V2.override(True),
        ):
            self.assertEqual(
                early_forkserver.start_method(enable_memory_saver=True), "spawn"
            )
            # Spawned workers keep the numactl path; only the forkserver flips it.
            self.assertTrue(envs.SGLANG_NUMA_BIND_V2.get())
        stop.assert_called_once()

    def test_the_forkserver_selects_in_process_numa_binding(self):
        with (
            mock.patch.object(early_forkserver, "_started", True),
            envs.SGLANG_NUMA_BIND_V2.override(True),
        ):
            self.assertEqual(
                early_forkserver.start_method(enable_memory_saver=False),
                early_forkserver.START_METHOD,
            )
            self.assertFalse(envs.SGLANG_NUMA_BIND_V2.get())


class _FakePynvml:
    def __init__(self):
        self.requested_index = None

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetHandleByIndex(self, index):
        self.requested_index = index
        return object()

    def nvmlDeviceGetCudaComputeCapability(self, handle):
        return (9, 0)

    def nvmlDeviceGetNumGpuCores(self, handle):
        return 132 * 128


class TestNvmlStandIn(CustomTestCase):
    def test_the_torch_ordinal_is_mapped_to_the_nvml_index(self):
        """Under CUDA_VISIBLE_DEVICES the two differ, and answering a probe for
        the wrong GPU would describe another architecture."""
        import torch

        fake = _FakePynvml()
        with (
            mock.patch.dict(sys.modules, {"pynvml": fake}),
            mock.patch.object(torch.cuda, "_get_nvml_device_index", lambda i: 3),
        ):
            device = early_forkserver._nvml_device(0)
        self.assertEqual(fake.requested_index, 3)
        self.assertEqual(device, (9, 0, 132))

    def test_an_attribute_nvml_cannot_supply_raises_by_name(self):
        """Falling back to the real query here initialized CUDA in the
        forkserver and broke every worker forked afterwards."""
        with mock.patch.object(
            early_forkserver,
            "_nvml_device",
            return_value=early_forkserver._NvmlDevice(9, 0, 132),
        ):
            props = early_forkserver._NvmlDeviceProperties(0)
        self.assertEqual(props.multi_processor_count, 132)
        with self.assertRaises(AttributeError) as ctx:
            props.L2_cache_size
        self.assertIn("L2_cache_size", str(ctx.exception))
        # copy/pickle probe dunders and must not read as an unsupported query.
        with self.assertRaises(AttributeError) as ctx:
            props.__deepcopy__
        self.assertNotIn("NVML", str(ctx.exception))


# The launcher of the end-to-end case. Runs as a script, not `-c`: forked
# children re-import __main__ to unpickle their target.
_LAUNCHER = """
import json, multiprocessing as mp, multiprocessing.forkserver as fs, os, resource, sys, time

from sglang.srt.entrypoints import early_forkserver
from sglang.srt.utils.common import get_parent_process, kill_itself_when_parent_died


def grandchild(q):
    q.put({"parent": get_parent_process().pid, "env": os.environ.get("FROM_CHILD"),
           "method": mp.get_start_method()})


def child(q):
    import setproctitle
    setproctitle.setproctitle("sglang::probe")
    kill_itself_when_parent_died()
    os.environ["FROM_CHILD"] = "1"
    gq = mp.Queue()
    g = mp.Process(target=grandchild, args=(gq,))
    g.start()
    report = {
        "pid": os.getpid(), "os_parent": os.getppid(), "parent": get_parent_process().pid,
        "method": mp.get_start_method(),
        "after": os.environ.get("AFTER_START"), "dropped": os.environ.get("DROPPED"),
        "nvml_check": os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK"),
        "nofile": resource.getrlimit(resource.RLIMIT_NOFILE)[0],
        "grandchild": gq.get(timeout=120),
    }
    g.join(60)
    q.put(report)
    time.sleep(600)  # killed by the test through the server's alive pipe


if __name__ == "__main__":
    early_forkserver.PRELOAD = (early_forkserver.__name__,)  # not the worker stack
    early_forkserver.start_early()
    server_pid = fs._forkserver._forkserver_pid
    os.environ["AFTER_START"] = "launcher"
    del os.environ["DROPPED"]
    # set_ulimit() runs after the server started; a spawned worker would inherit it.
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft - 1, hard))
    q = mp.Queue()
    p = mp.Process(target=child, args=(q,))
    p.start()
    report = q.get(timeout=120)
    report["launcher"] = os.getpid()
    report["launcher_nofile"] = soft - 1
    report["server"] = server_pid
    report["comm"] = open(f"/proc/{p.pid}/comm").read().strip()
    # The launcher is the only holder of the server's alive pipe: closing it
    # must end the server although a worker is still running, and the worker,
    # armed with PR_SET_PDEATHSIG against the server, must go with it.
    os.close(fs._forkserver._forkserver_alive_fd)
    report["server_exit"] = os.waitpid(server_pid, 0)[0] == server_pid
    for _ in range(100):
        state = open(f"/proc/{p.pid}/status").read() if os.path.exists(f"/proc/{p.pid}") else ""
        if not state or "State:\\tZ" in state:
            break
        time.sleep(0.1)
    report["worker_gone"] = not state or "State:\\tZ" in state
    print(json.dumps(report))
"""


@unittest.skipUnless(sys.platform == "linux", "forkserver children need Linux here")
class TestForkedWorkers(CustomTestCase):
    """The start method end to end: environment, parentage, nesting, title and
    lifetime of a worker forked from the preloaded server."""

    @classmethod
    def setUpClass(cls):
        env = dict(os.environ, SGLANG_ENABLE_EARLY_FORKSERVER="1", DROPPED="x")
        env.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "launcher.py")
            with open(script, "w") as f:
                f.write(textwrap.dedent(_LAUNCHER))
            res = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True,
                env=env,
                timeout=300,
            )
        assert res.returncode == 0, res.stderr[-3000:]
        cls.report = json.loads(res.stdout.strip().splitlines()[-1])

    def test_the_worker_runs_with_the_launchers_environment(self):
        # Written after start_early(), so only the launcher had it.
        self.assertEqual(self.report["after"], "launcher")
        self.assertIsNone(self.report["dropped"])
        # Set for the forkserver's exec only; the user had it unset.
        self.assertIsNone(self.report["nvml_check"])
        # rlimits are inherited at exec too; the launcher changed one after start.
        self.assertEqual(self.report["nofile"], self.report["launcher_nofile"])

    def test_the_worker_is_forked_from_the_server_and_reports_the_launcher(self):
        r = self.report
        self.assertEqual(r["os_parent"], r["server"])
        self.assertEqual(r["parent"], r["launcher"])
        self.assertEqual(r["method"], early_forkserver.START_METHOD)

    def test_a_nested_start_reuses_the_server_and_sees_the_workers_environment(self):
        g = self.report["grandchild"]
        self.assertEqual(g["method"], early_forkserver.START_METHOD)
        self.assertEqual(g["parent"], self.report["pid"])
        self.assertEqual(g["env"], "1")

    def test_the_worker_can_still_set_its_title(self):
        """The environment rewrite before run() once moved environ from under
        setproctitle, and every worker kept the server's title."""
        self.assertEqual(self.report["comm"], "sglang::probe"[:15])

    def test_the_server_and_its_workers_go_down_with_the_launcher(self):
        self.assertTrue(self.report["server_exit"])
        self.assertTrue(self.report["worker_gone"])


if __name__ == "__main__":
    unittest.main()
