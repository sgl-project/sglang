"""
CPU-only unit tests for the weight cache daemon destroying its NCCL
communicators before it serves (destroy_nccl_communicators,
WeightCacheDaemon.release_nccl_communicators and run_weight_cache_daemon in
weight_cache/daemon.py).

A daemon builds the model under the engine's process groups, and a group of
more than one rank creates a PyNccl communicator for the collectives of a
forward pass, which a daemon never runs. These tests check the Python side
with fake PyNccl objects (no CUDA, no NCCL): which communicators are destroyed
and in which order, that each one is off its group, unavailable, disabled and
without its handle by the time NCCL is called, that the group's other members
stay, that an error is raised, the log line, and where the entry point calls it
(between two readiness barriers, not for a one-rank daemon, and never serving
after a failed release).
"""

import gc
import re
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.weight_cache import daemon
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Group:
    """The parts of a GroupCoordinator the release uses (weak-referenceable:
    the registry holds weak references)."""

    get_available_pynccl_comm = (
        parallel_state.GroupCoordinator.get_available_pynccl_comm
    )

    def __init__(self, name, pynccl_comm=None):
        self.unique_name = name
        self.cpu_group = object()
        self.device_group = object()
        self.ca_comm = object()
        self.pynccl_comm = pynccl_comm
        if pynccl_comm is not None:
            pynccl_comm.test_group = self


def _pynccl(calls, name, available=True, fail=False):
    """A PyNccl whose ncclCommDestroy records the handle it got and the state
    of the communicator and its group at that moment."""
    comm = object.__new__(PyNcclCommunicator)
    comm.available = available
    comm.disabled = False
    comm.comm = f"{name}-handle"

    def destroy(handle):
        calls.append(
            {
                "handle": handle,
                "on_group": comm.test_group.pynccl_comm is comm,
                "available": comm.available,
                "disabled": comm.disabled,
                "own_handle": comm.comm,
            }
        )
        if fail:
            raise RuntimeError("NCCL error")

    comm.nccl = SimpleNamespace(ncclCommDestroy=destroy)
    return comm


class TestDestroyNcclCommunicators(CustomTestCase):
    def setUp(self):
        registry = mock.patch.dict(parallel_state._groups, clear=True)
        registry.start()
        self.addCleanup(registry.stop)
        self.calls = []

    def _register(self, *groups):
        for group in groups:
            parallel_state._groups[group.unique_name] = weakref.ref(group)

    def test_destroys_each_in_registration_order_once(self):
        world = _Group("world:0")
        tp = _Group("tp:0", _pynccl(self.calls, "tp"))
        pp = _Group("pp:0", _pynccl(self.calls, "pp"))
        self._register(world, tp, pp)
        # The same group under a second name goes once
        parallel_state._groups["alias"] = weakref.ref(tp)

        destroyed = daemon.destroy_nccl_communicators()

        self.assertEqual([c["handle"] for c in self.calls], ["tp-handle", "pp-handle"])
        self.assertEqual(destroyed, ["tp:0", "pp:0"])

    def test_the_communicator_is_unusable_before_nccl_is_called(self):
        comm = _pynccl(self.calls, "tp")
        tp = _Group("tp:0", comm)
        ca, cpu_group, device_group = tp.ca_comm, tp.cpu_group, tp.device_group
        self._register(tp)

        daemon.destroy_nccl_communicators()

        self.assertEqual(
            self.calls,
            [
                {
                    "handle": "tp-handle",
                    "on_group": False,
                    "available": False,
                    "disabled": True,
                    "own_handle": None,
                }
            ],
        )
        self.assertIsNone(tp.pynccl_comm)
        self.assertIsNone(tp.get_available_pynccl_comm())
        with self.assertRaises(RuntimeError):
            with comm.change_state(enable=True):
                pass
        self.assertIs(tp.ca_comm, ca)
        self.assertIs(tp.cpu_group, cpu_group)
        self.assertIs(tp.device_group, device_group)
        # Nothing left to destroy
        self.assertEqual(daemon.destroy_nccl_communicators(), [])
        self.assertEqual(len(self.calls), 1)

    def test_unavailable_absent_or_gone_are_left_alone(self):
        # Without an NCCL library PyNccl has no communicator; one-rank groups
        # (all of a TP1 x PP1 daemon) have no PyNccl at all
        unavailable = _pynccl(self.calls, "tp", available=False)
        tp = _Group("tp:0", unavailable)
        world = _Group("world:0")
        gone = _Group("gone:0", _pynccl(self.calls, "gone"))
        self._register(tp, world, gone)
        del gone
        gc.collect()  # the fake and its group refer to each other

        self.assertEqual(daemon.destroy_nccl_communicators(), [])
        self.assertEqual(self.calls, [])
        self.assertIs(tp.pynccl_comm, unavailable)
        self.assertEqual(unavailable.comm, "tp-handle")

    def test_an_error_is_raised_with_the_communicator_already_unusable(self):
        # A rank that cannot destroy its communicator must not go on to serve:
        # the error ends the daemon instead of being logged
        comm = _pynccl(self.calls, "tp", fail=True)
        tp = _Group("tp:0", comm)
        pp = _Group("pp:0", _pynccl(self.calls, "pp"))
        self._register(tp, pp)

        with self.assertRaisesRegex(RuntimeError, "NCCL error"):
            daemon.destroy_nccl_communicators()

        self.assertEqual([c["handle"] for c in self.calls], ["tp-handle"])
        self.assertIsNone(tp.pynccl_comm)
        self.assertFalse(comm.available)
        self.assertTrue(comm.disabled)
        self.assertIsNone(comm.comm)


class TestReleaseNcclCommunicators(CustomTestCase):
    def _daemon(self):
        wc_daemon = object.__new__(daemon.WeightCacheDaemon)
        wc_daemon.gpu_id = 3
        wc_daemon.tp_rank = 1
        return wc_daemon

    def test_logs_what_it_destroyed_and_how_long_it_took(self):
        with (
            mock.patch.object(
                daemon, "destroy_nccl_communicators", lambda: ["tp:0", "pp:0"]
            ),
            self.assertLogs(daemon.logger, level="INFO") as logs,
        ):
            self._daemon().release_nccl_communicators()
        line = "\n".join(logs.output)
        self.assertRegex(
            line,
            re.escape("[WeightCacheDaemon gpu=3 tp_rank=1] Destroyed the NCCL ")
            + re.escape("communicators of ['tp:0', 'pp:0'], which only loading ")
            + r"used, in \d+\.\d\ds",
        )

    def test_logs_nothing_when_there_was_nothing(self):
        with (
            mock.patch.object(daemon, "destroy_nccl_communicators", lambda: []),
            mock.patch.object(daemon.logger, "info") as info,
        ):
            self._daemon().release_nccl_communicators()
        info.assert_not_called()


class TestDaemonEntryPoint(CustomTestCase):
    def _run(self, world_size, release_error=None):
        calls = []

        class FakeDaemon:
            def __init__(self, **kwargs):
                calls.append(("init", kwargs["gpu_id"], kwargs["tp_rank"]))

            def load(self):
                calls.append("load")

            def release_nccl_communicators(self):
                calls.append("release")
                if release_error is not None:
                    raise release_error

            def serve(self):
                calls.append("serve")

        server_args = SimpleNamespace(weight_cache_timeout=7)
        with (
            mock.patch.object(daemon, "WeightCacheDaemon", FakeDaemon),
            mock.patch.object(
                daemon,
                "_await_cluster_ready_to_serve",
                lambda timeout: calls.append(("barrier", timeout)),
            ),
            mock.patch.object(daemon, "resolving_view", lambda args: args),
            mock.patch.object(daemon.dist, "get_world_size", lambda: world_size),
            mock.patch.object(daemon.logging, "basicConfig"),
            mock.patch("sglang.srt.utils.kill_itself_when_parent_died"),
        ):
            try:
                daemon.run_weight_cache_daemon(
                    server_args, gpu_id=2, tp_rank=1, pp_rank=0
                )
            except Exception as e:
                calls.append(("raised", repr(e)))
        return calls

    def test_releases_between_two_readiness_barriers_then_serves(self):
        self.assertEqual(
            self._run(world_size=4),
            [
                ("init", 2, 1),
                "load",
                ("barrier", 7),
                "release",
                ("barrier", 7),
                "serve",
            ],
        )

    def test_a_one_rank_daemon_goes_straight_to_serving(self):
        self.assertEqual(
            self._run(world_size=1),
            [("init", 2, 1), "load", ("barrier", 7), "serve"],
        )

    def test_a_failed_release_ends_the_daemon_before_it_serves(self):
        self.assertEqual(
            self._run(world_size=2, release_error=RuntimeError("NCCL error")),
            [
                ("init", 2, 1),
                "load",
                ("barrier", 7),
                "release",
                ("raised", "RuntimeError('NCCL error')"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
