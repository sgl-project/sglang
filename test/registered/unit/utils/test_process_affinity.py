"""Unit tests for srt/utils/process.py: the launcher's CPU plan and set_gpu_proc_affinity."""

import asyncio
import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.utils import device, numa_utils, process
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeProcess:
    """psutil.Process stand-in with sched_setaffinity's semantics: a mask is clipped to
    the cpuset, and one with no CPU inside it is rejected."""

    def __init__(self, allowed, denied=False):
        self.allowed = list(allowed)
        self.bound = list(allowed)
        self.denied = denied

    def cpu_affinity(self, cpus=None):
        if cpus is None:
            return list(self.bound)
        if self.denied:
            raise process.psutil.AccessDenied()
        inside = sorted(c for c in cpus if c in self.allowed)
        if not inside:
            raise ValueError(f"CPU number {max(cpus)} is not eligible")
        self.bound = inside


def _split_siblings(cores):
    """Linux's usual numbering: core c's threads are c and c + cores."""
    return lambda cpu: f"{cpu % cores},{cpu % cores + cores}"


def _adjacent_siblings(cpu):
    """Core k's threads are 2k and 2k + 1."""
    return f"{cpu - cpu % 2}-{cpu - cpu % 2 + 1}"


# A two-socket host: 128 cores, core c's threads c and c + 128.
_NODE_CPUS = {
    0: set(range(64)) | set(range(128, 192)),
    1: set(range(64, 128)) | set(range(192, 256)),
}


def _bind(
    gpu_id, *, allowed, siblings, tp_size=8, plan=None, numa_node=None, denied=False
):
    """set_gpu_proc_affinity in a scheduler spawned with ``plan`` (none: no plan), its
    GPU on ``numa_node`` of the two-socket host below."""
    proc = _FakeProcess(allowed, denied)
    parallel = SimpleNamespace(nnodes=1, pp_size=1, tp_size=tp_size)
    with (
        mock.patch.object(process.psutil, "Process", return_value=proc),
        mock.patch.object(process, "get_parallel", return_value=parallel),
        mock.patch.object(process, "_thread_siblings", side_effect=siblings),
        mock.patch.object(
            numa_utils, "get_numa_node_if_available", return_value=numa_node
        ),
        mock.patch.object(numa_utils, "_node_cpus", side_effect=_NODE_CPUS.get),
        process.gpu_proc_affinity_env(plan or {}, gpu_id),
    ):
        process.set_gpu_proc_affinity(gpu_id, SimpleNamespace())
    return proc.bound


def _plan(gpu_ids, *, launcher, gpu_nodes, reindex=None):
    """plan_gpu_proc_affinity with GPU g on NUMA node gpu_nodes[g] (None: no node)."""
    reindex = reindex or (lambda g: g)
    to_gpu = {reindex(g): g for g in gpu_ids}

    @contextlib.contextmanager
    def fake_reindex(gpu_id):
        yield reindex(gpu_id)

    with (
        envs.SGLANG_SET_CPU_AFFINITY.override(True),
        mock.patch.object(process.os, "sched_getaffinity", return_value=set(launcher)),
        mock.patch.object(device, "maybe_reindex_device_id", side_effect=fake_reindex),
        mock.patch.object(
            numa_utils,
            "get_numa_node_if_available",
            side_effect=lambda server_args, device_id: gpu_nodes[to_gpu[device_id]],
        ),
        mock.patch.object(numa_utils, "_node_cpus", side_effect=_NODE_CPUS.get),
        mock.patch.object(
            process, "_thread_siblings", side_effect=_split_siblings(128)
        ),
    ):
        return process.plan_gpu_proc_affinity(SimpleNamespace(), gpu_ids)


def _disjoint(slices):
    cpus = [c for s in slices for c in s]
    return len(cpus) == len(set(cpus))


class TestPlanGpuProcAffinity(CustomTestCase):
    def test_no_plan_without_cpu_affinity(self):
        self.assertEqual(process.plan_gpu_proc_affinity(SimpleNamespace(), [0, 1]), {})

    def test_splits_each_numa_node_among_the_gpus_on_it(self):
        """TP8 with four GPUs on each 64-core socket: every scheduler gets 16 cores of
        its own node."""
        plan = _plan(range(8), launcher=range(256), gpu_nodes=[0] * 4 + [1] * 4)
        for gpu, cpus in plan.items():
            self.assertEqual(len(cpus), 32)
            self.assertTrue(set(cpus) <= _NODE_CPUS[gpu // 4])
        self.assertTrue(_disjoint(plan.values()))

    def test_a_skipped_bind_shares_the_launchers_cpus(self):
        """One GPU per socket and only socket 0 allowed: GPU 1's bind is skipped, so both
        schedulers start from socket 0 and split it, 32 cores each."""
        plan = _plan([0, 1], launcher=sorted(_NODE_CPUS[0]), gpu_nodes=[0, 1])
        self.assertEqual(len(plan[0]), 64)
        self.assertEqual(len(plan[1]), 64)
        self.assertEqual(set(plan[0]) | set(plan[1]), _NODE_CPUS[0])
        self.assertTrue(_disjoint(plan.values()))

    def test_counts_every_pp_stage_on_a_node(self):
        """PP2/TP2 with GPU nodes [0, 1, 1, 0]: GPUs 0 and 3 split node 0, 1 and 2 node 1."""
        plan = _plan(range(4), launcher=range(256), gpu_nodes=[0, 1, 1, 0])
        for gpu, node in enumerate([0, 1, 1, 0]):
            self.assertEqual(len(plan[gpu]), 64)
            self.assertTrue(set(plan[gpu]) <= _NODE_CPUS[node])
        self.assertTrue(_disjoint(plan.values()))

    def test_a_gpu_without_a_node_takes_the_cores_left(self):
        """TP2 with GPU 0 on node 0 and GPU 1 on no node: GPU 1 starts from every CPU
        but takes node 1's cores, which GPU 0 does not hold."""
        plan = _plan([0, 1], launcher=range(256), gpu_nodes=[0, None])
        self.assertEqual(set(plan[0]), _NODE_CPUS[0])
        self.assertEqual(set(plan[1]), _NODE_CPUS[1])

    def test_splits_the_launchers_cpus_without_numa_nodes(self):
        """A container granted CPUs 0-95 and no NUMA node: eight disjoint 12-CPU slices."""
        plan = _plan(range(8), launcher=range(96), gpu_nodes=[None] * 8)
        self.assertTrue(all(len(cpus) == 12 for cpus in plan.values()))
        self.assertEqual(set().union(*map(set, plan.values())), set(range(96)))

    def test_looks_nodes_up_with_the_device_id_configure_subprocess_uses(self):
        """With one visible device per process the NUMA lookup takes the reindexed id."""
        plan = _plan(
            [0, 1],
            launcher=range(256),
            gpu_nodes=[0, 1],
            reindex=lambda g: 100 + g,
        )
        self.assertTrue(set(plan[0]) <= _NODE_CPUS[0])
        self.assertTrue(set(plan[1]) <= _NODE_CPUS[1])


class TestAllocateCpus(CustomTestCase):
    def test_leaves_a_core_for_each_scheduler_of_a_wider_mask(self):
        """TP4: GPU 0 may use cores 0-7 and the others 0-8. Each of the three gets its
        own cores instead of all sharing core 8."""
        starts = {0: [[c] for c in range(8)]}
        starts.update({g: [[c] for c in range(9)] for g in (1, 2, 3)})
        plan = process.allocate_cpus(starts)
        self.assertEqual(plan, {0: [0, 1, 2], 1: [3, 4], 2: [5, 6], 3: [7, 8]})

    def test_keeps_a_ranks_only_core_for_it(self):
        """TP4 with masks {0-3}, {0}, {1-3}, {1-3}: GPU 1's only core stays its own."""
        starts = {
            0: [[0], [1], [2], [3]],
            1: [[0]],
            2: [[1], [2], [3]],
            3: [[1], [2], [3]],
        }
        plan = process.allocate_cpus(starts)
        self.assertEqual(plan[1], [0])
        self.assertTrue(_disjoint(plan.values()))
        self.assertTrue(all(plan.values()))

    def test_shares_its_cores_when_none_is_left(self):
        plan = process.allocate_cpus({0: [[0]], 1: [[0]]})
        self.assertEqual(plan[0], [0])
        self.assertEqual(plan[1], [0])


class TestCpuShareBoard(CustomTestCase):
    def test_splits_each_host_among_its_schedulers(self):
        """Ray TP2 with one GPU per 32-core socket of each host: each actor gets its
        socket's cores; a host's split waits for both of its actors."""
        socket = lambda first: [[c] for c in range(first, first + 32)]

        async def run():
            board = process.CpuShareBoard()
            first = asyncio.ensure_future(board.register("a", 0, socket(0)))
            await asyncio.sleep(0)
            # the launcher reports its counts after some actors have registered
            await board.expect({"a": 2, "b": 1})
            return await asyncio.gather(
                first,
                board.register("a", 1, socket(32)),
                board.register("b", 0, socket(0)),
            )

        a0, a1, b0 = asyncio.run(run())
        self.assertEqual(a0, list(range(32)))
        self.assertEqual(a1, list(range(32, 64)))
        self.assertEqual(b0, list(range(32)))

    def test_counts_only_the_schedulers_the_launcher_placed(self):
        """TP1 using one of a host's two bundles: the actor does not wait for the other."""

        async def run():
            board = process.CpuShareBoard()
            await board.expect({"a": 1})
            return await board.register("a", 0, [[0], [1]], timeout=1)

        self.assertEqual(asyncio.run(run()), [0, 1])

    def test_gives_up_when_the_hosts_schedulers_do_not_all_report(self):
        async def run():
            board = process.CpuShareBoard()
            await board.expect({"a": 2})
            return await board.register("a", 0, [[0]], timeout=0.01)

        self.assertIsNone(asyncio.run(run()))


class TestSetGpuProcAffinity(CustomTestCase):
    def test_binds_the_cpus_the_launcher_planned(self):
        bound = _bind(
            1,
            allowed=range(256),
            siblings=_split_siblings(128),
            plan={1: [16, 17, 144, 145]},
        )
        self.assertEqual(bound, [16, 17, 144, 145])

    def test_binds_within_a_cpuset_smaller_than_the_host(self):
        """Without a plan, a container granted CPUs 0-95 of a 236-CPU host: each rank
        binds a disjoint slice of the CPUs it may use."""
        allowed = list(range(96))
        slices = [
            _bind(g, allowed=allowed, siblings=_split_siblings(118)) for g in range(8)
        ]
        for s in slices:
            self.assertTrue(set(s) <= set(allowed))
            self.assertEqual(len(s), 12)
        self.assertEqual(len(set().union(*map(set, slices))), 96)

    def test_keeps_both_threads_of_a_core_on_one_rank(self):
        """A cpuset holding cores 0-47 and their second threads 118-165: each rank gets
        whole cores."""
        allowed = list(range(48)) + list(range(118, 166))
        slices = [
            _bind(g, allowed=allowed, siblings=_split_siblings(118)) for g in range(8)
        ]
        for s in slices:
            self.assertEqual(len(s), 12)
        self.assertTrue(_disjoint(slices))
        self.assertEqual({c % 118 for s in slices for c in s}, set(range(48)))

    def test_unrestricted_host_keeps_core_and_sibling_layout(self):
        """With every CPU allowed, rank g keeps physical cores [g*n, (g+1)*n) and their HT siblings."""
        bound = _bind(3, allowed=list(range(256)), siblings=_split_siblings(128))
        self.assertEqual(bound, list(range(48, 64)) + list(range(176, 192)))

    def test_keeps_cores_whose_only_allowed_thread_is_the_upper_sibling(self):
        """CPUs 0-7 and 136-255 of a 128-core host: cores 8-127 are allowed only through
        their second thread, and still count, so every rank gets 16 cores."""
        allowed = list(range(8)) + list(range(136, 256))
        slices = [
            _bind(g, allowed=allowed, siblings=_split_siblings(128)) for g in range(8)
        ]
        for s in slices:
            self.assertEqual(len(s), 16)
        self.assertEqual(sorted(set().union(*map(set, slices))), allowed)

    def test_splits_within_the_numa_node_the_rank_is_bound_to_next(self):
        """Without a plan (Ray), GPU 0 on node 1 slices node 1's CPUs, so the NUMA bind
        that follows has CPUs to keep."""
        bound = _bind(0, allowed=range(256), siblings=_split_siblings(128), numa_node=1)
        self.assertTrue(bound and set(bound) <= _NODE_CPUS[1])

    def test_keeps_the_mask_when_the_cpu_bind_is_denied(self):
        allowed = list(range(256))
        bound = _bind(
            1,
            allowed=allowed,
            siblings=_split_siblings(128),
            plan={1: [16, 17]},
            denied=True,
        )
        self.assertEqual(bound, allowed)

    def test_a_denied_cpu_bind_raises_in_strict_mode(self):
        with (
            envs.SGLANG_CRASH_ON_NUMA_BIND_FAILURE.override(True),
            self.assertRaises(process.psutil.AccessDenied),
        ):
            _bind(
                1,
                allowed=range(256),
                siblings=_split_siblings(128),
                plan={1: [16, 17]},
                denied=True,
            )

    def test_groups_adjacent_smt_siblings_by_the_kernel_topology(self):
        """An 8-core host numbering core k's threads 2k and 2k + 1, TP2 with CPUs 0-3, 12
        and 14: CPUs 2 and 3 share a core, so they stay on one rank."""
        allowed = [0, 1, 2, 3, 12, 14]
        slices = [
            _bind(g, allowed=allowed, siblings=_adjacent_siblings, tp_size=2)
            for g in range(2)
        ]
        self.assertEqual(slices, [[0, 1, 2, 3], [12, 14]])


if __name__ == "__main__":
    unittest.main()
