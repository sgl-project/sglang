"""Unit tests for srt/utils/process.py: set_gpu_proc_affinity on a restricted cpuset."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.utils import numa_utils, process
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeProcess:
    """psutil.Process stand-in with sched_setaffinity's semantics: a mask is clipped to
    the cpuset, and one with no CPU inside it is rejected."""

    def __init__(self, allowed):
        self.allowed = list(allowed)
        self.bound = list(allowed)

    def cpu_affinity(self, cpus=None):
        if cpus is None:
            return list(self.bound)
        inside = sorted(c for c in cpus if c in self.allowed)
        if not inside:
            raise ValueError(f"CPU number {max(cpus)} is not eligible")
        self.bound = inside


def _bind(gpu_id, *, allowed, physical, logical, tp_size=8, server_args=None):
    proc = _FakeProcess(allowed)
    parallel = SimpleNamespace(nnodes=1, pp_size=1, tp_size=tp_size)
    counts = {False: physical, True: logical}
    with (
        mock.patch.object(process.psutil, "Process", return_value=proc),
        mock.patch.object(
            process.psutil,
            "cpu_count",
            side_effect=lambda logical=True: counts[logical],
        ),
        mock.patch.object(process, "get_parallel", return_value=parallel),
    ):
        process.set_gpu_proc_affinity(gpu_id, server_args)
    return proc.bound


# A two-socket host: 128 cores, GPUs 0-3 on node 0 and 4-7 on node 1.
_NODE_CPUS = {
    0: set(range(64)) | set(range(128, 192)),
    1: set(range(64, 128)) | set(range(192, 256)),
}


def _bind_numa(gpu_id, *, allowed):
    server_args = SimpleNamespace(base_gpu_id=0, gpu_id_step=1, numa_node=None)
    with (
        envs.SGLANG_NUMA_BIND_V2.override(True),
        mock.patch.object(
            numa_utils,
            "get_numa_node_if_available",
            side_effect=lambda server_args, gpu_id: gpu_id // 4,
        ),
        mock.patch.object(numa_utils, "_node_cpus", side_effect=_NODE_CPUS.get),
    ):
        return _bind(
            gpu_id,
            allowed=allowed,
            physical=128,
            logical=256,
            server_args=server_args,
        )


class TestSetGpuProcAffinity(CustomTestCase):
    def test_binds_within_a_cpuset_smaller_than_the_host(self):
        """A container granted CPUs 0-95 of a 236-CPU host: each rank binds a disjoint
        slice of the CPUs it may use, not host core ids outside its cpuset."""
        allowed = list(range(96))
        slices = [
            _bind(g, allowed=allowed, physical=118, logical=236) for g in range(8)
        ]
        for s in slices:
            self.assertTrue(set(s) <= set(allowed))
            self.assertEqual(len(s), 12)
        self.assertEqual(len(set().union(*map(set, slices))), 96)

    def test_keeps_both_threads_of_a_core_on_one_rank(self):
        """A cpuset holding cores 0-47 and their second threads 118-165: each rank gets
        whole cores, so no two ranks share a physical core."""
        allowed = list(range(48)) + list(range(118, 166))
        slices = [
            _bind(g, allowed=allowed, physical=118, logical=236) for g in range(8)
        ]
        owners = {}
        for g, s in enumerate(slices):
            self.assertEqual(len(s), 12)
            for cpu in s:
                owners.setdefault(cpu % 118, set()).add(g)
        self.assertEqual(len(owners), 48)
        self.assertTrue(all(len(ranks) == 1 for ranks in owners.values()))

    def test_unrestricted_host_keeps_core_and_sibling_layout(self):
        """With every CPU allowed, rank g keeps physical cores [g*n, (g+1)*n) and their HT siblings."""
        bound = _bind(3, allowed=list(range(256)), physical=128, logical=256)
        self.assertEqual(bound, list(range(48, 64)) + list(range(176, 192)))

    def test_keeps_cores_whose_only_allowed_thread_is_the_upper_sibling(self):
        """CPUs 0-7 and 136-255 of a 128-core host: cores 8-127 are allowed only through
        their second thread, and still count, so every rank gets 16 cores."""
        allowed = list(range(8)) + list(range(136, 256))
        slices = [
            _bind(g, allowed=allowed, physical=128, logical=256) for g in range(8)
        ]
        owners = {}
        for g, s in enumerate(slices):
            self.assertEqual(len(s), 16)
            for cpu in s:
                owners.setdefault(cpu % 128, set()).add(g)
        self.assertEqual(sorted(set().union(*map(set, slices))), allowed)
        self.assertTrue(all(len(ranks) == 1 for ranks in owners.values()))

    def test_splits_a_numa_node_mask_among_the_ranks_on_that_node(self):
        """Under NUMA binding each rank starts on its GPU's node CPUs; the four ranks
        of a node split its 64 cores, 16 each, instead of 8 as if all eight shared it."""
        for node, cpus in _NODE_CPUS.items():
            gpus = range(4 * node, 4 * node + 4)
            slices = [_bind_numa(g, allowed=sorted(cpus)) for g in gpus]
            for s in slices:
                self.assertEqual(len(s), 32)
            self.assertEqual(set().union(*map(set, slices)), cpus)

    def test_numa_split_needs_a_node_local_mask(self):
        """When the NUMA binding did not apply (the rank still holds every CPU), ranks on
        different nodes would split the same mask, so the host-wide split stays."""
        bound = _bind_numa(3, allowed=list(range(256)))
        self.assertEqual(bound, list(range(48, 64)) + list(range(176, 192)))


if __name__ == "__main__":
    unittest.main()
