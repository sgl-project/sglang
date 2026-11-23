"""Unit tests for srt/utils/process.py: set_gpu_proc_affinity on a restricted cpuset."""

import unittest
from types import SimpleNamespace
from unittest import mock

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


def _split_siblings(cores):
    """Linux's usual numbering: core c's threads are c and c + cores."""
    return lambda cpu: f"{cpu % cores},{cpu % cores + cores}"


def _adjacent_siblings(cpu):
    """Core k's threads are 2k and 2k + 1."""
    return f"{cpu - cpu % 2}-{cpu - cpu % 2 + 1}"


def _bind(gpu_id, *, allowed, siblings, tp_size=8, server_args=None, numa_node=None):
    proc = _FakeProcess(allowed)
    parallel = SimpleNamespace(nnodes=1, pp_size=1, tp_size=tp_size)

    def node_of(server_args, gpu):
        assert gpu == gpu_id, "only this rank's own GPU may be looked up"
        return numa_node

    with (
        mock.patch.object(process.psutil, "Process", return_value=proc),
        mock.patch.object(process, "get_parallel", return_value=parallel),
        mock.patch.object(process, "_thread_siblings", side_effect=siblings),
        mock.patch.object(
            numa_utils, "get_numa_node_if_available", side_effect=node_of
        ),
    ):
        process.set_gpu_proc_affinity(gpu_id, server_args)
    return proc.bound


def _owners(slices, core_of):
    owners = {}
    for g, s in enumerate(slices):
        for cpu in s:
            owners.setdefault(core_of(cpu), set()).add(g)
    return owners


class TestSetGpuProcAffinity(CustomTestCase):
    def test_binds_within_a_cpuset_smaller_than_the_host(self):
        """A container granted CPUs 0-95 of a 236-CPU host: each rank binds a disjoint
        slice of the CPUs it may use, not host core ids outside its cpuset."""
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
        whole cores, so no two ranks share a physical core."""
        allowed = list(range(48)) + list(range(118, 166))
        slices = [
            _bind(g, allowed=allowed, siblings=_split_siblings(118)) for g in range(8)
        ]
        for s in slices:
            self.assertEqual(len(s), 12)
        owners = _owners(slices, lambda cpu: cpu % 118)
        self.assertEqual(len(owners), 48)
        self.assertTrue(all(len(ranks) == 1 for ranks in owners.values()))

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
        owners = _owners(slices, lambda cpu: cpu % 128)
        self.assertTrue(all(len(ranks) == 1 for ranks in owners.values()))

    def test_groups_adjacent_smt_siblings_by_the_kernel_topology(self):
        """An 8-core host numbering core k's threads 2k and 2k + 1, TP2 with CPUs 0-3, 12
        and 14: CPUs 2 and 3 share a core, so they stay on one rank."""
        allowed = [0, 1, 2, 3, 12, 14]
        slices = [
            _bind(g, allowed=allowed, siblings=_adjacent_siblings, tp_size=2)
            for g in range(2)
        ]
        self.assertEqual(slices, [[0, 1, 2, 3], [12, 14]])

    def test_keeps_the_mask_numa_binding_gave_the_rank(self):
        """A rank NUMA binding placed on its GPU's node keeps that node's CPUs: ranks on
        different nodes start from different masks, so splitting by rank would overlap."""
        node0 = list(range(64)) + list(range(128, 192))
        bound = _bind(
            1,
            allowed=node0,
            siblings=_split_siblings(128),
            server_args=SimpleNamespace(numa_node=None),
            numa_node=0,
        )
        self.assertEqual(bound, node0)

    def test_splits_when_the_rank_has_no_numa_node(self):
        """Without a NUMA node for the GPU, every rank starts from the same CPUs and splits them."""
        bound = _bind(
            3,
            allowed=list(range(256)),
            siblings=_split_siblings(128),
            server_args=SimpleNamespace(numa_node=None),
            numa_node=None,
        )
        self.assertEqual(bound, list(range(48, 64)) + list(range(176, 192)))


if __name__ == "__main__":
    unittest.main()
