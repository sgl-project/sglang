"""Unit tests for srt/utils/process.py: set_gpu_proc_affinity on a restricted cpuset."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.utils import process
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


def _bind(gpu_id, *, allowed, physical, logical, tp_size=8):
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
        process.set_gpu_proc_affinity(gpu_id)
    return proc.bound


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


if __name__ == "__main__":
    unittest.main()
