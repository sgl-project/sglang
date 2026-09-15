"""``set_gpu_proc_affinity``: a rank may only bind to the CPUs the process was
actually given, and the slices must partition that allocation rather than the
machine. Under Slurm or a cgroup holding a subset -- typically the socket the
allocated GPUs sit on -- CPU ids derived from the machine's core count name CPUs
this process does not own, which psutil rejects. Pure arithmetic over patched
probes, so no GPU is needed.

The ROCm image sets ``SGLANG_SET_CPU_AFFINITY=1`` (docker/rocm.Dockerfile), so
this path is live for AMD users who never opted into it.
"""

import contextlib
import unittest
from unittest.mock import patch

from sglang.srt.utils.common import set_gpu_proc_affinity
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeProcess:
    """Stands in for ``psutil.Process``, keeping its CPU eligibility rule.

    Real psutil raises when asked to bind outside the cgroup. Reproducing that
    here is what makes these tests fail against slices built from the machine
    instead of from the allocation.
    """

    def __init__(self, allowed):
        self._allowed = set(allowed)
        self.bound = None

    def cpu_affinity(self, ids=None):
        if ids is None:
            return list(self.bound or sorted(self._allowed))
        if not ids:
            raise ValueError("cpu_affinity() got an empty CPU list")
        for cpu in ids:
            if cpu not in self._allowed:
                raise ValueError(
                    f"CPU number {cpu} is not eligible; choose between "
                    f"{sorted(self._allowed)}"
                )
        self.bound = sorted(ids)
        return self.bound


@contextlib.contextmanager
def machine(n_physical, hyperthreading, allowed):
    """Present a machine of ``n_physical`` cores exposing only ``allowed``.

    On a hyperthreaded topology the second sibling of core ``c`` is
    ``c + n_physical``, which is how the production code recognises siblings.
    """
    n_logical = n_physical * 2 if hyperthreading else n_physical

    def cpu_count(logical=True):
        return n_logical if logical else n_physical

    proc = FakeProcess(allowed)
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch("os.sched_getaffinity", return_value=set(allowed), create=True)
        )
        stack.enter_context(patch("psutil.cpu_count", side_effect=cpu_count))
        stack.enter_context(patch("psutil.Process", return_value=proc))
        yield proc


def bind(n_physical, hyperthreading, allowed, gpu_id, tp_size, pp_size=1, nnodes=1):
    with machine(n_physical, hyperthreading, allowed) as proc:
        set_gpu_proc_affinity(pp_size, tp_size, nnodes, gpu_id)
        return proc.bound


# A 64-core hyperthreaded node: cores 0-63, their siblings 64-127.
HT_NODE = dict(n_physical=64, hyperthreading=True)
# The upper socket plus its siblings, which is what Slurm hands out when the
# allocated GPUs sit on socket 1.
UPPER_SOCKET = list(range(32, 64)) + list(range(96, 128))


class TestSetGpuProcAffinity(CustomTestCase):
    def test_whole_machine_hyperthreaded_binds_slice_and_siblings(self):
        """Unrestricted allocation must keep its existing behaviour."""
        self.assertEqual(
            bind(**HT_NODE, allowed=range(128), gpu_id=1, tp_size=8),
            list(range(8, 16)) + list(range(72, 80)),
        )

    def test_whole_machine_without_hyperthreading_binds_contiguous_slice(self):
        self.assertEqual(
            bind(n_physical=64, hyperthreading=False, allowed=range(64),
                 gpu_id=3, tp_size=8),
            list(range(24, 32)),
        )

    def test_offset_cpuset_binds_within_the_allocation(self):
        """The Slurm second-socket case: every rank stays inside the cpuset."""
        first = bind(**HT_NODE, allowed=UPPER_SOCKET, gpu_id=0, tp_size=4)
        last = bind(**HT_NODE, allowed=UPPER_SOCKET, gpu_id=3, tp_size=4)
        self.assertEqual(first, list(range(32, 40)) + list(range(96, 104)))
        self.assertEqual(last, list(range(56, 64)) + list(range(120, 128)))

    def test_offset_cpuset_slices_are_disjoint_and_equal_width(self):
        """Intersecting machine-wide ranges leaves some ranks unbound and
        others oversized; partitioning the allocation cannot."""
        slices = [
            bind(**HT_NODE, allowed=UPPER_SOCKET, gpu_id=g, tp_size=4)
            for g in range(4)
        ]
        widths = {len(s) for s in slices}
        self.assertEqual(widths, {16}, f"uneven slice widths: {widths}")
        seen = set()
        for s in slices:
            self.assertTrue(set(s).isdisjoint(seen), f"overlapping slice {s}")
            seen |= set(s)

    def test_partial_cpuset_from_zero_does_not_oversubscribe(self):
        """Rank 0 gets its share of a small allocation, not all of it."""
        allowed = list(range(16)) + list(range(64, 80))
        self.assertEqual(
            bind(**HT_NODE, allowed=allowed, gpu_id=0, tp_size=4),
            [0, 1, 2, 3, 64, 65, 66, 67],
        )

    def test_non_contiguous_cpuset_slices_allowed_cpus_in_order(self):
        allowed = (
            list(range(8)) + list(range(32, 40))
            + list(range(64, 72)) + list(range(96, 104))
        )
        self.assertEqual(
            bind(**HT_NODE, allowed=allowed, gpu_id=2, tp_size=4),
            [32, 33, 34, 35, 96, 97, 98, 99],
        )

    def test_hyperthreaded_machine_without_sibling_cpus_skips_siblings(self):
        """``--hint=nomultithread`` leaves the siblings out of the cpuset."""
        self.assertEqual(
            bind(**HT_NODE, allowed=range(16), gpu_id=0, tp_size=4), [0, 1, 2, 3]
        )

    def test_sibling_only_cpuset_is_treated_as_the_pool(self):
        self.assertEqual(
            bind(**HT_NODE, allowed=range(64, 96), gpu_id=1, tp_size=4),
            list(range(72, 80)),
        )

    def test_uneven_cpu_count_yields_disjoint_nonempty_slices(self):
        allowed = list(range(20, 30))
        slices = [
            bind(n_physical=64, hyperthreading=False, allowed=allowed,
                 gpu_id=g, tp_size=4)
            for g in range(4)
        ]
        seen = set()
        for s in slices:
            self.assertTrue(s, "a rank was left with no CPUs")
            self.assertTrue(set(s) <= set(allowed), f"{s} escapes the cpuset")
            self.assertTrue(set(s).isdisjoint(seen), f"overlapping slice {s}")
            seen |= set(s)

    def test_fewer_allowed_cpus_than_ranks_binds_at_least_one(self):
        for gpu_id in range(4):
            with self.subTest(gpu_id=gpu_id):
                got = bind(n_physical=64, hyperthreading=False, allowed=[30, 31],
                           gpu_id=gpu_id, tp_size=4)
                self.assertTrue(got, "empty CPU list would raise in psutil")
                self.assertTrue(set(got) <= {30, 31})

    def test_parallelism_matrix_determines_slice_width(self):
        # (tp_size, pp_size, nnodes) -> ranks sharing this node's 32 cores.
        cases = [((8, 1, 1), 8), ((16, 1, 2), 8), ((8, 2, 2), 8),
                 ((4, 2, 4), 2), ((4, 4, 1), 4), ((1, 1, 1), 1)]
        for (tp_size, pp_size, nnodes), ranks_per_node in cases:
            with self.subTest(tp=tp_size, pp=pp_size, nnodes=nnodes):
                got = bind(**HT_NODE, allowed=UPPER_SOCKET, gpu_id=0,
                           tp_size=tp_size, pp_size=pp_size, nnodes=nnodes)
                self.assertEqual(len(got), (32 // ranks_per_node) * 2)
                self.assertTrue(set(got) <= set(UPPER_SOCKET))

    def test_data_parallel_gpu_id_wraps_to_the_first_slice(self):
        """More GPUs than TP ranks on a node: rank 5 reuses rank 1's slice."""
        self.assertEqual(
            bind(**HT_NODE, allowed=UPPER_SOCKET, gpu_id=5, tp_size=4),
            bind(**HT_NODE, allowed=UPPER_SOCKET, gpu_id=1, tp_size=4),
        )


if __name__ == "__main__":
    unittest.main()
