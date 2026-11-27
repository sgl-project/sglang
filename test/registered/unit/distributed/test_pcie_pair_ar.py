"""Unit tests for the PCIe pair all-reduce adapter: the gates that decide whether a
reduction reaches the kernel. Each fails silently in production (the reduction
falls back to NCCL), so they are pinned here. The kernel itself is tested on 4
GPUs in test/registered/kernels/ops/communication/test_pcie_pair_all_reduce.py.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.distributed.device_communicators import pcie_pair_ar
from sglang.srt.distributed.device_communicators.pcie_pair_ar import (
    PciePairCommunicator,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROW = 5120 * 2


class TestEligibleGroup(CustomTestCase):
    def test_off_by_default(self):
        self.assertFalse(pcie_pair_ar.eligible_group(group_name="tp", world_size=4))

    def test_tp_group_of_four_only(self):
        with envs.SGLANG_ENABLE_PCIE_PAIR_ALLREDUCE.override(True):
            self.assertTrue(pcie_pair_ar.eligible_group(group_name="tp", world_size=4))
            for world_size in (1, 2, 3, 8):
                self.assertFalse(
                    pcie_pair_ar.eligible_group(group_name="tp", world_size=world_size)
                )
            for name in ("attention_tp", "moe_tp", None):
                self.assertFalse(
                    pcie_pair_ar.eligible_group(group_name=name, world_size=4)
                )


class TestPlan(CustomTestCase):
    def test_measured_plan(self):
        self.assertEqual(pcie_pair_ar.plan(1 * ROW), (1, True))
        self.assertEqual(pcie_pair_ar.plan(2 * ROW), (1, True))
        self.assertEqual(pcie_pair_ar.plan(3 * ROW), (1, False))
        self.assertEqual(pcie_pair_ar.plan(6 * ROW), (2, False))
        self.assertEqual(pcie_pair_ar.plan(24 * ROW), (4, False))


class TestTopology(CustomTestCase):
    def _nvml(self, levels):
        nvml = MagicMock()
        nvml.NVML_TOPOLOGY_MULTIPLE = 20
        nvml.nvmlDeviceGetHandleByPciBusId.side_effect = lambda b: b
        nvml.nvmlDeviceGetTopologyCommonAncestor.side_effect = lambda a, b: levels[
            frozenset((a, b))
        ]
        return nvml

    def _check(self, same_switch, across):
        buses = ["a", "b", "c", "d"]
        levels = {}
        for i in range(4):
            for j in range(4):
                if i != j:
                    levels[frozenset((buses[i], buses[j]))] = (
                        same_switch if i ^ j == 1 else across
                    )
        with patch.dict(sys.modules, {"pynvml": self._nvml(levels)}):
            return pcie_pair_ar.is_two_switch_pairs(buses)

    def test_two_switch_pairs(self):
        # PIX (10) within a pair, SYS (50) across.
        self.assertTrue(self._check(same_switch=10, across=50))

    def test_all_behind_one_switch(self):
        self.assertFalse(self._check(same_switch=10, across=10))

    def test_no_shared_switch(self):
        self.assertFalse(self._check(same_switch=50, across=50))

    def test_no_nvml(self):
        with patch.dict(sys.modules, {"pynvml": None}):
            self.assertIsNone(pcie_pair_ar.is_two_switch_pairs(["a", "b", "c", "d"]))


class TestShouldUse(CustomTestCase):
    def _comm(self, max_bytes=24 * ROW):
        comm = PciePairCommunicator.__new__(PciePairCommunicator)
        comm.disabled = False
        comm.max_bytes = max_bytes
        return comm

    def _tensor(self, numel, dtype=torch.bfloat16, ptr=0, contiguous=True):
        return SimpleNamespace(
            is_cuda=True,
            dtype=dtype,
            numel=lambda: numel,
            is_contiguous=lambda: contiguous,
            data_ptr=lambda: ptr,
        )

    def test_decode_sized_bf16(self):
        self.assertTrue(self._comm().should_pcie_pair_ar(self._tensor(6 * 5120)))

    def test_size_bound(self):
        comm = self._comm()
        self.assertTrue(comm.should_pcie_pair_ar(self._tensor(24 * 5120)))
        self.assertFalse(comm.should_pcie_pair_ar(self._tensor(25 * 5120)))

    def test_rejects(self):
        comm = self._comm()
        self.assertFalse(comm.should_pcie_pair_ar(self._tensor(5120, torch.float16)))
        self.assertFalse(comm.should_pcie_pair_ar(self._tensor(5124)))  # % 8
        self.assertFalse(comm.should_pcie_pair_ar(self._tensor(5120, ptr=8)))
        self.assertFalse(comm.should_pcie_pair_ar(self._tensor(5120, contiguous=False)))
        comm.disabled = True
        self.assertFalse(comm.should_pcie_pair_ar(self._tensor(5120)))


if __name__ == "__main__":
    unittest.main()
