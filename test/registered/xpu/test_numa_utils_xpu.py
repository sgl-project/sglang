"""
XPU NUMA-binding tests for sglang.srt.utils.numa_utils.

Covers the Intel XPU path added to numa_utils (sysfs PCI topology lookup
instead of NVML, ZE_AFFINITY_MASK logical->physical remap). All cases are
mock-based, so they need no XPU hardware and run in XPU CI as well as
locally. The CUDA/CPU cases live in test/registered/utils/test_numa_utils.py.

Usage:
python3 -m unittest test_numa_utils_xpu
"""

import os
import unittest
from unittest.mock import mock_open, patch

from sglang.srt.utils.numa_utils import (
    _is_numa_available,
    _list_xpu_pci_addresses,
    _query_numa_node_for_gpu,
    _query_numa_node_for_xpu,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_xpu_ci

# Pure mock-based; register on XPU (real target) and CPU (so it also runs in
# the fast CPU lane without XPU hardware).
register_cpu_ci(est_time=7, suite="base-a-test-cpu")
register_xpu_ci(est_time=10, suite="stage-a-test-1-gpu-xpu")


def _fake_sysfs_reader(pci_meta: dict):
    """Return an ``open`` replacement that serves sysfs files from ``pci_meta``.

    ``pci_meta`` maps a sysfs file path to its text contents (e.g.
    ``/sys/bus/pci/devices/0000:18:00.0/vendor`` -> ``"0x8086\n"``). Paths not
    present raise ``FileNotFoundError`` (an ``OSError`` subclass), matching how
    the real sysfs behaves for absent attributes.
    """

    def _open(path, *args, **kwargs):
        if path in pci_meta:
            return mock_open(read_data=pci_meta[path])(path, *args, **kwargs)
        raise FileNotFoundError(path)

    return _open


class TestIsNumaAvailableXpu(unittest.TestCase):
    """_is_numa_available must admit XPU just like CUDA."""

    @patch("sglang.srt.utils.numa_utils._is_xpu", False)
    @patch("sglang.srt.utils.numa_utils._is_cuda", False)
    def test_returns_false_when_not_cuda_or_xpu(self):
        self.assertFalse(_is_numa_available())

    @patch("sglang.srt.utils.numa_utils._can_set_mempolicy", return_value=True)
    @patch("sglang.srt.utils.numa_utils.shutil.which", return_value="/usr/bin/numactl")
    @patch("sglang.srt.utils.numa_utils._is_xpu", True)
    @patch("sglang.srt.utils.numa_utils._is_cuda", False)
    @patch("os.path.isdir", return_value=True)
    def test_returns_true_on_xpu_numa_system(
        self, _mock_isdir, _mock_which, _mock_mempolicy
    ):
        # A regression that dropped `or _is_xpu` from the availability gate
        # would make this return False on XPU.
        self.assertTrue(_is_numa_available())


class TestListXpuPciAddresses(unittest.TestCase):
    """_list_xpu_pci_addresses filters to Intel display controllers and sorts."""

    def _run(self, devices: dict, present_dirs: list):
        """devices: basename -> (vendor, class). present_dirs: glob result."""
        pci_meta = {}
        for name, (vendor, cls) in devices.items():
            base = f"/sys/bus/pci/devices/{name}"
            pci_meta[f"{base}/vendor"] = vendor + "\n"
            pci_meta[f"{base}/class"] = cls + "\n"
        with patch(
            "sglang.srt.utils.numa_utils.glob.glob",
            return_value=[f"/sys/bus/pci/devices/{d}" for d in present_dirs],
        ), patch("builtins.open", _fake_sysfs_reader(pci_meta)):
            return _list_xpu_pci_addresses()

    def test_keeps_only_intel_display_controllers_sorted(self):
        # Intel GPU (0x8086 + class 0x03xx) kept; NVIDIA GPU and an Intel
        # non-display device (e.g. host bridge 0x06) dropped. Result sorted.
        devices = {
            "0000:29:00.0": ("0x8086", "0x030000"),  # Intel GPU  -> keep
            "0000:18:00.0": ("0x8086", "0x038000"),  # Intel GPU  -> keep
            "0000:3a:00.0": ("0x10de", "0x030000"),  # NVIDIA GPU -> drop
            "0000:00:00.0": ("0x8086", "0x060000"),  # Intel host bridge -> drop
        }
        result = self._run(devices, list(devices))
        self.assertEqual(result, ["0000:18:00.0", "0000:29:00.0"])

    def test_skips_devices_without_sysfs_attrs(self):
        # A device whose vendor/class files are absent must be skipped, not crash.
        with patch(
            "sglang.srt.utils.numa_utils.glob.glob",
            return_value=["/sys/bus/pci/devices/0000:18:00.0"],
        ), patch("builtins.open", _fake_sysfs_reader({})):
            self.assertEqual(_list_xpu_pci_addresses(), [])


class TestQueryNumaNodeForXpu(unittest.TestCase):
    """_query_numa_node_for_xpu: ZE_AFFINITY_MASK remap + numa_node read."""

    # A stable 4-GPU sysfs layout: device index i -> numa node NODES[i].
    ADDRS = [
        "0000:18:00.0",
        "0000:29:00.0",
        "0000:3a:00.0",
        "0000:5c:00.0",
    ]
    NODES = {
        "0000:18:00.0": "0",
        "0000:29:00.0": "0",
        "0000:3a:00.0": "1",
        "0000:5c:00.0": "-1",  # no NUMA affinity
    }

    def _query(self, device_id, mask=None):
        pci_meta = {
            f"/sys/bus/pci/devices/{a}/numa_node": self.NODES[a] + "\n"
            for a in self.ADDRS
        }
        env = {} if mask is None else {"ZE_AFFINITY_MASK": mask}
        with patch(
            "sglang.srt.utils.numa_utils._list_xpu_pci_addresses",
            return_value=list(self.ADDRS),
        ), patch("builtins.open", _fake_sysfs_reader(pci_meta)), patch.dict(
            os.environ, env, clear=False
        ):
            if mask is None:
                os.environ.pop("ZE_AFFINITY_MASK", None)
            return _query_numa_node_for_xpu(device_id)

    def test_plain_device_index(self):
        self.assertEqual(self._query(0), [0])
        self.assertEqual(self._query(2), [1])

    def test_negative_node_returns_empty(self):
        # Kernel reports -1 for a device with no NUMA affinity -> [].
        self.assertEqual(self._query(3), [])

    def test_affinity_mask_remaps_logical_to_physical(self):
        # ZE_AFFINITY_MASK="3,0": logical 0 -> physical GPU 3 (node -1 -> []),
        # logical 1 -> physical GPU 0 (node 0). This is the core derived
        # property: the mask must reorder the device list before indexing.
        self.assertEqual(self._query(0, mask="3,0"), [])
        self.assertEqual(self._query(1, mask="3,0"), [0])
        # logical 0 under "2,1" -> physical GPU 2 (node 1).
        self.assertEqual(self._query(0, mask="2,1"), [1])

    def test_device_id_out_of_range_returns_empty(self):
        self.assertEqual(self._query(9), [])
        # Also out of range relative to a shortened mask.
        self.assertEqual(self._query(2, mask="0,1"), [])

    def test_malformed_mask_falls_back_to_unmasked_order(self):
        # Composite "x.y" entries are unparsable -> keep unmasked order,
        # so device 2 still resolves to node 1.
        self.assertEqual(self._query(2, mask="0.0,1.0"), [1])

    def test_no_intel_gpus_returns_empty(self):
        with patch(
            "sglang.srt.utils.numa_utils._list_xpu_pci_addresses", return_value=[]
        ):
            self.assertEqual(_query_numa_node_for_xpu(0), [])

    @patch("sglang.srt.utils.numa_utils._is_xpu", True)
    def test_query_numa_node_for_gpu_delegates_to_xpu(self):
        # On XPU, _query_numa_node_for_gpu must route to the XPU sysfs path
        # (not pynvml). A regression dropping the delegation would call NVML.
        with patch(
            "sglang.srt.utils.numa_utils._query_numa_node_for_xpu",
            return_value=[2],
        ) as mock_xpu:
            self.assertEqual(_query_numa_node_for_gpu(5), [2])
            mock_xpu.assert_called_once_with(5)


if __name__ == "__main__":
    unittest.main()
