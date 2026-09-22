"""
XPU NUMA-binding tests for numa_utils. The CUDA/CPU cases live in
test/registered/utils/test_numa_utils.py.

Usage:
python3 -m unittest test_numa_utils_xpu
"""

import os
import unittest
from unittest.mock import mock_open, patch

import torch

from sglang.srt.utils.numa_utils import (
    _is_numa_available,
    _list_xpu_pci_addresses,
    _query_numa_node_for_gpu,
    _query_numa_node_for_xpu,
    _xpu_pci_address,
    _xpu_visible_device_indices,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_xpu_ci

# CPU too: only TestXpuOrderMatchesRuntime needs hardware.
register_cpu_ci(est_time=7, suite="base-a-test-cpu")
register_xpu_ci(est_time=10, suite="stage-a-test-1-gpu-xpu")


def _fake_sysfs_reader(pci_meta: dict):
    """``open`` replacement serving sysfs path -> contents from ``pci_meta``."""

    def _open(path, *args, **kwargs):
        if path in pci_meta:
            return mock_open(read_data=pci_meta[path])(path, *args, **kwargs)
        raise FileNotFoundError(path)

    return _open


def _fake_dri_glob(render_nodes: list):
    """``glob.glob`` replacement listing ``render_nodes`` under /dev/dri."""

    def _glob(pattern):
        if pattern == "/dev/dri/renderD*":
            return [f"/dev/dri/{node}" for node in render_nodes]
        raise AssertionError(f"unexpected glob pattern {pattern!r}")

    return _glob


def _bdf_from_xpu_uuid(uuid_bytes: bytes):
    """Intel device UUID -> PCI address, or None if not in the Intel layout."""
    # vendor(2 LE) device(2 LE) revision(2 LE) domain(2 LE) bus(1) device(1)
    # function(1) reserved(4) sub-device(1) -- intel/compute-runtime,
    # Device::generateUuidFromPciBusInfo.
    if len(uuid_bytes) != 16 or int.from_bytes(uuid_bytes[0:2], "little") != 0x8086:
        return None
    domain = int.from_bytes(uuid_bytes[6:8], "little")
    bus, device, function = uuid_bytes[8], uuid_bytes[9], uuid_bytes[10]
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function}"


class TestIsNumaAvailableXpu(unittest.TestCase):
    """_is_numa_available must admit XPU just like CUDA."""

    @patch("sglang.srt.utils.numa_utils._can_set_mempolicy", return_value=True)
    @patch("sglang.srt.utils.numa_utils.shutil.which", return_value="/usr/bin/numactl")
    @patch("sglang.srt.utils.numa_utils._is_xpu", True)
    @patch("sglang.srt.utils.numa_utils._is_cuda", False)
    @patch("os.path.isdir", return_value=True)
    def test_returns_true_on_xpu_numa_system(
        self, _mock_isdir, _mock_which, _mock_mempolicy
    ):
        self.assertTrue(_is_numa_available())


class TestListXpuPciAddresses(unittest.TestCase):
    """_list_xpu_pci_addresses filters and orders like Level Zero does."""

    def _run(self, devices: dict, env: dict = None, unopenable: tuple = ()):
        """``devices`` maps a render node to (PCI address, vendor or None)."""
        pci_meta = {
            f"/sys/class/drm/{node}/device/vendor": vendor + "\n"
            for node, (_, vendor) in devices.items()
            if vendor is not None
        }
        realpaths = {
            f"/sys/class/drm/{node}/device": f"/sys/devices/pci0000:00/{address}"
            for node, (address, _) in devices.items()
        }
        with (
            patch(
                "sglang.srt.utils.numa_utils.glob.glob", _fake_dri_glob(list(devices))
            ),
            patch("builtins.open", _fake_sysfs_reader(pci_meta)),
            patch(
                "os.access", lambda path, mode: os.path.basename(path) not in unopenable
            ),
            patch("os.path.realpath", lambda path: realpaths.get(path, path)),
            patch.dict(os.environ, env or {}, clear=False),
        ):
            if not (env or {}).get("ZE_ENABLE_PCI_ID_DEVICE_ORDER"):
                os.environ.pop("ZE_ENABLE_PCI_ID_DEVICE_ORDER", None)
            return _list_xpu_pci_addresses()

    def test_keeps_only_intel_render_nodes_sorted(self):
        devices = {
            "renderD129": ("0000:29:00.0", "0x8086"),  # Intel GPU  -> keep
            "renderD128": ("0000:18:00.0", "0x8086"),  # Intel GPU  -> keep
            "renderD130": ("0000:3a:00.0", "0x10de"),  # NVIDIA GPU -> drop (vendor)
        }
        self.assertEqual(self._run(devices), ["0000:18:00.0", "0000:29:00.0"])

    def test_render_node_this_process_cannot_open_is_skipped(self):
        # A container is handed a subset of /dev/dri; a node it cannot open is not
        # an XPU here, and counting it would shift every logical device index.
        devices = {
            "renderD128": ("0000:18:00.0", "0x8086"),
            "renderD129": ("0000:29:00.0", "0x8086"),
        }
        self.assertEqual(
            self._run(devices, unopenable=("renderD128",)), ["0000:29:00.0"]
        )

    def test_integrated_gpu_is_ordered_last_by_default(self):
        devices = {
            "renderD128": ("0000:00:02.0", "0x8086"),
            "renderD129": ("0000:03:00.0", "0x8086"),
        }
        self.assertEqual(self._run(devices), ["0000:03:00.0", "0000:00:02.0"])

    def test_bus_00_in_another_domain_is_not_integrated(self):
        # Only domain 0000 bus 00 is an iGPU; a second host bridge keeps BDF order.
        devices = {
            "renderD128": ("0001:00:00.0", "0x8086"),
            "renderD129": ("0001:03:00.0", "0x8086"),
        }
        self.assertEqual(self._run(devices), ["0001:00:00.0", "0001:03:00.0"])

    def test_pci_id_device_order_env_restores_plain_bdf_order(self):
        devices = {
            "renderD128": ("0000:00:02.0", "0x8086"),
            "renderD129": ("0000:03:00.0", "0x8086"),
        }
        result = self._run(devices, env={"ZE_ENABLE_PCI_ID_DEVICE_ORDER": "1"})
        self.assertEqual(result, ["0000:00:02.0", "0000:03:00.0"])

    def test_sorts_by_domain_bus_device_function(self):
        # Only case with hex digits in the BDF; catches decimal bus/device parsing.
        devices = {
            "renderD128": ("0001:0a:00.0", "0x8086"),
            "renderD129": ("0000:0a:00.1", "0x8086"),
            "renderD130": ("0000:0a:00.0", "0x8086"),
            "renderD131": ("0000:09:1f.0", "0x8086"),
        }
        self.assertEqual(
            self._run(devices),
            ["0000:09:1f.0", "0000:0a:00.0", "0000:0a:00.1", "0001:0a:00.0"],
        )

    def test_skips_nodes_without_sysfs_attrs(self):
        self.assertEqual(self._run({"renderD128": ("0000:18:00.0", None)}), [])


class TestXpuVisibleDeviceIndices(unittest.TestCase):
    """ZE_AFFINITY_MASK filters the device list; it does not permute it."""

    def _indices(self, mask, num_devices=4):
        env = {} if mask is None else {"ZE_AFFINITY_MASK": mask}
        with patch.dict(os.environ, env, clear=False):
            if mask is None:
                os.environ.pop("ZE_AFFINITY_MASK", None)
            return _xpu_visible_device_indices(num_devices)

    def test_unset_mask_exposes_every_device(self):
        self.assertEqual(self._indices(None), [0, 1, 2, 3])
        self.assertEqual(self._indices(""), [0, 1, 2, 3])
        self.assertEqual(self._indices("default"), [0, 1, 2, 3])

    def test_mask_order_is_ignored(self):
        # Verified against Level Zero on an 8-GPU host: "3,0" exposes physical
        # 0 as xpu:0 and physical 3 as xpu:1.
        self.assertEqual(self._indices("3,0"), [0, 3])
        self.assertEqual(self._indices("0,3"), [0, 3])
        self.assertEqual(self._indices("3,2,1,0"), [0, 1, 2, 3])

    def test_duplicate_and_out_of_range_entries_are_dropped(self):
        self.assertEqual(self._indices("3,3,0"), [0, 3])
        self.assertEqual(self._indices("1,99"), [1])
        self.assertEqual(self._indices("99"), [])
        self.assertEqual(self._indices("-1"), [])

    def test_composite_entry_selects_its_root_device(self):
        self.assertEqual(self._indices("1.0,3.0"), [1, 3])
        self.assertEqual(self._indices("2.1"), [2])

    def test_surrounding_whitespace_is_tolerated(self):
        self.assertEqual(self._indices(" 2 , 1 "), [1, 2])

    def test_unparsable_entries_are_dropped(self):
        self.assertEqual(self._indices("1,abc,2"), [1, 2])


class TestQueryNumaNodeForXpu(unittest.TestCase):
    """_query_numa_node_for_xpu: mask handling, runtime cross-check, node read."""

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

    def _query(
        self, device_id, mask=None, device_count=None, nodes=None, addresses=None
    ):
        addresses = self.ADDRS if addresses is None else addresses
        pci_meta = {
            f"/sys/bus/pci/devices/{a}/numa_node": node + "\n"
            for a, node in (self.NODES if nodes is None else nodes).items()
        }
        env = {} if mask is None else {"ZE_AFFINITY_MASK": mask}
        if device_count is None:
            device_count = len(addresses)
        with (
            patch(
                "sglang.srt.utils.numa_utils._list_xpu_pci_addresses",
                return_value=list(addresses),
            ),
            patch("builtins.open", _fake_sysfs_reader(pci_meta)),
            patch.dict(os.environ, env, clear=False),
            patch(
                "torch.xpu.device_count",
                return_value=device_count,
            ),
        ):
            if mask is None:
                os.environ.pop("ZE_AFFINITY_MASK", None)
            return _query_numa_node_for_xpu(device_id)

    def test_plain_device_index(self):
        self.assertEqual(self._query(0), [0])
        self.assertEqual(self._query(2), [1])

    def test_negative_node_returns_empty(self):
        self.assertEqual(self._query(3), [])

    def test_affinity_mask_selects_by_ascending_physical_index(self):
        # Treating "3,0" as a permutation would swap these two results.
        self.assertEqual(self._query(0, mask="3,0", device_count=2), [0])
        self.assertEqual(self._query(1, mask="3,0", device_count=2), [])
        self.assertEqual(self._query(0, mask="2,1", device_count=2), [0])
        self.assertEqual(self._query(1, mask="2,1", device_count=2), [1])

    def test_device_id_out_of_range_returns_empty(self):
        self.assertEqual(self._query(9, device_count=4), [])

    # Core + iGPU + 1x Arc Pro B60: sysfs lists both Intel GPUs but Level Zero
    # enumerates only the discrete one, so xpu:0 is the B60, not the iGPU.
    IGPU_ADDRS = ["0000:03:00.0", "0000:00:02.0"]  # discrete first, iGPU last
    IGPU_NODES = {"0000:03:00.0": "1", "0000:00:02.0": "0"}

    def test_integrated_gpu_the_runtime_does_not_enumerate_is_dropped(self):
        self.assertEqual(
            self._query(
                0, addresses=self.IGPU_ADDRS, nodes=self.IGPU_NODES, device_count=1
            ),
            [1],
        )

    def test_integrated_gpu_the_runtime_does_enumerate_stays_last(self):
        for device_id, node in ((0, [1]), (1, [0])):
            self.assertEqual(
                self._query(
                    device_id,
                    addresses=self.IGPU_ADDRS,
                    nodes=self.IGPU_NODES,
                    device_count=2,
                ),
                node,
            )

    def test_runtime_device_count_mismatch_returns_empty(self):
        self.assertEqual(self._query(0, device_count=8), [])
        self.assertEqual(self._query(0, device_count=0), [])

    def test_unreadable_numa_node_returns_empty(self):
        self.assertEqual(self._query(0, nodes={}), [])
        self.assertEqual(self._query(0, nodes={self.ADDRS[0]: "not-a-node"}), [])

    def test_no_intel_gpus_returns_empty(self):
        with patch(
            "sglang.srt.utils.numa_utils._list_xpu_pci_addresses", return_value=[]
        ):
            self.assertEqual(_query_numa_node_for_xpu(0), [])

    @patch("sglang.srt.utils.numa_utils._is_xpu", True)
    def test_query_numa_node_for_gpu_delegates_to_xpu(self):
        with patch(
            "sglang.srt.utils.numa_utils._query_numa_node_for_xpu",
            return_value=[2],
        ) as mock_xpu:
            self.assertEqual(_query_numa_node_for_gpu(5), [2])
            mock_xpu.assert_called_once_with(5)


@unittest.skipUnless(torch.xpu.is_available(), "requires XPU hardware")
class TestXpuOrderMatchesRuntime(unittest.TestCase):
    """The /dev/dri reconstruction must match the device order a worker sees."""

    def setUp(self):
        # A host whose /dev/dri and runtime disagree is one where declining to
        # bind is the correct answer, so there is no mapping to assert; the
        # mocked cases above pin that reconciliation logic.
        if _xpu_pci_address(0) is None:
            self.skipTest(
                f"/dev/dri lists {len(_list_xpu_pci_addresses())} Intel GPU(s) but "
                f"torch reports {torch.xpu.device_count()}; no mapping to check"
            )

    def test_logical_index_resolves_to_the_same_pci_address_as_torch(self):
        for device_id in range(torch.xpu.device_count()):
            device_uuid = torch.xpu.get_device_properties(device_id).uuid
            runtime_address = _bdf_from_xpu_uuid(device_uuid.bytes)
            if runtime_address is None:
                self.skipTest(
                    f"xpu:{device_id} UUID {device_uuid} is not in the Intel layout"
                )
            self.assertEqual(
                _xpu_pci_address(device_id),
                runtime_address,
                f"xpu:{device_id} mismatch",
            )

    def test_every_device_resolves_to_its_sysfs_numa_node(self):
        for device_id in range(torch.xpu.device_count()):
            address = _xpu_pci_address(device_id)
            with open(f"/sys/bus/pci/devices/{address}/numa_node") as f:
                sysfs_node = int(f.read().strip())
            expected = [] if sysfs_node < 0 else [sysfs_node]
            self.assertEqual(
                _query_numa_node_for_xpu(device_id), expected, f"xpu:{device_id}"
            )


if __name__ == "__main__":
    unittest.main()
