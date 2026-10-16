"""SWAComponent.create_match_validator: which host-only Full-KV tombstones are match boundaries.

A tree node whose SWA component has neither a device value nor a host value is a
"host-only tombstone" when its Full KV is still backed up on the host. The
validator must accept such a node as a match boundary whenever the SWA rows are
not stored in the tree at all, so that load_back restores the Full KV and the
window is rebuilt by the scheduler. Two layouts have that property:

* a per-request SWA ring (`is_swa_req_ring(allocator)`, the unified-KV layout);
* HiCache without an SWA host pool (a paged or request-window SWA pool: SWA
  never leaves the device, so a host-backed prefix always looks like this).

The second clause was dropped by #38269 and is restored here. These tests drive
the closure with plain stand-ins for the component, the tree core and the nodes;
no pools are built.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components import swa as swa_module
from sglang.srt.mem_cache.unified_cache.components.base import ComponentData
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_WINDOW = 4


def _component(*, has_swa_host_pool: bool, enable_hicache: bool) -> SimpleNamespace:
    return SimpleNamespace(
        sliding_window_size=_WINDOW,
        component_type=ComponentType.SWA,
        cache=SimpleNamespace(token_to_kv_pool_allocator=object()),
        tree_core=SimpleNamespace(
            has_swa_host_pool=has_swa_host_pool, enable_hicache=enable_hicache
        ),
    )


def _node(
    *,
    key_len: int = 2,
    swa_value=None,
    swa_host_value=None,
    full_value=None,
    full_host_value=None,
) -> SimpleNamespace:
    data = [ComponentData() for _ in ComponentType]
    data[ComponentType.FULL].value = full_value
    data[ComponentType.FULL].host_value = full_host_value
    data[ComponentType.SWA].value = swa_value
    data[ComponentType.SWA].host_value = swa_host_value
    # UnifiedTreeNode derives both from the FULL component's data.
    return SimpleNamespace(
        key=list(range(key_len)),
        component_data=data,
        backuped=full_host_value is not None,
        evicted=full_value is None,
    )


def _host_only_tombstone() -> SimpleNamespace:
    """Full KV evicted from the device and backed up on the host; SWA nowhere."""
    return _node(full_value=None, full_host_value="host-kv")


def _validator(component, *, swa_req_ring: bool, match_device_only: bool = False):
    with mock.patch.object(
        swa_module, "is_swa_req_ring", return_value=swa_req_ring
    ) as ring:
        validator = SWAComponent.create_match_validator(
            component, match_device_only=match_device_only
        )
        ring.assert_called_once_with(component.cache.token_to_kv_pool_allocator)
    return validator


class TestSWAHostOnlyTombstoneGate(unittest.TestCase):
    def test_paged_swa_with_hicache_and_no_swa_host_pool_accepts_tombstone(self):
        """The layout #38269 lost: paged / request-window SWA, HiCache on, no SWA
        host pool. The host-backed prefix must stay matchable (load_back runs)."""
        validator = _validator(
            _component(has_swa_host_pool=False, enable_hicache=True),
            swa_req_ring=False,
        )
        self.assertTrue(validator(_host_only_tombstone()))

    def test_per_request_ring_accepts_tombstone(self):
        """#38269's own case keeps working, with or without an SWA host pool."""
        for has_swa_host_pool in (False, True):
            validator = _validator(
                _component(has_swa_host_pool=has_swa_host_pool, enable_hicache=True),
                swa_req_ring=True,
            )
            self.assertTrue(validator(_host_only_tombstone()), has_swa_host_pool)

    def test_swa_host_pool_rejects_tombstone_without_swa_host_value(self):
        """With an SWA host tier the SWA rows are expected on the host; a node
        that has none is a real miss."""
        validator = _validator(
            _component(has_swa_host_pool=True, enable_hicache=True),
            swa_req_ring=False,
        )
        self.assertFalse(validator(_host_only_tombstone()))

    def test_hicache_off_rejects_tombstone(self):
        validator = _validator(
            _component(has_swa_host_pool=False, enable_hicache=False),
            swa_req_ring=False,
        )
        self.assertFalse(validator(_host_only_tombstone()))

    def test_gate_still_requires_backed_up_or_resident_full_kv(self):
        """A node evicted from the device with no host backup has nothing to load
        back: rejected under both gates."""
        gone = _node(full_value=None, full_host_value=None)
        for swa_req_ring in (False, True):
            validator = _validator(
                _component(has_swa_host_pool=False, enable_hicache=True),
                swa_req_ring=swa_req_ring,
            )
            self.assertFalse(validator(gone), swa_req_ring)

    def test_swa_host_value_is_not_a_tombstone_unless_device_only(self):
        """SWA on the host counts toward the window in a normal match and is a
        tombstone only when the match is restricted to device-resident data."""
        node = _node(key_len=_WINDOW, swa_host_value="host-swa", full_value="kv")
        component = _component(has_swa_host_pool=True, enable_hicache=True)
        self.assertTrue(_validator(component, swa_req_ring=False)(node))
        device_only = _validator(component, swa_req_ring=False, match_device_only=True)
        self.assertFalse(device_only(node))

    def test_device_resident_swa_keeps_the_window_accounting(self):
        """Nodes with SWA on the device follow the window count, which starts
        unbounded and restarts at zero after an accepted tombstone."""
        validator = _validator(
            _component(has_swa_host_pool=False, enable_hicache=True),
            swa_req_ring=False,
        )
        resident = _node(key_len=_WINDOW // 2, swa_value="swa", full_value="kv")
        self.assertTrue(validator(resident))
        self.assertTrue(validator(_host_only_tombstone()))
        self.assertFalse(validator(resident))
        self.assertTrue(validator(resident))


if __name__ == "__main__":
    unittest.main()
