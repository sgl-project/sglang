"""Sharing one Mooncake engine's memory registrations across its users."""

import unittest

from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeMemoryRegistry,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ERR_ADDRESS_OVERLAPPED = -7


class FakeMooncakeEngine:
    """Mooncake's registration rules: any overlap fails the whole batch."""

    def __init__(self):
        # ptr -> (length, remote_accessible)
        self.regions = {}
        self.register_calls = 0

    def _overlaps(self, ptr, length):
        return any(
            other < ptr + length and ptr < other + other_length
            for other, (other_length, _) in self.regions.items()
        )

    def batch_register_memory(self, ptrs, lengths):
        self.register_calls += 1
        if any(self._overlaps(ptr, length) for ptr, length in zip(ptrs, lengths)):
            return ERR_ADDRESS_OVERLAPPED
        for ptr, length in zip(ptrs, lengths):
            self.regions[ptr] = (length, True)
        return 0

    def batch_unregister_memory(self, ptrs):
        for ptr in ptrs:
            del self.regions[ptr]
        return 0

    def store_register_buffer(self, ptr, length):
        """What MooncakeDistributedStore.register_buffer does on this engine."""
        assert not self._overlaps(ptr, length)
        self.regions[ptr] = (length, False)


KV = (0x1000, 0x1000)
STATE = (0x4000, 0x200)
AUX = (0x8000, 0x100)


class TestMooncakeMemoryRegistry(CustomTestCase):
    def setUp(self):
        self.engine = FakeMooncakeEngine()
        self.registry = MooncakeMemoryRegistry(self.engine)

    def test_identical_region_is_registered_once(self):
        self.registry.register([KV[0]], [KV[1]])
        self.registry.register([KV[0]], [KV[1]])

        self.assertEqual(self.engine.register_calls, 1)
        self.assertEqual(self.engine.regions, {KV[0]: (KV[1], True)})

    def test_region_is_unregistered_after_its_last_user(self):
        self.registry.register([KV[0]], [KV[1]])
        self.registry.register([KV[0]], [KV[1]])

        self.registry.deregister([KV[0]])
        self.assertIn(KV[0], self.engine.regions)

        self.registry.deregister([KV[0]])
        self.assertNotIn(KV[0], self.engine.regions)

    def test_store_regions_are_shared_with_remote_access(self):
        # The linker registers the GPU KV pool through the store first; PD
        # transfer then registers the same pool plus its own buffers.
        for ptr, length in (KV, STATE):
            self.engine.store_register_buffer(ptr, length)
            self.registry.adopt_store_registration(ptr, length)

        self.registry.register([KV[0], STATE[0], AUX[0]], [KV[1], STATE[1], AUX[1]])

        self.assertEqual(
            self.engine.regions,
            {
                KV[0]: (KV[1], True),
                STATE[0]: (STATE[1], True),
                AUX[0]: (AUX[1], True),
            },
        )

    def test_views_inside_a_store_allocation_share_it(self):
        # MLA pools are one allocation the linker registers whole, while PD
        # transfer registers a view per layer inside it.
        self.engine.store_register_buffer(*KV)
        self.registry.adopt_store_registration(*KV)
        layer_views = [KV[0], KV[0] + 0x400, KV[0] + 0x800]

        self.registry.register(layer_views, [0x400] * len(layer_views))
        self.assertEqual(self.engine.regions, {KV[0]: (KV[1], True)})

        self.registry.deregister(layer_views)
        self.assertEqual(self.engine.regions, {KV[0]: (KV[1], True)})

    def test_store_keeps_its_regions_after_other_users_release_them(self):
        self.engine.store_register_buffer(*KV)
        self.registry.adopt_store_registration(*KV)
        self.registry.register([KV[0]], [KV[1]])

        self.registry.deregister([KV[0]])

        self.assertEqual(self.engine.regions, {KV[0]: (KV[1], True)})

    def test_partial_overlap_is_rejected_before_registering(self):
        self.registry.register([KV[0]], [KV[1]])

        with self.assertRaisesRegex(ValueError, "overlaps registered region"):
            self.registry.register([KV[0] + KV[1] - 0x10, AUX[0]], [0x20, AUX[1]])
        self.assertNotIn(AUX[0], self.engine.regions)

    def test_region_extending_past_a_registration_is_rejected(self):
        self.registry.register([KV[0]], [KV[1]])

        with self.assertRaisesRegex(ValueError, "overlaps registered region"):
            self.registry.register([KV[0]], [KV[1] * 2])

    def test_duplicate_regions_in_one_call_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.registry.register([KV[0], KV[0]], [KV[1], KV[1]])

    def test_failed_registration_raises_and_is_not_tracked(self):
        # A region registered on the engine outside the registry.
        self.engine.regions[KV[0]] = (KV[1], True)

        with self.assertRaisesRegex(RuntimeError, f"ret={ERR_ADDRESS_OVERLAPPED}"):
            self.registry.register([KV[0], AUX[0]], [KV[1], AUX[1]])
        with self.assertRaisesRegex(ValueError, "not registered"):
            self.registry.deregister([AUX[0]])

    def test_deregistering_an_unknown_region_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not registered"):
            self.registry.deregister([KV[0]])


if __name__ == "__main__":
    unittest.main()
