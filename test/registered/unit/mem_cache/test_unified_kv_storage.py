"""The unified pool's transfer surface: the external names and PD wire
entries of its sub-pools, and their layer routes."""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_P16 = dict(page_size=16, full_tokens=64 * 16, swa_tokens=64 * 16)
CHAINS = {
    "swa": UnifiedGeometry(
        chain="swa", full_layer_ids=(0, 1), swa_layer_ids=(2, 3), **_P16
    ),
    "mamba": UnifiedGeometry(
        chain="mamba", full_layer_ids=(0, 1), state_layer_ids=(2, 3), **_P16
    ),
    "mamba_mla": UnifiedGeometry(
        chain="mamba", full_layer_ids=(0, 1), state_layer_ids=(2, 3), mla=True, **_P16
    ),
    "tri": UnifiedGeometry(
        chain="tri",
        full_layer_ids=(0, 1),
        swa_layer_ids=(2, 3),
        state_layer_ids=(0, 2),
        **_P16,
    ),
}


def _bundle(name):
    setup_allocator_context()
    return build_unified_bundle(CHAINS[name])


class TestPlanNames(CustomTestCase):
    def test_sub_pools_and_wire_entries_per_chain(self):
        expected = {
            "swa": (("full", "kv", "FULL"), ("swa", "swa", "SWA")),
            "mamba": (("full", "kv", "FULL"), ("mamba", "mamba", "MAMBA")),
            "tri": (
                ("full", "kv", "FULL"),
                ("swa", "swa", "SWA"),
                ("mamba", "mamba", "MAMBA"),
            ),
        }
        wires = {
            "swa": [("swa", "kv", "swa")],
            "mamba": [("mamba", "state", "mamba")],
            "tri": [("swa", "kv", "swa"), ("mamba", "state", "mamba")],
        }
        envelopes = {"swa": ("full", "swa"), "mamba": None, "tri": None}
        for name in ("swa", "mamba", "tri"):
            with self.subTest(chain=name):
                storage = _bundle(name).token_to_kv_pool_allocator.storage
                self.assertEqual(storage.owner, "full")
                self.assertEqual(
                    tuple(
                        (s.name, s.external.host_pool, s.external.tree_component)
                        for s in storage.sub_pools()
                    ),
                    expected[name],
                )
                self.assertEqual(
                    [
                        (e.sub_pool, e.part, e.state_type)
                        for e in storage.wire_entries()
                    ],
                    wires[name],
                )
                self.assertEqual(storage.page_envelope_pair(), envelopes[name])

    def test_layer_indices_follow_the_pool_routes(self):
        for name in CHAINS:
            with self.subTest(chain=name):
                bundle = _bundle(name)
                storage = bundle.token_to_kv_pool_allocator.storage
                router = bundle.token_to_kv_pool.layer_router
                for s in storage.sub_pools():
                    if s.holds_state:
                        expected = bundle.token_to_kv_pool_allocator.state.layer_map
                    else:
                        expected = {
                            layer_id: router.route(layer_id).index
                            for layer_id in router.sub_pool(s.name).layer_ids
                        }
                    self.assertEqual(s.layer_indices, expected)


if __name__ == "__main__":
    unittest.main()
