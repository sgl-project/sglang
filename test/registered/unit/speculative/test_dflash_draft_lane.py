import importlib.util
import pathlib
import sys
import unittest
from dataclasses import replace

path = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python/sglang/srt/disaggregation/draft_lane.py"
)
spec = importlib.util.spec_from_file_location("draft_lane_under_test", path)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def peer(rank=0):
    regions = tuple(
        m.DraftRegion(
            layer,
            kind,
            "FULL" if layer == 4 else "WINDOW",
            1000000000 + layer * 100000000 + k * 10000000,
            16384 * 2048,
            2048,
            16384,
        )
        for layer in range(5)
        for k, kind in enumerate(("K", "V"))
    )
    return m.DraftPeer(
        f"reg{rank}",
        "cohort",
        "epoch",
        rank,
        f"engine{rank}",
        f"tcp://127.0.0.1:{2000 + rank}",
        "model-config-and-weights",
        "bfloat16",
        rank * 8,
        (rank + 1) * 8,
        16,
        128,
        64,
        12,
        regions,
    )


def lease():
    return m.DraftLease(
        "lease", 9, "request", 3, 0, 1, 5000, 8192, 64, 8256, 64, 2176, "epoch"
    )


from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class DraftLaneTests(CustomTestCase):
    def test_wire_roundtrip_and_full_head_coverage(self):
        peers = (peer(), peer(1))
        rec = m.decode_control(
            m.encode_control("REGISTER", {"peers": [m.peer_to_dict(p) for p in peers]})
        )
        decoded = tuple(m.peer_from_dict(p) for p in rec["payload"]["peers"])
        self.assertEqual(m.validate_cohort(decoded), peers)
        self.assertIsNone(m.decode_control([b"legacy"]))
        with self.assertRaises(ValueError):
            m.decode_control([m.CONTROL_TAG, b'{"version":2,"payload":{}}'])
        with self.assertRaises(ValueError):
            m.validate_cohort([peer(), replace(peer(1), head_begin=7)])
        with self.assertRaises(ValueError):
            m.validate_cohort([peer()])
        with self.assertRaises(ValueError):
            m.validate_cohort([peer(), replace(peer(1), fingerprint="other")])

    def test_copy_plan_coalesces_and_splits_ring_wrap(self):
        item = lease()
        full = peer().regions[-2]
        blocks = m.plan_region_copy(full, full, range(5000), range(5000), item)
        self.assertEqual(blocks, [(full.base, full.base + 64 * 2048, 5000 * 2048)])
        window = peer().regions[0]
        rows = [64 + p % 2176 for p in range(5000)]
        blocks = m.plan_region_copy(window, window, rows, range(5000), item)
        self.assertEqual(sum(b[2] for b in blocks), 2176 * 2048)
        self.assertEqual(len(blocks), 2)
        with self.assertRaises(ValueError):
            m.plan_region_copy(full, full, [0], [5000], item)
        with self.assertRaises(ValueError):
            m.plan_region_copy(full, full, [16384], [0], item)
        with self.assertRaises(ValueError):
            m.plan_region_copy(full, full, [0, 1], [1, 0], item)
        with self.assertRaises(ValueError):
            m.plan_region_copy(full, replace(full, row_stride=1024), [0], [0], item)

    def test_ready_requires_every_layer_head_kind_and_range(self):
        state = m.DraftLeaseState(lease(), peer())
        ranges = [
            (r.key, 0 if r.pool == "FULL" else lease().window_floor, 4000)
            for r in peer().regions
        ]
        state.transferred("chunk0", ranges)
        self.assertFalse(state.transferred("chunk0", ranges))
        with self.assertRaises(RuntimeError):
            state.mark_ready()
        state.transferred("chunk1", [(r.key, 4000, 5000) for r in peer().regions])
        state.mark_ready()
        self.assertEqual(state.phase, "READY")
        with self.assertRaises(RuntimeError):
            state.transferred("late", ranges)

    def test_uncertain_transport_quarantines_until_both_drained(self):
        state = m.DraftLeaseState(lease(), peer())
        self.assertFalse(state.retire(readers_drained=True, transport_drained=False))
        self.assertEqual(state.phase, "QUARANTINED")
        with self.assertRaises(RuntimeError):
            state.mark_ready()
        self.assertTrue(state.retire(readers_drained=False, transport_drained=True))
        self.assertEqual(state.phase, "RELEASED")
        with self.assertRaises(ValueError):
            m.DraftLeaseState(replace(lease(), cohort_epoch="stale"), peer())

    def test_refined_head_copy_has_correct_offsets_and_rejects_coarsening(self):
        source = peer().regions[-2]
        destination = replace(source, base=1 << 48, row_stride=1024)
        blocks = m.plan_region_head_copy(
            source,
            destination,
            [0, 1],
            [0, 1],
            lease(),
            source_head_begin=0,
            source_head_end=8,
            destination_head_begin=4,
            destination_head_end=8,
            head_dim=128,
        )
        self.assertEqual(
            blocks,
            [
                (source.base + 1024, destination.base + 64 * 1024, 1024),
                (source.base + 2048 + 1024, destination.base + 65 * 1024, 1024),
            ],
        )
        peers = tuple(
            replace(
                peer(rank // 2),
                registration_id=f"reg{rank}",
                peer=rank,
                head_begin=rank * 4,
                head_end=(rank + 1) * 4,
                regions=tuple(replace(r, row_stride=1024) for r in peer().regions),
            )
            for rank in range(4)
        )
        self.assertEqual(
            [p.peer for p in m.destination_head_shards(peers, 0, 8)], [0, 1]
        )
        self.assertEqual(
            [p.peer for p in m.destination_head_shards(peers, 8, 16)], [2, 3]
        )
        with self.assertRaisesRegex(ValueError, "crosses"):
            m.destination_head_shards((peer(), peer(1)), 0, 4)
        with self.assertRaisesRegex(ValueError, "ownership"):
            m.plan_region_head_copy(
                source,
                source,
                [0],
                [0],
                lease(),
                source_head_begin=0,
                source_head_end=8,
                destination_head_begin=4,
                destination_head_end=12,
                head_dim=128,
            )
        full = m.plan_region_head_copy(
            source,
            source,
            [0, 1],
            [0, 1],
            lease(),
            source_head_begin=0,
            source_head_end=8,
            destination_head_begin=0,
            destination_head_end=8,
            head_dim=128,
        )
        self.assertEqual(
            full, m.plan_region_copy(source, source, [0, 1], [0, 1], lease())
        )

    def test_all_addresses_are_64_bit_and_bounded(self):
        region = replace(
            peer().regions[-1],
            base=1 << 48,
            rows=1048704,
            registered_bytes=1048704 * 2048,
        )
        self.assertGreater(region.address(1048576), 1 << 48)
        with self.assertRaises(ValueError):
            region.address(1048704)
        with self.assertRaises(ValueError):
            replace(region, registered_bytes=16)
        with self.assertRaises(ValueError):
            replace(lease(), logical_end=8193).validate()


if __name__ == "__main__":
    unittest.main()
