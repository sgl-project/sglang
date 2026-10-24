"""Held routes and the held-routes store of routed-experts forcing."""

import unittest

import numpy as np
import torch

from sglang.srt.state_capturer.routed_experts_store import (
    HeldRoutes,
    HeldRoutesStore,
    RouteLayout,
    RouteSegment,
    _route_dtype,
    build_forcing_batch,
    unforced_recompute_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

LAYOUT = RouteLayout(moe_layer_ids=(1, 2), top_k=3, num_logical_experts=200)


def _segment(start, n):
    ids = np.arange(start * 6, (start + n) * 6) % 200
    return RouteSegment(start=start, routes=ids.reshape(n, 2, 3).astype(np.uint8))


def _held(*lengths):
    held, start = HeldRoutes(), 0
    for n in lengths:
        held = held.extended(_segment(start, n))
        start += n
    return held


def _all_routes(held):
    return np.concatenate([s.routes for s in held.segments])


class TestRouteLayout(CustomTestCase):
    def test_dtype_is_the_narrowest_that_holds_every_id(self):
        self.assertEqual(_route_dtype(8), np.uint8)
        self.assertEqual(_route_dtype(256), np.uint8)
        self.assertEqual(_route_dtype(257), np.int16)
        self.assertEqual(_route_dtype(32768), np.int16)
        for bad in (0, 32769):
            with self.assertRaises(ValueError):
                _route_dtype(bad)

    def test_narrow_keeps_moe_columns_only(self):
        captured = np.zeros((4, 4, 3), dtype=np.int32)  # layers 0 and 3 are dense
        captured[:, 1:3] = np.arange(24).reshape(4, 2, 3) + 230
        for experts, dtype in ((256, np.uint8), (257, np.int16)):
            layout = RouteLayout(
                moe_layer_ids=(1, 2), top_k=3, num_logical_experts=experts
            )
            rows = layout.narrow(captured)
            self.assertEqual(rows.dtype, dtype)
            np.testing.assert_array_equal(rows, captured[:, 1:3])

    def test_narrow_refuses_ids_it_cannot_hold(self):
        layout = RouteLayout(moe_layer_ids=(0,), top_k=1, num_logical_experts=255)
        for bad in (255, -1):
            with self.assertRaisesRegex(ValueError, "outside"):
                layout.narrow(np.full((2, 1, 1), bad, dtype=np.int32))
        with self.assertRaisesRegex(ValueError, "top-1"):
            layout.narrow(np.zeros((2, 1, 2), dtype=np.int32))


class TestHeldRoutes(CustomTestCase):
    def test_chain_truncate_and_copy_across_segments(self):
        held = _held(5, 3, 4)
        self.assertEqual(held.end, 12)
        with self.assertRaisesRegex(ValueError, "not 5"):
            HeldRoutes(segments=(_segment(0, 5), _segment(6, 2)))

        cut = held.truncated(7)
        self.assertEqual((cut.end, len(cut.segments), held.end), (7, 2, 12))

        out = np.full((6, 2, 3), 99, dtype=np.uint8)
        held.copy_rows(3, 9, out=out)
        np.testing.assert_array_equal(out, _all_routes(held)[3:9])
        with self.assertRaisesRegex(ValueError, "outside"):
            held.copy_rows(10, 13, out=out)

    def test_segments_are_read_only(self):
        segment = _segment(0, 2)
        with self.assertRaises(ValueError):
            segment.routes[0, 0, 0] = 1


class TestHeldRoutesStore(CustomTestCase):
    def test_admit_is_bounded_by_force_len_held_routes_and_matching_tokens(self):
        store = HeldRoutesStore(max_tokens=1000)
        tokens = list(range(20))
        first = store.admit("s", token_ids=tokens, force_len=0)
        self.assertTrue(
            store.commit("s", token_ids=tokens, rows=_held(10, 9), admission=first)
        )

        def held_end(key, token_ids, force_len):
            return store.admit(key, token_ids=token_ids, force_len=force_len).rows.end

        self.assertEqual(held_end("s", tokens, 50), 19)
        self.assertEqual(held_end("s", tokens, 7), 7)
        self.assertEqual(held_end("missing", tokens, 50), 0)
        # Row i depends on tokens [0, i]: a change at token 12 keeps rows [0, 12).
        edited = tokens.copy()
        edited[12] = 999
        self.assertEqual(held_end("s", edited, 50), 12)
        edited[0] = 999
        self.assertEqual(held_end("s", edited, 50), 0)
        # A longer continuation of the same sequence keeps every held row.
        self.assertEqual(held_end("s", list(range(40)), 50), 19)

    def test_commit_replaces_and_evicts_least_recently_used_by_tokens(self):
        store = HeldRoutesStore(max_tokens=30)

        def commit(key, n):
            admission = store.admit(key, token_ids=range(n + 1), force_len=0)
            return store.commit(
                key, token_ids=range(n + 1), rows=_held(n), admission=admission
            )

        def held(*keys):
            return [
                store.admit(key, token_ids=range(41), force_len=40).rows.end
                for key in keys
            ]

        commit("a", 10)
        commit("b", 10)
        store.admit("a", token_ids=range(11), force_len=10)  # b is now the oldest
        commit("c", 15)
        self.assertEqual(held("b", "a", "c"), [0, 10, 15])

        # Replacing "a" frees 7 tokens, so a 12-token key fits without evicting.
        commit("a", 3)
        commit("d", 12)
        self.assertEqual(held("a", "c", "d"), [3, 15, 12])
        # A key larger than the whole store is dropped without evicting the others.
        self.assertTrue(commit("huge", 40))
        self.assertEqual(held("huge", "a", "c", "d"), [0, 3, 15, 12])

        admission = store.admit("a", token_ids=range(3), force_len=0)
        with self.assertRaisesRegex(ValueError, "3 rows for 3 tokens"):
            store.commit("a", token_ids=range(3), rows=_held(3), admission=admission)

    def test_stale_commit_invalidates_key(self):
        """Two attempts of one turn (a retry and the attempt it abandoned) are
        admitted against the same rows. The store cannot tell which one the
        client kept, so the later commit drops the key's rows, even when both
        attempts produced the same tokens: the next turn is refused, never
        forced to the abandoned attempt's routes."""
        tokens = list(range(12))
        for first in ("kept", "abandoned"):
            with self.subTest(first=first):
                store = HeldRoutesStore(max_tokens=1000)
                turn1 = store.admit("s", token_ids=range(9), force_len=0)
                store.commit("s", token_ids=range(9), rows=_held(8), admission=turn1)
                attempts = {
                    name: store.admit("s", token_ids=tokens, force_len=8)
                    for name in ("kept", "abandoned")
                }
                second = "abandoned" if first == "kept" else "kept"
                for name, applied in ((first, True), (second, False)):
                    self.assertEqual(
                        store.commit(
                            "s",
                            token_ids=tokens,
                            rows=_held(8, 3),
                            admission=attempts[name],
                        ),
                        applied,
                    )
                self.assertEqual(
                    store.admit("s", token_ids=tokens, force_len=11).rows.end, 0
                )
                # A new first turn of the key starts over.
                fresh = store.admit("s", token_ids=tokens, force_len=0)
                self.assertTrue(
                    store.commit("s", token_ids=tokens, rows=_held(11), admission=fresh)
                )

    def test_evicted_key_can_commit(self):
        """Evicting other sessions' entries, even ones committed after a request
        was admitted, never costs that request its commit; only a newer commit
        of its own key does."""
        store = HeldRoutesStore(max_tokens=10)
        first = store.admit("s", token_ids=range(9), force_len=0)
        store.commit("s", token_ids=range(9), rows=_held(8), admission=first)
        running = store.admit("s", token_ids=range(12), force_len=8)
        abandoned = store.admit("s", token_ids=range(12), force_len=8)
        for key in ("t", "u"):  # evict "s", then "t", committed after `running`
            admission = store.admit(key, token_ids=range(9), force_len=0)
            store.commit(key, token_ids=range(9), rows=_held(8), admission=admission)
        rows = running.rows.extended(_segment(8, 1))
        self.assertTrue(
            store.commit("s", token_ids=range(10), rows=rows, admission=running)
        )
        self.assertEqual(store.admit("s", token_ids=range(10), force_len=9).rows.end, 9)

        for key in ("t", "u"):  # evict "s" again
            admission = store.admit(key, token_ids=range(9), force_len=0)
            store.commit(key, token_ids=range(9), rows=_held(8), admission=admission)
        self.assertFalse(
            store.commit("s", token_ids=range(10), rows=rows, admission=abandoned)
        )
        self.assertEqual(store.admit("s", token_ids=range(10), force_len=9).rows.end, 0)

    def test_dropped_eviction_record_fails_closed(self):
        """With its eviction record gone, the store cannot tell whether the key
        changed after an admission, so a commit admitted before the newest
        dropped record is refused: the key's next turn is refused."""
        store = HeldRoutesStore(max_tokens=10, max_evicted_keys=1)
        first = store.admit("s", token_ids=range(9), force_len=0)
        store.commit("s", token_ids=range(9), rows=_held(8), admission=first)
        kept = store.admit("s", token_ids=range(12), force_len=8)
        abandoned = store.admit("s", token_ids=range(12), force_len=8)
        rows = kept.rows.extended(_segment(8, 1))
        store.commit("s", token_ids=range(10), rows=rows, admission=kept)
        for key in ("t", "u"):  # evict "s", then drop its record for "t"'s
            admission = store.admit(key, token_ids=range(9), force_len=0)
            store.commit(key, token_ids=range(9), rows=_held(8), admission=admission)
        self.assertFalse(
            store.commit("s", token_ids=range(10), rows=rows, admission=abandoned)
        )
        self.assertEqual(store.admit("s", token_ids=range(10), force_len=9).rows.end, 0)

    def test_budget_counts_owned_rows(self):
        store = HeldRoutesStore(max_tokens=1000)
        first = store.admit("s", token_ids=range(11), force_len=0)
        store.commit("s", token_ids=range(11), rows=_held(10), admission=first)
        # The client kept only 4 rows of the 10-row turn (e.g. a re-rendered history).
        tokens = list(range(5)) + [50, 51, 52]
        admission = store.admit("s", token_ids=tokens, force_len=4)
        self.assertIsNotNone(admission.rows.segments[0].routes.base)

        rows = admission.rows.extended(_segment(4, 3))
        store.commit("s", token_ids=tokens, rows=rows, admission=admission)
        held = store.admit("s", token_ids=tokens, force_len=7).rows
        self.assertEqual(held.end, 7)
        self.assertEqual(
            sum(s.routes.nbytes for s in held.segments), 7 * 6, "held bytes"
        )
        for segment in held.segments:
            self.assertIsNone(segment.routes.base)
            self.assertFalse(segment.routes.flags.writeable)
        np.testing.assert_array_equal(_all_routes(held), _all_routes(rows))


class TestForcingBatch(CustomTestCase):
    def test_unforced_recompute_tokens(self):
        for (prefix_len, held_end, force_len), unforced in (
            ((0, 4, 7), 3),
            ((5, 0, 4), 0),
            ((0, 8, 8), 0),
        ):
            self.assertEqual(
                unforced_recompute_tokens(
                    prefix_len=prefix_len,
                    extend_len=10,
                    held_end=held_end,
                    force_len=force_len,
                ),
                unforced,
            )

    def test_build_places_rows_at_their_forward_rows(self):
        held = [_held(10), _held(6), HeldRoutes()]
        batch = build_forcing_batch(
            held=held, prefix_lens=[8, 0, 0], extend_lens=[5, 4, 3], layout=LAYOUT
        )
        self.assertEqual((int(batch.mask.sum()), len(batch.mask)), (6, 12))
        self.assertEqual(
            batch.mask.tolist(), [True, True] + [False] * 3 + [True] * 4 + [False] * 3
        )
        self.assertEqual(batch.routes.dtype, torch.uint8)
        np.testing.assert_array_equal(
            batch.routes[:2].numpy(), _all_routes(held[0])[8:10]
        )
        np.testing.assert_array_equal(
            batch.routes[5:9].numpy(), _all_routes(held[1])[:4]
        )
        self.assertFalse(batch.routes[2:5].any() or batch.routes[9:].any())

        device = batch.padded_to(15).to_device("cpu")
        self.assertEqual(len(device.mask), 15)
        self.assertFalse(device.mask[12:].any())
        self.assertEqual(device.routes.dtype, torch.int32)
        self.assertTrue(torch.equal(device.routes[:12], batch.routes.int()))

        self.assertIsNone(
            build_forcing_batch(
                held=held, prefix_lens=[10, 6, 0], extend_lens=[2, 2, 2], layout=LAYOUT
            )
        )

    def test_wide_ids_survive_the_device_copy(self):
        layout = RouteLayout(moe_layer_ids=(0,), top_k=2, num_logical_experts=512)
        routes = layout.narrow(np.array([[[511, 300]], [[0, 256]]], dtype=np.int32))
        held = HeldRoutes(segments=(RouteSegment(start=0, routes=routes),))
        batch = build_forcing_batch(
            held=[held], prefix_lens=[0], extend_lens=[2], layout=layout
        ).to_device("cpu")
        self.assertEqual(batch.routes[:, 0].tolist(), [[511, 300], [0, 256]])


if __name__ == "__main__":
    unittest.main()
