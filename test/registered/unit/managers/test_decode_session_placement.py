"""Verify sticky decode dispatch, capacity projection and bounded history."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.data_parallel_controller import (
    DataParallelController,
    DPBudget,
)
from sglang.srt.managers.decode_session_placement import DecodeSessionPlacement
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDecodeSessionPlacement(CustomTestCase):
    def select(
        self,
        placement,
        key,
        prompt=50,
        output=10,
        active=None,
        live=None,
        capacity=None,
        forced=None,
    ):
        return placement.select(
            key,
            prompt,
            output,
            active or [0, 1],
            live or [0, 0],
            [0, 0],
            capacity or [100, 100],
            forced,
        )

    def test_capacity_and_observed_output_budget_affect_new_sessions(self):
        placement = DecodeSessionPlacement(2)
        self.assertEqual(self.select(placement, ("one", None), capacity=[100, 200]), 1)
        self.assertEqual(self.select(placement, ("two", None), output=80), 0)
        self.assertEqual(placement.retained, [130, 60])

    def test_existing_session_keeps_rank_and_updates_growth(self):
        placement = DecodeSessionPlacement(2)
        self.assertEqual(self.select(placement, ("one", None)), 0)
        self.assertEqual(
            self.select(placement, ("one", None), prompt=70, live=[20, 0]), 0
        )
        self.assertEqual(placement.retained, [100, 0])

    def test_full_sticky_home_spills_to_rank_with_capacity(self):
        """A long prompt must not wait behind a full home while another DP fits."""
        placement = DecodeSessionPlacement(2)
        self.select(placement, ("long-context", None), prompt=59543, forced=0)
        rank = placement.select(
            ("long-context", None),
            60783,
            0,
            [0, 1],
            [254000, 79000],
            [2, 2],
            [291840, 291840],
        )
        self.assertEqual(rank, 1)
        self.assertEqual(placement.sessions[("long-context", None)].rank, 1)
        self.assertEqual(placement.retained, [0, 62023])

    def test_spill_preserves_affinity_without_a_better_known_target(self):
        """Keep a fitting/better home, unknown capacities and explicit ranks."""
        for live, capacity, forced in (
            ([100000, 0], [291840, 291840], None),
            ([254000, 270000], [291840, 291840], None),
            ([254000, 0], [291840, 0], None),
            ([254000, 0], [291840, 291840], 0),
        ):
            with self.subTest(live=live, capacity=capacity, forced=forced):
                placement = DecodeSessionPlacement(2, reserved_decode_tokens=512)
                self.select(placement, ("one", None), prompt=60783, forced=0)
                rank = placement.select(
                    ("one", None),
                    60783,
                    0,
                    [0, 1],
                    live,
                    [2, 2],
                    capacity,
                    forced,
                )
                self.assertEqual(rank, 0)
                self.assertEqual(placement.retained, [60783, 0])

    def test_all_full_ranks_spill_to_smallest_kv_deficit(self):
        """An oversubscribed home must not retain a request just because every
        alternate also needs to wait; that can put short later turns behind
        long decodes while a much smaller backlog exists elsewhere.
        """
        placement = DecodeSessionPlacement(8, reserved_decode_tokens=512)
        active = list(range(8))
        capacity = [291840] * 8
        key = ("queued-turn", None)
        placement.select(
            key, 114253, 0, active, [0] * 8, [0] * 8, capacity, forced_rank=3
        )
        rank = placement.select(
            key,
            114441,
            0,
            active,
            [282267, 215594, 297421, 424238, 393556, 300326, 350710, 392037],
            [5, 7, 4, 8, 7, 6, 5, 4],
            capacity,
        )
        self.assertEqual(rank, 1)
        self.assertEqual(placement.sessions[key].rank, 1)
        self.assertEqual(placement.retained, [0, 114629, 0, 0, 0, 0, 0, 0])

    def test_full_rank_fallback_prices_missing_tokens_not_utilization_ratio(self):
        """The saturated fallback minimizes the tokens that must be released.
        With unequal capacities, that can differ from the lowest usage ratio.
        """
        placement = DecodeSessionPlacement(2)
        key = ("unequal-pools", None)
        self.select(placement, key, prompt=100000, output=0, forced=0)
        rank = placement.select(
            key, 100000, 0, [0, 1], [1100000, 150000], [10, 3], [1000000, 100000]
        )
        # Rank 0 needs 200k tokens released (120% projected usage); rank 1
        # needs only 150k (250%). No destination prefix hit is assumed.
        self.assertEqual(rank, 1)

    def test_all_full_load_tie_keeps_existing_home(self):
        placement = DecodeSessionPlacement(2, reserved_decode_tokens=512)
        key = ("equal-pressure", None)
        self.select(placement, key, prompt=60783, output=0, forced=1)
        rank = placement.select(
            key, 60783, 0, [0, 1], [270000, 270000], [3, 3], [291840, 291840]
        )
        self.assertEqual(rank, 1)
        self.assertEqual(placement.retained, [0, 60783])

    def test_spill_accounts_for_all_active_decode_reservations(self):
        """A raw prompt fit can still exceed KV once decode growth is reserved."""
        for reserve, expected_rank in ((0, 0), (512, 1)):
            with self.subTest(reserve=reserve):
                placement = DecodeSessionPlacement(2, reserved_decode_tokens=reserve)
                self.select(placement, ("one", None), prompt=60783, forced=0)
                rank = placement.select(
                    ("one", None),
                    60783,
                    0,
                    [0, 1],
                    [229000, 0],
                    [5, 0],
                    [291840, 291840],
                )
                self.assertEqual(rank, expected_rank)

    def test_spill_rebinds_only_the_matching_cache_salt(self):
        """Moving a hot session must preserve other namespaces and token counts."""
        placement = DecodeSessionPlacement(2, reserved_decode_tokens=512)
        for salt in ("a", "b"):
            self.select(placement, ("same", salt), prompt=60783, output=0, forced=0)
        placement.select(
            ("same", "a"),
            60783,
            0,
            [0, 1],
            [254000, 0],
            [2, 0],
            [291840, 291840],
        )
        self.assertEqual(placement.sessions[("same", "a")].rank, 1)
        self.assertEqual(placement.sessions[("same", "b")].rank, 0)
        self.assertEqual(placement.retained, [60783, 60783])

    def test_cache_salt_separates_reservations(self):
        placement = DecodeSessionPlacement(2)
        self.assertEqual(self.select(placement, ("same", "a")), 0)
        self.assertEqual(self.select(placement, ("same", "b")), 1)
        self.assertEqual(len(placement.sessions), 2)

    def test_inactive_rank_and_override_rebind(self):
        placement = DecodeSessionPlacement(2)
        self.select(placement, ("one", None))
        self.assertEqual(self.select(placement, ("one", None), active=[1]), 1)
        self.assertEqual(self.select(placement, ("one", None), forced=0), 0)
        self.assertEqual(placement.retained, [60, 0])
        with self.assertRaises(ValueError):
            self.select(placement, ("one", None), active=[1], forced=0)
        self.assertEqual(placement.retained, [60, 0])

    def test_unknown_capacity_uses_raw_load(self):
        placement = DecodeSessionPlacement(2)
        self.assertEqual(
            self.select(placement, None, live=[90, 20], capacity=[0, 100]), 1
        )
        self.assertEqual(len(placement.sessions), 0)

    def test_size_and_ttl_evict_old_estimates(self):
        placement = DecodeSessionPlacement(2, max_sessions=1, idle_seconds=10)
        with patch(
            "sglang.srt.managers.decode_session_placement.time.monotonic",
            return_value=0,
        ):
            self.select(placement, ("one", None))
            self.select(placement, ("two", None))
        self.assertEqual(len(placement.sessions), 1)
        with patch(
            "sglang.srt.managers.decode_session_placement.time.monotonic",
            return_value=11,
        ):
            self.select(placement, None)
        self.assertEqual(placement.retained, [0, 0])

    def test_burst_output_allowance_does_not_create_false_capacity_spill(self):
        """A dispatch estimate must use the same units as the next native
        snapshot: used KV plus queued prompts. Decode reserves are already
        included by the placement policy, not the full max_new_tokens cap.
        """
        for output_limit in (0, 32000):
            with self.subTest(output_limit=output_limit):
                controller = DataParallelController.__new__(DataParallelController)
                controller.workers = [Mock(), Mock()]
                controller._active_workers = [0, 1]
                controller.dp_budget = DPBudget(2)
                controller.dp_budget.total_tokens = [220000, 50000]
                controller.dp_budget.total_requests = [2, 1]
                controller.dp_budget.capacity = [291840, 291840]
                controller.decode_sessions = DecodeSessionPlacement(
                    2, reserved_decode_tokens=512
                )
                requests = []
                for session, prompt in (("first", 40000), ("second", 25000)):
                    self.select(
                        controller.decode_sessions,
                        (session, None),
                        prompt=prompt,
                        output=0,
                        capacity=[291840, 291840],
                        forced=0,
                    )
                    requests.append(
                        SimpleNamespace(
                            routed_dp_rank=None,
                            input_ids=[1] * prompt,
                            sampling_params=SimpleNamespace(
                                max_new_tokens=output_limit
                            ),
                            session_id=session,
                            cache_salt=None,
                        )
                    )
                with patch(
                    "sglang.srt.managers.data_parallel_controller.sock_send"
                ) as send:
                    for req in requests:
                        controller.decode_session_tokens_scheduler(req)
                    self.assertEqual(
                        [call.args[0] for call in send.call_args_list],
                        [controller.workers[0], controller.workers[0]],
                    )
                self.assertEqual(controller.dp_budget.total_tokens, [285000, 50000])
                self.assertEqual(controller.dp_budget.total_requests, [4, 1])

    def test_controller_sends_override_once_and_remembers_it(self):
        controller = DataParallelController.__new__(DataParallelController)
        controller.workers = [Mock(), Mock()]
        controller._active_workers = [0, 1]
        controller.dp_budget = DPBudget(2)
        controller.decode_sessions = DecodeSessionPlacement(2)
        req = SimpleNamespace(
            routed_dp_rank=1,
            input_ids=[1, 2, 3],
            sampling_params=SimpleNamespace(max_new_tokens=10),
            session_id="one",
            cache_salt=None,
        )
        with patch("sglang.srt.managers.data_parallel_controller.sock_send") as send:
            controller.decode_session_tokens_scheduler(req)
            send.assert_called_once_with(controller.workers[1], req)
            send.reset_mock()
            req.routed_dp_rank = None
            controller.decode_session_tokens_scheduler(req)
            send.assert_called_once_with(controller.workers[1], req)
        self.assertEqual(controller.dp_budget.total_requests, [0, 2])


if __name__ == "__main__":
    unittest.main()
