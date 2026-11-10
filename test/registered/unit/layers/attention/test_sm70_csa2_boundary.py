"""CPU tests for SM70 CSA2 ring/pending checkpoints."""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.dsv4.sm70_csa2_boundary import (
    cap_verify_commit,
    csa2_finish_forward,
    csa2_image_len,
    csa2_prepare_decode,
    csa2_prepare_extend,
    evict_recent,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _State:
    def __init__(self) -> None:
        self.swa_ring = {0: torch.zeros(4, dtype=torch.uint8)}
        self.pending_kv = {1: torch.zeros(2, dtype=torch.float32)}
        self.pending_score = {1: torch.zeros(2, dtype=torch.float32)}


class _RingState:
    """Ring slot ``p % W`` holds position ``p``; pending holds the last token."""

    W = 8

    def __init__(self, pending: bool = True) -> None:
        self.swa_ring = {0: torch.full((self.W, 1), 255, dtype=torch.uint8)}
        self.pending_kv = {1: torch.zeros(2)} if pending else {}
        self.pending_score = {1: torch.zeros(2)} if pending else {}
        self.verify_pending_kv_traj = {}
        self.verify_pending_score_traj = {}

    def extend(self, start: int, end: int) -> None:
        for pos in range(start, end):
            self.swa_ring[0][pos % self.W] = pos
        for table in (self.pending_kv, self.pending_score):
            for vec in table.values():
                vec.fill_(end - 1)

    def verify(self, start: int, commit: int, block: int = 4) -> None:
        """A verify block whose first ``commit`` inputs are published."""
        if self.pending_kv:
            traj = torch.arange(start, start + block, dtype=torch.float32)
            traj = traj[:, None].repeat(1, 2)
            self.verify_pending_kv_traj = {1: traj.clone()}
            self.verify_pending_score_traj = {1: traj.clone()}
        self.extend(start, start + commit)

    def window(self, length: int) -> list:
        """Positions the ring should hold at ``length``, by slot."""
        slots = [255] * self.W
        for pos in range(max(0, length - self.W), length):
            slots[pos % self.W] = pos
        return slots


class _Backend:
    def __init__(self, state=None) -> None:
        if state is not None:
            self._sm70_csa2 = state


class TestCsa2Boundary(CustomTestCase):
    def test_cap_verify_commit_drops_the_overshoot(self):
        # Live failure: prompt 65, max_new_tokens 8, pin 73, uncapped image 78.
        self.assertEqual(cap_verify_commit(71, 6, 73), 2)
        self.assertEqual(71 + cap_verify_commit(71, 6, 73), 73)
        self.assertEqual(cap_verify_commit(65, 6, 73), 6)
        self.assertEqual(cap_verify_commit(78, 6, 73), 0)
        self.assertEqual(cap_verify_commit(71, 6, None), 6)

    def test_evict_recent_keeps_the_tail(self):
        self.assertEqual(evict_recent([1, 2, 3], 2), [2, 3])
        self.assertEqual(evict_recent([1], 8), [1])

    def test_rewind_restores_ring_and_drops_the_tail(self):
        target = _Backend(_State())
        backends = [target]
        csa2_prepare_extend(backends, 0)
        target._sm70_csa2.swa_ring[0].fill_(1)
        target._sm70_csa2.pending_kv[1].fill_(1)
        csa2_finish_forward(backends, 4, from_extend=True)
        csa2_prepare_decode(backends)
        store = target._csa2_boundary
        self.assertIn(4, store.history)
        self.assertEqual(store.resident_end, 4)

        target._sm70_csa2.swa_ring[0].fill_(7)
        target._sm70_csa2.pending_kv[1].fill_(7)
        csa2_finish_forward(backends, 6, from_extend=False)
        csa2_prepare_extend(backends, 4)

        self.assertTrue(
            torch.equal(
                target._sm70_csa2.swa_ring[0], torch.ones(4, dtype=torch.uint8)
            )
        )
        self.assertTrue(
            torch.equal(target._sm70_csa2.pending_kv[1], torch.ones(2))
        )
        self.assertEqual(store.resident_end, 4)
        self.assertNotIn(6, store.history)
        self.assertIn(4, store.order)

    def test_prefix_zero_drops_the_resident_image(self):
        target = _Backend(_State())
        backends = [target]
        target._sm70_csa2.swa_ring[0].fill_(3)
        csa2_finish_forward(backends, 4, from_extend=True)
        csa2_prepare_decode(backends)
        self.assertIn(4, target._csa2_boundary.history)
        csa2_prepare_extend(backends, 0)
        self.assertEqual(target._csa2_boundary.history, {})
        self.assertEqual(target._csa2_boundary.resident_end, 0)
        self.assertEqual(int(target._sm70_csa2.swa_ring[0].sum()), 12)

    def test_missing_length_raises(self):
        target = _Backend(_State())
        backends = [target]
        csa2_finish_forward(backends, 4, from_extend=True)
        csa2_prepare_decode(backends)
        with self.assertRaises(RuntimeError):
            csa2_prepare_extend(backends, 2)

    def test_bonus_snap_is_filed_under_the_pin(self):
        # Scheduler pin 16, worker tip 17. The bytes stay, under the pin.
        target = _Backend(_State())
        backends = [target]
        target._sm70_csa2.swa_ring[0].fill_(1)
        csa2_finish_forward(backends, 5, from_extend=True)
        csa2_prepare_decode(backends)
        target._sm70_csa2.swa_ring[0].fill_(9)
        csa2_finish_forward(backends, 17, from_extend=False)
        csa2_prepare_extend(backends, 16)
        self.assertEqual(int(target._sm70_csa2.swa_ring[0][0]), 9)
        self.assertEqual(target._csa2_boundary.resident_end, 16)
        self.assertIn(16, target._csa2_boundary.history)
        self.assertNotIn(17, target._csa2_boundary.history)

    def test_rewind_to_bonus_pin_after_a_later_turn(self):
        # Live failure: pin 23995 filed as 23996, then the tip moved to 24717.
        target = _Backend(_State())
        backends = [target]
        target._sm70_csa2.swa_ring[0].fill_(1)
        csa2_finish_forward(backends, 5, from_extend=True)
        csa2_prepare_decode(backends)
        target._sm70_csa2.swa_ring[0].fill_(9)
        csa2_finish_forward(backends, 17, from_extend=False)
        csa2_prepare_extend(backends, 16)
        target._sm70_csa2.swa_ring[0].fill_(3)
        csa2_finish_forward(backends, 20, from_extend=False)
        csa2_prepare_extend(backends, 16)
        self.assertEqual(int(target._sm70_csa2.swa_ring[0][0]), 9)
        self.assertEqual(target._csa2_boundary.resident_end, 16)
        self.assertNotIn(20, target._csa2_boundary.history)
        self.assertNotIn(17, target._csa2_boundary.history)

    def test_verify_gap_of_two_is_filed_under_the_pin(self):
        # Third turn: pin 23696, verify tip 23698. Nothing snapshotted between.
        target = _Backend(_State())
        backends = [target]
        target._sm70_csa2.swa_ring[0].fill_(1)
        csa2_finish_forward(backends, 5, from_extend=True)
        csa2_prepare_decode(backends)
        target._sm70_csa2.swa_ring[0].fill_(9)
        csa2_finish_forward(backends, 18, from_extend=False)
        csa2_prepare_extend(backends, 16)
        self.assertEqual(int(target._sm70_csa2.swa_ring[0][0]), 9)
        self.assertEqual(target._csa2_boundary.resident_end, 16)
        self.assertIn(16, target._csa2_boundary.history)
        self.assertNotIn(18, target._csa2_boundary.history)

    def test_verify_slack_stops_at_an_intermediate_snap(self):
        target = _Backend(_State())
        backends = [target]
        target._sm70_csa2.swa_ring[0].fill_(1)
        csa2_finish_forward(backends, 16, from_extend=True)
        csa2_prepare_decode(backends)
        target._sm70_csa2.swa_ring[0].fill_(4)
        csa2_finish_forward(backends, 20, from_extend=False)
        with self.assertRaises(RuntimeError):
            csa2_prepare_extend(backends, 10)
        self.assertNotIn(10, target._csa2_boundary.history)
        self.assertIn(20, target._csa2_boundary.history)
        self.assertEqual(int(target._sm70_csa2.swa_ring[0][0]), 4)

    def test_extend_stop_one_past_the_pin_is_not_relabeled(self):
        target = _Backend(_State())
        backends = [target]
        target._sm70_csa2.swa_ring[0].fill_(1)
        csa2_finish_forward(backends, 4, from_extend=True)
        csa2_prepare_decode(backends)
        target._sm70_csa2.swa_ring[0].fill_(6)
        csa2_finish_forward(backends, 6, from_extend=True)
        with self.assertRaises(RuntimeError):
            csa2_prepare_extend(backends, 5)
        self.assertEqual(int(target._sm70_csa2.swa_ring[0][0]), 6)
        self.assertIn(6, target._csa2_boundary.history)
        self.assertNotIn(5, target._csa2_boundary.history)

    def test_draft_ring_rewinds_with_the_target(self):
        target = _Backend(_State())
        draft = _Backend(_State())
        backends = [target, draft]
        target._sm70_csa2.swa_ring[0].fill_(1)
        draft._sm70_csa2.swa_ring[0].fill_(2)
        csa2_finish_forward(backends, 4, from_extend=True)
        csa2_prepare_decode(backends)
        target._sm70_csa2.swa_ring[0].fill_(8)
        draft._sm70_csa2.swa_ring[0].fill_(9)
        csa2_finish_forward(backends, 6, from_extend=False)
        csa2_prepare_extend(backends, 4)
        self.assertEqual(int(target._sm70_csa2.swa_ring[0][0]), 1)
        self.assertEqual(int(draft._sm70_csa2.swa_ring[0][0]), 2)

    def test_disk_session_restores_rows_after_clobber(self):
        import tempfile

        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.dsv4.sm70_csa2_session import (
            request_load,
            request_spill,
            reset_handoff,
            session_key,
        )

        state = _State()
        state.kv_rows = {2: torch.arange(8, dtype=torch.uint8)}
        state.index_rows = {2: torch.arange(4, dtype=torch.uint8)}
        target = _Backend(state)
        backends = [target]
        target._sm70_csa2.swa_ring[0].fill_(4)
        csa2_finish_forward(backends, 4, from_extend=True)
        csa2_prepare_decode(backends)
        key = session_key([1, 2, 3, 4], None, None)
        directory = tempfile.mkdtemp()
        envs.SGLANG_DSV41_CSA2_SESSION_DIR.set(directory)
        envs.SGLANG_DSV41_CSA2_SESSION_KEEP.set(2)
        try:
            reset_handoff()
            request_spill(key)
            csa2_prepare_extend(backends, 0)
            self.assertEqual(target._csa2_boundary.resident_end, 0)
            target._sm70_csa2.kv_rows[2].fill_(9)
            target._sm70_csa2.swa_ring[0].fill_(9)
            request_load(key)
            csa2_prepare_extend(backends, 4)
            self.assertEqual(int(target._sm70_csa2.kv_rows[2][0]), 0)
            self.assertEqual(int(target._sm70_csa2.kv_rows[2][3]), 3)
            self.assertEqual(int(target._sm70_csa2.swa_ring[0][0]), 4)
            self.assertEqual(target._csa2_boundary.resident_end, 4)
        finally:
            envs.SGLANG_DSV41_CSA2_SESSION_DIR.clear()
            envs.SGLANG_DSV41_CSA2_SESSION_KEEP.clear()
            reset_handoff()

    def test_spill_frees_its_host_copy(self):
        """A spilled image's host copy must not outlive the spill.

        Left to the automatic GC, it can stay resident for hours behind the
        engine's frozen startup objects.
        """
        import gc
        import tempfile

        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.dsv4.sm70_csa2_session import (
            request_spill,
            reset_handoff,
            session_key,
        )

        state = _State()
        state.kv_rows = {2: torch.arange(8, dtype=torch.uint8)}
        target = _Backend(state)
        backends = [target]
        csa2_finish_forward(backends, 4, from_extend=True)
        csa2_prepare_decode(backends)
        envs.SGLANG_DSV41_CSA2_SESSION_DIR.set(tempfile.mkdtemp())
        gc.collect()
        gc.disable()
        try:
            reset_handoff()
            request_spill(session_key([1, 2, 3, 4], None, None))
            csa2_prepare_extend(backends, 0)
            gc.set_debug(gc.DEBUG_SAVEALL)
            gc.collect()
            leaked = [o for o in gc.garbage if isinstance(o, torch.UntypedStorage)]
            self.assertEqual(leaked, [])
        finally:
            gc.set_debug(0)
            gc.garbage.clear()
            gc.enable()
            envs.SGLANG_DSV41_CSA2_SESSION_DIR.clear()
            reset_handoff()

    def test_spill_files_verify_tip_under_the_pin(self):
        import tempfile

        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.dsv4.sm70_csa2_session import (
            remember,
            request_load,
            request_spill,
            reset_handoff,
            session_key,
        )

        target = _Backend(_State())
        backends = [target]
        target._sm70_csa2.swa_ring[0].fill_(1)
        csa2_finish_forward(backends, 4, from_extend=True)
        csa2_prepare_decode(backends)
        target._sm70_csa2.swa_ring[0].fill_(9)
        csa2_finish_forward(backends, 6, from_extend=False)
        ids = list(range(5))
        key = session_key(ids, None, None)
        directory = tempfile.mkdtemp()
        envs.SGLANG_DSV41_CSA2_SESSION_DIR.set(directory)
        try:
            remember(key, ids, [4, 5], None, None)
            reset_handoff()
            request_spill(key)
            csa2_prepare_extend(backends, 0)
            target._sm70_csa2.swa_ring[0].fill_(3)
            request_load(key)
            csa2_prepare_extend(backends, 5)
            self.assertEqual(int(target._sm70_csa2.swa_ring[0][0]), 9)
            self.assertEqual(target._csa2_boundary.resident_end, 5)
        finally:
            envs.SGLANG_DSV41_CSA2_SESSION_DIR.clear()
            reset_handoff()

    def _ring(self, backend) -> list:
        return [int(v) for v in backend._sm70_csa2.swa_ring[0].reshape(-1)]

    def _pending(self, backend) -> int:
        return int(backend._sm70_csa2.pending_kv[1][0])

    def test_stop_on_a_draft_cuts_the_verify_image_to_the_pin(self):
        # Prompt 10; one verify publishes positions 10..13; the request
        # stopped at position 11, so the pin is 12.
        target = _Backend(_RingState())
        draft = _Backend(_RingState(pending=False))
        backends = [target, draft]
        target._sm70_csa2.extend(0, 10)
        draft._sm70_csa2.extend(0, 10)
        csa2_finish_forward(backends, 10, from_extend=True)
        csa2_prepare_decode(backends)
        target._sm70_csa2.verify(10, 4)
        draft._sm70_csa2.verify(10, 4)
        csa2_finish_forward(backends, 14, from_extend=False, start=10)
        csa2_prepare_extend(backends, 12)
        want = _RingState().window(12)
        self.assertEqual(self._ring(target), want)
        self.assertEqual(self._ring(draft), want)
        self.assertEqual(self._pending(target), 11)
        store = target._csa2_boundary
        self.assertEqual(store.resident_end, 12)
        self.assertIn(12, store.history)
        self.assertNotIn(14, store.history)

    def test_overlap_step_after_the_stop_is_cut_too(self):
        # The stop step publishes 10..12 (stop at 11, pin 12); the overlap
        # loop's extra step publishes 13..14 before the pin lands.
        target = _Backend(_RingState())
        backends = [target]
        target._sm70_csa2.extend(0, 10)
        csa2_finish_forward(backends, 10, from_extend=True)
        csa2_prepare_decode(backends)
        target._sm70_csa2.verify(10, 3)
        csa2_finish_forward(backends, 13, from_extend=False, start=10)
        target._sm70_csa2.verify(13, 2)
        csa2_finish_forward(backends, 15, from_extend=False, start=13)
        csa2_prepare_extend(backends, 12)
        self.assertEqual(self._ring(target), _RingState().window(12))
        self.assertEqual(self._pending(target), 11)
        self.assertEqual(target._csa2_boundary.resident_end, 12)

    def test_pin_at_an_earlier_verify_image_restores_it(self):
        target = _Backend(_RingState())
        backends = [target]
        target._sm70_csa2.extend(0, 10)
        csa2_finish_forward(backends, 10, from_extend=True)
        csa2_prepare_decode(backends)
        target._sm70_csa2.verify(10, 3)
        csa2_finish_forward(backends, 13, from_extend=False, start=10)
        target._sm70_csa2.verify(13, 1)
        csa2_finish_forward(backends, 14, from_extend=False, start=13)
        csa2_prepare_extend(backends, 13)
        self.assertEqual(self._ring(target), _RingState().window(13))
        self.assertEqual(self._pending(target), 12)

    def test_cut_then_spill_writes_the_pin_image(self):
        import tempfile

        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.dsv4.sm70_csa2_session import (
            remember,
            request_load,
            request_spill,
            reset_handoff,
            session_key,
        )

        target = _Backend(_RingState())
        backends = [target]
        target._sm70_csa2.extend(0, 10)
        csa2_finish_forward(backends, 10, from_extend=True)
        csa2_prepare_decode(backends)
        target._sm70_csa2.verify(10, 4)
        csa2_finish_forward(backends, 14, from_extend=False, start=10)
        ids = list(range(12))
        key = session_key(ids, None, None)
        envs.SGLANG_DSV41_CSA2_SESSION_DIR.set(tempfile.mkdtemp())
        try:
            remember(key, ids, [10, 12], None, None)
            reset_handoff()
            request_spill(key)
            csa2_prepare_extend(backends, 0)
            target._sm70_csa2.swa_ring[0].fill_(7)
            target._sm70_csa2.pending_kv[1].fill_(7)
            request_load(key)
            csa2_prepare_extend(backends, 12)
            self.assertEqual(self._ring(target), _RingState().window(12))
            self.assertEqual(self._pending(target), 11)
            self.assertEqual(target._csa2_boundary.resident_end, 12)
        finally:
            envs.SGLANG_DSV41_CSA2_SESSION_DIR.clear()
            reset_handoff()

    def test_image_len_reports_the_newest_tip(self):
        target = _Backend(_RingState())
        backends = [target]
        target._sm70_csa2.extend(0, 10)
        csa2_finish_forward(backends, 10, from_extend=True)
        self.assertEqual(csa2_image_len(), 10)
        csa2_prepare_extend(backends, 0)
        self.assertIsNone(csa2_image_len())
