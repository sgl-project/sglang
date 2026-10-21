"""Exercise staging retirement without importing torch or creating GPU storage."""

import pickle
import unittest
from collections import deque
from types import SimpleNamespace

from sglang.srt.disaggregation.deferred_draft_prefill import DeferredDraftPrefillPool
from sglang.srt.disaggregation.deferred_draft_prefill_lifecycle import (
    DeferredDraftPrefillReleaseRids,
)
from sglang.srt.disaggregation.deferred_draft_prefill_lifecycle import (
    DeferredDraftPrefillRetirementState as State,
)
from sglang.srt.disaggregation.deferred_draft_prefill_lifecycle import (
    DeferredDraftPrefillSourceFence,
    DeferredDraftPrefillSourceRetirements,
    combine_deferred_draft_prefill_retirement_states,
    deferred_draft_prefill_retirement_key,
    merge_deferred_draft_prefill_retirements,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class MetadataAllocator:
    def __init__(self, size):
        self.free_slots = deque(range(size))

    def alloc(self):
        return self.free_slots.popleft()

    def free(self, index):
        self.free_slots.append(index)

    def available_size(self):
        return len(self.free_slots)


def pool(size=1):
    return DeferredDraftPrefillPool(
        size, 8, 4, "torch.bfloat16", "cpu", allocate_storage=False
    )


def request(rid="request", room=100, generation="generation"):
    return SimpleNamespace(
        rid=rid,
        bootstrap_room=room,
        pd_deferred_draft_source_generation=generation,
        origin_input_ids=[1, 2, 3],
        metadata_buffer_index=-1,
        disagg_kv_sender=SimpleNamespace(
            deferred_draft_prefill_fence=DeferredDraftPrefillSourceFence()
        ),
    )


def stage(size=1, rid="request", room=100):
    hidden = pool(size)
    allocator = MetadataAllocator(size)
    registry = DeferredDraftPrefillSourceRetirements(allocator)
    req = request(rid, room)
    hidden.alloc(req)
    req.deferred_draft_prefill_pool = hidden
    req.metadata_buffer_index = allocator.alloc()
    return req, hidden, allocator, registry


def fence(req):
    return req.disagg_kv_sender.deferred_draft_prefill_fence


class TestDeferredDraftPrefillSourceFence(unittest.TestCase):
    def test_abort_before_publication_is_safe_and_closes_future_submission(self):
        source = DeferredDraftPrefillSourceFence()
        self.assertEqual(source.retirement_state, State.HOLD)
        self.assertEqual(source.seal(), State.FREE)
        self.assertFalse(source.submit())
        self.assertFalse(source.submitted)

    def test_queued_but_not_dequeued_final_chunk_is_not_drained(self):
        source = DeferredDraftPrefillSourceFence()
        self.assertTrue(source.submit())
        self.assertFalse(source.submit())
        # No worker has touched the queued chunk, so its outstanding counter
        # can still be zero. Publication must nevertheless hold ownership.
        self.assertEqual(source.seal(), State.HOLD)
        self.assertEqual(source.settled(), State.FREE)

    def test_completed_worker_does_not_free_before_sender_retirement(self):
        source = DeferredDraftPrefillSourceFence()
        source.submit()
        self.assertEqual(source.settled(), State.HOLD)
        self.assertEqual(source.seal(), State.FREE)

    def test_timeout_poison_survives_a_worker_finally_completion(self):
        source = DeferredDraftPrefillSourceFence()
        source.submit()
        source.seal()
        self.assertEqual(source.poison("timeout left a batch alive"), State.QUARANTINE)
        self.assertEqual(source.settled(), State.QUARANTINE)
        self.assertEqual(source.seal(), State.QUARANTINE)
        self.assertFalse(source.submit())

    def test_old_generation_completion_cannot_settle_a_new_sender(self):
        old, new = DeferredDraftPrefillSourceFence(), DeferredDraftPrefillSourceFence()
        self.assertNotEqual(old.generation, new.generation)
        old.submit()
        new.submit()
        old.seal()
        new.seal()
        old.settled()
        self.assertEqual(old.retirement_state, State.FREE)
        self.assertEqual(new.retirement_state, State.HOLD)


class TestDeferredDraftPrefillQuarantine(unittest.TestCase):
    def test_quarantine_is_permanent_and_repeated_free_cannot_restore_it(self):
        hidden = pool()
        req = request()
        hidden.alloc(req)
        with self.assertLogs(
            "sglang.srt.disaggregation.deferred_draft_prefill", "WARNING"
        ):
            self.assertTrue(hidden.quarantine(req, reason="late RDMA write"))
        self.assertEqual(hidden.quarantined_slots, 1)
        self.assertEqual(hidden.available_slots, 0)
        self.assertFalse(hidden.quarantine(req, reason="duplicate retirement"))
        self.assertFalse(hidden.free_source(req))
        hidden.reclaim()
        self.assertFalse(hidden.try_alloc(request("next")))

    def test_same_rid_after_quarantine_gets_fresh_storage(self):
        hidden = pool(2)
        old, new = request(), request(room=101)
        old_slot = hidden.alloc(old)
        hidden.quarantine(old, reason="unknown drain")
        self.assertFalse(hidden.has_staged_rid(old.rid))
        self.assertNotEqual(hidden.alloc(new), old_slot)
        self.assertFalse(hidden.free_source(old))
        self.assertFalse(hidden.quarantine(old, reason="stale generation"))
        with self.assertRaises(KeyError):
            hidden.get_slot(old)
        self.assertTrue(hidden.owns(new))

    def test_active_duplicate_rid_is_not_an_idempotent_allocation(self):
        hidden = pool(2)
        owner = request()
        self.assertEqual(hidden.alloc(owner), hidden.alloc(owner))
        self.assertTrue(hidden.has_staged_rid(owner.rid))
        self.assertFalse(hidden.owns(request(room=101)))
        with self.assertRaisesRegex(ValueError, "Another active request"):
            hidden.alloc(request(room=101))
        self.assertEqual(hidden.available_slots, 1)


class TestDeferredDraftPrefillSourceRetirements(unittest.TestCase):
    def test_ordinary_queued_abort_releases_both_resources_after_worker_ack(self):
        req, hidden, allocator, registry = stage()
        queued_chunk_fence = fence(req)
        queued_chunk_fence.submit()
        self.assertTrue(registry.register(req))
        self.assertTrue(registry.register(req))
        self.assertEqual(registry.resolve(), [])
        self.assertEqual(hidden.available_slots, allocator.available_size())
        self.assertEqual(hidden.available_slots, 0)
        # The transfer worker skips the canceled final chunk and confirms it.
        queued_chunk_fence.settled()
        self.assertEqual(
            registry.resolve(), [deferred_draft_prefill_retirement_key(req)]
        )
        self.assertEqual(req.metadata_buffer_index, -1)
        self.assertEqual(hidden.available_slots, allocator.available_size())
        self.assertEqual(hidden.available_slots, 1)
        self.assertEqual(hidden.quarantined_slots, 0)
        self.assertFalse(registry)
        self.assertFalse(registry.register(req))

    def test_never_submitted_early_abort_releases_without_worker_or_slot_loss(self):
        req, hidden, allocator, registry = stage()
        registry.register(req)
        self.assertEqual(
            registry.snapshot(),
            {deferred_draft_prefill_retirement_key(req): State.FREE},
        )
        registry.resolve()
        self.assertEqual(hidden.available_slots, 1)
        self.assertEqual(allocator.available_size(), 1)
        self.assertFalse(fence(req).submit())

    def test_timeout_keeps_hidden_and_metadata_out_of_admission(self):
        req, hidden, allocator, registry = stage()
        fence(req).submit()
        fence(req).poison("timeout; backend batch handle lost")
        registry.register(req)
        fence(req).settled()  # A generic finally block does not prove drain.
        registry.resolve()
        self.assertEqual(req.metadata_buffer_index, -1)
        self.assertEqual(hidden.available_slots, allocator.available_size())
        self.assertEqual(hidden.available_slots, 0)
        self.assertEqual(hidden.quarantined_slots, 1)
        self.assertEqual(registry.quarantined_metadata_slots, 1)
        self.assertEqual(registry.resolve(), [])

    def test_old_pp_payload_cannot_retire_same_rid_in_a_new_bootstrap_room(self):
        req, hidden, allocator, registry = stage()
        registry.register(req)
        old_vote = registry.snapshot()
        registry.resolve(old_vote)
        newer = request(room=req.bootstrap_room + 1)
        hidden.alloc(newer)
        newer.deferred_draft_prefill_pool = hidden
        newer.metadata_buffer_index = allocator.alloc()
        registry.register(newer)
        self.assertEqual(registry.resolve(old_vote), [])
        self.assertTrue(hidden.owns(newer))
        self.assertGreaterEqual(newer.metadata_buffer_index, 0)
        self.assertEqual(len(registry), 1)

    def test_stale_request_identity_cannot_release_a_reassigned_pool_slot(self):
        req, hidden, allocator, registry = stage(2)
        registry.register(req)
        hidden.free_source(req)  # Simulate a caller violating ownership retention.
        newer = request(room=101)
        hidden.alloc(newer)
        with self.assertRaisesRegex(RuntimeError, "ownership changed"):
            registry.resolve()
        self.assertTrue(hidden.owns(newer))
        self.assertEqual(allocator.available_size(), 1)

    def test_old_pp_vote_cannot_consume_same_rid_and_room_with_new_intake_nonce(self):
        req, hidden, allocator, registry = stage()
        registry.register(req)
        old_vote = registry.snapshot()
        registry.resolve(old_vote)
        newer = request(req.rid, req.bootstrap_room, generation="new-generation")
        hidden.alloc(newer)
        newer.deferred_draft_prefill_pool = hidden
        newer.metadata_buffer_index = allocator.alloc()
        registry.register(newer)
        self.assertEqual(registry.resolve(old_vote), [])
        self.assertTrue(hidden.owns(newer))
        self.assertGreaterEqual(newer.metadata_buffer_index, 0)
        self.assertEqual(len(registry), 1)
        registry.resolve()
        self.assertEqual(hidden.available_slots, 1)
        self.assertEqual(allocator.available_size(), 1)

    def test_free_consensus_does_not_override_a_local_pending_fence(self):
        req, hidden, allocator, registry = stage()
        fence(req).submit()
        registry.register(req)
        with self.assertRaisesRegex(RuntimeError, "Unsafe"):
            registry.resolve({deferred_draft_prefill_retirement_key(req): State.FREE})
        self.assertEqual(hidden.available_slots, 0)
        self.assertEqual(allocator.available_size(), 0)

    def test_duplicate_generation_cannot_replace_a_registered_owner(self):
        req, hidden, allocator, registry = stage(2)
        registry.register(req)
        other = request(req.rid, req.bootstrap_room)
        other_pool = pool()
        other_pool.alloc(other)
        other.deferred_draft_prefill_pool = other_pool
        other.metadata_buffer_index = allocator.alloc()
        with self.assertRaisesRegex(ValueError, "already owns retirement"):
            registry.register(other)
        registry.resolve()
        self.assertEqual(req.metadata_buffer_index, -1)
        self.assertGreaterEqual(other.metadata_buffer_index, 0)
        self.assertTrue(other_pool.owns(other))

    def test_retirement_key_requires_the_shared_intake_nonce(self):
        for generation in (None, ""):
            with (
                self.subTest(generation=generation),
                self.assertRaisesRegex(ValueError, "intake generation"),
            ):
                deferred_draft_prefill_retirement_key(request(generation=generation))

    def test_missing_fence_is_unknown_provenance_not_never_submitted(self):
        req, hidden, allocator, registry = stage()
        del req.disagg_kv_sender.deferred_draft_prefill_fence
        registry.register(req)
        self.assertEqual(list(registry.snapshot().values()), [State.QUARANTINE])
        registry.resolve()
        self.assertEqual(hidden.available_slots, 0)
        self.assertEqual(allocator.available_size(), 0)


class TestDeferredDraftPrefillPipelineConsensus(unittest.TestCase):
    def merge(self, registries):
        combined = registries[0].snapshot()
        for registry in registries[1:]:
            combined = merge_deferred_draft_prefill_retirements(
                combined, registry.snapshot()
            )
        return combined

    def test_four_pipeline_ranks_hold_then_release_one_ordinary_abort_together(self):
        stages = [stage() for _ in range(4)]
        for rank, (req, _, _, registry) in enumerate(stages):
            if rank >= 2:
                fence(req).submit()
            if rank == 2:
                fence(req).settled()
            registry.register(req)
        combined = self.merge([s[3] for s in stages])
        self.assertEqual(list(combined.values()), [State.HOLD])
        for _, hidden, allocator, registry in stages:
            registry.resolve(combined)
            self.assertEqual(hidden.available_slots, 0)
            self.assertEqual(allocator.available_size(), 0)
        fence(stages[-1][0]).settled()
        combined = self.merge([s[3] for s in stages])
        self.assertEqual(list(combined.values()), [State.FREE])
        for _, hidden, allocator, registry in stages:
            registry.resolve(combined)
            self.assertEqual(hidden.available_slots, 1)
            self.assertEqual(allocator.available_size(), 1)

    def test_one_poisoned_rank_quarantines_every_source_pair_with_backpressure(self):
        stages = [stage(2) for _ in range(4)]
        for rank, (req, _, _, registry) in enumerate(stages):
            if rank:
                fence(req).submit()
            if rank == 1:
                fence(req).settled()
            if rank == 3:
                fence(req).poison("unknown backend completion")
            registry.register(req)
        combined = self.merge([s[3] for s in stages])
        self.assertEqual(list(combined.values()), [State.QUARANTINE])
        for _, hidden, allocator, registry in stages:
            registry.resolve(combined)
            self.assertEqual(hidden.quarantined_slots, 1)
            self.assertEqual(registry.quarantined_metadata_slots, 1)
            self.assertEqual(hidden.available_slots, allocator.available_size())
            next_req = request("remaining", 200)
            hidden.alloc(next_req)
            next_req.deferred_draft_prefill_pool = hidden
            next_req.metadata_buffer_index = allocator.alloc()
            self.assertEqual(hidden.available_slots, 0)
            self.assertEqual(allocator.available_size(), 0)
            registry.register(next_req)
            registry.resolve()  # A later never-published abort wastes no capacity.
            self.assertEqual(hidden.available_slots, allocator.available_size())
            self.assertEqual(hidden.available_slots, 1)

    def test_generation_intersection_requires_registration_on_every_rank(self):
        old, newer = ("same-rid", 10, "old"), ("same-rid", 10, "new")
        self.assertEqual(
            merge_deferred_draft_prefill_retirements(
                {old: State.FREE, newer: State.HOLD}, {old: State.QUARANTINE}
            ),
            {old: State.QUARANTINE},
        )
        with self.assertRaises(ValueError):
            combine_deferred_draft_prefill_retirement_states([])

    def test_release_payload_preserves_rid_membership_and_pickles_votes(self):
        key = ("request", 100, "generation")
        payload = DeferredDraftPrefillReleaseRids(["request"], {key: State.HOLD})
        restored = pickle.loads(pickle.dumps(payload))
        self.assertIsInstance(restored, DeferredDraftPrefillReleaseRids)
        self.assertIn("request", restored)
        self.assertEqual(restored, ["request"])
        self.assertEqual(restored.retirement_states, {key: State.HOLD})


if __name__ == "__main__":
    unittest.main()
