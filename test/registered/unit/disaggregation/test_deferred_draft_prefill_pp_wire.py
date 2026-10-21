"""PP retirement votes survive real transport and protect source ownership."""

import tempfile
import unittest
from array import array
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.disaggregation.deferred_draft_prefill import (
    DeferredDraftPrefillPool,
)
from sglang.srt.disaggregation.deferred_draft_prefill import (
    DeferredDraftPrefillRetirementState as State,
)
from sglang.srt.disaggregation.deferred_draft_prefill import (
    DeferredDraftPrefillSourceFence,
    DeferredDraftPrefillSourceRetirements,
    merge_deferred_draft_prefill_retirements,
)
from sglang.srt.disaggregation.utils import ReqToMetadataIdxAllocator
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils.common import broadcast_pyobj, point_to_point_pyobj
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _request(generation):
    req = Req(
        "same-rid",
        "",
        array("q", [1, 2]),
        SamplingParams(max_new_tokens=2),
        bootstrap_room=7,
    )
    req.pd_deferred_draft_source_generation = generation
    req.disagg_kv_sender = SimpleNamespace(
        deferred_draft_prefill_fence=DeferredDraftPrefillSourceFence()
    )
    return req


def _transport_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        # The tuple stays nonempty even after the normal RID queue drains.
        payload = ([], {("same-rid", 7, "generation"): State.HOLD})
        received = point_to_point_pyobj(
            payload if rank == 0 else [], rank, group=dist.group.WORLD, src=0, dst=1
        )
        received = broadcast_pyobj(
            received if rank == 1 else [], rank, dist_group=dist.group.WORLD, src=1
        )
        assert received == payload
    finally:
        dist.destroy_process_group()


class TestDeferredDraftPrefillPipeline(CustomTestCase):
    def test_empty_rid_votes_survive_real_p2p_then_broadcast(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _transport_worker,
                args=(str(Path(directory) / "rendezvous"),),
                nprocs=2,
                join=True,
            )

    def test_pp_releases_only_drained_generations_and_quarantines_unknown_writes(self):
        for final_state in (State.FREE, State.QUARANTINE):
            with self.subTest(final_state=final_state):
                stages = []
                for rank in range(2):
                    pool = DeferredDraftPrefillPool(
                        1, 8, 4, "torch.bfloat16", "cpu", allocate_storage=False
                    )
                    allocator = ReqToMetadataIdxAllocator(1)
                    registry = DeferredDraftPrefillSourceRetirements(allocator)
                    req = _request("current")
                    req.deferred_draft_prefill_pool = pool
                    pool.alloc(req)
                    req.metadata_buffer_index = allocator.alloc()
                    if rank == 1:
                        req.disagg_kv_sender.deferred_draft_prefill_fence.submit()
                    registry.register(req)
                    stages.append((req, pool, allocator, registry))

                def votes():
                    return merge_deferred_draft_prefill_retirements(
                        stages[0][3].snapshot(), stages[1][3].snapshot()
                    )

                held = votes()
                self.assertEqual(list(held.values()), [State.HOLD])
                for _, pool, allocator, registry in stages:
                    registry.resolve(held)
                    self.assertEqual(
                        (pool.available_slots, allocator.available_size()), (0, 0)
                    )
                fence = stages[1][0].disagg_kv_sender.deferred_draft_prefill_fence
                if final_state == State.FREE:
                    fence.settled()
                else:
                    fence.poison("Unknown transfer drain")
                    fence.settled()  # Cleanup cannot turn an unknown write into proof.
                agreed = votes()
                self.assertEqual(list(agreed.values()), [final_state])
                for req, pool, allocator, registry in stages:
                    registry.resolve(agreed)
                    self.assertEqual(len(registry), 0)
                    if final_state == State.QUARANTINE:
                        self.assertEqual(
                            (pool.available_slots, allocator.available_size()), (0, 0)
                        )
                        self.assertFalse(pool.try_alloc(_request("next")))
                    else:
                        self.assertEqual(
                            (pool.available_slots, allocator.available_size()), (1, 1)
                        )
                        newer = _request("next")
                        newer.deferred_draft_prefill_pool = pool
                        pool.alloc(newer)
                        newer.metadata_buffer_index = allocator.alloc()
                        registry.register(newer)
                        registry.resolve(agreed)  # A delayed vote for the old intake.
                        self.assertTrue(pool.owns(newer))
                        self.assertFalse(pool.free_source(req))
                        registry.resolve()
                        self.assertEqual(
                            (pool.available_slots, allocator.available_size()), (1, 1)
                        )


if __name__ == "__main__":
    unittest.main()
