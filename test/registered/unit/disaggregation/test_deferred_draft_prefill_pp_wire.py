"""Preserve retirement-only PP messages through the real CPU transports."""

import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.disaggregation.deferred_draft_prefill_lifecycle import (
    DeferredDraftPrefillReleaseRids,
    DeferredDraftPrefillRetirementState,
    pack_deferred_draft_prefill_release,
    unpack_deferred_draft_prefill_release,
)
from sglang.srt.utils.common import broadcast_pyobj, point_to_point_pyobj
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _cpu_transport_worker(rank, rendezvous, output_dir):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    key = ("request", 100, "generation")
    votes = {key: DeferredDraftPrefillRetirementState.FREE}
    try:
        for rids in ([], ["request"]):
            release = DeferredDraftPrefillReleaseRids(rids, retirement_states=votes)
            # The source's RID list may be empty once normal inflight requests
            # have drained. Generic transport must still carry its votes.
            wire = pack_deferred_draft_prefill_release(release)
            assert len(wire) > 0
            received = point_to_point_pyobj(
                wire if rank == 0 else [],
                rank,
                group=dist.group.WORLD,
                src=0,
                dst=1,
            )
            # Exercise the second len(data)==0 fast path before unpacking,
            # matching the PP receiver's subsequent attention TP broadcast.
            received = broadcast_pyobj(
                received if rank == 1 else [],
                rank,
                dist_group=dist.group.WORLD,
                src=1,
            )
            restored = unpack_deferred_draft_prefill_release(received)
            assert isinstance(restored, DeferredDraftPrefillReleaseRids)
            assert list(restored) == rids
            assert restored.retirement_states == votes
        for plain in ([], ["ordinary_request"]):
            assert pack_deferred_draft_prefill_release(plain) is plain
            received = point_to_point_pyobj(
                plain, rank, group=dist.group.WORLD, src=0, dst=1
            )
            received = broadcast_pyobj(
                received if rank == 1 else [],
                rank,
                dist_group=dist.group.WORLD,
                src=1,
            )
            assert type(received) is list and received == plain
            assert unpack_deferred_draft_prefill_release(received) is received
        # Demonstrate why direct pickle-only coverage missed this regression:
        # the actual transport intentionally discards an empty list subclass.
        raw_empty = DeferredDraftPrefillReleaseRids([], retirement_states=votes)
        received = point_to_point_pyobj(
            raw_empty, rank, group=dist.group.WORLD, src=0, dst=1
        )
        if rank == 1:
            assert received == [] and not hasattr(received, "retirement_states")
        dist.barrier()
        Path(output_dir, f"rank{rank}.json").write_text(
            json.dumps({"rank": rank, "status": "PASS", "cuda_used": False}) + "\n"
        )
    finally:
        dist.destroy_process_group()


class TestDeferredDraftPrefillPipelineWire(unittest.TestCase):
    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(), "requires CPU Gloo"
    )
    def test_two_rank_real_p2p_then_broadcast_keeps_empty_rid_votes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mp.spawn(
                _cpu_transport_worker,
                args=(str(root / "rendezvous"), str(root)),
                nprocs=2,
                join=True,
            )
            for rank in range(2):
                result = json.loads((root / f"rank{rank}.json").read_text())
                self.assertEqual(result["status"], "PASS")
                self.assertFalse(result["cuda_used"])


if __name__ == "__main__":
    unittest.main()
