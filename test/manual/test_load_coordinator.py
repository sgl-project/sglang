# SPDX-License-Identifier: Apache-2.0
"""Real CPU/Gloo collectives with the same calls as SGLang GroupCoordinator.

The adapter avoids importing the rest of SGLang's accelerator runtime. This is
a transport/contract test, not a full-engine integration test.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path

import torch.distributed as dist

from sglang.srt.weight_cache.coordinator import Ballot, vote


class CpuGroup:
    def __init__(self, ranks, cpu_group):
        self.ranks = ranks
        self.world_size = len(ranks)
        self.cpu_group = cpu_group

    def barrier(self):
        dist.barrier(group=self.cpu_group)

    def all_gather_object(self, obj):
        objects = [None] * self.world_size
        dist.all_gather_object(objects, obj, group=self.cpu_group)
        return objects


def _capture(group, ballot):
    try:
        ballots = vote(group=group, ballot=ballot)
        return {"payloads": [b.payload.hex() for b in ballots]}
    except RuntimeError as error:
        return {"error": str(error)}


def _worker(rank, root, mode, finished):
    world_size = 4 if mode == "agreement" else 2
    dist.init_process_group(
        "gloo",
        init_method=Path(root, "rendezvous").as_uri(),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=15),
    )
    try:
        if mode == "missing":
            pg = dist.new_group([0, 1], backend="gloo", timeout=timedelta(seconds=2))
            if rank == 0:
                result = _capture(CpuGroup([0, 1], pg), Ballot(agree=b"v1"))
                finished.set()
            else:
                # Stay alive but deliberately omit this phase's collective.
                finished.wait(8)
                result = {"omitted_vote": True}
        else:
            group = CpuGroup(list(range(world_size)), dist.group.WORLD)
            result = {}
            scenarios = {
                "success": Ballot(agree=b"spec/v1", payload=bytes([rank])),
                "raw": Ballot(agree=b"spec/"),
                "error": Ballot(
                    agree=b"spec/v1", error="checkpoint missing" if rank == 1 else None
                ),
                "version": Ballot(agree=b"spec/v2" if rank == 2 else b"spec/v1"),
                "partial_hit": Ballot(agree=b"spec/" if rank == 3 else b"spec/v1"),
                "spec": Ballot(agree=b"other/v1" if rank == 1 else b"spec/v1"),
                "empty_error": Ballot(error="" if rank == 0 else None),
                "multiple_errors": Ballot(
                    error=f"failure-{rank}" if rank in (1, 3) else None
                ),
                "success_after_failure": Ballot(
                    agree=b"next-phase", payload=bytes([rank])
                ),
            }
            for name, ballot in scenarios.items():
                result[name] = _capture(group, ballot)

            # All world ranks create groups in order, but only members vote.
            subgroup = dist.new_group([1, 3], backend="gloo")
            if rank in (1, 3):
                result["subgroup"] = _capture(
                    CpuGroup([1, 3], subgroup),
                    Ballot(
                        agree=b"same",
                        error="subgroup failure" if rank == 3 else None,
                    ),
                )
            singleton = dist.new_group([2], backend="gloo")
            if rank == 2:
                result["singleton"] = _capture(
                    CpuGroup([2], singleton), Ballot(payload=b"rank2")
                )
            dist.barrier()
        Path(root, f"rank-{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


class CoordinatorTests(unittest.TestCase):
    def run_workers(self, mode):
        context = mp.get_context("spawn")
        count = 4 if mode == "agreement" else 2
        with tempfile.TemporaryDirectory() as root:
            finished = context.Event()
            processes = [
                context.Process(target=_worker, args=(rank, root, mode, finished))
                for rank in range(count)
            ]
            try:
                for process in processes:
                    process.start()
                deadline = time.monotonic() + 45
                for process in processes:
                    process.join(max(0, deadline - time.monotonic()))
                self.assertFalse(
                    any(p.is_alive() for p in processes), "workers hung"
                )
                self.assertEqual([p.exitcode for p in processes], [0] * count)
                return [
                    json.loads(Path(root, f"rank-{rank}.json").read_text())
                    for rank in range(count)
                ]
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(5)

    def test_agreement_and_failures(self):
        results = self.run_workers("agreement")
        for name in results[0]:
            with self.subTest(case=name):
                self.assertTrue(
                    all(row[name] == results[0][name] for row in results)
                )
        self.assertEqual(results[0]["success"]["payloads"], ["00", "01", "02", "03"])
        self.assertEqual(results[0]["raw"]["payloads"], [""] * 4)
        self.assertEqual(
            results[0]["success_after_failure"]["payloads"],
            ["00", "01", "02", "03"],
        )
        for name in ("version", "partial_hit", "spec"):
            self.assertIn("agreement mismatch", results[0][name]["error"])
        self.assertIn("rank 1: checkpoint missing", results[0]["error"]["error"])
        self.assertIn("rank 0:", results[0]["empty_error"]["error"])
        self.assertIn("rank 1: failure-1", results[0]["multiple_errors"]["error"])
        self.assertIn("rank 3: failure-3", results[0]["multiple_errors"]["error"])
        self.assertEqual(results[1]["subgroup"], results[3]["subgroup"])
        self.assertIn("rank 3: subgroup failure", results[1]["subgroup"]["error"])
        self.assertEqual(results[2]["singleton"]["payloads"], [b"rank2".hex()])

    def test_missing_participant_times_out(self):
        results = self.run_workers("missing")
        self.assertIn("error", results[0])
        self.assertNotIn("Model load vote failed", results[0]["error"])
        self.assertTrue(results[1]["omitted_vote"])


if __name__ == "__main__":
    unittest.main()
