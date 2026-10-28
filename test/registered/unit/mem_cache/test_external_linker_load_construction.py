"""CPU regressions for rank-local allocation failure before a linker load."""

import datetime
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.unified_cache.components import (
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    ExternalCacheHitMarker,
    UnifiedCacheLinkerWrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Component:
    def __init__(self, name, failure=None):
        self.name = name
        self.failure = failure
        self.live = 0

    def build_external_linker_transfer(self, phase, node, keys):
        assert phase is LinkerTransferPhase.LOAD
        if self.failure == "none":
            return None
        if self.failure == "error":
            raise RuntimeError("allocation failed")
        if self.failure == "interrupt":
            raise KeyboardInterrupt("allocation interrupted")
        self.live += 1
        return SimpleNamespace(name=self.name, device_indices=torch.arange(4))

    def update_external_linker_load(self, phase, req, full, transfer, prefix_len):
        if phase is ExternalLinkerLoadPhase.ABORT:
            self.live -= 1
        return transfer


def _wrapper(failure, reduce):
    components = (_Component(PoolName.KV), _Component(PoolName.SWA, failure))
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = SimpleNamespace(page_size=4, _all_reduce_attn_groups=reduce)
    wrapper.cache_linker = SimpleNamespace(load=Mock(return_value=True))
    wrapper._components = components
    wrapper._skip_swa = False
    wrapper.hit_markers = {"r": ExternalCacheHitMarker(["page"], 0)}
    wrapper.private_load_indices = {}
    wrapper.inflight_load_rids = []
    req = SimpleNamespace(rid="r", last_node=7, swa_branching_seqlen=None)
    return wrapper, components, req


def _construction_rank(rank, rendezvous, results_dir):
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        outcomes = {}
        for failure in ("none", "error", "interrupt", None):
            reductions = []

            def reduce(value, op):
                reductions.append(value.numel())
                dist.all_reduce(value, op=op)

            wrapper, components, req = _wrapper(failure if rank == 0 else None, reduce)
            result, error = None, None
            try:
                result = wrapper.load_back(req)[0]
            except BaseException as exception:
                error = type(exception).__name__
            # Every rank must still be able to enter the next scheduler verdict.
            reduce(torch.ones(3, dtype=torch.int), dist.ReduceOp.MIN)
            outcomes[str(failure)] = {
                "loaded": result,
                "error": error,
                "live": [component.live for component in components],
                "queued": wrapper.cache_linker.load.call_count,
                "reductions": reductions,
            }
        Path(results_dir, f"rank-{rank}.json").write_text(json.dumps(outcomes))
    finally:
        dist.destroy_process_group()


class TestExternalLoadConstruction(unittest.TestCase):
    def test_single_rank_failures_roll_back_every_rank(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(
                _construction_rank,
                args=(str(Path(root, "rendezvous")), root),
                nprocs=2,
                join=True,
            )
            for rank in range(2):
                outcomes = json.loads(Path(root, f"rank-{rank}.json").read_text())
                for failure in ("none", "error", "interrupt"):
                    with self.subTest(rank=rank, failure=failure):
                        outcome = outcomes[failure]
                        error = (
                            {
                                "error": "RuntimeError",
                                "interrupt": "KeyboardInterrupt",
                            }.get(failure)
                            if rank == 0
                            else None
                        )
                        self.assertEqual(outcome["error"], error)
                        self.assertEqual(outcome["loaded"], None if error else 0)
                        self.assertEqual(outcome["live"], [0, 0])
                        self.assertEqual(outcome["queued"], 0)
                        self.assertEqual(outcome["reductions"], [1, 3])
                self.assertEqual(
                    outcomes["None"],
                    {
                        "loaded": 4,
                        "error": None,
                        "live": [1, 1],
                        "queued": 1,
                        "reductions": [1, 3],
                    },
                )

    def test_collective_error_rolls_back_local_allocations(self):
        def reduce(value, op):
            raise RuntimeError("collective failed")

        wrapper, components, req = _wrapper(None, reduce)
        with self.assertRaisesRegex(RuntimeError, "collective failed"):
            wrapper.load_back(req)
        self.assertEqual([component.live for component in components], [0, 0])
        wrapper.cache_linker.load.assert_not_called()

    def test_construction_error_remains_primary_if_rollback_fails(self):
        wrapper, components, req = _wrapper("error", lambda value, op: None)
        components[0].update_external_linker_load = Mock(
            side_effect=RuntimeError("rollback failed")
        )
        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
            wrapper.load_back(req)
        wrapper.cache_linker.load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
