"""UMBP offload must not copy KV before the forward that writes it finishes."""

import unittest
from queue import Queue
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import UMBPDirectLinker
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="extra-a", runner_config="1-gpu-small")


def _linker(producer_stream=None):
    linker = UMBPDirectLinker.__new__(UMBPDirectLinker)
    linker._offload_queue = Queue()
    linker._gc_frozen = True  # short-circuits _freeze_gc_once
    linker._stats = {"offload": 0, "offload_batches": 0}
    linker.pool_group = SimpleNamespace(
        resolve_transfers=lambda _transfers, allow_partial: ["transfer"]
    )
    if producer_stream is not None:
        linker.set_kv_producer_stream(producer_stream)
    return linker


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestOffloadWaitsForProducer(CustomTestCase):
    def test_offload_copies_after_the_producer_finishes(self):
        forward = torch.cuda.Stream()
        kv = torch.full((1 << 20,), -1.0, device="cuda")
        torch.cuda.synchronize()
        linker = _linker(forward)
        # A checkpointed chunk whose forward is still writing its pages.
        with torch.cuda.stream(forward):
            torch.cuda._sleep(600_000_000)
            kv.fill_(42.0)
        seen = []

        def build_plans(*_args, **_kwargs):
            # The offload copy starts here; snapshot what it would read.
            seen.append(kv.cpu())
            return []

        linker._build_load_plans = build_plans
        try:
            self.assertTrue(linker.offload(["transfer"]))
            self.assertTrue(linker._run_offload([linker._offload_queue.get_nowait()]))
            self.assertEqual(len(seen), 1)
            self.assertTrue(
                torch.all(seen[0] == 42.0).item(),
                f"copied {seen[0].unique().tolist()}",
            )
        finally:
            forward.synchronize()

    def test_without_a_producer_stream_only_the_caller_is_awaited(self):
        linker = _linker()
        self.assertTrue(linker.offload(["transfer"]))
        _, ready_events = linker._offload_queue.get_nowait()
        self.assertEqual(len(ready_events), 1)


if __name__ == "__main__":
    unittest.main()
