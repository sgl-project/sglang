"""Overlap mixed-chunk relay gather: mix_running_indices lifetime.

One allocator-lifetime seam of the overlap scheduler, demonstrated with the
production ownership graph intact. What is REAL in every arm: the producer
functions (ScheduleBatch record + ring semantics, resolve_forward_inputs),
the stream discipline (the gather enqueued on a real forward stream), and
the reference lifetimes (objects die by natural scope exit or by the
production sequence that drops them -- no reference is manually deleted that
production keeps). What is COMPRESSED for determinism, and only that: kernel
latency (torch.cuda._sleep holds the forward stream the way a deep kernel
queue does) and allocation pressure (an immediate same-stream allocation
stands in for the next scheduler round's allocations). Reachability is
allocator-config-dependent: the default native allocator's free-list
eventing happens to order the reuse write after pending cross-stream
readers in this pattern (measured 0/40 corruptions), so the corruption arm
runs in a subprocess pinned to PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,
where the same production ordering corrupts ~19/20 (backend:cudaMallocAsync
behaves the same); the contract (torch.Tensor.record_stream docs) is
violated either way.

The seam (non-spec overlap + mixed chunked prefill): mix_with_running
aliases the running batch's req_pool_indices into mix_running_indices.
last_batch / the 2-slot record ring usually keep the running batch alive,
but that ages out: intervening non-mixed prefills advance last_batch and
overwrite both ring slots; a later mix leaves the staging field as the LAST
owner, and resolve_forward_inputs clears it right after enqueueing the
forward-stream gather. The fix record_streams the indices on the gathering
stream (CPU-device guarded).
"""

import dataclasses
import gc
import unittest
import weakref

import torch

from sglang.srt.managers.overlap_utils import FutureMap, resolve_forward_inputs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")

_GPU_DELAY_CYCLES = 2_000_000_000  # ~1s: must outlast full-process gc.collect()

_DIST_READY = False


def _ensure_single_rank_parallel_state():
    """resolve_forward_inputs' staging path consults the parallel state;
    without it the run leaves a retaining reference on the staged indices
    (verified: the weakref arm fails in isolation without this init). Give
    it the real single-rank state, same as other unit tests that drive
    real production functions."""
    global _DIST_READY
    if _DIST_READY:
        return
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        backend="nccl",
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29811",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)
    _DIST_READY = True


def _make_schedule_batch(**attrs):
    """A ScheduleBatch shell with every dataclass field present (None): the
    real record/snapshot paths getattr every field, and the real
    prepare_for_draft_extend setattrs onto it like on any batch."""
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    batch = object.__new__(ScheduleBatch)
    for f in dataclasses.fields(ScheduleBatch):
        setattr(batch, f.name, None)
    for name, value in attrs.items():
        setattr(batch, name, value)
    return batch


def _record_ring_slot(batch):
    """Mirror Scheduler.record_batch_in_overlap exactly (scheduler.py): the
    dataclass-fields snapshot plus the batch object, as a list so workers can
    extend it with extra_keep_alive_refs."""
    attr_snapshot = [getattr(batch, f.name, None) for f in dataclasses.fields(batch)]
    return [batch, attr_snapshot]


class TestMixRunningIndicesLifetime(CustomTestCase):
    """Seam 2: the aging-out sequence that makes the mixed gather's indices
    lose their last owner, walked with the production data structures."""

    @classmethod
    def setUpClass(cls):
        _ensure_single_rank_parallel_state()

    @staticmethod
    def _future_map(vocab_rows, device):
        fm = object.__new__(FutureMap)
        fm.output_tokens_buf = vocab_rows
        fm.spec_algo = SpeculativeAlgorithm.NONE
        return fm

    def _age_out_sequence(self, device):
        """scheduler-lifecycle walk: R decode batch -> two non-mixed prefills
        overwrite the 2-slot ring and advance last_batch -> a later prefill
        mixes R -> running_batch replaced -> staging cleared by
        resolve_forward_inputs. Returns (weakref to R.req_pool_indices,
        the mixed batch, the future map)."""
        rpi = torch.tensor([2, 5], dtype=torch.int64, device=device)
        running = _make_schedule_batch(req_pool_indices=rpi, reqs=[])
        wr = weakref.ref(rpi)

        ring = [None, None]
        ring[0] = _record_ring_slot(running)  # R's decode round
        last_batch = running

        for _ in range(2):  # non-mixed prefills age R out of the ring
            prefill = _make_schedule_batch(reqs=[])
            ring[0], ring[1] = ring[1], _record_ring_slot(prefill)
            last_batch = prefill

        mixed = _make_schedule_batch(
            reqs=[],
            device=device,
            # the alias mix_with_running establishes (schedule_batch.py):
            mix_running_indices=running.req_pool_indices,
            prefill_input_ids_cpu=torch.tensor([11, 12], dtype=torch.int64),
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        # scheduler replaces the running batch; R's object dies.
        del running, rpi
        return wr, mixed, last_batch, ring

    def test_aging_out_drops_last_reference(self):
        device = "cpu"
        wr, mixed, last_batch, ring = self._age_out_sequence(device)
        # Before the forward: staging is the last owner.
        gc.collect()
        self.assertIsNotNone(wr(), "staging field should still own the indices")
        fm = self._future_map(torch.arange(10, dtype=torch.int64), device)
        resolve_forward_inputs(mixed, fm)  # real function; clears staging
        gc.collect()
        self.assertIsNone(
            wr(),
            "after resolve_forward_inputs clears the staging field, nothing "
            "owns the indices while the gather may still be queued",
        )
        del last_batch, ring

    def _gather_round_corrupted(self) -> bool:
        """One aging-out round on the GPU: real resolve_forward_inputs on a
        real side stream; returns whether the gathered decode ids were
        corrupted by scheduler-stream block reuse."""
        device = "cuda"
        torch.cuda.synchronize()
        wr, mixed, last_batch, ring = self._age_out_sequence(device)
        pool = torch.arange(100, 110, dtype=torch.int64, device=device)
        fm = self._future_map(pool, device)
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            torch.cuda._sleep(_GPU_DELAY_CYCLES)
            resolve_forward_inputs(mixed, fm)  # real fn (carries the fix)
            out = mixed.input_ids + 0
        gc.collect()  # last host ref (staging already cleared) dies here
        # scheduler-stream reuse pressure (indices were allocated on the
        # default/scheduler stream in _age_out_sequence)
        decoys = [
            torch.full((2,), 999, dtype=torch.int64, device=device) for _ in range(4)
        ]
        torch.cuda.synchronize()
        del decoys, last_batch, ring
        result = out.cpu().tolist()
        del out
        gc.collect()
        torch.cuda.synchronize()
        return result != [11, 12, 102, 105]

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_gather_survives_aging_out(self):
        self.assertEqual(_run_arm_in_subprocess("seam2_fix"), "CLEAN")

    def test_cpu_device_mixed_gather_still_works(self):
        device = "cpu"
        wr, mixed, last_batch, ring = self._age_out_sequence(device)
        fm = self._future_map(torch.arange(100, 110, dtype=torch.int64), device)
        resolve_forward_inputs(mixed, fm)
        self.assertEqual(mixed.input_ids.tolist(), [11, 12, 102, 105])


def _run_arm_in_subprocess(arm: str) -> str:
    """Run a corruption arm in a subprocess pinned to the allocator config
    under which the lifetime violation is observable (the parent process may
    already have initialized CUDA with a different allocator)."""
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    out = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--arm", arm],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    marker = [l for l in out.stdout.splitlines() if l.startswith("ARM_RESULT:")]
    assert (
        marker
    ), f"arm subprocess produced no result: {out.stdout[-2000:]} {out.stderr[-2000:]}"
    return marker[-1].split(":", 1)[1].strip()


def _arm_main(arm: str) -> None:
    torch.cuda.init()
    case = TestMixRunningIndicesLifetime("test_gather_survives_aging_out")
    # The violation fires ~19/20 per attempt under this allocator; bound the
    # retry so a miss is negligible while a FIXED build stays clean across
    # every attempt.
    results = []
    for _ in range(3):
        corrupted_once = case._gather_round_corrupted()
        results.append(corrupted_once)
        if arm == "seam2_nofix" and corrupted_once:
            break
    corrupted = any(results)
    print(f"ARM_RESULT: {'CORRUPTED' if corrupted else 'CLEAN'}")


if __name__ == "__main__":
    import sys

    if "--arm" in sys.argv:
        _arm_main(sys.argv[sys.argv.index("--arm") + 1])
    else:
        unittest.main()
