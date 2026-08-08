"""Spec-V2 overlap lifetime: exit-time SB pin, relay-read pin.

Two use-after-free seams of the overlap scheduler, each with a deterministic
repro (real CUDA streams where the hazard is cross-stream):

1. Exit-time SB attr pin (Scheduler._forward_isolation): spec-V2 workers
   rebind ScheduleBatch fields to fresh tensors DURING the forward (e.g.
   eagle_prepare_for_verify replaces input_ids / out_cache_loc), after the
   pre-forward snapshot was taken. The isolation restore then drops the
   rebound tensors' only Python ref while forward-stream kernels may still
   read them. The exit pin captures the CURRENT attr values (vars(), covering
   ad-hoc attrs) into the 2-iter batch_record_buf ring slot before restoring.

2. mix_running_indices relay-read pin (resolve_forward_inputs): the mixed-in
   running batch's req_pool_indices are freed by the scheduler right after
   the batch is staged, while the forward stream's output_tokens_buf gather
   still reads them -- record_stream defers block reuse.
"""

import dataclasses
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.overlap_utils import FutureMap, resolve_forward_inputs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

_GPU_DELAY_CYCLES = 100_000_000  # ~tens of ms >> host enqueue latency


def _make_schedule_batch(**attrs):
    """A ScheduleBatch shell with every dataclass field present (None) so the
    isolation snapshot/restore paths can getattr/setattr them freely."""
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    batch = object.__new__(ScheduleBatch)
    for f in dataclasses.fields(ScheduleBatch):
        setattr(batch, f.name, None)
    for name, value in attrs.items():
        setattr(batch, name, value)
    return batch


def _flatten(obj, out):
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _flatten(item, out)
    else:
        out.append(obj)


class TestForwardIsolationExitPin(CustomTestCase):
    """The exit pin must capture mid-forward rebinds (dataclass fields AND
    ad-hoc instance attrs) into the batch_record_buf ring slot the pre-forward
    snapshot lives in, BEFORE the restore drops them."""

    @staticmethod
    def _scheduler():
        from sglang.test.test_utils import maybe_stub_sgl_kernel

        maybe_stub_sgl_kernel()
        from sglang.srt.managers.scheduler import Scheduler

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.batch_record_buf = [None] * 2
        scheduler.batch_record_ct = 0
        return scheduler

    def test_exit_pin_covers_mid_forward_rebinds(self):
        scheduler = self._scheduler()
        batch = _make_schedule_batch(
            spec_algorithm=SpeculativeAlgorithm.EAGLE,  # full-snapshot path
            sampling_info=None,
        )
        pre_binding = object()
        batch.input_ids = pre_binding

        rebound_field = object()  # stands in for the verify input_ids rebind
        rebound_adhoc = object()  # stands in for plan-stream ad-hoc attrs
        with scheduler._forward_isolation(batch, overlap=True):
            batch.input_ids = rebound_field
            batch.mid_forward_adhoc_attr = rebound_adhoc

        # The restore reverted the field (transactional SB)...
        self.assertIs(batch.input_ids, pre_binding)
        # ...but both rebound objects must survive in the ring slot: the
        # restore dropped the batch's refs while (in prod) forward-stream
        # kernels may still read the underlying tensors.
        pinned = []
        _flatten(scheduler.batch_record_buf[scheduler.batch_record_ct], pinned)
        self.assertTrue(any(item is rebound_field for item in pinned))
        self.assertTrue(any(item is rebound_adhoc for item in pinned))

    def test_non_overlap_path_pins_nothing(self):
        # The sync path runs on a single stream and never allocates the ring
        # slot; the exit pin must not fire there.
        scheduler = self._scheduler()
        batch = _make_schedule_batch(
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            sampling_info=None,
        )
        with scheduler._forward_isolation(batch, overlap=False):
            batch.input_ids = object()
        self.assertEqual(scheduler.batch_record_buf, [None, None])


class TestMixRunningIndicesCpuDevice(CustomTestCase):
    def test_cpu_device_mix_gather_unaffected(self):
        # CPU-device overlap has no cross-stream reuse hazard, and
        # Tensor.record_stream rejects CPU tensors -- the keep-alive pin must
        # not fire there. Pins the device-type guard.
        future_map = FutureMap(
            device=torch.device("cpu"),
            spec_algo=SpeculativeAlgorithm.NONE,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros((8, 16), dtype=torch.int32)
            ),
        )
        future_map.output_tokens_buf[:8] = torch.arange(8, dtype=torch.int64) * 100
        batch = SimpleNamespace(
            prefill_input_ids_cpu=torch.tensor([11, 12], dtype=torch.int64),
            mix_running_indices=torch.tensor([2, 3], dtype=torch.int64),
            input_ids=None,
            device="cpu",
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        resolve_forward_inputs(batch, future_map)
        self.assertEqual(batch.input_ids.tolist(), [11, 12, 200, 300])


@unittest.skipUnless(
    torch.cuda.is_available(), "relay-read UAF repro needs real CUDA streams"
)
class TestMixRunningIndicesKeepAlive(CustomTestCase):
    @staticmethod
    def _make_batch(prefill_cpu, indices):
        return SimpleNamespace(
            prefill_input_ids_cpu=prefill_cpu,
            mix_running_indices=indices,
            input_ids=None,
            device="cuda",
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )

    def test_indices_survive_forward_stream_gather(self):
        """Drives the REAL resolve_forward_inputs on the forward stream while
        the scheduler-side ref to mix_running_indices drops and its block is
        reallocated with DECOY pool rows. Unprotected, the queued
        output_tokens_buf gather reads the decoy rows and the mixed batch's
        decode input_ids are silently wrong (in-range values, no crash).

        Choreography notes (mirrors the steady-state scheduler): a WARM pass
        runs the same resolve on the forward stream first so the raced pass
        only enqueues async work -- a first-touch cudaMalloc on a cold stream
        pool synchronizes the device and would mask the race. The raced
        indices get a distinctive size class (128 x int64) so their freed
        block is deterministically the one the decoy reallocation reuses."""
        dev = torch.device("cuda")
        future_map = FutureMap(
            device=dev,
            spec_algo=SpeculativeAlgorithm.NONE,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros((8, 16), dtype=torch.int32, device=dev)
            ),
        )
        future_map.output_tokens_buf[:8] = (
            torch.arange(8, dtype=future_map.output_tokens_buf.dtype, device=dev) * 100
        )

        decoy_vals = torch.full((128,), 5, dtype=torch.int64, device=dev)
        prefill_warm = torch.tensor([11, 12], dtype=torch.int64).pin_memory()
        prefill_cpu = torch.tensor([11, 12], dtype=torch.int64).pin_memory()
        # The raced indices: 128 entries of relay rows 2,3 -- the only
        # default-pool allocation in this size class.
        indices = torch.tensor([2, 3] * 64, dtype=torch.int64, device=dev)

        fwd_stream = torch.cuda.Stream()
        # Warm pass: same shapes through the REAL resolve on the forward
        # stream (indices allocated inside the stream ctx land in the forward
        # stream's allocator pool, away from the raced default-pool block).
        with torch.cuda.stream(fwd_stream):
            warm_indices = torch.ones(128, dtype=torch.int64, device=dev)
            warm_batch = self._make_batch(prefill_warm, warm_indices)
            torch.cuda._sleep(1)
            resolve_forward_inputs(warm_batch, future_map)
        torch.cuda.synchronize()
        del warm_indices, warm_batch

        batch = self._make_batch(prefill_cpu, indices)
        with torch.cuda.stream(fwd_stream):
            torch.cuda._sleep(_GPU_DELAY_CYCLES)
            resolve_forward_inputs(batch, future_map)  # consumes + clears staging

        # The scheduler-side free: resolve cleared batch.mix_running_indices,
        # so this local is the last ref. The same-size reallocation stamps
        # DECOY rows into the (reused, unless kept alive) block while the
        # gather is still queued behind the sleep.
        self.assertIsNone(batch.mix_running_indices)
        del indices
        decoy = torch.empty(128, dtype=torch.int64, device=dev)
        decoy.copy_(decoy_vals)
        torch.cuda.synchronize()

        # Rows 2,3 of the relay buf -- not the decoy row 5.
        self.assertEqual(batch.input_ids[:2].tolist(), [11, 12])
        self.assertEqual(batch.input_ids[2:].tolist(), [200, 300] * 64)
        del decoy


if __name__ == "__main__":
    unittest.main()
