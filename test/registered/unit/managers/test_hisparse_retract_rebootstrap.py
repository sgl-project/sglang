"""A HiSparse decode retraction must not try to back the KV up to host.

`release_req` frees the request's HiSparse resources through
`hisparse_coordinator.retract_req` before the retraction backup would run, and
the HiSparse allocator has no `get_cpu_copy` to take that backup with. So
`retract_decode` skips the backup and `DecodePreallocQueue.add` sends the
request down the PD rebootstrap path instead of the host-resume queue.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, ScheduleBatch
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeReq:
    """Minimal stand-in for Req that keeps the real offload body."""

    # The real method is the one that reaches the allocator; keep it bound to
    # the fake so a regression shows up as the production NotImplementedError.
    offload_kv_cache = Req.offload_kv_cache
    _mamba_pool_needing_backup = Req._mamba_pool_needing_backup

    def __init__(self, output_len: int, input_len: int = 8):
        self.rid = f"req-{output_len}"
        self.origin_input_ids = [0] * input_len
        self.output_ids = [1] * output_len
        self.seqlen = input_len + output_len
        self.kv = ReqKvInfo(req_pool_idx=0, kv_allocated_len=self.seqlen)
        self.beam_group = None
        self.priority = None
        self.sampling_params = SimpleNamespace(max_new_tokens=64)
        self.to_finish = None
        self.was_reset = False
        self.bootstrap_host = "10.0.0.1"
        self.bootstrap_room = 7
        self.retraction_mb_id = 3
        self.pd_rebootstrap_forced_output_id = None
        self.pd_rebootstrap_in_progress = False
        self.time_stats = SimpleNamespace(set_retract_time=lambda: None)

    def finished(self) -> bool:
        return False

    def owned_kv_len(self) -> int:
        return self.kv.kv_allocated_len

    def reset_for_retract(self) -> None:
        self.was_reset = True


class _NoOffloadAllocator:
    """Stands in for the HiSparse allocator: no CPU copy support."""

    def __init__(self):
        self.get_cpu_copy_calls = 0

    def get_kvcache(self):
        return SimpleNamespace(cpu_copy_carries_mamba=True)

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        self.get_cpu_copy_calls += 1
        raise NotImplementedError()


class _FakeTreeCache:
    def is_chunk_cache(self) -> bool:
        return True

    def cache_finished_req(self, req, is_insert=True, owned_kv_len=None):
        # Mirrors the real cache: the request's row and KV are released.
        req.kv.req_pool_idx = None
        req.kv.mark_kv_released()


class _FakeReqToTokenPool:
    def __init__(self, seqlen: int):
        self.req_to_token = torch.zeros((1, seqlen), dtype=torch.int64)


def _build_batch(hisparse: bool, reqs):
    batch = ScheduleBatch.__new__(ScheduleBatch)
    batch.reqs = reqs
    batch.req_to_token_pool = _FakeReqToTokenPool(max(r.seqlen for r in reqs))
    batch.token_to_kv_pool_allocator = _NoOffloadAllocator()
    batch.tree_cache = _FakeTreeCache()
    # A coordinator is attached exactly when HiSparse is enabled; its
    # retract_req is what frees the KV a backup would otherwise copy.
    batch.hisparse_coordinator = (
        SimpleNamespace(retract_req=lambda req: None) if hisparse else None
    )
    # Satisfied again right away: the loop's first iteration is unconditional,
    # so exactly one request is retracted and none is aborted for OOM.
    batch.check_decode_mem = lambda selected_indices=None: True
    batch.filter_batch = lambda keep_indices=None: None
    return batch


def _build_prealloc_queue(hisparse: bool):
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.scheduler = SimpleNamespace(
        hisparse_coordinator=(SimpleNamespace() if hisparse else None)
    )
    queue.token_to_kv_pool_allocator = _NoOffloadAllocator()
    queue.retracted_queue = []
    queue.enqueued = []

    def _create_receiver_and_enqueue(req, is_rebootstrap=False):
        decode_req = SimpleNamespace(
            req=req,
            is_rebootstrap=is_rebootstrap,
            kv_receiver=SimpleNamespace(init=lambda rank: None),
        )
        queue.enqueued.append(decode_req)
        return decode_req

    queue._create_receiver_and_enqueue = _create_receiver_and_enqueue
    queue._check_if_req_exceed_kv_capacity = lambda req: False
    queue._resolve_prefill_dp_rank = lambda req: 0
    return queue


class TestHiSparseRetractRebootstrap(CustomTestCase):
    def setUp(self):
        publish(
            ServerArgs(
                model_path="dummy",
                disaggregation_mode="decode",
                disaggregation_decode_retraction_backup="cpu_tensor",
            ),
            role="test",
        )
        self.addCleanup(reset_context)

    def test_hisparse_decode_retract_skips_host_offload(self):
        reqs = [_FakeReq(5), _FakeReq(1)]
        batch = _build_batch(hisparse=True, reqs=reqs)

        retracted, _, reqs_to_abort = batch.retract_decode()

        self.assertEqual(batch.token_to_kv_pool_allocator.get_cpu_copy_calls, 0)
        # The request is retracted for real, not aborted as "host pool
        # exhausted": release_req reports success because no backup was needed.
        self.assertEqual([r.rid for r in retracted], ["req-1"])
        self.assertEqual(reqs_to_abort, [])
        self.assertTrue(reqs[1].was_reset)

    def test_non_hisparse_decode_retract_still_offloads(self):
        # Guard against silently disabling host offload for plain PD decode,
        # whose resume path (resume_retracted_reqs -> load_kv_cache) needs it.
        reqs = [_FakeReq(5), _FakeReq(1)]
        batch = _build_batch(hisparse=False, reqs=reqs)

        with self.assertRaises(NotImplementedError):
            batch.retract_decode()
        self.assertEqual(batch.token_to_kv_pool_allocator.get_cpu_copy_calls, 1)

    def test_hisparse_retracted_request_goes_to_rebootstrap(self):
        queue = _build_prealloc_queue(hisparse=True)
        req = _FakeReq(3)

        queue.add(req, is_retracted=True)

        # The host-resume queue would restore a backup that does not exist.
        self.assertEqual(queue.retracted_queue, [])
        self.assertEqual(len(queue.enqueued), 1)
        self.assertTrue(queue.enqueued[0].is_rebootstrap)
        self.assertTrue(req.pd_rebootstrap_in_progress)
        # The boundary token is popped here and replayed at transfer commit.
        self.assertEqual(req.pd_rebootstrap_forced_output_id, 1)
        self.assertEqual(len(req.output_ids), 2)

    def test_non_hisparse_retracted_request_goes_to_resume_queue(self):
        queue = _build_prealloc_queue(hisparse=False)
        req = _FakeReq(3)

        queue.add(req, is_retracted=True)

        self.assertEqual(queue.retracted_queue, [req])
        self.assertEqual(queue.enqueued, [])
        self.assertFalse(req.pd_rebootstrap_in_progress)


if __name__ == "__main__":
    unittest.main()
