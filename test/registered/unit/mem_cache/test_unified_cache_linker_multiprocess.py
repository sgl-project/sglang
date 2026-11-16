"""External-linker prefetch yielding to a memory-parked chunked continuation,
run on two gloo ranks of one attention group."""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

# Every collective the prefetch path makes comes from one of these.
_EXISTING_SITES = {
    "_sync_restorable_prefix",
    "prefetch",
    "check_prefetch_events",
    "check_hicache_events",
}


def _yield_worker(rank, path):
    """Drive one scenario on each rank; ranks differ only in which read is
    still running when the continuation parks and which submission fails."""
    import sys
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    from test_unified_cache_linker import TestLinkerDevicePrefetch

    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.schedule_policy import PrefillAdder

    dist.init_process_group(
        "gloo",
        init_method="file://" + path,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        case = TestLinkerDevicePrefetch(
            "test_layer_wise_still_uses_queued_loads_without_prefetch"
        )
        cache, backend, _, allocator = case._fixture()
        linker = cache.linker
        cache.tp_group = dist.group.WORLD
        cache.tp_world_size = 2
        trace = []
        reduce = cache._all_reduce_attn_groups

        def traced(tensor, op):
            trace.append(
                (
                    sys._getframe(1).f_code.co_name,
                    str(op),
                    tuple(tensor.shape),
                    tuple((h.rid, h.attempt_id) for h in linker.prefetches),
                )
            )
            reduce(tensor, op)

        cache._all_reduce_attn_groups = traced

        held = allocator.alloc(14)
        burst = [case._request(f"burst-{i}", start=10 * i + 1) for i in range(3)]
        for req in burst:
            case._enqueue(cache, req)
        # Rank 0's oldest read is already running; rank 1's is still queued.
        if rank == 0:
            assert backend.prefetch_futures[0].set_running_or_notify_cancel()

        # The continuation whose earlier chunks hold KV; real admission
        # decides each pass, called in the scheduler's order.
        continuation = case._request("continuation", start=61)
        continuation.full_untruncated_fill_ids = list(range(1000, 1040))
        continuation.prefix_len = 8
        owner = continuation.cache_request_handle

        def admission_pass():
            linker.release_protection(chunked_request=owner)
            adder = PrefillAdder(
                page_size=cache.page_size,
                tree_cache=cache,
                token_to_kv_pool_allocator=allocator,
                running_batch=ScheduleBatch(reqs=[]),
                new_token_ratio=1.0,
                rem_input_tokens=10000,
                rem_chunk_tokens=8,
                num_mixed_decode_tokens=0,
                priority_scheduling_preemption_threshold=0,
            )
            adder.add_chunked_req(continuation)
            admitted = any(r is continuation for r in adder.can_run_list)
            linker.report_continuation(
                request=owner,
                admitted=admitted,
                parked_for_memory=adder.continuation_parked_for_memory,
            )
            return admitted, adder.continuation_parked_for_memory

        assert admission_pass() == (False, True)
        late = case._request("late", start=41)
        case._enqueue(cache, late)
        cache.check_hicache_events()
        # The running peer keeps the read's slots on both ranks until it stops.
        assert [h.rid for h in linker.prefetches] == ["burst-0"]
        assert admission_pass() == (True, False)
        # A request re-queued between selection and allocation stays deferred.
        requeued = case._request("requeued", start=51)
        case._enqueue(cache, requeued)
        assert requeued.cache_request_handle in linker.deferred_prefetches
        chunk = allocator.alloc(8)
        assert chunk is not None
        linker.release_protection(chunked_request=owner)
        assert linker.protected_continuation is None
        waiting = [*burst, late, requeued]
        # Rank 1 fails to submit one read; the accept vote still matches.
        if rank == 1:
            submit = backend.prefetch.side_effect

            submissions = []

            def fail_once(transfers):
                submissions.append(transfers)
                if len(submissions) == 1:
                    raise RuntimeError("submission failed")
                return submit(transfers)

            backend.prefetch.side_effect = fail_once
        linker.issue_deferred_prefetches(waiting)
        if rank == 0:
            backend.prefetch_futures[0].set_result(True)
        cache.check_hicache_events()
        linker.issue_deferred_prefetches(waiting)
        assert not linker.deferred_prefetches

        for future in backend.prefetch_futures:
            if not future.done():
                future.set_result(False)
        cache.check_hicache_events()
        assert not linker.prefetches
        allocator.free(torch.cat([held, chunk]))

        sites = {site for site, *_ in trace}
        assert sites <= _EXISTING_SITES, sites
        every = [None] * 2
        dist.all_gather_object(every, trace)
        assert every[0] == every[1], every
    finally:
        dist.destroy_process_group()


class TestLinkerYieldAcrossRanks(CustomTestCase):
    def test_ranks_reclaim_retire_and_retry_in_lockstep(self):
        """A memory-parked continuation's reclaim, retirement and retries keep
        both ranks in the same collectives over the same requests, although
        one rank's read is still running and another's submission fails."""
        import tempfile

        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(_yield_worker, args=(tmp + "/rendezvous",), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
