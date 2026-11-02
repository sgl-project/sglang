"""Exercise real SWA freeing with speculative iteration clocks and CPU lengths."""

import sys
from contextlib import ExitStack
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.managers import schedule_batch as sb
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def simulate(
    enabled,
    *,
    algorithm="DFLASH",
    block=8,
    accepted=None,
    page=64,
    window=128,
    protected=0,
    mode="decode",
    steps=260
):
    accepted = accepted or [block]
    req = SimpleNamespace(
        decode_batch_idx=0,
        seqlen=1025,
        req_pool_idx=0,
        cache_protected_len=protected,
        swa_evicted_seqlen=0,
        extend_batch_idx=2,
    )
    slots = torch.arange(100000).view(1, -1)
    freed = torch.zeros(100000, dtype=torch.bool)
    calls = []

    def free_swa(indices):
        assert not freed[indices].any(), "No slot may be returned twice"
        assert not (indices < protected).any(), "Preserve the radix-owned prefix"
        assert indices.max() < req.seqlen - 1 - window
        freed[indices] = True

    cache = SimpleNamespace(
        supports_swa=lambda: True,
        sliding_window_size=window,
        page_size=page,
        is_chunk_cache=lambda: True,
    )
    batch = SimpleNamespace(
        tree_cache=cache,
        reqs=[req],
        forward_mode=SimpleNamespace(
            is_decode=lambda: mode == "decode", is_extend=lambda: mode == "extend"
        ),
        req_to_token_pool=SimpleNamespace(req_to_token=slots),
        token_to_kv_pool_allocator=SimpleNamespace(free_swa=free_swa),
        prefix_lens=[1024],
        enable_overlap=True,
    )

    def evict(self, request, pre_len):
        calls.append((request.decode_batch_idx, pre_len))
        sb.ScheduleBatch._evict_swa(self, request, pre_len)

    batch._evict_swa = MethodType(evict, batch)
    args = SimpleNamespace(
        speculative_algorithm=algorithm,
        speculative_dflash_block_size=block,
        chunked_prefill_size=256,
    )
    retained = []
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(sb, "get_global_server_args", return_value=args)
        )
        for setting, value in (
            (envs.SGLANG_DFLASH_SWA_TOKEN_INTERVAL, enabled),
            (envs.SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER, 1.0),
            (envs.SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW, False),
            (envs.SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN, False),
        ):
            stack.enter_context(patch.object(setting, "get", return_value=value))
        for i in range(steps):
            req.decode_batch_idx = i
            sb.ScheduleBatch.maybe_evict_swa(batch)
            if mode == "decode" and i == 0:
                assert not calls, "The previous extend can still be in flight"
            # The active window and speculative reservation beyond the committed
            # CPU length remain valid even when the CPU view lags the worker.
            safe_start = max(protected, req.seqlen - window - 1)
            assert not freed[safe_start : req.seqlen + 4 * block].any()
            if i > 0:
                retained.append(req.seqlen - req.swa_evicted_seqlen)
            req.seqlen += accepted[i % len(accepted)]
    return calls, retained, freed


@pytest.mark.parametrize(
    "enabled,algorithm", [(False, "DFLASH"), (True, "EAGLE"), (True, None)]
)
def test_existing_iteration_schedule_is_preserved(enabled, algorithm):
    calls, _, _ = simulate(enabled, algorithm=algorithm)
    assert [i for i, _ in calls] == [1, 129, 257]


@pytest.mark.parametrize("block", [1, 4, 8, 16])
def test_maximum_token_growth_bounds_retained_swa(block):
    calls, retained, _ = simulate(True, block=block)
    period = max(1, 128 // block)
    assert [i for i, _ in calls] == list(range(1, 260, period))
    assert max(retained) <= 128 + 64 + 63 + 128 + 1


@pytest.mark.parametrize("accepted", [[1], [8], [1, 8, 2, 5, 8, 1]])
@pytest.mark.parametrize("protected", [0, 768])
def test_variable_acceptance_and_radix_prefix_keep_live_pages(accepted, protected):
    simulate(True, accepted=accepted, protected=protected)


def test_large_block_still_evicts_after_first_iteration():
    calls, _, _ = simulate(True, block=256, steps=10)
    assert [i for i, _ in calls] == list(range(1, 10))


def test_prefill_eviction_frontier_is_unchanged():
    old, _, old_freed = simulate(False, mode="extend", steps=3)
    new, _, new_freed = simulate(True, mode="extend", steps=3)
    assert old == new == [(0, 768), (1, 768), (2, 768)]
    torch.testing.assert_close(old_freed, new_freed)


def test_memory_reduction_at_full_dflash_acceptance():
    _, old, _ = simulate(False)
    _, new, _ = simulate(True)
    assert max(old) > 1000
    assert max(new) < 400


def test_opt_out_preserves_the_legacy_unit_interval_edge_case():
    calls, _, _ = simulate(False, block=1, page=1, window=1, steps=10)
    assert calls == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
