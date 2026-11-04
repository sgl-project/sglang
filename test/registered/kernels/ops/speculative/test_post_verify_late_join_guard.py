"""Post-verify late join on a real GPU: a slot freed under a running side-stream commit, then taken by a new owner.

The side stream sleeps before it writes the slot (as a delayed commit would), so without the guard the new owner's
write lands first and the commit then overwrites it; with the guard the new owner waits for the commit. The same
holds for wait() on another stream (a HiCache load or backup).
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA streams"),
]

_DELAY_CYCLES = 50_000_000  # tens of ms: long enough that the host has enqueued the new owner's write
_SLOTS = 8
_COMMIT_VALUE, _OWNER_VALUE = 7.0, 3.0


def _pending_commit(state: torch.Tensor, slot: int):
    """Enqueue a delayed commit writing ``slot`` on a side stream; returns the graphs view the guard reads."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    done = torch.cuda.Event()
    with torch.cuda.stream(side):
        torch.cuda._sleep(_DELAY_CYCLES)
        state[slot].fill_(_COMMIT_VALUE)
    done.record(side)
    return SimpleNamespace(_commit_pending=True, _commit_done=done, _side_stream=side)


def _reuse_slot(late_join, *, guarded: bool) -> float:
    from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator

    state = torch.full((_SLOTS + 1, 1024), float("nan"), device="cuda")
    alloc = MambaSlotAllocator(size=_SLOTS, device="cuda")
    slot = int(alloc.alloc(1)[0])
    alloc.alloc(_SLOTS - 1)  # the freed slot is the next one handed out
    torch.cuda.synchronize()
    graphs = _pending_commit(state, slot)
    with mock.patch.object(late_join, "_graphs", graphs if guarded else None):
        alloc.free(
            torch.tensor([slot], device="cuda")
        )  # the row finished while its commit runs
        again = int(alloc.alloc(1)[0])
        assert again == slot
        state[again].fill_(_OWNER_VALUE)  # the new owner initializes its state
    torch.cuda.synchronize()
    return float(state[slot, 0])


def test_slot_reuse_waits_for_the_commit():
    from sglang.srt.speculative import post_verify_late_join

    assert _reuse_slot(post_verify_late_join, guarded=True) == _OWNER_VALUE
    # Control: the same interleaving without the guard loses the new owner's write.
    assert _reuse_slot(post_verify_late_join, guarded=False) == _COMMIT_VALUE


def test_wait_orders_another_stream():
    from sglang.srt.speculative import post_verify_late_join

    for guarded, want in ((True, _OWNER_VALUE), (False, _COMMIT_VALUE)):
        state = torch.full((2, 1024), float("nan"), device="cuda")
        torch.cuda.synchronize()
        graphs = _pending_commit(state, 1)
        load = torch.cuda.Stream()
        with mock.patch.object(
            post_verify_late_join, "_graphs", graphs if guarded else None
        ):
            post_verify_late_join.wait("hicache_load", stream=load)
            with torch.cuda.stream(load):
                state[1].fill_(_OWNER_VALUE)
        torch.cuda.synchronize()
        assert float(state[1, 0]) == want, guarded
