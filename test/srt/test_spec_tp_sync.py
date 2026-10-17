"""Tests for SpecTpSync.available_memory_gb group semantics (#39886)."""
import os
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite


def _mem_call(sync, group):
    with patch(
        "sglang.srt.speculative.spec_tp_sync.get_available_gpu_memory",
        return_value=2.0,
    ) as get_memory:
        sync.available_memory_gb(SpecTpSyncSite.DSPARK_MEM, "cuda", 0, group=group)
    return get_memory.call_args.kwargs["distributed"]


def test_foreign_multi_rank_group_reduces():
    # The #39886 repro: sync bound to single-rank attention group, caller
    # passes the full TP group -> the reduction must happen.
    sync = SpecTpSync(SimpleNamespace(world_size=1))
    full_tp2 = SimpleNamespace(world_size=2, rank_in_group=0, cpu_group=object())
    assert _mem_call(sync, full_tp2) is True


def test_constructor_multi_rank_group_reduces_by_default():
    # Multi-rank constructor group: sites default to all -> reduced.
    sync = SpecTpSync(SimpleNamespace(world_size=2, rank_in_group=0, cpu_group=object()))
    assert _mem_call(sync, sync._tp_group) is True


def test_site_gate_still_applies_on_constructor_group():
    # Same-group call with every site disabled stays rank-local.
    os.environ["SGLANG_SPEC_TP_SYNC"] = "-all"
    try:
        sync = SpecTpSync(SimpleNamespace(world_size=2, rank_in_group=0, cpu_group=object()))
        assert _mem_call(sync, sync._tp_group) is False
    finally:
        del os.environ["SGLANG_SPEC_TP_SYNC"]


def test_single_rank_group_stays_local():
    sync = SpecTpSync(SimpleNamespace(world_size=1))
    assert _mem_call(sync, sync._tp_group) is False


test_foreign_multi_rank_group_reduces()
test_constructor_multi_rank_group_reduces_by_default()
test_site_gate_still_applies_on_constructor_group()
test_single_rank_group_stays_local()
print("ALL 4 PASS")
