import sys

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.spec_tp_sync import (
    SpecTpSync,
    SpecTpSyncSite,
    parse_spec_tp_sync,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTpGroup:
    def __init__(self):
        self.world_size = 8
        self.rank_in_group = 0
        self.broadcast_calls = []

    def broadcast(self, values, src):
        self.broadcast_calls.append((values, src))


def test_default_includes_dflash_greedy_sites(monkeypatch):
    monkeypatch.delenv(envs.SGLANG_SPEC_TP_SYNC.name, raising=False)
    sites = parse_spec_tp_sync(envs.SGLANG_SPEC_TP_SYNC.get())
    assert sites == frozenset(SpecTpSyncSite)
    assert SpecTpSyncSite.DFLASH_ACCEPT_GREEDY in sites
    assert SpecTpSyncSite.DFLASH_DRAFT_GREEDY in sites
    assert int(SpecTpSyncSite.DFLASH_DRAFT_GREEDY) == 18
    assert SpecTpSyncSite.DFLASH_DRAFT_GREEDY.slug == "dflash-draft-greedy"


def test_default_greedy_sites_are_broadcast():
    tp_group = _FakeTpGroup()
    sync = SpecTpSync(tp_group)
    values = torch.tensor([1])

    assert sync.sync(SpecTpSyncSite.DFLASH_ACCEPT_GREEDY, values) is values
    assert len(tp_group.broadcast_calls) == 1

    for site in (
        SpecTpSyncSite.DFLASH_TARGET,
        SpecTpSyncSite.DFLASH_ACCEPT_SAMPLE,
        SpecTpSyncSite.DFLASH_SELECTOR,
        SpecTpSyncSite.DFLASH_DRAFT_GREEDY,
    ):
        call_count = len(tp_group.broadcast_calls)
        assert sync.sync(site, values) is values
        assert len(tp_group.broadcast_calls) == call_count + 1


def test_all_preset_opts_greedy_back_in():
    tp_group = _FakeTpGroup()
    with envs.SGLANG_SPEC_TP_SYNC.override("all"):
        sync = SpecTpSync(tp_group)
        values = torch.tensor([1])
        assert sync.sync(SpecTpSyncSite.DFLASH_ACCEPT_GREEDY, values) is values
    assert len(tp_group.broadcast_calls) == 1


def test_default_does_not_warn(caplog):
    tp_group = _FakeTpGroup()
    with envs.SGLANG_SPEC_TP_SYNC.override(envs.SGLANG_SPEC_TP_SYNC.default):
        with caplog.at_level("WARNING"):
            SpecTpSync(tp_group)
    assert not any("Speculative TP sync" in record.message for record in caplog.records)

    caplog.clear()
    with envs.SGLANG_SPEC_TP_SYNC.override("rng"):
        with caplog.at_level("WARNING"):
            SpecTpSync(tp_group)
    assert any("Speculative TP sync" in record.message for record in caplog.records)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
