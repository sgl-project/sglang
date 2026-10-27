"""SGLANG_OPT_DSV41_DSPARK_DRAFT_PREP_FORK: the draft graph's in-graph metadata prep
runs on a side stream and the first draft KV store joins it, before anything reads
the metadata it builds."""

import sys
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_kv_store_location_joins_before_the_deferred_commit_hook(monkeypatch):
    """The first draft KV store reads its location through get_swa_out_cache_loc,
    which must join the forked prep before reading the metadata it builds."""
    from sglang.srt.layers.attention import deepseek_v4_backend as dsv4

    order = []
    monkeypatch.setattr(
        dsv4, "join_in_graph_metadata_prep", lambda: order.append("prep") or True
    )

    class _Stop(Exception):
        pass

    def deferred_commit_join():
        order.append("commit")
        raise _Stop

    backend = dsv4.DeepseekV4AttnBackend.__new__(dsv4.DeepseekV4AttnBackend)
    backend.pre_kv_store_hook = deferred_commit_join
    with pytest.raises(_Stop):
        backend.get_swa_out_cache_loc(SimpleNamespace())
    assert order == ["prep", "commit"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
