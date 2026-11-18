"""DeepSelect's host-side module dispatch, with stand-in modules (no GPU)."""

import logging

import pytest
import torch

from sglang.kernels.ops.attention import deep_select
from sglang.kernels.ops.attention.deep_select import (
    _CLUSTER_MIN_VOCAB_SIZE,
    _CachedModuleDispatch,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# (max rows in one wave, cluster size), widest cluster first: a 212-SM SM107 GPU's tiers
_TIERS = ((8, 16), (22, 8), (46, 4), (60, 3), (106, 2))
_LONG = _CLUSTER_MIN_VOCAB_SIZE + 1


def _dispatch(**kwargs) -> _CachedModuleDispatch:
    return _CachedModuleDispatch(
        [f"cluster{size}" for _, size in _TIERS],
        [rows for rows, _ in _TIERS],
        "normal",
        cluster_sizes=[size for _, size in _TIERS],
        **kwargs,
    )


def test_default_keeps_one_wave_tiers():
    dispatch = _dispatch()
    assert dispatch(1, _LONG) == "cluster16"
    assert dispatch(9, _LONG) == "cluster8"
    assert dispatch(106, _LONG) == "cluster2"
    assert dispatch(107, _LONG) == "normal"
    assert dispatch(240, _LONG) == "normal"
    assert dispatch(8, _CLUSTER_MIN_VOCAB_SIZE) == "normal"


def test_multiwave_past_the_widest_tier():
    dispatch = _dispatch(multiwave_cluster_size=2, multiwave_max_rows=512)
    # One-wave tiers are unchanged.
    assert dispatch(8, _LONG) == "cluster16"
    assert dispatch(106, _LONG) == "cluster2"
    # Past them, the cluster2 module runs several waves, up to the row cap.
    assert dispatch(107, _LONG) == "cluster2"
    assert dispatch(240, _LONG) == "cluster2"
    assert dispatch(512, _LONG) == "cluster2"
    assert dispatch(513, _LONG) == "normal"
    # Short rows never take a cluster kernel.
    assert dispatch(240, _CLUSTER_MIN_VOCAB_SIZE) == "normal"


def test_multiwave_other_tier():
    dispatch = _dispatch(multiwave_cluster_size=4, multiwave_max_rows=300)
    assert dispatch(100, _LONG) == "cluster2"
    assert dispatch(200, _LONG) == "cluster4"
    assert dispatch(301, _LONG) == "normal"


def test_multiwave_size_without_a_tier_warns(caplog):
    with caplog.at_level(logging.WARNING, logger=deep_select.__name__):
        dispatch = _dispatch(multiwave_cluster_size=5, multiwave_max_rows=512)
    assert "not a cluster tier" in caplog.text
    assert dispatch(240, _LONG) == "normal"


def test_no_cluster_modules_ignore_multiwave(caplog):
    # fp32 / large top-k signatures compile no cluster module.
    with caplog.at_level(logging.WARNING, logger=deep_select.__name__):
        dispatch = _CachedModuleDispatch(
            [], [], "normal", multiwave_cluster_size=2, multiwave_max_rows=512
        )
    assert caplog.text == ""
    assert dispatch(240, _LONG) == "normal"


@pytest.mark.parametrize("cluster,expected", [(0, "normal"), (2, "cluster2")])
def test_precompile_reads_env(monkeypatch, cluster, expected):
    monkeypatch.setattr(deep_select, "_cluster_tiers", lambda: _TIERS)
    # Each job is partial(_jit_deep_select_module, *signature, bucket, cluster_size, page_transform)
    monkeypatch.setattr(
        deep_select,
        "parallel_compile",
        lambda jobs: [
            f"cluster{job.args[-2]}" if job.args[-2] > 1 else "normal" for job in jobs
        ],
    )
    precompile = deep_select._precompile.__wrapped__
    with (
        envs.SGLANG_DEEP_SELECT_MULTIWAVE_CLUSTER.override(cluster),
        envs.SGLANG_DEEP_SELECT_MULTIWAVE_MAX_ROWS.override(512),
    ):
        dispatch = precompile(
            dtype=torch.bfloat16,
            topk=512,
            sorted=False,
            indices_type=torch.int32,
            sorted_index=False,
            return_value=False,
            page_transform=True,
        )
    assert dispatch(106, _LONG) == "cluster2"
    assert dispatch(240, _LONG) == expected
