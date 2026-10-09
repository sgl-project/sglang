"""Routing boundaries for sparse-prefill direct cache reads."""

from types import SimpleNamespace

import pytest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.srt.layers.attention.deepseek_v4_backend import _prefill_reads_fp8_direct


register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-large")

@pytest.mark.parametrize(
    "prefix,rows,in_graph,expected",
    [
        (None, [1024], False, False),
        ([0], [8192], False, False),
        ([4095], [819, 1], False, False),
        ([4096], [819, 1], False, False),
        ([5000], [1000], False, True),
        ([4999], [1000], False, False),
        ([7168], [1024], False, True),
        ([7168], [1024], True, False),
        ([8000], [1000], True, True),
        ([7999], [1000], True, False),
        ([10000, 0], [1000, 1000], False, True),
        ([9999, 0], [1000, 1000], False, False),
    ],
)
def test_prefix_per_query_row(prefix, rows, in_graph, expected):
    batch = SimpleNamespace(
        extend_prefix_lens_cpu=prefix, extend_seq_lens_cpu=rows
    )
    with (
        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(True),
        envs.SGLANG_DSV4_SPARSE_PREFILL_DIRECT_PREFIX_PER_ROW.override(5),
        envs.SGLANG_DSV4_SPARSE_PREFILL_DIRECT_PREFIX_PER_ROW_GRAPH.override(8),
    ):
        assert _prefill_reads_fp8_direct(batch, in_prefill_graph=in_graph) == expected


@pytest.mark.parametrize("in_graph", [False, True])
def test_zero_threshold_and_explicit_direct_override(in_graph):
    batch = SimpleNamespace(
        extend_prefix_lens_cpu=[65536], extend_seq_lens_cpu=[1024]
    )
    with (
        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(True),
        envs.SGLANG_DSV4_SPARSE_PREFILL_DIRECT_PREFIX_PER_ROW.override(0),
        envs.SGLANG_DSV4_SPARSE_PREFILL_DIRECT_PREFIX_PER_ROW_GRAPH.override(0),
    ):
        assert not _prefill_reads_fp8_direct(batch, in_prefill_graph=in_graph)
        with envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(False):
            assert _prefill_reads_fp8_direct(batch, in_prefill_graph=in_graph)
