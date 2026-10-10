"""Routing boundaries for sparse-prefill direct cache reads."""

from types import SimpleNamespace

import pytest

from sglang.srt.environ import envs
from sglang.srt.layers.attention import deepseek_v4_backend as backend_module
from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    _prefill_reads_fp8_direct,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    context as bcg_context,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
    batch = SimpleNamespace(extend_prefix_lens_cpu=prefix, extend_seq_lens_cpu=rows)
    with (
        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(True),
        envs.SGLANG_DSV4_SPARSE_PREFILL_DIRECT_PREFIX_PER_ROW.override(5),
        envs.SGLANG_DSV4_SPARSE_PREFILL_DIRECT_PREFIX_PER_ROW_GRAPH.override(8),
    ):
        assert _prefill_reads_fp8_direct(batch, in_prefill_graph=in_graph) == expected


@pytest.mark.parametrize("in_graph", [False, True])
def test_zero_threshold_and_explicit_direct_override(in_graph):
    batch = SimpleNamespace(extend_prefix_lens_cpu=[65536], extend_seq_lens_cpu=[1024])
    with (
        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(True),
        envs.SGLANG_DSV4_SPARSE_PREFILL_DIRECT_PREFIX_PER_ROW.override(0),
        envs.SGLANG_DSV4_SPARSE_PREFILL_DIRECT_PREFIX_PER_ROW_GRAPH.override(0),
    ):
        assert not _prefill_reads_fp8_direct(batch, in_prefill_graph=in_graph)
        with envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(False):
            assert _prefill_reads_fp8_direct(batch, in_prefill_graph=in_graph)


@pytest.mark.parametrize(
    "is_sm100,prefix,in_graph,expected",
    [
        (True, [5000], False, True),
        (True, [7999], True, False),
        (True, [8000], True, True),
        (False, [65536], False, False),
        (False, [65536], True, False),
    ],
)
def test_unset_thresholds_apply_only_on_sm100(
    monkeypatch, is_sm100, prefix, in_graph, expected
):
    monkeypatch.setattr(
        backend_module, "get_platform", lambda: SimpleNamespace(is_sm100=is_sm100)
    )
    batch = SimpleNamespace(extend_prefix_lens_cpu=prefix, extend_seq_lens_cpu=[1000])
    with (
        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(True),
        envs.SGLANG_DSV4_SPARSE_PREFILL_DIRECT_PREFIX_PER_ROW.override(None),
        envs.SGLANG_DSV4_SPARSE_PREFILL_DIRECT_PREFIX_PER_ROW_GRAPH.override(None),
    ):
        assert _prefill_reads_fp8_direct(batch, in_prefill_graph=in_graph) == expected


@pytest.mark.parametrize(
    "direct,in_graph,expected",
    [(False, False, False), (True, False, True), (False, True, True)],
)
def test_page_indices_only_when_a_step_can_read_direct(
    monkeypatch, direct, in_graph, expected
):
    monkeypatch.setattr(
        backend_module,
        "get_platform",
        lambda: SimpleNamespace(is_sm100=True, is_sm120=False),
    )
    monkeypatch.setattr(bcg_context, "is_in_breakable_cuda_graph", lambda: in_graph)
    backend = object.__new__(DeepseekV4AttnBackend)
    backend.trtllm_attn = False
    backend.tail_forward_metadata = None
    backend.forward_metadata = SimpleNamespace(
        late_layer_tail=None, sparse_prefill_direct=direct
    )
    backend.token_to_kv_pool = SimpleNamespace(request_window=None)
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    with (
        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(True),
        envs.SGLANG_DSV4_SPARSE_PREFILL_DIRECT_PREFIX_PER_ROW_GRAPH.override(None),
    ):
        assert backend._low_ratio_prefill_reads_page_indices(batch) == expected
