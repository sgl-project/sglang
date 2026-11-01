"""CPU policy checks; graph execution/quality still requires GPU qualification."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.srt.model_executor import dsv4_prefill_graph as policy
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


@pytest.fixture
def state():
    config = CudaGraphConfig(
        decode=PhaseConfig(backend=Backend.FULL, bs=[1, 2, 4], max_bs=4),
        prefill=PhaseConfig(backend=Backend.DISABLED, max_context_size=16384),
    )
    context = Mock()
    args = SimpleNamespace(
        disable_cuda_graph=False,
        disable_prefill_cuda_graph=False,
        disaggregation_mode="null",
        cuda_graph_backend_prefill=None,
        cuda_graph_bs_prefill=None,
        cuda_graph_config=None,
    )
    execution = SimpleNamespace(
        graph=SimpleNamespace(cuda_graph_config=config),
        overlap=SimpleNamespace(enable_two_batch_overlap=False),
    )
    parallel = SimpleNamespace(pp_size=1, attn_dp_size=1, attn_cp_size=1, dcp_size=1)
    schedule = SimpleNamespace(chunked_prefill_size=8192, max_prefill_tokens=16384)
    with (
        patch.object(policy, "get_exec", return_value=execution),
        patch.object(policy, "get_parallel", return_value=parallel),
        patch.object(policy, "get_schedule", return_value=schedule),
        patch.object(policy, "get_context", return_value=context),
        patch.object(policy, "is_deepseek_v4", return_value=True),
        patch("sglang.srt.utils.is_cuda", return_value=True),
        patch("sglang.srt.layers.moe.utils.get_moe_a2a_backend") as a2a,
    ):
        a2a.return_value.is_none.return_value = True
        yield SimpleNamespace(
            config=config,
            context=context,
            args=args,
            execution=execution,
            parallel=parallel,
            schedule=schedule,
            a2a=a2a,
        )


def enable(state):
    return policy.maybe_enable_dsv4_prefill_cuda_graph(
        server_args=state.args, hf_config=SimpleNamespace()
    )


@pytest.mark.parametrize(
    "maximum,expected",
    [
        (512, [512]),
        (8192, [512, 1024, 2048, 4096, 8192]),
        (6000, [512, 1024, 2048, 4096, 6000]),
    ],
)
def test_buckets_include_ceiling_once(maximum, expected):
    assert policy.dsv4_prefill_graph_buckets(maximum) == expected


def test_enables_only_prefill_without_mutating_published_config(state):
    assert enable(state)
    name = state.context.override.call_args.args[0]
    result = state.context.override.call_args.kwargs["cuda_graph_config"]
    assert name == "dsv4_prefill_cuda_graph"
    assert result.prefill.backend == Backend.BREAKABLE
    assert result.prefill.bs == [512, 1024, 2048, 4096, 8192]
    assert result.prefill.max_context_size == 16384
    assert result.decode == state.config.decode
    assert state.config.prefill.backend == Backend.DISABLED
    assert state.config.prefill.bs is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("cuda_graph_backend_prefill", Backend.DISABLED),
        ("cuda_graph_bs_prefill", [1024]),
        ("cuda_graph_config", {}),
    ],
)
def test_explicit_configuration_wins(state, field, value):
    setattr(state.args, field, value)
    assert not enable(state)
    state.context.override.assert_not_called()


@pytest.mark.parametrize(
    "field", ["pp_size", "attn_dp_size", "attn_cp_size", "dcp_size"]
)
def test_unsupported_parallel_topology_stays_eager(state, field):
    setattr(state.parallel, field, 2)
    assert not enable(state)
    state.context.override.assert_not_called()


@pytest.mark.parametrize("chunk", [None, -1, 0, 511])
def test_unbounded_or_small_chunks_stay_eager(state, chunk):
    state.schedule.chunked_prefill_size = chunk
    assert not enable(state)
    state.context.override.assert_not_called()


def test_max_prefill_tokens_caps_bucket_ladder(state):
    state.schedule.max_prefill_tokens = 6000
    assert enable(state)
    result = state.context.override.call_args.kwargs["cuda_graph_config"]
    assert result.prefill.bs == [512, 1024, 2048, 4096, 6000]


@pytest.mark.parametrize("backend", [Backend.FULL, Backend.BREAKABLE])
def test_existing_prefill_backend_is_preserved(state, backend):
    state.config.prefill.backend = backend
    assert not enable(state)
    state.context.override.assert_not_called()


def test_disabled_decode_is_preserved(state):
    state.config.decode.backend = Backend.DISABLED
    assert not enable(state)


def test_missing_config_is_preserved(state):
    state.execution.graph.cuda_graph_config = None
    assert not enable(state)


def test_two_batch_overlap_stays_eager(state):
    state.execution.overlap.enable_two_batch_overlap = True
    assert not enable(state)


def test_moe_all_to_all_stays_eager(state):
    state.a2a.return_value.is_none.return_value = False
    assert not enable(state)


def test_non_cuda_stays_eager(state):
    with patch("sglang.srt.utils.is_cuda", return_value=False):
        assert not enable(state)


def test_other_models_are_unchanged(state):
    with patch.object(policy, "is_deepseek_v4", return_value=False):
        assert not enable(state)


@pytest.mark.parametrize("field", ["disable_cuda_graph", "disable_prefill_cuda_graph"])
def test_legacy_disable_switches_are_preserved(state, field):
    setattr(state.args, field, True)
    assert not enable(state)
    state.context.override.assert_not_called()


def test_pd_decode_role_stays_eager(state):
    state.args.disaggregation_mode = "decode"
    assert not enable(state)
    state.context.override.assert_not_called()
