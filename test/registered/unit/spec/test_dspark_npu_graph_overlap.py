"""CPU contracts for token-keyed NPU graphs and overlapped confidence relay."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.hardware_backend.npu.attention.dspark_compact import (
    NpuCompactGraphMetadata,
)
from sglang.srt.managers.overlap_utils import (
    CONFIDENCE_RELAY_RING_DEPTH,
    CONFIDENCE_RELAY_RING_LAG,
    ConfidenceRelay,
)
from sglang.srt.speculative.dspark_components import dspark_planner
from sglang.srt.speculative.dspark_components.dspark_planner import DSparkScheduleConfig
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout, RaggedVerifyMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def method(path, cls, name, **namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = copy.deepcopy(
        next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    )
    fn.decorator_list = []
    for n in ast.walk(fn):
        if isinstance(n, ast.arg):
            n.annotation = None
        if isinstance(n, ast.FunctionDef):
            n.returns = None
    scope = dict(torch=torch, **namespace)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
            str(path),
            "exec",
        ),
        scope,
    )
    return scope[name]


def test_graph_buffers_reused_without_changing_live_causal_positions():
    state = NpuCompactGraphMetadata(
        num_slots=3, num_tokens=16, table_width=4, device="cpu"
    )
    ptrs = [t.data_ptr() for t in (state.query_ends, state.kv_lens, state.block_tables)]
    table = torch.arange(16, 16 + 4 * 16).view(4, 16)
    for lens, prefixes, reqs in (
        ([2, 4, 1], [3, 7, 8], [2, 0, 3]),
        ([1, 2], [5, 9], [3, 2]),
        ([4, 4, 4], [4, 8, 12], [1, 3, 0]),
    ):
        state.update(
            prefix_lens=torch.tensor(prefixes),
            req_pool_indices=torch.tensor(reqs),
            req_to_token=table,
            verify_lens=torch.tensor(lens),
            page_size=4,
        )
        assert ptrs == [
            t.data_ptr() for t in (state.query_ends, state.kv_lens, state.block_tables)
        ]
        assert state.query_ends[-1] == 16
        assert (
            state.query_ends[: len(lens)].tolist()
            == torch.tensor(lens).cumsum(0).tolist()
        )
        assert state.kv_lens[: len(lens)].tolist() == [
            p + v for p, v in zip(prefixes, lens)
        ]
        assert not state.block_tables[-1].any()
        assert not state.kv_lens[len(lens) : -1].any()
        # In right-aligned causal attention, q row j sees prefix+j+1 keys.
        starts = [0] + state.query_ends[: len(lens) - 1].tolist()
        for i, (p, v) in enumerate(zip(prefixes, lens)):
            q_len = int(state.query_ends[i]) - starts[i]
            assert int(state.kv_lens[i]) - q_len == p
            assert q_len == v


def test_metadata_capture_tiers_with_same_slot_count_do_not_alias():
    init = method(
        "hardware_backend/npu/attention/ascend_backend.py",
        "AscendAttnBackend",
        "_init_compact_graph_metadata",
        NpuCompactGraphMetadata=NpuCompactGraphMetadata,
        ForwardMetadata=SimpleNamespace,
    )
    backend = SimpleNamespace(
        graph_metadata={},
        req_to_token=torch.arange(256).view(4, 64),
        page_size=8,
        device="cpu",
    )
    pointers = []
    for total in (8, 16, 8):
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.tensor([1, 3]), graph_num_tokens=total
        )
        fb = SimpleNamespace(
            batch_size=2,
            seq_lens=torch.tensor([7, 15]),
            req_pool_indices=torch.tensor([2, 1]),
        )
        init(backend, fb, layout)
        pointers.append(backend.forward_metadata.actual_seq_lengths_q.data_ptr())
        assert backend.forward_metadata.actual_seq_lengths_q.tolist() == [1, 4, total]
    assert pointers[0] == pointers[2] != pointers[1]


def test_npu_runner_defers_compact_to_common_token_tier_loader():
    prepare = method(
        "hardware_backend/npu/graph_runner/npu_cudagraph_backend.py",
        "NPUCudaGraphBackend",
        "prepare_replay",
    )
    runner = SimpleNamespace(ragged_verify_mode=True)
    backend = SimpleNamespace(_decode_runner=runner)
    fb = SimpleNamespace(
        needs_forward_metadata_init=lambda: False, ragged_verify_layout=object()
    )
    assert prepare(backend, fb) is False


@pytest.mark.parametrize(
    "is_npu,local_budget,root_budget,expected_budget,expected_tier",
    [
        (True, 1, 6, 6, 8),
        (True, None, 6, 6, 8),
        (True, 1, None, None, -1),
        (False, 1, 6, 1, 3),
    ],
)
def test_tp_broadcast_budget_precedes_tier_choice(
    is_npu, local_budget, root_budget, expected_budget, expected_tier
):
    planner = dspark_planner.DSparkVerifyPlanner.__new__(
        dspark_planner.DSparkVerifyPlanner
    )
    planner._budget_planner = Mock()
    planner.model_runner = SimpleNamespace(
        attn_backend=SimpleNamespace(_dspark_dsa_ragged_verify=is_npu)
    )
    planner._budget_from_resolved = Mock(return_value=local_budget)
    planner._maybe_gather_dp_verify_tier = Mock()
    planner.verify_num_draft_tokens = 8
    planner._schedule_cfg = DSparkScheduleConfig(gamma=7)
    batch = SimpleNamespace(
        spec_info=SimpleNamespace(),
        batch_size=lambda: 2,
        forward_mode=SimpleNamespace(is_extend=lambda: False),
        is_extend_in_batch=False,
        req_pool_indices_cpu=torch.tensor([2, 5]),
    )
    group = SimpleNamespace(broadcast_object=Mock(return_value=root_budget))
    with (
        patch.object(dspark_planner, "_is_npu", is_npu),
        patch.object(dspark_planner, "get_dp_tp_group", return_value=group) as parallel,
    ):
        planner.prepare_verify_budget(batch, Mock())
    assert batch.spec_info.verify_token_budget == expected_budget
    assert batch.spec_verify_tier_num_tokens == expected_tier
    if is_npu:
        group.broadcast_object.assert_called_once_with(local_budget, src=0)
    else:
        parallel.assert_not_called()
        group.broadcast_object.assert_not_called()


@pytest.mark.parametrize("is_npu,is_glm", [(True, True), (True, False), (False, False)])
def test_sync_budget_uses_runtime_tp_group(is_npu, is_glm):
    planner = dspark_planner.DSparkVerifyPlanner.__new__(
        dspark_planner.DSparkVerifyPlanner
    )
    planner._budget_planner = Mock()
    planner._budget_from_resolved = Mock(return_value=1)
    planner.model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_generation=torch.tensor([10, 20, 30])),
        attn_backend=SimpleNamespace(_dspark_dsa_ragged_verify=is_glm),
    )
    group = SimpleNamespace(broadcast_object=Mock(return_value=6))
    draft_input = SimpleNamespace()
    with (
        patch.object(dspark_planner, "_is_npu", is_npu),
        patch.object(dspark_planner, "get_dp_tp_group", return_value=group) as parallel,
        patch.object(
            dspark_planner,
            "get_schedule",
            return_value=SimpleNamespace(disable_overlap_schedule=True),
        ),
    ):
        budget = planner.resolve_verify_token_budget(
            draft_input=draft_input,
            confidence=torch.ones(2, 7),
            prefix_lens=torch.tensor([11, 22]),
            req_pool_indices=torch.tensor([2, 0]),
        )
    assert budget == (6 if is_npu and is_glm else 1)
    assert draft_input.verify_token_budget == budget
    resolved = planner._budget_from_resolved.call_args.kwargs["resolved"]
    assert resolved.generation.tolist() == [30, 10]
    if is_npu and is_glm:
        group.broadcast_object.assert_called_once_with(1, src=0)
    else:
        parallel.assert_not_called()
        group.broadcast_object.assert_not_called()


@pytest.mark.parametrize("mode", [RaggedVerifyMode.STATIC, RaggedVerifyMode.COMPACT])
def test_planner_constructed_before_backend_detects_glm_after_initialization(mode):
    # Match Scheduler: create worker/planner, allocate pools, then initialize
    # attention backends. Constructor must neither read a missing backend nor
    # permanently cache a negative capability result from that startup phase.
    runner = SimpleNamespace()
    with (
        patch.object(dspark_planner, "_is_npu", True),
        patch.object(dspark_planner, "read_ragged_verify_mode", return_value=mode),
        patch.object(
            dspark_planner,
            "get_spec",
            return_value=SimpleNamespace(
                speculative_dspark_align_verify_tokens_to_graph_tier=False,
                speculative_dspark_confidence_sts_path=None,
            ),
        ),
        patch.object(
            dspark_planner,
            "get_schedule",
            return_value=SimpleNamespace(
                disable_overlap_schedule=False,
            ),
        ),
        patch.object(dspark_planner, "is_dp_attention_enabled", return_value=False),
        patch.object(
            dspark_planner,
            "build_sps_cost_table",
            return_value=dspark_planner.build_uninitialized_sps_table(
                max_batch_tokens=16
            ),
        ),
    ):
        planner = dspark_planner.DSparkVerifyPlanner(
            draft_model=SimpleNamespace(confidence_head=object()),
            gamma=3,
            model_runner=runner,
            device="cpu",
            tp_rank=1,
            verify_num_draft_tokens=4,
            tp_sync=Mock(),
        )
        assert not hasattr(runner, "attn_backend")
        assert not planner._npu_glm_compact
        runner.req_to_token_pool = SimpleNamespace(req_generation=torch.tensor([0, 0]))
        runner.attn_backend = SimpleNamespace(_dspark_dsa_ragged_verify=True)
        assert planner._npu_glm_compact
        if mode is RaggedVerifyMode.COMPACT:
            planner._budget_from_resolved = Mock(return_value=1)
            group = SimpleNamespace(broadcast_object=Mock(return_value=3))
            with patch.object(
                dspark_planner,
                "get_dp_tp_group",
                return_value=group,
            ):
                budget = planner.compute_budget_sync(
                    confidence=torch.ones(2, 3),
                    prefix_lens=torch.tensor([4, 8]),
                    req_pool_indices=torch.tensor([0, 1]),
                )
            assert budget == 3
            group.broadcast_object.assert_called_once_with(1, src=0)
        runner.attn_backend = SimpleNamespace(_dspark_dsa_ragged_verify=False)
        assert not planner._npu_glm_compact


def test_confidence_relay_not_ready_generation_and_source_fence():
    events = [
        Mock(query=Mock(return_value=True)) for _ in range(CONFIDENCE_RELAY_RING_DEPTH)
    ]
    relay = ConfidenceRelay(
        device=torch.device("cpu"),
        req_pool_size=3,
        pool=SimpleNamespace(req_generation=torch.tensor([2, 3, 4])),
    )
    relay.initialized = True
    relay.confidence_buf = torch.zeros(3, 2)
    relay.conf_ring = torch.zeros(CONFIDENCE_RELAY_RING_DEPTH, 3, 2)
    relay.gen_ring = torch.zeros(CONFIDENCE_RELAY_RING_DEPTH, 3, dtype=torch.int64)
    relay.copy_done = events
    from contextlib import nullcontext

    stream, ready = Mock(), Mock()
    module = SimpleNamespace(
        stream=lambda s: nullcontext(), current_stream=Mock(return_value=stream)
    )
    batch = SimpleNamespace(
        spec_info=SimpleNamespace(future_indices=torch.tensor([1])),
        req_pool_indices_cpu=torch.tensor([1]),
    )
    with patch("torch.get_device_module", return_value=module):
        for i in range(CONFIDENCE_RELAY_RING_LAG):
            relay.scatter(torch.tensor([1]), torch.tensor([[0.1 + i * 0.1, 0.5]]))
            relay.issue_ring_copy(stream=stream, publish_ready=ready)
        events[0].query.return_value = False
        assert relay.resolve(batch, stream=stream, publish_ready=ready) is None
        events[0].query.return_value = True
        resolved = relay.resolve(batch, stream=stream, publish_ready=ready)
        assert resolved.generation.tolist() == [3]
        torch.testing.assert_close(resolved.confidence, torch.tensor([[0.1, 0.5]]))
        with patch("sglang.srt.managers.overlap_utils._is_npu", True):
            relay.scatter(torch.tensor([1]), torch.tensor([[0.8, 0.9]]))
        stream.wait_event.assert_called_with(
            events[(relay.ring_pos - 1) % CONFIDENCE_RELAY_RING_DEPTH]
        )


def test_device_metadata_update_has_no_host_reads():
    state = NpuCompactGraphMetadata(
        num_slots=3, num_tokens=16, table_width=8, device="cpu"
    )
    args = dict(
        prefix_lens=torch.tensor([7, 17]),
        req_pool_indices=torch.tensor([1, 2]),
        req_to_token=torch.arange(256).view(4, 64),
        verify_lens=torch.tensor([1, 4]),
        page_size=8,
    )
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("D2H")),
        patch.object(torch.Tensor, "item", side_effect=AssertionError("scalar sync")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host read")),
    ):
        state.update(**args)
    assert state.query_ends.tolist() == [1, 5, 5, 16]


def test_npu_executor_keeps_host_prefix_and_device_only_layout():
    from sglang.srt.speculative.dspark_components import dspark_verify

    executor = dspark_verify.TargetVerifyExecutor.__new__(
        dspark_verify.TargetVerifyExecutor
    )
    executor.verify_num_draft_tokens = 8
    executor.model_runner = SimpleNamespace(
        attn_backend=SimpleNamespace(_dspark_dsa_ragged_verify=True)
    )
    executor._forward_prepared_verify = Mock(return_value="launched")
    prefix = torch.tensor([7, 15])
    batch = SimpleNamespace(seq_lens_cpu=prefix, seq_lens_sum=22)
    layout = RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=torch.tensor([2, 4]), graph_num_tokens=8
    )
    window = SimpleNamespace(
        verify_ids=torch.arange(8),
        positions=torch.arange(8),
        verify_cache_loc=torch.arange(8),
        window_index=torch.arange(8),
    )
    with (
        patch.object(dspark_verify, "_is_npu", True),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("D2H")),
    ):
        result = executor._run_ragged(
            batch=batch,
            layout=layout,
            ragged_window=window,
            sampling_info=None,
            iteration_ids=None,
        )
    assert result == "launched"
    assert batch.seq_lens_cpu is prefix
    assert batch.seq_lens_sum == 22


def test_dsv4_ragged_verify_keeps_existing_cpu_length_contract():
    from sglang.srt.speculative.dspark_components import dspark_verify

    executor = dspark_verify.TargetVerifyExecutor.__new__(
        dspark_verify.TargetVerifyExecutor
    )
    executor.verify_num_draft_tokens = 8
    executor.model_runner = SimpleNamespace(
        attn_backend=SimpleNamespace(
            supports_ragged_verify_graph=True, _dspark_dsa_ragged_verify=False
        )
    )
    executor._forward_prepared_verify = Mock(return_value="launched")
    prefix = torch.tensor([7, 15])
    batch = SimpleNamespace(seq_lens_cpu=prefix, seq_lens_sum=22)
    layout = RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=torch.tensor([2, 4]), graph_num_tokens=8
    )
    window = SimpleNamespace(
        verify_ids=torch.arange(8),
        positions=torch.arange(8),
        verify_cache_loc=torch.arange(8),
        window_index=torch.arange(8),
    )
    with patch.object(dspark_verify, "_is_npu", True):
        result = executor._run_ragged(
            batch=batch,
            layout=layout,
            ragged_window=window,
            sampling_info=None,
            iteration_ids=None,
        )
    assert result == "launched"
    assert batch.seq_lens_cpu.tolist() == [9, 19]
    assert batch.seq_lens_sum == 28


def test_future_map_accepts_scheduler_string_device():
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    pool = SimpleNamespace(req_to_token=torch.zeros(4, 32, dtype=torch.int64))
    future = FutureMap(
        "cpu", SpeculativeAlgorithm.DSPARK, pool, needs_confidence_relay=True
    )
    assert future.npu_confidence_stream is None
