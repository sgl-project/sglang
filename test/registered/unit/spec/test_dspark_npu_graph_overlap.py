"""CPU contracts for token-keyed NPU graphs and overlapped confidence relay."""

import ast
import copy
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.hardware_backend.npu import dspark_dp
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


@pytest.mark.parametrize("glm", [False, True])
@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("needs_metadata", [False, True])
def test_npu_runner_defers_only_glm_compact_to_common_loader(
    glm, ragged, needs_metadata
):
    prepare = method(
        "hardware_backend/npu/graph_runner/npu_cudagraph_backend.py",
        "NPUCudaGraphBackend",
        "prepare_replay",
        envs=SimpleNamespace(
            SGLANG_ENABLE_OVERLAP_PLAN_STREAM=SimpleNamespace(get=lambda: False)
        ),
    )
    # An unrelated backend (including DSV4) need not expose the GLM capability.
    attn_backend = SimpleNamespace(
        **({"_dspark_dsa_ragged_verify": True} if glm else {})
    )
    runner = SimpleNamespace(
        ragged_verify_mode=ragged,
        attn_backend=attn_backend,
        model_runner=SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_dflash=lambda: False)
        ),
        buffers=SimpleNamespace(
            input_ids=torch.zeros(4, dtype=torch.int64),
            positions=torch.zeros(4, dtype=torch.int64),
        ),
        raw_num_token=4,
        bs=2,
        _make_graph_key=Mock(return_value="legacy-key"),
    )
    backend = SimpleNamespace(_decode_runner=runner)
    fb = SimpleNamespace(
        needs_forward_metadata_init=lambda: needs_metadata,
        ragged_verify_layout=object() if ragged else None,
        input_ids=torch.tensor([1, 2, 3, 4]),
        positions=torch.tensor([10, 11, 20, 21]),
        mrope_positions=None,
    )
    deferred = needs_metadata or (glm and ragged)
    assert prepare(backend, fb) is (not deferred)
    if deferred:
        assert runner.buffers.input_ids.tolist() == [0] * 4
        runner._make_graph_key.assert_not_called()
    else:
        torch.testing.assert_close(runner.buffers.input_ids, fb.input_ids)
        torch.testing.assert_close(runner.buffers.positions, fb.positions)
        runner._make_graph_key.assert_called_once_with(2)
        assert runner._replay_graph_key == "legacy-key"


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
    planner._verify_hooks = (
        dspark_planner.DSparkVerifyHooks(
            sync_budget=partial(
                dspark_dp.sync_verify_budget, model_runner=planner.model_runner
            )
        )
        if is_npu
        else None
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
        patch.object(dspark_dp, "get_dp_tp_group", return_value=group) as parallel,
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
    planner._verify_hooks = (
        dspark_planner.DSparkVerifyHooks(
            sync_budget=partial(
                dspark_dp.sync_verify_budget, model_runner=planner.model_runner
            )
        )
        if is_npu
        else None
    )
    group = SimpleNamespace(broadcast_object=Mock(return_value=6))
    draft_input = SimpleNamespace()
    with (
        patch.object(dspark_dp, "get_dp_tp_group", return_value=group) as parallel,
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
            verify_hooks=dspark_planner.DSparkVerifyHooks(
                sync_budget=partial(dspark_dp.sync_verify_budget, model_runner=runner)
            ),
        )
        assert not hasattr(runner, "attn_backend")
        assert planner.sync_verify_budget(1) == 1
        runner.req_to_token_pool = SimpleNamespace(req_generation=torch.tensor([0, 0]))
        runner.attn_backend = SimpleNamespace(_dspark_dsa_ragged_verify=True)
        if mode is RaggedVerifyMode.COMPACT:
            planner._budget_from_resolved = Mock(return_value=1)
            group = SimpleNamespace(broadcast_object=Mock(return_value=3))
            with patch.object(
                dspark_dp,
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
        assert planner.sync_verify_budget(1) == 1


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


def test_npu_executor_keeps_host_prefix_and_device_only_layout():
    from sglang.srt.speculative.dspark_components import dspark_verify

    executor = dspark_verify.TargetVerifyExecutor.__new__(
        dspark_verify.TargetVerifyExecutor
    )
    executor.verify_num_draft_tokens = 8
    executor.model_runner = SimpleNamespace(
        attn_backend=SimpleNamespace(ragged_verify_requires_cpu_seq_lens=False)
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
            supports_ragged_verify_graph=True, ragged_verify_requires_cpu_seq_lens=True
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


@pytest.mark.parametrize("idle_seq_len", [0, 1])
def test_idle_executor_uses_backend_prefix_contract_and_prepare_hook(idle_seq_len):
    from sglang.srt.speculative.dspark_components import dspark_verify

    backend = SimpleNamespace(ragged_verify_idle_seq_len=idle_seq_len)
    runner = SimpleNamespace(device="cpu", attn_backend=backend)
    executor = dspark_verify.TargetVerifyExecutor.__new__(
        dspark_verify.TargetVerifyExecutor
    )
    executor.model_runner = runner
    executor.verify_epilogue = None
    executor.verify_num_draft_tokens = 9
    executor._prepare_verify_forward_batch = Mock()
    executor.target_worker = Mock()
    layout = RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=torch.tensor([0]), graph_num_tokens=8
    )
    batch = SimpleNamespace()
    forward_batch = object()

    def prepare(verify_input, actual_batch, worker):
        assert actual_batch is batch
        assert batch.seq_lens.tolist() == [idle_seq_len]
        assert batch.seq_lens_cpu.tolist() == [idle_seq_len]
        assert batch.seq_lens_sum == idle_seq_len
        assert verify_input.live_seq_lens_cpu is batch.seq_lens_cpu
        assert verify_input.draft_token.numel() == 8
        return forward_batch, False

    def forward(**kwargs):
        executor._prepare_verify_forward_batch.assert_called_once_with(forward_batch)
        assert kwargs["forward_batch"] is forward_batch

    executor.target_worker.forward_batch_generation.side_effect = forward
    with patch.object(dspark_verify.DFlashVerifyInput, "prepare_for_verify", prepare):
        executor.run_idle_participation(batch=batch, idle_layout=layout)
    executor.target_worker.forward_batch_generation.assert_called_once()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
