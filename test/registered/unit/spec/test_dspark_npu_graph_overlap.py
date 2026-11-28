"""CPU contracts for token-keyed NPU graphs and overlapped confidence relay."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

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


@pytest.mark.parametrize("idle", [False, True])
def test_npu_compact_verify_leaves_replay_metadata_to_graph_loader(idle):
    from sglang.srt.hardware_backend.npu.dsv4 import dsv4_common_hooks
    from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
        NPUCudaGraphBackend,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.srt.speculative import dflash_info, spec_utils
    from sglang.srt.speculative.dspark_components import dspark_verify

    runner = SimpleNamespace(
        device="cpu", decode_cuda_graph_runner=Mock(ragged_verify_mode=True)
    )
    # Use the real readiness flag; mock batch allocation and NPU operations only.
    forward_batch = ForwardBatch.__new__(ForwardBatch)
    forward_batch.forward_metadata_ready = False
    backend = NPUCudaGraphBackend.__new__(NPUCudaGraphBackend)
    backend._decode_runner = runner.decode_cuda_graph_runner
    target = Mock(model_runner=runner)
    executor = dspark_verify.TargetVerifyExecutor.__new__(
        dspark_verify.TargetVerifyExecutor
    )
    executor.model_runner = runner
    executor.target_worker = target
    executor.verify_epilogue = None
    executor.verify_num_draft_tokens = 9
    executor._target_is_dsv41 = False
    layout = RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=torch.tensor([0 if idle else 3]), graph_num_tokens=8
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.IDLE if idle else ForwardMode.DECODE,
        seq_lens_cpu=torch.tensor([7]),
        seq_lens_sum=7,
    )
    with (
        patch.object(ForwardBatch, "init_new", return_value=forward_batch) as init,
        patch.object(dflash_info, "_is_npu", True),
        patch.object(dspark_verify, "_is_npu", True),
        patch.object(dsv4_common_hooks, "maybe_build_dsv4_verify_bundle"),
        patch.object(spec_utils, "prepare_mamba_track_for_verify"),
    ):
        if idle:
            executor.run_idle_participation(batch=batch, idle_layout=layout)
        else:
            executor._run_ragged(
                batch=batch,
                layout=layout,
                ragged_window=SimpleNamespace(
                    verify_ids=torch.arange(8),
                    positions=torch.arange(8),
                    verify_cache_loc=torch.arange(8),
                    window_index=torch.arange(8),
                ),
                iteration_ids=None,
            )

    target.forward_batch_generation.assert_called_once_with(
        batch=None, forward_batch=forward_batch, is_verify=True
    )
    assert init.call_args.kwargs["ragged_verify_layout"] is layout
    runner.decode_cuda_graph_runner.load_batch.assert_not_called()
    assert forward_batch.needs_forward_metadata_init()
    assert not backend.prepare_replay(forward_batch)


@pytest.mark.parametrize("local_budget,expected_tier", [(4, 6), (0, 2), (None, -1)])
def test_budget_preparation_preserves_upstream_decision(local_budget, expected_tier):
    planner = dspark_planner.DSparkVerifyPlanner.__new__(
        dspark_planner.DSparkVerifyPlanner
    )
    planner._budget_planner = Mock()
    planner.plans_verify_budget = Mock(return_value=True)
    planner._budget_from_resolved = Mock(return_value=local_budget)
    planner._maybe_gather_dp_verify_tier = Mock()
    planner.verify_num_draft_tokens = 9
    planner._schedule_cfg = DSparkScheduleConfig(gamma=8)
    batch = SimpleNamespace(
        spec_info=SimpleNamespace(),
        batch_size=lambda: 2,
        forward_mode=SimpleNamespace(is_extend=lambda: False),
        is_extend_in_batch=False,
        req_pool_indices_cpu=torch.tensor([2, 5]),
    )
    future_map = Mock()
    with patch("torch.distributed.broadcast_object_list") as broadcast:
        planner.prepare_verify_budget(batch, future_map)
    assert batch.spec_info.verify_token_budget == local_budget
    assert batch.spec_verify_tier_num_tokens == expected_tier
    planner._budget_from_resolved.assert_called_once_with(
        resolved=future_map.resolve_confidence_cpu.return_value,
        req_pool_indices_cpu=batch.req_pool_indices_cpu,
    )
    broadcast.assert_not_called()


def test_sync_budget_computation_preserves_upstream_decision():
    planner = dspark_planner.DSparkVerifyPlanner.__new__(
        dspark_planner.DSparkVerifyPlanner
    )
    planner._budget_planner = Mock()
    planner.plans_verify_budget = Mock(return_value=True)
    planner._budget_from_resolved = Mock(return_value=1)
    planner.model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_generation=torch.tensor([10, 20, 30])),
    )
    draft_input = SimpleNamespace()
    with (
        patch("torch.distributed.broadcast_object_list") as broadcast,
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
    assert budget == 1
    assert draft_input.verify_token_budget == budget
    resolved = planner._budget_from_resolved.call_args.kwargs["resolved"]
    assert resolved.generation.tolist() == [30, 10]
    broadcast.assert_not_called()


@pytest.mark.parametrize("mode", [RaggedVerifyMode.STATIC, RaggedVerifyMode.COMPACT])
def test_planner_constructor_does_not_read_backend_or_run_budget_collectives(mode):
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
        )
        assert not hasattr(runner, "attn_backend")
        runner.req_to_token_pool = SimpleNamespace(req_generation=torch.tensor([0, 0]))
        runner.attn_backend = SimpleNamespace(_dspark_dsa_ragged_verify=True)
        if mode is RaggedVerifyMode.COMPACT:
            planner._budget_from_resolved = Mock(return_value=1)
            with patch("torch.distributed.broadcast_object_list") as broadcast:
                budget = planner.compute_budget_sync(
                    confidence=torch.ones(2, 3),
                    prefix_lens=torch.tensor([4, 8]),
                    req_pool_indices=torch.tensor([0, 1]),
                )
            assert budget == 1
            broadcast.assert_not_called()


def test_confidence_relay_lag_generation_and_source_fence():
    events = [Mock() for _ in range(CONFIDENCE_RELAY_RING_DEPTH)]
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
            if i + 1 < CONFIDENCE_RELAY_RING_LAG:
                assert relay.resolve(batch, stream=stream, publish_ready=ready) is None
        resolved = relay.resolve(batch, stream=stream, publish_ready=ready)
        events[0].synchronize.assert_called_once_with()
        assert resolved.generation.tolist() == [3]
        torch.testing.assert_close(resolved.confidence, torch.tensor([[0.1, 0.5]]))
        with patch("sglang.srt.managers.overlap_utils._is_npu", True):
            relay.scatter(torch.tensor([1]), torch.tensor([[0.8, 0.9]]))
        stream.wait_event.assert_called_with(
            events[(relay.ring_pos - 1) % CONFIDENCE_RELAY_RING_DEPTH]
        )


def test_ragged_verify_keeps_existing_cpu_length_contract():
    from sglang.srt.speculative.dspark_components import dspark_verify

    executor = dspark_verify.TargetVerifyExecutor.__new__(
        dspark_verify.TargetVerifyExecutor
    )
    executor.verify_num_draft_tokens = 8
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


def test_idle_executor_keeps_upstream_prefix():
    from sglang.srt.speculative.dspark_components import dspark_verify

    idle_seq_len = 1
    backend = SimpleNamespace()
    runner = SimpleNamespace(device="cpu", attn_backend=backend)
    executor = dspark_verify.TargetVerifyExecutor.__new__(
        dspark_verify.TargetVerifyExecutor
    )
    executor.model_runner = runner
    executor.verify_epilogue = None
    executor.verify_num_draft_tokens = 9
    executor.target_worker = Mock()
    layout = RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=torch.tensor([0]), graph_num_tokens=8
    )
    batch = SimpleNamespace()
    forward_batch = Mock()

    def prepare(verify_input, actual_batch, worker):
        assert actual_batch is batch
        assert batch.seq_lens.tolist() == [idle_seq_len]
        assert batch.seq_lens_cpu.tolist() == [idle_seq_len]
        assert batch.seq_lens_sum == idle_seq_len
        assert verify_input.live_seq_lens_cpu is batch.seq_lens_cpu
        assert verify_input.draft_token.numel() == 8
        return forward_batch, False

    def forward(**kwargs):
        assert kwargs["forward_batch"] is forward_batch

    executor.target_worker.forward_batch_generation.side_effect = forward
    with patch.object(dspark_verify.DFlashVerifyInput, "prepare_for_verify", prepare):
        executor.run_idle_participation(batch=batch, idle_layout=layout)
    executor.target_worker.forward_batch_generation.assert_called_once()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
