from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)


def test_cuda_profiler_activity_can_use_nsys_nvtx_capture_range(tmp_path):
    ps = SimpleNamespace(gpu_id=0)
    manager = SchedulerProfilerManager(
        ps=ps,
        dp_tp_cpu_group=MagicMock(),
        get_forward_ct=lambda: 0,
    )
    manager.torch_profiler_output_dir = tmp_path
    manager.torch_profiler_with_stack = None
    manager.torch_profiler_record_shapes = None
    manager.profiler_activities = ["CUDA_PROFILER"]
    manager.profile_id = "nsys-nvtx-test"
    manager.profile_prefix = ""

    device = SimpleNamespace(base_gpu_id=0)
    with (
        patch.dict(
            "os.environ",
            {"SGLANG_NSYS_NVTX_CAPTURE_RANGE": "agentx_decode_capture"},
        ),
        patch(
            "sglang.srt.managers.scheduler_components.profiler_manager.get_device",
            return_value=device,
        ),
        patch("torch.cuda.nvtx.range_start", return_value=73) as range_start,
        patch("torch.cuda.nvtx.range_end") as range_end,
        patch("torch.cuda.synchronize") as synchronize,
        patch("torch.cuda.cudart") as cudart,
    ):
        start_result = manager._start_profile()
        stop_result = manager._stop_profile()

    assert start_result.success
    assert stop_result.success
    range_start.assert_called_once_with("agentx_decode_capture")
    synchronize.assert_called_once_with()
    range_end.assert_called_once_with(73)
    cudart.assert_not_called()
    assert not manager.nsys_nvtx_capture_active
    assert manager.nsys_nvtx_capture_handle is None


def test_nsys_exact_running_batch_defers_and_rebases_capture_window(tmp_path):
    forward_ct = 100
    ps = SimpleNamespace(gpu_id=0)
    with patch.dict(
        "os.environ", {"SGLANG_NSYS_EXACT_RUNNING_BATCH": "32"}
    ):
        manager = SchedulerProfilerManager(
            ps=ps,
            dp_tp_cpu_group=MagicMock(),
            get_forward_ct=lambda: forward_ct,
        )
    manager.profiler_start_forward_ct = 100
    manager.profiler_target_forward_ct = 200

    with (
        patch.object(manager, "_start_profile") as start_profile,
        patch.object(manager, "_wait_for_nsys_capture_start") as start_latch,
        patch("torch.distributed.all_reduce") as all_reduce,
    ):
        manager._profile_batch_predicate(
            SimpleNamespace(reqs=[object()] * 31, forward_mode=MagicMock())
        )
        assert not start_profile.called

        forward_ct = 107
        manager._profile_batch_predicate(
            SimpleNamespace(reqs=[object()] * 32, forward_mode=MagicMock())
        )

    start_profile.assert_called_once_with()
    start_latch.assert_called_once_with()
    assert all_reduce.call_count == 2
    assert manager.profiler_target_forward_ct == 207


def test_nsys_exact_running_batch_waits_for_every_dp_rank():
    forward_ct = 100
    ps = SimpleNamespace(gpu_id=0)
    request_plane_group = MagicMock(name="request_plane_group")
    exact_nsys_group = MagicMock(name="exact_nsys_group")
    with (
        patch.dict(
            "os.environ",
            {
                "SGLANG_NSYS_EXACT_RUNNING_BATCH": "32",
                "SGLANG_NSYS_EXACT_SYNC_WORLD_SIZE": "4",
            },
        ),
        patch("torch.distributed.get_world_size", return_value=4) as get_world_size,
    ):
        manager = SchedulerProfilerManager(
            ps=ps,
            dp_tp_cpu_group=request_plane_group,
            get_forward_ct=lambda: forward_ct,
            exact_nsys_cpu_group=exact_nsys_group,
        )
    get_world_size.assert_called_once_with(group=exact_nsys_group)
    manager.profiler_start_forward_ct = 100
    manager.profiler_target_forward_ct = 132
    batch = SimpleNamespace(reqs=[object()] * 32, forward_mode=MagicMock())

    def peer_readiness(ready, **_kwargs):
        if peer_readiness.calls == 0:
            ready.zero_()
        peer_readiness.calls += 1

    peer_readiness.calls = 0
    with (
        patch.object(manager, "_start_profile") as start_profile,
        patch.object(manager, "_wait_for_nsys_capture_start") as start_latch,
        patch("torch.distributed.all_reduce", side_effect=peer_readiness),
    ):
        manager._profile_batch_predicate(batch)
        start_profile.assert_not_called()
        manager._profile_batch_predicate(batch)

    start_profile.assert_called_once_with()
    start_latch.assert_called_once_with()
    assert all(
        call.kwargs["group"] is exact_nsys_group for call in all_reduce.call_args_list
    )


def test_nsys_exact_capture_waits_for_two_real_decode_batches_after_idle_steps():
    forward_ct = 100
    ps = SimpleNamespace(gpu_id=0)
    with patch.dict(
        "os.environ", {"SGLANG_NSYS_EXACT_RUNNING_BATCH": "32"}
    ):
        manager = SchedulerProfilerManager(
            ps=ps,
            dp_tp_cpu_group=MagicMock(),
            get_forward_ct=lambda: forward_ct,
        )
    manager.profiler_start_forward_ct = 100
    manager.profiler_target_forward_ct = 101

    decode_mode = SimpleNamespace(is_decode=lambda: True)
    idle_mode = SimpleNamespace(is_decode=lambda: False)
    with (
        patch.object(manager, "_start_profile") as start_profile,
        patch.object(manager, "_wait_for_nsys_capture_start") as start_latch,
        patch.object(manager, "_stop_profile") as stop_profile,
        patch("torch.distributed.all_reduce"),
        patch("torch.distributed.barrier") as barrier,
    ):
        start_profile.side_effect = lambda: setattr(
            manager, "profile_in_progress", True
        )
        stop_profile.side_effect = lambda: setattr(
            manager, "profile_in_progress", False
        )

        manager._profile_batch_predicate(
            SimpleNamespace(reqs=[object()] * 32, forward_mode=decode_mode)
        )
        assert manager.nsys_exact_decode_batches_seen == 1

        forward_ct = 1000
        for _ in range(5):
            manager._profile_batch_predicate(
                SimpleNamespace(reqs=[], forward_mode=idle_mode)
            )
        stop_profile.assert_not_called()

        manager._profile_batch_predicate(
            SimpleNamespace(reqs=[object()] * 32, forward_mode=decode_mode)
        )
        assert manager.nsys_exact_decode_batches_seen == 2
        stop_profile.assert_not_called()

        manager._profile_batch_predicate(
            SimpleNamespace(reqs=[], forward_mode=idle_mode)
        )

    start_profile.assert_called_once_with()
    start_latch.assert_called_once_with()
    stop_profile.assert_called_once_with()
    # The rank-0 filesystem latch replaces the deadlocking post-start barrier;
    # the existing pre/post-stop pair remains.
    assert barrier.call_count == 2


def test_nsys_exact_capture_fails_closed_if_batch_shape_changes():
    decode_mode = SimpleNamespace(is_decode=lambda: True)
    with patch.dict(
        "os.environ", {"SGLANG_NSYS_EXACT_RUNNING_BATCH": "32"}
    ):
        manager = SchedulerProfilerManager(
            ps=SimpleNamespace(gpu_id=0),
            dp_tp_cpu_group=MagicMock(),
            get_forward_ct=lambda: 100,
        )
    manager.profile_in_progress = True

    with pytest.raises(RuntimeError, match="expected 32, got 31"):
        manager._profile_batch_predicate(
            SimpleNamespace(reqs=[object()] * 31, forward_mode=decode_mode)
        )


def test_nsys_any_rank_gate_captures_variable_shape_window():
    decode_mode = SimpleNamespace(is_decode=lambda: True)
    forward_ct = 100
    with patch.dict(
        "os.environ",
        {
            "SGLANG_NSYS_EXACT_RUNNING_BATCH": "32",
            "SGLANG_NSYS_EXACT_GATE_REDUCTION": "any",
            "SGLANG_NSYS_REQUIRE_FIXED_CAPTURE": "0",
        },
    ):
        manager = SchedulerProfilerManager(
            ps=SimpleNamespace(gpu_id=0),
            dp_tp_cpu_group=MagicMock(),
            get_forward_ct=lambda: forward_ct,
        )
    manager.profiler_start_forward_ct = 100
    manager.profiler_target_forward_ct = 102

    def mark_any_rank_ready(ready, **_kwargs):
        ready.fill_(1)

    with (
        patch.object(manager, "_start_profile") as start_profile,
        patch.object(manager, "_wait_for_nsys_capture_start") as start_latch,
        patch.object(manager, "_stop_profile") as stop_profile,
        patch("torch.distributed.all_reduce", side_effect=mark_any_rank_ready) as reduce,
        patch("torch.distributed.barrier") as barrier,
    ):
        start_profile.side_effect = lambda: setattr(
            manager, "profile_in_progress", True
        )
        stop_profile.side_effect = lambda: setattr(
            manager, "profile_in_progress", False
        )

        manager._profile_batch_predicate(
            SimpleNamespace(reqs=[object()] * 31, forward_mode=decode_mode)
        )
        assert manager.profile_in_progress
        assert reduce.call_args.kwargs["op"] == torch.distributed.ReduceOp.MAX

        forward_ct = 101
        manager._profile_batch_predicate(
            SimpleNamespace(reqs=[object()] * 29, forward_mode=decode_mode)
        )
        forward_ct = 102
        manager._profile_batch_predicate(
            SimpleNamespace(reqs=[object()] * 27, forward_mode=decode_mode)
        )
        forward_ct = 103
        manager._profile_batch_predicate(
            SimpleNamespace(reqs=[object()] * 25, forward_mode=decode_mode)
        )

    start_profile.assert_called_once_with()
    start_latch.assert_called_once_with()
    stop_profile.assert_called_once_with()
    assert barrier.call_count == 2


def test_nsys_capture_start_latch_is_profile_specific(tmp_path):
    manager = SchedulerProfilerManager(
        ps=SimpleNamespace(gpu_id=0),
        dp_tp_cpu_group=MagicMock(),
        get_forward_ct=lambda: 0,
    )
    manager.torch_profiler_output_dir = tmp_path
    manager.profile_id = "worker-specific-profile"

    with patch(
        "sglang.srt.managers.scheduler_components.profiler_manager.get_device",
        return_value=SimpleNamespace(base_gpu_id=0),
    ):
        manager._wait_for_nsys_capture_start()

    assert (tmp_path / ".nsys-capture-ready-worker-specific-profile").is_file()
