from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
        patch("torch.cuda.nvtx.range_push") as range_push,
        patch("torch.cuda.nvtx.range_pop") as range_pop,
        patch("torch.cuda.cudart") as cudart,
    ):
        start_result = manager._start_profile()
        stop_result = manager._stop_profile()

    assert start_result.success
    assert stop_result.success
    range_push.assert_called_once_with("agentx_decode_capture")
    range_pop.assert_called_once_with()
    cudart.assert_not_called()
    assert not manager.nsys_nvtx_capture_active


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

    with patch.object(manager, "_start_profile") as start_profile:
        manager._profile_batch_predicate(
            SimpleNamespace(reqs=[object()] * 31, forward_mode=MagicMock())
        )
        assert not start_profile.called

        forward_ct = 107
        manager._profile_batch_predicate(
            SimpleNamespace(reqs=[object()] * 32, forward_mode=MagicMock())
        )

    start_profile.assert_called_once_with()
    assert manager.profiler_target_forward_ct == 207


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
        patch.object(manager, "_stop_profile") as stop_profile,
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
            SimpleNamespace(reqs=[object()] * 31, forward_mode=decode_mode)
        )
        assert manager.nsys_exact_decode_batches_seen == 2
        stop_profile.assert_not_called()

        manager._profile_batch_predicate(
            SimpleNamespace(reqs=[], forward_mode=idle_mode)
        )

    start_profile.assert_called_once_with()
    stop_profile.assert_called_once_with()
