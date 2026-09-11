from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def make_manager(*, tp_rank: int = 0, profile_ranks: str = "0"):
    with patch.dict(
        "os.environ",
        {"SGLANG_PROFILE_RANKS": profile_ranks, "SGLANG_PROFILE_V2": "0"},
    ):
        return SchedulerProfilerManager(
            ps=SimpleNamespace(tp_size=8, tp_rank=tp_rank),
            dp_tp_cpu_group=None,
            get_forward_ct=lambda: 0,
        )


def configure_profile(manager):
    result = manager._init_profile(
        output_dir="/tmp/sglang-test-profile",
        start_step=None,
        num_steps=None,
        activities=["CPU", "GPU"],
        with_stack=False,
        record_shapes=False,
        profile_by_stage=False,
        profile_id="test-profile",
    )
    assert result.success


def test_parse_profile_ranks():
    assert SchedulerProfilerManager._parse_profile_ranks(None) is None
    assert SchedulerProfilerManager._parse_profile_ranks("") is None
    assert SchedulerProfilerManager._parse_profile_ranks("0, 2,2") == {0, 2}
    with pytest.raises(ValueError):
        SchedulerProfilerManager._parse_profile_ranks("0,-1")


def test_reject_profile_rank_outside_tp_world():
    with pytest.raises(ValueError, match="outside the TP world"):
        make_manager(profile_ranks="8")


def test_non_selected_rank_never_constructs_profiler():
    manager = make_manager(tp_rank=1)
    configure_profile(manager)

    result = manager._start_profile(ForwardMode.DECODE)

    assert result.success
    assert manager.profile_in_progress
    assert manager.torch_profiler is None

    result = manager._stop_profile(ForwardMode.DECODE)
    assert result.success
    assert not manager.profile_in_progress


def test_selected_rank_stop_has_no_cross_rank_barrier(tmp_path):
    manager = make_manager()
    manager.profile_in_progress = True
    manager.torch_profiler = Mock()
    manager.torch_profiler_output_dir = Path(tmp_path)
    manager.profile_prefix = ""
    manager.profile_id = "test-profile"
    manager.profiler_activities = ["CPU", "GPU"]
    manager.merge_profiles = False
    manager.ps.dp_size = 1
    manager.ps.pp_size = 1
    manager.ps.moe_ep_size = 1

    module = "sglang.srt.managers.scheduler_components.profiler_manager"
    with (
        patch(f"{module}._is_npu", True),
        patch(f"{module}.torch.distributed.barrier") as barrier,
    ):
        result = manager._stop_profile(ForwardMode.DECODE)

    assert result.success
    barrier.assert_not_called()
