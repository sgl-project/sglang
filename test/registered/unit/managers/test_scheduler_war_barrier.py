from unittest.mock import patch

from sglang.srt.managers.scheduler import _should_enable_war_barrier
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_war_barrier_is_enabled_for_xpu_without_env_flag():
    with (
        patch("sglang.srt.managers.scheduler.is_cuda", return_value=False),
        patch("sglang.srt.managers.scheduler.is_xpu", return_value=True),
        patch(
            "sglang.srt.managers.scheduler.envs.SGLANG_ENABLE_WAR_BARRIER.get",
            return_value=False,
        ),
    ):
        assert _should_enable_war_barrier()


def test_war_barrier_remains_opt_in_for_other_devices():
    with (
        patch("sglang.srt.managers.scheduler.is_cuda", return_value=False),
        patch("sglang.srt.managers.scheduler.is_xpu", return_value=False),
        patch(
            "sglang.srt.managers.scheduler.envs.SGLANG_ENABLE_WAR_BARRIER.get",
            return_value=False,
        ),
    ):
        assert not _should_enable_war_barrier()


def test_war_barrier_env_flag_still_enables_other_devices():
    with (
        patch("sglang.srt.managers.scheduler.is_cuda", return_value=False),
        patch("sglang.srt.managers.scheduler.is_xpu", return_value=False),
        patch(
            "sglang.srt.managers.scheduler.envs.SGLANG_ENABLE_WAR_BARRIER.get",
            return_value=True,
        ),
    ):
        assert _should_enable_war_barrier()
