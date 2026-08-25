from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.dp_attn import (
    should_skip_scheduler_all_gather,
)


def test_dp1_skips_scheduler_all_gather_by_default():
    with envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False):
        assert should_skip_scheduler_all_gather(dp_size=1)


def test_multi_dp_preserves_default_and_explicit_override():
    with envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False):
        assert not should_skip_scheduler_all_gather(dp_size=2)
    with envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(True):
        assert should_skip_scheduler_all_gather(dp_size=2)
