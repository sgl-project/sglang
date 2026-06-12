"""EAGLE spec-decoding core on CPU: the standard config (topk=1, page_size=1)
plus a topk=4 tree-drafting suite, both on the synchronous (non-overlap) path.

CPU mirror of test/registered/spec/eagle/test_spec_eagle.py. Unlike the CUDA
file there is no Overlap class: ServerArgs force-disables the overlap
scheduler for EAGLE-family speculative decoding on CPU (see "Overlap schedule
is disabled for speculative decoding on CPU" in
sglang/srt/arg_groups/speculative_hook.py), so only the non-overlap path is
reachable. intel_amx is pinned (the x86 CPU default) so a default-selection
change can't silently alter what this exercises.
"""

import os
import unittest

# Must be set before any sglang import: is_cpu() is lru_cached, and device
# auto-detection in popen_launch_server relies on it to append `--device cpu`
# (the env is also inherited by the server subprocess). CPU CI sets it
# already; this keeps manual runs working.
os.environ["SGLANG_USE_CPU_ENGINE"] = "1"

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecCorrectnessKit,
    SpecFeatureKit,
    SpecLogprobKit,
    SpecPenaltyKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import EagleLlama2Base

register_cpu_ci(est_time=900, suite="base-b-test-cpu")

_KITS = (
    SpecCorrectnessKit,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecFeatureKit,
    MatchedStopMixin,
)


class _Core(EagleLlama2Base):
    """EAGLE (Llama-2) preset on CPU.

    SGLANG_USE_CPU_ENGINE=1 is set at module import (above), so
    popen_launch_server auto-appends ``--device cpu`` and the server
    subprocess inherits the CPU-engine env.
    """

    attention_backend = "intel_amx"
    # Overlap is force-disabled for CPU spec decoding by ServerArgs (see module
    # docstring); pass the flag explicitly so the launch args state what runs.
    disable_overlap = True
    # CPU sizes the static pool from host RAM; 0.3 matches the other CPU
    # server tests (the CUDA preset's 0.7 would grab most of the host's RAM).
    mem_fraction_static = 0.3
    # 200 gsm8k examples don't fit the CPU runner's time budget.
    gsm8k_num_examples = 64
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagleLlama2NoOverlap(_Core, *_KITS):
    """Spec v1 (overlap scheduler off) -- the only mode reachable on CPU."""

    # Standard chain config (topk=1, page_size=1), same shape as the CUDA core.
    spec_steps = 5
    spec_topk = 1
    spec_tokens = 6
    # EAGLE/Llama-2 topk=1 accepts modestly; tune against CI if needed.
    acc_length_thres = 1.6
    batch_accept_len_thres = 1.3
    gsm8k_accept_len_thres = 1.3


class TestEagleLlama2Topk4(
    _Core,
    SpecCorrectnessKit,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecFeatureKit,
):
    """EAGLE/Llama-2 topk=4 tree coverage (kits listed in bases), mirroring
    test_spec_eagle_topk.py's TestEagleLlama2Suite at a CPU-sized tree."""

    spec_steps = 3
    spec_topk = 4
    spec_tokens = 8
    # A 3-step / 8-token tree caps acceptance at 4 tokens per verify, so the
    # CUDA suite's topk=8 thresholds don't transfer; tune against CI if needed.
    acc_length_thres = 2.4
    batch_accept_len_thres = 1.6
    gsm8k_accept_len_thres = 2.0


if __name__ == "__main__":
    unittest.main()
