"""Collect the hardware-free evaluator assertions, not the serving drivers."""

import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from registered.ppu.answer_eval.test_ppu_answer_eval_unit import (  # noqa: E402,F401
    TestPPUAnswerEval,
)
from registered.ppu.perf_eval.test_ppu_perf_eval_unit import (  # noqa: E402,F401
    TestPPUPerfEval,
)

register_cpu_ci(est_time=20, suite="stage-a-test-cpu-ppu")

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
