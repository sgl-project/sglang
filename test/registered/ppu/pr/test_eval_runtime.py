"""Keep runtime-driver exchange coverage on the worker with the real PPU SDK."""

import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits import answer_suite_kit, perf_suite_kit

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from registered.ppu.answer_eval.test_ppu_answer_eval_unit import (  # noqa: E402,F401
    TestPPUAnswerMultiNodeExchange,
)
from registered.ppu.perf_eval.test_ppu_perf_eval_unit import (  # noqa: E402,F401
    TestPPUPerfMultiNodeExchange,
)

register_ppu_ci(est_time=20, suite="stage-b-test-1-gpu-ppu")

# These drivers must import from the tested checkout, never fall back to skips.
_checkout = Path(__file__).resolve().parents[4] / "python"
for _module in (answer_suite_kit, perf_suite_kit):
    assert Path(_module.__file__).resolve().is_relative_to(_checkout)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
