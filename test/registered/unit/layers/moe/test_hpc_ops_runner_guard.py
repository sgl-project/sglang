"""The hpc_ops MoE runner backend makes the standard dispatcher keep global
expert ids, so a quant method that silently falls back to another runner
(e.g. an unquantized MoE) would misroute tokens under EP>1. MoeRunner must
reject that combination loudly at startup.
"""

import sys

import pytest

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.runtime_context import get_flags
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


@pytest.fixture
def _runner_backend_flag():
    moe = get_flags().moe
    saved = moe.runner_backend
    yield moe
    moe.runner_backend = saved


def test_non_hpc_runner_rejected_when_hpc_ops_requested(_runner_backend_flag):
    _runner_backend_flag.runner_backend = MoeRunnerBackend.HPC_OPS
    with pytest.raises(ValueError, match="hpc_ops"):
        MoeRunner(MoeRunnerBackend.TRITON, MoeRunnerConfig())


def test_triton_runner_allowed_without_hpc_ops(_runner_backend_flag):
    _runner_backend_flag.runner_backend = MoeRunnerBackend.TRITON
    runner = MoeRunner(MoeRunnerBackend.TRITON, MoeRunnerConfig())
    assert runner.runner_core is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
