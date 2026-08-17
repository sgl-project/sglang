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


def test_direct_kernel_quant_method_rejected_when_hpc_ops_requested(
    _runner_backend_flag,
):
    # W4AFp8MoEMethod never constructs a MoeRunner (apply() calls its kernel
    # directly), so it bypasses the MoeRunner-level guard; the layer-level
    # check must reject it.
    from sglang.srt.layers.moe.fused_moe_triton.layer import (
        _validate_hpc_ops_quant_method,
    )
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
    from sglang.srt.layers.quantization.w4afp8 import W4AFp8MoEMethod

    _runner_backend_flag.runner_backend = MoeRunnerBackend.HPC_OPS
    with pytest.raises(ValueError, match="hpc_ops"):
        _validate_hpc_ops_quant_method(object.__new__(W4AFp8MoEMethod))
    # The FP8 method (the one the hpc_ops runner supports) passes.
    _validate_hpc_ops_quant_method(object.__new__(Fp8MoEMethod))
    # Without hpc_ops requested, any quant method passes.
    _runner_backend_flag.runner_backend = MoeRunnerBackend.TRITON
    _validate_hpc_ops_quant_method(object.__new__(W4AFp8MoEMethod))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
