"""Startup guards for MoE runner and dispatcher quantization contracts."""

import sys
from types import SimpleNamespace

import pytest

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.runtime_context import get_flags
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="stage-b-test-cpu-intel")


@pytest.fixture
def _moe_flags():
    moe = get_flags().moe
    saved = (moe.runner_backend, moe.a2a_backend)
    yield moe
    moe.runner_backend, moe.a2a_backend = saved


def test_non_hpc_runner_rejected_when_hpc_ops_requested(_moe_flags):
    _moe_flags.runner_backend = MoeRunnerBackend.HPC_OPS
    with pytest.raises(ValueError, match="hpc_ops"):
        MoeRunner(MoeRunnerBackend.TRITON, MoeRunnerConfig())


def test_triton_runner_allowed_without_hpc_ops(_moe_flags):
    _moe_flags.runner_backend = MoeRunnerBackend.TRITON
    runner = MoeRunner(MoeRunnerBackend.TRITON, MoeRunnerConfig())
    assert runner.runner_core is not None


def test_direct_kernel_quant_method_rejected_when_hpc_ops_requested(
    _moe_flags,
):
    # W4AFp8MoEMethod never constructs a MoeRunner (apply() calls its kernel
    # directly), so it bypasses the MoeRunner-level guard; the layer-level
    # check must reject it.
    from sglang.srt.layers.moe.fused_moe_triton.layer import (
        _validate_hpc_ops_quant_method,
    )
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
    from sglang.srt.layers.quantization.w4afp8 import W4AFp8MoEMethod

    _moe_flags.runner_backend = MoeRunnerBackend.HPC_OPS
    with pytest.raises(ValueError, match="hpc_ops"):
        _validate_hpc_ops_quant_method(object.__new__(W4AFp8MoEMethod))
    # The FP8 method (the one the hpc_ops runner supports) passes.
    _validate_hpc_ops_quant_method(object.__new__(Fp8MoEMethod))
    # Without hpc_ops requested, any quant method passes.
    _moe_flags.runner_backend = MoeRunnerBackend.TRITON
    _validate_hpc_ops_quant_method(object.__new__(W4AFp8MoEMethod))


def _fp8_method(**overrides):
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

    values = {
        "activation_scheme": "dynamic",
        "weight_block_size": (128, 128),
        "use_mxfp8": False,
        "is_fp4_expert": False,
    }
    values.update(overrides)
    method = object.__new__(Fp8MoEMethod)
    method.quant_config = SimpleNamespace(
        activation_scheme=values["activation_scheme"],
    )
    method.weight_block_size = values["weight_block_size"]
    method.use_mxfp8 = values["use_mxfp8"]
    method.is_fp4_expert = values["is_fp4_expert"]
    return method


def test_deepep_v2_quant_contract_accepts_blockwise_fp8(_moe_flags):
    from sglang.srt.layers.moe.fused_moe_triton.layer import (
        _validate_deepep_v2_quant_method,
    )

    _moe_flags.a2a_backend = MoeA2ABackend.DEEPEP_V2
    _validate_deepep_v2_quant_method(_fp8_method(weight_block_size=[128, 128]))


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"activation_scheme": "static"}, "activation_scheme"),
        ({"weight_block_size": None}, "weight_block_size"),
        ({"weight_block_size": (1, 32), "use_mxfp8": True}, "MXFP8"),
        ({"is_fp4_expert": True}, "FP4 experts"),
    ],
)
def test_deepep_v2_quant_contract_rejects_incompatible_fp8(
    _moe_flags, overrides, expected
):
    from sglang.srt.layers.moe.fused_moe_triton.layer import (
        _validate_deepep_v2_quant_method,
    )

    _moe_flags.a2a_backend = MoeA2ABackend.DEEPEP_V2
    with pytest.raises(ValueError, match=expected):
        _validate_deepep_v2_quant_method(_fp8_method(**overrides))


def test_deepep_v2_quant_contract_rejects_incompatible_methods(_moe_flags):
    from sglang.srt.layers.moe.fused_moe_triton.layer import (
        _validate_deepep_v2_quant_method,
    )
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    from sglang.srt.layers.quantization.w4afp8 import W4AFp8MoEMethod

    _moe_flags.a2a_backend = MoeA2ABackend.DEEPEP_V2
    for method_type in (UnquantizedFusedMoEMethod, W4AFp8MoEMethod):
        with pytest.raises(ValueError, match=method_type.__name__):
            _validate_deepep_v2_quant_method(object.__new__(method_type))


def test_deepep_v2_quant_contract_does_not_affect_other_backends(_moe_flags):
    from sglang.srt.layers.moe.fused_moe_triton.layer import (
        _validate_deepep_v2_quant_method,
    )
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    _moe_flags.a2a_backend = MoeA2ABackend.DEEPEP
    _validate_deepep_v2_quant_method(object.__new__(UnquantizedFusedMoEMethod))


def test_deepep_v2_runner_backstop(_moe_flags):
    _moe_flags.a2a_backend = MoeA2ABackend.DEEPEP_V2
    with pytest.raises(ValueError, match="deep_gemm"):
        MoeRunner(MoeRunnerBackend.TRITON, MoeRunnerConfig())
    assert MoeRunner(MoeRunnerBackend.DEEP_GEMM, MoeRunnerConfig()).runner_core


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
