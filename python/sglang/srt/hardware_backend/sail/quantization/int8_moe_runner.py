# SPDX-License-Identifier: Apache-2.0

"""Pick the INT8 MoE runner for SAIL.

Both INT8 MoE paths need the same answer: ``quantization/w8a8_int8.W8A8Int8MoEMethod``
and the compressed-tensors scheme next door. DeepEP and Mooncake have no Triton
INT8 all-to-all, so those pairs must run on DeepGEMM; a plain run stays on Triton
unless DeepGEMM is asked for explicitly.
"""

from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.utils import (
    MoeRunnerBackend,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)


def select_int8_moe_runner_backend() -> MoeRunnerBackend:
    moe_runner_backend = MoeRunnerBackend.TRITON
    requested_backend = get_moe_runner_backend()
    if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and (
        get_moe_a2a_backend().is_deepep()
        or get_moe_a2a_backend().is_mooncake()
        or requested_backend.is_deep_gemm()
        or (requested_backend.is_auto() and envs.SGLANG_SAIL_DEEPGEMM_MOE.get())
    ):
        moe_runner_backend = MoeRunnerBackend.DEEP_GEMM
    elif get_moe_a2a_backend().is_deepep() or requested_backend.is_deep_gemm():
        raise ValueError("SAIL INT8 DeepEP requires an available DeepGEMM backend")
    return moe_runner_backend
