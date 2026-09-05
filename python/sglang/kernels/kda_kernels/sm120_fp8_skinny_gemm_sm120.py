# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# KDA provenance: this kernel was automatically optimized by the Humanize2
# workflow (https://github.com/PolyArch/humanize) and Kernel Design Agents
# (https://github.com/mit-han-lab/kernel-design-agents).
# Source: https://github.com/BBuf/KDA-Pilot/pull/199 @
# 3c0294ba40e00b23138cd2e256b22a9519298da0.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This file integrates the KDA-Pilot SM120 FP8 skinny GEMM with SGLang's JIT
# kernel and KDA backend interfaces.

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kda_kernels import _cuda_source

if TYPE_CHECKING:
    import torch
    from tvm_ffi.module import Module


@cache_once
def _jit_sm120_fp8_skinny_module() -> Module:
    import torch

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("KDA FP8 skinny GEMM requires CUDA SM120")
    return load_jit(
        "kda_sm120_fp8_skinny_gemm",
        cuda_files=[_cuda_source("gemm/sm120_fp8_skinny_gemm.cuh")],
        cuda_wrappers=[
            ("run_quantized", "KdaSm120Fp8SkinnyGemm::run_quantized"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "-DCUTLASS_ENABLE_GDC_FOR_SM100",
            "--expt-relaxed-constexpr",
            "-static-global-template-stub=false",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
        extra_dependencies=["cutlass"],
    )


def _run_sm120_fp8_skinny_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
) -> torch.Tensor:
    """Compute a static per-tensor FP8 linear for an accepted decode shape.

    ``input`` is BF16 ``[M, K]``. ``weight`` is the column-major ``[K, N]``
    view used by SGLang's FP8 linear path. ``output_scale`` is the precomputed
    ``input_scale * weight_scale`` used by FlashInfer's cuBLAS backend. The
    activation is quantized with SGLang's reference static-FP8 kernel before
    the SM120 CUTLASS GEMM so the full path stays within BF16 tolerance of the
    FlashInfer fallback for real model activations, including saturation edges.
    """
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    quantized, _ = static_quant_fp8(input, input_scale, repeat_scale=False)
    return _run_sm120_fp8_skinny_gemm_quantized(quantized, weight, output_scale)


def _run_sm120_fp8_skinny_gemm_quantized(
    input: torch.Tensor,
    weight: torch.Tensor,
    output_scale: torch.Tensor,
) -> torch.Tensor:
    """Run the low-level FP8-input entry used for kernel qualification."""
    import torch

    output = torch.empty(
        (input.shape[0], weight.shape[1]),
        dtype=torch.bfloat16,
        device=input.device,
    )
    _jit_sm120_fp8_skinny_module().run_quantized(
        input, weight, output_scale.reshape(()), output
    )
    return output
