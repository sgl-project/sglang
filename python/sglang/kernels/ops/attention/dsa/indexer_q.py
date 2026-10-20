"""DSA only. Indexer Q kernels (JIT)."""

from typing import Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.attention.dsv4.utils import make_name


# V3.2 lays q out as [rope | nope] (V4 is [nope | rope]) -> kRopeFirst=true, and
# drops the Hadamard rotation (kHadamard=false).
@cache_once
def _jit_main_q_indexer_rope_first_quant_module(dtype: torch.dtype):
    args = make_cpp_args(dtype, is_arch_support_pdl(), True, False)
    return load_jit(
        make_name("main_q_indexer_rope_first_quant"),
        *args,
        cuda_files=["deepseek_v4/main_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedQIndexerRopeHadamardQuantKernel<{args}>::forward"),
        ],
    )


def fused_q_indexer_rope_first_quant(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: float,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek-V3.2 only. Indexer Q: RoPE on the leading dims + fp8 act-quant. CUDA only."""
    q_fp8 = torch.empty(q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device)
    weights_out = torch.empty(
        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device
    )
    module = _jit_main_q_indexer_rope_first_quant_module(q_input.dtype)
    module.forward(
        q_input,
        q_fp8,
        weight,
        weights_out,
        float(weight_scale),
        cos_sin_cache,
        positions,
    )
    return q_fp8, weights_out
