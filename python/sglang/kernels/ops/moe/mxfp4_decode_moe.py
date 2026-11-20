"""Decode routed MoE (TP-MoE rank or one EP rank's local experts) with MXFP4 weights in the engine's
flashinfer_mxfp4 layout x MXFP8 activations, SiTU or clamped SwiGLU, on SM100: a routing kernel and a persistent
FC1 + FC2 kernel chained with PDL, launched by one call. The output is flashinfer's deferred-finalize contract
(permuted BF16 FC2 rows + expanded_idx); the engine applies the top-k weights in its finalize / all-reduce pass.

Kernel source: ``kernels/jit/csrc/moe/mxfp4_decode/moe_mxfp4.cuh``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

MAX_TOKENS = 256  # kMaxM: tokens per call
# FC1 activation functions of the kernel source -> number of float template constants
ACTIVATIONS = {"situ": 2, "swiglu": 1}
# items a CTA may hold per phase (claimed, not yet released); see moe_kernel. FC2's 6 is the info ring's own bound.
FC1_PREFETCH, FC2_PREFETCH = 2, 6


def is_supported_shape(num_experts: int, top_k: int, hidden: int, inter: int) -> bool:
    """Whether the kernels serve this per-rank shape (``inter`` after the 128 padding of the flashinfer_mxfp4
    weights); mirrors the static_asserts of the kernel source."""
    # routing: one thread per expert, a [MAX_TOKENS / 32, E] bitmask + row table in 48 KB of static smem
    return (
        32 <= num_experts <= 1024
        and num_experts % 4 == 0
        and num_experts * MAX_TOKENS // 32 * 6 + 128 <= 48 * 1024
        and 1 <= top_k <= 4096 // MAX_TOKENS
        and hidden % 512 == 0
        and inter % 128 == 0
    )


def default_block_n(num_tokens: int) -> int:
    """Tokens per virtual expert (MMA N). Measured on B300 with real routing: 16 wins at small T, 32 at large T
    (hot experts split into fewer virtual experts)."""
    return 16 if num_tokens <= 64 else 32


@cache_once
def _jit_module(
    num_experts: int,
    top_k: int,
    hidden: int,
    inter: int,
    block_n: int,
    activation: str,
    activation_constants: tuple[float, ...],
    fc1_prefetch: int = FC1_PREFETCH,
    fc2_prefetch: int = FC2_PREFETCH,
) -> tuple[Module, int]:
    if len(activation_constants) != ACTIVATIONS[activation]:
        raise ValueError(
            f"activation {activation!r} takes {ACTIVATIONS[activation]} constants, "
            f"got {activation_constants}"
        )
    # the kernel source's activation function with its constants, as a C++ template argument
    constants = [repr(float(c)) for c in activation_constants]
    cpp_activation = f"{activation}<{', '.join(c + 'f' for c in constants)}>"
    tag = "_".join([activation] + constants).replace(".", "p").replace("-", "m")
    args = make_cpp_args(
        num_experts, top_k, hidden, inter, block_n, fc1_prefetch, fc2_prefetch
    )
    cls = f"MXFP4DecodeMoE<{args}, {cpp_activation}>"
    module = load_jit(
        "mxfp4_decode_moe",
        *args,
        tag,
        cuda_files=["moe/mxfp4_decode/moe_mxfp4.cuh"],
        cuda_wrappers=[
            ("run", f"{cls}::run"),
            ("workspace_bytes", f"{cls}::workspace_bytes"),
        ],
    )
    return module, int(module.workspace_bytes())


def mxfp4_decode_moe(
    x: torch.Tensor,  # [T, H] float8_e4m3fn
    x_sf: torch.Tensor,  # [T, H / 32] E8M0 bytes (linear)
    topk_ids: torch.Tensor,  # [T, top-k] int32 global ids, or the packed (id << 16) | bf16(weight) form
    w13: torch.Tensor,  # [E, 2 I, H / 2] uint8, flashinfer_mxfp4 layout
    w13_sf: torch.Tensor,  # [E, 2 I, H / 32] E8M0 bytes, 128x4 atoms
    w2: torch.Tensor,  # [E, H, I / 2] uint8
    w2_sf: torch.Tensor,  # [E, H, I / 32]
    *,
    activation: str,  # a key of ACTIVATIONS: situ (beta, linear_beta) or swiglu (limit,)
    activation_constants: tuple[float, ...],
    expert_offset: int = 0,
    packed: bool = False,
    block_n: Optional[int] = None,
    fc1_prefetch: int = FC1_PREFETCH,
    fc2_prefetch: int = FC2_PREFETCH,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Routing kernel + MoE kernel. Returns (gemm2_out [T * top-k, H] bf16, expanded_idx [T * top-k] int32, weights):
    out[t] = sum_k topk_weights[t, k] * gemm2_out[expanded_idx[t * top-k + k]], where expanded_idx is -1 for ids
    outside the local experts [expert_offset, expert_offset + E) (weight 0). With `packed`, `weights` is the
    [T, top-k] bf16 half of the packed form, else None. The scratch holds no state across calls (the routing kernel
    resets the counters), so it is allocated per call like the outputs."""
    num_experts, top_k = w13.shape[0], topk_ids.shape[1]
    hidden, inter = x.shape[1], w13.shape[1] // 2
    if block_n is None:
        block_n = default_block_n(x.shape[0])
    module, workspace_bytes = _jit_module(
        num_experts,
        top_k,
        hidden,
        inter,
        block_n,
        activation,
        tuple(activation_constants),
        fc1_prefetch,
        fc2_prefetch,
    )
    num_tokens, device = x.shape[0], x.device
    rows = num_tokens * top_k
    gemm2_out = torch.empty(rows, hidden, dtype=torch.bfloat16, device=device)
    expanded_idx = torch.empty(rows, dtype=torch.int32, device=device)
    weights = (
        torch.empty(num_tokens, top_k, dtype=torch.bfloat16, device=device)
        if packed
        else None
    )
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    module.run(
        x.view(torch.uint8),
        x_sf.view(torch.uint8),
        topk_ids,
        w13.view(torch.uint8),
        w13_sf.view(torch.uint8),
        w2.view(torch.uint8),
        w2_sf.view(torch.uint8),
        expert_offset,
        workspace,
        gemm2_out,
        expanded_idx,
        weights,
    )
    return gemm2_out, expanded_idx, weights
