# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.multimodal_gen.runtime.layers.utils import direct_register_custom_op


@triton.jit
def _mhc_mix_output_kernel(
    streams_ptr,
    block_out_ptr,
    post_ptr,
    res_ptr,
    out_ptr,
    hidden,
    NUM_STREAM: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token = tl.program_id(0)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < hidden
    offs_n = tl.arange(0, NUM_STREAM)

    acc = tl.zeros([NUM_STREAM, BLOCK_C], dtype=tl.float32)
    for j in tl.static_range(NUM_STREAM):
        stream = tl.load(
            streams_ptr + (token * NUM_STREAM + j) * hidden + offs_c,
            mask=mask_c,
            other=0.0,
        ).to(tl.float32)
        # res[token, :, j]: the column every output stream mixes this stream by.
        res = tl.load(
            res_ptr + token * NUM_STREAM * NUM_STREAM + offs_n * NUM_STREAM + j
        )
        acc += res[:, None] * stream[None, :]

    written = tl.load(block_out_ptr + token * hidden + offs_c, mask=mask_c, other=0.0)
    post = tl.load(post_ptr + token * NUM_STREAM + offs_n)
    acc += post[:, None] * written.to(tl.float32)[None, :]

    tl.store(
        out_ptr + (token * NUM_STREAM + offs_n[:, None]) * hidden + offs_c[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_c[None, :],
    )


def _mhc_mix_output(
    streams: torch.Tensor,
    block_out: torch.Tensor,
    post: torch.Tensor,
    res: torch.Tensor,
) -> torch.Tensor:
    """``res @ streams + post outer block_out`` in one pass, accumulating in fp32.

    The torch spelling reads the 4-stream residual three times and materializes the
    ``post`` outer product at full width only to add it.
    """
    # The kernel indexes every tensor by raw offset (token-major, last dim 1),
    # so a strided view would be read wrong with no error. Every in-tree
    # producer is contiguous; normalise the two activations and keep the
    # explicit copies of the small parameter tensors.
    if not streams.is_contiguous():
        streams = streams.contiguous()
    if not block_out.is_contiguous():
        block_out = block_out.contiguous()
    tokens, num_stream, hidden = streams.shape
    # Always contiguous so the fake and the real impl agree on output strides.
    out = torch.empty(streams.shape, dtype=streams.dtype, device=streams.device)
    block_c = min(1024, triton.next_power_of_2(hidden))
    _mhc_mix_output_kernel[(tokens, triton.cdiv(hidden, block_c))](
        streams,
        block_out,
        post.contiguous(),
        res.contiguous(),
        out,
        hidden,
        NUM_STREAM=num_stream,
        BLOCK_C=block_c,
        num_warps=4,
    )
    return out


def _mhc_mix_output_fake(
    streams: torch.Tensor,
    block_out: torch.Tensor,
    post: torch.Tensor,
    res: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(streams)


# Registered at import, not on first call: a lazily registered op inside a compiled
# region breaks the graph and pays Triton's JIT per block. As an op inductor emits an
# opaque call, which is how the reference keeps its own kernels out of its compiler.
direct_register_custom_op(
    op_name="magi2_mhc_mix_output",
    op_func=_mhc_mix_output,
    mutates_args=[],
    fake_impl=_mhc_mix_output_fake,
)

mhc_mix_output = torch.ops.sglang.magi2_mhc_mix_output
