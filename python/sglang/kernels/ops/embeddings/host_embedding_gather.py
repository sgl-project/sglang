"""Zero-copy gather of embedding rows from a page-locked host table.

The embedding table stays in pinned host memory and the GPU reads the requested
rows directly over PCIe through unified virtual addressing (UVA). Compared with
a host-side ``index_select`` followed by a device copy, this needs no
host/device synchronisation and no staging buffer, and the launch is a plain
kernel launch, so it can be captured in a CUDA graph.

A decode step gathers a handful of rows (a few microseconds); a 1024-token
prefill chunk moves ~10 MB for a 5120-wide table and runs at PCIe bandwidth
(~0.4 ms on PCIe 4.0 x16).
"""

import torch
import triton
import triton.language as tl

_TORCH_TO_TRITON_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def _host_embedding_gather_kernel(
    input_ptr,
    # Raw host address of the pinned table, cast to a typed pointer below. The
    # launcher receives a plain integer so no device-placement check is applied
    # to what is, from PyTorch's point of view, a CPU tensor.
    table_addr,
    out_ptr,
    hidden_dim: tl.constexpr,
    table_stride0: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Widen first: Triton specialises small integer arguments to i32 and
    # int_to_ptr needs a 64-bit operand.
    table_ptr = table_addr.to(tl.int64).to(tl.pointer_type(DTYPE))
    row = tl.program_id(0).to(tl.int64)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_H + tl.arange(0, BLOCK_H)
    col_mask = cols < hidden_dim

    token = tl.load(input_ptr + row).to(tl.int64)
    vals = tl.load(table_ptr + token * table_stride0 + cols, mask=col_mask, other=0.0)
    tl.store(out_ptr + row * hidden_dim + cols, vals, mask=col_mask)


def host_embedding_gather(input_: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Return ``table[input_]`` where ``table`` lives in pinned host memory.

    ``input_`` is a CUDA int32/int64 tensor of any shape; the result has shape
    ``(*input_.shape, hidden_dim)`` on the same device with the table's dtype.
    Out-of-range ids are not checked here (the caller masks or asserts them):
    an out-of-range host read is an illegal address for the GPU.
    """
    assert input_.is_cuda
    assert input_.dtype in (torch.int32, torch.int64)
    assert table.device.type == "cpu"
    # cudaPointerGetAttributes is not a stream op, but keep it out of capture
    # anyway: the table's residency is fixed for the process lifetime.
    if not torch.cuda.is_current_stream_capturing():
        assert table.is_pinned(), "host embedding table must be in pinned memory"
    assert table.ndim == 2
    assert table.stride(1) == 1
    assert table.dtype in _TORCH_TO_TRITON_DTYPE

    input_ = input_.contiguous()
    hidden_dim = table.shape[1]
    output = torch.empty(
        (*input_.shape, hidden_dim), dtype=table.dtype, device=input_.device
    )
    n_tokens = input_.numel()
    if n_tokens == 0:
        return output

    block_h = min(1024, triton.next_power_of_2(hidden_dim))
    grid = (n_tokens, triton.cdiv(hidden_dim, block_h))
    _host_embedding_gather_kernel[grid](
        input_,
        table.data_ptr(),
        output,
        hidden_dim,
        table.stride(0),
        DTYPE=_TORCH_TO_TRITON_DTYPE[table.dtype],
        BLOCK_H=block_h,
        num_warps=4,
    )
    return output
