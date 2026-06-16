import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_gate_mul_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    g = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    out = x * tl.sigmoid(g)
    tl.store(out_ptr + offsets, out.to(x_ptr.dtype.element_ty), mask=mask)


def sigmoid_gate_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Compute x * sigmoid(gate) in a single fused kernel."""
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _sigmoid_gate_mul_kernel[grid](x, gate, out, n, BLOCK_SIZE=1024)
    return out
