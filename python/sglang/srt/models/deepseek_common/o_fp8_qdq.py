"""In-place FP8 (e4m3) quantize-dequantize of an activation in blocks along its last dim,
with power-of-two scales: the reference act_quant(x, block, "ue8m0", ..., inplace=True)
of DeepSeek's inference code. amax is floored at 1e-4, the scale is 2^ceil(log2(amax/448)),
the value x/scale is clamped to +-448, rounded to e4m3 and multiplied back (bit-exact
against the reference: the scale is a power of two).

DeepSeek-V4.1-Flash-1017 runs the attention output o through it (block 32) before the
bf16 wo_a einsum; the earlier V4.1 checkpoint did not.

The e4m3 rounding uses the hardware conversion on sm89 and newer. Older GPUs (Ampere
and before, where Triton has no e4m3 type) get the same rounding in fp32 arithmetic
(round half to even onto the e4m3 grid, signed zero kept); SGLANG_DSV4_O_FP8_QDQ_EMULATE=1 forces
that path anywhere, for testing.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs


@triton.jit
def _o_fp8_qdq_kernel(
    x_ptr, n_blocks, BLOCK: tl.constexpr, ROWS: tl.constexpr, NATIVE: tl.constexpr
):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    mask = rows[:, None] < n_blocks
    offs = rows[:, None].to(tl.int64) * BLOCK + cols[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    t = amax * (1.0 / 448.0)
    bits = t.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) - 127 + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
    s = ((e + 127) << 23).to(tl.float32, bitcast=True)
    v = tl.minimum(tl.maximum(x / s[:, None], -448.0), 448.0)
    if NATIVE:
        q = v.to(tl.float8e4nv).to(tl.float32)
    else:
        # the e4m3 grid: 3 mantissa bits, exponents down to -6, subnormal step 2^-9
        a = tl.abs(v)
        ea = tl.maximum(((a.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 127, -6)
        ulp = ((ea - 3 + 127) << 23).to(tl.float32, bitcast=True)
        r = a / ulp  # exact: ulp is a power of two; r < 16
        r = (r + 12582912.0) - 12582912.0  # round half to even (1.5 * 2^23)
        q = r * ulp
        # the sign by its bit: Triton's -q is 0 - q, which turns -0 into +0
        sign = v.to(tl.int32, bitcast=True) & -2147483648
        q = (q.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True)
    y = q * s[:, None]
    tl.store(x_ptr + offs, y.to(x_ptr.dtype.element_ty), mask=mask)


_NATIVE = {}


def _native_fp8(device) -> bool:
    if envs.SGLANG_DSV4_O_FP8_QDQ_EMULATE.get():
        return False
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _NATIVE:
        _NATIVE[idx] = torch.cuda.get_device_capability(idx) >= (8, 9)
    return _NATIVE[idx]


def o_fp8_qdq_(x: torch.Tensor, block: int) -> torch.Tensor:
    """x in place (made contiguous first if it is not); returns the tensor holding the result."""
    if x.shape[-1] % block:
        raise ValueError(f"last dim {x.shape[-1]} is not a multiple of block {block}")
    if not x.is_contiguous():
        x = x.contiguous()
    n_blocks = x.numel() // block
    if n_blocks:
        rows = 64
        _o_fp8_qdq_kernel[(triton.cdiv(n_blocks, rows),)](
            x, n_blocks, BLOCK=block, ROWS=rows, NATIVE=_native_fp8(x.device)
        )
    return x


def e4m3_round_emulated(v: torch.Tensor) -> torch.Tensor:
    """The non-native kernel path's rounding in torch (fp32 in, fp32 out, |v| <= 448)."""
    a = v.abs()
    ea = (((a.view(torch.int32) >> 23) & 0xFF) - 127).clamp_min(-6)
    ulp = ((ea - 3 + 127) << 23).view(torch.float32)
    r = a / ulp
    r = (r + 12582912.0) - 12582912.0
    q = r * ulp
    return torch.where(v.view(torch.int32) < 0, -q, q)


def o_fp8_qdq_reference(x: torch.Tensor, block: int) -> torch.Tensor:
    """The same in plain torch, out of place (for the self-test)."""
    shape = x.shape
    f = x.float().reshape(-1, block)
    amax = f.abs().amax(dim=1).clamp_min(1e-4)
    bits = (amax * (1.0 / 448.0)).view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    s = ((e + 127) << 23).view(torch.float32)[:, None]
    v = (f / s).clamp(-448.0, 448.0)
    return (v.to(torch.float8_e4m3fn).float() * s).to(x.dtype).reshape(shape)
