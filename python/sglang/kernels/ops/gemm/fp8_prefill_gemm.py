"""Prefill-width FP8 dense GEMMs on gfx950 (Muse Spark attention QKV / O projections, MoE pooler / unpooler).

Above the row counts ``fp8w_small_m_gemm`` serves, the GEMM is aiter's CK a8w8 row-wise kernel: FP8 e4m3
activation rows with one FP32 scale each, the FP8 weight copy ``quantize_weight`` makes at load (one FP32 scale per
output channel), BF16 output. The activation's per-row quantization is fused into the boundary kernel that produces
it where one exists (``fused_norm_residual`` with ``fp8_out``) and otherwise is ``quantize_rows_fp8`` here, one
Triton pass over the BF16 rows (5.9-6.9 TB/s; aiter's ``pertoken_quant`` reads these widths at 0.3 TB/s and would
cost more than the GEMM saves). Measured cold on MI355X at a 32K-token TP4 extend's shard width of 8192 rows,
quant + GEMM vs BF16 hipBLASLt (bench_prefill_fp8_dense.py): gathered Ulysses QKV [18432, 8192] 1649 -> 1259 us,
Ulysses O [8192, 8192] 734 -> 468, pooler [4096, 8192] 362 -> 315, unpooler [8192, 4096] 367 -> 319; per-rank QKV
[4608, 8192] 550 -> 394 and O [8192, 2048] 207 -> 176 at the same rows. aiter has no tuned a8w8 config for any
of these shapes (its default CK config runs at 1.7-2.6 PF/s).
"""

import torch
import triton
import triton.language as tl

F8 = torch.float8_e4m3fn
# Rows from which the CK GEMM (+ the row quantization) beats hipBLASLt, measured cold on MI355X
# (bench_prefill_fp8_dense.py): the per-rank TP4 / TP8 shapes only from 2048 rows (aiter's default a8w8 config is
# slower than BF16 at 129..1024 rows, +5..+90 %, -4..-18 % from 4096); the gathered Ulysses QKV / O [18432, 8192] /
# [8192, 8192] from any SP shard width (-3..-38 % at 256..2048 rows, -24..-39 % above).
MIN_ROWS = 2048
_MIN_ROWS = {(18432, 8192): 129, (8192, 8192): 129}
# K -> (rows per program, warps) for the row quantization; tuned on MI355X at 32768 rows (bench_prefill_fp8_dense.py
# --tune-quant): 29 / 68 / 135 us for K 2048 / 4096 / 8192, 5.9-6.9 TB/s.
_ROW_QUANT_CONFIGS = {2048: (1, 2), 4096: (1, 2), 8192: (1, 4)}
_gemm_a8w8_ck = None


@triton.jit
def _quantize_rows_kernel(
    X,
    XQ,
    XS,
    M,
    stride_w,
    K: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_K)
    mask = (rows[:, None] < M) & (cols[None, :] < K)
    # Row r is K // HW pieces of HW contiguous columns, piece w at X + w * stride_w + r * HW (HW = K: row-major).
    src = (
        X
        + rows.to(tl.int64)[:, None] * HW
        + (cols // HW).to(tl.int64)[None, :] * stride_w
        + (cols % HW)[None, :]
    )
    x = tl.load(src, mask=mask, other=0.0).to(tl.float32)
    # 448 = e4m3 max; the floor keeps an all-zero row at zeros instead of NaN (same as the boundary kernel).
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-20)
    f8_max = tl.full(amax.shape, 448.0, tl.float32)
    # Correctly rounded per-row divisions (Triton's `/` is an approximate reciprocal on AMD), then one exact
    # multiply per element: bit-identical to torch's ``x * (448 / amax)`` and ``amax / 448``.
    scale = tl.div_rn(amax, f8_max)
    inv = tl.div_rn(f8_max, amax)
    xq = (x * inv[:, None]).to(tl.float8e4nv)
    tl.store(XQ + rows.to(tl.int64)[:, None] * K + cols[None, :], xq, mask=mask)
    tl.store(XS + rows, scale, mask=rows < M)


def _quantize(x: torch.Tensor, m: int, k: int, hw: int, stride_w: int):
    xq = torch.empty((m, k), device=x.device, dtype=F8)
    xs = torch.empty((m,), device=x.device, dtype=torch.float32)
    if m == 0:
        return xq, xs
    block_m, warps = _ROW_QUANT_CONFIGS.get(k, (1, 4))
    _quantize_rows_kernel[(triton.cdiv(m, block_m),)](
        x,
        xq,
        xs,
        m,
        stride_w,
        k,
        hw,
        block_m,
        triton.next_power_of_2(k),
        num_warps=warps,
    )
    return xq, xs


def quantize_rows_fp8(x: torch.Tensor):
    """Per-row FP8 e4m3 copy of the BF16 rows ``x [M, K]`` and their FP32 scales [M]: ``xs = amax / 448`` and
    ``xq = fp8(x * (448 / amax))`` per row, so ``xq * xs[:, None] ~= x``."""
    assert x.dim() == 2 and x.dtype == torch.bfloat16
    x = x.contiguous()
    return _quantize(x, x.shape[0], x.shape[1], x.shape[1], 0)


def quantize_head_rows_fp8(x: torch.Tensor):
    """``quantize_rows_fp8(x.transpose(0, 1).reshape(M, W * HW))`` of ``x [W, M, HW]`` read in place: the Ulysses
    all-to-all output, this rank's token rows with the head blocks outermost."""
    assert x.dim() == 3 and x.dtype == torch.bfloat16
    x = x.contiguous()
    w, m, hw = x.shape
    return _quantize(x, m, w * hw, hw, m * hw)


def min_rows(n: int, k: int) -> int:
    """Rows from which an [N, K] weight's GEMM runs on the FP8 copy."""
    return _MIN_ROWS.get((n, k), MIN_ROWS)


def fp8_prefill_gemm_applies(x: torch.Tensor, fp8_weight) -> bool:
    """Whether ``fp8_prefill_gemm`` serves these operands: an activation (BF16 rows, or their FP8 copy) at or above
    the weight's ``min_rows`` against ``quantize_weight(W)``. The K / N multiples are what the CK row-wise kernels
    were checked with."""
    if fp8_weight is None or x.dim() != 2:
        return False
    n, k = fp8_weight[0].shape
    return (
        x.shape[0] >= min_rows(n, k)
        and x.dtype in (torch.bfloat16, F8)
        and x.shape[1] == k
        and k % 64 == 0
        and n % 64 == 0
    )


def _gemm():
    global _gemm_a8w8_ck
    if _gemm_a8w8_ck is None:
        from aiter import gemm_a8w8_CK

        _gemm_a8w8_ck = gemm_a8w8_CK
    return _gemm_a8w8_ck


def fp8_prefill_gemm(xq: torch.Tensor, xs: torch.Tensor, fp8_weight) -> torch.Tensor:
    """``(xq * xs[:, None]) @ (wq * ws[:, None]).T`` in BF16 through aiter's CK a8w8 GEMM; it takes the activation
    scales as [M, 1] and the channel scales as [N, 1], both FP32."""
    wq, ws = fp8_weight
    return _gemm()(
        xq, wq, xs.view(-1, 1), ws.view(-1, 1), bias=None, dtype=torch.bfloat16
    )
