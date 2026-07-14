import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------- #
# sum bitmatrix rows


@triton.jit
def vpopc(x):
    """
    Vertical popcount
    Input  x : uint32[..., N]
    Output y : uint32[..., 32]
    semantics : y[..., i] = sum_j((x[..., j] >> i) & 1)
    credits: @apgoucher
    """

    tl.static_assert(x.dtype == tl.uint32, "x should consist of 32-bit unsigned integers")

    BLOCK_N: tl.constexpr = x.shape[-1]  # summation axis
    BATCHES: tl.constexpr = x.numel // BLOCK_N  # number of batches
    if BLOCK_N >= 8:
        sa1: tl.constexpr = 8
    else:
        sa1: tl.constexpr = BLOCK_N
    # create 8-way sums in 4-bit fields:
    y = tl.reshape(x, [BATCHES, BLOCK_N // sa1, sa1, 1])
    y = (y >> tl.arange(0, 4)[None, None, None, :]) & 0x11111111
    y = tl.sum(y, 2)  # [BATCHES, BLOCK_N // sa1, 4]
    if BLOCK_N >= 128:
        sa2: tl.constexpr = 16
    else:
        sa2: tl.constexpr = BLOCK_N // sa1
    # create 128-way sums in 8-bit fields:
    y = tl.reshape(y, [BATCHES, BLOCK_N // (sa1 * sa2), sa2, 1, 4])
    y = (y >> (4 * tl.arange(0, 2))[None, None, None, :, None]) & 0x0f0f0f0f
    y = tl.sum(y, 2)  # [BATCHES, BLOCK_N // (sa1 * sa2), 2, 4]
    sa3: tl.constexpr = BLOCK_N // (sa1 * sa2)
    # create N-way sums in 32-bit fields:
    y = tl.reshape(y, [BATCHES, 1, sa3, 8])
    y = (y >> (8 * tl.arange(0, 4))[None, :, None, None]) & 0x000000ff
    y = tl.sum(y, 2)  # [BATCHES, 4, 8]
    y = tl.reshape(y, x.shape[:-1] + [32])
    return y


@triton.jit
def _sum_bitmatrix_rows(B, shape_bm, stride_bm: tl.constexpr, stride_bn: tl.constexpr,  # input bitmatrix
                        Out, OutPartials, stride_pm: tl.constexpr, stride_pn, shape_pn,  # outputs
                        BLOCK_MM: tl.constexpr, BLOCK_M: tl.constexpr):
    tl.static_assert(BLOCK_MM % BLOCK_M == 0)
    TILE_SIZE: tl.constexpr = BLOCK_MM // BLOCK_M
    if isinstance(shape_bm, tl.tensor) and shape_bm.dtype.is_ptr():
        shape_bm = tl.load(shape_bm)
    # load input bits
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_bm = pid_m * BLOCK_MM + tl.arange(0, BLOCK_MM)
    bits = tl.load(B + pid_n * stride_bn + offs_bm * stride_bm, mask=offs_bm < shape_bm, other=0)
    bits = tl.reshape(bits, [TILE_SIZE, BLOCK_M])
    # partial row sum
    partial_row_sum = vpopc(bits)  # [TILE_SIZE, 32]
    # write-back partial row sum
    offs_pm = pid_m * TILE_SIZE + tl.arange(0, TILE_SIZE)
    offs_n = pid_n * 32 + tl.arange(0, 32)
    tl.store(OutPartials + offs_pm[:, None] * stride_pm + offs_n[None, :] * stride_pn, partial_row_sum)
    # update final row sum
    tl.atomic_add(Out + offs_n, tl.sum(partial_row_sum, 0), sem="relaxed")


def cdiv(x, y):
    return (x + y - 1) // y


def sum_bitmatrix_rows(x, partials_block_size=None):
    assert partials_block_size is not None
    PARTIALS_BLOCK_M = partials_block_size
    n_rows, n_cols = x.shape
    n_rows_max = x.shape_max[0]

    TILE_SIZE = max(1, 128 // PARTIALS_BLOCK_M)
    BLOCK_MM = PARTIALS_BLOCK_M * TILE_SIZE

    grid_m = cdiv(n_rows_max, BLOCK_MM)
    grid_n = cdiv(n_cols, 32)
    out = torch.zeros((cdiv(n_cols, 128) * 128, ), device=x.device, dtype=torch.int32)[:n_cols]
    out_partials = torch.empty((grid_n * 32, grid_m * TILE_SIZE), device=x.device, dtype=torch.int32)
    out_partials = torch.transpose(out_partials, 0, 1)
    # output tensors
    _sum_bitmatrix_rows[(grid_m, grid_n)](
        x.storage.data, n_rows, x.stride(0), x.stride(1),  # input
        out,  # output [final reduction]
        out_partials, out_partials.stride(0), out_partials.stride(1),
        out_partials.shape[1],  # output [partial reductions]
        BLOCK_M=PARTIALS_BLOCK_M, BLOCK_MM=BLOCK_MM,  # constants
        num_warps=8)
    out_partials = out_partials[:cdiv(n_rows_max, PARTIALS_BLOCK_M), :]
    return out, out_partials
