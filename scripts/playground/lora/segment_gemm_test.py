import time

import numpy as np
import torch
import triton
import triton.language as tl
from flashinfer import SegmentGEMMWrapper

S = 64
R = 16
H = 4096
BS = 1
SEQ_LENS = [64]
W_INDICES = [0]


@triton.jit
def _segment_gemm_kernel_expand(
    output,  # (s, h)
    x,  # (s, r)
    weights,  # (num_lora, h, r)
    seg_lens,
    seg_start,
    weight_indices,
    H,
    R,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    INPLACE: tl.constexpr,
):
    batch_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)

    width = tl.cdiv(H, BLOCK_H)
    pid_s = pid // width
    pid_h = pid % width

    seg_len = tl.load(seg_lens + batch_id)
    w_index = tl.load(weight_indices + batch_id)

    seg_start = tl.load(seg_start + batch_id)
    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    h_offset = tl.arange(0, BLOCK_H) + pid_h * BLOCK_H
    r_offset = tl.arange(0, BLOCK_R)

    # (BLOCK_S, BLOCK_R)
    x_ptrs = x + seg_start * R + s_offset[:, None] * R + r_offset[None, :]
    # (BLOCK_R, BLOCK_H)
    w_ptrs = weights + w_index * H * R + r_offset[:, None] + h_offset[None, :] * R

    partial_sum = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)
    for rid in range(tl.cdiv(R, BLOCK_R)):
        tiled_x = tl.load(x_ptrs)
        tiled_w = tl.load(w_ptrs)
        partial_sum += tl.dot(tiled_x, tiled_w)
        x_ptrs += BLOCK_R
        w_ptrs += BLOCK_R

    partial_sum = partial_sum.to(x.dtype.element_ty)
    out_ptr = output + seg_start * H + s_offset[:, None] * H + h_offset[None, :]
    if INPLACE:
        partial_sum += tl.load(out_ptr)
    tl.store(out_ptr, partial_sum)


def segment_gemm_triton_expand(
    x,  # (s, r)
    weights,  # (num_lora, h, r)
    batch_size,
    weight_column_major,
    seg_lens,
    weight_indices,
):
    assert weights.ndim == 3
    assert batch_size == seg_lens.shape[0] == weight_indices.shape[0]
    assert weight_column_major
    assert x.shape[-1] == weights.shape[-1]
    assert x.is_contiguous()
    assert weights.is_contiguous()

    BLOCK_S = 16
    BLOCK_H = 32
    BLOCK_R = 16
    sum_S = x.shape[0]
    H = weights.shape[-2]
    R = weights.shape[-1]
    assert H % BLOCK_H == 0 and R % BLOCK_R == 0
    seg_start = torch.cat(
        [torch.tensor([0], device="cuda"), torch.cumsum(seg_lens, dim=0)[:-1]]
    )
    max_S = int(torch.max(seg_lens))
    # TODO: fix the case of S % BLOCK_S != 0

    grid = (
        triton.cdiv(max_S, BLOCK_S) * triton.cdiv(H, BLOCK_H),
        batch_size,
    )

    output = torch.empty(sum_S, H, dtype=x.dtype, device="cuda")
    _segment_gemm_kernel_expand[grid](
        output,  # (s, h)
        x,  # (s, r)
        weights,  # (num_lora, h, r)
        seg_lens,
        seg_start,
        weight_indices,
        H,
        R,
        BLOCK_S,
        BLOCK_H,
        BLOCK_R,
        INPLACE=False,
    )
    return output


def bench_flashinfer():
    x = torch.rand((S, R), dtype=torch.float16, device="cuda")
    w = torch.rand((8, H, R), dtype=torch.float16, device="cuda")
    bs = BS
    seq_lens = torch.tensor(SEQ_LENS, dtype=torch.int32, device="cuda")
    w_indices = torch.tensor(W_INDICES, dtype=torch.int32, device="cuda")

    workspace_buffer = torch.empty(1 * 1024 * 1024, dtype=torch.int8, device="cuda")
    segment_gemm = SegmentGEMMWrapper(workspace_buffer)

    bench_t = []
    for i in range(40):
        torch.cuda.synchronize()
        tic = time.perf_counter()
        output = segment_gemm.run(
            x=x,
            weights=w,
            batch_size=bs,
            weight_column_major=True,
            seg_lens=seq_lens,
            weight_indices=w_indices,
        )
        torch.cuda.synchronize()
        bench_t.append(time.perf_counter() - tic)

    print(output.shape)
    # print([f"{t * 1000:.4f} ms" for t in bench_t])
    print(f"{np.mean(np.array(bench_t[20:])) * 1000:.4f} ms")
    return output


def bench_triton():
    x = torch.rand((S, R), dtype=torch.float16, device="cuda")
    w = torch.rand((8, H, R), dtype=torch.float16, device="cuda")
    bs = BS
    seq_lens = torch.tensor(SEQ_LENS, dtype=torch.int32, device="cuda")
    w_indices = torch.tensor(W_INDICES, dtype=torch.int32, device="cuda")

    bench_t = []
    for i in range(40):
        torch.cuda.synchronize()
        tic = time.perf_counter()
        output = segment_gemm_triton_expand(
            x=x,  # (s, r)
            weights=w,  # (num_lora, h, r)
            batch_size=bs,
            weight_column_major=True,
            seg_lens=seq_lens,
            weight_indices=w_indices,
        )
        torch.cuda.synchronize()
        bench_t.append(time.perf_counter() - tic)

    print(output.shape)
    # print([f"{t * 1000:.4f} ms" for t in bench_t])
    print(f"{np.mean(np.array(bench_t[20:])) * 1000:.4f} ms")
    return output


def bench_torch():
    x = torch.rand((S, R), dtype=torch.float16, device="cuda")
    w = torch.rand((R, H), dtype=torch.float16, device="cuda")

    bench_t = []
    for i in range(40):
        torch.cuda.synchronize()
        tic = time.perf_counter()
        output = torch.matmul(x, w)
        torch.cuda.synchronize()
        bench_t.append(time.perf_counter() - tic)

    print(output.shape)
    # print([f"{t * 1000:.4f} ms" for t in bench_t])
    print(f"{np.mean(np.array(bench_t[20:])) * 1000:.4f} ms")
    return output


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    set_random_seed(42)
    o1 = bench_flashinfer()
    set_random_seed(42)
    o2 = bench_torch()
    set_random_seed(42)
    o3 = bench_triton()

    assert torch.allclose(o1, o3)
