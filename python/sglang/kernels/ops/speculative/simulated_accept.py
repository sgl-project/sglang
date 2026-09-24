import torch
import triton
import triton.language as tl


@triton.jit
def _simulated_accept_kernel(
    source_index,
    output_index,
    predict,
    correct,
    candidates,
    target,
    BS: tl.constexpr,
    WIDTH: tl.constexpr,
    LENGTH: tl.constexpr,
    PREDICT_SIZE: tl.constexpr,
    INDEX_ROW: tl.constexpr,
    PREDICT_STRIDE: tl.constexpr,
    CORRECT_STRIDE: tl.constexpr,
    CANDIDATE_ROW: tl.constexpr,
    CANDIDATE_COL: tl.constexpr,
    TARGET_ROW: tl.constexpr,
    TARGET_COL: tl.constexpr,
    REAL_TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    row = offset // WIDTH
    col = offset % WIDTH
    valid = row < BS
    base = tl.load(source_index + row * INDEX_ROW, valid, other=0).to(tl.int64)
    tl.store(output_index + offset, tl.where(col < LENGTH, base + col, -1), valid)
    tl.store(correct + offset * CORRECT_STRIDE, LENGTH - 1, offset < BS)
    if REAL_TOKENS:
        draft = tl.load(
            candidates + row * CANDIDATE_ROW + (col + 1) * CANDIDATE_COL,
            valid & (col + 1 < LENGTH),
            other=0,
        )
        bonus = tl.load(
            target + row * TARGET_ROW + (LENGTH - 1) * TARGET_COL,
            valid & (col == LENGTH - 1),
            other=0,
        )
        tl.store(
            predict + (base + col) * PREDICT_STRIDE,
            tl.where(col == LENGTH - 1, bonus, draft),
            valid & (col < LENGTH),
        )
    else:
        tl.store(predict + offset * PREDICT_STRIDE, 100, offset < PREDICT_SIZE)


def simulated_accept(
    accept_index, predict, correct, candidates, target, width, length, real_tokens
):
    bs = accept_index.shape[0]
    output = torch.empty((bs, width), dtype=torch.int32, device=accept_index.device)
    size = max(bs * width, predict.numel())
    if size:
        _simulated_accept_kernel[(triton.cdiv(size, 128),)](
            accept_index,
            output,
            predict,
            correct,
            candidates,
            target,
            bs,
            width,
            length,
            predict.numel(),
            accept_index.stride(0),
            predict.stride(0),
            correct.stride(0),
            candidates.stride(0) if real_tokens else 0,
            candidates.stride(1) if real_tokens else 0,
            target.stride(0) if real_tokens else 0,
            target.stride(1) if real_tokens else 0,
            REAL_TOKENS=real_tokens,
            BLOCK=128,
        )
    return output
