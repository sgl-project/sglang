import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "BS",
        "NUM_ACCEPT_TOKENS",
        "PREDICT_SIZE",
        "NUM_CORRECT_DRAFTS_SIZE",
        "INDEX_ROW",
        "PREDICT_STRIDE",
        "NUM_CORRECT_DRAFTS_STRIDE",
        "CANDIDATE_ROW",
        "CANDIDATE_COL",
        "TARGET_ROW",
        "TARGET_COL",
    ]
)
def _simulated_accept_kernel(
    source_indices,
    output_indices,
    predict_tokens,
    num_correct_drafts,
    candidate_tokens,
    target_tokens,
    BS,
    WIDTH: tl.constexpr,
    NUM_ACCEPT_TOKENS,
    PREDICT_SIZE,
    NUM_CORRECT_DRAFTS_SIZE,
    INDEX_ROW,
    PREDICT_STRIDE,
    NUM_CORRECT_DRAFTS_STRIDE,
    CANDIDATE_ROW,
    CANDIDATE_COL,
    TARGET_ROW,
    TARGET_COL,
    REAL_TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    row = offset // WIDTH
    col = offset % WIDTH
    valid = row < BS
    base = tl.load(source_indices + row * INDEX_ROW, valid, other=0).to(tl.int64)
    tl.store(
        output_indices + offset,
        tl.where(col < NUM_ACCEPT_TOKENS, base + col, -1),
        valid,
    )
    tl.store(
        num_correct_drafts + offset * NUM_CORRECT_DRAFTS_STRIDE,
        NUM_ACCEPT_TOKENS - 1,
        offset < NUM_CORRECT_DRAFTS_SIZE,
    )
    if REAL_TOKENS:
        draft = tl.load(
            candidate_tokens + row * CANDIDATE_ROW + (col + 1) * CANDIDATE_COL,
            valid & (col + 1 < NUM_ACCEPT_TOKENS),
            other=0,
        )
        bonus = tl.load(
            target_tokens + row * TARGET_ROW + (NUM_ACCEPT_TOKENS - 1) * TARGET_COL,
            valid & (col == NUM_ACCEPT_TOKENS - 1),
            other=0,
        )
        tl.store(
            predict_tokens + (base + col) * PREDICT_STRIDE,
            tl.where(col == NUM_ACCEPT_TOKENS - 1, bonus, draft),
            valid & (col < NUM_ACCEPT_TOKENS),
        )
    else:
        tl.store(predict_tokens + offset * PREDICT_STRIDE, 100, offset < PREDICT_SIZE)


def simulated_accept(
    *,
    accept_indices,
    predict_tokens,
    num_correct_drafts,
    candidate_tokens,
    target_tokens,
    width,
    num_accept_tokens,
    real_tokens,
):
    assert predict_tokens.dim() == 1
    assert num_correct_drafts.dim() == 1
    bs = accept_indices.shape[0]
    output_indices = torch.empty(
        (bs, width), dtype=torch.int32, device=accept_indices.device
    )
    size = max(bs * width, predict_tokens.numel(), num_correct_drafts.numel())
    if size:
        _simulated_accept_kernel[(triton.cdiv(size, 128),)](
            source_indices=accept_indices,
            output_indices=output_indices,
            predict_tokens=predict_tokens,
            num_correct_drafts=num_correct_drafts,
            candidate_tokens=candidate_tokens if real_tokens else predict_tokens,
            target_tokens=target_tokens if real_tokens else predict_tokens,
            BS=bs,
            WIDTH=width,
            NUM_ACCEPT_TOKENS=num_accept_tokens,
            PREDICT_SIZE=predict_tokens.numel(),
            NUM_CORRECT_DRAFTS_SIZE=num_correct_drafts.numel(),
            INDEX_ROW=accept_indices.stride(0),
            PREDICT_STRIDE=predict_tokens.stride(0),
            NUM_CORRECT_DRAFTS_STRIDE=num_correct_drafts.stride(0),
            CANDIDATE_ROW=candidate_tokens.stride(0) if real_tokens else 0,
            CANDIDATE_COL=candidate_tokens.stride(1) if real_tokens else 0,
            TARGET_ROW=target_tokens.stride(0) if real_tokens else 0,
            TARGET_COL=target_tokens.stride(1) if real_tokens else 0,
            REAL_TOKENS=real_tokens,
            BLOCK=128,
        )
    return output_indices
