# Adapted from OpenAI-Partners/artemis-kernel-integrations PR #11.
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _load_bf16x4(P, offsets, OUT: gl.constexpr):
    words_ptr = P.to(gl.pointer_type(gl.uint64))
    wide = gl.amd.cdna4.buffer_load(words_ptr, offsets)
    low = wide.to(gl.uint32)
    high = (wide >> 32).to(gl.uint32)
    if len(offsets.shape) == 2:
        words = gl.join(low, high).reshape((offsets.shape[0], offsets.shape[1] * 2))
    else:
        words = gl.join(low, high).reshape(
            (offsets.shape[0], offsets.shape[1], offsets.shape[2] * 2)
        )
    lo = words.to(gl.uint16).to(gl.bfloat16, bitcast=True)
    hi = (words >> 16).to(gl.uint16).to(gl.bfloat16, bitcast=True)
    if len(offsets.shape) == 2:
        values = gl.join(lo, hi).reshape((offsets.shape[0], offsets.shape[1] * 4))
    else:
        values = gl.join(lo, hi).reshape(
            (offsets.shape[0], offsets.shape[1], offsets.shape[2] * 4)
        )
    return gl.convert_layout(values, OUT)


@gluon.jit
def _small_gemm(X, W, Y, M: gl.constexpr):
    UNROLL: gl.constexpr = 4 if M == 8 else 8
    SHIFT: gl.constexpr = 1
    TRANSPOSED: gl.constexpr = M >= 4
    XLOAD: gl.constexpr = 0 if M == 8 else 2
    WLOAD: gl.constexpr = 2 if M == 2 or M == 4 else 0
    SPLITS: gl.constexpr = 2
    BK: gl.constexpr = 128
    STEPS: gl.constexpr = 16
    ADVANCE: gl.constexpr = 256
    load_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 8], [1, 4, 16], [1, 1, 1], [2, 1, 0]
    )
    mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=TRANSPOSED,
        warps_per_cta=[1, 1, 1],
    )
    split = gl.arange(0, SPLITS, gl.SliceLayout(1, gl.SliceLayout(2, load_layout)))
    rows = gl.arange(0, M, gl.SliceLayout(0, gl.SliceLayout(2, load_layout)))
    cols = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(2, load_layout)))
    kk = gl.arange(0, BK, gl.SliceLayout(0, gl.SliceLayout(1, load_layout)))
    k = split[:, None, None] * BK + kk[None, None, :]
    x_offsets = rows[None, :, None] * 4096 + k
    w_offsets = cols[None, :, None] * 4096 + k
    if M <= 4:
        word_layout: gl.constexpr = gl.BlockedLayout(
            [1, 1, 2], [1, 4, 16], [1, 1, 1], [2, 1, 0]
        )
        ws = gl.arange(0, 2, gl.SliceLayout(1, gl.SliceLayout(2, word_layout)))
        wr = gl.arange(0, M, gl.SliceLayout(0, gl.SliceLayout(2, word_layout)))
        wn = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(2, word_layout)))
        wk = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(1, word_layout)))
        word_k = ws[:, None, None] * 32 + wk[None, None, :]
        x_words = wr[None, :, None] * 1024 + word_k
        w_words = wn[None, :, None] * 1024 + word_k
    tile = gl.program_id(0)
    gl.assume(tile >= 0)
    gl.assume(tile < 384)
    w_base = W + tile * 16 * 4096
    if M == 8:
        phase = (tile >> SHIFT) * 9 & STEPS - 1
    else:
        phase = tile >> SHIFT & STEPS - 1
    acc = gl.zeros((SPLITS, M, 16), gl.float32, mma_layout)
    if XLOAD == 0:
        x = gl.load(X + x_offsets + phase * ADVANCE)
    else:
        x = _load_bf16x4(X + phase * ADVANCE, x_words, load_layout)
    if WLOAD == 0:
        w = gl.load(w_base + w_offsets + phase * ADVANCE)
    else:
        w = _load_bf16x4(w_base + phase * ADVANCE, w_words, load_layout)
    for group in range(STEPS // UNROLL):
        for inner in gl.static_range(UNROLL):
            step = group * UNROLL + inner
            next_panel = step + 1 & STEPS - 1 ^ phase
            current_x = x
            current_w = w
            if XLOAD == 0:
                x = gl.load(X + x_offsets + next_panel * ADVANCE)
            else:
                x = _load_bf16x4(X + next_panel * ADVANCE, x_words, load_layout)
            if WLOAD == 0:
                w = gl.load(w_base + w_offsets + next_panel * ADVANCE)
            else:
                w = _load_bf16x4(w_base + next_panel * ADVANCE, w_words, load_layout)
            a = gl.convert_layout(current_x, gl.DotOperandLayout(0, mma_layout, 8))
            b = gl.convert_layout(
                current_w.permute(0, 2, 1), gl.DotOperandLayout(1, mma_layout, 8)
            )
            acc = gl.amd.cdna4.mfma(a, b, acc)
    result_layout: gl.constexpr = gl.SliceLayout(0, mma_layout)
    result = gl.sum(acc, 0)
    out_m = gl.arange(0, M, gl.SliceLayout(1, result_layout))
    out_n = tile * 16 + gl.arange(0, 16, gl.SliceLayout(0, result_layout))
    if M <= 4:
        out_ptr = Y + (out_m[:, None] * 6144 + out_n[None, :])
    else:
        out_ptr = Y + out_m[:, None] * 6144 + out_n[None, :]
    gl.store(out_ptr, result.to(gl.bfloat16))


def bf16_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Exact M=4, N=6144, K=4096 target O projection."""
    assert x.shape == (4, 4096) and weight.shape == (6144, 4096)
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_contiguous() and weight.is_contiguous()
    out = torch.empty((4, 6144), dtype=torch.bfloat16, device=x.device)
    _small_gemm[384,](x, weight, out, 4, num_warps=1)
    return out
