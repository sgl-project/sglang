"""Row-tiled ``_fused_qk_rmsnorm_rope_gate_kernel`` for prefill (opt-in, default off).

``SGLANG_FUSED_QK_NORM_ROPE_ROWS=1``: calls with at least
``SGLANG_FUSED_QK_NORM_ROPE_ROWS_MIN_TOKENS`` tokens (default 1024) run one
program per ``ROWS`` (token, head) rows instead of one 128-thread program per
row: a 16K-token prefill launches 27K programs instead of 426K, each with all
its loads in flight at once.

Bitwise equal to the stock kernel as the b72221178d image's Triton 3.9 compiles
it: each row keeps the stock register layout along the head dim (2 elements
per thread, 32 lanes, 4 warps, written out as a Gluon layout), so the RMS sum
reduces in the same order; the RoPE products use the stock compile's FMA
contraction, written out; the gate is a copy. Another Triton may contract the
stock kernel differently: ``..._CHECK_CALLS`` compares on real inputs. Only the
BF16, head_dim 256 / rotary 64, gated shape with 16-byte aligned rows is
covered; anything else keeps the stock kernel.
"""
import functools
import logging
import os
import time

import torch

logger = logging.getLogger(__name__)

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    _HAS_GLUON = True
except ImportError:  # pragma: no cover - older Triton or non-NVIDIA builds
    _HAS_GLUON = False


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


ENABLED = _env_flag("SGLANG_FUSED_QK_NORM_ROPE_ROWS") and _HAS_GLUON
_MIN_TOKENS = int(os.environ.get("SGLANG_FUSED_QK_NORM_ROPE_ROWS_MIN_TOKENS", "1024"))
# Rows per program (VR200 microbench, h449).
_ROWS = int(os.environ.get("SGLANG_FUSED_QK_NORM_ROPE_ROWS_ROWS", "16"))
# The stock kernel's launch: 4 warps over a 256-wide head, 2 elements per thread.
_NUM_WARPS = 4
_HEAD_DIM = 256
_ROTARY_DIM = 64
# Debug: the first N eligible eager calls also run the stock kernel and log time and
# bitwise equality. Leave unset in served runs: it syncs and runs both kernels.
_CHECK_CALLS = int(os.environ.get("SGLANG_FUSED_QK_NORM_ROPE_ROWS_CHECK_CALLS", "0"))
_state = {"checked": 0, "bypass": False}


if _HAS_GLUON:

    @gluon.jit
    def _mul_rn(a, b):
        return gl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=gl.float32, is_pure=True, pack=1
        )

    @gluon.jit
    def _fma_rn(a, b, c):
        return gl.inline_asm_elementwise(
            "fma.rn.f32 $0, $1, $2, $3;", "=r,r,r,r", [a, b, c], dtype=gl.float32, is_pure=True, pack=1
        )

    @gluon.jit
    def _rope_pair(xr1, xr2, cos, sin):
        # The stock kernel's contraction as Triton 3.9 compiles it; explicit, so the
        # compiler cannot pick another for this tile.
        first = _fma_rn(xr1, cos, -_mul_rn(xr2, sin))
        second = _fma_rn(xr1, sin, _mul_rn(xr2, cos))
        return first, second

    @gluon.jit
    def _rows_body(
        in_ptr,
        out_ptr,
        gate_out_ptr,
        w_ptr,
        cos_sin_cache_ptr,
        positions_ptr,
        mrope_axis_map_ptr,
        block,
        n_rows,
        stride_in_t,
        stride_out_t,
        stride_gate_t,
        stride_cos_t,
        stride_pos_axis,
        HEADS: gl.constexpr,
        IN_HEAD_STRIDE: gl.constexpr,
        HEAD_DIM: gl.constexpr,
        ROTARY_DIM: gl.constexpr,
        HALF_ROTARY: gl.constexpr,
        EPS: gl.constexpr,
        HAS_GATE: gl.constexpr,
        MROPE: gl.constexpr,
        ROWS: gl.constexpr,
        NUM_WARPS: gl.constexpr,
        WEIGHT_SHIFT: gl.constexpr,
    ):
        # Along the head dim, the stock 1-D tile's layout: the RMS sum's order.
        norm_l: gl.constexpr = gl.BlockedLayout(
            [1, HEAD_DIM // (32 * NUM_WARPS)], [1, 32], [1, NUM_WARPS], [1, 0]
        )
        rot_l: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [NUM_WARPS, 1], [1, 0])
        copy_l: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [NUM_WARPS, 1], [1, 0])
        out_dtype: gl.constexpr = gl.bfloat16

        # RMSNorm over the whole head, then the pass-through tail [ROTARY_DIM, HEAD_DIM).
        r = block * ROWS + gl.arange(0, ROWS, gl.SliceLayout(1, norm_l))
        valid = r < n_rows
        token = (r // HEADS).to(gl.int64)
        head = (r % HEADS).to(gl.int64)
        in_row = token * stride_in_t + head * IN_HEAD_STRIDE
        out_row = token * stride_out_t + head * HEAD_DIM
        d = gl.arange(0, HEAD_DIM, gl.SliceLayout(0, norm_l))
        mask = valid[:, None] & (d[None, :] < HEAD_DIM)
        x = gl.load(in_ptr + in_row[:, None] + d[None, :], mask=mask, other=0.0).to(
            gl.float32
        )
        w = gl.load(w_ptr + d).to(gl.float32)
        var = gl.sum(x * x, axis=1) / HEAD_DIM
        inv_rms = gl.rsqrt(var + EPS)
        x_norm = (
            (x * inv_rms[:, None] * (w[None, :] + WEIGHT_SHIFT))
            .to(out_dtype)
            .to(gl.float32)
        )
        gl.store(
            out_ptr + out_row[:, None] + d[None, :],
            x_norm.to(out_dtype),
            mask=mask & (d[None, :] >= ROTARY_DIM),
        )

        # The rotary part: re-normalized from the input, as stock, then NeoX RoPE.
        r2 = block * ROWS + gl.arange(0, ROWS, gl.SliceLayout(1, rot_l))
        valid2 = r2 < n_rows
        token2 = (r2 // HEADS).to(gl.int64)
        head2 = (r2 % HEADS).to(gl.int64)
        in_row2 = token2 * stride_in_t + head2 * IN_HEAD_STRIDE
        out_row2 = token2 * stride_out_t + head2 * HEAD_DIM
        j = gl.arange(0, HALF_ROTARY, gl.SliceLayout(0, rot_l))
        rmask = valid2[:, None] & (j[None, :] < HALF_ROTARY)
        inv2 = gl.convert_layout(inv_rms, gl.SliceLayout(1, rot_l))
        xr1 = gl.load(in_ptr + in_row2[:, None] + j[None, :], mask=rmask, other=0.0).to(
            gl.float32
        )
        xr2 = gl.load(
            in_ptr + in_row2[:, None] + HALF_ROTARY + j[None, :], mask=rmask, other=0.0
        ).to(gl.float32)
        wr1 = gl.load(w_ptr + j).to(gl.float32)
        wr2 = gl.load(w_ptr + HALF_ROTARY + j).to(gl.float32)
        xr1 = (
            (xr1 * inv2[:, None] * (wr1[None, :] + WEIGHT_SHIFT))
            .to(out_dtype)
            .to(gl.float32)
        )
        xr2 = (
            (xr2 * inv2[:, None] * (wr2[None, :] + WEIGHT_SHIFT))
            .to(out_dtype)
            .to(gl.float32)
        )
        if MROPE:
            axis = gl.load(mrope_axis_map_ptr + j)
            pos = gl.load(
                positions_ptr + axis[None, :] * stride_pos_axis + token2[:, None],
                mask=rmask,
                other=0,
            )
        else:
            pos1 = gl.load(positions_ptr + token2, mask=valid2, other=0)
            pos = pos1[:, None] + gl.zeros_like(j)[None, :]
        cache_off = pos.to(gl.int64) * stride_cos_t
        cos = gl.load(
            cos_sin_cache_ptr + cache_off + j[None, :], mask=rmask, other=0.0
        ).to(gl.float32)
        sin = gl.load(
            cos_sin_cache_ptr + cache_off + HALF_ROTARY + j[None, :],
            mask=rmask,
            other=0.0,
        ).to(gl.float32)
        first, second = _rope_pair(xr1, xr2, cos, sin)
        gl.store(out_ptr + out_row2[:, None] + j[None, :], first.to(out_dtype), mask=rmask)
        gl.store(
            out_ptr + out_row2[:, None] + HALF_ROTARY + j[None, :],
            second.to(out_dtype),
            mask=rmask,
        )

        # Gate deinterleave (Q rows only): a copy.
        if HAS_GATE:
            r3 = block * ROWS + gl.arange(0, ROWS, gl.SliceLayout(1, copy_l))
            valid3 = r3 < n_rows
            token3 = (r3 // HEADS).to(gl.int64)
            head3 = (r3 % HEADS).to(gl.int64)
            d3 = gl.arange(0, HEAD_DIM, gl.SliceLayout(0, copy_l))
            gmask = valid3[:, None] & (d3[None, :] < HEAD_DIM)
            g = gl.load(
                in_ptr
                + (token3 * stride_in_t + head3 * IN_HEAD_STRIDE + HEAD_DIM)[:, None]
                + d3[None, :],
                mask=gmask,
                other=0.0,
            )
            gl.store(
                gate_out_ptr
                + (token3 * stride_gate_t + head3 * HEAD_DIM)[:, None]
                + d3[None, :],
                g,
                mask=gmask,
            )

    # Row and program counts change with every prefill size; unspecialized, they never recompile.
    @gluon.jit(do_not_specialize=["q_rows", "k_rows", "q_programs"])
    def _fused_qk_rmsnorm_rope_gate_rows_kernel(
        q_gate_ptr,
        k_ptr,
        q_out_ptr,
        k_out_ptr,
        gate_out_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_sin_cache_ptr,
        positions_ptr,
        mrope_axis_map_ptr,
        q_rows,
        k_rows,
        q_programs,
        stride_qg_t,
        stride_k_t,
        stride_qo_t,
        stride_ko_t,
        stride_gate_t,
        stride_cos_t,
        stride_pos_axis,
        NUM_Q_HEADS: gl.constexpr,
        NUM_KV_HEADS: gl.constexpr,
        HEAD_DIM: gl.constexpr,
        ROTARY_DIM: gl.constexpr,
        HALF_ROTARY: gl.constexpr,
        EPS: gl.constexpr,
        MROPE: gl.constexpr,
        ROWS: gl.constexpr,
        NUM_WARPS: gl.constexpr,
        WEIGHT_SHIFT: gl.constexpr = 1.0,
    ):
        # Programs [0, q_programs) take Q heads (with the gate), the rest K heads.
        pid = gl.program_id(0)
        if pid < q_programs:
            _rows_body(
                q_gate_ptr,
                q_out_ptr,
                gate_out_ptr,
                q_weight_ptr,
                cos_sin_cache_ptr,
                positions_ptr,
                mrope_axis_map_ptr,
                pid,
                q_rows,
                stride_qg_t,
                stride_qo_t,
                stride_gate_t,
                stride_cos_t,
                stride_pos_axis,
                NUM_Q_HEADS,
                2 * HEAD_DIM,
                HEAD_DIM,
                ROTARY_DIM,
                HALF_ROTARY,
                EPS,
                True,
                MROPE,
                ROWS,
                NUM_WARPS,
                WEIGHT_SHIFT,
            )
        else:
            _rows_body(
                k_ptr,
                k_out_ptr,
                gate_out_ptr,
                k_weight_ptr,
                cos_sin_cache_ptr,
                positions_ptr,
                mrope_axis_map_ptr,
                pid - q_programs,
                k_rows,
                stride_k_t,
                stride_ko_t,
                stride_gate_t,
                stride_cos_t,
                stride_pos_axis,
                NUM_KV_HEADS,
                HEAD_DIM,
                HEAD_DIM,
                ROTARY_DIM,
                HALF_ROTARY,
                EPS,
                False,
                MROPE,
                ROWS,
                NUM_WARPS,
                WEIGHT_SHIFT,
            )


def _aligned(t: torch.Tensor) -> bool:
    # 16-byte rows: the stock kernel's vectorized-load specialization.
    return t.data_ptr() % 16 == 0 and t.stride(0) % 8 == 0 and t.stride(-1) == 1


def applies(q_gate, k, positions, num_tokens, head_dim, rotary_dim, has_gate) -> bool:
    return (
        ENABLED
        and not _state["bypass"]
        and num_tokens >= _MIN_TOKENS
        and has_gate
        and (head_dim, rotary_dim) == (_HEAD_DIM, _ROTARY_DIM)
        and q_gate.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and q_gate.is_cuda
        and _aligned(q_gate)
        and _aligned(k)
        and positions.dtype in (torch.int32, torch.int64)
    )


@functools.lru_cache(maxsize=None)
def _log_once(num_q_heads: int, num_kv_heads: int, mrope: bool) -> None:
    logger.info(
        "Fused QK norm rope rows: on (rows=%d min_tokens=%d heads=%d/%d mrope=%s "
        "check_calls=%d) (SGLANG_FUSED_QK_NORM_ROPE_ROWS)",
        _ROWS,
        _MIN_TOKENS,
        num_q_heads,
        num_kv_heads,
        mrope,
        _CHECK_CALLS,
    )


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def checked(run_rows, run_stock, num_tokens: int):
    """Serve ``run_rows``; the first ``_CHECK_CALLS`` eager calls also time
    ``run_stock`` (the stock kernel, with this path bypassed) and log both."""
    if _state["checked"] >= _CHECK_CALLS or torch.cuda.is_current_stream_capturing():
        return run_rows()
    _state["checked"] += 1
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    new = run_rows()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    _state["bypass"] = True
    try:
        ref = run_stock()
    finally:
        _state["bypass"] = False
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    logger.info(
        "Fused QK norm rope rows check %d/%d: tokens=%d rows_ms=%.3f stock_ms=%.3f "
        "bitwise=%s",
        _state["checked"],
        _CHECK_CALLS,
        num_tokens,
        (t1 - t0) * 1e3,
        (t2 - t1) * 1e3,
        all(_same_bits(a, b) for a, b in zip(new, ref)),
    )
    return new


def launch(
    *,
    q_gate,
    k,
    q_out,
    k_out,
    gate_out,
    q_weight,
    k_weight,
    cos_sin_cache,
    positions,
    mrope_axis_map,
    eps,
    num_q_heads,
    num_kv_heads,
):
    """Fill ``q_out``, ``k_out`` and ``gate_out`` as the stock kernel does."""
    num_tokens = q_gate.shape[0]
    mrope = positions.dim() == 2
    q_rows = num_tokens * num_q_heads
    k_rows = num_tokens * num_kv_heads
    q_programs = -(-q_rows // _ROWS)
    grid = (q_programs + -(-k_rows // _ROWS),)
    _log_once(num_q_heads, num_kv_heads, mrope)
    _fused_qk_rmsnorm_rope_gate_rows_kernel[grid](
        q_gate,
        k,
        q_out,
        k_out,
        gate_out,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        mrope_axis_map if mrope else positions,
        q_rows,
        k_rows,
        q_programs,
        q_gate.stride(0),
        k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        gate_out.stride(0),
        cos_sin_cache.stride(0),
        positions.stride(0),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=_HEAD_DIM,
        ROTARY_DIM=_ROTARY_DIM,
        HALF_ROTARY=_ROTARY_DIM // 2,
        EPS=eps,
        MROPE=mrope,
        ROWS=_ROWS,
        NUM_WARPS=_NUM_WARPS,
        num_warps=_NUM_WARPS,
    )
