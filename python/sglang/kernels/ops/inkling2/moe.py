import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit

PDL = is_arch_support_pdl()
# Below this many rows the Triton _mxfp8_quantize is as fast.
MXFP8_CUDA_MIN_ROWS = 64


@triton.jit
def _mxfp8_block(groups):
    amax = tl.max(tl.abs(groups), 1)
    raw_scale = amax * (1.0 / 448.0)
    min_scale = tl.full((), 2.0**-127, tl.float32)
    exponent = ((raw_scale.to(tl.int32, bitcast=True) + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = tl.where(raw_scale <= min_scale, 0, tl.minimum(exponent, 254))
    exponent = tl.where(raw_scale != raw_scale, 255, exponent)
    scale = tl.maximum((exponent << 23).to(tl.float32, bitcast=True), min_scale)
    scale = tl.where(exponent == 255, float("nan"), scale)
    inverse = tl.where(exponent == 0, 2.0**127, libdevice.rcp_rn(scale))
    inverse = tl.where(exponent == 255, 0.0, inverse)
    values = groups * inverse[:, None]
    values = tl.minimum(
        tl.maximum(values, -448.0), 448.0, propagate_nan=tl.PropagateNan.ALL
    )
    return values.to(tl.float8e4nv), exponent


@triton.jit
def _mxfp8_quantize(X, Y, S, N: tl.constexpr, BLOCK: tl.constexpr):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * N + cols, cols < N, other=0).to(tl.float32)
    values, exponent = _mxfp8_block(tl.reshape(x, (BLOCK // 32, 32)))
    groups = tl.program_id(1) * (BLOCK // 32) + tl.arange(0, BLOCK // 32)
    tl.store(S + row * (N // 32) + groups, exponent, groups < N // 32)
    tl.store(Y + row * N + cols, tl.reshape(values, (BLOCK,)), cols < N)


def mxfp8_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim not in (2, 3) or not x.is_contiguous() or x.shape[-1] % 32:
        raise ValueError("MXFP8 tensors must be contiguous with width divisible by 32")
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] < 10:
        raise ValueError("Native MXFP8 expert computation requires a Blackwell GPU")
    values = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (*x.shape[:-1], x.shape[-1] // 32), dtype=torch.uint8, device=x.device
    )
    if (
        x.numel() >= MXFP8_CUDA_MIN_ROWS * x.shape[-1]
        and x.dtype == torch.bfloat16
        and x.shape[-1] % (32 * 3 * 16) == 0
    ):
        from sglang.kernels.ops.inkling2.nvfp4 import routed_quant_module

        routed_quant_module(x.shape[-1]).mxfp8_quantize(
            x, values.view(torch.uint8), scales, _sm_count(x.device)
        )
    elif x.numel():
        _mxfp8_quantize[(x.numel() // x.shape[-1], triton.cdiv(x.shape[-1], 256))](
            x, values, scales, x.shape[-1], 256, enable_fp_fusion=False, launch_pdl=PDL
        )
    return values, scales


@triton.jit
def _swiglu(
    X,
    G,
    Y,
    N: tl.constexpr,
    SIZE: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(X + 2 * i, i < SIZE, other=0).to(tl.float32)
    up = tl.load(X + 2 * i + 1, i < SIZE, other=0).to(tl.float32)
    gamma = tl.load(G + i // N, i < SIZE, other=0)
    tl.store(Y + i, _capped_swiglu(gate, up, gamma, GATE_CAP, UP_CAP), i < SIZE)


@triton.jit
def _capped_swiglu(gate, up, gamma, GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr):
    sigmoid = tl.inline_asm_elementwise(
        "{ .reg .f32 t; mul.rn.f32 t, $1, 0fBFB8AA3B; "
        "ex2.approx.ftz.f32 t, t; add.rn.f32 t, t, 0f3F800000; "
        "rcp.approx.ftz.f32 $0, t; }",
        "=f,f",
        [gate],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    if GATE_CAP > 0:
        gate = GATE_CAP * tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            [tl.div_rn(gate, GATE_CAP)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    if UP_CAP > 0:
        up = UP_CAP * tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            [tl.div_rn(up, UP_CAP)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    return ((gate * sigmoid) * up) * gamma


def swiglu(x: torch.Tensor, gammas: torch.Tensor, gate_cap: float, up_cap: float):
    out = x.new_empty(x.shape[0], x.shape[1] // 2)
    if out.numel():
        _swiglu[(triton.cdiv(out.numel(), 256),)](
            x,
            gammas,
            out,
            out.shape[1],
            out.numel(),
            gate_cap,
            up_cap,
            256,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _row_expert(Offsets, row, E: tl.constexpr, SEARCH: tl.constexpr):
    experts = tl.arange(0, 1 << SEARCH)
    starts = tl.load(Offsets + 1 + experts, experts < E - 1, other=0x7FFFFFFF)
    return tl.sum((starts <= row).to(tl.int32), 0)


@triton.jit
def _nvfp4_row_scale(XAmax, WAmax, WDenominator, expert, row):
    denominator = 1536.0 * tl.load(WDenominator)
    return (tl.load(XAmax + row) * tl.load(WAmax + expert)) * libdevice.rcp_rn(
        denominator
    )


@triton.jit
def _routed_swiglu(
    X,
    XAmax,
    WAmax,
    WDenominator,
    Offsets,
    G,
    Y,
    N: tl.constexpr,
    E: tl.constexpr,
    SCALED: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
    BLOCK: tl.constexpr,
    SEARCH: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = (row >= tl.load(Offsets)) & (row < tl.load(Offsets + E))
    mask = (cols < N) & inside
    gate = tl.load(X + row.to(tl.int64) * 2 * N + 2 * cols, mask, other=0)
    up = tl.load(X + row.to(tl.int64) * 2 * N + 2 * cols + 1, mask, other=0)
    if SCALED:
        scale = _nvfp4_row_scale(
            XAmax, WAmax, WDenominator, _row_expert(Offsets, row, E, SEARCH), row
        )
        gate = tl.where(mask, (gate * scale).to(Y.dtype.element_ty), 0)
        up = tl.where(mask, (up * scale).to(Y.dtype.element_ty), 0)
    gamma = tl.load(G + row)
    y = _capped_swiglu(gate.to(tl.float32), up.to(tl.float32), gamma, GATE_CAP, UP_CAP)
    tl.store(Y + row.to(tl.int64) * N + cols, y, cols < N)


def routed_swiglu(
    x, gammas, offsets, gate_cap, up_cap, *, out_dtype=None, nvfp4_scales=None
):
    """swiglu over rows in [offsets[0], offsets[-1]); other rows act as zero input.

    With nvfp4_scales=(x_amax, w_amax, w_denominator), x is the raw FP32 GEMM
    accumulator and each row is first scaled and rounded like the GEMM output."""
    experts = offsets.numel() - 1
    out = torch.empty(
        (x.shape[0], x.shape[1] // 2), device=x.device, dtype=out_dtype or x.dtype
    )
    if out.numel():
        x_amax, w_amax, w_denominator = nvfp4_scales or (None, None, None)
        _routed_swiglu[(out.shape[0], triton.cdiv(out.shape[1], 512))](
            x,
            x_amax,
            w_amax,
            w_denominator,
            offsets,
            gammas,
            out,
            out.shape[1],
            experts,
            nvfp4_scales is not None,
            gate_cap,
            up_cap,
            512,
            experts.bit_length(),
            enable_fp_fusion=False,
            launch_pdl=PDL,
        )
    return out


@triton.jit
def fence_system(THREADS: tl.constexpr):
    """fence.sc.sys once per thread: its earlier stores, including NVLink stores
    to peers, are visible system-wide before any later signal."""
    tl.inline_asm_elementwise(
        "fence.sc.sys; mov.b32 $0, $1;",
        "=r,r",
        [tl.full([THREADS], 0, tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _routed_rows(
    X,
    XAmax,
    WAmax,
    WDenominator,
    Offsets,
    Y,
    N: tl.constexpr,
    E: tl.constexpr,
    SCALED: tl.constexpr,
    BLOCK: tl.constexpr,
    SEARCH: tl.constexpr,
    Calls=None,
    slot_size=0,
    OWNED_ONLY: tl.constexpr = False,
    STRIDED: tl.constexpr = False,
    Dst2Src=None,
    TOP_K: tl.constexpr = 1,
    per_rank=1,
    THREADS: tl.constexpr = 128,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    if STRIDED:
        chunks: tl.constexpr = (N + BLOCK - 1) // BLOCK
        lo = tl.load(Offsets)
        for item in range(
            tl.program_id(0), (tl.load(Offsets + E) - lo) * chunks, tl.num_programs(0)
        ):
            row = lo + item // chunks
            cols = (item % chunks) * BLOCK + tl.arange(0, BLOCK)
            x = tl.load(X + row.to(tl.int64) * N + cols, cols < N)
            if SCALED:
                x = x * _nvfp4_row_scale(
                    XAmax,
                    WAmax,
                    WDenominator,
                    _row_expert(Offsets, row, E, SEARCH),
                    row,
                )
            combiner = tl.load(Dst2Src + row) // TOP_K // per_rank
            out = tl.load(Y + combiner).to(tl.pointer_type(tl.bfloat16))
            tl.store(out + row.to(tl.int64) * N + cols, x, cols < N)
        fence_system(THREADS)
        return
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = (row >= tl.load(Offsets)) & (row < tl.load(Offsets + E))
    if OWNED_ONLY:
        if not inside:
            return
        if Calls is not None:
            Y += (tl.load(Calls) & 1).to(tl.int64) * slot_size
    mask = (cols < N) & inside
    x = tl.load(X + row.to(tl.int64) * N + cols, mask, other=0)
    if SCALED:
        scale = _nvfp4_row_scale(
            XAmax, WAmax, WDenominator, _row_expert(Offsets, row, E, SEARCH), row
        )
        x = x * scale
    tl.store(Y + row.to(tl.int64) * N + cols, tl.where(mask, x, 0), cols < N)


@cache_once
def _routed_rows_module(hidden, dtype, scaled, owned):
    cuda_type = "float" if dtype == torch.float32 else "__nv_bfloat16"
    flags = f"{str(scaled).lower()}, {str(owned).lower()}"
    return load_jit(
        "inkling2_routed_rows",
        str(hidden),
        str(dtype).split(".")[-1],
        str(scaled),
        str(owned),
        cuda_files=["inkling2/routed_rows.cuh"],
        cuda_wrappers=[
            ("run", f"inkling2_routed_rows::run<{hidden}, {cuda_type}, {flags}>")
        ],
    )


def _cuda_routed_rows(x, offsets, out, nvfp4_scales, exchange):
    """_routed_rows (not the prefill push) in CUDA; False when unsupported."""
    experts = offsets.numel() - 1
    if not (
        out.dtype == torch.bfloat16
        and x.dtype == (torch.float32 if nvfp4_scales is not None else torch.bfloat16)
        and x.is_contiguous()
        and x.shape[1] % 8 == 0
        and x.shape[1] // 8 <= 1024
        and experts <= 256
    ):
        return False
    owned_only = exchange is not None
    x_amax, w_amax, w_denominator = nvfp4_scales or (None, None, None)
    ptr = lambda t: 0 if t is None else t.data_ptr()  # noqa: E731
    _routed_rows_module(x.shape[1], x.dtype, nvfp4_scales is not None, owned_only).run(
        x,
        ptr(x_amax),
        ptr(w_amax),
        ptr(w_denominator),
        offsets,
        out.data_ptr(),
        ptr(getattr(exchange, "calls", None)),
        exchange.max_rows * exchange.hidden if owned_only else 0,
        min(x.shape[0], 2 * _sm_count(x.device)),
    )
    return True


def routed_rows(
    x, offsets, *, out_dtype=None, nvfp4_scales=None, exchange=None, dst2src=None
):
    """Rows in [offsets[0], offsets[-1]) of x (scaled like the NVFP4 GEMM output
    when nvfp4_scales is given); all other rows zero. With an exchange, only
    those rows are written, into its buffer (current slot when it alternates
    slots), and nothing is returned. With a PrefillExchange (dst2src given),
    each row goes to the row buffer of the rank combining its token."""
    experts = offsets.numel() - 1
    owned_only = exchange is not None
    push = dst2src is not None
    out = (
        exchange.buffer[: x.numel()].view(x.shape)
        if owned_only
        else torch.empty(x.shape, device=x.device, dtype=out_dtype or x.dtype)
    )
    if (
        out.numel()
        and not push
        and _cuda_routed_rows(x, offsets, out, nvfp4_scales, exchange)
    ):
        return None if owned_only else out
    if out.numel():
        x_amax, w_amax, w_denominator = nvfp4_scales or (None, None, None)
        chunks = triton.cdiv(out.shape[1], 1024)
        grid = (
            (min(out.shape[0] * chunks, 8 * _sm_count(x.device)),)
            if push
            else (out.shape[0], chunks)
        )
        _routed_rows[grid](
            x,
            x_amax,
            w_amax,
            w_denominator,
            offsets,
            exchange.row_ptrs if push else out,
            out.shape[1],
            experts,
            nvfp4_scales is not None,
            1024,
            experts.bit_length(),
            None if push else getattr(exchange, "calls", None),
            exchange.max_rows * exchange.hidden if owned_only else 0,
            owned_only,
            push,
            dst2src,
            exchange.top_k if push else 1,
            exchange.per_rank(x.shape[0] // exchange.top_k) if push else 1,
            enable_fp_fusion=False,
            launch_pdl=PDL,
        )
    return None if owned_only else out


def _sm_count(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


@triton.jit
def _unweighted_post_reorder(
    X, Y, Src2Dst, TOP_K: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    src = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H
    total = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        dst = tl.load(Src2Dst + src * TOP_K + k).to(tl.int64)
        total += tl.load(X + dst * H + cols, mask) * 1.0
    tl.store(Y + src * H + cols, total, mask)


@triton.jit
def _routed_partial(
    X,
    XAmax,
    WAmax,
    WDenominator,
    Offsets,
    Src2Dst,
    Y,
    N: tl.constexpr,
    E: tl.constexpr,
    TOP_K: tl.constexpr,
    SCALED: tl.constexpr,
    BLOCK: tl.constexpr,
    SEARCH: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    src = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    lo = tl.load(Offsets)
    hi = tl.load(Offsets + E)
    total = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        row = tl.load(Src2Dst + src * TOP_K + k)
        inside = (row >= lo) & (row < hi)
        x = tl.load(X + row.to(tl.int64) * N + cols, (cols < N) & inside, other=0)
        if SCALED:
            x = x * _nvfp4_row_scale(
                XAmax, WAmax, WDenominator, _row_expert(Offsets, row, E, SEARCH), row
            )
        total += tl.where(inside, x.to(tl.bfloat16).to(tl.float32), 0.0)
    tl.store(Y + src * N + cols, total, cols < N)


def routed_partial(x, offsets, src2dst, top_k, *, nvfp4_scales=None):
    """Per token, the FP32 sum in top-k order of its rows in [offsets[0],
    offsets[-1]), each rounded to bfloat16 (scaled like the NVFP4 GEMM output
    when nvfp4_scales is given)."""
    experts = offsets.numel() - 1
    out = torch.empty(
        (src2dst.numel() // top_k, x.shape[1]), device=x.device, dtype=torch.float32
    )
    if out.numel():
        x_amax, w_amax, w_denominator = nvfp4_scales or (None, None, None)
        _routed_partial[(out.shape[0], triton.cdiv(out.shape[1], 1024))](
            x,
            x_amax,
            w_amax,
            w_denominator,
            offsets,
            src2dst,
            out,
            out.shape[1],
            experts,
            top_k,
            nvfp4_scales is not None,
            1024,
            experts.bit_length(),
            launch_pdl=PDL,
        )
    return out


def unweighted_post_reorder(x, src2dst, top_k):
    """post_reorder with unit routing weights."""
    out = torch.empty(
        (src2dst.numel() // top_k, x.shape[1]), device=x.device, dtype=x.dtype
    )
    if out.numel():
        _unweighted_post_reorder[(out.shape[0], triton.cdiv(out.shape[1], 1024))](
            x, out, src2dst, top_k, out.shape[1], 1024, launch_pdl=PDL
        )
    return out
