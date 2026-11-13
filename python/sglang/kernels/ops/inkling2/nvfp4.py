import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit
from sglang.kernels.ops.inkling2.cute_grouped_nvfp4 import routed_swap_matmul

PDL = is_arch_support_pdl()
SWAP_AB_MAX_ROWS = 20480


@triton.jit
def _pack_roundtrip(x):
    low, high = tl.split(x)
    packed, halves = tl.inline_asm_elementwise(
        "{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, $3, $2; "
        "cvt.u32.u8 $0, b; cvt.rn.f16x2.e2m1x2 $1, b; }",
        "=r,=r,f,f",
        [low, high],
        dtype=(tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )
    low = (halves & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    high = (halves >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
    return packed.to(tl.uint8), tl.join(low, high).to(tl.float32)


@triton.jit
def _block_error(x, reconstructed):
    groups: tl.constexpr = x.shape[0]
    error = tl.abs(reconstructed - x)
    for width in tl.static_range(3, -1, -1):
        pairs = tl.reshape(error, (groups, 1 << width, 2))
        low, high = tl.split(pairs)
        error = low + high
    return tl.reshape(error, (groups,))


@triton.jit
def _row_amax(X, Amax, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + cols, cols < K, other=0).to(tl.float32)
    tl.store(Amax + row, tl.max(tl.abs(x), 0))


@triton.jit
def _quantize_rows(
    X,
    Values,
    Scales,
    Amax,
    K: tl.constexpr,
    AMAX_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    cols = tl.arange(0, AMAX_BLOCK)
    x = tl.load(X + row * K + cols, cols < K, other=0).to(tl.float32)
    amax = tl.max(tl.abs(x), 0)
    if chunk == 0:
        tl.store(Amax + row, amax)
    _quantize_chunk(X, Values, Scales, amax, row, chunk, K, BLOCK)


@triton.jit
def _quantize(X, Values, Scales, Amax, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    _quantize_chunk(
        X, Values, Scales, tl.load(Amax + row), row, tl.program_id(1), K, BLOCK
    )


@triton.jit
def _quantize_chunk(
    X, Values, Scales, amax, row, chunk, K: tl.constexpr, BLOCK: tl.constexpr
):
    packed, scales = _quantize_values(X, amax, row, chunk, K, BLOCK)
    packed_cols = chunk * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
    scale_cols = chunk * (BLOCK // 16) + tl.arange(0, BLOCK // 16)
    tl.store(Values + row * (K // 2) + packed_cols, packed, packed_cols < K // 2)
    tl.store(Scales + row * (K // 16) + scale_cols, scales, scale_cols < K // 16)


@triton.jit
def _quantize_values(X, amax, row, chunk, K: tl.constexpr, BLOCK: tl.constexpr):
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * K + cols, cols < K, other=0).to(tl.float32)
    x = tl.reshape(x, (BLOCK // 16, 16))
    encoder_scale = tl.minimum(tl.div_rn(1536.0, amax), 3.4028234663852886e38)
    decoder_scale = tl.div_rn(1.0, encoder_scale)
    outer_scale = amax * (1.0 / 1536.0)
    block_max = tl.max(tl.abs(x), 1)
    scale6 = tl.minimum(block_max * (encoder_scale * (1.0 / 6.0)), 448.0)
    scale4 = tl.minimum(block_max * (encoder_scale * 0.25), 448.0)
    scale6 = scale6.to(tl.float8e4nv).to(tl.float32)
    scale4 = scale4.to(tl.float8e4nv).to(tl.float32)
    inverse6 = tl.minimum(tl.div_rn(1.0, scale6 * decoder_scale), 3.4028234663852886e38)
    inverse4 = tl.minimum(tl.div_rn(1.0, scale4 * decoder_scale), 3.4028234663852886e38)
    packed6, rounded6 = _pack_roundtrip(
        tl.reshape(x * inverse6[:, None], (BLOCK // 16, 8, 2))
    )
    packed4, rounded4 = _pack_roundtrip(
        tl.reshape(x * inverse4[:, None], (BLOCK // 16, 8, 2))
    )
    reconstructed6 = tl.reshape(rounded6, (BLOCK // 16, 16)) * scale6[:, None]
    reconstructed4 = tl.reshape(rounded4, (BLOCK // 16, 16)) * scale4[:, None]
    error6 = _block_error(x, reconstructed6 * outer_scale)
    error4 = _block_error(x, reconstructed4 * outer_scale)
    choose4 = error4 < error6
    packed = tl.reshape(tl.where(choose4[:, None], packed4, packed6), (BLOCK // 2,))
    scales = tl.where(choose4, scale4, scale6).to(tl.float8e4nv)
    return packed, scales


def nvfp4_quantize(x: torch.Tensor):
    if x.ndim not in (2, 3) or not x.is_contiguous() or x.shape[-1] % 16:
        raise ValueError("NVFP4 tensors must be contiguous with width divisible by 16")
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] < 10:
        raise ValueError("Native NVFP4 expert computation requires a Blackwell GPU")
    width = x.shape[-1]
    values = torch.empty(
        (*x.shape[:-1], width // 2), dtype=torch.uint8, device=x.device
    )
    scales = torch.empty(
        (*x.shape[:-1], width // 16), dtype=torch.float8_e4m3fn, device=x.device
    )
    amax = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
    if x.numel():
        rows = x.numel() // width
        if rows * width <= 256 * 2304:
            _quantize_rows[(rows, triton.cdiv(width, 256))](
                x,
                values,
                scales,
                amax,
                width,
                triton.next_power_of_2(width),
                256,
                enable_fp_fusion=False,
                launch_pdl=PDL,
            )
        else:
            _row_amax[(rows,)](x, amax, width, triton.next_power_of_2(width))
            _quantize[(rows, triton.cdiv(width, 256))](
                x, values, scales, amax, width, 256, enable_fp_fusion=False
            )
    return values, scales, amax


@cache_once
def _matmul_module():
    return load_jit(
        "inkling2_nvfp4",
        cuda_files=["inkling2/nvfp4.cuh"],
        cuda_wrappers=[
            ("run", "inkling2_nvfp4::run"),
            ("run_swap", "inkling2_nvfp4::run_swap"),
            ("run_wide_preswizzled", "inkling2_nvfp4::run_wide_preswizzled"),
            ("wide_x_scales_offset", "inkling2_nvfp4::wide_x_scales_offset"),
            ("wide_workspace_size", "inkling2_nvfp4::wide_workspace_size"),
            ("workspace_size", "inkling2_nvfp4::workspace_size"),
            ("swap_workspace_size", "inkling2_nvfp4::swap_workspace_size"),
            ("swizzle_weight_scales", "inkling2_nvfp4::swizzle_weight_scales"),
        ],
        extra_cuda_cflags=["-DCUTLASS_ENABLE_GDC_FOR_SM100=1"],
        extra_dependencies=["cutlass"],
    )


@cache_once
def routed_quant_module(width: int):
    return load_jit(
        "inkling2_routed_quant",
        str(width),
        cuda_files=["inkling2/routed_quant.cuh"],
        cuda_wrappers=[
            ("quantize", f"inkling2_routed_quant::quantize<{width}>"),
            ("swiglu_quantize", f"inkling2_routed_quant::swiglu_quantize<{width}>"),
            (
                "mxfp8_swiglu_quantize",
                f"inkling2_routed_quant::mxfp8_swiglu_quantize<{width}>",
            ),
            ("mxfp8_quantize", f"inkling2_routed_quant::mxfp8_quantize<{width}>"),
        ],
    )


def swizzled_scale_size(output_size: int, width: int) -> int:
    return triton.cdiv(output_size, 128) * 128 * triton.cdiv(width, 64) * 4


def swizzle_weight_scales(scales: torch.Tensor) -> torch.Tensor:
    """Convert (experts, n, k // 16) block scales into the GEMM's scale layout."""
    if (
        scales.ndim != 3
        or scales.dtype != torch.float8_e4m3fn
        or not scales.is_contiguous()
        or not scales.is_cuda
    ):
        raise ValueError(
            f"expected contiguous CUDA float8_e4m3fn scales of shape (experts, n, k // 16), "
            f"got {tuple(scales.shape)} {scales.dtype}"
        )
    experts, output_size, scale_width = scales.shape
    out = torch.zeros(
        (experts, swizzled_scale_size(output_size, scale_width * 16)),
        dtype=scales.dtype,
        device=scales.device,
    )
    if out.numel():
        _matmul_module().swizzle_weight_scales(scales, out)
    return out


@triton.jit
def _scale_output(
    Acc,
    XAmax,
    WAmax,
    WDenominator,
    Y,
    Counts,
    Offsets,
    Schedule,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if Schedule is None:
        expert = tl.program_id(0).to(tl.int64)
        rows = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        row_ids = expert * M + rows
        row_mask = rows < M
    else:
        entry = tl.load(Schedule + tl.program_id(0))
        if entry == -1:
            return
        expert = (entry & 0xFFFF).to(tl.int64)
        rows = (entry >> 16) * BLOCK + tl.arange(0, BLOCK)
        row_ids = tl.load(Offsets + expert).to(tl.int64) + rows
        row_mask = rows < tl.load(Counts + expert)
    cols = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.load(
        Acc + row_ids[:, None] * N + cols[None, :],
        row_mask[:, None] & (cols[None, :] < N),
        other=0.0,
    )
    x_amax = tl.load(XAmax + row_ids, row_mask, other=0)
    w_amax = tl.load(WAmax + expert)
    denominator = 1536.0 * tl.load(WDenominator)
    scale = (x_amax * w_amax) * libdevice.rcp_rn(denominator)
    tl.store(
        Y + row_ids[:, None] * N + cols[None, :],
        acc * scale[:, None],
        row_mask[:, None] & (cols[None, :] < N),
    )


def expert_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    amax: torch.Tensor,
    outer_scale_denominator: torch.Tensor,
    *,
    metadata=None,
    raw=False,
):
    """With raw=True (grouped inputs only), return the unscaled FP32 accumulator and
    the per-row activation amax instead of the scaled output."""
    experts, output_size, packed_width = weight.shape
    width = packed_width * 2
    if metadata is None:
        if x.ndim == 2:
            x = x.unsqueeze(0).expand(experts, -1, -1)
        if x.ndim != 3 or x.shape[0] != experts or x.shape[-1] != width:
            raise ValueError("NVFP4 expert input and weight shapes do not match")
        rows = x.shape[1]
        out_shape = (experts, rows, output_size)
        counts = torch.full((experts,), rows, dtype=torch.int32, device=x.device)
        offsets = torch.arange(experts + 1, dtype=torch.int32, device=x.device) * rows
        schedule = None
        grid = (experts, triton.cdiv(rows, 128), triton.cdiv(output_size, 128))
    else:
        if x.ndim != 2 or x.shape[1] != width:
            raise ValueError(
                "Grouped NVFP4 inputs must be a matrix of reordered tokens"
            )
        rows = x.shape[0]
        out_shape = (rows, output_size)
        counts, offsets, schedule = metadata
        grid = (schedule.numel(), 1, triton.cdiv(output_size, 128))
    # Keep x alive until return: the PDL prepare kernel writes its workspace
    # before waiting on the quantize kernel, which may still be reading x.
    x = x.contiguous()
    values, x_scales, x_amax = nvfp4_quantize(x)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    if rows:
        total_rows = values.numel() // packed_width
        accumulator = torch.empty(
            (total_rows, output_size), dtype=torch.float32, device=x.device
        )
        sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
        module = _matmul_module()
        swap = schedule is not None and total_rows <= SWAP_AB_MAX_ROWS
        workspace = torch.empty(
            (module.swap_workspace_size if swap else module.workspace_size)(
                experts, total_rows, output_size, width, sm_count
            ),
            dtype=torch.uint8,
            device=x.device,
        )
        (module.run_swap if swap else module.run)(
            values.reshape(total_rows, packed_width),
            x_scales,
            weight,
            scales,
            counts,
            offsets,
            accumulator,
            workspace,
            sm_count,
        )
        if raw:
            return accumulator, x_amax
        _scale_output[grid](
            accumulator,
            x_amax,
            amax,
            outer_scale_denominator,
            out,
            counts,
            offsets,
            schedule,
            rows,
            output_size,
            128,
            num_warps=4,
            num_stages=3,
            enable_fp_fusion=False,
        )
    return out


def _routed_buffers(module, device, wide, experts, rows, n, k, sm_count):
    """Workspace (wide CUTLASS GEMM only) and the activation scale buffer, laid
    out as one 128-row scale block run per expert."""
    if not wide:
        scale_rows = (rows + 127 * experts + 127) // 128 * 128
        return None, torch.empty(
            scale_rows * (k // 16), dtype=torch.uint8, device=device
        )
    workspace = torch.empty(
        module.wide_workspace_size(experts, rows, n, k, sm_count),
        dtype=torch.uint8,
        device=device,
    )
    return workspace, workspace[module.wide_x_scales_offset(experts, rows, n, k) :]


def routed_expert_matmul(x, dst2src, top_k, weight, scales, counts, offsets, schedule):
    """Grouped NVFP4 GEMM of the routed token rows of x (row r reads token
    dst2src[r] // top_k). Returns the unscaled FP32 accumulator and per-row amax;
    rows outside offsets[0]..offsets[-1] are left unset."""
    experts, output_size, packed_width = weight.shape
    width = packed_width * 2
    rows = dst2src.numel()
    module = _matmul_module()
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    wide = rows > SWAP_AB_MAX_ROWS
    workspace, x_scales = _routed_buffers(
        module, x.device, wide, experts, rows, output_size, width, sm_count
    )
    values = x.new_empty((rows, packed_width), dtype=torch.uint8)
    amax = torch.empty(rows, dtype=torch.float32, device=x.device)
    routed_quant_module(width).quantize(
        x, dst2src, values, x_scales, amax, offsets, top_k, sm_count
    )
    accumulator = torch.empty((rows, output_size), dtype=torch.float32, device=x.device)
    if wide:
        module.run_wide_preswizzled(
            values, weight, scales, counts, offsets, accumulator, workspace, sm_count
        )
    else:
        routed_swap_matmul(
            values, x_scales, weight, scales, counts, offsets, schedule, accumulator
        )
    return accumulator, amax


def routed_swiglu_expert_matmul(
    accumulator,
    accumulator_amax,
    accumulator_weight_amax,
    accumulator_weight_denominator,
    gammas,
    gate_cap,
    up_cap,
    weight,
    scales,
    counts,
    offsets,
    out_dtype,
    schedule,
):
    """Scale the previous GEMM's accumulator, apply capped swiglu, quantize, and
    run the next grouped NVFP4 GEMM, for rows in offsets[0]..offsets[-1]."""
    if out_dtype != torch.bfloat16:
        raise ValueError("Routed NVFP4 swiglu rounds activations to bfloat16")
    experts, output_size, packed_width = weight.shape
    width = packed_width * 2
    rows = accumulator.shape[0]
    module = _matmul_module()
    sm_count = torch.cuda.get_device_properties(
        accumulator.device
    ).multi_processor_count
    wide = rows > SWAP_AB_MAX_ROWS
    workspace, x_scales = _routed_buffers(
        module, accumulator.device, wide, experts, rows, output_size, width, sm_count
    )
    values = accumulator.new_empty((rows, packed_width), dtype=torch.uint8)
    amax = torch.empty(rows, dtype=torch.float32, device=accumulator.device)
    routed_quant_module(width).swiglu_quantize(
        accumulator,
        accumulator_amax,
        accumulator_weight_amax,
        accumulator_weight_denominator,
        gammas,
        values,
        x_scales,
        amax,
        offsets,
        gate_cap,
        up_cap,
        sm_count,
    )
    out = torch.empty(
        (rows, output_size), dtype=torch.float32, device=accumulator.device
    )
    if wide:
        module.run_wide_preswizzled(
            values, weight, scales, counts, offsets, out, workspace, sm_count
        )
    else:
        routed_swap_matmul(
            values, x_scales, weight, scales, counts, offsets, schedule, out
        )
    return out, amax
