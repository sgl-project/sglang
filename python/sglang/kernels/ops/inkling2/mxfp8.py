import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit
from sglang.kernels.ops.inkling2.moe import _row_expert, mxfp8_quantize
from sglang.kernels.ops.inkling2.nvfp4 import routed_quant_module

PDL = is_arch_support_pdl()


@triton.jit
def _expert_matmul(
    X,
    XS,
    W,
    WS,
    Y,
    Counts,
    Offsets,
    Schedule,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
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
    col0 = tl.program_id(2) * BLOCK
    cols = col0 + tl.arange(0, BLOCK)
    k = tl.arange(0, BLOCK)
    scale_k = tl.arange(0, BLOCK // 32)
    acc = tl.zeros((BLOCK, BLOCK), tl.float32)
    for tile in range(tl.cdiv(K, BLOCK)):
        offsets = tile * BLOCK + k
        scales = tile * (BLOCK // 32) + scale_k
        x = tl.load(
            X + row_ids[:, None] * K + offsets[None, :],
            row_mask[:, None] & (offsets[None, :] < K),
            other=0.0,
        )
        w = W.load([(expert * N + col0).to(tl.int32), tile * BLOCK])
        xs = tl.load(
            XS + row_ids[:, None] * (K // 32) + scales[None, :],
            row_mask[:, None] & (scales[None, :] < K // 32),
            other=0,
        )
        ws = tl.load(
            WS + expert * N * (K // 32) + cols[:, None] * (K // 32) + scales[None, :],
            (cols[:, None] < N) & (scales[None, :] < K // 32),
            other=0,
        )
        acc = tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3", acc, fast_math=True)
    tl.store(
        Y + row_ids[:, None] * N + cols[None, :],
        acc,
        row_mask[:, None] & (cols[None, :] < N),
    )


def expert_matmul(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, *, metadata=None
):
    experts, output_size, width = weight.shape
    if metadata is None:
        if x.ndim == 2:
            x = x.unsqueeze(0).expand(experts, -1, -1)
        if x.ndim != 3 or x.shape[0] != experts or x.shape[-1] != width:
            raise ValueError("MXFP8 expert input and weight shapes do not match")
        rows = x.shape[1]
        out_shape = (experts, rows, output_size)
        counts, offsets, schedule = None, None, None
        grid = (experts, triton.cdiv(rows, 128), triton.cdiv(output_size, 128))
    else:
        if x.ndim != 2 or x.shape[1] != width:
            raise ValueError(
                "Grouped MXFP8 inputs must be a matrix of reordered tokens"
            )
        rows = x.shape[0]
        out_shape = (rows, output_size)
        counts, offsets, schedule = metadata
        grid = (min(schedule.numel(), rows), 1, triton.cdiv(output_size, 128))
    values, x_scales = mxfp8_quantize(x.contiguous())
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    if rows:
        _expert_matmul[grid](
            values,
            x_scales,
            TensorDescriptor.from_tensor(
                weight.view(experts * output_size, width), [128, 128]
            ),
            scales,
            out,
            counts,
            offsets,
            schedule,
            rows,
            output_size,
            width,
            128,
            num_warps=4,
            num_stages=4,
            launch_pdl=PDL,
        )
    return out


@triton.jit
def _routed_swap_ab_matmul(
    X,
    XS,
    W,
    WS,
    Y,
    Counts,
    Offsets,
    Schedule,
    Dst2Src,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    TOP_K: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    entry = tl.load(Schedule + tl.program_id(0))
    if entry == -1:
        return
    expert = (entry & 0xFFFF).to(tl.int64)
    block = (entry >> 16) * 128
    count = tl.minimum(tl.load(Counts + expert) - block, 128)
    col0 = tl.program_id(1) * BN
    cols = col0 + tl.arange(0, BN)
    for m0 in range(0, count, BM):
        local = block + m0 + tl.arange(0, BM)
        row_mask = m0 + tl.arange(0, BM) < count
        row_ids = tl.load(Offsets + expert).to(tl.int64) + local
        src_ids = row_ids
        if Dst2Src is not None:
            src_ids = (
                tl.load(Dst2Src + row_ids, row_mask, other=0).to(tl.int64) // TOP_K
            )
        acc = tl.zeros((BN, BM), tl.float32)
        for k0 in range(0, K, BK):
            offsets = k0 + tl.arange(0, BK)
            scales = k0 // 32 + tl.arange(0, BK // 32)
            w = W.load([(expert * N + col0).to(tl.int32), k0])
            ws = tl.load(
                WS + (expert * N + cols)[:, None] * (K // 32) + scales[None, :]
            )
            x = tl.load(
                X + src_ids[:, None] * K + offsets[None, :],
                row_mask[:, None],
                other=0.0,
            )
            xs = tl.load(
                XS + src_ids[:, None] * (K // 32) + scales[None, :],
                row_mask[:, None],
                other=0,
            )
            acc = tl.dot_scaled(
                w, ws, "e4m3", tl.trans(x), xs, "e4m3", acc, fast_math=True
            )
        tl.store(Y + row_ids[None, :] * N + cols[:, None], acc, row_mask[None, :])


def _routed_matmul(
    values, x_scales, rows, tokens, weight, scales, metadata, dst2src, top_k
):
    """Routed rows of the grouped MXFP8 GEMM as weight-rows x tokens tiles;
    bitwise equal to _expert_matmul's tokens x weight-rows tiles."""
    experts, output_size, width = weight.shape
    counts, offsets, schedule = metadata
    out = torch.empty((rows, output_size), dtype=torch.bfloat16, device=values.device)
    block_k = next(b for b in (512, 256, 128, 64, 32) if width % b == 0 and width > b)
    if rows:
        _routed_swap_ab_matmul[(min(schedule.numel(), rows), output_size // 128)](
            values,
            x_scales,
            TensorDescriptor.from_tensor(
                weight.view(experts * output_size, width), [128, block_k]
            ),
            scales,
            out,
            counts,
            offsets,
            schedule,
            dst2src,
            output_size,
            width,
            16 if tokens <= 16 else 32,
            128,
            block_k,
            top_k,
            num_warps=4,
            num_stages=2 if block_k == 512 else 3,
            launch_pdl=PDL,
        )
    return out


def routed_expert_matmul(x, dst2src, top_k, weight, scales, metadata):
    """expert_matmul over pre_reorder(x) rows, quantizing the token rows once and
    gathering them through dst2src."""
    values, x_scales = mxfp8_quantize(x.contiguous())
    return _routed_matmul(
        values,
        x_scales,
        dst2src.numel(),
        x.shape[0],
        weight,
        scales,
        metadata,
        dst2src,
        top_k,
    )


def routed_swiglu_expert_matmul(
    gate_up, gammas, gate_cap, up_cap, weight, scales, metadata, tokens
):
    """expert_matmul over routed_swiglu(gate_up, ...) rows, with the swiglu fused
    into the activation quantization."""
    rows, width = gate_up.shape[0], gate_up.shape[1] // 2
    counts, offsets, _ = metadata
    values = torch.empty(
        (rows, width), dtype=torch.float8_e4m3fn, device=gate_up.device
    )
    x_scales = torch.empty(
        (rows, width // 32), dtype=torch.uint8, device=gate_up.device
    )
    if rows:
        routed_quant_module(width).mxfp8_swiglu_quantize(
            gate_up,
            gammas,
            values.view(torch.uint8),
            x_scales,
            offsets,
            gate_cap,
            up_cap,
            torch.cuda.get_device_properties(gate_up.device).multi_processor_count,
        )
    return _routed_matmul(
        values, x_scales, rows, tokens, weight, scales, metadata, None, 1
    )


@triton.jit
def _sf_offsets(local, group, K: tl.constexpr):
    """Byte offset of the scale of (row local, 32-wide K group) in CUTLASS's
    blocked scale layout: 128x4 atoms, K atom fastest."""
    SCALE_K: tl.constexpr = (K + 127) // 128 * 4
    return (
        ((local // 128) * (SCALE_K // 4) + group // 4) * 512
        + (local % 32) * 16
        + ((local // 32) % 4) * 4
        + group % 4
    )


@triton.jit
def _sf_base(Offsets, expert, K: tl.constexpr):
    """Each expert's scales start on a 128-row boundary (mxfp8.cuh prepare)."""
    SCALE_K: tl.constexpr = (K + 127) // 128 * 4
    start = tl.load(Offsets + expert)
    return (start - tl.load(Offsets) + expert * 127) // 128 * 128 * SCALE_K, start


@triton.jit
def _zero_sf_tails(
    S, Offsets, E: tl.constexpr, GROUPS: tl.constexpr, GROUPS_P2: tl.constexpr
):
    """Zero the scales of rows count..count rounded up to 128 of each expert."""
    K: tl.constexpr = GROUPS * 32
    for expert in range(tl.program_id(0), E, tl.num_programs(0)):
        base, start = _sf_base(Offsets, expert, K)
        count = tl.load(Offsets + expert + 1) - start
        local = count + tl.arange(0, 128)
        groups = tl.arange(0, GROUPS_P2)
        mask = (local[:, None] < (count + 127) // 128 * 128) & (
            groups[None, :] < GROUPS
        )
        tl.store(
            S + base + _sf_offsets(local[:, None], groups[None, :], K),
            tl.zeros(mask.shape, tl.uint8),
            mask,
        )


@triton.jit
def _gather_routed_rows(
    Values,
    Scales,
    Dst2Src,
    A,
    S,
    Offsets,
    K: tl.constexpr,
    TOP_K: tl.constexpr,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
    SEARCH: tl.constexpr,
    GROUPS_P2: tl.constexpr,
):
    """Copy this rank's routed rows of the quantized tokens into expert order,
    with their scales in the CUTLASS layout."""
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    _zero_sf_tails(S, Offsets, E, K // 32, GROUPS_P2)
    groups = tl.arange(0, GROUPS_P2)
    for row in range(
        tl.load(Offsets) + tl.program_id(0), tl.load(Offsets + E), tl.num_programs(0)
    ):
        source = (tl.load(Dst2Src + row) // TOP_K).to(tl.int64)
        for chunk in tl.static_range((K + BLOCK - 1) // BLOCK):
            cols = chunk * BLOCK + tl.arange(0, BLOCK)
            tl.store(
                A + row.to(tl.int64) * K + cols,
                tl.load(Values + source * K + cols, cols < K),
                cols < K,
            )
        expert = _row_expert(Offsets, row, E, SEARCH)
        base, start = _sf_base(Offsets, expert, K)
        tl.store(
            S + base + _sf_offsets(row - start, groups, K),
            tl.load(Scales + source * (K // 32) + groups, groups < K // 32),
            groups < K // 32,
        )


@cache_once
def _cutlass_module():
    return load_jit(
        "inkling2_mxfp8",
        cuda_files=["inkling2/mxfp8.cuh"],
        cuda_wrappers=[
            ("run", "inkling2_mxfp8::run"),
            ("workspace_size", "inkling2_mxfp8::workspace_size"),
            ("x_scales_offset", "inkling2_mxfp8::x_scales_offset"),
            ("swizzle_weight_scales", "inkling2_mxfp8::swizzle_weight_scales"),
        ],
        extra_dependencies=["cutlass"],
    )


def cutlass_weight_scales(scales: torch.Tensor, out: torch.Tensor | None = None):
    """(experts, n, k // 32) scales in the CUTLASS grouped-GEMM layout, written
    into out when given (in place, for CUDA-graph-held buffers)."""
    experts, n, groups = scales.shape
    size = triton.cdiv(n, 128) * 128 * triton.cdiv(groups * 32, 128) * 4
    if out is None:
        out = torch.zeros((experts, size), dtype=torch.uint8, device=scales.device)
    if out.numel():
        _cutlass_module().swizzle_weight_scales(scales.contiguous(), out)
    return out


def _cutlass_grouped(values, weight, weight_scales, counts, offsets, workspace, out):
    sm_count = torch.cuda.get_device_properties(values.device).multi_processor_count
    _cutlass_module().run(
        values,
        workspace.new_empty(0),
        weight,
        weight_scales,
        counts,
        offsets,
        out,
        workspace,
        sm_count,
    )
    return out


def _cutlass_workspace(device, experts, rows, n, k):
    module = _cutlass_module()
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    workspace = torch.empty(
        module.workspace_size(experts, rows, n, k, sm_count),
        dtype=torch.uint8,
        device=device,
    )
    return workspace, workspace[module.x_scales_offset(experts, rows, k) :]


def prefill_expert_matmul(x, dst2src, top_k, weight, weight_scales, counts, offsets):
    """expert_matmul over pre_reorder(x) rows of this rank's experts on the CUTLASS
    grouped GEMM: token rows are quantized once and gathered in expert order."""
    experts, output_size, width = weight.shape
    rows = dst2src.numel()
    values, scales = mxfp8_quantize(x.contiguous())
    workspace, x_scales = _cutlass_workspace(
        x.device, experts, rows, output_size, width
    )
    gathered = torch.empty((rows, width), dtype=torch.float8_e4m3fn, device=x.device)
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    _gather_routed_rows[(min(rows + experts, 8 * sm_count),)](
        values.view(torch.uint8),
        scales,
        dst2src,
        gathered.view(torch.uint8),
        x_scales,
        offsets,
        width,
        top_k,
        experts,
        1024,
        experts.bit_length(),
        triton.next_power_of_2(width // 32),
        launch_pdl=PDL,
    )
    out = torch.empty((rows, output_size), dtype=x.dtype, device=x.device)
    return _cutlass_grouped(
        gathered, weight, weight_scales, counts, offsets, workspace, out
    )


def prefill_swiglu_expert_matmul(
    gate_up, gammas, gate_cap, up_cap, weight, weight_scales, counts, offsets
):
    """prefill_expert_matmul over routed_swiglu(gate_up, ...) rows, with the
    swiglu fused into the activation quantization."""
    experts, output_size, width = weight.shape
    rows = gate_up.shape[0]
    workspace, _ = _cutlass_workspace(gate_up.device, experts, rows, output_size, width)
    values = torch.empty(
        (rows, width), dtype=torch.float8_e4m3fn, device=gate_up.device
    )
    x_scales = torch.empty(
        (rows, width // 32), dtype=torch.uint8, device=gate_up.device
    )
    sm_count = torch.cuda.get_device_properties(gate_up.device).multi_processor_count
    routed_quant_module(width).mxfp8_swiglu_quantize(
        gate_up,
        gammas,
        values.view(torch.uint8),
        x_scales,
        offsets,
        gate_cap,
        up_cap,
        sm_count,
    )
    out = torch.empty((rows, output_size), dtype=gate_up.dtype, device=gate_up.device)
    _cutlass_module().run(
        values,
        x_scales,
        weight,
        weight_scales,
        counts,
        offsets,
        out,
        workspace,
        sm_count,
    )
    return out
