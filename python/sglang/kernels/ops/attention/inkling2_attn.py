import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit

PDL = is_arch_support_pdl()
# Below this many rows the Triton _gate_sink_kernel is faster.
GATE_SINK_CUDA_MIN_ROWS = 64


@triton.jit
def _sigmoid(x):
    exponent = -x * 1.4426950408889634
    exp = tl.inline_asm_elementwise(
        "ex2.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[exponent],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    denominator = exp + 1.0
    return tl.inline_asm_elementwise(
        "rcp.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[denominator],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _gate_sink_kernel(
    X,
    Gate,
    LSE,
    Sink,
    Y,
    SX: tl.constexpr,
    SG: tl.constexpr,
    SL0: tl.constexpr,
    SL1: tl.constexpr,
    SS: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    PADDED_GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < H * D
    head = col // D
    source_head = head + (head // GROUP) * (PADDED_GROUP - GROUP)
    x = tl.load(X + row * SX + source_head * D + col % D, mask, other=0).to(tl.float32)
    gate = tl.load(Gate + row * SG + col, mask, other=0).to(tl.float32)
    lse = tl.load(LSE + row * SL0 + source_head * SL1, mask, other=0).to(tl.float32)
    sink = tl.load(Sink + row * SS + head, mask, other=0).to(tl.float32)
    result = x * _sigmoid(gate)
    result = result * _sigmoid(lse - sink)
    tl.store(Y + row * H * D + col, result, mask)


@triton.jit
def _approx(x, OP: tl.constexpr):
    return tl.inline_asm_elementwise(
        OP + " $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _combine_gate_sink_kernel(
    OPartial,
    MPartial,
    LPartial,
    Gate,
    Sink,
    Y,
    splits,
    B,
    SM0,
    SMB,
    SMI,
    SMH,
    SQ: tl.constexpr,
    SG: tl.constexpr,
    SS: tl.constexpr,
    SY: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SPLITS_P: tl.constexpr,
    HAS_L: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    head = tl.program_id(1)
    cols = tl.arange(0, D)
    stat = row * H + head
    m_stat = (row // SQ) * SMB + (row % SQ) * SMI + head * SMH
    split_ids = tl.arange(0, SPLITS_P)
    maxes = tl.load(
        MPartial + split_ids * SM0 + m_stat, split_ids < splits, other=float("-inf")
    )
    max_final = tl.max(maxes, 0)
    total = tl.zeros((), tl.float32)
    out = tl.zeros((D,), tl.float32)
    for split in tl.static_range(SPLITS_P):
        max_partial = tl.load(
            MPartial + split * SM0 + m_stat, split < splits, other=float("-inf")
        )
        active = max_partial > float("-inf")
        correction = _approx(
            (max_partial - max_final) * 1.4426950408889634, "ex2.approx.ftz.f32"
        )
        if HAS_L:
            partial_sum = tl.load(LPartial + split * B * H + stat, active, other=0)
        else:
            partial_sum = tl.where(active, 1.0, 0.0)
        partial_out = tl.load(
            OPartial + (split * B * H + stat) * D + cols, active, other=0
        )
        total = tl.where(active, tl.fma(correction, partial_sum, total), total)
        out = tl.where(active, tl.fma(correction, partial_out, out), out)
    out = tl.where(
        max_final > float("-inf"), out * _approx(total, "rcp.approx.ftz.f32"), out
    )
    lse = max_final + libdevice.log(total)
    gate = tl.load(Gate + row * SG + head * D + cols).to(tl.float32)
    sink = tl.load(Sink + row * SS + head).to(tl.float32)
    result = out * _sigmoid(gate)
    result = result * _sigmoid(lse - sink)
    tl.store(Y + row * SY + head * D + cols, result)


def combine_gate_sink(o_partial, m_partial, l_partial, *, gate, sinks):
    """gate_sink over the split-KV attention partials (splits, tokens, 1, heads[, d])
    of paged_rel_bias_decode, combined as its reduction kernel does. With l_partial
    None, o_partial is normalized per split and m_partial is its LSE. m_partial is
    any strided (splits, batch, qlen, heads) view, tokens = batch * qlen."""
    splits, heads, d = o_partial.shape[0], o_partial.shape[-2], o_partial.shape[-1]
    assert o_partial.is_contiguous()
    tokens = o_partial[0].numel() // (heads * d)
    _, batch, qlen, _ = m_partial.shape
    assert batch * qlen == tokens
    m_strides = m_partial.stride()
    out = torch.empty((tokens, heads * d), device=gate.device, dtype=gate.dtype)
    if tokens:
        _combine_gate_sink_kernel[(tokens, heads)](
            o_partial,
            m_partial,
            l_partial,
            gate,
            sinks,
            out,
            splits,
            tokens,
            m_strides[0],
            m_strides[1],
            m_strides[2],
            m_strides[3],
            qlen,
            gate.stride(0),
            sinks.stride(0),
            out.stride(0),
            heads,
            d,
            triton.next_power_of_2(splits),
            l_partial is not None,
            enable_fp_fusion=False,
            launch_pdl=PDL,
        )
    return out


def rel_bias_partials(
    q,
    k_cache,
    v_cache,
    page_table,
    seqused_k,
    rel_bias,
    softmax_scale,
    out,
    *,
    window_size_left,
    verify,
):
    """Split-KV partials of paged attention with the compact relative bias, for
    combine_gate_sink; verify: several queries per sequence (MTP verify)."""
    from sglang.kernels.ops.attention.flash_attn.cute.flash_fwd_sm100_bias_decode import (
        paged_rel_bias_decode,
    )
    from sglang.srt.utils import get_device_core_count

    return paged_rel_bias_decode(
        q,
        k_cache,
        v_cache,
        page_table,
        seqused_k,
        rel_bias,
        softmax_scale,
        out,
        window_size_left=window_size_left,
        reduce=False,
        num_sms=(
            get_device_core_count(q.device.index)
            if verify and window_size_left is None
            else None
        ),
    )


@cache_once
def _gate_sink_module(heads, head_dim, group, padded_group):
    args = f"{heads}, {head_dim}, {group}, {padded_group}"
    return load_jit(
        "inkling2_gate_sink",
        *(str(v) for v in (heads, head_dim, group, padded_group)),
        cuda_files=["inkling2/gate_sink.cuh"],
        cuda_wrappers=[("run", f"inkling2_gate_sink::run<{args}>")],
    )


def _cuda_gate_sink(output, gate, lse, sinks, result, group, padded_group):
    """_gate_sink_kernel in CUDA; False when an operand layout is unsupported."""
    heads = sinks.shape[-1]
    head_dim = result.shape[1] // heads
    if not (
        result.shape[0] >= GATE_SINK_CUDA_MIN_ROWS
        and output.dtype == lse.dtype == torch.float32
        and gate.dtype == sinks.dtype == result.dtype == torch.bfloat16
        and output.stride(-1) == gate.stride(-1) == result.stride(-1) == 1
        and result.is_contiguous()
        and head_dim % 4 == 0
        and heads * head_dim // 4 <= 1024
        and output.data_ptr() % 16 == 0
        and output.stride(0) % 4 == 0
        and gate.data_ptr() % 8 == 0
        and gate.stride(0) % 4 == 0
        and result.data_ptr() % 8 == 0
    ):
        return False
    _gate_sink_module(heads, head_dim, group, padded_group).run(
        output, gate, lse, sinks, result
    )
    return True


def gate_sink(output, gate, lse, sinks, *, output_dtype, head_groups=None):
    """head_groups=(group, padded_group): output and lse hold padded_group heads per
    group of group real heads (padding heads are ignored)."""
    if not output.is_cuda:
        scale = torch.sigmoid(lse.float() - sinks.float())
        output = output.float() * torch.sigmoid(gate.float())
        return (
            (output.view(*sinks.shape, -1) * scale.unsqueeze(-1))
            .flatten(1)
            .to(output_dtype)
        )
    heads = sinks.shape[-1]
    group, padded_group = head_groups or (heads, heads)
    assert output.shape[1] == heads // group * padded_group * (gate.shape[1] // heads)
    result = torch.empty(gate.shape, device=output.device, dtype=output_dtype)
    if result.numel() and not _cuda_gate_sink(
        output, gate, lse, sinks, result, group, padded_group
    ):
        _gate_sink_kernel[(result.shape[0], triton.cdiv(result.shape[1], 256))](
            output,
            gate,
            lse,
            sinks,
            result,
            output.stride(0),
            gate.stride(0),
            *lse.stride(),
            sinks.stride(0),
            heads,
            result.shape[1] // heads,
            group,
            padded_group,
            256,
            enable_fp_fusion=False,
            launch_pdl=PDL,
        )
    return result


@triton.jit
def _tau_kernel(
    Pos,
    Out,
    FLOOR: tl.constexpr,
    ALPHA: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos = tl.load(Pos + i, i < N, other=0)
    ratio = tl.maximum((pos + 1).to(tl.float32) * (1.0 / FLOOR), 1.0)
    tau = 1.0 + ALPHA * libdevice.log(ratio)
    tl.store(Out + i, tau, i < N)


def compute_log_scaling_tau(positions, n_floor, alpha):
    if not positions.is_cuda:
        return 1.0 + alpha * torch.log(
            torch.clamp((positions + 1).float() / n_floor, min=1.0)
        )
    output = torch.empty(positions.shape, device=positions.device, dtype=torch.float32)
    if positions.numel():
        _tau_kernel[(triton.cdiv(positions.numel(), 256),)](
            positions,
            output,
            n_floor,
            alpha,
            positions.numel(),
            256,
            enable_fp_fusion=False,
            launch_pdl=PDL,
        )
    return output


@triton.jit
def _pad_head_groups(
    X,
    Y,
    rows,
    HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    PADDED: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    padded_heads: tl.constexpr = HEADS // GROUP * PADDED
    mask = cols < padded_heads * D
    head = cols // D
    source = (head // PADDED) * GROUP + head % PADDED
    keep = mask & (head % PADDED < GROUP)
    x = tl.load(X + row * HEADS * D + source * D + cols % D, keep, other=0)
    tl.store(Y + row * padded_heads * D + cols, x, mask)


def pad_head_groups(x, group, padded_group):
    """(tokens, heads, d) -> (tokens, heads // group * padded_group, d), zero-filling
    the padded_group - group extra heads after each group."""
    tokens, heads, width = x.shape
    out = x.new_empty((tokens, heads // group * padded_group, width))
    if tokens:
        _pad_head_groups[(tokens, triton.cdiv(out.shape[1] * width, 1024))](
            x.contiguous(),
            out,
            tokens,
            heads,
            group,
            padded_group,
            width,
            1024,
            launch_pdl=PDL,
        )
    return out
