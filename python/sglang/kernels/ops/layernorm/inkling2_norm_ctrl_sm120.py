import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

NORM_CTRL_HIDDEN_SIZE = 2304
NORM_CTRL_ATTN_RES_MAX_TOKENS = 64
NORM_CTRL_LOOP_MAX_TOKENS = 32
NORM_CTRL_GATE_RANK = 64


def _require_sm12x() -> None:
    if not current_arch().is_sm(12):
        raise RuntimeError("inkling2_norm_ctrl_sm120 requires NVIDIA SM12x")


@cache_once
def _module():
    _require_sm12x()
    return load_jit(
        "inkling2_norm_ctrl_sm120",
        cuda_files=["inkling2/norm_ctrl_sm120.cuh"],
        cuda_wrappers=[
            (
                "attn_res_norm_bf16",
                "inkling2_norm_ctrl_sm120::attn_res_norm_bf16",
            ),
            (
                "pre_attn_mix_cross_s1",
                "inkling2_norm_ctrl_sm120::pre_attn_mix_cross_s1",
            ),
            (
                "pre_mlp_mix_s1",
                "inkling2_norm_ctrl_sm120::pre_mlp_mix_s1",
            ),
            (
                "ctrl_bottleneck_rmsnorm",
                "inkling2_norm_ctrl_sm120::ctrl_bottleneck_rmsnorm",
            ),
        ],
    )


def _blocks_ok(blocks, partial) -> bool:
    return all(
        b.is_cuda
        and b.dtype == torch.bfloat16
        and b.device == partial.device
        and b.shape == partial.shape
        for b in blocks
    )


def can_use_attn_res_norm_sm120(
    blocks,
    partial,
    queries,
    norm_w,
    *,
    normalize_values,
    pre_weight=None,
    addend=None,
):
    if (
        not normalize_values
        or pre_weight is not None
        or not partial.is_cuda
        or partial.dtype != torch.bfloat16
        or partial.ndim != 2
        or not (0 < partial.shape[0] <= NORM_CTRL_ATTN_RES_MAX_TOKENS)
        or partial.shape[1] != NORM_CTRL_HIDDEN_SIZE
        or not (0 <= len(blocks) <= 4)
        or not _blocks_ok(blocks, partial)
        or (
            addend is not None
            and (
                not addend.is_cuda
                or addend.dtype != torch.bfloat16
                or addend.device != partial.device
                or addend.shape != partial.shape
            )
        )
        or not queries.is_cuda
        or queries.dtype != torch.float32
        or queries.device != partial.device
        or queries.ndim not in (1, 2)
        or queries.shape[-1] != NORM_CTRL_HIDDEN_SIZE
        or norm_w is None
        or not norm_w.is_cuda
        or norm_w.dtype != torch.bfloat16
        or norm_w.device != partial.device
        or norm_w.ndim not in (1, 2)
        or norm_w.shape[-1] != NORM_CTRL_HIDDEN_SIZE
        or not current_arch().is_sm(12)
        or is_batch_invariant_mode_enabled()
    ):
        return False
    num_q = 1 if queries.ndim == 1 else queries.shape[0]
    num_nw = 1 if norm_w.ndim == 1 else norm_w.shape[0]
    return num_q in (1, 3) and num_nw == num_q


def can_use_loop_ctrl_sm120(
    blocks,
    partial,
    prev_blocks,
    ctrl_weights,
    norm_w,
    gate_rank,
    *,
    normalize_values,
    addend=None,
):
    if (
        not normalize_values
        or ctrl_weights is None
        or norm_w is None
        or not partial.is_cuda
        or partial.dtype != torch.bfloat16
        or partial.ndim != 2
        or not (0 < partial.shape[0] <= NORM_CTRL_LOOP_MAX_TOKENS)
        or partial.shape[1] != NORM_CTRL_HIDDEN_SIZE
        or gate_rank != NORM_CTRL_GATE_RANK
        or not (1 <= len(blocks) <= 3)
        or len(prev_blocks) != 2
        or not _blocks_ok(blocks, partial)
        or not _blocks_ok(prev_blocks, partial)
        or (
            addend is not None
            and (
                not addend.is_cuda
                or addend.dtype != torch.bfloat16
                or addend.device != partial.device
                or addend.shape != partial.shape
            )
        )
        or not norm_w.is_cuda
        or norm_w.dtype != torch.bfloat16
        or norm_w.device != partial.device
        or not all(
            w.is_cuda and w.dtype == torch.bfloat16 and w.device == partial.device
            for w in ctrl_weights
        )
        or not current_arch().is_sm(12)
        or is_batch_invariant_mode_enabled()
    ):
        return False
    return True


def _pad_blocks(blocks, max_blocks: int, *, min_blocks: int = 0):
    if not (min_blocks <= len(blocks) <= max_blocks):
        raise ValueError(
            f"expected between {min_blocks} and {max_blocks} blocks, got {len(blocks)}"
        )
    blocks_c = [b.contiguous() for b in blocks]
    while len(blocks_c) < max_blocks:
        blocks_c.append(None)
    return tuple(blocks_c)


def attn_res_norm_bf16(
    blocks,
    partial,
    addend,
    scaled_queries,
    norm_w,
    *,
    eps=1e-6,
    norm_eps=1e-6,
):
    b0, b1, b2, b3 = _pad_blocks(blocks, 4)
    q = scaled_queries.unsqueeze(0) if scaled_queries.ndim == 1 else scaled_queries
    q = q.contiguous()
    nw = norm_w.unsqueeze(0) if norm_w.ndim == 1 else norm_w
    nw = nw.contiguous()
    partial_c = partial.contiguous()
    addend_c = addend.contiguous() if addend is not None else None
    num_q = q.shape[0]
    tokens, width = partial_c.shape
    out = torch.empty(
        (num_q, tokens, width), dtype=torch.bfloat16, device=partial_c.device
    )
    if tokens:
        _module().attn_res_norm_bf16(
            b0,
            b1,
            b2,
            b3,
            partial_c,
            addend_c,
            q,
            nw,
            out,
            float(eps),
            float(norm_eps),
        )
    return out


def loop_pre_attn_rmsnorm(
    blocks,
    partial,
    attn_q,
    prev_blocks,
    cross_q,
    ctrl_weights,
    norm_w,
    *,
    is_local,
    eps=1e-6,
    norm_eps=1e-6,
):
    b0, b1, b2 = _pad_blocks(blocks, 3, min_blocks=1)
    if len(prev_blocks) != 2:
        raise ValueError(
            f"loop_pre_attn_rmsnorm requires exactly 2 prev_blocks, got {len(prev_blocks)}"
        )
    partial_c = partial.contiguous()
    tokens, width = partial_c.shape
    dev = partial_c.device
    q_attn = attn_q.unsqueeze(0) if attn_q.ndim == 1 else attn_q
    q_attn = q_attn.contiguous()
    q_cross = cross_q.contiguous()
    prev0, prev1 = prev_blocks[0].contiguous(), prev_blocks[1].contiguous()
    mix_qkv = torch.empty(
        (1 if is_local else 3, tokens, width), dtype=torch.float32, device=dev
    )
    msg_buf = torch.empty((4, tokens, width), dtype=torch.float32, device=dev)
    out = torch.empty((3, tokens, width), dtype=torch.bfloat16, device=dev)
    if tokens == 0:
        return out, msg_buf[3]

    cat_3x = torch.empty((3, tokens, 2 * width), dtype=torch.bfloat16, device=dev)
    mod = _module()
    mod.pre_attn_mix_cross_s1(
        b0,
        b1,
        b2,
        partial_c,
        q_attn,
        prev0,
        prev1,
        q_cross,
        mix_qkv,
        msg_buf,
        cat_3x,
        1 if is_local else 0,
        float(eps),
    )
    down, gain, forget = ctrl_weights
    nw = norm_w.unsqueeze(0) if norm_w.ndim == 1 else norm_w
    h_buf = torch.empty(
        (3, tokens, NORM_CTRL_GATE_RANK), dtype=torch.bfloat16, device=dev
    )
    fg_buf = torch.empty((3, tokens, 2 * width), dtype=torch.float32, device=dev)
    mod.ctrl_bottleneck_rmsnorm(
        cat_3x,
        down[:3].contiguous(),
        gain[:3].contiguous(),
        forget[:3].contiguous(),
        mix_qkv,
        msg_buf[:3].contiguous(),
        nw.contiguous(),
        h_buf,
        fg_buf,
        out,
        float(norm_eps),
    )
    return out, msg_buf[3]


def loop_pre_mlp_rmsnorm(
    blocks,
    partial,
    addend,
    mlp_q,
    mlp_msg,
    ctrl_weights,
    norm_w,
    *,
    eps=1e-6,
    norm_eps=1e-6,
):
    b0, b1, b2 = _pad_blocks(blocks, 3, min_blocks=1)
    partial_c = partial.contiguous()
    addend_c = addend.contiguous() if addend is not None else None
    tokens, width = partial_c.shape
    dev = partial_c.device
    out = torch.empty((1, tokens, width), dtype=torch.bfloat16, device=dev)
    if tokens == 0:
        return out[0]

    q_mlp = mlp_q.unsqueeze(0) if mlp_q.ndim == 1 else mlp_q
    msg_c = mlp_msg.contiguous()
    mix_mlp = torch.empty((tokens, width), dtype=torch.float32, device=dev)
    cat_1x = torch.empty((1, tokens, 2 * width), dtype=torch.bfloat16, device=dev)
    mod = _module()
    mod.pre_mlp_mix_s1(
        b0,
        b1,
        b2,
        partial_c,
        addend_c,
        q_mlp.contiguous(),
        msg_c,
        mix_mlp,
        cat_1x,
        float(eps),
    )
    down, gain, forget = ctrl_weights
    sl = slice(3, 4) if down.shape[0] == 4 else slice(0, 1)
    nw = norm_w.unsqueeze(0) if norm_w.ndim == 1 else norm_w
    h_buf = torch.empty(
        (1, tokens, NORM_CTRL_GATE_RANK), dtype=torch.bfloat16, device=dev
    )
    fg_buf = torch.empty((1, tokens, 2 * width), dtype=torch.float32, device=dev)
    mod.ctrl_bottleneck_rmsnorm(
        cat_1x,
        down[sl].contiguous(),
        gain[sl].contiguous(),
        forget[sl].contiguous(),
        mix_mlp.unsqueeze(0),
        msg_c.unsqueeze(0),
        nw.contiguous(),
        h_buf,
        fg_buf,
        out,
        float(norm_eps),
    )
    return out[0]
