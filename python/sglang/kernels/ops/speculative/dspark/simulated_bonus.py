from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _simulated_bonus_row_partial_kernel(
    logits_ptr,
    logits_stride,
    temperatures_ptr,
    correct_len_ptr,
    correct_len_stride,
    seed_ptr,
    part_val_ptr,
    part_idx_ptr,
    vocab,
    ROWS: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    split = tl.program_id(1)
    row = req * ROWS + tl.load(correct_len_ptr + req * correct_len_stride)
    t = tl.load(temperatures_ptr + req).to(tl.float32)
    offs = split * BLOCK + tl.arange(0, BLOCK)
    mask = offs < vocab
    x = tl.load(
        logits_ptr + row.to(tl.int64) * logits_stride + offs,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    seed = tl.load(seed_ptr)
    u = tl.rand(seed, req * vocab + offs)
    noisy = x / tl.maximum(t, 1e-5) - tl.log(-tl.log(u))
    v = tl.where(mask, tl.where(t > 0.0, noisy, x), float("-inf"))
    top = tl.max(v, axis=0)
    idx = tl.min(tl.where(v == top, offs, vocab), axis=0)
    tl.store(part_val_ptr + req * NSPLIT + split, top)
    tl.store(part_idx_ptr + req * NSPLIT + split, idx)


@triton.jit
def _residual_lse_partial_kernel(
    logits_ptr,
    logits_stride,
    temperatures_ptr,
    correct_len_ptr,
    correct_len_stride,
    draft_ptr,
    draft_stride0,
    draft_stride1,
    draft_temperatures_ptr,
    part_ptr,
    vocab,
    width,
    ROWS: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Per (request, vocab split): (max, sum exp(. - max)) of the target row at correct_len
    # (x / T) and of the draft step row that proposed the stop token (d / T_d).
    req = tl.program_id(0)
    split = tl.program_id(1)
    c = tl.load(correct_len_ptr + req * correct_len_stride)
    row = req * ROWS + c
    t = tl.maximum(tl.load(temperatures_ptr + req).to(tl.float32), 1e-5)
    td = tl.maximum(tl.load(draft_temperatures_ptr + req).to(tl.float32), 1e-5)
    dc = tl.minimum(c, ROWS - 2)
    offs = split * BLOCK + tl.arange(0, BLOCK)
    x = (
        tl.load(
            logits_ptr + row.to(tl.int64) * logits_stride + offs,
            mask=offs < vocab,
            other=float("-inf"),
        ).to(tl.float32)
        / t
    )
    m_t = tl.max(x, axis=0)
    s_t = tl.sum(tl.exp(x - m_t), axis=0)
    d = (
        tl.load(
            draft_ptr + req.to(tl.int64) * draft_stride0 + dc * draft_stride1 + offs,
            mask=offs < width,
            other=float("-inf"),
        ).to(tl.float32)
        / td
    )
    m_d = tl.max(d, axis=0)
    s_d = tl.where(m_d > float("-inf"), tl.sum(tl.exp(d - m_d), axis=0), 0.0)
    base = (req * NSPLIT + split) * 4
    tl.store(part_ptr + base + 0, m_t)
    tl.store(part_ptr + base + 1, s_t)
    tl.store(part_ptr + base + 2, m_d)
    tl.store(part_ptr + base + 3, s_d)


@triton.jit
def _combine_lse(part_ptr, req, which: tl.constexpr, NSPLIT: tl.constexpr):
    k = tl.arange(0, NSPLIT)
    base = (req * NSPLIT + k) * 4 + 2 * which
    m = tl.load(part_ptr + base)
    s = tl.load(part_ptr + base + 1)
    top = tl.max(m, axis=0)
    w = tl.where(m > float("-inf"), tl.exp(m - top), 0.0)
    return top + tl.log(tl.sum(s * w, axis=0))


@triton.jit
def _residual_weights(
    logits_ptr,
    logits_stride,
    draft_ptr,
    draft_stride0,
    draft_stride1,
    lse_part_ptr,
    req,
    row,
    dc,
    offs,
    t,
    td,
    residual_row,
    vocab,
    width,
    NSPLIT: tl.constexpr,
):
    # (max(p - q, 0), p) over ``offs`` of the stop row, normalised by the row logsumexps.
    x = tl.load(
        logits_ptr + row.to(tl.int64) * logits_stride + offs,
        mask=offs < vocab,
        other=float("-inf"),
    ).to(tl.float32)
    p = tl.exp(x / t - _combine_lse(lse_part_ptr, req, 0, NSPLIT))
    d = tl.load(
        draft_ptr + req.to(tl.int64) * draft_stride0 + dc * draft_stride1 + offs,
        mask=residual_row & (offs < width),
        other=float("-inf"),
    ).to(tl.float32)
    q = tl.exp(d / td - _combine_lse(lse_part_ptr, req, 1, NSPLIT))
    r = tl.where(residual_row, tl.maximum(p - q, 0.0), p)
    return r, p, x


@triton.jit
def _residual_mass_kernel(
    logits_ptr,
    logits_stride,
    temperatures_ptr,
    correct_len_ptr,
    correct_len_stride,
    draft_ptr,
    draft_stride0,
    draft_stride1,
    draft_temperatures_ptr,
    lse_part_ptr,
    mass_ptr,
    part_val_ptr,
    part_idx_ptr,
    vocab,
    width,
    ROWS: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Per (request, split): the residual mass and the target mass (inverse-CDF pass 1);
    # argmax rows (T <= 0) keep the plain per-split argmax for the select.
    req = tl.program_id(0)
    split = tl.program_id(1)
    c = tl.load(correct_len_ptr + req * correct_len_stride)
    row = req * ROWS + c
    t_raw = tl.load(temperatures_ptr + req).to(tl.float32)
    td_raw = tl.load(draft_temperatures_ptr + req).to(tl.float32)
    t = tl.maximum(t_raw, 1e-5)
    td = tl.maximum(td_raw, 1e-5)
    residual_row = (c < ROWS - 1) & (td_raw > 0.0)
    dc = tl.minimum(c, ROWS - 2)
    offs = split * BLOCK + tl.arange(0, BLOCK)
    r, p, x = _residual_weights(
        logits_ptr,
        logits_stride,
        draft_ptr,
        draft_stride0,
        draft_stride1,
        lse_part_ptr,
        req,
        row,
        dc,
        offs,
        t,
        td,
        residual_row,
        vocab,
        width,
        NSPLIT,
    )
    tl.store(mass_ptr + (req * NSPLIT + split) * 2, tl.sum(r, axis=0))
    tl.store(mass_ptr + (req * NSPLIT + split) * 2 + 1, tl.sum(p, axis=0))
    if t_raw <= 0.0:
        v = tl.where(offs < vocab, x, float("-inf"))
        top = tl.max(v, axis=0)
        tl.store(part_val_ptr + req * NSPLIT + split, top)
        tl.store(
            part_idx_ptr + req * NSPLIT + split,
            tl.min(tl.where(v == top, offs, vocab), axis=0),
        )


@triton.jit
def _residual_pick_kernel(
    logits_ptr,
    logits_stride,
    temperatures_ptr,
    correct_len_ptr,
    correct_len_stride,
    seed_ptr,
    draft_ptr,
    draft_stride0,
    draft_stride1,
    draft_temperatures_ptr,
    lse_part_ptr,
    mass_ptr,
    part_val_ptr,
    part_idx_ptr,
    vocab,
    width,
    ROWS: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Inverse-CDF pass 2: one uniform per request picks the split, then the token inside
    # it. Draws from max(p - q, 0) at a stop row (correct_len < gamma), as a rejection's
    # resample would, and from p on the all-accepted row or when the residual is empty.
    # The pick is written as split 0 of the (value, index) partials the select reduces.
    req = tl.program_id(0)
    t_raw = tl.load(temperatures_ptr + req).to(tl.float32)
    if t_raw > 0.0:
        c = tl.load(correct_len_ptr + req * correct_len_stride)
        row = req * ROWS + c
        td_raw = tl.load(draft_temperatures_ptr + req).to(tl.float32)
        t = tl.maximum(t_raw, 1e-5)
        td = tl.maximum(td_raw, 1e-5)
        k = tl.arange(0, NSPLIT)
        mass_r = tl.load(mass_ptr + (req * NSPLIT + k) * 2)
        mass_p = tl.load(mass_ptr + (req * NSPLIT + k) * 2 + 1)
        residual_row = (c < ROWS - 1) & (td_raw > 0.0) & (tl.sum(mass_r, axis=0) > 0.0)
        mass = tl.where(residual_row, mass_r, mass_p)
        total = tl.sum(mass, axis=0)
        u = tl.sum(tl.rand(tl.load(seed_ptr), req + tl.zeros([1], tl.int32)), axis=0)
        target = u * total
        cum = tl.cumsum(mass, axis=0)
        last_split = tl.max(tl.where(mass > 0.0, k, 0), axis=0)
        pick = tl.minimum(tl.sum((cum <= target).to(tl.int32), axis=0), last_split)
        rem = target - tl.sum(tl.where(k < pick, mass, 0.0), axis=0)
        dc = tl.minimum(c, ROWS - 2)
        offs = pick * BLOCK + tl.arange(0, BLOCK)
        r, p, x = _residual_weights(
            logits_ptr,
            logits_stride,
            draft_ptr,
            draft_stride0,
            draft_stride1,
            lse_part_ptr,
            req,
            row,
            dc,
            offs,
            t,
            td,
            residual_row,
            vocab,
            width,
            NSPLIT,
        )
        cw = tl.cumsum(r, axis=0)
        last = tl.max(tl.where(r > 0.0, tl.arange(0, BLOCK), 0), axis=0)
        idx = tl.minimum(tl.sum((cw <= rem).to(tl.int32), axis=0), last)
        k1 = tl.arange(0, NSPLIT)
        tl.store(
            part_val_ptr + req * NSPLIT + k1, tl.where(k1 == 0, 0.0, float("-inf"))
        )
        tl.store(
            part_idx_ptr + req * NSPLIT + k1, tl.where(k1 == 0, pick * BLOCK + idx, 0)
        )


@triton.jit
def _simulated_bonus_row_select_kernel(
    part_val_ptr,
    part_idx_ptr,
    out_ptr,
    vocab,
    NSPLIT: tl.constexpr,
):
    req = tl.program_id(0)
    s = tl.arange(0, NSPLIT)
    val = tl.load(part_val_ptr + req * NSPLIT + s)
    idx = tl.load(part_idx_ptr + req * NSPLIT + s)
    top = tl.max(val, axis=0)
    tl.store(
        out_ptr + req, tl.min(tl.where(val == top, idx, vocab), axis=0).to(tl.int64)
    )


def simulated_bonus_sample_graph(
    *,
    target_logits: torch.Tensor,
    correct_len: torch.Tensor,
    temperatures: torch.Tensor,
    seed: torch.Tensor,
    bs: int,
    rows_per_request: int,
    draft_logits: torch.Tensor | None = None,
    draft_temperatures: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gumbel-max temperature sample of each request's row at ``correct_len``; seed
    ([1] int) and per-request temperatures ([bs] float; <= 0 means argmax) are read
    from device buffers, so it is graph-safe. [bs] int64."""
    part_val, part_idx = simulated_bonus_partials_graph(
        target_logits=target_logits,
        correct_len=correct_len,
        temperatures=temperatures,
        seed=seed,
        bs=bs,
        rows_per_request=rows_per_request,
        draft_logits=draft_logits,
        draft_temperatures=draft_temperatures,
    )
    out = torch.empty((bs,), dtype=torch.int64, device=target_logits.device)
    _simulated_bonus_row_select_kernel[(bs,)](
        part_val, part_idx, out, target_logits.shape[1], NSPLIT=part_val.shape[1]
    )
    return out


def simulated_bonus_partials_graph(
    *,
    target_logits: torch.Tensor,
    correct_len: torch.Tensor,
    temperatures: torch.Tensor,
    seed: torch.Tensor,
    bs: int,
    rows_per_request: int,
    draft_logits: torch.Tensor | None = None,
    draft_temperatures: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """simulated_bonus_sample_graph's per-split (value, index) maxima, [bs, NSPLIT],
    reduced by ``simulated_bonus_select_partials``. ``correct_len`` may be stride-0."""
    assert target_logits.stride(1) == 1
    assert correct_len.dim() == 1 and correct_len.shape[0] == bs
    device = target_logits.device
    vocab = target_logits.shape[1]
    block = 4096
    nsplit = triton.next_power_of_2(triton.cdiv(vocab, block))
    part_val = torch.empty((bs, nsplit), dtype=torch.float32, device=device)
    part_idx = torch.empty((bs, nsplit), dtype=torch.int32, device=device)
    if draft_logits is not None:
        # SGLANG_SIMULATE_ACC_RESIDUAL_BONUS: draft_logits [bs, gamma, width] (width <= vocab),
        # the draft step logits that proposed each verify row.
        assert (
            draft_logits.stride(2) == 1
            and draft_logits.shape[1] == rows_per_request - 1
        )
        lse_part = torch.empty((bs, nsplit, 4), dtype=torch.float32, device=device)
        width = min(int(draft_logits.shape[2]), vocab)
        _residual_lse_partial_kernel[(bs, nsplit)](
            target_logits,
            target_logits.stride(0),
            temperatures,
            correct_len,
            correct_len.stride(0),
            draft_logits,
            draft_logits.stride(0),
            draft_logits.stride(1),
            draft_temperatures,
            lse_part,
            vocab,
            width,
            ROWS=rows_per_request,
            NSPLIT=nsplit,
            BLOCK=block,
            num_warps=8,
        )
        mass = torch.empty((bs, nsplit, 2), dtype=torch.float32, device=device)
        args = (
            target_logits,
            target_logits.stride(0),
            temperatures,
            correct_len,
            correct_len.stride(0),
        )
        draft_args = (
            draft_logits,
            draft_logits.stride(0),
            draft_logits.stride(1),
            draft_temperatures,
            lse_part,
            mass,
            part_val,
            part_idx,
            vocab,
            width,
        )
        meta = dict(ROWS=rows_per_request, NSPLIT=nsplit, BLOCK=block, num_warps=8)
        _residual_mass_kernel[(bs, nsplit)](*args, *draft_args, **meta)
        _residual_pick_kernel[(bs,)](*args, seed, *draft_args, **meta)
        return part_val, part_idx
    _simulated_bonus_row_partial_kernel[(bs, nsplit)](
        target_logits,
        target_logits.stride(0),
        temperatures,
        correct_len,
        correct_len.stride(0),
        seed,
        part_val,
        part_idx,
        vocab,
        ROWS=rows_per_request,
        NSPLIT=nsplit,
        BLOCK=block,
        num_warps=8,
    )
    return part_val, part_idx


def simulated_bonus_select_partials(
    *, part_val: torch.Tensor, part_idx: torch.Tensor, vocab: int
) -> torch.Tensor:
    """Torch reference of _simulated_bonus_row_select_kernel: the largest value,
    ties to the lowest index. [bs] int64."""
    top = part_val.max(dim=1, keepdim=True).values
    sentinel = torch.full_like(part_idx, vocab)
    return torch.where(part_val == top, part_idx, sentinel).min(dim=1).values.long()
