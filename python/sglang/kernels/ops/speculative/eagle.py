import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_cpu, next_power_of_2

_is_cpu = is_cpu()

if _is_cpu:
    from sgl_kernel import fill_accept_out_cache_loc_cpu, fill_bonus_tokens_cpu


@triton.jit
def fill_bonus_tokens(
    accept_tokens,
    accept_lens,
    bonus_tokens_ptr,
    accept_stride: tl.constexpr,
):
    # NOTE: we cannot fuse any in-place operations of `accept_lens` inside this kernel
    # because this kernel reads accept_lens
    pid = tl.program_id(axis=0)
    # `accept_lens` includes the bonus token; the last accepted slot is at -1.
    accept_len = tl.load(accept_lens + pid)

    # accept_stride = per-req width of accept_tokens (= accept_index.shape[1]).
    bonus_token_idx = accept_stride * pid + accept_len - 1
    bonus_token = tl.load(accept_tokens + bonus_token_idx)
    tl.store(bonus_tokens_ptr + pid, bonus_token)


def fill_bonus_tokens_func(
    accept_tokens: torch.Tensor,
    accept_lens: torch.Tensor,
    bonus_tokens: torch.Tensor,  # mutable
    accept_stride: int,
    batch_size: int,
):
    if _is_cpu:
        fill_bonus_tokens_cpu(
            accept_tokens,
            accept_lens,
            bonus_tokens,
            accept_stride,
        )
        return
    fill_bonus_tokens[(batch_size,)](
        accept_tokens,
        accept_lens,
        bonus_tokens,
        accept_stride,
    )


@triton.jit
def fill_accept_out_cache_loc(
    accept_index,
    out_cache_loc,
    accept_out_cache_loc,
    size_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = tl.arange(0, size_upper)

    masks = (tl.load(accept_index + offset, offset < pid, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks)
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accept_out_cache_loc + dst, value)


def fill_accept_out_cache_loc_func(
    accept_index: torch.Tensor,
    out_cache_loc: torch.Tensor,
    accept_out_cache_loc: torch.Tensor,  # mutable
    size: int,
):
    if _is_cpu:
        fill_accept_out_cache_loc_cpu(
            accept_index,
            out_cache_loc,
            accept_out_cache_loc,
        )
        return
    fill_accept_out_cache_loc[(size,)](
        accept_index,
        out_cache_loc,
        accept_out_cache_loc,
        next_power_of_2(size),
    )


@triton.jit
def nextn_mamba_commit_prologue(
    accept_lens_ptr,
    seq_lens_ptr,
    last_correct_ptr,
    steps_to_track_ptr,
    mamba_track_interval,
    HAS_TRACK: tl.constexpr,
):
    # In topk=1 chains, accepted node indices equal per-request step indices, so the eager accept_index gathers are identities.
    pid = tl.program_id(axis=0)
    al = tl.load(accept_lens_ptr + pid).to(tl.int64)
    # Eager accept_index gathers yield int32; draft-step bounds make this narrowing exact.
    tl.store(last_correct_ptr + pid, (al - 1).to(tl.int32))
    if HAS_TRACK:
        pre = tl.load(seq_lens_ptr + pid).to(tl.int64)
        post = pre + al
        interval = mamba_track_interval.to(tl.int64)
        crossed = (pre // interval) != (post // interval)
        tracking_point = (post // interval) * interval
        ith = tl.maximum(tracking_point - pre - 1, 0)
        out = tl.where(crossed, ith, -1)
        tl.store(steps_to_track_ptr + pid, out.to(tl.int32))


def nextn_mamba_commit_prologue_func(
    accept_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    mamba_track_interval: int,
    has_track: bool,
):
    """Return int32 chain-step indices equivalent to the eager accept_index gathers."""
    bs = accept_lens.shape[0]
    last_correct = torch.empty(bs, dtype=torch.int32, device=accept_lens.device)
    steps_to_track = (
        torch.empty(bs, dtype=torch.int32, device=accept_lens.device)
        if has_track
        else last_correct  # unused dummy write target is never launched
    )
    nextn_mamba_commit_prologue[(bs,)](
        accept_lens,
        seq_lens,
        last_correct,
        steps_to_track,
        mamba_track_interval,
        HAS_TRACK=has_track,
    )
    return last_correct, (steps_to_track if has_track else None)
