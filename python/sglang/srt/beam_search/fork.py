"""Fork primitives for beam member rows (copy-on-fork KV).

Members are not requests: each is one physical req_to_token row spawned
decode-ready after the leader's prefill -- no member prefill; its first
decode step computes the selected token's KV like normal decode.

- alias: members share the leader's prompt KV mapping read-only; each member
  owns only its decode suffix, which standard alloc_for_decode extends.
- reparent: all beams in a group share the same length, so reparenting is a
  pure KV **data** copy onto the destination's own slots: no allocator
  traffic, no req_to_token remapping, stream-safe and capturable.
- free: member rows own exactly their decode-suffix slots
  [prompt_len, leader_allocated); the aliased prompt belongs to the leader's
  accounting and is never freed here.
"""

from __future__ import annotations

from typing import Sequence

import torch

# Members outlive the group's own length checks by a margin so that row-side
# length limits can never truncate before the coordinator's deterministic
# advance_final.
MEMBER_LENGTH_MARGIN = 4


def neutral_member_sampling_params(leader_params):
    """Neutral params: raw logprob scoring, and no leader-side finish path.

    The leader row never self-finishes (the coordinator owns all stop/length
    semantics), so stop conditions are stripped and ignore_eos is forced.
    """
    from sglang.srt.sampling.sampling_params import SamplingParams

    return SamplingParams(
        max_new_tokens=(leader_params.max_new_tokens or 0) + MEMBER_LENGTH_MARGIN,
        temperature=1.0,
        top_p=1.0,
        min_p=0.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        min_new_tokens=0,
        n=1,
        ignore_eos=True,
        skip_special_tokens=leader_params.skip_special_tokens,
        spaces_between_special_tokens=leader_params.spaces_between_special_tokens,
    )


def alias_members_prompt_kv(
    req_to_token: torch.Tensor,
    dst_rows: torch.Tensor,
    leader_row: int,
    prompt_len: int,
) -> None:
    """Alias the leader's prompt KV mapping onto every member row in one
    batched assignment (dst_rows all share the same leader_row + prompt_len).

    The prompt KV indices are aliased read-only from the leader's row; each
    member owns only its decode suffix, which standard alloc_for_decode extends
    from here. Any tree lock on a matched prompt prefix is held by the leader
    for the whole group's lifetime; member rows never touch the tree. Batching
    all members of a group into one indexed copy avoids a per-member kernel
    launch.
    """
    req_to_token[dst_rows, :prompt_len] = req_to_token[leader_row, :prompt_len]


def free_member_rows(group, req_to_token_pool, token_to_kv_pool_allocator) -> None:
    """Release a group's member rows: their decode-suffix KV slots
    [prompt_len, leader_allocated) plus the row slots themselves. Idempotent.

    Must run while the leader still holds its kv info (leader_allocated is the
    lockstep allocated length of every member row, including any overlap
    overshoot slot). The aliased prompt mapping is the leader's to free.
    """
    if group.member_rows is None:
        return
    leader = group.leader
    start = group.prompt_len
    end = leader.kv.kv_allocated_len if leader.kv is not None else start
    if end > start:
        slots = req_to_token_pool.req_to_token[group.member_rows, start:end]
        token_to_kv_pool_allocator.free(slots.flatten())
    req_to_token_pool.free_raw(group.member_rows_cpu.tolist())
    group.member_rows = None
    group.member_rows_cpu = None
    group.all_rows = None


# ==================== reparent ====================


def reparent_kv(
    req_to_token: torch.Tensor,
    kv_buffers: Sequence[torch.Tensor],
    dst_rows: torch.Tensor,
    src_rows: torch.Tensor,
    prefix_len: int,
    seq_len: int,
) -> None:
    """Copy the decode-suffix KV data of src rows onto dst rows.

    Synchronized lengths make this a pure data move: dst keeps its own KV
    slots and its req_to_token mapping; only buffer contents change. All
    index math is tensor-side (no host sync), so the whole call can be
    enqueued in-stream and captured. Rows with parent == self must simply be
    omitted from dst_rows/src_rows.
    """
    src_slots = req_to_token[src_rows, prefix_len:seq_len].reshape(-1)
    dst_slots = req_to_token[dst_rows, prefix_len:seq_len].reshape(-1)
    for buf in kv_buffers:
        buf[dst_slots] = buf[src_slots]
