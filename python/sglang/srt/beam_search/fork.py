"""Fork primitives for beam member rows (share-on-fork KV).

Members are not requests: each is one physical req_to_token row spawned
decode-ready after the leader's prefill -- no member prefill; its first
decode step computes the selected token's KV like normal decode.

- alias: members share the leader's prompt KV mapping read-only; each member
  owns only its decode suffix, which standard alloc_for_decode extends.
- reparent: a survivor's history IS its parent's, so reparenting only remaps
  req_to_token onto the parent's slots -- no KV data copy. Slots nobody
  inherits are reclaimed separately (collect_orphan_slots), off the launch
  path because that set difference has a data-dependent shape.
- free: sharing means several rows can name one slot, so the decode region is
  owned by the GROUP and released once, deduped; the aliased prompt stays the
  leader's.
"""

from __future__ import annotations

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
    """One indexed copy for all members (they share leader_row + prompt_len,
    so batching saves a kernel launch per member).

    Any tree lock on a matched prompt prefix is the leader's for the group's
    whole lifetime; member rows never touch the tree.
    """
    req_to_token[dst_rows, :prompt_len] = req_to_token[leader_row, :prompt_len]


def free_member_rows(group, req_to_token_pool, token_to_kv_pool_allocator) -> None:
    """Release the decode-suffix KV [prompt_len, leader_allocated) plus the
    member row slots. Idempotent.

    Must run while the leader still holds its kv info: leader_allocated is the
    lockstep allocated length of every member row, overlap overshoot slot
    included.
    """
    if group.member_rows is None:
        return
    leader = group.leader
    start = group.prompt_len
    end = leader.kv.kv_allocated_len if leader.kv is not None else start
    if end > start:
        # Share-on-fork lets several rows (incl. the leader's) point at one
        # slot, so the GROUP owns the decode region and frees it once, deduped.
        # Rewinding the leader's kv lengths then leaves its per-Req release
        # path exactly the region still its own -- [0, prompt_len), the aliased
        # prompt. Without the rewind it frees the decode region a second time.
        rows = group.all_rows if group.all_rows is not None else group.member_rows
        slots = req_to_token_pool.req_to_token[rows, start:end]
        token_to_kv_pool_allocator.free(slots.flatten().unique())
        if leader.kv is not None:
            leader.kv_committed_len = start
            leader.kv.kv_allocated_len = start
    req_to_token_pool.free_raw(group.member_rows_cpu.tolist())
    group.member_rows = None
    group.member_rows_cpu = None
    group.all_rows = None


def remap_kv_mapping(
    req_to_token: torch.Tensor,
    rows: torch.Tensor,
    parent_idx: torch.Tensor,
    prefix_len: int,
    seq_len: int,
):
    """Returns (old_mapping, new_mapping) so the caller can reclaim the slots
    no surviving row references any more.

    All rows are length-synchronized, so a survivor's history is exactly its
    parent's window [prefix_len, seq_len) -- including the token just computed
    at seq_len-1. The row's own slot there becomes garbage unless some other
    survivor inherits it.
    """
    window = req_to_token[rows, prefix_len:seq_len]
    old_mapping = window.clone()
    new_mapping = old_mapping[parent_idx]
    req_to_token[rows, prefix_len:seq_len] = new_mapping
    return old_mapping, new_mapping


def collect_orphan_slots(old_mapping: torch.Tensor, new_mapping: torch.Tensor):
    """Slots referenced before the remap and by nobody after it.

    Data-dependent output shape (unique/isin), so this synchronizes -- callers
    must keep it off the launch path.
    """
    old_slots = old_mapping.flatten().unique()
    new_slots = new_mapping.flatten().unique()
    return old_slots[~torch.isin(old_slots, new_slots)]
