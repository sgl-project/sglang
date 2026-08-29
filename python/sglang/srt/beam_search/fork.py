# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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

from typing import NamedTuple

import torch


class StagedOrphans(NamedTuple):
    """A remap's before/after mapping, awaiting the deferred set difference."""

    tick: int
    old_mapping: torch.Tensor
    new_mapping: torch.Tensor


# Margin so row-side length limits can never truncate before the coordinator's
# deterministic advance_final.
MEMBER_LENGTH_MARGIN = 4


def neutral_member_sampling_params(leader_params):
    """Raw logprob scoring, and no leader-side finish path: the coordinator owns
    all stop/length semantics, so stops are stripped and ignore_eos forced."""
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
    """One indexed copy for all members, which share leader_row + prompt_len.
    Any tree lock stays the leader's; members never touch the tree."""
    req_to_token[dst_rows, :prompt_len] = req_to_token[leader_row, :prompt_len]


def free_member_rows(group, req_to_token_pool, token_to_kv_pool_allocator) -> None:
    """Release the decode-suffix KV plus the member row slots. Idempotent; must
    run while the leader's kv info still carries the lockstep allocated length."""
    if group.member_rows is None:
        return
    leader = group.leader
    start = group.prompt_len
    end = leader.kv.kv_allocated_len
    if end > start:
        # The rewind below is required: without it the leader's own per-Req
        # release frees this decode region a second time.
        slots = req_to_token_pool.req_to_token[group.all_rows, start:end]
        token_to_kv_pool_allocator.free(slots.flatten().unique())
        leader.kv_committed_len = start
        leader.kv.kv_allocated_len = start
    req_to_token_pool.free_rows(group.member_rows_cpu.tolist())
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
    """Returns (old_mapping, new_mapping) so the caller can reclaim the slots no
    surviving row references any more."""
    # Rows are length-synchronized, so a survivor's history is exactly its
    # parent's window, including the token just computed at seq_len-1.
    window = req_to_token[rows, prefix_len:seq_len]
    old_mapping = window.clone()
    new_mapping = old_mapping[parent_idx]
    req_to_token[rows, prefix_len:seq_len] = new_mapping
    return old_mapping, new_mapping


def collect_orphan_slots(old_mapping: torch.Tensor, new_mapping: torch.Tensor):
    """Slots referenced before the remap and by nobody after it. Data-dependent
    shape (unique/isin), so it synchronizes -- keep it off the launch path."""
    old_slots = old_mapping.flatten().unique()
    new_slots = new_mapping.flatten().unique()
    return old_slots[~torch.isin(old_slots, new_slots)]
