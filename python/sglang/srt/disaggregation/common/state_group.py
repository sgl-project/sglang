"""PD pairing when the linear-attention state group spans attention-DP groups.

RFC / partial implementation. Nothing here is wired into the receiver yet;
see ``CommonKVManager.try_ensure_parallel_info``, which refuses such peers.

Stock SGLang lays linear-attention (GDN / Mamba) state out over the attention
TP group, so the ranks that hold a request's state are the ranks that hold its
KV. Some deployments shard the state over a wider group than attention: e.g.
decode attention TP8 x DP16 with GDN TP64 x DP2, where one GDN state group is
``k = 64 / 8 = 8`` consecutive attention-DP groups. Then:

* every rank of the request's state group (not only its attention group) must
  hold a receiver for the room and register with the prefill;
* the prefill must count those extra decode ranks in ``required_dst_info_num``;
* each side must know the other's state TP size, and each pair its state TP
  ranks, to slice the state.

Rank numbering. Within one role, the global attention rank of
``(attn_dp_rank, attn_tp_rank)`` is ``attn_dp_rank * attn_tp + attn_tp_rank``.
State group ``g`` is global ranks ``[g * state_tp, (g + 1) * state_tp)``, i.e.
attention-DP groups ``[g * k, (g + 1) * k)``; a rank's state TP rank is its
global rank modulo ``state_tp``.

Edges for one room (prefill attention-DP group ``pd``, decode attention-DP
group ``dd``):

* KV: decode attention group ``dd`` <- prefill attention group ``pd``, paired
  by attention TP exactly as ``CommonKVManager._resolve_rank_mapping`` pairs
  non-MLA ranks today.
* State: decode state group ``dd // k_d`` <- prefill state group ``pd // k_p``,
  paired by *state* TP with the same rule.

A decode rank sends one metadata message per prefill rank it has an edge to
(KV and state edges to the same prefill rank are merged), so a prefill rank's
``required_dst_info_num`` is the number of decode ranks with an edge to it and
a decode rank's ``required_prefill_response_num`` is its number of senders.
With ``state_tp == attn_tp`` on both sides this reduces to today's mapping.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Set


@dataclasses.dataclass(frozen=True)
class StateGroupLayout:
    """One PD role's attention and linear-attention-state sharding."""

    attn_tp: int
    attn_dp: int
    # None: the state is sharded like attention (stock SGLang).
    state_tp: Optional[int] = None

    def __post_init__(self):
        if self.attn_tp <= 0 or self.attn_dp <= 0:
            raise ValueError(f"invalid attention layout {self}")
        st = self.state_tp_size
        if st % self.attn_tp or (self.attn_tp * self.attn_dp) % st:
            raise ValueError(
                f"state TP {st} must be a multiple of attention TP {self.attn_tp} "
                f"and divide attention TP x DP {self.attn_tp * self.attn_dp}"
            )

    @property
    def state_tp_size(self) -> int:
        return self.state_tp if self.state_tp is not None else self.attn_tp

    @property
    def dp_span(self) -> int:
        """Attention-DP groups per state group (1 for stock SGLang)."""
        return self.state_tp_size // self.attn_tp

    def global_rank(self, attn_dp_rank: int, attn_tp_rank: int) -> int:
        return attn_dp_rank * self.attn_tp + attn_tp_rank

    def attention_group(self, attn_dp_rank: int) -> List[int]:
        base = attn_dp_rank * self.attn_tp
        return list(range(base, base + self.attn_tp))

    def state_group_of(self, attn_dp_rank: int) -> List[int]:
        """Global ranks of the state group holding attention-DP group ``attn_dp_rank``."""
        base = (attn_dp_rank // self.dp_span) * self.state_tp_size
        return list(range(base, base + self.state_tp_size))

    def state_tp_rank(self, global_rank: int) -> int:
        return global_rank % self.state_tp_size


def tp_targets(src_tp: int, dst_tp: int, dst_rank: int) -> List[int]:
    """Source TP ranks a destination TP rank pairs with (non-MLA rule of
    ``CommonKVManager._resolve_rank_mapping``)."""
    if max(src_tp, dst_tp) % min(src_tp, dst_tp):
        raise ValueError(f"TP sizes must divide: src={src_tp}, dst={dst_tp}")
    dst_rank %= dst_tp
    if src_tp == dst_tp:
        return [dst_rank]
    if dst_tp > src_tp:
        return [dst_rank // (dst_tp // src_tp)]
    k = src_tp // dst_tp
    return list(range(dst_rank * k, (dst_rank + 1) * k))


@dataclasses.dataclass
class StateGroupRoomPlan:
    """Who registers / sends for one room. Ranks are global per role."""

    # decode rank -> prefill ranks it receives from (KV and state merged)
    senders: Dict[int, List[int]]
    # decode rank -> prefill ranks it receives KV from
    kv_senders: Dict[int, List[int]]
    # decode rank -> prefill ranks it receives state from
    state_senders: Dict[int, List[int]]
    # prefill rank -> number of decode ranks registering for the room there
    required_dst_info_num: Dict[int, int]

    @property
    def decode_participants(self) -> List[int]:
        return sorted(self.senders)

    @property
    def prefill_participants(self) -> List[int]:
        return sorted(self.required_dst_info_num)

    def required_prefill_response_num(self, decode_rank: int) -> int:
        return len(self.senders.get(decode_rank, ()))


def plan_state_group_room(
    prefill: StateGroupLayout,
    decode: StateGroupLayout,
    prefill_dp_rank: int,
    decode_dp_rank: int,
) -> StateGroupRoomPlan:
    """Edges for one room; identical on every rank that computes it."""
    p_attn = prefill.attention_group(prefill_dp_rank)
    p_state = prefill.state_group_of(prefill_dp_rank)
    d_attn = decode.attention_group(decode_dp_rank)
    d_state = decode.state_group_of(decode_dp_rank)

    kv_senders: Dict[int, List[int]] = {}
    for i, d in enumerate(d_attn):
        kv_senders[d] = [
            p_attn[t] for t in tp_targets(prefill.attn_tp, decode.attn_tp, i)
        ]

    state_senders: Dict[int, List[int]] = {}
    for i, d in enumerate(d_state):
        state_senders[d] = [
            p_state[t]
            for t in tp_targets(prefill.state_tp_size, decode.state_tp_size, i)
        ]

    senders: Dict[int, List[int]] = {}
    for d in sorted(set(kv_senders) | set(state_senders)):
        merged: Set[int] = set(kv_senders.get(d, ())) | set(state_senders.get(d, ()))
        senders[d] = sorted(merged)

    required: Dict[int, int] = {}
    for srcs in senders.values():
        for p in srcs:
            required[p] = required.get(p, 0) + 1

    return StateGroupRoomPlan(
        senders=senders,
        kv_senders=kv_senders,
        state_senders=state_senders,
        required_dst_info_num=required,
    )


def prefill_sender_id(
    layout: StateGroupLayout,
    attn_dp_rank: int,
    attn_tp_rank: int,
    pp_rank: int = 0,
    pp_size: int = 1,
    attn_cp_rank: int = 0,
    attn_cp_size: int = 1,
) -> int:
    """Per-sender id the decode counts responses by.

    Today's id (``CommonKVManager._prefill_unique_rank``) is unique within one
    attention-DP group. When the state group spans several, senders from
    different attention-DP groups answer the same room, so the attention-DP
    rank is folded in; with ``dp_span == 1`` the id is unchanged.
    """
    base = attn_tp_rank * (pp_size * attn_cp_size) + pp_rank * attn_cp_size
    base += attn_cp_rank
    if layout.dp_span == 1:
        return base
    per_dp = layout.attn_tp * pp_size * attn_cp_size
    return (attn_dp_rank % layout.dp_span) * per_dp + base
