# Copyright 2023-2026 SGLang Team
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
"""The chain: the `MultiEndedKVPool`s that share one `UnifiedKVPool` buffer."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import torch
from torch.profiler import record_function

from sglang.srt.mem_cache.allocator.unified_sub_pool import (
    FloatMultiEndedKVPool,
    MultiEndedKVPool,
    _chain_byte_accounting_violations,
    _float_open_short_side,
)
from sglang.srt.runtime_context import get_parallel


class UnifiedChain:
    """The members of one shared byte buffer, ordered low -> high address, and
    everything that is about the shared bytes rather than about one component:
    the joint capacity view, the shortfall ladder, the lazy-compaction lifecycle
    fan-out and the byte-accounting audit.

    ``token_members`` are the pools a token allocation draws one page from each
    (full, swa); a ``state_member`` (mamba) is allocated per request instead.
    """

    def __init__(
        self,
        members: Sequence[MultiEndedKVPool],
        *,
        token_members: Sequence[MultiEndedKVPool],
        state_member: Optional[MultiEndedKVPool] = None,
        lazy_compaction: bool,
    ):
        self.members = tuple(members)
        self.token_members = tuple(token_members)
        self.state_member = state_member
        self.lazy_compaction = lazy_compaction
        for low, high in zip(self.members, self.members[1:]):
            low.bind_high_peer(high)
            high.bind_low_peer(low)
        floats = [m for m in self.members if isinstance(m, FloatMultiEndedKVPool)]
        assert len(floats) <= 1, "a chain has at most one float middle"
        self.float: Optional[FloatMultiEndedKVPool] = floats[0] if floats else None
        # Epoch-keyed memo for the joint view; any member's mutation moves the
        # chain epoch (see `MultiEndedKVPool._chain_capacity_epoch`).
        self._joint_avail_memo_epoch: Optional[int] = None
        self._joint_avail_memo_tokens: int = 0

    def _epoch(self) -> int:
        return self.members[0]._chain_capacity_epoch()

    # -- joint capacity --

    def joint_available_tokens(self) -> int:
        """Tokens a composite `alloc(N)` may take: N pages on EVERY token member."""
        epoch = self._epoch()
        if self._joint_avail_memo_epoch != epoch:
            self._joint_avail_memo_tokens = self._compute_joint_available_tokens()
            self._joint_avail_memo_epoch = epoch
        return self._joint_avail_memo_tokens

    def _compute_joint_available_tokens(self) -> int:
        if len(self.token_members) == 1:
            return self.token_members[0].available_size()
        assert len(self.token_members) == 2, "joint view is defined for a pair"
        if self.float is None:
            return self._joint_two_ends()
        return self._joint_with_float()

    def _joint_two_ends(self) -> int:
        """Two END pools facing each other across one shared gap."""
        fa, sa = self.token_members
        page_size = fa.page_size
        e_f = fa.entry_bytes_per_page
        e_s = sa.entry_bytes_per_page
        # Direction-agnostic shared gap: the free byte band between the two pools.
        if fa.grow_direction == "up":
            gap_bytes = max(0, sa._byte_low_frontier() - fa._byte_high_frontier())
        else:
            gap_bytes = max(0, fa._byte_low_frontier() - sa._byte_high_frontier())
        R_f = fa.num_pages - fa.min_page_index - fa._allocated_pages()
        R_s = sa.num_pages - sa.min_page_index - sa._allocated_pages()

        if not self.lazy_compaction:
            pages_by_bytes = gap_bytes // (e_f + e_s)
            return min(pages_by_bytes, R_f, R_s) * page_size

        H_f = len(fa._free_phys_pages)
        H_s = len(sa._free_phys_pages)

        K1 = min(H_f, H_s)  # Phase 1: both drain

        # Phase 2: fewer-holes side extends; more-holes side keeps draining.
        if H_f <= H_s:
            e_phase2 = e_f
            K_phase2_max = H_s
        else:
            e_phase2 = e_s
            K_phase2_max = H_f
        K2_room = K_phase2_max - K1
        K2 = min(K2_room, gap_bytes // e_phase2) if e_phase2 > 0 else K2_room
        gap_bytes -= K2 * e_phase2

        K3 = gap_bytes // (e_f + e_s)  # Phase 3: both extend

        K_total = K1 + K2 + K3
        K_total = min(K_total, H_f + R_f, H_s + R_s)  # index-space caps
        return K_total * page_size

    def _joint_with_float(self) -> int:
        """END full + FLOAT swa: N costs N full pages AND N swa pages, drawn from
        DIFFERENT bands -- full extends only into the high band, the float into
        either side but only ONE per batch alloc. Feasibility is monotone in N, so
        binary search; the order matches the alloc path (full takes the high band).
        """
        fa = next(m for m in self.token_members if m is not self.float)
        sa = self.float
        page_size = fa.page_size
        e_f = fa.entry_bytes_per_page
        # full is grow-down: its chain gap IS the high band.
        b_high = fa._current_gap_bytes()
        h_f = len(fa._free_phys_pages) if fa.lazy_compaction else 0
        h_s = sa._hole_pages()
        r_f = fa.num_pages - fa.min_page_index - fa._allocated_pages()
        r_s = sa.num_pages - sa.min_page_index - sa._allocated_pages()

        def feasible(n: int) -> bool:
            if n > h_f + r_f or n > h_s + r_s:
                return False
            ext_f = max(0, n - h_f)
            if ext_f * e_f > b_high:
                return False
            ext_s = max(0, n - h_s)
            # On the float's page grid, never in raw bytes: a byte budget
            # credits a page `take_physical_pages` cannot yield.
            full_low_after = fa._byte_low_frontier() - ext_f * e_f
            if sa._is_frontier_transparent():
                room = sa.pages_in_band(
                    low_byte=sa._chain_high_frontier_below_bytes(),
                    high_byte=full_low_after,
                )
                return ext_s <= room
            p_low = sa.pages_in_band(
                low_byte=sa._chain_high_frontier_below_bytes(),
                high_byte=sa._byte_low_frontier(),
            )
            p_high = sa.pages_in_band(
                low_byte=sa._byte_high_frontier(),
                high_byte=full_low_after,
            )
            return ext_s <= max(p_low, p_high)

        lo_n, hi_n = 0, min(h_f + r_f, h_s + r_s)
        while lo_n < hi_n:
            mid = (lo_n + hi_n + 1) // 2
            if feasible(mid):
                lo_n = mid
            else:
                hi_n = mid - 1
        return lo_n * page_size

    # -- shortfall ladder --

    def relieve(self, need_tokens: int) -> bool:
        """The composite's shortfall ladder: flush every member (a one-sided hole
        is unusable, compacting it yields SHARED gap), then ask the float to open
        the short side. Same steps as `_relieve_for_alloc` for one band."""
        for m in self._flush_targets():
            m._flush(urgent=True)
        if need_tokens <= self.joint_available_tokens():
            return True
        self._ask_float_for_room(need_tokens)
        return need_tokens <= self.joint_available_tokens()

    def _flush_targets(self) -> tuple:
        """Float FIRST: its zero-copy boundary absorption must land before the
        deficit math prices a relocation it already covered."""
        rest = [m for m in self.token_members if m is not self.float]
        if self.state_member is not None:
            rest.append(self.state_member)
        return tuple(([self.float] if self.float is not None else []) + rest)

    def _alloc_demand(self, need_tokens: int):
        """Demand VECTOR for one composite allocation, in PAGES per band. A token
        never draws a state slot, so a state member is an explicit 0."""
        page_size = self.token_members[0].page_size
        need_n = -(-need_tokens // page_size)
        demand = {m: need_n for m in self.token_members}
        if self.state_member is not None:
            demand[self.state_member] = 0
        return demand

    def _ask_float_for_room(self, need_tokens: int) -> None:
        if self.float is None:
            return  # two ENDs: nothing can slide
        _float_open_short_side(self.float, self._alloc_demand(need_tokens))

    # -- lazy compaction lifecycle --

    def set_latest_forward_done_event(self, event: Optional[torch.cuda.Event]) -> None:
        with record_function("UnifiedChain.set_latest_forward_done_event"):
            for m in self.members:
                m.set_latest_forward_done_event(event)

    def set_inflight_forward(
        self,
        forward_done: torch.cuda.Event,
        out_cache_loc_virtual: Optional[torch.Tensor],
    ) -> None:
        """Token members materialize their write-set from the forward's virtual
        `out_cache_loc`; a state member is written by its own kernels, so it
        gets None."""
        with record_function("UnifiedChain.set_inflight_forward"):
            for m in self.token_members:
                m.set_inflight_forward(forward_done, out_cache_loc_virtual)
            if self.state_member is not None:
                self.state_member.set_inflight_forward(forward_done, None)

    def flush_opportunistic(self) -> int:
        """Non-urgent flush of every member; sync-free."""
        with record_function("UnifiedChain.flush_opportunistic"):
            if all(
                m._free_phys_pages.numel() == 0 and not m._pending_reuse
                for m in self.members
            ):
                return 0
            return sum(m.flush_opportunistic() for m in self.members)

    def set_disagg_move_gate(self, gate: Callable[[], bool]) -> None:
        """Install the PD-disaggregation move gate on every member."""
        assert self.lazy_compaction, (
            "PD disaggregation with the unified memory pool requires lazy "
            "compaction (eager free-path compaction moves pages under "
            "in-flight transfers)."
        )
        for m in self.members:
            m.disagg_move_gate = gate

    # -- byte accounting --

    def mamba_slot_full_token_cost(self) -> int:
        """Full-token-equivalents one state slot removes from the shared buffer:
        a token costs the sum of the token members' entry bytes, rounded UP. The
        `dcp_size` factor is there because that budget is in widened tokens, one
        of which is `entry_bytes / dcp_size` local bytes."""
        assert self.state_member is not None, "chain has no state member"
        e_tok = sum(m.entry_bytes for m in self.token_members)
        return -(
            -self.state_member.entry_bytes_per_page
            * get_parallel().attn_dcp_size
            // e_tok
        )

    def verify_byte_accounting(self) -> List[str]:
        return (
            _chain_byte_accounting_violations(list(self.members))
            + self._joint_capacity_memo_violations()
        )

    def _joint_capacity_memo_violations(self) -> List[str]:
        """Idle-time twin of `MultiEndedKVPool._capacity_memo_violations` for
        the joint view. Empty == healthy."""
        if len(self.token_members) < 2 or self._joint_avail_memo_epoch != self._epoch():
            return []
        actual = self._compute_joint_available_tokens()
        if self._joint_avail_memo_tokens == actual:
            return []
        return [
            f"[joint] stale available_size memo: "
            f"cached={self._joint_avail_memo_tokens}, actual={actual}"
        ]
