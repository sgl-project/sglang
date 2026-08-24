"""Admission accounting for the HiCache HiSparse backing.

The backing hands a request's prefix to the radix tree and lets HiCache evict it
to host; the swap-in kernel then reads whichever tier holds a selected position.
That only works while every admitted prefix token keeps ONE home, so admission
has to be rationed -- and rationed *before* the prefill forward runs, because by
the time a request is offered to the coordinator its KV is already in the pool.

Integer bookkeeping over three quantities: expanded indexer pages (one per prefix
page, for the request's lifetime), host + usable device tokens (the "one home per
token" ceiling), and the temp device buffer each admitted request pins.

Split out of the coordinator because it is the half with no tensors in it -- the
ceiling arithmetic is where over-admission bugs live, and it is testable without a
GPU. See `managers/hisparse_hicache_coordinator.py` for the half that moves KV.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import msgspec

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

logger = logging.getLogger(__name__)

# Slack beyond the named reserves below, for the one thing no capacity term can
# express: admission is instantaneous, write-back is bandwidth-bound. One page per
# live request is the whole margin -- the reserves are upper bounds already, and
# retraction handles a genuine device shortfall.
_ALIGNMENT_SLACK_PAGES_PER_REQ = 1


class _NodeClaim(msgspec.Struct):
    """One tree node's residency claim, shared by every request referencing it.

    `tokens` is counted once per node however many requests hold it, so a shared
    prefix bills once; `host_locked` is sticky for the entry's lifetime and marks
    the node's tokens as pinned in host by an eviction taken on some request's
    behalf.
    """

    refs: int
    tokens: int
    host_locked: bool


class _ActiveRequest(msgspec.Struct):
    """One admitted request's residency accounting, established and dropped as a
    unit -- as parallel dicts, every exit path had to pop each one."""

    # Tree-owned page-aligned prefix, the span that must keep a home.
    tree_len: int
    # Output tokens still to come, device-only until write-back takes them.
    decode_reserve: int
    # Prefix positions already on host, no longer blocking a device eviction.
    evicted_positions: int = 0


class HiCacheAdmitBudget:
    """One `PrefillAdder` round's view of the admission quotas.

    PREDICTS admission so the adder can pick each candidate's device reservation;
    the authoritative accounting stays in `AdmissionLedger` and is re-snapshotted
    into every new adder. Within a round the quotas act as shadow copies, drawn
    down on commit.
    """

    def __init__(
        self,
        *,
        page_size: int,
        temp_slot_tokens: int,
        expanded_pages_left: int,
        host_tokens_left: int,
        device_pool_tokens: int,
        device_evictable_overhang: int,
        ledger: Optional[AdmissionLedger] = None,
    ):
        self._page_size = page_size
        self.temp_slot_tokens = temp_slot_tokens
        self._expanded_left = expanded_pages_left
        self._host_left = host_tokens_left
        self._device_pool_tokens = device_pool_tokens
        self.device_evictable_overhang = device_evictable_overhang
        # Committed reservations are published here so the next round's snapshot
        # sees them; this round's local _host_left already accounts for them.
        self._ledger = ledger

    def _infeasible(self, total_seq_len: int, max_new: int) -> bool:
        """True when the candidate's reservation cannot fit the device pool even
        at full idle, so it must be budgeted as a standard request.

        The entry-side max_new clamp only guarantees the STANDARD budget fits, so
        adding the temp buffer on top could push the adder's total past
        rem_total_tokens forever -- a livelock on an idle system. The coordinator
        may still admit such a request later; with the
        `tree_len >= temp_slot_tokens` gate its footprint never exceeds the
        standard reservation made here, so this stays sound.
        """
        return (
            total_seq_len + self.temp_slot_tokens + max_new + self._page_size
            >= self._device_pool_tokens
        )

    def _ineligible(self, total_seq_len: int, max_new: int) -> bool:
        """Reasons a candidate gains nothing from admission, quotas aside."""
        num_pages = total_seq_len // self._page_size
        return (
            # Nothing page-aligned to hand the tree, so nothing evictable.
            num_pages <= 0
            # A prefix smaller than the temp buffer makes admission a net
            # capacity LOSS: the buffer is pinned for the request's lifetime
            # while only tree_len becomes evictable.
            or num_pages * self._page_size < self.temp_slot_tokens
            or self._infeasible(total_seq_len, max_new)
        )

    def future_tokens(
        self, total_seq_len: int, max_new: int, commit: bool, req=None
    ) -> int:
        """Device-pool tokens to reserve for this candidate.

        Predicted admission -> `temp_slot_tokens + max_new` (the temp device
        buffer comes out of the regular pool at admission and is held for the
        request's lifetime); predicted rejection -> the standard `max_new`.
        """
        num_pages = total_seq_len // self._page_size
        tree_len = num_pages * self._page_size
        # In-flight candidates are billed their FULL footprint, with no credit for
        # an already-resident match, even though per-node claims DO dedup shared
        # prefixes once admitted. Crediting the match is accurate for capacity and
        # CRASHES: a re-hit burst admits ~2x as deep and write_back cannot drain
        # fast enough -- prefill alloc OOMs with the pool at 45%. The binding
        # constraint there is eviction RATE, which no ceiling expresses, so this
        # basis is the pacing margin. Do not tighten it without a rate limiter.
        if (
            self._ineligible(total_seq_len, max_new)
            or num_pages > self._expanded_left
            or tree_len > self._host_left
        ):
            return max_new
        if commit:
            self._expanded_left -= num_pages
            self._host_left -= tree_len
            if self._ledger is not None and req is not None:
                self._ledger.note_pending(req.rid, tree_len)
        return self.temp_slot_tokens + max_new

    def admission_exhausted(self, total_seq_len: int, max_new: int) -> bool:
        """True when only a depleted quota blocks admission.

        Such a candidate should stay queued rather than run as standard: the
        quota frees up when an admitted request finishes. A candidate that is
        ineligible on its own merits is not "exhausted" -- it runs as standard
        and nothing it waits for would change that.
        """
        if self._ineligible(total_seq_len, max_new):
            return False
        num_pages = total_seq_len // self._page_size
        # Same basis as future_tokens, or the gate would queue candidates that
        # future_tokens would happily admit.
        return (
            num_pages > self._expanded_left
            or num_pages * self._page_size > self._host_left
        )


class AdmissionLedger:
    """What admitted and in-flight requests have promised the two KV tiers.

    Two quantities with no overlap: admitted requests are charged their per-node
    claims (deduped, so a shared prefix bills once no matter how many requests
    reference it -- and a re-hit prefix still bills, which is what un-accounted
    let a re-hit burst do until its evictions pinned the host tier solid), and
    in-flight candidates are charged their pending footprint until admission
    resolves one way or the other.
    """

    def __init__(
        self,
        *,
        device_pool_tokens: int,
        temp_slot_tokens: int,
        page_size: int,
        chunk_tokens: int,
    ):
        self.temp_slot_tokens = temp_slot_tokens
        # Participates in the ceiling: see reservable_left.
        self._device_pool_tokens = device_pool_tokens
        self._page_size = page_size
        # Tokens one prefill step writes before its request can be admitted; 0
        # when chunked prefill is off, where the prompt is charged as pending.
        self._chunk_tokens = max(0, chunk_tokens)
        # Host tier size in tokens; 0 until the tree cache attaches.
        self._host_capacity_tokens = 0
        # req_pool_idx -> residency accounting of every admitted request.
        self.active_reqs: Dict[int, _ActiveRequest] = {}
        # rid -> tokens claimed at prefill-admission time but not yet resolved by
        # the coordinator, so a later adder round cannot re-promise the space: the
        # request's KV is in the pool but not yet on host. Summed on demand -- a
        # running total is derived state, and the way it can drift (a charge
        # dropped twice) raises the ceiling and over-admits.
        self._pending: Dict[str, int] = {}
        self._node_claims: Dict[int, _NodeClaim] = {}
        self._claimed_tokens = 0
        self._host_locked_tokens = 0

    # ---- attachment ---------------------------------------------------

    def set_host_capacity(self, host_capacity_tokens: int) -> None:
        self._host_capacity_tokens = host_capacity_tokens

    # ---- per-node claims ----------------------------------------------

    @staticmethod
    def _node_token_len(node) -> int:
        cd = node.component_data[ComponentType.FULL]
        if cd.value is not None:
            return len(cd.value)
        if cd.host_value is not None:
            return len(cd.host_value)
        return 0

    def claim_node(self, node, *, host_locked: bool = False) -> None:
        """Register one residency claim on a node.

        Two sources share this registry so they can never double-bill a node:
        admission (every matched-path node of an admitted request, for its
        lifetime) and eviction attribution (a node evicted to host on that
        request's behalf, which flips `host_locked`).
        """
        claim = self._node_claims.get(node.id)
        if claim is None:
            claim = _NodeClaim(
                refs=0, tokens=self._node_token_len(node), host_locked=False
            )
            self._node_claims[node.id] = claim
            self._claimed_tokens += claim.tokens
        claim.refs += 1
        if host_locked and not claim.host_locked:
            claim.host_locked = True
            self._host_locked_tokens += claim.tokens

    def release_node(self, node) -> None:
        claim = self._node_claims[node.id]
        claim.refs -= 1
        if claim.refs == 0:
            del self._node_claims[node.id]
            self._claimed_tokens -= claim.tokens
            if claim.host_locked:
                self._host_locked_tokens -= claim.tokens

    # ---- in-flight claims ---------------------------------------------

    def note_pending(self, rid: str, tokens: int) -> None:
        """Record a prefill-time claim so later adder rounds see it.

        The coordinator only records the real reservation after the prefill
        forward completes, so without this a round snapshots stale headroom and
        re-promises capacity an earlier round already claimed. Idempotent per
        request: chunked prefill re-budgets the same request across rounds.
        """
        if rid in self._pending:
            return
        self._pending[rid] = tokens

    def drop_pending(self, rid: str) -> None:
        """Release a request's prefill-time claim once admission resolves.

        Both outcomes release it: on success `activate` calls this itself, the
        charge having been converted into per-node claims; on rejection the
        request runs as a standard fallback and holds no host space. Callers
        settle the paths that reach neither -- an abort during prefill.
        """
        self._pending.pop(rid, None)

    # ---- admitted requests --------------------------------------------

    def activate(
        self, req_pool_idx: int, tree_len: int, *, rid: str, decode_reserve: int
    ) -> None:
        """Record a successful admission, superseding its in-flight charge.

        The pending charge and the per-node claims stand for the SAME tokens, so
        an admitted request must never carry both -- it bills its prefix twice and
        the ceiling runs out a whole request early. Dropped here rather than at
        the caller so "pending XOR active" is the ledger's invariant instead of
        call-site discipline.

        `decode_reserve` is held for the request's lifetime rather than decayed
        per step: tokens it already produced still occupy pool pages, so releasing
        the reserve as they land would just stop counting them.
        """
        self.drop_pending(rid)
        self.active_reqs[req_pool_idx] = _ActiveRequest(
            tree_len=tree_len, decode_reserve=decode_reserve
        )

    def deactivate(self, req_pool_idx: int) -> None:
        self.active_reqs.pop(req_pool_idx, None)

    def note_evicted_positions(self, req_pool_idx: int, count: int) -> None:
        self.active_reqs[req_pool_idx].evicted_positions += count

    # ---- ceilings ------------------------------------------------------

    def reservable_left(self) -> int:
        """Tokens still reservable for new admissions, across both tiers.

        A prefix token needs exactly ONE home -- device or host -- because the
        swap-in kernel reads whichever holds it. Demanding a host slot for tokens
        sitting on device double-books the device tier and caps admission at the
        host size alone, which left the pool ~84% idle while admission stalled.

        Reservation-based, not measured availability: HiCache keeps written-back KV
        as reusable cache, so measured availability stays low even though eviction
        can reclaim it on demand, and budgeting against it starves admission.
        Safety comes from eviction time instead -- the copy-less-drop veto keeps
        data backing an active request on device, and device pressure resolves by
        retraction.
        """
        if self._host_capacity_tokens <= 0:
            # No host tier attached yet (the tree cache brings it). Nowhere to
            # evict to means this ceiling is not what protects the pool, so it
            # must not bind; 0 would read as "quota exhausted" and stop prefill.
            # An int, not inf: the round's shadow quota is decremented from it.
            return 1 << 60
        return (
            self._host_capacity_tokens
            + self._usable_device_tokens()
            - self._claimed_tokens
            - sum(self._pending.values())
        )

    def _usable_device_tokens(self) -> int:
        """Device tokens available to hold resident prefixes.

        A device token counts toward "every admitted prefix token keeps one home"
        only if it will still be free when the prefix needs it, so everything with
        a prior claim comes off first. Each term is an upper bound on something
        host CANNOT stand in for: one temp buffer per live request plus one for
        this round's admission (pinned before the request enters `active_reqs`),
        the decode tail every admitted request may still append, the chunk being
        prefilled right now, and one page per request for alignment.
        """
        if self._device_pool_tokens <= 0:
            return 0
        live_reqs = len(self.active_reqs) + 1
        reserve = (
            live_reqs * self.temp_slot_tokens
            + sum(a.decode_reserve for a in self.active_reqs.values())
            + self._chunk_tokens
            + live_reqs * _ALIGNMENT_SLACK_PAGES_PER_REQ * self._page_size
        )
        return max(0, self._device_pool_tokens - reserve)

    def device_evictable_overhang(self, tree_evictable_tokens: int) -> int:
        """Device-evictable tokens the `PrefillAdder` must NOT budget against.

        Under write_back a demotion needs host space for the backup; without it
        the copy-less drop is vetoed for data backing active requests and the node
        stays device-resident. Whatever exceeds what host can absorb is therefore
        not reclaimable, and counting it in rem_total_tokens over-admits until
        prefill allocation hard-fails.

        Only tokens BACKING ACTIVE REQUESTS are blocked -- the veto does not
        protect unrelated cache nodes. The blocked set is bounded by each active
        request's still-device-resident positions, clamped by the tree-wide
        evictable size, plus slack for backups landing after this snapshot.

        A deliberate upper bound, not the exact veto-blocked set: host copies are
        not pinned until eviction time, so a span that looks host-backed can have
        its copy reclaimed by the host LRU before the demotion that needs it.
        Deducting such spans was tried and reproducibly ended in prefill-alloc OOM
        under a re-hit burst. Over-estimation only throttles prefill earlier; it
        never corrupts.
        """
        if self._host_capacity_tokens <= 0 or not self.active_reqs:
            return 0
        active_backing = sum(
            a.tree_len - a.evicted_positions for a in self.active_reqs.values()
        )
        blocked = min(active_backing, tree_evictable_tokens)
        slack = self.temp_slot_tokens * (len(self.active_reqs) + 1)
        absorbable = self._host_capacity_tokens - self._host_locked_tokens - slack
        return max(0, blocked - max(0, absorbable))

    def make_budget(
        self,
        *,
        expanded_pages_left: int,
        tree_evictable_tokens: int,
    ) -> HiCacheAdmitBudget:
        """Snapshot the quotas for one `PrefillAdder` round.

        Page and pool sizes come from the ledger's own fields: taking them per
        round let `_infeasible`'s threshold disagree with the ceiling it is
        compared against.
        """
        return HiCacheAdmitBudget(
            page_size=self._page_size,
            temp_slot_tokens=self.temp_slot_tokens,
            expanded_pages_left=expanded_pages_left,
            host_tokens_left=self.reservable_left(),
            device_pool_tokens=self._device_pool_tokens,
            device_evictable_overhang=self.device_evictable_overhang(
                tree_evictable_tokens
            ),
            ledger=self,
        )
