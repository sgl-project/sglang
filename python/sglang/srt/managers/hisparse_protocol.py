"""The interface every HiSparse coordinator implements, whatever backs it.

HiSparse keeps the indexer KV GPU-resident over an expanded region, holds a
small hot attention working set per request, and fetches the rest per decode
step by top-k swap-in. Only the *logical KV pool* -- where a token's attention
KV lives while it is not in that hot set -- differs between deployments; see
`mem_cache/sparsity/factory.py` for the two backings and how one is chosen.

Everything in this module is the part both backings must answer, so scheduler,
model runner and attention backends can drive HiSparse without knowing which
one they got. Paths only one backing has (PD-disaggregation direct-to-host,
DeepSeek V4, shared-index prefetch) deliberately stay off the protocol: they
key on `coordinator.backing` and talk to the concrete class, so this interface
does not grow a member with exactly one real implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, NamedTuple, Optional, Protocol

if TYPE_CHECKING:
    import torch

    from sglang.srt.managers.hisparse_hicache_admission import HiCacheAdmitBudget
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.sparsity.factory import HiSparseBacking
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode


class HiSparseTokenStats(NamedTuple):
    """Device / host occupancy of the logical KV pool, for pool-stats logging."""

    device_tokens: int
    device_token_usage: float
    host_tokens: int
    host_token_usage: float


class HiSparseEvictionHooks(NamedTuple):
    """What the tree owes HiSparse when it moves a node's KV, and nothing more.

    The other direction of the seam from `HiSparseCoordinator` below: what the
    cache layer calls on HiSparse rather than what the scheduler calls. The tree
    core only annotates against this, so it imports it under TYPE_CHECKING and
    nothing in `mem_cache` gains a runtime dependency on `managers`.

    The HiCache HiSparse backing lets the tree evict the attention KV of a request
    that is still decoding, which no other feature does. Two things therefore have
    to cross from the tree to the coordinator, and they are two callables rather
    than a coordinator reference so the tree can only do these two. Registered as
    one value so the pair cannot be half-installed: with the veto missing, a
    copy-less drop would strand a live request's positions silently.

    `on_device_released` gets the node whose device KV is about to be freed, while
    `component_data[FULL].value` still holds its device indices and `.host_value`
    its host indices (None when it is being dropped without a host copy). It must
    not write to the GPU: it runs on the scheduler thread, possibly with a forward
    in flight. `backs_live_request` answers whether a node's device KV is part of a
    live HiSparse request's prefix.

    The private-host backing registers neither: its staging owns the only copy.
    """

    on_device_released: Callable[[UnifiedTreeNode], None]
    backs_live_request: Callable[[UnifiedTreeNode], bool]


class HiSparseCoordinator(Protocol):
    """Owns HiSparse's per-request KV residency for one backing."""

    # ---- identity -----------------------------------------------------
    backing: HiSparseBacking

    # ---- data plane ---------------------------------------------------
    # Number of real (non-padded) requests in the batch, as a device scalar the
    # swap-in kernels read at graph-replay time so padded blocks early-return.
    num_real_reqs: torch.Tensor

    def indexer_page_table(
        self, *, req_pool_indices: torch.Tensor, num_pages: int
    ) -> Optional[torch.Tensor]:
        """The page table the indexer must score against, or None.

        None means the standard `req_to_token`-derived table is already correct
        -- true whenever the logical token index space *is* the indexer space.
        A backing that lets the tree evict attention KV out from under a live
        request returns a hybrid table instead, resolving evicted prefix pages
        to the request's private expanded-region indexer pages.

        The result is only valid against the indexer buffer: expanded page ids
        are outside the attention KV buffer's range. Attention resolves its own
        positions through `swap_in_selected_pages`.
        """
        ...

    def translate_page_table(self, page_table: torch.Tensor) -> torch.Tensor:
        """Map logical page ids to attention-KV page ids for the sparse kernels.

        Identity for a backing whose logical ids already address the attention
        pool; the private-host backing resolves its extra indirection here.
        """
        ...

    def swap_in_selected_pages(
        self,
        *,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Make one layer's top-k KV device-resident; return its slot table.

        Every returned entry is either an already-resident slot (zero copy) or a
        slot this call DMAs into; -1 marks a position with no copy anywhere,
        which the sparse attention kernels treat as masked. `compressed_seq_lens`
        is in the indexer's compressed space and is unused by backings that
        derive residency per position instead of per length.
        """
        ...

    def prepare_decode_batch(
        self,
        *,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """The one controlled point between two decode forwards.

        Each backing does its own pre-forward bookkeeping here: routing the new
        decode token into the hot buffer, or applying queued eviction events
        before the allocator can hand a freed slot to someone else.
        """
        ...

    # ---- request lifecycle --------------------------------------------
    def on_prefill_complete(self, req: Req) -> bool:
        """Offer a finished prefill to HiSparse; True when HiSparse takes it.

        Called once per request, after its last chunk and after the request has
        been inserted into the tree cache, so the prefix it reports is final.

        False means declined -- nothing evictable, prefix too short, a quota
        exhausted -- and the request runs as an ordinary non-HiSparse request.
        That is a normal outcome, not an error; a backing that declines for a
        reason worth acting on warns for itself.

        True carries no promise about *when*: a backing may still be copying the
        KV out (the scheduler keeps such a request out of the running batch until
        `collect_ready_reqs()` returns it) or may not have taken it over yet, in
        which case it stays an ordinary request until `admit_pending()` runs.
        Both are invisible here on purpose -- the scheduler drives them through
        `collect_ready_reqs` / `has_ongoing_staging` and `admit_pending`, so a
        richer return value would be a second description of the same state,
        free to drift from the one the scheduler acts on.
        """
        ...

    def on_prefill_finished_early(self, req: Req) -> None:
        """Release what a request reserved when it finishes *during* prefill.

        `max_new == 0`, or EOS/stop on the first token: the request was budgeted
        at prefill admission but never reaches `on_prefill_complete` or
        `request_finished`, so a backing that charged it anything has to settle
        up here or leak that charge for the server's lifetime. A backing that
        holds nothing until admission does nothing here.
        """
        ...

    def admit_pending(self) -> None:
        """Finish admitting the requests that `on_prefill_complete` deferred.

        Called once per prefill-result pass, AFTER that pass has flushed its
        batched frees -- which is the whole reason a backing defers. Admission
        allocates from the device pool and may have to evict to get there, and
        inside the result pass the pages an eviction frees are parked in the
        allocator's free group, where neither `available_size()` nor `alloc()`
        can see them. It also has to run before the next scheduling round, or the
        next prefill takes the pages first.

        No-op for a backing that admits synchronously.
        """
        ...

    def collect_ready_reqs(self) -> List[Req]:
        """Requests whose staging copy has landed and that may now run.

        Empty for a backing that admits synchronously.
        """
        ...

    def has_ongoing_staging(self) -> bool:
        """True while any request is still waiting on a staging copy.

        The scheduler must not go idle while this holds. False for a backing
        that admits synchronously.
        """
        ...

    def request_finished(self, req: Req) -> None:
        """Release everything the request holds (slots, host space, claims)."""
        ...

    def retract_req(self, req: Req) -> None:
        """Undo admission for a retracted request, at whatever stage it is in."""
        ...

    # ---- scheduler hooks ----------------------------------------------
    def admit_budget(self) -> Optional[HiCacheAdmitBudget]:
        """This round's admission budget, or None when the backing has no quota.

        None rather than a permissive stub, so the scheduler's fast path stays
        free of per-candidate calls. Takes no pool size: a backing that rations
        knows its own, and passing one in let the round's feasibility threshold
        be computed against a different number than the ceiling it is compared
        with.

        The one member typed against a backing's own class, against the rule
        above: only the HiCache backing rations, and its budget is two methods a
        second Protocol would restate. If a second rationing backing appears,
        that Protocol is what to reintroduce.
        """
        ...

    def wait_for_pending_backup(self) -> None:
        """Block until in-flight device-to-host copies are visible on this stream."""
        ...

    def get_token_stats(self) -> HiSparseTokenStats:
        """Device / host occupancy for pool-stats logging."""
        ...

    # ---- setup / teardown ---------------------------------------------
    def set_decode_producer_stream(self, stream) -> None:
        """Register the forward stream that produces decode KV.

        Release and backup paths order themselves after it, so a freed slot is
        never reused while an overlapped forward is still reading it.
        """
        ...

    def set_tree_cache(self, tree_cache) -> None:
        """Attach the tree cache. No-op for a backing that runs without one."""
        ...

    def destroy(self) -> None:
        """Release whatever must not outlive the coordinator (in-flight
        transfers, host registrations, non-owning pointers into pools torn
        down by their real owner)."""
        ...
