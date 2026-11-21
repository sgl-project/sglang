"""Share prompt KV among the requests of one prefill batch.

The radix tree holds a request's KV only once its prefill result is processed,
so requests admitted to one prefill batch with a common prompt (the samples of
an RL rollout group, a client fanning out samples as separate requests) would
each compute it. Instead, a request (the follower) takes as its prefix the
leading full pages that an earlier request of the batch (its leader) computes
in the same forward:

- ``PrefixSharePlanner`` picks a leader for each request as PrefillAdder
  admits it and raises the request's prefix to the shared length, so
  admission charges, and allocation gives it, KV slots only past it.
- ``borrowed_prefix_indices`` points the follower's row at the leader's slots
  once allocation has given the leader its slots.
- ``publish_prefix_shares`` inserts the leader's shared span into the tree
  before the forward, so the tree owns those slots, and moves each follower's
  tree lock onto them as if it had matched them.

Each layer stores the batch's new KV before it reads cached prefixes (see
``AttentionBackend.stores_extend_kv_before_reading_prefix``), so a follower
reads the KV its leader writes in that layer. Later batches read the published
slots after this forward on the same stream.
"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING, Optional, Sequence

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.cp.base import is_cp_sharded_padded
from sglang.srt.mem_cache.allocator.page_interleave import page_interleave_shard_size
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.common import match_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_memory,
    get_parallel,
    uses_mla_backend,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


# Requests hold token ids as array("q").
_TOKEN_BYTES = 8


class PrefixShare(msgspec.Struct, frozen=True):
    """Batch row ``follower`` takes KV ``[0, length)`` of batch row ``leader``;
    ``length`` is page-aligned."""

    follower: int
    leader: int
    length: int


class InBatchPrefixSharing:
    """--enable-in-batch-prefix-sharing: the server-wide settings, and a planner
    for each prefill batch."""

    def __init__(self, *, tree_cache: BasePrefixCache, page_size: int):
        self.tree_cache = tree_cache
        self.page_size = page_size

    @staticmethod
    def unsupported_reason(
        *,
        tree_cache: BasePrefixCache,
        disable_radix_cache: bool,
        attn_backend: AttentionBackend,
        spec_algorithm: SpeculativeAlgorithm,
        is_dllm: bool,
    ) -> Optional[str]:
        """Why the server cannot share prompt KV within a prefill batch, or None."""
        memory = get_memory()
        parallel = get_parallel()
        if disable_radix_cache:
            return "the radix cache is disabled"
        if tree_cache.supports_mamba() or tree_cache.supports_swa():
            # A follower would need the leader's recurrent state or window at the
            # shared boundary, which exists only after the forward.
            return "hybrid (mamba / sliding-window) caches are not supported"
        if not attn_backend.stores_extend_kv_before_reading_prefix:
            return (
                f"attention backend {type(attn_backend).__name__} may read a cached "
                "prefix before it stores the batch's new KV"
            )
        if uses_mla_backend():
            # Chunked-prefix MLA reads the cached prefix outside the attention backend.
            return "MLA models are not supported"
        if not spec_algorithm.is_none():
            return "speculative decoding is not supported"
        if get_exec().overlap.enable_two_batch_overlap:
            # A follower in the first micro-batch would read a layer's KV before
            # its leader in the second one writes it.
            return "two-batch overlap is not supported"
        if (
            memory.enable_hierarchical_cache
            or memory.enable_lmcache
            or memory.enable_flexkv
            or memory.enable_unified_cache_external_linker
            or memory.enable_hisparse
        ):
            # These copy tree KV off the forward stream, possibly before the
            # forward that writes a published span has run.
            return "hierarchical, external and sparse KV caches are not supported"
        if get_disagg().disaggregation_mode != "null":
            return "PD disaggregation is not supported"
        if parallel.pp_size > 1 or parallel.attn_cp_size > 1 or parallel.dcp_enabled:
            return "pipeline and context parallelism are not supported"
        if is_cp_sharded_padded() or (
            page_interleave_shard_size(tree_cache.token_to_kv_pool_allocator) > 1
        ):
            return "sharded KV pages are not supported"
        if is_dllm:
            return "diffusion LLMs are not supported"
        if envs.SGLANG_RADIX_FORCE_MISS.get():
            return "SGLANG_RADIX_FORCE_MISS is set"
        return None

    def new_planner(self) -> PrefixSharePlanner:
        return PrefixSharePlanner(tree_cache=self.tree_cache, page_size=self.page_size)


class PrefixSharePlanner:
    """Plans the shares of one prefill batch while PrefillAdder admits it."""

    def __init__(self, *, tree_cache: BasePrefixCache, page_size: int):
        self._tree_cache = tree_cache
        self._page_size = page_size
        # Admitted leader candidates as (namespace, token bytes, admission
        # index), sorted: among them, the longest common prefix with a request
        # is next to its insertion point.
        self._leaders: list[tuple[str, bytes, int]] = []
        self._admitted: list[Req] = []
        # The tree's match of each admitted request, before any share.
        self._tree_prefix_len: list[int] = []
        # Whether the tree holds no more of an admitted leader's span than its
        # match, by admission index (see _publishable).
        self._publishable: dict[int, bool] = {}
        # (follower, leader) admission indices and the shared length.
        self._shares: list[tuple[int, int, int]] = []
        self._proposal: Optional[tuple[str, bytes, int, Optional[int]]] = None

    def propose(self, req: Req) -> None:
        """Before PrefillAdder prices ``req``: raise its prefix to what it can
        take from an admitted request."""
        self._proposal = None
        if not _is_shareable(req) or req.kv.holds_kv:
            return
        key = (_namespace(req), req.full_untruncated_fill_ids.tobytes())
        leader, length = self._best_leader(req, key)
        self._proposal = (*key, req.prefix_len, leader)
        if leader is not None:
            req.prefix_len = length

    def settle(self, req: Req, *, admitted: bool) -> None:
        """After PrefillAdder decided on the request ``propose`` saw."""
        if self._proposal is None:
            return
        namespace, token_bytes, tree_prefix_len, leader = self._proposal
        self._proposal = None
        if not admitted:
            req.prefix_len = tree_prefix_len
            return
        index = len(self._admitted)
        self._admitted.append(req)
        self._tree_prefix_len.append(tree_prefix_len)
        if leader is not None:
            self._shares.append((index, leader, req.prefix_len))
        if self._published_len(req) > tree_prefix_len:
            bisect.insort(self._leaders, (namespace, token_bytes, index))

    def shares(self, batch_reqs: Sequence[Req]) -> tuple[PrefixShare, ...]:
        """The planned shares, by row of the batch built from the admitted
        requests; a follower comes after its leader."""
        # A preemption during admission releases KV into the tree, which can
        # leave a leader stale after its followers were admitted; they then
        # compute their prefix themselves.
        self._publishable.clear()
        row = {id(req): i for i, req in enumerate(batch_reqs)}
        shares = []
        for follower, leader, length in self._shares:
            if not self._is_publishable(leader):
                self._admitted[follower].prefix_len = self._tree_prefix_len[follower]
                continue
            shares.append(
                PrefixShare(
                    follower=row[id(self._admitted[follower])],
                    leader=row[id(self._admitted[leader])],
                    length=length,
                )
            )
        return tuple(shares)

    def _best_leader(
        self, req: Req, key: tuple[str, bytes]
    ) -> tuple[Optional[int], int]:
        # At least one token stays to compute the logits from.
        limit = req._compute_max_prefix_len(len(req.full_untruncated_fill_ids))
        pos = bisect.bisect_left(self._leaders, key)
        best, best_len = None, req.prefix_len
        for namespace, token_bytes, leader in self._leaders[max(pos - 1, 0) : pos + 1]:
            if namespace != key[0]:
                continue
            length = self._common_pages_len(
                key[1],
                token_bytes,
                limit=min(limit, self._published_len(self._admitted[leader])),
            )
            if length <= req.prefix_len or not self._is_publishable(leader):
                continue
            if best is None or (length, -leader) > (best_len, -best):
                best, best_len = leader, length
        return best, best_len

    def _published_len(self, leader: Req) -> int:
        # The leader computes up to extend_end in this forward; the tree takes
        # only full pages.
        return leader.extend_end // self._page_size * self._page_size

    def _is_publishable(self, leader: int) -> bool:
        """Whether the tree holds none of the leader's span past its match. A
        stale match would make the insert free the leader's duplicate slots,
        which the forward still writes."""
        if leader not in self._publishable:
            req = self._admitted[leader]
            match = self._tree_cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(
                        req.full_untruncated_fill_ids,
                        req.extra_key,
                        limit=self._published_len(req),
                        cache_salt=req.cache_salt,
                    )
                )
            )
            self._publishable[leader] = (
                match.device_prefix_len == self._tree_prefix_len[leader]
            )
        return self._publishable[leader]

    def _common_pages_len(self, a: bytes, b: bytes, *, limit: int) -> int:
        """Tokens in the longest common prefix of two int64 token byte strings,
        in full pages, at most limit tokens."""
        page_bytes = self._page_size * _TOKEN_BYTES
        num_pages = min(len(a) // _TOKEN_BYTES, len(b) // _TOKEN_BYTES, limit)
        num_pages //= self._page_size
        if a[: num_pages * page_bytes] == b[: num_pages * page_bytes]:
            return num_pages * self._page_size
        lo, hi = 0, num_pages
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if a[: mid * page_bytes] == b[: mid * page_bytes]:
                lo = mid
            else:
                hi = mid
        return lo * self._page_size


def publish_prefix_shares(
    *,
    tree_cache: BasePrefixCache,
    reqs: Sequence[Req],
    shares: Sequence[PrefixShare],
) -> None:
    """Lock each follower's shared prefix and insert each leader's shared span
    into the tree, in row order so a leader's own share is in the tree before
    it extends it. Runs once allocation has written every row."""
    follower_len = {share.follower: share.length for share in shares}
    leader_len: dict[int, int] = {}
    for share in shares:
        leader_len[share.leader] = max(leader_len.get(share.leader, 0), share.length)
    for row in sorted(follower_len.keys() | leader_len.keys()):
        req = reqs[row]
        if row in follower_len:
            _lock_shared_prefix(tree_cache, req, follower_len[row])
        # A span inside the leader's prefix is already in the tree.
        if leader_len.get(row, 0) > req.prefix_len:
            tree_cache.checkpoint(req, up_to=leader_len[row])


def _lock_shared_prefix(tree_cache: BasePrefixCache, req: Req, length: int) -> None:
    old_lock = req.lock
    match_kv_cache(
        req, tree_cache, req.full_untruncated_fill_ids, max_prefix_len=length
    )
    assert req.prefix_len == length, (req.rid, req.prefix_len, length)
    req.kv.protect_prefix(length)
    req.lock = tree_cache.lock(req.last_node)
    tree_cache.unlock(old_lock)


def borrowed_prefix_indices(
    *,
    prefix_indices: Sequence[torch.Tensor],
    out_cache_loc: torch.Tensor,
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    shares: Sequence[PrefixShare],
) -> list[torch.Tensor]:
    """Each row's prefix KV indices, the followers' taken from their leader's
    prefix and the leader's slots in ``out_cache_loc`` (laid out by row).
    Shares come in follower order, so a leader's own share resolves first."""
    starts = [0]
    for extend_len in extend_lens:
        starts.append(starts[-1] + extend_len)
    result = list(prefix_indices)
    for share in shares:
        leader = share.leader
        computed_len = share.length - prefix_lens[leader]
        if computed_len <= 0:
            result[share.follower] = result[leader][: share.length]
            continue
        result[share.follower] = torch.cat(
            [
                result[leader].to(out_cache_loc.device, torch.int64),
                out_cache_loc[starts[leader] : starts[leader] + computed_len].to(
                    torch.int64
                ),
            ]
        )
    return result


def _is_shareable(req: Req) -> bool:
    # Rows whose KV the token ids alone do not determine, or that a session,
    # a host load or a beam group owns, neither lead nor follow.
    return (
        req.session is None
        and not req.skip_radix_cache_insert
        and req.multimodal_inputs is None
        and req.input_embeds is None
        and not req.is_dllm()
        and req.beam_group is None
        and req.host_hit_length == 0
        and req.discard_output_reason is None
    )


def _namespace(req: Req) -> str:
    # Requests share KV only within one radix-key namespace and LoRA adapter.
    return repr((req.extra_key, req.cache_salt, req.lora_id))
