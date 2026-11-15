"""ShardQ: token-sharded prefill attention under attention TP.

The model's attention weights (``wq_b``, ``wo_a``/``wo_b``) split heads across
the attention-TP group, so every rank's query only ever carries its own
``n_local_heads`` heads -- but the low-ratio index sources are replicated
(``ReplicatedLinear``, ``n_local_heads == n_heads``), so every rank scores
every prefill token with every index head, and the dense MQA logits scan
dominates long-context prefill without growing any cheaper as TP widens.

ShardQ removes that redundancy by cutting the query dimension instead of the
head dimension for the layers that read the compressed context before the
two-level candidate scheme cuts in (see ``DeepseekV4AttnBackend._plan_shard_q``
for the exact cutoff). Each rank attends a contiguous *zigzag* share of the
step's prefill tokens with *every* head:

1. A rank's query rows carry only its local heads, ``[T, H / tp, D]``. An
   all-to-all turns that into ``[T_rank, H, D]`` -- every head of this rank's
   own tokens (``scatter_q``).
2. The low-ratio sources (compressor, indexer) and the compressed-KV / SWA
   attention run once per rank over ``T_rank`` tokens with full heads, instead
   of every rank repeating the same ``T`` tokens with a slice of the heads.
   KV reads and causal lengths are unaffected: every rank's KV cache already
   holds every token (the compressor wrote it before this split), so a
   zigzag chunk is just an ordinary prefill chunk whose context ends at its
   own last token.
3. A second all-to-all restores ``[T_rank, H, D]`` back to ``[T, H / tp, D]``
   (``gather_o``), handing the rest of the layer (``wo_a``/``wo_b``) the shape
   it already expects.

The zigzag split (each request cut into ``2 * world`` equal chunks, rank ``r``
taking chunks ``r`` and ``2 * world - 1 - r``) balances the causal attention
cost across ranks: a rank that only took early chunks would see short causal
windows and finish early, so every rank gets one short (early) and one long
(late) share.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import msgspec
import numpy as np
import torch

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator

# Below this many prefill tokens in the step, the two all-to-alls' fixed
# latency and the per-request zigzag runs outweigh the indexer/attention
# time ShardQ removes; the measured crossover on H20 is ~8K tokens.
MIN_TOKENS = 8192

# A small step is still worth sharding as the tail chunk of a long-context
# prefill: the indexer cost per row grows with the context length.
LONG_CONTEXT_MIN_SEQ = 32768
LONG_CONTEXT_MIN_TOKENS = 1024


def step_amortizes_shard_q(extend_lens_cpu: List[int], seq_lens_cpu: List[int]) -> bool:
    """Whether a step's prefill tokens amortize ShardQ's exchanges.

    A step below ``MIN_TOKENS`` still pays as the tail chunk of a
    long-context prefill, but only the tokens of its own long-context
    requests count toward that: a long context this step barely extends must
    not drag a batch of short prefills into the split.
    """
    if sum(extend_lens_cpu) >= MIN_TOKENS:
        return True
    return (
        sum(
            num
            for num, seq_len in zip(extend_lens_cpu, seq_lens_cpu)
            if seq_len >= LONG_CONTEXT_MIN_SEQ
        )
        >= LONG_CONTEXT_MIN_TOKENS
    )


def zigzag_owner(extend_lens: List[int], world: int) -> np.ndarray:
    """The owning rank of every token across all requests, request-major.

    Each request's tokens are divided by proportional boundaries into
    ``2 * world`` chunks. Rank ``r`` owns chunks ``r`` and
    ``2 * world - 1 - r``, pairing an early (short causal window) chunk with a
    late (long) one so every rank's share costs about the same. Remainder
    tokens are distributed across those proportional chunks.
    """
    lens = np.asarray(extend_lens, dtype=np.int64)
    if lens.size == 0 or np.any(lens <= 0):
        return np.empty(0, dtype=np.int64)
    num_chunks = 2 * world
    total = int(lens.sum())
    # Position of every token within its own request, request-major order.
    local_pos = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
    chunk = ((local_pos + 1) * num_chunks - 1) // np.repeat(lens, lens)
    chunk = np.minimum(chunk, num_chunks - 1)
    return np.minimum(chunk, num_chunks - 1 - chunk)


def split_runs(
    token_ids: np.ndarray,
    extend_lens: List[int],
    seq_lens: List[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Describe ascending ``token_ids`` as contiguous runs of real requests.

    Returns the request of every token, each run's query length, each run's
    request, and the causal sequence length at each run's last token. The last
    value is deliberately not the request's final sequence length: an early
    zigzag run must not see tokens from the request's later run.
    """
    token_ids = np.asarray(token_ids, dtype=np.int64)
    extend = np.asarray(extend_lens, dtype=np.int64)
    seq = np.asarray(seq_lens, dtype=np.int64)
    if token_ids.size == 0 or extend.size != seq.size:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, empty
    query_start = np.r_[0, np.cumsum(extend)]
    req = np.searchsorted(query_start[1:], token_ids, side="right")
    boundary = np.ones(token_ids.size, dtype=bool)
    boundary[1:] = (token_ids[1:] != token_ids[:-1] + 1) | (req[1:] != req[:-1])
    starts = np.flatnonzero(boundary)
    ends = np.r_[starts[1:], token_ids.size]
    run_qlen = (ends - starts).astype(np.int64)
    run_req = req[starts]
    run_last = token_ids[ends - 1]
    run_seq = (seq[run_req] - query_start[run_req + 1] + run_last + 1).astype(np.int64)
    return req, run_qlen, run_req, run_seq


class ShardQ(msgspec.Struct, frozen=True, kw_only=True):
    """One step's token-sharded exchange plan for one attention-TP group.

    ``token_ids`` lists this rank's tokens in ascending global-batch order;
    the attention call reads exactly those tokens' worth of metadata (see
    ``DeepseekV4AttnBackend._build_shard_q_metadata``). ``send_ids``/
    ``recv_ids`` describe the permutation the two all-to-alls apply, built
    once and shared by ``scatter_q``/``gather_o`` across every layer of the
    step.
    """

    group: GroupCoordinator  # attention TP
    rows_per_rank: int  # every rank's padded share (the largest actual share)
    num_tokens_per_rank: List[int]
    token_ids: torch.Tensor  # [T_rank] this rank's tokens, ascending
    token_ids_cpu: List[int]
    rows_per_request: List[int]  # this rank's token count of each request

    # [world * rows_per_rank]: all-to-all slot i holds send_ids[i]'s global row.
    send_ids: torch.Tensor
    # [T]: global row i is received at (and gathered back from) slot recv_ids[i].
    recv_ids: torch.Tensor

    @classmethod
    def build(
        cls,
        *,
        extend_lens: List[int],
        seq_lens: List[int],
        group: GroupCoordinator,
        device: torch.device,
    ) -> Optional[ShardQ]:
        """Build one rank's exchange plan, or return ``None`` on every rank.

        Eligibility is checked from the complete split, not just this rank's
        rows. That is required before the first collective: if one rank falls
        back because one of its runs is invalid while another rank enters the
        all-to-all, the process group deadlocks.
        """
        world, rank = group.world_size, group.rank_in_group
        if world <= 1 or len(extend_lens) != len(seq_lens):
            return None
        owner = zigzag_owner(extend_lens, world)
        if owner.size == 0:
            return None
        sizes = np.bincount(owner, minlength=world)
        if sizes.min() == 0:
            return None
        # Stable sort groups tokens by owner while preserving each owner's
        # original (ascending, request-major) order within its group.
        order = np.argsort(owner, kind="stable")
        rows = int(sizes.max())
        group_starts = np.cumsum(sizes) - sizes
        for r in range(world):
            rank_tokens = order[group_starts[r] : group_starts[r] + sizes[r]]
            _, run_qlen, _, run_seq = split_runs(rank_tokens, extend_lens, seq_lens)
            if (
                run_qlen.size == 0
                or int(run_qlen.min()) <= 0
                or int(run_seq.min()) <= 0
            ):
                return None

        # slot[i] is where global row order[i] lands in the padded send buffer.
        slot = np.repeat(np.arange(world) * rows, sizes) + (
            np.arange(order.size) - np.repeat(group_starts, sizes)
        )
        send_ids = np.zeros(world * rows, dtype=np.int64)
        send_ids[slot] = order
        recv_ids = np.empty(order.size, dtype=np.int64)
        recv_ids[order] = slot
        local = order[group_starts[rank] : group_starts[rank] + sizes[rank]]
        request = np.repeat(np.arange(len(extend_lens)), extend_lens)
        local_cpu = local.tolist()
        return cls(
            group=group,
            rows_per_rank=rows,
            num_tokens_per_rank=sizes.tolist(),
            token_ids=torch.from_numpy(local).to(device),
            token_ids_cpu=local_cpu,
            rows_per_request=np.bincount(
                request[local], minlength=len(extend_lens)
            ).tolist(),
            send_ids=torch.from_numpy(send_ids).to(device),
            recv_ids=torch.from_numpy(recv_ids).to(device),
        )

    def scatter_q(self, q: torch.Tensor) -> torch.Tensor:
        """``[T, H_local, D]`` (this rank's local-head query of every token)
        to ``[T_rank, H_local * world, D]`` (every head of this rank's own
        tokens, head block ``r`` supplied by rank ``r``)."""
        world, rows = self.group.world_size, self.rows_per_rank
        num_tokens = self.recv_ids.numel()
        assert q.ndim == 3 and q.shape[0] == num_tokens, (
            f"ShardQ expects replicated, unpadded query rows: {q.shape[0]=}, "
            f"{num_tokens=}"
        )
        assert self.send_ids.numel() == world * rows
        assert int(self.send_ids.max()) < num_tokens
        send = q.index_select(0, self.send_ids)
        recv = torch.empty_like(send)
        self.group.all_to_all_single(recv, send)
        num_local = self.token_ids.shape[0]
        assert num_local == self.num_tokens_per_rank[self.group.rank_in_group]
        recv = recv.view(world, rows, *q.shape[1:])[:, :num_local]
        return recv.transpose(0, 1).reshape(num_local, -1, q.shape[-1])

    def gather_o(self, o: torch.Tensor) -> torch.Tensor:
        """The inverse of ``scatter_q``'s head layout: ``[T_rank, H_local *
        world, D]`` back to ``[T, H_local, D]``."""
        world, rows = self.group.world_size, self.rows_per_rank
        num_local, num_heads, dim = o.shape
        assert o.ndim == 3 and num_local == self.token_ids.numel()
        assert num_heads % world == 0, (num_heads, world)
        assert int(self.recv_ids.max()) < world * rows
        send = o.new_empty(world, rows, num_heads // world, dim)
        send[:, :num_local] = o.view(num_local, world, -1, dim).transpose(0, 1)
        recv = torch.empty_like(send)
        self.group.all_to_all_single(recv, send)
        restored = recv.view(world * rows, -1, dim).index_select(0, self.recv_ids)
        assert restored.shape[0] == self.recv_ids.numel()
        return restored
