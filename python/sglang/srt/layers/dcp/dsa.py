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

"""Decode context parallel (DCP) helpers for DSA sparse attention (GLM-5 / V3.2).

Owner rule matches the widened allocator: slot % W == rank, local row = slot // W.
"""

import contextlib
from typing import Callable, List, NamedTuple, Optional, Tuple

import torch

from sglang.kernels.ops.attention.dcp_kernels import (
    dcp_compact_owned_slots,
    dcp_topk_merge,
    dcp_topk_pack,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.srt.runtime_context import get_parallel


def _localize(loc: torch.Tensor) -> torch.Tensor:
    parallel = get_parallel()
    w, r = parallel.attn_dcp_size, parallel.attn_dcp_rank
    return torch.where(loc % w == r, loc // w, torch.zeros_like(loc))


def _root(t: torch.Tensor) -> torch.Tensor:
    return t if t._base is None else t._base


class DcpStep:
    """Per-forward DCP state: the localized write loc and the per-step indexer
    metadata, derived once and read by every layer.

    Lives only for one model forward (``dcp_forward_scope``), so it can never
    serve a later step whose buffers were refilled in place. Under CUDA graph
    capture the derivation is recorded in the graph and replays once per step.
    """

    def __init__(self, out_cache_loc: Optional[torch.Tensor]):
        self.src_loc = out_cache_loc
        self.local_loc = None if out_cache_loc is None else _localize(out_cache_loc)
        self._memo = {}

    def local_loc_for(self, loc: torch.Tensor) -> Optional[torch.Tensor]:
        src = self.src_loc
        if (
            src is None
            or _root(loc) is not _root(src)
            or loc.data_ptr() != src.data_ptr()
            or loc.numel() > src.numel()
            or loc.dim() != 1
            or loc.stride(0) != 1
        ):
            return None
        return self.local_loc[: loc.numel()]

    def memo(self, name: str, src: torch.Tensor, fn: Callable):
        hit = self._memo.get(name)
        if hit is not None and hit[0] is src:
            return hit[1]
        out = fn()
        self._memo[name] = (src, out)
        return out


_STEP: Optional[DcpStep] = None


@contextlib.contextmanager
def dcp_forward_scope(forward_batch):
    """Open the per-forward ``DcpStep`` around one model forward."""
    global _STEP
    if not get_parallel().dcp_enabled or forward_batch.forward_mode.is_idle():
        yield
        return
    prev, _STEP = _STEP, DcpStep(getattr(forward_batch, "out_cache_loc", None))
    try:
        yield
    finally:
        _STEP = prev


def dcp_step() -> Optional[DcpStep]:
    return _STEP


def dcp_localize_write_loc(loc: torch.Tensor) -> torch.Tensor:
    """Widened write loc -> this rank's row; non-owned ids go to reserved row 0."""
    if not get_parallel().dcp_enabled:
        return loc
    if _STEP is not None:
        local = _STEP.local_loc_for(loc)
        if local is not None:
            return local
    return _localize(loc)


def dcp_local_lens(seq_lens: torch.Tensor) -> torch.Tensor:
    """This rank's int32 token counts, memoized per forward on the source tensor."""
    parallel = get_parallel()

    def fn():
        return get_dcp_lens(
            seq_lens, parallel.attn_dcp_size, parallel.attn_dcp_rank
        ).to(torch.int32)

    return _STEP.memo("lens", seq_lens, fn) if _STEP is not None else fn()


def dcp_compact_read_table(
    page_table_1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Widened top-k slot table (-1 = invalid) -> (this rank's local rows packed
    to the front with a -1 tail, per-row owned count)."""
    parallel = get_parallel()
    return dcp_compact_owned_slots(
        page_table_1, parallel.attn_dcp_size, parallel.attn_dcp_rank
    )


def dcp_local_index_block_table(page_table_1: torch.Tensor, page_size: int):
    """Per-request local index-K page table and its column capacity in tokens.

    One widened page (page_size * W slots) holds page_size local tokens per rank.
    """
    span = page_size * get_parallel().attn_dcp_size

    def fn():
        return (page_table_1[:, ::span] // span).to(torch.int32).contiguous()

    block_tables = (
        _STEP.memo(f"block_table_{page_size}", page_table_1, fn)
        if _STEP is not None
        else fn()
    )
    return block_tables, block_tables.shape[1] * page_size


def dcp_exchange_topk(
    local_logits: torch.Tensor,
    local_lens: torch.Tensor,
    topk: int,
    topk_func: Callable,
) -> torch.Tensor:
    """Local top-k -> all-gather (score, global pos) -> global top-k (-1 padded).

    A token in the global top-k is in its owner's local top-k, so the merge is exact.
    """
    parallel = get_parallel()
    w, r = parallel.attn_dcp_size, parallel.attn_dcp_rank
    rows = local_logits.shape[0]
    if local_logits.shape[1] < topk:
        local_logits = torch.nn.functional.pad(
            local_logits, (0, topk - local_logits.shape[1]), value=-float("inf")
        )
    local_idx = topk_func(local_logits, local_lens, topk)
    if local_logits.is_cuda:
        send = dcp_topk_pack(local_logits, local_idx, local_lens, w, r)
        recv = parallel.dcp_group.all_gather(send, dim=0)
        return dcp_topk_merge(recv, w, topk_func)

    valid = (local_idx >= 0) & (local_idx < local_lens.view(rows, 1))
    safe_idx = torch.where(valid, local_idx, torch.zeros_like(local_idx)).long()
    send = torch.empty((2, rows, topk), dtype=torch.float32, device=local_logits.device)
    send[0] = torch.where(
        valid, local_logits.gather(1, safe_idx), torch.full_like(send[0], -float("inf"))
    )
    # Pack the global position as int32 bits so one collective moves both planes.
    send.view(torch.int32)[1] = torch.where(valid, local_idx * w + r, -1)
    # Gather as fp32 (bit-exact) so ROCm can use the custom all-gather.
    recv = parallel.dcp_group.all_gather(send, dim=0).view(torch.int32)
    recv = recv.view(w, 2, rows, topk).permute(2, 1, 0, 3).reshape(rows, 2, w * topk)
    scores = recv[:, 0].contiguous().view(torch.float32)
    best, pick = torch.topk(scores, topk, dim=1)
    gids = recv[:, 1].gather(1, pick)
    return torch.where(best > -float("inf"), gids, -1).to(torch.int32)


def dcp_use_owned_prefill(
    extend_prefix_lens_cpu: List[int], extend_num_tokens: int, gathered_heads: int
) -> bool:
    """Whether a DSA prefill should attend owned slots + LSE merge.

    The gathered path moves the cached prefix KV (one 576 B latent per prefix
    token); the owned path moves every extend token's Q for all heads plus the
    head partials back, independent of the prefix.
    """
    ratio = envs.SGLANG_DCP_DSA_OWNED_PREFILL_RATIO.get()
    if ratio < 0:
        return False
    return sum(extend_prefix_lens_cpu) >= ratio * gathered_heads * extend_num_tokens


def dcp_use_split_indexer(
    extend_prefix_lens_cpu: List[int], extend_seq_lens_cpu: List[int]
) -> bool:
    """Whether the gathered-KV DSA prefill splits indexer rows across ranks.

    Splitting saves (W - 1) / W of the logits and top-k, which grow with the
    key length of each query token, for a fixed topk-wide all-gather per token.
    """
    tokens = sum(extend_seq_lens_cpu)
    if tokens == 0:
        return False
    keys = sum(
        e * p + e * (e + 1) // 2
        for p, e in zip(extend_prefix_lens_cpu, extend_seq_lens_cpu)
    )
    return keys >= envs.SGLANG_DCP_DSA_SPLIT_INDEXER_MIN_KV.get() * tokens


def dcp_split_rows(rows: int) -> Tuple[int, int, int]:
    """This rank's slice [lo, hi) of ``rows`` and the per-rank padded count."""
    parallel = get_parallel()
    per = -(-rows // parallel.attn_dcp_size)
    lo = min(parallel.attn_dcp_rank * per, rows)
    return lo, min(lo + per, rows), per


def dcp_all_gather_rows(part: torch.Tensor, rows: int, per: int) -> torch.Tensor:
    """Inverse of ``dcp_split_rows``: every rank's slice, back in row order."""
    send = part
    if part.shape[0] != per:
        send = part.new_full((per,) + tuple(part.shape[1:]), -1)
        send[: part.shape[0]] = part
    return get_parallel().dcp_group.all_gather(send, dim=0)[:rows]


def dcp_gather_index_k_prefill(
    pool,
    layer_id: int,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    page_table_1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rebuild the full-sequence index K (flat, per-request order) from DCP shards.

    Every rank reads ceil(len / W) local tokens per request so the all-gather is even.
    """
    parallel = get_parallel()
    w = parallel.attn_dcp_size
    shard = dcp_local_index_k_prefill(
        pool, layer_id, seq_lens, seq_lens_cpu, page_table_1
    )
    k_all = parallel.dcp_group.all_gather(
        shard.k_fp8.contiguous().view(torch.uint8), dim=0
    )
    s_all = parallel.dcp_group.all_gather(shard.k_scale.contiguous(), dim=0)
    pad_lens_cpu = (seq_lens_cpu + w - 1) // w
    src = _dcp_flat_gather_index(
        seq_lens_cpu.tolist(), pad_lens_cpu.tolist(), shard.pad_sum, w
    )
    src = src.to(shard.k_fp8.device, non_blocking=True)
    return (
        k_all.index_select(0, src).view(shard.k_fp8.dtype),
        s_all.index_select(0, src),
    )


def _dcp_flat_gather_index(
    seq_lens: List[int], pad_lens: List[int], pad_sum: int, w: int
) -> torch.Tensor:
    """Row in the rank-major gathered buffer for each (request, position)."""
    parts, base = [], 0
    for seq_len, pad_len in zip(seq_lens, pad_lens):
        pos = torch.arange(seq_len, dtype=torch.int64)
        parts.append((pos % w) * pad_sum + base + pos // w)
        base += pad_len
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.int64)


class DcpPrefillIndexK(NamedTuple):
    """This rank's index-K rows for a prefill batch, packed per request."""

    k_fp8: torch.Tensor  # [pad_sum, D]
    k_scale: torch.Tensor  # [pad_sum, ...]
    req_starts: torch.Tensor  # [bs] int32 first row of each request
    pad_sum: int  # sum(ceil(seq_len / W)), identical on every rank


def dcp_local_index_k_prefill(
    pool,
    layer_id: int,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    page_table_1: torch.Tensor,
) -> DcpPrefillIndexK:
    """Read this rank's index-K shard (ceil(len / W) rows per request).

    Local row j of a request is sequence position j * W + rank; rows at or past
    the rank's own length are padding and never scored.
    """
    w = get_parallel().attn_dcp_size
    block_tables, _ = dcp_local_index_block_table(page_table_1, pool.page_size)
    pad_lens_cpu = (seq_lens_cpu + w - 1) // w
    pad_sum = int(pad_lens_cpu.sum())
    pad_lens = ((seq_lens + w - 1) // w).to(torch.int32)
    k_fp8, k_scale = pool.get_index_k_scale_buffer(
        layer_id, pad_lens, block_tables, pad_sum, int(pad_lens_cpu.max())
    )
    req_starts = (torch.cumsum(pad_lens, dim=0) - pad_lens).to(torch.int32)
    return DcpPrefillIndexK(k_fp8, k_scale, req_starts, pad_sum)


def dcp_exchange_topk_prefill(
    local_logits: torch.Tensor,
    row_starts: torch.Tensor,
    local_lens: torch.Tensor,
    topk: int,
    topk_func: Callable,
) -> torch.Tensor:
    """Prefill rows: local top-k over packed logits -> global top-k (-1 padded).

    Like ``dcp_exchange_topk`` with per-token windows into the packed shard.
    """
    parallel = get_parallel()
    w, r = parallel.attn_dcp_size, parallel.attn_dcp_rank
    local_idx = topk_func(local_logits, local_lens, topk, row_starts=row_starts)
    send = dcp_topk_pack(
        local_logits, local_idx, local_lens, w, r, row_starts=row_starts
    )
    recv = parallel.dcp_group.all_gather(send, dim=0)
    return dcp_topk_merge(recv, w, topk_func)


def dcp_prefill_page_table(
    dcp_kv_indptr: torch.Tensor,
    dcp_kv_indices: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """[bs, width] position -> row of the all-gathered ``dcp_kv_buffer``."""
    bs = seq_lens_cpu.shape[0]
    device = dcp_kv_indices.device
    table = torch.zeros((bs, width), dtype=torch.int32, device=device)
    seq_lens = seq_lens_cpu.to(device=device, dtype=torch.int64)
    rows = torch.repeat_interleave(torch.arange(bs, device=device), seq_lens)
    cols = torch.arange(rows.shape[0], device=device) - dcp_kv_indptr[:-1].long()[rows]
    table[rows, cols] = dcp_kv_indices[: rows.shape[0]].to(torch.int32)
    return table
