"""Context-sharded MiniMax block scoring and exact ROCm top-k selection.

The cache is replicated. Rank r reads logical blocks r, r + WORLD, ...,
scoring all index heads while each cache tile is resident. Selection follows
the ROCm minimax_decode_topk contract: score descending, block ID ascending.
"""

import torch
import triton
import triton.language as tl

# Rows per scoring tile; unrelated to the selector's top-k, which is also 16.
TILE_ROWS = 16


@triton.jit
def _score_shard(
    Q,
    K,
    ReqToToken,
    Slots,
    Lengths,
    Scores,
    BATCH: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    LOCAL_BLOCKS: tl.constexpr,
    Q_HEAD_STRIDE: tl.constexpr,
    Q_ROW_STRIDE: tl.constexpr,
    K_SLOT_STRIDE: tl.constexpr,
    K_DIM_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    TABLE_ROWS: tl.constexpr,
    CACHE_SLOTS: tl.constexpr,
    BLOCKS_PER_CHUNK: tl.constexpr,
    INIT_BLOCKS: tl.constexpr,
    LOCAL_KEEP: tl.constexpr,
    sm_scale,
    k_scale,
):
    row, chunk = tl.program_id(0), tl.program_id(1)
    heads = tl.arange(0, 16)
    dims = tl.arange(0, 128)
    positions = tl.arange(0, 128)
    length = tl.load(Lengths + row)
    request = tl.load(Slots + row).to(tl.int64)
    # Match the native scorer's request/slot addressing, including padding.
    request = (request + CACHE_SLOTS) % CACHE_SLOTS
    active = (length > 0) & (request >= 0) & (request < TABLE_ROWS)
    num_blocks = tl.cdiv(length, 128)
    local_start = tl.maximum(0, num_blocks - LOCAL_KEEP)
    q = tl.load(
        Q + heads[:, None] * Q_HEAD_STRIDE + row * Q_ROW_STRIDE + dims[None, :],
        mask=heads[:, None] < WORLD,
        other=0,
    )
    start = chunk * BLOCKS_PER_CHUNK
    end = tl.minimum(start + BLOCKS_PER_CHUNK, LOCAL_BLOCKS)
    for local in range(start, end):
        block = local * WORLD + RANK
        valid = active & (block < MAX_BLOCKS) & (block < num_blocks)
        # At <=top-k blocks the native selector emits every block without
        # reading scores. In particular, graph padding need not read K.
        score = tl.full((16,), 0.0, tl.float32)
        if valid & (num_blocks > 16):
            pos = block * 128 + positions
            token_valid = pos < length
            slots = tl.load(
                ReqToToken + request * TABLE_STRIDE + pos,
                mask=token_valid,
                other=0,
            ).to(tl.int64)
            slots = (slots + CACHE_SLOTS) % CACHE_SLOTS
            k = tl.load(
                K + dims[:, None] * K_DIM_STRIDE + slots[None, :] * K_SLOT_STRIDE,
                mask=token_valid[None, :],
                other=0.0,
            ).to(q.dtype)
            # Preserve the baseline's dot orientation and scale order.
            dot = tl.dot(q, k) * (sm_scale * 1.4426950409 * k_scale)
            dot = tl.where(token_valid[None, :], dot, float("-inf"))
            score = tl.max(dot, 1)
            score = tl.where(
                block >= local_start,
                1e29,
                tl.where(block < INIT_BLOCKS, 1e30, score),
            )
        score = tl.where(valid, score, float("-inf"))
        # Every slot is written on every replay, including empty shards.
        tl.store(
            Scores + (heads * BATCH + row) * LOCAL_BLOCKS + local,
            score,
            mask=heads < WORLD,
        )


@triton.jit
def _score_shard_packed(
    Q,
    K,
    ReqToToken,
    Slots,
    Lengths,
    Scores,
    BATCH: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    PACK: tl.constexpr,
    TILE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    LOCAL_BLOCKS: tl.constexpr,
    Q_HEAD_STRIDE: tl.constexpr,
    Q_ROW_STRIDE: tl.constexpr,
    K_SLOT_STRIDE: tl.constexpr,
    K_DIM_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    TABLE_ROWS: tl.constexpr,
    CACHE_SLOTS: tl.constexpr,
    BLOCKS_PER_CHUNK: tl.constexpr,
    INIT_BLOCKS: tl.constexpr,
    LOCAL_KEEP: tl.constexpr,
    sm_scale,
    k_scale,
):
    # tile row i is (draft row i // WORLD, head i % WORLD): one K read per request
    group, chunk = tl.program_id(0), tl.program_id(1)
    tile = tl.arange(0, TILE)
    head, sub = tile % WORLD, tile // WORLD
    in_tile = tile < PACK * WORLD
    rows = group * PACK + sub
    dims = tl.arange(0, 128)
    positions = tl.arange(0, 128)
    lengths = tl.load(Lengths + rows, mask=in_tile, other=0)
    group_len = tl.max(lengths, 0)
    # caller contract: a group's PACK rows share one request, so one slot addresses K
    request = tl.load(Slots + group * PACK).to(tl.int64)
    # Match the native scorer's request/slot addressing, including padding.
    request = (request + CACHE_SLOTS) % CACHE_SLOTS
    active = (lengths > 0) & (request >= 0) & (request < TABLE_ROWS) & in_tile
    num_blocks = tl.cdiv(lengths, 128)
    local_start = tl.maximum(0, num_blocks - LOCAL_KEEP)
    q = tl.load(
        Q
        + head[:, None] * Q_HEAD_STRIDE
        + rows[:, None] * Q_ROW_STRIDE
        + dims[None, :],
        mask=in_tile[:, None],
        other=0,
    )
    start = chunk * BLOCKS_PER_CHUNK
    end = tl.minimum(start + BLOCKS_PER_CHUNK, LOCAL_BLOCKS)
    for local in range(start, end):
        block = local * WORLD + RANK
        valid = active & (block < MAX_BLOCKS) & (block < num_blocks)
        # At <=top-k blocks the native selector emits every block without
        # reading scores. In particular, graph padding need not read K.
        score = tl.full((TILE,), 0.0, tl.float32)
        # scalar branch on the group's longest row; num_blocks restores each row's rule
        group_blocks = tl.cdiv(group_len, 128)
        if (block < group_blocks) & (group_blocks > 16):
            pos = block * 128 + positions
            token_valid = pos < group_len
            slots = tl.load(
                ReqToToken + request * TABLE_STRIDE + pos,
                mask=token_valid,
                other=0,
            ).to(tl.int64)
            slots = (slots + CACHE_SLOTS) % CACHE_SLOTS
            k = tl.load(
                K + dims[:, None] * K_DIM_STRIDE + slots[None, :] * K_SLOT_STRIDE,
                mask=token_valid[None, :],
                other=0.0,
            ).to(q.dtype)
            # Preserve the baseline's dot orientation and scale order.
            dot = tl.dot(q, k) * (sm_scale * 1.4426950409 * k_scale)
            dot = tl.where(pos[None, :] < lengths[:, None], dot, float("-inf"))
            scored = tl.max(dot, 1)
            scored = tl.where(
                block >= local_start,
                1e29,
                tl.where(block < INIT_BLOCKS, 1e30, scored),
            )
            score = tl.where(num_blocks > 16, scored, 0.0)
        score = tl.where(valid, score, float("-inf"))
        # Every slot is written on every replay, including empty shards.
        tl.store(
            Scores + (head * BATCH + rows) * LOCAL_BLOCKS + local,
            score,
            mask=in_tile,
        )


@triton.jit
def _pack(score, block, valid):
    # Same fp32 total order as ROCm TopKTrait::pack_score_id. 16 ID bits
    # suffice for the supported <=16384 blocks; signed int64 top-k is safe.
    score = tl.where(score != score, float("-inf"), score)
    bits = score.to(tl.uint32, bitcast=True)
    ordered = tl.where(bits & 0x80000000 != 0, ~bits, bits | 0x80000000)
    key = (ordered.to(tl.int64) << 16) | (65535 - block).to(tl.int64)
    return tl.where(valid, key, 0)


@triton.jit
def _local_candidates(
    Scores,
    Lengths,
    Keys,
    BATCH: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    LOCAL_BLOCKS: tl.constexpr,
    TOPK: tl.constexpr,
    TILE: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    blocks = tl.cdiv(tl.load(Lengths + row), 128)
    off = tl.arange(0, TILE)
    local = off
    block = local * WORLD + RANK
    scores = tl.load(
        Scores + (head * BATCH + row) * LOCAL_BLOCKS + local,
        mask=local < LOCAL_BLOCKS,
        other=float("-inf"),
    )
    winners = tl.topk(
        _pack(
            scores,
            block,
            (local < LOCAL_BLOCKS) & (block < MAX_BLOCKS) & (block < blocks),
        ),
        TOPK,
    )
    for start in range(TILE, LOCAL_BLOCKS, TILE):
        local = start + off
        block = local * WORLD + RANK
        scores = tl.load(
            Scores + (head * BATCH + row) * LOCAL_BLOCKS + local,
            mask=local < LOCAL_BLOCKS,
            other=float("-inf"),
        )
        tile = tl.topk(
            _pack(
                scores,
                block,
                (local < LOCAL_BLOCKS) & (block < MAX_BLOCKS) & (block < blocks),
            ),
            TOPK,
        )
        winners = tl.topk(tl.cat(winners, tile, can_reorder=True), TOPK)
    tl.store(Keys + (head * BATCH + row) * TOPK + tl.arange(0, TOPK), winners)


@triton.jit
def _merge_candidates(
    Gathered,
    Output,
    BATCH: tl.constexpr,
    WORLD: tl.constexpr,
    HEAD: tl.constexpr,
    TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    off = tl.arange(0, WORLD * TOPK)
    source, candidate = off // TOPK, off % TOPK
    keys = tl.load(
        Gathered + ((source * WORLD + HEAD) * BATCH + row) * TOPK + candidate
    )
    winners = tl.topk(keys, TOPK)
    # Consumers require ascending logical IDs followed by -1, not score order.
    ids = tl.where(winners != 0, 65535 - (winners & 65535), 2147483647)
    ids = tl.sort(ids.to(tl.int32), descending=False)
    tl.store(
        Output + row * TOPK + tl.arange(0, TOPK), tl.where(ids == 2147483647, -1, ids)
    )


def score_local_blocks(
    gathered_q,
    k_cache,
    req_to_token,
    slot_ids,
    seq_lens,
    max_seqlen,
    rank,
    init_blocks,
    local_blocks,
    sm_scale,
    k_scale,
    packed_queries=1,
):
    """Score [world,batch,128] queries through SGLang's token-slot mapping.

    ``packed_queries > 1``: rows come in groups of that many consecutive verify rows of one
    request (same slot), scored per K block together.
    """
    world, batch, _ = gathered_q.shape
    blocks = triton.cdiv(max_seqlen, 128)
    local = triton.cdiv(blocks, world)
    scores = torch.empty(
        (world, batch, local), dtype=torch.float32, device=gathered_q.device
    )
    if (
        packed_queries > 1
        and batch % packed_queries == 0
        and packed_queries * world <= TILE_ROWS
    ):
        groups = batch // packed_queries
        chunks = min(local, max(1, min(256, 4096 // max(groups, 1))))
        _score_shard_packed[(groups, chunks)](
            gathered_q,
            k_cache,
            req_to_token,
            slot_ids,
            seq_lens,
            scores,
            BATCH=batch,
            WORLD=world,
            RANK=rank,
            PACK=packed_queries,
            TILE=TILE_ROWS,
            MAX_BLOCKS=blocks,
            LOCAL_BLOCKS=local,
            Q_HEAD_STRIDE=gathered_q.stride(0),
            Q_ROW_STRIDE=gathered_q.stride(1),
            K_SLOT_STRIDE=k_cache.stride(0),
            K_DIM_STRIDE=k_cache.stride(2),
            TABLE_STRIDE=req_to_token.stride(0),
            TABLE_ROWS=req_to_token.shape[0],
            CACHE_SLOTS=k_cache.shape[0],
            BLOCKS_PER_CHUNK=triton.cdiv(local, chunks),
            INIT_BLOCKS=init_blocks,
            LOCAL_KEEP=local_blocks,
            sm_scale=sm_scale,
            k_scale=k_scale,
            num_warps=4,
            num_stages=1,
        )
        return scores
    chunks = min(local, max(1, min(256, 4096 // max(batch, 1))))
    if batch:
        _score_shard[(batch, chunks)](
            gathered_q,
            k_cache,
            req_to_token,
            slot_ids,
            seq_lens,
            scores,
            BATCH=batch,
            WORLD=world,
            RANK=rank,
            MAX_BLOCKS=blocks,
            LOCAL_BLOCKS=local,
            Q_HEAD_STRIDE=gathered_q.stride(0),
            Q_ROW_STRIDE=gathered_q.stride(1),
            K_SLOT_STRIDE=k_cache.stride(0),
            K_DIM_STRIDE=k_cache.stride(2),
            TABLE_STRIDE=req_to_token.stride(0),
            TABLE_ROWS=req_to_token.shape[0],
            CACHE_SLOTS=k_cache.shape[0],
            BLOCKS_PER_CHUNK=triton.cdiv(local, chunks),
            INIT_BLOCKS=init_blocks,
            LOCAL_KEEP=local_blocks,
            sm_scale=sm_scale,
            k_scale=k_scale,
            num_warps=4,
            num_stages=1,
        )
    return scores


def select_local_candidates(scores, seq_lens, rank, max_seqlen, topk=16):
    """Keep topk per head per shard, never topk/world."""
    world, batch, local = scores.shape
    keys = torch.empty((world, batch, topk), dtype=torch.int64, device=scores.device)
    if batch:
        _local_candidates[(batch, world)](
            scores,
            seq_lens,
            keys,
            BATCH=batch,
            RANK=rank,
            WORLD=world,
            MAX_BLOCKS=triton.cdiv(max_seqlen, 128),
            LOCAL_BLOCKS=local,
            TOPK=topk,
            TILE=min(512, max(topk, triton.next_power_of_2(local))),
            num_warps=4,
        )
    return keys


def merge_candidates(gathered_keys, head):
    """Consume [source,head,batch,topk] without a rearrangement copy."""
    world, _, batch, topk = gathered_keys.shape
    indices = torch.empty(
        (1, batch, topk), dtype=torch.int32, device=gathered_keys.device
    )
    if batch:
        _merge_candidates[(batch,)](
            gathered_keys,
            indices,
            BATCH=batch,
            WORLD=world,
            HEAD=head,
            TOPK=topk,
            num_warps=4,
        )
    return indices
