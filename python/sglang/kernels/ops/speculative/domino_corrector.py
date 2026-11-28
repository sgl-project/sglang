"""Steps of the Domino transformer corrector's rollout, one launch each.

The corrector advances a draft block slot by slot with two streams per request
(``sglang.srt.speculative.domino_corrector``). Each slot attends twice per block,
to the slots before it and to a window of committed target context, and ends by
drawing its token from a candidate pool. Each is a small problem per request,
where the cost of the generic route is its launches.

Coverage: CUDA. The attention steps take BF16 or FP32 activations and a
power-of-two head dimension; products and the softmax accumulate in FP32 and the
result is rounded once on store. The draw takes FP32 scores over at most
``DRAW_MAX_POOL`` candidates.
"""

import torch
import triton
import triton.language as tl

# Arbitrary: any finite value below every real score keeps the running maximum finite.
_MASKED = tl.constexpr(-1e30)
_NEG_INF = tl.constexpr(float("-inf"))
# The smallest positive normal FP32: a zero draw would divide by zero.
_TINY = tl.constexpr(1.1754943508222875e-38)
# One program holds a request's whole pool in registers.
DRAW_MAX_POOL = 8192


@triton.jit
def _history_attention_kernel(
    QKV,
    HISTORY,
    OUT,
    slot,
    WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SLOTS: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
):
    pid = tl.program_id(0)
    head = pid % HEADS
    stream = (pid // HEADS) % 2
    request = pid // (2 * HEADS)
    dim = tl.arange(0, HEAD_DIM)
    # [request, stream, 3 * WIDTH]: the query, then each head's key and value.
    token_row = QKV + request * 6 * WIDTH
    query = tl.load(token_row + stream * 3 * WIDTH + head * HEAD_DIM + dim)
    # Only the token stream (0) contributes keys and values, to both streams.
    current = token_row + WIDTH + head * 2 * HEAD_DIM
    key = tl.load(current + dim)
    value = tl.load(current + HEAD_DIM + dim)
    # [request, head, slot, 2 * HEAD_DIM]. The other stream reads this slot from
    # QKV above, never from HISTORY, so the store races with no load.
    past = HISTORY + (request * HEADS + head) * SLOTS * 2 * HEAD_DIM
    append = (dim < HEAD_DIM) & (stream == 0)
    tl.store(past + slot * 2 * HEAD_DIM + dim, key, mask=append)
    tl.store(past + slot * 2 * HEAD_DIM + HEAD_DIM + dim, value, mask=append)

    query = query.to(tl.float32)
    index = tl.arange(0, BLOCK_SLOTS)
    seen = index < slot
    rows = past + index[:, None] * 2 * HEAD_DIM + dim[None, :]
    keys = tl.load(rows, mask=seen[:, None], other=0.0).to(tl.float32)
    values = tl.load(rows + HEAD_DIM, mask=seen[:, None], other=0.0).to(tl.float32)
    scores = tl.where(seen, tl.sum(keys * query[None, :], axis=1), _MASKED)
    score = tl.sum(key.to(tl.float32) * query, axis=0)
    top = tl.maximum(tl.max(scores, axis=0), score)
    weights = tl.where(seen, tl.exp(scores - top), 0.0)
    weight = tl.exp(score - top)
    attended = tl.sum(weights[:, None] * values, axis=0) + weight * value.to(tl.float32)
    attended = attended / (tl.sum(weights, axis=0) + weight)
    tl.store(OUT + (request * 2 + stream) * WIDTH + head * HEAD_DIM + dim, attended)


@triton.jit
def _context_attention_kernel(
    QUERY,
    CONTEXT,
    VALID,
    COS,
    SIN,
    OUT,
    WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    HALF: tl.constexpr,
    WINDOW: tl.constexpr,
    LAYERS: tl.constexpr,
    LAYER: tl.constexpr,
    TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    head = pid % HEADS
    row = pid // HEADS
    request = row // 2
    dim = tl.arange(0, HALF)
    query = QUERY + row * WIDTH + head * 2 * HALF
    query_left = tl.load(query + dim).to(tl.float32)
    query_right = tl.load(query + HALF + dim).to(tl.float32)
    # [request, window, layer, 2 * WIDTH]: each window row's key, then its value.
    stride = LAYERS * 2 * WIDTH
    keys = CONTEXT + request * WINDOW * stride + LAYER * 2 * WIDTH + head * 2 * HALF

    top = tl.full([], _MASKED, tl.float32)
    total = tl.zeros([], tl.float32)
    left = tl.zeros([HALF], tl.float32)
    right = tl.zeros([HALF], tl.float32)
    for start in range(0, WINDOW, TILE):
        index = start + tl.arange(0, TILE)
        inside = index < WINDOW
        valid = inside & (
            tl.load(VALID + request * WINDOW + index, mask=inside, other=0) != 0
        )
        rows = keys + index[:, None] * stride + dim[None, :]
        take = inside[:, None]
        key_left = tl.load(rows, mask=take, other=0.0).to(tl.float32)
        key_right = tl.load(rows + HALF, mask=take, other=0.0).to(tl.float32)
        # The rotation of a window row, by its offset from the block.
        table = index[:, None] * HALF + dim[None, :]
        cos = tl.load(COS + table, mask=take, other=0.0)
        sin = tl.load(SIN + table, mask=take, other=0.0)
        scores = tl.sum(
            query_left[None, :] * (key_left * cos - key_right * sin)
            + query_right[None, :] * (key_right * cos + key_left * sin),
            axis=1,
        )
        scores = tl.where(valid, scores, _MASKED)
        new_top = tl.maximum(top, tl.max(scores, axis=0))
        carry = tl.exp(top - new_top)
        weights = tl.where(valid, tl.exp(scores - new_top), 0.0)[:, None]
        value_left = tl.load(rows + WIDTH, mask=take, other=0.0).to(tl.float32)
        value_right = tl.load(rows + WIDTH + HALF, mask=take, other=0.0).to(tl.float32)
        left = left * carry + tl.sum(weights * value_left, axis=0)
        right = right * carry + tl.sum(weights * value_right, axis=0)
        total = total * carry + tl.sum(weights)
        top = new_top
    # A request with no valid window row attends to nothing.
    scale = tl.where(total > 0.0, 1.0 / total, 0.0)
    out = OUT + row * WIDTH + head * 2 * HALF
    tl.store(out + dim, left * scale)
    tl.store(out + HALF + dim, right * scale)


@triton.jit
def _draw_kernel(
    SCORES,
    NOISE,
    TEMPERATURES,
    GREEDY,
    CANDIDATES,
    TOKENS,
    Q,
    POOL: tl.constexpr,
    BLOCK_POOL: tl.constexpr,
):
    request = tl.program_id(0)
    index = tl.arange(0, BLOCK_POOL)
    inside = index < POOL
    row = request * POOL + index
    scores = tl.load(SCORES + row, mask=inside, other=_NEG_INF)
    greedy = tl.load(GREEDY + request) != 0
    top = tl.argmax(scores, axis=0, tie_break_left=True)
    scaled = scores / tl.load(TEMPERATURES + request)
    weights = tl.exp(scaled - tl.max(scaled, axis=0))
    q = weights / tl.sum(weights, axis=0)
    # Exponential race: argmax(q / E), E ~ Exp(1), is an exact draw from q.
    noise = tl.maximum(tl.load(NOISE + row, mask=inside, other=1.0), _TINY)
    drawn = tl.argmax(q / noise, axis=0, tie_break_left=True)
    choice = tl.where(greedy, top, drawn)
    tl.store(TOKENS + request, tl.load(CANDIDATES + request * POOL + choice))
    point_mass = tl.where(index == top, 1.0, 0.0)
    tl.store(Q + row, tl.where(greedy, point_mass, q), mask=inside)


def _check(x: torch.Tensor, *, width: int, heads: int) -> int:
    assert x.is_cuda and x.is_contiguous()
    assert x.dtype in (torch.bfloat16, torch.float32)
    head_dim = width // heads
    assert head_dim * heads == width, (width, heads)
    assert head_dim >= 2 and head_dim & (head_dim - 1) == 0, head_dim
    return head_dim


def corrector_history_attention(
    qkv: torch.Tensor, history: torch.Tensor, slot: int, *, heads: int
) -> torch.Tensor:
    """Attend a slot's two streams to the slots up to it, and record the slot.

    ``qkv`` is [batch, 2 streams, 3 * width]: a stream's query, then each head's
    key and value. ``history`` is [batch, heads, slots, 2 * head_dim], a slot's key
    then its value: slot ``slot`` is written from the token stream (0) and the
    slots before it are read. Returns softmax(query . key) @ value over slots
    ``0 .. slot``, [batch, 2, width]. The query carries the attention scale.
    """
    bs, streams, triple = qkv.shape
    width = triple // 3
    head_dim = _check(qkv, width=width, heads=heads)
    slots = history.shape[2]
    assert streams == 2 and triple == 3 * width and 0 <= slot < slots
    assert history.shape == (bs, heads, slots, 2 * head_dim)
    assert history.dtype == qkv.dtype and history.is_contiguous()
    out = qkv.new_empty((bs, 2, width))
    if bs:
        _history_attention_kernel[(bs * 2 * heads,)](
            qkv,
            history,
            out,
            slot,
            WIDTH=width,
            HEADS=heads,
            HEAD_DIM=head_dim,
            SLOTS=slots,
            BLOCK_SLOTS=triton.next_power_of_2(slots),
            num_warps=1,
        )
    return out


def corrector_context_attention(
    query: torch.Tensor,
    context: torch.Tensor,
    valid: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    layer: int,
    heads: int,
) -> torch.Tensor:
    """Attend a slot's two streams to a window of projected target context.

    ``query`` is [batch, 2 streams, width] and carries the attention scale.
    ``context`` is [batch, window, layers, 2 * width], a window row's key then its
    value for each layer; ``valid`` [batch, window] (uint8) marks the rows a request
    has. ``cos`` / ``sin`` are [window, head_dim / 2] FP32: the rotation applied to
    a row's key, pairing dimension ``d`` with ``d + head_dim / 2``. Returns
    softmax(query . rotated key) @ value over the valid rows, [batch, 2, width],
    and zeros for a request with none.
    """
    bs, streams, width = query.shape
    head_dim = _check(query, width=width, heads=heads)
    half = head_dim // 2
    window, layers = context.shape[1], context.shape[2]
    assert streams == 2 and context.shape == (bs, window, layers, 2 * width)
    assert context.dtype == query.dtype and context.is_contiguous()
    assert valid.shape == (bs, window) and valid.dtype == torch.uint8
    assert cos.shape == sin.shape == (window, half)
    assert cos.dtype == sin.dtype == torch.float32
    assert cos.is_contiguous() and sin.is_contiguous() and valid.is_contiguous()
    out = torch.empty_like(query)
    if bs:
        _context_attention_kernel[(bs * 2 * heads,)](
            query,
            context,
            valid,
            cos,
            sin,
            out,
            WIDTH=width,
            HEADS=heads,
            HALF=half,
            WINDOW=window,
            LAYERS=layers,
            LAYER=layer,
            # Measured on B300 at window 256: one 256-row pass at 4 warps takes
            # 3.9 us, 32-row tiles 10.3 us; re-tune when the kernel changes.
            TILE=min(256, triton.next_power_of_2(window)),
            num_warps=4,
        )
    return out


def corrector_draw(
    scores: torch.Tensor,
    noise: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw each request's token from q = softmax(scores / T) over its candidates.

    ``scores`` and ``noise`` are [batch, K] FP32, ``noise`` holding Exp(1) draws;
    ``temperatures`` [batch] FP32 is positive; ``greedy_mask`` [batch] is uint8;
    ``candidate_ids`` [batch, K] is int64. A sampled row takes
    argmax(q / noise) and a greedy row argmax(scores), the first on a tie.
    Returns the tokens, [batch] int64, and each row's q, [batch, K] FP32: the
    point mass on its token for a greedy row.
    """
    bs, pool = scores.shape
    assert scores.is_cuda and 0 < pool <= DRAW_MAX_POOL
    assert scores.dtype == noise.dtype == temperatures.dtype == torch.float32
    assert greedy_mask.dtype == torch.uint8 and candidate_ids.dtype == torch.int64
    assert noise.shape == candidate_ids.shape == (bs, pool)
    assert temperatures.shape == greedy_mask.shape == (bs,)
    for tensor in (scores, noise, temperatures, greedy_mask, candidate_ids):
        assert tensor.is_contiguous()
    tokens = candidate_ids.new_empty((bs,))
    q = torch.empty_like(scores)
    if bs:
        _draw_kernel[(bs,)](
            scores,
            noise,
            temperatures,
            greedy_mask,
            candidate_ids,
            tokens,
            q,
            POOL=pool,
            BLOCK_POOL=triton.next_power_of_2(pool),
            num_warps=8,
        )
    return tokens, q
