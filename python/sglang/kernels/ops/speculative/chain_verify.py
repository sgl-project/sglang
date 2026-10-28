"""Greedy verification of top-1 draft chains in one launch.

The verify step's target tokens are the argmax of its logits rows; a request's
drafts are accepted while each equals the target token of the row before it.
``verify_tree_greedy`` does that walk once ``row_argmax`` has reduced the rows,
on buffers filled beforehand. Here one launch does all of it: each CTA reduces a
vocabulary split of a row and folds its (value, lowest index) into the row's
64-bit key with an atomic max; the last CTA to arrive turns the keys into the
target tokens and walks every request's chain.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

_BLOCK = 8192
INT64_MIN = tl.constexpr(-(2**63))


@triton.jit
def _greedy_chain_verify_kernel(
    logits,
    logits_row_stride,
    best,
    arrived,
    target_predict,
    candidates,
    retrieve_index,
    retrieve_next_token,
    predicts,
    accept_index,
    accept_token_num,
    accept_lens,
    bs,
    n_programs,
    vocab_size: tl.constexpr,
    W: tl.constexpr,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS_BLOCK: tl.constexpr,
    BS_BLOCK: tl.constexpr,
    WITH_ACCEPT_LENS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    offsets = split * BLOCK + tl.arange(0, BLOCK)
    vals = tl.load(
        logits + row * logits_row_stride + offsets,
        mask=offsets < vocab_size,
        other=-float("inf"),
    )
    vals = tl.where(vals == vals, vals, -1e30)  # NaN reads as -1e30
    max_val = tl.max(vals, axis=0)
    max_val = tl.where(max_val == 0.0, 0.0, max_val)  # -0.0 as +0.0: equal values tie
    index = split * BLOCK + tl.argmax(vals, axis=0)
    # the value's bits ordered as signed integers, then the index reversed, so a
    # larger key is a larger value or, on a tie, a lower index
    bits = max_val.to(tl.int32, bitcast=True)
    bits = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    key = (bits.to(tl.int64) << 32) | (0x7FFFFFFF - index).to(tl.int64)
    tl.atomic_max(best + row, key, sem="relaxed")
    if tl.atomic_add(arrived, 1, sem="acq_rel") == n_programs - 1:
        # The last CTA: every row's key is final. It walks each request's chain
        # (verify_tree_greedy with no siblings) over all W nodes, so every predict
        # is written once: the target token on the accepted path and its bonus, 0
        # after it. A row's target is read with an atomic from its key, so no
        # thread reads what another stored.
        b = tl.arange(0, BS_BLOCK)
        req = b < bs
        node = tl.zeros((BS_BLOCK,), dtype=tl.int64)
        pos = tl.load(retrieve_index + b * W, mask=req, other=0).to(tl.int64)
        on = req
        n = tl.zeros((BS_BLOCK,), dtype=tl.int32)
        # the previous node's target token, which node j > 0's draft must equal
        target = tl.zeros((BS_BLOCK,), dtype=tl.int64)
        for j in tl.static_range(W):
            if j > 0:
                node = tl.load(retrieve_next_token + b * W + node, mask=req, other=-1)
                walk = req & (node != -1)
                node = tl.where(walk, node, 0)
                pos = tl.load(retrieve_index + b * W + node, mask=walk, other=0).to(
                    tl.int64
                )
                draft = tl.load(candidates + b * W + node, mask=walk, other=-1)
                on = on & walk & (draft == target)
                n += on.to(tl.int32)
                if j < S:
                    tl.store(
                        accept_index + b * S + j,
                        tl.where(on, pos, -1).to(accept_index.dtype.element_ty),
                        mask=req,
                    )
            else:
                walk = req
                tl.store(
                    accept_index + b * S,
                    pos.to(accept_index.dtype.element_ty),
                    mask=req,
                )
            k = tl.atomic_add(best + pos, 0, mask=walk, sem="relaxed")
            target = 0x7FFFFFFF - (k & 0x7FFFFFFF)
            tl.store(
                predicts + pos,
                tl.where(on, target, 0).to(predicts.dtype.element_ty),
                mask=walk,
            )
        tl.store(
            accept_token_num + b, n.to(accept_token_num.dtype.element_ty), mask=req
        )
        if WITH_ACCEPT_LENS:
            tl.store(
                accept_lens + b, (n + 1).to(accept_lens.dtype.element_ty), mask=req
            )
        # every read above has returned; now read the keys out once more and reset
        tl.debug_barrier()
        r = tl.arange(0, ROWS_BLOCK)
        rows = r < bs * W
        # (an atomic min to INT64_MIN: the exchange has no buffer-atomic form)
        keys = tl.atomic_min(best + r, INT64_MIN, mask=rows, sem="relaxed")
        tl.store(target_predict + r, 0x7FFFFFFF - (keys & 0x7FFFFFFF), mask=rows)
        tl.atomic_xchg(arrived, 0, sem="relaxed")


# per device: each row's best key and the launch's arrival count, left reset by
# every launch (a device's verify steps run in stream order)
_STATE: dict = {}
# outgrown states stay allocated: captured graphs keep their addresses
_OUTGROWN: list = []


def _state(device: torch.device, rows: int):
    state = _STATE.get(device)
    if state is None or state[0].shape[0] < rows:
        if state is not None:
            _OUTGROWN.append(state)
        n = max(rows, 1024)
        state = (
            torch.full((n,), -(2**63), dtype=torch.int64, device=device),
            torch.zeros((1,), dtype=torch.int32, device=device),
        )
        _STATE[device] = state
    return state


def greedy_chain_verify(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    num_speculative_tokens: int,
    accept_lens: Optional[torch.Tensor] = None,
):
    """Greedy verify of top-1 draft chains from f32 logits [bs * W, vocab], in one
    launch: (predicts int32 [bs * W], accept_index int32 [bs, S], accept_token_num
    int32 [bs]: consecutive views of one buffer; target_predict int64 [bs, W]; the
    buffer), as ``row_argmax`` (ties to the
    lowest index) then ``verify_tree_greedy`` on zeroed predicts and an accept
    index filled with -1 give them. ``retrieve_next_token`` holds each request's
    chain (no siblings); NaN reads as -1e30. accept_lens (optional, [bs]): also
    written, accept_token_num + 1."""
    assert logits.dim() == 2 and logits.dtype == torch.float32
    assert logits.stride(1) == 1
    bs, W = candidates.shape
    S = num_speculative_tokens
    rows, vocab = logits.shape
    assert rows == bs * W and S == W
    for t in (candidates, retrieve_index, retrieve_next_token):
        assert t.shape == (bs, W) and t.is_contiguous()
    dev = logits.device
    # consecutive views of one buffer, so a broadcast of the results needs no packing
    out = torch.empty((rows + bs * S + bs,), dtype=torch.int32, device=dev)
    predicts = out[:rows]
    accept_index = out[rows : rows + bs * S].view(bs, S)
    accept_token_num = out[rows + bs * S :]
    target_predict = torch.empty((bs, W), dtype=torch.int64, device=dev)
    if bs == 0:
        return predicts, accept_index, accept_token_num, target_predict, out
    best, arrived = _state(dev, rows)
    splits = triton.cdiv(vocab, _BLOCK)
    _greedy_chain_verify_kernel[(rows, splits)](
        logits,
        logits.stride(0),
        best,
        arrived,
        target_predict,
        candidates,
        retrieve_index,
        retrieve_next_token,
        predicts,
        accept_index,
        accept_token_num,
        accept_token_num if accept_lens is None else accept_lens,
        bs,
        rows * splits,
        vocab_size=vocab,
        W=W,
        S=S,
        BLOCK=_BLOCK,
        ROWS_BLOCK=triton.next_power_of_2(rows),
        BS_BLOCK=triton.next_power_of_2(bs),
        WITH_ACCEPT_LENS=accept_lens is not None,
        num_warps=8,
    )
    return predicts, accept_index, accept_token_num, target_predict, out
