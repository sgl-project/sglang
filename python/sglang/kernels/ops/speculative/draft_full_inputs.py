"""Capture-owned FN HC4 bs1 inputs, published by the preceding extend graph.

Native argmax runs before publish. STEP2 remains the authority for whether its
length/slot staging can be used; a changed batch loads current inputs instead.
"""
from dataclasses import dataclass, fields, replace
import logging
import os

import torch
import triton
import triton.language as tl


@dataclass(eq=False)
class FullInputFrame:
    buffers: object
    weight: torch.Tensor
    embedding: torch.Tensor
    tokens: torch.Tensor


def make_frame(model, buffers, steps, eligible):
    if not eligible or os.environ.get("Q38_MTP_FULL_INPUT_FOLD", "0") != "1":
        return None
    from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP
    from sglang.srt.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

    if not isinstance(model, Qwen4ExpForCausalLMMTP):
        return None
    e = model.model.embed_tokens
    w = e.weight
    if not (model.hc_count == 4 and steps == 3 and e.tp_size == 1
            and isinstance(e.quant_method, UnquantizedEmbeddingMethod)
            and e.output_dtype in (None, torch.bfloat16)
            and e.num_embeddings == e.org_vocab_size
            and w.is_cuda and w.dtype == torch.bfloat16
            and w.ndim == 2 and w.shape[1] == 2560 and w.is_contiguous()
            and buffers.hidden_states is not None
            and buffers.hidden_states.shape[1] == 10240
            and buffers.rids_int is None and buffers.bootstrap_room_ids_int is None
            and buffers.draft_probs is None):
        return None
    # Including host mirrors: no captured graph shares a mutable input frame.
    private = replace(buffers, **{
        f.name: v.clone() for f in fields(buffers)
        if isinstance(v := getattr(buffers, f.name), torch.Tensor)
    })
    frame = FullInputFrame(
        private, w, torch.empty((1, 2560), dtype=w.dtype, device=w.device),
        torch.empty((1, 3), dtype=torch.int64, device=w.device),
    )
    logging.getLogger(__name__).info("Q38_FULL_INPUT_FOLD attached bs1 HC4 K3 private frame")
    return frame


@triton.jit
def _publish(Token, Hidden, Select, Seed, Seq, Pos, Req, Loc, MPos,
             Weight, DToken, DP, DHidden, DSeed, DSeq, DPos, DReq, DLoc, DMPos,
             Embedding, Chain, H: tl.constexpr, HS: tl.constexpr,
             SEED: tl.constexpr, SEED_STRIDE: tl.constexpr,
             SELECTED: tl.constexpr, HAS_MROPE: tl.constexpr,
             MSTRIDE: tl.constexpr, DMSTRIDE: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    x = tile * BLOCK + tl.arange(0, BLOCK)
    token = tl.load(Token).to(tl.int64)
    row = tl.load(Select).to(tl.int64) if SELECTED else 0
    hidden = tl.load(Hidden + row * HS + x, x < HS, other=0)
    tl.store(DHidden + x, hidden, x < HS)
    emb = tl.load(Weight + token * H + x, x < H, other=0)
    tl.store(Embedding + x, emb, x < H)
    if SEED:
        seed = tl.load(Seed + row * SEED_STRIDE + x, x < SEED, other=0)
        tl.store(DSeed + x, seed, x < SEED)
    if tile == 0:
        tl.store(DToken, token)
        tl.store(DP, 1.0)
        tl.store(Chain, token)
        tl.store(DSeq, tl.load(Seq))
        tl.store(DPos, tl.load(Pos))
        tl.store(DReq, tl.load(Req))
        loc = tl.load(Loc + x, x < 3, other=0)
        tl.store(DLoc + x, loc, x < 3)
        if HAS_MROPE:
            mpos = tl.load(MPos + x * MSTRIDE, x < 3, other=0)
        else:
            mpos = tl.full((BLOCK,), 0, tl.int64)
        tl.store(DMPos + x * DMSTRIDE, mpos, x < 3)


def publish(frame, *, token, hidden, seed, source, select=None, request_slots=None):
    b = frame.buffers
    seed_width = b.dsa_seed_topk.shape[1] if b.dsa_seed_topk is not None else 0
    # Missing seed in a fallback round must be zero, matching the native fill.
    if seed_width and seed is None:
        b.dsa_seed_topk[:1].zero_()
        seed_width = 0
    dummy = token
    _publish[(triton.cdiv(max(10240, seed_width), 256),)](
        token, hidden, select if select is not None else dummy,
        seed if seed is not None else dummy,
        source.seq_lens, source.positions,
        source.req_pool_indices if request_slots is None else request_slots, source.out_cache_loc,
        source.mrope_positions if source.mrope_positions is not None else dummy,
        frame.weight, b.topk_index, b.topk_p, b.hidden_states,
        b.dsa_seed_topk if b.dsa_seed_topk is not None else dummy,
        b.seq_lens, b.positions, b.req_pool_indices, b.out_cache_loc, b.mrope_positions,
        frame.embedding, frame.tokens,
        H=2560, HS=10240, SEED=seed_width,
        SEED_STRIDE=seed.stride(0) if seed is not None else 0,
        SELECTED=select is not None, HAS_MROPE=source.mrope_positions is not None,
        MSTRIDE=source.mrope_positions.stride(0) if source.mrope_positions is not None else 0,
        DMSTRIDE=b.mrope_positions.stride(0), BLOCK=256,
        num_warps=4,
    )
