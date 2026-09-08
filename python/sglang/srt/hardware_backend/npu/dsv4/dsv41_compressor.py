"""Ascend bring-up path for the V4.1 C1/C2 compressor.

These are ordinary device-side torch operations, not the C4/C128 fused
compressor ABI. The result deliberately stops at the normalized, pre-RoPE
latent: the indexer must consume that latent before attention applies RoPE
and FP4 quantization. No indexer, quantization or KV-cache writer is called.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.layers.attention.dsv4.dsv41_compressor import (
    DeepseekV41Compressor,
    last_token_per_request,
    pair_partners_decode,
)


@dataclass
class LowRatioCompressResult:
    """Normalized BF16 latents and their destination metadata.

    ``positions`` are absolute *group-start* positions. ``out_loc`` is a full
    token-pool address divided by the ratio, never an SWA address. Extend
    returns only completed groups; decode keeps one row per input token for
    graph capture. Decode consumers MUST honor ``valid_mask``/``out_loc >= 0``:
    unfinished pairs and padding have unspecified latents and location -1.
    """

    latent: torch.Tensor
    positions: torch.Tensor
    out_loc: torch.Tensor
    valid_mask: torch.Tensor


def compress_low_ratio(
    compressor: DeepseekV41Compressor,
    x: torch.Tensor,
    *,
    positions: torch.Tensor,
    req_indices: torch.Tensor,
    raw_out_loc: torch.Tensor,
    is_decode: bool,
    state_kv: Optional[torch.Tensor] = None,
    state_score: Optional[torch.Tensor] = None,
    pad_row: Optional[int] = None,
) -> LowRatioCompressResult:
    """Compress unsharded, request-major extend or one-token/request decode.

    C2 state is owned by a KV *source layer*, indexed by persistent request
    slot, with a separate padding row. It contains the latest even token's
    FP32 projection/score. A chunk starting at odd position p requires state
    from the same request's token p-1, produced/restored before this call. This API
    does not implement prefix-state restoration or speculative rollback.
    Full token slot 0 (and negative slots) denotes padding.
    Live request indices must be in [0, pad_row), unique in decode, and
    contiguous with consecutive positions within an extend batch. Full token
    groups must be physically ratio-aligned, as in the shared GPU allocator.
    """
    ratio = compressor.compress_ratio
    if ratio not in (1, 2):
        raise ValueError(f"NPU low-ratio compressor requires C1/C2, got C{ratio}")
    if x.ndim != 2 or x.dtype != torch.bfloat16:
        raise ValueError("C1/C2 compressor expects BF16 x with shape [tokens, hidden]")
    num_tokens = x.shape[0]
    for name, value in (
        ("positions", positions),
        ("req_indices", req_indices),
        ("raw_out_loc", raw_out_loc),
    ):
        if value.ndim != 1 or value.shape[0] != num_tokens:
            raise ValueError(f"{name} must have one entry per input token")
        if value.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must use int32 or int64 metadata")
        if value.device != x.device:
            raise ValueError(f"{name} must be on the same device as x")

    if ratio == 2:
        if state_kv is None or state_score is None or pad_row is None:
            raise ValueError("C2 requires per-request KV/score state and a padding row")
        head_dim = compressor.wkv.out_features
        if (
            state_kv.ndim != 2
            or state_kv.shape != state_score.shape
            or state_kv.shape[1] != head_dim
            or pad_row != state_kv.shape[0] - 1
        ):
            raise ValueError(
                "C2 state must be [request slots + 1, head_dim], pad row last"
            )
        if state_kv.dtype != torch.float32 or state_score.dtype != torch.float32:
            raise ValueError("C2 KV and score state must remain FP32")
        if state_kv.device != x.device or state_score.device != x.device:
            raise ValueError("C2 state must be on the same device as x")

    pos = positions.to(torch.int64)
    req = req_indices.to(torch.int64)
    raw_loc = raw_out_loc.to(torch.int64)
    valid_token = (raw_loc > 0) & (pos >= 0)
    complete = valid_token & ((pos + 1) % ratio == 0)
    out_loc = torch.where(complete, raw_loc // ratio, -1)

    kv, score = compressor.project(x)
    if ratio == 1:
        pooled = kv
        group_pos = pos
    else:
        odd = pos % 2 == 1
        # Padding can carry req_pool_idx=0, which may also be a live request.
        # Never read or overwrite request 0's pending pair on behalf of padding.
        safe_req = torch.where(valid_token, req, pad_row)
        if is_decode:
            partner_kv, partner_score = pair_partners_decode(
                kv, score, odd, safe_req, state_kv, state_score
            )
        else:
            partner_kv = state_kv[safe_req].clone()
            partner_score = state_score[safe_req].clone()
            paired_in_batch = torch.zeros_like(odd)
            paired_in_batch[1:] = (
                valid_token[1:]
                & valid_token[:-1]
                & (req[1:] == req[:-1])
                & (pos[1:] == pos[:-1] + 1)
            )
            in_batch = (complete & paired_in_batch).nonzero().squeeze(1)
            partner_kv[in_batch] = kv[in_batch - 1]
            partner_score[in_batch] = score[in_batch - 1]

            # Read cross-chunk partners BEFORE storing the last even token of
            # this chunk: e.g. a chunk [1,2,3] still needs state from token 0.
            pending = last_token_per_request(valid_token & ~odd, req)
            state_kv[req[pending]] = kv[pending]
            state_score[req[pending]] = score[pending]

        group_pos = torch.where(odd, pos - 1, pos)
        if not is_decode:
            kv, score = kv[complete], score[complete]
            partner_kv = partner_kv[complete]
            partner_score = partner_score[complete]
        pooled = compressor.pool_pairs(
            torch.stack((partner_kv, kv), dim=1),
            torch.stack((partner_score, score), dim=1),
        )

    if not is_decode:
        if ratio == 1:
            pooled = pooled[complete]
        group_pos = group_pos[complete]
        out_loc = out_loc[complete]
        complete = torch.ones_like(out_loc, dtype=torch.bool)

    # Match GPU/private-reference ordering: FP32 C2 pooling -> BF16 rounding
    # -> RMSNorm with FP32 statistics/weight multiply -> BF16, still pre-RoPE.
    return LowRatioCompressResult(
        latent=compressor.finish(pooled),
        positions=group_pos,
        out_loc=out_loc,
        valid_mask=complete,
    )


def compress_low_ratio_batch(
    *,
    compressor: DeepseekV41Compressor,
    x: torch.Tensor,
    positions: torch.Tensor,
    forward_batch,
    layer_id: int,
    token_to_kv_pool,
) -> Optional[LowRatioCompressResult]:
    """Adapt ForwardBatch and the existing source-owned C2 state to the core.

    The caller must supply globally ordered (not CP-sharded) tokens. Only
    regular extend and decode are supported; reject speculative modes before
    modifying state. Idle is a no-op, matching the old compressor entry.
    """
    mode = forward_batch.forward_mode
    if mode.is_idle():
        return None
    is_decode = mode.is_decode()
    # ForwardMode.is_extend() also includes TARGET_VERIFY and DLLM_EXTEND.
    # Those need state rollback/non-causal semantics and must not mutate the
    # ordinary pending-pair state. MIXED is regular request-major prefill.
    if mode.name not in ("EXTEND", "MIXED", "DECODE"):
        raise NotImplementedError(
            "NPU C1/C2 Compress supports regular extend/decode only"
        )
    req = forward_batch.req_pool_indices.to(torch.int64)
    if not is_decode:
        req = torch.repeat_interleave(
            req,
            forward_batch.extend_seq_lens.to(torch.int64),
            output_size=x.shape[0],
        )
    # NPU allocators can expose a separate full/SWA bundle. Low ratios always
    # derive compressed slots from FULL locations, never out_swa_loc.
    bundle = getattr(forward_batch, "out_cache_loc_dsv4", None)
    raw_loc = bundle.out_full_loc if bundle is not None else forward_batch.out_cache_loc
    state_kv = state_score = pad_row = None
    if compressor.compress_ratio == 2:
        state_kv = token_to_kv_pool.c2_pair_kv_state[layer_id]
        state_score = token_to_kv_pool.c2_pair_score_state[layer_id]
        pad_row = token_to_kv_pool.c2_pair_pad_row
    return compress_low_ratio(
        compressor,
        x,
        positions=positions,
        req_indices=req,
        raw_out_loc=raw_loc,
        is_decode=is_decode,
        state_kv=state_kv,
        state_score=state_score,
        pad_row=pad_row,
    )
