"""NPU compatibility sampler for the newer MarkovSampleStep interface.

The tiled selection kernels below are adapted from sglang commit
95325f36ec1879a16b474d4f07c4846c97b9cef3, dspark_draft_model.py, with
guards for entirely masked (-inf) tiles during the online-max rescaling.
The new branch keeps its bias projection, step-logit dtype, rejection sampling,
confidence scheduling, and KV logic. Noise comes from the native NPU generator
as in the older branch, rather than the new counter-based Philox implementation.
Consequently sampled sequences need not be identical to the newer RNG path.

Scope: one vocabulary shard per attention-DP rank, as in TP16/attention-DP16.
Multi-rank vocabulary sampling is intentionally rejected rather than allowing
unsynchronized native RNG draws on replicas of the gathered logits.
"""

import logging
from typing import Callable, Optional

import torch
import triton
import triton.language as tl

_BLOCK_V = 1024
_IDX_SENTINEL = tl.constexpr(2147483647)
_reported = False
logger = logging.getLogger(__name__)


@triton.jit
def _online_partial_kernel(
    logits_ptr,
    temperatures_ptr,
    greedy_mask_ptr,
    exp_noise_ptr,
    tile_max_ptr,
    partial_key_ptr,
    partial_idx_ptr,
    V,
    stride_row,
    n_tiles,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = offs < V
    logits = tl.load(
        logits_ptr + row * stride_row + offs, mask=mask, other=float("-inf")
    ).to(tl.float32)
    temperature = tl.load(temperatures_ptr + row)
    s = logits / temperature
    tile_max = tl.max(s, axis=0)
    greedy = tl.load(greedy_mask_ptr + row) != 0
    noise = tl.load(exp_noise_ptr + row * V + offs, mask=mask, other=1.0)
    denom = tl.where(greedy, 1.0, noise)
    shifted = tl.where(tile_max != -float("inf"), s - tile_max, -float("inf"))
    key = tl.exp(shifted) / denom
    key = tl.where(mask, key, -1.0)
    tile_best = tl.max(key, axis=0)
    idx = tl.where(key == tile_best, offs, _IDX_SENTINEL)
    tl.store(tile_max_ptr + row * n_tiles + tile, tile_max)
    tl.store(partial_key_ptr + row * n_tiles + tile, tile_best)
    tl.store(partial_idx_ptr + row * n_tiles + tile, tl.min(idx, axis=0))


@triton.jit
def _online_combine_kernel(
    tile_max_ptr,
    partial_key_ptr,
    partial_idx_ptr,
    next_tokens_ptr,
    n_tiles,
    BLOCK_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_TILES)
    mask = offs < n_tiles
    tile_max = tl.load(
        tile_max_ptr + row * n_tiles + offs, mask=mask, other=float("-inf")
    )
    keys = tl.load(partial_key_ptr + row * n_tiles + offs, mask=mask, other=-1.0)
    idxs = tl.load(
        partial_idx_ptr + row * n_tiles + offs, mask=mask, other=_IDX_SENTINEL
    )
    global_max = tl.max(tile_max, axis=0)
    shifted = tl.where(
        global_max != -float("inf"), tile_max - global_max, -float("inf")
    )
    rescaled = keys * tl.exp(shifted)
    rescaled = tl.where(mask, rescaled, -1.0)
    best = tl.max(rescaled, axis=0)
    cand = tl.where(rescaled == best, idxs, _IDX_SENTINEL)
    next_token = tl.min(cand, axis=0)
    # Degenerate rows (e.g. all -inf logits) leave cand all-sentinel; clamp to 0
    # so a valid token id is emitted instead of an out-of-range 2147483647.
    next_token = tl.where(next_token == _IDX_SENTINEL, 0, next_token)
    tl.store(next_tokens_ptr + row, next_token.to(tl.int64))


def sample_step_tokens_triton(
    *,
    step_logits: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    exp_noise: torch.Tensor,
) -> torch.Tensor:
    bs, V = step_logits.shape
    device = step_logits.device
    assert step_logits.stride(1) == 1, "step_logits rows must be contiguous"
    stride_row = step_logits.stride(0)
    temperatures = temperatures.to(torch.float32).contiguous()
    greedy_mask = greedy_mask.to(torch.int32).contiguous()
    exp_noise = exp_noise.to(torch.float32).contiguous()

    n_tiles = triton.cdiv(V, _BLOCK_V)
    block_tiles = triton.next_power_of_2(n_tiles)

    tile_max = torch.empty((bs, n_tiles), dtype=torch.float32, device=device)
    partial_key = torch.empty((bs, n_tiles), dtype=torch.float32, device=device)
    partial_idx = torch.empty((bs, n_tiles), dtype=torch.int32, device=device)
    next_tokens = torch.empty((bs,), dtype=torch.int64, device=device)

    tile_grid = (bs, n_tiles)
    row_grid = (bs,)

    _online_partial_kernel[tile_grid](
        step_logits,
        temperatures,
        greedy_mask,
        exp_noise,
        tile_max,
        partial_key,
        partial_idx,
        V,
        stride_row,
        n_tiles,
        BLOCK_V=_BLOCK_V,
    )
    _online_combine_kernel[row_grid](
        tile_max,
        partial_key,
        partial_idx,
        next_tokens,
        n_tiles,
        BLOCK_TILES=block_tiles,
    )
    return next_tokens


def sample_markov_step_npu(
    *,
    base_logits: torch.Tensor,
    bias: torch.Tensor,
    temperatures: torch.Tensor,
    key: int,
    offset: torch.Tensor,
    step: int,
    step_dtype: torch.dtype,
    vocab_start: int = 0,
    gather: Optional[Callable] = None,
    gather_logits: Optional[Callable] = None,
    logits_out: Optional[torch.Tensor] = None,
    has_sampled_rows: Optional[bool] = None,
) -> torch.Tensor:
    global _reported
    del key, offset, step  # Native server-seeded generator replaces counter RNG.
    if base_logits.device.type != "npu":
        raise ValueError("The NPU compatibility sampler requires NPU tensors.")
    if base_logits.ndim != 2 or bias.ndim != 2:
        raise ValueError("base_logits and bias must be 2-D.")
    rows, padded = base_logits.shape
    width = bias.shape[1]
    if bias.shape[0] != rows or not (0 < width <= padded):
        raise ValueError("Unexpected Markov bias shape or empty vocabulary shard.")
    if temperatures.shape != (rows,):
        raise ValueError("temperatures must contain one value per row.")
    if step_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError(f"Unsupported step-logit dtype: {step_dtype}")
    if vocab_start != 0:
        raise ValueError(
            "This compatibility sampler supports a single full-vocabulary rank only."
        )
    if gather is not None and gather_logits is None:
        raise ValueError("A sharded head must provide gather_logits.")
    if rows == 0:
        return torch.empty((0,), device=base_logits.device, dtype=torch.int64)

    # Match the newer implementation's FP32 add and round to step dtype. Do not
    # change logits_out's storage: the graph and rejection sampler retain it.
    value = base_logits.to(torch.float32, copy=True)
    value[:, :width].add_(bias.float())
    if width < padded:
        value[:, width:] = -torch.inf
    value = value.to(step_dtype)
    if gather is not None:
        gathered = gather_logits(value)
        if gathered.shape[0] != rows or gathered.shape[1] > padded:
            raise ValueError(
                "Multi-rank vocabulary gather is unsupported by this native-noise adapter; "
                "use TP size equal to attention-DP size with DP LM head."
            )
        value = gathered
    if logits_out is not None:
        if value.shape[1] < logits_out.shape[1] or logits_out.shape[0] != rows:
            raise ValueError("Gathered step logits do not cover logits_out.")
        value = value[:, : logits_out.shape[1]]
        if logits_out.dtype != step_dtype:
            raise ValueError("logits_out dtype differs from step_dtype.")
        logits_out.copy_(value)
        step_logits = logits_out
    else:
        step_logits = value

    # temperature=0 means argmax. A nonzero safe temperature keeps the legacy
    # kernel from dividing by zero; greedy rows use unit noise and still argmax.
    if has_sampled_rows is False:
        greedy_mask = torch.ones(rows, dtype=torch.int32, device=value.device)
        safe_temperatures = torch.ones(rows, dtype=torch.float32, device=value.device)
        exp_noise = torch.ones(
            step_logits.shape, dtype=torch.float32, device=value.device
        )
    else:
        greedy_rows = temperatures.eq(0)
        greedy_mask = greedy_rows.to(torch.int32)
        safe_temperatures = torch.where(
            greedy_rows, 1.0, temperatures.float().clamp_min(1e-5)
        )
        exp_noise = torch.empty(
            step_logits.shape, dtype=torch.float32, device=value.device
        )
        exp_noise.exponential_(1.0)
    if not _reported:
        logger.info(
            "DSpark NPU sampler: legacy Triton selection + native exponential RNG; "
            "score/token-ID buffers are separate float32/int32 tensors."
        )
        _reported = True
    return sample_step_tokens_triton(
        step_logits=step_logits,
        temperatures=safe_temperatures,
        greedy_mask=greedy_mask,
        exp_noise=exp_noise,
    )
