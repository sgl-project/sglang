from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_SAMPLE_STEP_TOKENS.get()

_BLOCK_V = 1024
_IDX_SENTINEL = tl.constexpr(2147483647)


class SampleStepTokens:
    @classmethod
    def execute(
        cls,
        *,
        step_logits: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        exp_noise: torch.Tensor,
    ) -> torch.Tensor:
        if _KERNEL_IMPL == "torch":
            return cls.torch(
                step_logits=step_logits,
                temperatures=temperatures,
                greedy_mask=greedy_mask,
                exp_noise=exp_noise,
            )
        return cls.triton(
            step_logits=step_logits,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
            exp_noise=exp_noise,
        )

    @classmethod
    def torch(
        cls,
        *,
        step_logits: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        exp_noise: torch.Tensor,
    ) -> torch.Tensor:
        return sample_step_tokens(
            step_logits=step_logits,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
            exp_noise=exp_noise,
        )

    @classmethod
    def triton(
        cls,
        *,
        step_logits: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        exp_noise: torch.Tensor,
    ) -> torch.Tensor:
        return sample_step_tokens_triton(
            step_logits=step_logits,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
            exp_noise=exp_noise,
        )


def sample_step_tokens(
    *,
    step_logits: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    exp_noise: torch.Tensor,
) -> torch.Tensor:
    probs = torch.softmax(step_logits.float() / temperatures[:, None], dim=-1)
    noise = torch.where(greedy_mask[:, None], 1.0, exp_noise)
    return probs.div_(noise).argmax(dim=-1)


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
    key = tl.exp(s - tile_max) / denom
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
    rescaled = keys * tl.exp(tile_max - global_max)
    rescaled = tl.where(mask, rescaled, -1.0)
    best = tl.max(rescaled, axis=0)
    cand = tl.where(rescaled == best, idxs, _IDX_SENTINEL)
    tl.store(next_tokens_ptr + row, tl.min(cand, axis=0).to(tl.int64))


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
