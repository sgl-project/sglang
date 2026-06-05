# SPDX-License-Identifier: Apache-2.0
"""SANA-WM self-forcing chunk SAMPLER utilities (not a diffusers scheduler).

Owns the autoregressive chunk grid, per-block KV-cache accumulation /
eviction, and the explicit per-chunk sigma list for streaming (self-forcing)
SANA-WM generation. The actual stepping scheduler is the shared
``FlowMatchEulerDiscreteScheduler`` (``per_token_timesteps`` path).

Why not ``SelfForcingFlowMatchScheduler`` (the LingBot causal-DMD scheduler):

  * SANA-WM's distilled sigma grid is an explicit NON-uniform list
    ((1000, 960, 889, 727, 0)/1000); its ``set_timesteps`` only expresses
    ``linspace(sigma_max -> sigma_min)`` + shift.
  * SANA-WM pins the condition frame at timestep 0 INSIDE chunk 0 (per-frame
    timesteps within one step); its ``step`` applies a single per-sample
    sigma. LingBot instead warms the cond frames into the KV cache, so a
    scalar sigma suffices there.

The model-calling denoise loop stays in the stage.

  * segmentation FRONT-LOADS the remainder into chunk 0;
  * the KV accumulator concatenates softmax cache slots on **dim=1** (our
    ``(B, N, H, D)`` softmax cache layout);
  * GDN/STATE blocks copy-forward the previous chunk's recurrent state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.multimodal_gen.runtime.models.dits.sana_wm_components import (
    _NUM_STREAM_CACHE_SLOTS,
    _SLOT_CAM_K,
    _SLOT_CAM_V,
    _SLOT_FFN_TCONV,
    _SLOT_K,
    _SLOT_SHORTCONV,
    _SLOT_TYPE_FLAG,
    _SLOT_V,
)


@dataclass
class SanaWMSelfForcingSamplerConfig:
    """Streaming self-forcing knobs (defaults mirror ``SanaWMPipelineConfig``)."""

    num_frame_per_block: int = 3
    num_cached_blocks: int = 2
    sink_token: bool = True
    denoising_step_list: tuple = (1000, 960, 889, 727, 0)
    streaming_cfg_scale: float = 1.0

    @classmethod
    def from_pipeline_config(cls, pcfg) -> "SanaWMSelfForcingSamplerConfig":
        """Read the streaming knobs off a pipeline config (note the ``or 1.0`` cfg-scale guard)."""
        return cls(
            num_frame_per_block=int(getattr(pcfg, "num_frame_per_block", 3)),
            num_cached_blocks=int(getattr(pcfg, "num_cached_blocks", 2)),
            sink_token=bool(getattr(pcfg, "sink_token", True)),
            denoising_step_list=tuple(
                getattr(pcfg, "denoising_step_list", (1000, 960, 889, 727, 0))
            ),
            streaming_cfg_scale=float(getattr(pcfg, "streaming_cfg_scale", 1.0) or 1.0),
        )


class SanaWMSelfForcingSampler:
    """Self-forcing chunk scheduler: segmentation + per-block KV-cache carry.

    Stateless; methods are static (every input is passed explicitly). A
    ``config`` may be attached for callers that prefer the instance form.
    """

    def __init__(self, config: SanaWMSelfForcingSamplerConfig | None = None):
        self.config = config or SanaWMSelfForcingSamplerConfig()

    # ----------------------------------------------------------------- #
    # Chunk schedule + KV cache accumulation
    # ----------------------------------------------------------------- #
    @staticmethod
    def create_autoregressive_segments(
        total_frames: int, num_frame_per_block: int
    ) -> list[int]:
        base = int(num_frame_per_block)
        remained = total_frames % base
        num_chunks = total_frames // base
        chunk_indices = [0]
        for idx in range(num_chunks):
            cur = chunk_indices[-1] + base + (remained if idx == 0 else 0)
            chunk_indices.append(cur)
        return chunk_indices

    @staticmethod
    def accumulate_kv_cache(
        kv_cache: list,
        chunk_idx: int,
        chunk_indices: list[int],
        num_cached_blocks: int,
        sink_token: bool,
        num_blocks: int,
    ) -> tuple[list, int]:
        """Build chunk ``chunk_idx``'s read-only KV prefix from prior chunks.

        GDN/STATE blocks (type flag > 0.5) copy-forward the PREVIOUS chunk's
        recurrent state; softmax/CONCAT blocks concatenate the rolling-window +
        sink K/V along **dim=1** (token axis of our (B,N,H,D) softmax cache)."""
        if chunk_idx == 0:
            return kv_cache[0], 0

        cur = kv_cache[chunk_idx]
        start_chunk = max(chunk_idx - num_cached_blocks, 0) if num_cached_blocks > 0 else 0
        valid = list(range(start_chunk, chunk_idx))
        sink_num = 0
        if sink_token and num_cached_blocks > 0:
            sink_start = max(chunk_idx - num_cached_blocks + 1, 0)
            if sink_start > 0:
                valid = [0] + list(range(sink_start, chunk_idx))
                sink_num = chunk_indices[1] - chunk_indices[0]

        for block_id in range(num_blocks):
            prev_last = kv_cache[chunk_idx - 1][block_id]
            type_flag = prev_last[_SLOT_TYPE_FLAG]
            if type_flag is not None and float(type_flag.item()) > 0.5:
                # STATE (GDN) block: carry the previous chunk's recurrent state.
                cur[block_id] = [
                    prev_last[_SLOT_K],
                    prev_last[_SLOT_V],
                    prev_last[_SLOT_CAM_K],
                    prev_last[_SLOT_CAM_V],
                    prev_last[_SLOT_SHORTCONV],
                    None,
                    prev_last[_SLOT_TYPE_FLAG],
                    None,
                    None,
                    prev_last[_SLOT_FFN_TCONV],
                ]
                continue

            # CONCAT (softmax) block: concat cached K/V over the valid window.
            acc: list[torch.Tensor | None] = [None] * _NUM_STREAM_CACHE_SLOTS
            for idx in valid:
                prev = kv_cache[idx][block_id]
                if prev[_SLOT_K] is None:
                    continue
                for slot in (_SLOT_K, _SLOT_V, _SLOT_CAM_K, _SLOT_CAM_V):
                    if prev[slot] is None:
                        continue
                    acc[slot] = (
                        prev[slot].clone()
                        if acc[slot] is None
                        else torch.cat([acc[slot], prev[slot]], dim=1)  # (B,N,H,D) token axis
                    )
            cur[block_id] = [
                acc[_SLOT_K],
                acc[_SLOT_V],
                acc[_SLOT_CAM_K],
                acc[_SLOT_CAM_V],
                prev_last[_SLOT_SHORTCONV],
                None,
                prev_last[_SLOT_TYPE_FLAG],
                None,
                None,
                prev_last[_SLOT_FFN_TCONV],
            ]

        SanaWMSelfForcingSampler.evict_stale_kv_cache(
            kv_cache, chunk_idx, valid, num_cached_blocks, num_blocks
        )
        return cur, sink_num

    @staticmethod
    def evict_stale_kv_cache(
        kv_cache: list,
        chunk_idx: int,
        valid: list[int],
        num_cached_blocks: int,
        num_blocks: int,
    ) -> None:
        if num_cached_blocks <= 0:
            return
        keep = set(valid)
        keep.add(chunk_idx)
        for stale in range(chunk_idx):
            if stale in keep:
                continue
            kv_cache[stale] = [
                [None] * _NUM_STREAM_CACHE_SLOTS for _ in range(num_blocks)
            ]

    # ----------------------------------------------------------------- #
    # Per-chunk flow-Euler sigma schedule
    # ----------------------------------------------------------------- #
    @staticmethod
    def build_per_chunk_sigmas(denoising_step_list) -> list[float]:
        """Explicit flow-Euler sigmas for one chunk's short self-forcing schedule.

        The ``denoising_step_list`` (e.g. (1000, 960, 889, 727, 0)) must end with
        0; sigmas are the non-terminal steps divided by 1000."""
        schedule = list(denoising_step_list)
        if len(schedule) < 2 or schedule[-1] != 0:
            raise ValueError(f"denoising_step_list must end with 0, got {schedule}")
        return [float(t) / 1000.0 for t in schedule[:-1]]
