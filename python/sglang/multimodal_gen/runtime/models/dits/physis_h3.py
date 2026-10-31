# SPDX-License-Identifier: Apache-2.0
"""Same-step noisy video history for the Physis-AI H3 checkpoint.

Keys are normalized, but cached before RoPE. Rebasing the rolling window must
preserve the five-latent VAE clock, while the opening sink stays at position zero.
Text is recomputed with each chunk and attends the same history as video.
"""

import math

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalSelfAttentionKVCache,
)
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all_qkv,
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    _apply_qk_norm,
    _apply_rope,
    _rope_cos_sin_cache,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    _axis_from_sqrt_area,
    _video_t_grid,
)

PHYSIS_H3_SIGMAS = (1.0, 0.9097071290016174, 0.7173174619674683, 0.0244140625, 0.0)


def physis_h3_frame_positions(chunk: int, history: int) -> list[int]:
    if chunk < 0 or history < 0:
        raise ValueError("Chunk and history must be non-negative")
    if chunk == 0:
        return [0]
    first = max(1, chunk - history)
    offset = (first - 1) // 5 * 5
    return [0, *range(first - offset, chunk + 1 - offset)]


def physis_h3_positions(text_len, height, width, frames):
    """H3 coordinates for a text prefix and selected latent frames."""
    text = torch.zeros(text_len, 3, dtype=torch.float64)
    text[:, 0] = torch.arange(text_len)
    area = math.sqrt(height * width)
    times = _video_t_grid(max(frames) + 1, float(text_len))[frames]
    grid = torch.meshgrid(
        times,
        _axis_from_sqrt_area(height, 2, area),
        _axis_from_sqrt_area(width, 2, area),
        indexing="ij",
    )
    video = torch.stack(grid, dim=-1).reshape(-1, 3)
    return torch.cat((text, video))


def physis_h3_step(sample, prediction, sigma, next_sigma, noise):
    # H3 predicts x0-epsilon, the opposite sign of the usual flow velocity.
    clean = (sample.float() + sigma * prediction.float()).to(sample.dtype)
    if next_sigma == 0:
        return clean
    return ((1 - next_sigma) * clean.float() + next_sigma * noise).to(sample.dtype)


class PhysisH3AttentionContext:
    def __init__(
        self, model, text_len, height, width, history, sp_size=1, chunk_frames=1
    ):
        self.model = model
        self.text_len = text_len
        self.height, self.width = height, width
        self.frame_tokens = height // 2 * (width // 2)
        self.chunk_frames = chunk_frames
        self.query_tokens = self.frame_tokens * chunk_frames
        self.live_tokens = text_len + self.query_tokens
        self.history = history
        self.sp_size = sp_size
        self.caches = {}
        self.sinks = {}
        self.chunk = 0
        self.step = 0
        self.layer = 0
        self.sink_only = False
        self.query_rope = self.key_rope = None

    def set_chunk(self, chunk):
        self.chunk = chunk
        if self.chunk_frames > 1:
            if chunk != 0:
                raise ValueError("Multi-frame H3 chunks are only used for bootstrap")
            frames = list(range(self.chunk_frames))
        else:
            frames = physis_h3_frame_positions(chunk, self.history)
        positions = physis_h3_positions(
            self.text_len, self.height, self.width, frames
        ).to(self.model.rope.inv_freq.device)
        self.key_rope = _rope_cos_sin_cache(
            self.model.rope(positions[None]), dtype=torch.bfloat16
        )
        self.query_rope = torch.cat(
            (self.key_rope[: self.text_len], self.key_rope[-self.query_tokens :])
        )

    def _new_cache(self, key):
        capacity = self.frame_tokens * (
            self.chunk_frames if self.sink_only else self.history + 2
        )
        shape = (1, capacity, *key.shape[1:])
        return CausalSelfAttentionKVCache(
            k=key.new_empty(shape),
            v=key.new_empty(shape),
            global_end_index=torch.zeros((), device=key.device, dtype=torch.long),
            local_end_index=torch.zeros((), device=key.device, dtype=torch.long),
            global_end_index_int=0,
            local_end_index_int=0,
            sink_tokens=self.frame_tokens,
        )

    def attend(self, attention, q, k, v):
        q, k = _apply_qk_norm(
            q, k, attention.q_norm, attention.k_norm, attention.head_dim
        )
        if self.sp_size > 1:
            q, k, v = (
                x[0] for x in _usp_input_all_to_all_qkv(q[None], k[None], v[None])
            )
        padded_tokens = q.shape[0]
        q, k, v = (x[: self.live_tokens] for x in (q, k, v))
        cache_key = (self.step, self.layer)
        cache = self.caches.get(cache_key)
        if cache is None:
            cache = self.caches[cache_key] = self._new_cache(k)
            if self.chunk > 0:
                sink_k, sink_v = self.sinks[self.layer]
                cache.update_and_get_attention_kv(
                    key=sink_k, value=sink_v, current_chunk_start=0
                )
        visible = cache.update_and_get_attention_kv(
            key=k[None, self.text_len :],
            value=v[None, self.text_len :],
            current_chunk_start=self.chunk * self.frame_tokens,
        )
        keys = torch.cat((k[: self.text_len], visible.k[0]))
        values = torch.cat((v[: self.text_len], visible.v[0]))
        q = _apply_rope(q, self.query_rope)
        keys = _apply_rope(keys, self.key_rope)
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            keys.transpose(0, 1)[None],
            values.transpose(0, 1)[None],
        )[0].transpose(0, 1)
        if self.sp_size > 1:
            out = F.pad(out, (0, 0, 0, 0, 0, padded_tokens - self.live_tokens))
            out = _usp_output_all_to_all(out[None], head_dim=2)[0]
        return out
