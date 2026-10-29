# SPDX-License-Identifier: Apache-2.0
"""H3 same-step sampler using native DiT blocks and the causal pipeline plan."""

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_group
from sglang.multimodal_gen.runtime.models.dits.physis_h3 import (
    PHYSIS_H3_SIGMAS,
    PhysisH3AttentionContext,
    physis_h3_step,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpatchify_video_tokens,
)


class PhysisH3Denoiser:
    """Request-owned KV and noise. A packet carries the latent and text/video rows."""

    def __init__(
        self,
        model,
        prompt_embeds,
        latent_shape,
        seed,
        history=18,
        sp_size=1,
        sp_rank=0,
        refined_text=None,
    ):
        self.model = model
        self.shape = latent_shape
        self.seed = seed
        self.device = prompt_embeds.device
        self.sp_size, self.sp_rank = sp_size, sp_rank
        self.patch_size = model.arch.patch_size
        self.prompt_embeds = prompt_embeds
        text_len = prompt_embeds.shape[0]
        cu = torch.tensor([0, text_len], device=self.device, dtype=torch.int32)
        self.text = (
            model.refine_prompt_embeds(
                prompt_embeds, cu, device=self.device, unfused_qkv=True
            )
            if refined_text is None
            else refined_text
        )
        self.context = PhysisH3AttentionContext(
            model,
            text_len,
            latent_shape[-2],
            latent_shape[-1],
            history,
            sp_size,
            chunk_frames=latent_shape[2],
        )
        live = self.context.live_tokens
        self.padded_tokens = (live + sp_size - 1) // sp_size * sp_size
        tags = torch.zeros(self.padded_tokens, device=self.device, dtype=torch.long)
        tags[:text_len] = 1
        self.tags = tags.chunk(sp_size)[sp_rank]
        self.inverse_indices = torch.zeros_like(self.tags)
        self.cu = torch.tensor([0, live], device=self.device, dtype=torch.int32)
        self.adaln_inputs = [
            F.silu(
                model._time_embedding(
                    torch.tensor([1 - sigma], device=self.device, dtype=torch.float32)
                )
            ).to(torch.bfloat16)
            for sigma in (*PHYSIS_H3_SIGMAS[:-1], 0.001)
        ]

    def noise(self, chunk, step, dtype=torch.float32):
        # Match the checkpoint's reference seed schedule: slot 0 is initial
        # noise, slots 1..3 re-noise the denoising passes, and slot 5 is the sink.
        shape = (*self.shape[:2], 1, *self.shape[3:])
        frames = []
        for frame in range(chunk, chunk + self.shape[2]):
            generator = torch.Generator(device=self.device).manual_seed(
                self.seed * 1_000_003 + frame * 4096 + step
            )
            frames.append(
                torch.randn(shape, generator=generator, device=self.device, dtype=dtype)
            )
        return frames[0] if len(frames) == 1 else torch.cat(frames, dim=2)

    def initial_latent(self, chunk):
        return self.noise(chunk, 0, dtype=torch.bfloat16)

    def _embed(self, latent):
        rows = minimax_h3_patchify_video_latent(latent, patch_size=self.patch_size)
        video, _ = self.model.video_patch_proj(rows.float())
        hidden = torch.cat((self.text, video.to(torch.bfloat16)))
        hidden = F.pad(hidden, (0, 0, 0, self.padded_tokens - hidden.shape[0]))
        return hidden.chunk(self.sp_size)[self.sp_rank].contiguous()

    def _predict(self, hidden, step):
        video, _ = self.model.final_layer(
            hidden,
            adaln_input=self.adaln_inputs[step],
            inverse_indices=self.inverse_indices,
        )
        if self.sp_size > 1:
            video = get_sp_group().all_gather(video, dim=0)
        video = video[self.text.shape[0] : self.context.live_tokens]
        _, channels, frames, height, width = self.shape
        pt, ph, pw = self.patch_size
        return minimax_h3_unpatchify_video_tokens(
            video,
            latent_shape=(frames // pt, height // ph, width // pw, channels),
            patch_size=self.patch_size,
        )

    def run_slice(self, latent, hidden, chunk, step, block_start, block_end):
        context = self.context
        if context.chunk != chunk or context.query_rope is None:
            context.set_chunk(chunk)
        context.step = step
        if block_start == 0:
            hidden = self._embed(latent)
        for layer in range(block_start, block_end):
            context.layer = layer
            hidden = self.model.blocks[layer](
                hidden,
                adaln_input=self.adaln_inputs[step],
                combined_indices=self.tags,
                rope_cache=None,
                cu_seqlens=self.cu,
                max_seqlen=context.live_tokens,
                causal_context=context,
            )
        if block_end == len(self.model.blocks):
            prediction = self._predict(hidden, step)
            if step < len(PHYSIS_H3_SIGMAS) - 1:
                latent = physis_h3_step(
                    latent,
                    prediction,
                    PHYSIS_H3_SIGMAS[step],
                    PHYSIS_H3_SIGMAS[step + 1],
                    self.noise(chunk, step + 1),
                )
            hidden = None
        return latent, hidden

    def bootstrap(self, num_frames=1, owned_slices=None):
        """Denoise the opening prefix jointly and retain KV for the owned slices."""
        if not 1 <= num_frames <= self.context.history + 1:
            raise ValueError("Bootstrap must fit in the H3 history window")
        prefix = self
        if num_frames > 1:
            prefix = PhysisH3Denoiser(
                self.model,
                self.prompt_embeds,
                (*self.shape[:2], num_frames, *self.shape[3:]),
                self.seed,
                history=self.context.history,
                sp_size=self.sp_size,
                sp_rank=self.sp_rank,
                refined_text=self.text,
            )
        prefix.context.sink_only = True
        latent = prefix.initial_latent(0)
        for step in range(len(PHYSIS_H3_SIGMAS) - 1):
            latent, _ = prefix.run_slice(
                latent, None, 0, step, 0, len(self.model.blocks)
            )
        pages = prefix.context.caches
        prefix.context.caches = {}
        # The sink forward is conditioned at sigma=0.001, so its input must
        # have that noise level too. Keep the returned opening latent clean.
        sink_sigma = 0.001
        sink_latent = (
            (1 - sink_sigma) * latent[:, :, :1].float() + sink_sigma * self.noise(0, 5)
        ).to(latent.dtype)
        self.context.sink_only = True
        self.run_slice(sink_latent, None, 0, 4, 0, len(self.model.blocks))
        self.context.sinks = {
            layer: (cache.k, cache.v)
            for (_, layer), cache in self.context.caches.items()
        }
        self.context.caches.clear()
        self.context.sink_only = False
        if num_frames == 1:
            return latent
        owned = (
            {
                (part.timestep, layer)
                for part in owned_slices
                for layer in range(part.block_start, part.block_end)
            }
            if owned_slices is not None
            else set(pages)
        )
        frame_tokens = self.context.frame_tokens
        # Pop pages as they are copied, keeping startup peak memory bounded.
        for key in list(pages):
            page = pages.pop(key)
            if key not in owned:
                continue
            sink_k, sink_v = self.context.sinks[key[1]]
            k = torch.cat((sink_k, page.k[:, frame_tokens:]), dim=1)
            v = torch.cat((sink_v, page.v[:, frame_tokens:]), dim=1)
            cache = self.context._new_cache(k[0])
            cache.update_and_get_attention_kv(key=k, value=v, current_chunk_start=0)
            self.context.caches[key] = cache
        return latent

    def run_chunk(self, chunk, slices, latent=None, hidden=None):
        if latent is None:
            latent = self.initial_latent(chunk)
        for part in slices:
            latent, hidden = self.run_slice(
                latent, hidden, chunk, part.timestep, part.block_start, part.block_end
            )
        return latent, hidden
