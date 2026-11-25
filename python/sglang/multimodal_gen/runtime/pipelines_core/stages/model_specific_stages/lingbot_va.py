# SPDX-License-Identifier: Apache-2.0
"""Video-conditioned action chunks and observed history for LingBot-VA policies."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from sglang.multimodal_gen.configs.pipeline_configs.wan import _wan_whitespace_clean
from sglang.multimodal_gen.runtime.models.vaes.wanvae import forward_context, patchify
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.vla import vla_state
from sglang.multimodal_gen.runtime.realtime.action_chunk import remaining_ms
from sglang.multimodal_gen.runtime.realtime.session import BaseRealtimeState


class LingBotVAState(BaseRealtimeState):
    def __init__(self):
        self.frame = 0
        self.history = []
        self.vae_cache = []
        self.anchor = None
        self.text = None
        self.prompt = None


def grid_ids(shape, frame, action, device):
    frames, height, width = shape
    f, h, w = torch.meshgrid(
        torch.arange(frame, frame + frames, device=device),
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    if action:
        f = f + (h.float() + 1) / (height + 1)
        h = torch.full_like(h, -1)
        w = torch.full_like(w, -1)
    return torch.stack((f, h, w)).flatten(1).unsqueeze(0)


def append_kv(history, new):
    if not history:
        return new
    return [
        (torch.cat((a, c), dim=1), torch.cat((b, d), dim=1))
        for (a, b), (c, d) in zip(history, new, strict=True)
    ]


@dataclass
class PreparedChunk:
    request: object
    state: LingBotVAState
    prompt: str
    frame: int
    text: torch.Tensor
    anchor: torch.Tensor | None
    history: list
    cache: list
    visual: torch.Tensor
    actions: torch.Tensor
    prefix: torch.Tensor | None


class LingBotVAStage(PipelineStage):
    def __init__(self, modules, config):
        super().__init__()
        self.model = modules["transformer"]
        self.vae = modules["vae"]
        self.vae.clear_cache()
        self.text_encoder = modules["text_encoder"]
        self.tokenizer = modules["tokenizer"]
        self.config = config
        self.channels = list(config.channels)
        self.inactive = [i for i in range(30) if i not in self.channels]
        h, w = config.latent_shape
        self.frames = config.frame_chunk_size
        self.capacity = (config.attention_window // 2) * (
            self.frames * h * w // 4 + config.action_horizon
        )

    def encode_text(self, prompt, device):
        clean = _wan_whitespace_clean(prompt)
        tokens = self.tokenizer(
            [clean, ""],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        embeds = self.text_encoder(**tokens).last_hidden_state
        return embeds.masked_fill(~tokens.attention_mask.bool().unsqueeze(-1), 0)

    def encode_frames(self, images, expected, cache, device, dtype):
        views = []
        for index, key in enumerate(self.config.image_keys):
            if key not in images:
                raise ValueError(f"Missing LingBot camera {key}")
            source = images[key]
            view = np.asarray(source)
            if (
                isinstance(source, list)
                and view.size
                and view.dtype.kind in "iu"
                and view.min() >= 0
                and view.max() <= 255
            ):
                view = view.astype(np.uint8)
            if view.ndim == 3:
                view = view[None]
            if (
                view.ndim != 4
                or view.shape[0] != expected
                or view.shape[-1] != 3
                or view.dtype != np.uint8
            ):
                raise ValueError(f"{key} needs {expected} uint8 RGB frames [T,H,W,3]")
            frames = (
                torch.as_tensor(np.ascontiguousarray(view), device=device)
                .permute(0, 3, 1, 2)
                .float()
            )
            size = self.config.image_size
            if self.config.wrist_half_resolution and index > 0:
                size = tuple(n // 2 for n in size)
            views.append(
                F.interpolate(
                    frames, size=size, mode="bilinear", align_corners=False
                ).transpose(0, 1)
            )

        def encode(group, feature_cache):
            pixels = ((torch.stack(group) / 255) * 2 - 1).to(dtype)
            pixels = patchify(pixels, patch_size=self.vae.config.patch_size)
            with forward_context(feat_cache_arg=feature_cache, feat_idx_arg=0):
                return self.vae.quant_conv(self.vae.encoder(pixels))

        if self.config.wrist_half_resolution:
            high = encode(views[:1], cache[0])
            wrists = torch.cat(encode(views[1:], cache[1]).split(1), dim=-1)
            stats = torch.cat((wrists, high), dim=-2)
        else:
            stats = encode(views, cache[0])
        mu = stats.chunk(2, dim=1)[0].float()
        mean = mu.new_tensor(self.vae.latents_mean).view(1, -1, 1, 1, 1)
        std = mu.new_tensor(self.vae.latents_std).view(1, -1, 1, 1, 1)
        return torch.cat(((mu - mean) * std.reciprocal()).to(dtype).split(1), dim=-1)

    def predict(self, value, time, frame, text, history, *, action, save=None):
        f, h, w = value.shape[-3:]
        scale = 1 if action else 2
        grid = grid_ids((f, h // scale, w // scale), frame, action, value.device)
        # Leave room for the current tokens within the checkpoint attention window.
        capacity = self.capacity
        keep = capacity - grid.shape[-1]
        history = [(k[:, -keep:], v[:, -keep:]) for k, v in history]
        return self.model(
            value.repeat(2, 1, 1, 1, 1),
            time,
            grid,
            text,
            action=action,
            history=history,
            save=save,
        )

    def ground(self, state, images, executed, low, span, device, dtype, cache):
        count = self.config.action_per_frame * (self.frames - int(state.frame == 0))
        rows = torch.as_tensor(executed, device=device, dtype=torch.float32)
        if (
            rows.shape != (count, self.config.action_dim)
            or not torch.isfinite(rows).all()
        ):
            raise ValueError(
                f"executed_actions needs {count} finite controller commands"
            )
        normalized = (rows - low) / span * 2 - 1
        if state.frame == 0:
            normalized = F.pad(normalized, (0, 0, self.config.action_per_frame, 0))
        actions = torch.zeros(
            (1, 30, self.frames, self.config.action_per_frame, 1),
            device=device,
            dtype=dtype,
        )
        actions[0, self.channels, :, :, 0] = normalized.T.reshape(
            self.config.action_dim, self.frames, -1
        ).to(dtype)
        visual = self.encode_frames(
            images, 4 * (self.frames - int(state.frame == 0)), cache, device, dtype
        )
        if state.frame == 0:
            visual = torch.cat((state.anchor, visual), dim=2)
        history = state.history
        for value, action in ((visual, False), (actions, True)):
            new = []
            self.predict(
                value,
                torch.zeros(1, self.frames, device=device),
                state.frame,
                state.text,
                history,
                action=action,
                save=new,
            )
            history = append_kv(history, new)
            history = [
                (
                    k[:, -self.capacity :].contiguous(),
                    v[:, -self.capacity :].contiguous(),
                )
                for k, v in history
            ]
        return history

    def prepare(self, batch):
        record = vla_state(batch)
        obs = record.get("observation") or {}
        images = obs.get("images", {})
        realtime = batch.extra.get("action_realtime", {})
        device = next(self.model.parameters()).device
        dtype = self.model.action_embedder.weight.dtype
        state = (
            batch.session.get_or_create_state(LingBotVAState)
            if batch.session is not None
            else LingBotVAState()
        )
        prompt = batch.prompt or ""
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise ValueError("Expected one prompt per observation")
            prompt = prompt[0]
        cold = state.text is None
        if batch.is_warmup:
            images = {
                key: np.zeros((*self.config.image_size, 3), np.uint8)
                for key in self.config.image_keys
            }
        low = torch.tensor(self.config.low, device=device)
        span = torch.tensor(self.config.high, device=device) - low + 1e-6
        cache = (
            [list(c) for c in state.vae_cache]
            if not cold
            else [
                [None] * self.vae._enc_conv_num
                for _ in range(2 if self.config.wrist_half_resolution else 1)
            ]
        )
        if cold:
            if obs.get("executed_actions") is not None:
                raise ValueError("Missing LingBot session for continuation")
            anchor = self.encode_frames(images, 1, cache, device, dtype)
            text = self.encode_text(prompt, device)
            history = []
            frame = 0
        else:
            if obs.get("executed_actions") is None:
                raise ValueError(
                    "A LingBot continuation needs camera history and executed_actions"
                )
            history = self.ground(
                state, images, obs["executed_actions"], low, span, device, dtype, cache
            )
            # Observed history used the previous condition. The next action
            # may consume a new instruction from the reasoning stage.
            anchor = None
            text = (
                state.text
                if prompt == state.prompt
                else self.encode_text(prompt, device)
            )
            frame = state.frame + self.frames
        seed = batch.seed[0] if isinstance(batch.seed, list) else batch.seed
        generator = torch.Generator(device=device).manual_seed(int(seed) + frame)

        def noise(shape):
            return torch.randn(shape, device=device, dtype=dtype, generator=generator)

        visual = noise((1, 48, self.frames, *self.config.latent_shape))
        actions = noise((1, 30, self.frames, self.config.action_per_frame, 1))
        prefix = realtime.get("action_prefix")
        if prefix is not None:
            prefix = torch.as_tensor(prefix, device=device)
            if len(prefix) > self.config.action_per_frame * (self.frames - int(cold)):
                raise ValueError("Action prefix exceeds executable chunk")
            prefix = ((prefix - low) / span * 2 - 1).T.to(dtype)
        return PreparedChunk(
            batch,
            state,
            prompt,
            frame,
            text,
            anchor,
            history,
            cache,
            visual,
            actions,
            prefix,
        )

    def denoise(self, chunks, *, check_deadline=True):
        first = chunks[0]
        frame, device = first.frame, first.visual.device
        cold = frame == 0
        start = self.config.action_per_frame if cold else 0

        # Reference CFG layout is conditional rows followed by unconditional rows.
        def cfg_stack(tensors):
            return (
                torch.stack(tensors, dim=1).flatten(0, 1)
                if len(tensors) > 1
                else tensors[0]
            )

        history = []
        for layer in zip(*(c.history for c in chunks), strict=True):
            keys, values = zip(*layer, strict=True)
            history.append((cfg_stack(keys), cfg_stack(values)))
        text = cfg_stack([c.text for c in chunks])
        anchor = torch.cat([c.anchor for c in chunks]) if cold else None
        visual = torch.cat([c.visual for c in chunks])
        actions = torch.cat([c.actions for c in chunks])
        prefix_noise = actions[:, self.channels, :, :, 0].flatten(2).clone()
        video_cache = []
        for action, steps, shift in (
            (False, self.config.video_steps, 5.0),
            (True, first.request.num_inference_steps, self.config.action_shift),
        ):
            sigmas = torch.linspace(1.0, 0.0, steps + 1)
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
            value = actions if action else visual
            for index, sigma in enumerate(sigmas):
                if check_deadline:
                    deadline = first.request.extra.get("action_realtime", {}).get(
                        "deadline_unix_ms"
                    )
                    if deadline is not None:
                        remaining_ms(deadline)
                time = torch.full((1, self.frames), float(sigma * 1000), device=device)
                if cold:
                    value[:, :, 0:1] = 0 if action else anchor
                    time[:, 0] = 0
                if action:
                    value[:, self.inactive] = 0
                    for i, chunk in enumerate(chunks):
                        prefix = chunk.prefix
                        if prefix is not None:
                            sl = slice(start, start + prefix.shape[1])
                            value[i, :, :, :, 0].flatten(1)[self.channels, sl] = (
                                prefix * (1 - sigma) + prefix_noise[i, :, sl] * sigma
                            )
                last = index == steps
                if action and last:
                    break
                pred = self.predict(
                    value,
                    time,
                    frame,
                    text,
                    history,
                    action=action,
                    save=video_cache if last else None,
                )
                if last:
                    break
                conditional, unconditional = pred.chunk(2)
                if action:
                    velocity = rearrange(
                        conditional, "b (f h) c -> b c f h 1", f=self.frames
                    )
                else:
                    velocity = unconditional + 5 * (conditional - unconditional)
                    velocity = rearrange(
                        velocity,
                        "b (f h w y x) c -> b c f (h y) (w x)",
                        f=self.frames,
                        h=self.config.latent_shape[0] // 2,
                        w=self.config.latent_shape[1] // 2,
                        y=2,
                        x=2,
                    )
                value = value + velocity * (sigmas[index + 1] - sigma)
            if not action:
                history = append_kv(history, video_cache)
        low = torch.tensor(self.config.low, device=device)
        span = torch.tensor(self.config.high, device=device) - low + 1e-6
        rows = (
            value[:, self.channels, :, :, 0]
            .flatten(2)
            .transpose(1, 2)
            .float()[:, start:]
        )
        commands = (rows + 1) * 0.5 * span + low
        for i, chunk in enumerate(chunks):
            record = vla_state(chunk.request)
            if chunk.prefix is not None:
                prefix = chunk.request.extra["action_realtime"]["action_prefix"]
                commands[i, : len(prefix)] = torch.as_tensor(prefix, device=device)
            record["actions"] = commands[i : i + 1]
            record["extras"] = {"policy_batch_size": len(chunks)}
            state = chunk.state
            state.anchor, state.history, state.frame = (
                chunk.anchor,
                chunk.history,
                chunk.frame,
            )
            state.text, state.prompt, state.vae_cache = (
                chunk.text,
                chunk.prompt,
                chunk.cache,
            )

    def run_grouped_requests(self, batches, server_args):
        groups = defaultdict(list)
        for request in batches:
            chunk = self.prepare(request)
            groups[(chunk.frame, request.num_inference_steps)].append(chunk)
        for chunks in groups.values():
            self.denoise(chunks, check_deadline=False)
        return batches

    def forward(self, batch, server_args):
        self.denoise([self.prepare(batch)])
        return batch
