# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Adapted for inference from huggingface/lerobot at f58f7fb9dbb241304df96e967aea0f5b42ff9a4d.
# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flow-matching action sampling of LingBot-VLA 2.0: suffix embedding, velocity prediction and Euler steps."""

from typing import TYPE_CHECKING

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from .utils import create_sinusoidal_pos_embedding, make_att_2d_masks

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.vla.cuda_graph import VLATensorGraphRunner


class FlowMatching(nn.Module):
    @staticmethod
    def _fp32_linear(module, x):
        """Compute linear layer in fp32 regardless of module's current parameter dtype."""
        return F.linear(
            x.float(),
            module.weight.float(),
            module.bias.float() if module.bias is not None else None,
        )

    def embed_suffix(
        self, state, noisy_actions, timestep
    ):  # (torch.Size([state_bs, 32]), torch.Size([1, state_bs*50, 32]), torch.Size([1]))
        bsize = state.shape[0]  # state_bs = img_bs
        device = state.device
        dtype = state.dtype
        _fp32 = self.config.action_fp32
        state_emb = (
            self._fp32_linear(self.state_proj, state)
            if _fp32
            else self.state_proj(state)
        )

        time_emb = create_sinusoidal_pos_embedding(  # 1, 1024
            timestep,  # torch.Size([1]))
            self.proj_width,
            min_period=4e-3,
            max_period=4.0,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb_ori = time_emb

        action_emb = (
            self._fp32_linear(self.action_in_proj, noisy_actions)
            if _fp32
            else self.action_in_proj(noisy_actions)
        )  # torch.Size([1, state_bs*50, 1024])
        time_emb = einops.repeat(
            time_emb, "b d -> b n d", n=action_emb.shape[1]
        )  # [1, 1024] -> [1, state_bs*50, 1024]
        action_time_emb = torch.cat(
            [action_emb, time_emb], dim=-1
        )  # [1, state_bs*50, 2048]

        action_time_emb = (
            self._fp32_linear(self.action_time_mlp_in, action_time_emb)
            if _fp32
            else self.action_time_mlp_in(action_time_emb)
        )
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = (
            self._fp32_linear(self.action_time_mlp_out, action_time_emb)
            if _fp32
            else self.action_time_mlp_out(action_time_emb)
        )  # [1, state_bs*50, 1024]
        action_time_dim = action_time_emb.shape[1]

        embs = torch.cat([state_emb[:, None], action_time_emb], dim=1)
        pad_masks = torch.ones(
            (bsize, action_time_dim + 1), device=device, dtype=torch.bool
        )

        # note (0xtoward): only the action tokens share a bidirectional attention block.
        att_masks = torch.zeros(
            (bsize, action_time_dim + 1), device=device, dtype=torch.bool
        )
        att_masks[:, :2] = True

        return time_emb_ori, embs, pad_masks, att_masks

    def _build_full_position_ids(
        self, prefix_position_ids, prefix_pad_masks, suffix_pad_masks
    ):
        valid_prefix_pos = prefix_position_ids.masked_fill(
            ~prefix_pad_masks.unsqueeze(0), 0
        )
        prefix_offsets = valid_prefix_pos.amax(dim=(0, 2)) + 1
        suffix_1d = (
            prefix_offsets[:, None] + torch.cumsum(suffix_pad_masks.long(), dim=1) - 1
        )
        suffix_1d = suffix_1d.masked_fill(~suffix_pad_masks, 1)
        suffix_position_ids = suffix_1d.unsqueeze(0).expand(3, -1, -1)
        return torch.cat([prefix_position_ids, suffix_position_ids], dim=-1)

    def predict_velocity(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        prefix_position_ids=None,
        _denoise_cache: dict | None = None,
    ):
        """Predict the flow velocity using the observation prefix and request-local masks."""
        if prefix_position_ids is None:
            raise ValueError(
                "LingBotVLAV2ForActionPrediction.predict_velocity requires Qwen3-VL prefix_position_ids."
            )

        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state,
            x_t,
            timestep,
        )

        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        cache = _denoise_cache if _denoise_cache is not None else {}
        if "full_att_2d_masks" not in cache:
            batch_size = prefix_pad_masks.shape[0]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size,
                suffix_len,
                prefix_len,
            )
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat(
                [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
            )
            full_att_2d_masks = self._block_suffix_to_queries_(
                full_att_2d_masks, 0, prefix_len
            )

            full_position_ids = self._build_full_position_ids(
                prefix_position_ids,
                prefix_pad_masks,
                suffix_pad_masks,
            )
            position_ids = full_position_ids[:, :, -suffix_len:]
            core = self.qwenvl_with_expert
            position_embeddings = core.qwenvl.model.language_model.rotary_emb(
                suffix_embs, position_ids
            )
            cache["full_att_2d_masks"] = full_att_2d_masks
            cache["position_ids"] = position_ids
            cache["position_embeddings"] = position_embeddings

        outputs_embeds, _ = self.qwenvl_with_expert.forward(
            attention_mask=cache["full_att_2d_masks"],
            position_ids=cache["position_ids"],
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            ada_cond=time_embs if self.config.adanorm_time else None,
            position_embeddings=cache["position_embeddings"],
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        if self.config.action_fp32:
            v_t = self._fp32_linear(self.action_out_proj, suffix_out)
        else:
            if suffix_out.dtype != self.action_out_proj.weight.dtype:
                suffix_out = suffix_out.to(self.action_out_proj.weight.dtype)
            v_t = self.action_out_proj(suffix_out)
        return v_t

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        image_grid_thw,
        noise=None,
        num_steps=None,
        generator=None,
        denoise_graph: "VLATensorGraphRunner | None" = None,
    ):
        steps = self.config.num_steps if num_steps is None else num_steps
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("num_inference_steps must be a positive integer")
        shape = (state.shape[0], self.config.chunk_size, self.config.max_action_dim)
        if noise is None:
            x_t = torch.randn(
                shape, device=state.device, dtype=state.dtype, generator=generator
            )
        else:
            if tuple(noise.shape) != shape:
                raise ValueError(
                    f"Expected noise shape {shape}, got {tuple(noise.shape)}"
                )
            x_t = noise.to(device=state.device, dtype=state.dtype).clone()
        masks, positions, kv = self._embed_and_fill_prefix(
            images, img_masks, lang_tokens, lang_masks, image_grid_thw
        )
        dt = torch.tensor(-1.0 / steps, device=state.device, dtype=state.dtype)
        time = torch.tensor(1.0, device=state.device, dtype=state.dtype)
        if denoise_graph is not None:
            return self._run_denoise_graph(
                denoise_graph, state, masks, positions, kv, x_t, dt, time, steps
            )
        return self._denoise_actions(state, masks, positions, kv, x_t, dt, time, steps)

    def _denoise_actions(self, state, masks, positions, kv, x_t, dt, time, steps):
        cache = {}
        for _ in range(steps):
            velocity = self.predict_velocity(
                state, masks, kv, x_t, time.expand(state.shape[0]), positions, cache
            )
            x_t = x_t + dt * velocity
            time = time + dt
        return x_t

    def _run_denoise_graph(
        self, runner, state, masks, positions, kv, x_t, dt, time, steps
    ):
        kv_inputs = tuple(
            kv[layer][kind]
            for layer in range(len(kv))
            for kind in ("key_states", "value_states")
        )

        def run(state, masks, positions, x_t, dt, time, *tensors):
            kv = {
                layer: {"key_states": key, "value_states": value}
                for layer, (key, value) in enumerate(
                    zip(tensors[::2], tensors[1::2], strict=True)
                )
            }
            return (
                self._denoise_actions(
                    state, masks, positions, kv, x_t, dt, time, steps
                ),
            )

        (actions,) = runner.run(
            run, (state, masks, positions, x_t, dt, time, *kv_inputs), key=steps
        )
        return actions.clone()
