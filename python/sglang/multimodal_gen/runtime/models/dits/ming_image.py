# SPDX-License-Identifier: Apache-2.0
# Adapted from inclusionAI/Ming-Image.
import math

import torch
from torch import nn
from torch.nn import functional as F

from sglang.multimodal_gen.runtime.distributed import get_sp_world_size
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
    gather_seq,
    shard_like,
    shard_seq,
    tail_attn_meta,
)
from sglang.multimodal_gen.runtime.models.dits.zimage import ZImageTransformer2DModel
from sglang.srt.layers.layernorm import RMSNorm


class MingRMSNorm(RMSNorm):
    def __init__(self, dim, eps=1e-5):
        # unlike Z-Image, Ming's reference accumulates the variance in fp32
        super().__init__(dim, eps=eps, cast_x_before_out_mul=True, force_native=True)


class MingSiluAndMul(nn.Module):
    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up


class MingImageTransformer2DModel(ZImageTransformer2DModel):
    """Ming-Image's joint DiT, sharing Z-Image blocks and TP/SP attention."""

    norm_cls = MingRMSNorm
    _aliases = ["DiffusionTransformer"]

    def __init__(self, config, hf_config, quant_config=None):
        super().__init__(config, hf_config, quant_config)
        self.learned_padding = config.alignment_padding_mode == "learned"
        if not self.learned_padding:
            del self.x_pad_token, self.cap_pad_token
            self.register_buffer(
                "x_pad_token", torch.zeros(1, self.dim, device="cpu"), persistent=False
            )
            self.register_buffer(
                "cap_pad_token",
                torch.zeros(1, self.dim, device="cpu"),
                persistent=False,
            )
        self.layer_names = ["noise_refiner", "context_refiner", "layers"]
        for layer in (*self.noise_refiner, *self.context_refiner, *self.layers):
            layer.feed_forward.act = MingSiluAndMul()

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        direct_embeddings,
        reference_latents=None,
        **kwargs,
    ):
        x = hidden_states
        batch_size, channels, output_frames, height, width = x.shape
        if reference_latents is not None:
            x = torch.cat((x, reference_latents), 2)
        frames = x.shape[2]
        x = x.reshape(batch_size, channels, frames, height // 2, 2, width // 2, 2)
        x = x.permute(0, 2, 3, 5, 4, 6, 1).flatten(1, 3).flatten(2)
        x_length = x.shape[1]
        sp_size = get_sp_world_size()
        x_target = math.ceil(x_length / 32) * 32
        x = F.pad(x, (0, 0, 0, x_target - x_length))
        x, _ = self.all_x_embedder["2-1"](x)
        x = self._replace_padding_with_token_mask(
            x,
            (torch.arange(x_target, device=x.device) < x_length).unsqueeze(0),
            self.x_pad_token,
        )
        cap, _ = self.cap_embedder(encoder_hidden_states)
        cap = torch.cat((cap, direct_embeddings), 1)
        cap_length = cap.shape[1]
        cap_target = math.ceil(cap_length / 32) * 32
        cap = F.pad(cap, (0, 0, 0, cap_target - cap_length))
        cap = self._replace_padding_with_token_mask(
            cap,
            (torch.arange(cap_target, device=x.device) < cap_length).unsqueeze(0),
            self.cap_pad_token,
        )

        cap_ids = self.create_coordinate_grid(
            (cap_target, 1, 1), (1, 0, 0), x.device
        ).flatten(0, 2)
        x_ids = self.create_coordinate_grid(
            (frames, height // 2, width // 2), (cap_target + 1, 0, 0), x.device
        ).flatten(0, 2)
        x_ids = F.pad(x_ids, (0, 0, 0, x_target - x_length))
        x_valid = x_target if self.learned_padding else x_length
        cap_valid = cap_target if self.learned_padding else cap_length
        x_freqs, cap_freqs = self.rotary_emb(x_ids), self.rotary_emb(cap_ids)
        x_mask, x_meta = self._get_attn_mask_and_meta(
            "_ming_x_mask", [x_valid] * batch_size, x_target, x.device
        )
        cap_mask, cap_meta = self._get_attn_mask_and_meta(
            "_ming_cap_mask", [cap_valid] * batch_size, cap_target, x.device
        )
        adaln = self.t_embedder(((1000.0 - timestep) / 1000.0) * self.t_scale).to(
            x.dtype
        )
        for layer in self.noise_refiner:
            x = layer(
                x,
                x_freqs,
                adaln,
                attn_mask=x_mask,
                attn_mask_meta=x_meta,
                skip_sequence_parallel_override=True,
            )
        for layer in self.context_refiner:
            cap = layer(cap, cap_freqs, attn_mask=cap_mask, attn_mask_meta=cap_meta)
        if sp_size > 1:
            # masked alignment rows never contribute; learned registers must stay
            x, cap = x[:, :x_valid], cap[:, :cap_valid]
            x_freqs = tuple(freq[:x_valid] for freq in x_freqs)
            cap_freqs = tuple(freq[:cap_valid] for freq in cap_freqs)
        unified = torch.cat((x, cap), 1)
        freqs = tuple(
            torch.cat((image_freq, text_freq), -2)
            for image_freq, text_freq in zip(x_freqs, cap_freqs)
        )
        mask, meta = self._get_joint_attn_mask_and_meta(
            [x_valid] * batch_size,
            x_target,
            [cap_valid] * batch_size,
            cap_target,
            x.device,
        )
        if sp_size > 1:
            # shard the complete joint sequence, with only global-tail padding
            unified, shard = shard_seq(unified)
            freqs = tuple(shard_like(freq, shard, dim=0) for freq in freqs)
            mask, meta = None, tail_attn_meta(shard, batch_size, x.device)
        for layer in self.layers:
            unified = layer(
                unified,
                freqs,
                adaln,
                attn_mask=mask,
                attn_mask_meta=meta,
            )
        if sp_size == 1:
            unified = unified[:, :x_target]
        output = self.all_final_layer["2-1"](unified, adaln)
        if sp_size > 1:
            output = gather_seq(output, shard.orig_len)
        output = output[:, :x_length].reshape(
            batch_size, frames, height // 2, width // 2, 2, 2, channels
        )
        output = output.permute(0, 6, 1, 2, 4, 3, 5).reshape(
            batch_size, channels, frames, height, width
        )
        return -output[:, :, :output_frames]


EntryClass = MingImageTransformer2DModel
