# SPDX-License-Identifier: Apache-2.0
#
# Wan-Animate-2 DiT. Subclasses WanTransformer3DModel so TP / Ulysses SP shard it
# unchanged; only the per-block self-attention is swapped for the in-context mechanism
# (build_reference_kv runs forward_ref once per clip and returns each block's rotated reference-video
# K/V, forward_gen attends over [gen tokens, then that reference-video K/V]). No extra parameters.
# Names kept from the reference implementation: the forward_ref / forward_gen methods and the
# kv_cache_mode values "ref" / "gen"; there "ref" means the reference video (the motion source), not the reference image.
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from sglang.multimodal_gen.configs.models.dits.wan_animate_2 import (
    WanAnimate2Config,
)
from sglang.multimodal_gen.runtime.distributed import get_sp_world_size
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
    gather_seq,
    shard_seq,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.models.dits.wan_animate_2_block import (
    WanAnimate2TransformerBlock,
    _InContextLayout,
    _reference_video_key_mask,
)
from sglang.multimodal_gen.runtime.models.dits.wan_animate_2_clip_conditioning import (
    WanAnimate2ClipConditioning,
    WanAnimate2ReferenceKV,
)
from sglang.multimodal_gen.runtime.models.dits.wanvideo import WanTransformer3DModel
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class WanAnimate2Transformer3DModel(WanTransformer3DModel):
    """Wan-Animate-2 DiT (in-context variant) on the Wan2.2-I2V-14B backbone.

    Attention goes through USPAttention, so callers must wrap the forward in
    ``set_forward_context(...)``.
    """

    # Registry alias: the diffusers class name of this transformer.
    _aliases = ["WanAnimate2Transformer3DModel"]

    param_names_mapping = WanAnimate2Config().param_names_mapping
    reverse_param_names_mapping = WanAnimate2Config().reverse_param_names_mapping
    lora_param_names_mapping = WanAnimate2Config().lora_param_names_mapping

    def __init__(
        self,
        config: WanAnimate2Config,
        hf_config: dict[str, Any] | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(
            config=config, hf_config=hf_config or {}, quant_config=quant_config
        )

        # RoPE placement of the reference-video tokens relative to the gen grid; plain
        # ints from the config, not parameters, so the Wan state dict is unchanged.
        self.refer_offset_t = config.refer_offset_t
        self.refer_offset_h = config.refer_offset_h
        # <0: place reference-video tokens right of the gen grid (w offset = gen W in tokens).
        self.refer_offset_w = config.refer_offset_w
        self.refer_stride = config.refer_stride
        # Prompt embeddings are padded/truncated to this many tokens.
        self.max_text_len = config.text_len

        self._block_masks: OrderedDict[tuple[int, int, int, str], BlockMask] = (
            OrderedDict()
        )
        # RoPE tables keyed by the (f, h, w) position ranges and device; see _rope_tables.
        self._rope_cache: OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]] = (
            OrderedDict()
        )

    def _make_block(
        self,
        *,
        index: int,
        config: WanAnimate2Config,
        quant_config: QuantizationConfig | None,
    ) -> nn.Module:
        # Same constructor args as the Wan block and no new params, so the Wan checkpoint
        # mapping and TP sharding apply unchanged.
        return WanAnimate2TransformerBlock(
            config.num_attention_heads * config.attention_head_dim,
            config.ffn_dim,
            config.num_attention_heads,
            config.qk_norm,
            config.cross_attn_norm,
            config.eps,
            config.added_kv_proj_dim,
            self._supported_attention_backends
            | {AttentionBackendEnum.VIDEO_SPARSE_ATTN},
            prefix=f"blocks.{index}",
            attention_type=config.attention_type,
            sla_topk=config.sla_topk,
            quant_config=quant_config,
            log_scale=config.log_scale,
        )

    # Device and dtype of the resident weights; the patch embedding is never offloaded.
    @property
    def _device(self) -> torch.device:
        return self.patch_embedding.proj.weight.device

    @property
    def _dtype(self) -> torch.dtype:
        return self.patch_embedding.proj.weight.dtype

    def _rope_tables(
        self,
        f_range: tuple[int, int, int],  # (start, length, stride)
        h_range: tuple[int, int],  # (start, length)
        w_range: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RoPE ``(cos, sin)``, each ``[f*h*w, D//2]`` fp32, for the full grid.

        Row i holds the position of token i, flattened in (f, h, w) row-major order
        like the patch embedding output.

        The table is cached per ranges."""
        key = (f_range, h_range, w_range, str(self._device))
        if key in self._rope_cache:
            self._rope_cache.move_to_end(key)
            return self._rope_cache[key]
        # Bound retained GPU tables; evict before allocating the next geometry.
        if len(self._rope_cache) >= 4:
            self._rope_cache.popitem(last=False)
        f_start, f_len, f_stride = f_range
        f_idx = torch.arange(f_len, device=self._device) * f_stride + f_start
        h_idx = torch.arange(h_range[1], device=self._device) + h_range[0]
        w_idx = torch.arange(w_range[1], device=self._device) + w_range[0]
        # Positions are built explicitly (not via the rank-sharding forward_from_grid) because
        # the in-context blocks all-to-all-gather the full sequence before applying RoPE.
        ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
        positions = torch.stack([ff.reshape(-1), hh.reshape(-1), ww.reshape(-1)], dim=1)
        cos, sin = self.rotary_emb.forward_uncached(positions.long())
        tables = (cos.float(), sin.float())
        self._rope_cache[key] = tables
        return tables

    def _generation_video_rope(
        self, f: int, h: int, w: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RoPE tables for the generated video's ``f x h x w`` token grid; positions start at 0."""
        return self._rope_tables((0, f, 1), (0, h), (0, w))

    def _reference_video_rope(
        self, f: int, h: int, w: int, generation_video_w: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RoPE tables for the reference video's ``f x h x w`` token grid, shifted by the
        ``refer_*`` offsets. When ``refer_offset_w`` is -1, the reference-video grid is placed
        directly to the right of the generated grid (width offset = ``generation_video_w``)."""
        offset_w = (
            generation_video_w if self.refer_offset_w == -1 else self.refer_offset_w
        )
        return self._rope_tables(
            (self.refer_offset_t, f, self.refer_stride),
            (self.refer_offset_h, h),
            (offset_w, w),
        )

    def _get_block_mask_from_layout(self, layout: _InContextLayout) -> BlockMask:
        """Cached flex_attention ``BlockMask`` over the padded [gen | reference-video] layout."""
        key = (
            layout.max_generation_video_num_tokens,
            layout.max_reference_video_num_tokens,
            layout.max_tokens_per_frame,
            str(self._device),
        )
        if key in self._block_masks:
            self._block_masks.move_to_end(key)
            return self._block_masks[key]
        if len(self._block_masks) >= 2:
            self._block_masks.popitem(last=False)

        # _compile=True keeps the block-sparse layout consistent with the compiled
        # flex_attention; built once per key, so the compile cost is paid once.
        block_mask = create_block_mask(
            layout.block_mask_fn(),
            B=None,
            H=None,
            Q_LEN=layout.padded_generation_video_len,
            KV_LEN=layout.padded_kv_len,
            device=self._device,
            _compile=True,
        )
        self._block_masks[key] = block_mask
        return block_mask

    def _patch_embeddings(
        self,
        latent_36ch: torch.Tensor,  # [B, 36, f, H, W]
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Patch-embed the latent into the token stream ``[B, f*h*w, C]`` and return the
        token grid_size ``(f, h, w)``; tokens are flattened row-major, w fastest."""
        p_t, p_h, p_w = self.patch_size
        _, _, num_frames, height, width = latent_36ch.shape
        grid_size = (num_frames // p_t, height // p_h, width // p_w)
        hidden_states = self.patch_embedding(latent_36ch)
        hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()
        return hidden_states, grid_size

    def _time_embeddings(
        self, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep_embeddings = self.condition_embedder.time_embedder(timestep, None)
        timestep_modulation = self.condition_embedder.time_modulation(
            timestep_embeddings
        ).unflatten(1, (6, -1))
        return timestep_embeddings, timestep_modulation

    def prepare_context(
        self,
        prompt_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor | None,
    ) -> torch.Tensor:
        """Project one branch's timestep-invariant context; owned by the current clip."""
        prompt_embeddings = self._pad_or_truncate_prompt_embeddings(prompt_embeddings)
        context = self.condition_embedder.text_embedder(prompt_embeddings)
        if image_embeddings is not None:
            image_context = self.condition_embedder.image_embedder(image_embeddings)
            context = torch.cat([image_context, context], dim=1)
        return context.to(self._dtype)

    def _pad_or_truncate_prompt_embeddings(
        self, prompt_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Zero-pad or truncate the prompt embeddings (``[L, C]`` or ``[B, L, C]``) to
        ``[B, max_text_len, C]``, as the official model does; cross-attention attends the pad rows too."""
        if prompt_embeddings.dim() == 2:
            prompt_embeddings = prompt_embeddings.unsqueeze(0)  # [L, C] -> [1, L, C]

        batch_size, num_tokens, channels = prompt_embeddings.shape
        if num_tokens < self.max_text_len:
            pad = prompt_embeddings.new_zeros(
                batch_size, self.max_text_len - num_tokens, channels
            )
            prompt_embeddings = torch.cat([prompt_embeddings, pad], dim=1)
        elif num_tokens > self.max_text_len:
            prompt_embeddings = prompt_embeddings[:, : self.max_text_len]
        return prompt_embeddings.to(self._dtype)

    @torch.no_grad()
    def build_reference_kv(
        self, *, clip_cond: WanAnimate2ClipConditioning
    ) -> WanAnimate2ReferenceKV:
        """One forward_ref pass over the clip's reference video, returning every block's rotated K and V.

        Text- and timestep-independent, so one result serves every step and CFG branch of the
        clip; the caller owns it and the DiT keeps no copy. Attention goes through USPAttention,
        so callers must wrap this in ``set_forward_context(...)`` like ``forward``.
        """

        # 36-channel reference-video input: fp32 latents (16) over the bf16 condition (20),
        # both cast to the param dtype so the cat runs in bf16 (see _make_generation_input).
        reference_video_latent_36ch = torch.cat(
            [
                clip_cond.reference_video_latents.to(self._dtype),
                clip_cond.reference_video_condition.to(self._dtype),
            ],
            dim=0,
        ).unsqueeze(0)  # [1, 36, cf, H', W']
        reference_video_frame_0_image_embeddings = (
            clip_cond.reference_video_frame_0_image_embeddings
        )  # [1, 257, 1280] bf16

        # reference-video timestep is forced to 1.
        reference_video_timestep = torch.ones(1, device=self._device, dtype=torch.long)
        hidden_states, (reference_video_f, reference_video_h, reference_video_w) = (
            self._patch_embeddings(reference_video_latent_36ch)
        )
        _, timestep_modulation = self._time_embeddings(reference_video_timestep)
        reference_video_encoder_hidden_states = self.prepare_context(
            clip_cond.prompt_ref_embeddings, reference_video_frame_0_image_embeddings
        )

        reference_video_rope_cos, reference_video_rope_sin = self._reference_video_rope(
            reference_video_f,
            reference_video_h,
            reference_video_w,
            generation_video_w=clip_cond.generation_video_grid_sizes[2],
        )
        reference_video_num_tokens = (
            reference_video_f * reference_video_h * reference_video_w
        )

        # Ulysses SP: shard the reference-video sequence; each block a2a-gathers it and caches
        # full-seq / head-shard K/V. The reference-video hidden states are discarded, so no gather.
        # The gathered sequence is tail-padded to a multiple of the SP size; one key mask over
        # that padding serves every block (None when nothing is padded).
        reference_video_key_mask = None
        sp_size = get_sp_world_size()
        if sp_size > 1:
            hidden_states, _ = shard_seq(hidden_states, dim=1)
            gathered_seq_len = hidden_states.shape[1] * sp_size
            if gathered_seq_len != reference_video_num_tokens:
                reference_video_key_mask = _reference_video_key_mask(
                    batch_size=hidden_states.shape[0],
                    seq_len=gathered_seq_len,
                    num_valid=reference_video_num_tokens,
                    device=hidden_states.device,
                )

        k_by_block: dict[int, torch.Tensor] = {}
        v_by_block: dict[int, torch.Tensor] = {}
        # Call via __call__ (not forward_ref) so the layerwise-offload hooks fire.
        for idx, block in enumerate(self.blocks):
            hidden_states = block(
                kv_cache_mode="ref",
                hidden_states=hidden_states,
                encoder_hidden_states=reference_video_encoder_hidden_states,
                timestep_modulation=timestep_modulation,
                reference_video_rope_cos=reference_video_rope_cos,
                reference_video_rope_sin=reference_video_rope_sin,
                reference_video_num_tokens=reference_video_num_tokens,
                index=idx,
                k_cache=k_by_block,
                v_cache=v_by_block,
                reference_video_key_mask=reference_video_key_mask,
            )
        return WanAnimate2ReferenceKV(k=k_by_block, v=v_by_block)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,  # [16, f, h, w] gen latent (B=1)
        encoder_hidden_states: torch.Tensor,  # this CFG branch's prompt embeddings [L, C]
        timestep: torch.Tensor,  # [1]
        # [1, 257, 1280] bf16 CLIP embed of the reference image
        encoder_hidden_states_image: torch.Tensor | None = None,
        guidance: torch.Tensor | None = None,
        *,
        # This clip's K/V from build_reference_kv; the same object for every step and CFG branch.
        reference_kv: WanAnimate2ReferenceKV,
        # This clip's conditioning from build_clip_conditioning.
        clip_cond: WanAnimate2ClipConditioning,
        projected_context: torch.Tensor | None = None,
        # The unconditional CFG branch skips block 9 (see _run_generation_blocks).
        is_unconditional: bool = False,
    ) -> torch.Tensor:
        sp_size = get_sp_world_size()

        latent_36ch = self._make_generation_input(
            hidden_states, clip_cond.generation_condition
        )
        hidden_states, (f, h, w) = self._patch_embeddings(latent_36ch)
        timestep_embeddings, timestep_modulation = self._time_embeddings(timestep)
        generation_encoder_hidden_states = (
            projected_context
            if projected_context is not None
            else self.prepare_context(
                encoder_hidden_states, encoder_hidden_states_image
            )
        )
        layout = _InContextLayout(
            (f, h, w),
            clip_cond.reference_video_grid_sizes,
            clip_cond.full_clip_grid_sizes,
        )

        # No seq padding here: hidden_states is exactly f*h*w tokens and the in-context
        # layout pads internally. Under Ulysses SP each block a2a-gathers the full
        # sequence, so shard tail padding lands at the global tail where forward_gen
        # expects it.
        if sp_size > 1:
            hidden_states, _ = shard_seq(hidden_states, dim=1)
        hidden_states = self._run_generation_blocks(
            hidden_states,
            encoder_hidden_states=generation_encoder_hidden_states,
            timestep_modulation=timestep_modulation,
            layout=layout,
            reference_kv=reference_kv,
            is_unconditional=is_unconditional,
        )
        if sp_size > 1:
            # Gather the shards and trim tail padding back to the clip's token count.
            hidden_states = gather_seq(
                hidden_states, layout.generation_video_num_tokens, dim=1
            )

        return self._unpatchify(hidden_states, timestep_embeddings, (f, h, w))

    def _make_generation_input(
        self,
        latent: torch.Tensor,  #  [16, f, h, w] fp32
        generation_condition: torch.Tensor,  #  [20, f, h, w] bf16
    ) -> torch.Tensor:
        """36-channel DiT input ``[1, 36, f, h, w]``: the fp32 noisy latent over the bf16 generation condition."""
        # Both cast to the param dtype so the cat runs in bf16 and does not rely on autocast;
        # the fp32 latent would be rounded to bf16 by the patch embedding anyway.
        return torch.cat(
            [latent.to(self._dtype), generation_condition.to(self._dtype)], dim=0
        ).unsqueeze(0)

    def _run_generation_blocks(
        self,
        hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        timestep_modulation: torch.Tensor,
        layout: _InContextLayout,
        reference_kv: WanAnimate2ReferenceKV,
        is_unconditional: bool,
    ) -> torch.Tensor:
        """forward_gen through every block with the in-context RoPE tables and block mask."""
        generation_video_rope_cos, generation_video_rope_sin = (
            self._generation_video_rope(
                layout.generation_video_f,
                layout.generation_video_h,
                layout.generation_video_w,
            )
        )
        block_mask = self._get_block_mask_from_layout(layout)

        # Call via __call__ (not forward_gen) so the layerwise-offload hooks fire.
        for idx, block in enumerate(self.blocks):
            # Block 9 is dropped on the unconditional branch, as in the reference
            # wan_animate_2_model.forward_gen; the authors give no rationale.
            if is_unconditional and idx == 9:
                continue
            hidden_states = block(
                kv_cache_mode="gen",
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep_modulation=timestep_modulation,
                gen_cos=generation_video_rope_cos,
                gen_sin=generation_video_rope_sin,
                block_mask=block_mask,
                layout=layout,
                index=idx,
                k_cache=reference_kv.k,
                v_cache=reference_kv.v,
            )
        return hidden_states


EntryClass = WanAnimate2Transformer3DModel
