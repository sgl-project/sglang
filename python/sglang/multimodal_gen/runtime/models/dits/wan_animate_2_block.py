# SPDX-License-Identifier: Apache-2.0
#
# Wan-Animate-2 DiT block: in-context self-attention over [gen tokens, then cached
# reference-video K/V], its RoPE / Ulysses helpers, and the per-clip _InContextLayout
# geometry.
# Names kept from the reference implementation: the forward_ref / forward_gen methods and the
# kv_cache_mode values "ref" / "gen"; there "ref" means the reference video (the motion source), not the reference image.
from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache, partial
from typing import Any, Literal

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from sglang.multimodal_gen.runtime.distributed import (
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all_qkv as _sp_gather_qkv,
)
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.models.dits.wanvideo import WanTransformerBlock

# Eager flex_attention materializes the full [Q, KV] scores over the padded [gen | reference-video]
# sequence (tens of GiB at video resolution); compiled, it is one fused kernel.
_flex_attention_compiled = torch.compile(
    flex_attention,
    dynamic=False,
    mode="max-autotune",
    fullgraph=True,
    backend="inductor",
)


def _apply_rope_interleaved(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Interleaved (GPT-J style) RoPE, identical to ``_apply_rotary_emb(is_neox_style=False)``;
    ``x`` is ``[B, S, H, D]``, ``cos`` / ``sin`` are ``[S, D//2]`` fp32."""
    cos = cos[None, :, None, :]  # [1, S, 1, D//2]
    sin = sin[None, :, None, :]
    x1 = x[..., 0::2].float()
    x2 = x[..., 1::2].float()
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    out = torch.stack([o1, o2], dim=-1).flatten(-2)
    return out.type_as(x)


def _score_mod_impl(
    score: torch.Tensor,
    b_idx: torch.Tensor,
    h_idx: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    hw: int,
    log_scale: float,
) -> torch.Tensor:
    # Biases the keys of generation frame 1, [hw, 2*hw) in the padded layout, exactly as the
    # official Wan-Animate-2 code does; non-zero only when the checkpoint config sets log_scale.
    condition = (kv_idx >= hw) & (kv_idx < 2 * hw)
    return torch.where(condition, score + log_scale, score)


@lru_cache(maxsize=None)
def _make_score_mod(hw: int, log_scale: float) -> Callable[..., torch.Tensor]:
    # Stable object identity per (hw, log_scale): compiled flex_attention guards on the
    # score_mod object, so a fresh closure per call would recompile every clip.
    return partial(_score_mod_impl, hw=hw, log_scale=log_scale)


def _sp_scatter_out(attn_output: torch.Tensor) -> torch.Tensor:
    """Ulysses output a2a: [B, S_full, N_local, C] -> [B, S_local, N_tp, C]."""
    return _usp_output_all_to_all(attn_output, head_dim=2)


def _reference_video_key_mask(
    *, batch_size: int, seq_len: int, num_valid: int, device: torch.device
) -> torch.Tensor:
    """``[B, S]`` bool key mask: True for the leading ``num_valid`` reference-video tokens."""
    key_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    key_mask[:, :num_valid] = True
    return key_mask


class WanAnimate2TransformerBlock(WanTransformerBlock):
    """Wan block with the Wan-Animate-2 in-context self-attention.

    No new parameters: forward_ref / forward_gen reuse the inherited modules and only
    route the self-attention differently.
    """

    def __init__(self, *args: Any, log_scale: float = 0.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.use_offline_qk_rotation:
            raise ValueError(
                "Wan-Animate-2 blocks do not apply the offline Q/K rotation that "
                "SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION selects for this checkpoint."
            )
        # Additive score bias on the keys of generation frame 1 (_score_mod_impl); 0.0 skips it.
        self.log_scale = log_scale

    def _modulate(
        self, timestep_modulation: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """14B modulation split of ``timestep_modulation`` ``[B, 6, C]`` into six ``[B, 1, C]`` fp32 chunks; mirrors wanvideo.py."""
        assert timestep_modulation.dim() == 3, (
            "WanAnimate2TransformerBlock expects the 14B timestep modulation [B, 6, inner]; the ti2v "
            "seq-len modulation path is not used by Wan-Animate-2."
        )
        e = self.scale_shift_table + timestep_modulation.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
            6, dim=1
        )
        return shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa

    def forward(
        self, *, kv_cache_mode: Literal["ref", "gen"], **kwargs: Any
    ) -> torch.Tensor:
        """Dispatch to forward_ref / forward_gen through ``nn.Module.__call__`` so the
        layerwise-offload module hooks fire for both passes; keyword arguments only."""
        if kv_cache_mode == "ref":
            return self.forward_ref(**kwargs)
        if kv_cache_mode == "gen":
            return self.forward_gen(**kwargs)
        raise ValueError(
            f"WanAnimate2TransformerBlock: unknown kv_cache_mode {kv_cache_mode!r} "
            "(expected 'ref' or 'gen')."
        )

    def forward_ref(
        self,
        *,
        hidden_states: torch.Tensor,  # [B, S_reference_video, C] reference-video latent stream
        encoder_hidden_states: torch.Tensor,  # [B, 257 + L, C] reference-video context (CLIP tokens, then text)
        timestep_modulation: torch.Tensor,  # [B, 6, C], reference-video timestep forced to 1
        reference_video_rope_cos: torch.Tensor,  # reference-video RoPE cos [reference_video_num_tokens, D//2]
        reference_video_rope_sin: torch.Tensor,
        reference_video_num_tokens: int,  # reference_video_f * reference_video_h * reference_video_w (valid reference-video tokens)
        index: int,
        k_cache: dict[int, torch.Tensor],
        v_cache: dict[int, torch.Tensor],
        # [B, S_reference_video] key mask for the SP tail padding, built once per clip by the
        # caller; None when the gathered sequence has no padding.
        reference_video_key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cache this block's rotated reference-video K and V, then run the reference-video self-attention
        (reference-video RoPE, pad keys masked) and return the updated reference-video stream."""
        orig_dtype = hidden_states.dtype
        shift, scale, gate, c_shift, c_scale, c_gate = self._modulate(
            timestep_modulation
        )

        norm_hidden_states = self.norm1(hidden_states, shift, scale)
        query, key, value = self._qkv(norm_hidden_states, self.local_num_heads)

        # Ulysses SP: gather the full reference-video sequence / shard heads before caching, so the
        # cache matches forward_gen's post-a2a layout.
        sp_size = get_sp_world_size()
        if sp_size > 1:
            query, key, value = _sp_gather_qkv(query, key, value)

        # Reference-video self-attention. Under SP, S_reference_video is the gathered
        # (tail-padded) length: RoPE the valid slice and mask the pad keys. Without padding the
        # whole sequence is valid and no mask is passed: USPAttention serves a dense key mask
        # through SDPA instead of the configured backend.
        reference_video_seq_len = query.shape[1]
        if reference_video_seq_len == reference_video_num_tokens:
            q_roped = _apply_rope_interleaved(
                query, reference_video_rope_cos, reference_video_rope_sin
            )
            k_roped = _apply_rope_interleaved(
                key, reference_video_rope_cos, reference_video_rope_sin
            )
            key_mask = None
        else:
            q_roped = query.clone()
            k_roped = key.clone()
            q_roped[:, :reference_video_num_tokens] = _apply_rope_interleaved(
                query[:, :reference_video_num_tokens],
                reference_video_rope_cos,
                reference_video_rope_sin,
            )
            k_roped[:, :reference_video_num_tokens] = _apply_rope_interleaved(
                key[:, :reference_video_num_tokens],
                reference_video_rope_cos,
                reference_video_rope_sin,
            )
            key_mask = reference_video_key_mask
            if key_mask is None:
                key_mask = _reference_video_key_mask(
                    batch_size=hidden_states.shape[0],
                    seq_len=reference_video_seq_len,
                    num_valid=reference_video_num_tokens,
                    device=hidden_states.device,
                )
        # Reference positions are invariant within a clip. Keep the rotated key in the
        # original cache storage; SP's value can share the gathered QKV allocation.
        k_cache[index] = key.copy_(k_roped)
        v_cache[index] = value
        if sp_size > 1:
            # q/k/v are already gathered: skip USPAttention's own a2a, then scatter back.
            attn_output = self.attn1(
                q_roped,
                k_roped,
                value,
                attn_mask=key_mask,
                skip_sequence_parallel_override=True,
            )
            attn_output = _sp_scatter_out(attn_output)
        else:
            attn_output = self.attn1(q_roped, k_roped, value, attn_mask=key_mask)
        attn_output = attn_output.flatten(2)

        return self._cross_and_ffn(
            hidden_states,
            attn_output,
            encoder_hidden_states,
            gate,
            c_shift,
            c_scale,
            c_gate,
            orig_dtype,
        )

    def forward_gen(
        self,
        *,
        hidden_states: torch.Tensor,  # [B, seq_len, C] gen stream (padded)
        encoder_hidden_states: torch.Tensor,  # [B, 257 + L, C] gen context (CLIP tokens, then text)
        timestep_modulation: torch.Tensor,  # [B, 6, C]
        gen_cos: torch.Tensor,  # gen-RoPE cos [generation_video_num_tokens, D//2]
        gen_sin: torch.Tensor,
        block_mask: BlockMask,  # per full-length clip grid
        layout: _InContextLayout,
        index: int,
        k_cache: dict[int, torch.Tensor],
        v_cache: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """One generation block with in-context masked self-attention over
        [gen tokens, then cached reference-video]."""
        orig_dtype = hidden_states.dtype
        shift, scale, gate, c_shift, c_scale, c_gate = self._modulate(
            timestep_modulation
        )

        norm_hidden_states = self.norm1(hidden_states, shift, scale)
        query, key, value = self._qkv(norm_hidden_states, self.local_num_heads)

        # Ulysses SP: gather the full gen sequence / shard heads. The cached reference-video K/V are
        # already in this layout.
        sp_size = get_sp_world_size()
        if sp_size > 1:
            query, key, value = _sp_gather_qkv(query, key, value)

        # gen-RoPE on the valid gen tokens only (padding kept unrotated).
        num_generation_tokens = layout.generation_video_num_tokens
        q_roped = _apply_rope_interleaved(
            query[:, :num_generation_tokens], gen_cos, gen_sin
        )
        k_roped = _apply_rope_interleaved(
            key[:, :num_generation_tokens], gen_cos, gen_sin
        )
        if query.shape[1] != num_generation_tokens:
            q_roped = torch.cat([q_roped, query[:, num_generation_tokens:]], dim=1)
            k_roped = torch.cat([k_roped, key[:, num_generation_tokens:]], dim=1)

        # Reference-video K is already rotated by forward_ref, including SP tail padding.
        reference_video_k = k_cache[index]
        reference_video_v = v_cache[index]

        attn_output = self._in_context_attention(
            q_roped,
            k_roped,
            value,
            reference_video_k,
            reference_video_v,
            block_mask,
            layout,
        )
        if sp_size > 1:
            # scatter the sequence back / gather heads.
            attn_output = _sp_scatter_out(attn_output)
        attn_output = attn_output.flatten(2)

        return self._cross_and_ffn(
            hidden_states,
            attn_output,
            encoder_hidden_states,
            gate,
            c_shift,
            c_scale,
            c_gate,
            orig_dtype,
        )

    def _in_context_attention(
        self,
        q: torch.Tensor,  # [B, seq_len, H, D] (gen-RoPE'd)
        k: torch.Tensor,  # [B, seq_len, H, D] (gen-RoPE'd)
        v: torch.Tensor,  # [B, seq_len, H, D]
        reference_video_k: torch.Tensor,  # [B, reference_video_seq_len, H, D] (reference-video RoPE'd)
        reference_video_v: torch.Tensor,  # [B, reference_video_seq_len, H, D]
        block_mask: BlockMask,
        layout: _InContextLayout,
    ) -> torch.Tensor:
        """Build the padded [gen | reference-video] layout and run compiled flex_attention.

        The block-mask / score_mod are over sequence positions only, so TP composes
        trivially, and under Ulysses SP the callers hand in the full sequence with sharded
        heads, so the layout and mask are identical to sp=1."""
        batch_size, _, num_heads, head_dim = q.shape
        device, dtype = q.device, q.dtype
        padded_generation_video_len = layout.padded_generation_video_len
        padded_kv_len = layout.padded_kv_len

        q_padding = q[:, layout.generation_video_num_tokens :]

        q_padded = torch.zeros(
            batch_size,
            padded_generation_video_len,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        k_padded = torch.zeros(
            batch_size, padded_kv_len, num_heads, head_dim, device=device, dtype=dtype
        )
        v_padded = torch.zeros(
            batch_size, padded_kv_len, num_heads, head_dim, device=device, dtype=dtype
        )

        num_latent_frames = layout.generation_video_f
        latent_hw = layout.generation_video_tokens_per_frame
        q_valid = q[:, : layout.generation_video_num_tokens].view(
            batch_size, num_latent_frames, latent_hw, num_heads, head_dim
        )
        k_valid = k[:, : layout.generation_video_num_tokens].view(
            batch_size, num_latent_frames, latent_hw, num_heads, head_dim
        )
        v_valid = v[:, : layout.generation_video_num_tokens].view(
            batch_size, num_latent_frames, latent_hw, num_heads, head_dim
        )
        max_tokens_per_frame = layout.max_tokens_per_frame
        q_padded[:, : num_latent_frames * max_tokens_per_frame].view(
            batch_size, num_latent_frames, max_tokens_per_frame, num_heads, head_dim
        )[:, :, :latent_hw] = q_valid
        k_padded[:, : num_latent_frames * max_tokens_per_frame].view(
            batch_size, num_latent_frames, max_tokens_per_frame, num_heads, head_dim
        )[:, :, :latent_hw] = k_valid
        v_padded[:, : num_latent_frames * max_tokens_per_frame].view(
            batch_size, num_latent_frames, max_tokens_per_frame, num_heads, head_dim
        )[:, :, :latent_hw] = v_valid

        reference_video_f, reference_video_tokens_per_frame = (
            layout.reference_video_f,
            layout.reference_video_tokens_per_frame,
        )
        reference_video_k_valid = reference_video_k[
            :, : layout.reference_video_num_tokens
        ].view(
            batch_size,
            reference_video_f,
            reference_video_tokens_per_frame,
            num_heads,
            head_dim,
        )
        reference_video_v_valid = reference_video_v[
            :, : layout.reference_video_num_tokens
        ].view(
            batch_size,
            reference_video_f,
            reference_video_tokens_per_frame,
            num_heads,
            head_dim,
        )
        k_padded[
            :,
            padded_generation_video_len : padded_generation_video_len
            + reference_video_f * max_tokens_per_frame,
        ].view(
            batch_size, reference_video_f, max_tokens_per_frame, num_heads, head_dim
        )[:, :, :reference_video_tokens_per_frame] = reference_video_k_valid
        v_padded[
            :,
            padded_generation_video_len : padded_generation_video_len
            + reference_video_f * max_tokens_per_frame,
        ].view(
            batch_size, reference_video_f, max_tokens_per_frame, num_heads, head_dim
        )[:, :, :reference_video_tokens_per_frame] = reference_video_v_valid

        # At 0.0 the bias is an exact identity, so the kernel is not asked to apply it;
        # otherwise the cached partial keeps the identity compiled flex guards on (_make_score_mod).
        log_scale = float(self.log_scale)
        _score_mod = (
            None
            if log_scale == 0.0
            else _make_score_mod(max_tokens_per_frame, log_scale)
        )

        # flex_attention takes [B, H, L, D]; the default 1/sqrt(D) scale matches USPAttention.
        attn_output_padded = _flex_attention_compiled(
            q_padded.transpose(1, 2),
            k_padded.transpose(1, 2),
            v_padded.transpose(1, 2),
            block_mask=block_mask,
            score_mod=_score_mod,
        ).transpose(1, 2)  # -> [B, padded_generation_video_len, N, C]

        attn_output_valid = attn_output_padded[
            :, : num_latent_frames * max_tokens_per_frame
        ].view(
            batch_size, num_latent_frames, max_tokens_per_frame, num_heads, head_dim
        )[:, :, :latent_hw]
        attn_output_valid = attn_output_valid.reshape(
            batch_size, num_latent_frames * latent_hw, num_heads, head_dim
        )
        if q_padding.shape[1] == 0:
            return attn_output_valid
        return torch.cat([attn_output_valid, q_padding], dim=1)  # [B, seq_len, N, C]


class _InContextLayout:
    """Token geometry of the padded [generation | reference-video] in-context layout.

    Three sets of sizes: this clip's token counts; those of a full-length clip, which every
    clip is padded up to; and the full-length sizes rounded up to 128 for flex_attention.
    The attention pads to the last set and the block mask is built from the same numbers,
    so one compiled mask serves every clip of a request."""

    def __init__(
        self,
        generation_video_grid: tuple[int, int, int],
        reference_video_grid: tuple[int, int, int],
        full_clip_grid: tuple[int, int, int],
    ) -> None:
        f, h, w = generation_video_grid
        reference_video_f, reference_video_h, reference_video_w = reference_video_grid
        full_clip_f, full_clip_h, full_clip_w = full_clip_grid
        self.generation_video_f, self.generation_video_h, self.generation_video_w = (
            f,
            h,
            w,
        )
        self.reference_video_f, self.reference_video_h, self.reference_video_w = (
            reference_video_f,
            reference_video_h,
            reference_video_w,
        )

        # This clip; shorter than full length only for the last clip of a video.
        self.generation_video_tokens_per_frame = h * w
        self.reference_video_tokens_per_frame = reference_video_h * reference_video_w
        self.generation_video_num_tokens = f * h * w
        self.reference_video_num_tokens = (
            reference_video_f * reference_video_h * reference_video_w
        )

        # A full-length clip. The generation video carries one extra reference-image frame,
        # the reference video does not.
        self.max_tokens_per_frame = full_clip_h * full_clip_w
        self.max_generation_video_num_tokens = full_clip_f * self.max_tokens_per_frame
        self.max_reference_video_num_tokens = (
            full_clip_f - 1
        ) * self.max_tokens_per_frame

        # Padded to flex_attention's 128-token blocks; keys are [generation | reference video].
        self.padded_generation_video_len = (
            math.ceil(self.max_generation_video_num_tokens / 128) * 128
        )
        self.padded_reference_video_len = (
            math.ceil(self.max_reference_video_num_tokens / 128) * 128
        )
        self.padded_kv_len = (
            self.padded_generation_video_len + self.padded_reference_video_len
        )

    def block_mask_fn(self) -> Callable[..., torch.Tensor]:
        """Returns the block mask function based on the [gen | reference-video] in-context layout.

        A valid gen query attends every valid gen key plus the reference-video keys of its own frame.

        keys:     [generation tokens | generation pad | reference tokens | reference pad]
                  0        padded_generation_video_len                      padded_kv_len
        queries:  [generation tokens | generation pad]
                  0        padded_generation_video_len
        """
        tokens_per_frame = self.max_tokens_per_frame
        num_generation_tokens = self.max_generation_video_num_tokens
        num_reference_tokens = self.max_reference_video_num_tokens
        reference_start = self.padded_generation_video_len

        def _mask_function(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ) -> torch.Tensor:
            # Pad positions are not tokens; pad queries attend nothing.
            q_is_generation_token = q_idx < num_generation_tokens
            # Generation frame of the query; frame 0 is the reference image.
            q_frame = q_idx // tokens_per_frame

            # Every generation token attends every generation token.
            kv_is_generation_token = kv_idx < num_generation_tokens

            # Reference-video tokens are attended only by the queries of their own frame.
            kv_reference_idx = kv_idx - reference_start
            kv_is_reference_token = (kv_idx >= reference_start) & (
                kv_reference_idx < num_reference_tokens
            )
            # +1: the reference video has no counterpart for the generation stream's
            # reference-image frame 0.
            kv_reference_frame = kv_reference_idx // tokens_per_frame + 1
            kv_is_same_frame_reference_token = kv_is_reference_token & (
                q_frame == kv_reference_frame
            )
            return q_is_generation_token & (
                kv_is_generation_token | kv_is_same_frame_reference_token
            )

        return _mask_function
