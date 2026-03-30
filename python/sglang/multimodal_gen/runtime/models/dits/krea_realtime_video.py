# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache

import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    get_sp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    NDRotaryEmbedding,
    get_rotary_pos_embed,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.models.dits.causal_wan_common import (
    BaseCausalWanSelfAttention,
    BaseCausalWanTransformer3DModel,
    BaseCausalWanTransformerBlock,
    _ForwardShapeInfo,
)
from sglang.multimodal_gen.runtime.pipelines_core.kv_cache import (
    CrossAttentionKVCache,
    SelfAttentionKVCache,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)


def _is_krea_sequence_shard_enabled() -> bool:
    if get_sp_world_size() <= 1:
        return False
    try:
        forward_batch = get_forward_context().forward_batch
    except AssertionError:
        return False
    if forward_batch is None:
        return False
    return bool(getattr(forward_batch, "enable_sequence_shard", False))


def _get_krea_sequence_shard_layout() -> tuple[int, int] | None:
    """Return per-frame shard layout: (local_slots, valid_tokens).

    local_slots includes right-padding used for even SP sharding, while
    valid_tokens excludes those padded tokens.
    """
    try:
        forward_batch = get_forward_context().forward_batch
    except AssertionError:
        return None
    if forward_batch is None:
        return None
    local = getattr(forward_batch, "_krea_sp_local_tokens_per_frame", None)
    valid = getattr(forward_batch, "_krea_sp_valid_tokens_per_frame", None)
    if local is None or valid is None:
        return None
    return int(local), int(valid)


class KreaCausalWanSelfAttention(BaseCausalWanSelfAttention):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ulysses_attn = USPAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=(
                AttentionBackendEnum.FA,
                AttentionBackendEnum.AITER,
                AttentionBackendEnum.TORCH_SDPA,
            ),
        )

    def _should_use_flex_attention(self, block_mask, kv_cache) -> bool:
        if _is_krea_sequence_shard_enabled():
            # flex_attention path has no Ulysses collectives; use regular attention
            # with KV cache updates instead.
            return False
        return kv_cache is None or block_mask is not None

    def _prepare_flex_cache(
        self,
        roped_key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: SelfAttentionKVCache | None,
    ) -> None:
        if kv_cache is not None:
            # Bulk write mode: populate cache from position 0 and use flex_attention.
            kv_cache.bulk_write(roped_key, value)

    def _incremental_attention(
        self,
        roped_query: torch.Tensor,
        roped_key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: SelfAttentionKVCache | None,
        current_start: int,
        cache_start: int,
    ) -> torch.Tensor:
        attn = self.ulysses_attn if _is_krea_sequence_shard_enabled() else self.attn
        key_for_attn = roped_key
        value_for_attn = value
        append_start = current_start
        shard_layout = (
            _get_krea_sequence_shard_layout()
            if _is_krea_sequence_shard_enabled()
            else None
        )
        if shard_layout is not None:
            # Keep communication-friendly fixed local slots per frame, but strip
            # padded tail tokens before updating KV cache to avoid cache pollution.
            local_tokens_per_frame, valid_tokens_per_frame = shard_layout
            if (
                0 < valid_tokens_per_frame < local_tokens_per_frame
                and local_tokens_per_frame > 0
                and roped_key.shape[1] % local_tokens_per_frame == 0
            ):
                num_frames = roped_key.shape[1] // local_tokens_per_frame
                bsz, _, nheads, hdim = roped_key.shape
                key_for_attn = roped_key.view(
                    bsz, num_frames, local_tokens_per_frame, nheads, hdim
                )[:, :, :valid_tokens_per_frame].reshape(
                    bsz, num_frames * valid_tokens_per_frame, nheads, hdim
                )
                value_for_attn = value.view(
                    bsz, num_frames, local_tokens_per_frame, nheads, hdim
                )[:, :, :valid_tokens_per_frame].reshape(
                    bsz, num_frames * valid_tokens_per_frame, nheads, hdim
                )
                # current_start is in "local slot" units; convert it to
                # "valid token" units so cache indices stay consistent.
                frame_idx = current_start // local_tokens_per_frame
                append_start = frame_idx * valid_tokens_per_frame

        if kv_cache is None:
            return attn(roped_query, key_for_attn, value_for_attn)
        kv_cache.append(key_for_attn, value_for_attn, append_start)
        active_k, active_v = kv_cache.get_active_kv(self.max_attention_size)
        return attn(roped_query, active_k, active_v)


class KreaCausalWanTransformerBlock(BaseCausalWanTransformerBlock):
    self_attn_cls = KreaCausalWanSelfAttention


class KreaCausalWanTransformer3DModel(BaseCausalWanTransformer3DModel):
    block_cls = KreaCausalWanTransformerBlock

    def _get_rope_embed_kwargs(self, hidden_states: torch.Tensor) -> dict:
        # Krea path keeps rotary embeddings on-device to avoid extra transfers.
        return {"device": hidden_states.device}

    def _use_gradient_checkpointing_inference(self) -> bool:
        # Keep Krea inference path simple and avoid extra graph machinery.
        return False

    @lru_cache(maxsize=1)
    def _get_nd_rotary_embedder(self) -> NDRotaryEmbedding:
        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        return NDRotaryEmbedding(
            rope_dim_list=rope_dim_list,
            rope_theta=10000,
            dtype=(
                torch.float32
                if current_platform.is_mps() or current_platform.is_musa()
                else torch.float64
            ),
        )

    def _compute_rope_for_spatial_shard(
        self,
        post_patch_num_frames: int,
        post_patch_width: int,
        frame_seq_start: int,
        frame_seq_len: int,
        start_frame: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_offsets = torch.arange(
            frame_seq_start,
            frame_seq_start + frame_seq_len,
            device=device,
            dtype=torch.long,
        )
        t_idx = (
            torch.arange(
                start_frame,
                start_frame + post_patch_num_frames,
                device=device,
                dtype=torch.long,
            )
            .unsqueeze(1)
            .expand(post_patch_num_frames, frame_seq_len)
            .reshape(-1)
        )
        rem = token_offsets.unsqueeze(0).expand(post_patch_num_frames, frame_seq_len)
        rem = rem.reshape(-1)
        h_idx = rem // post_patch_width
        w_idx = rem % post_patch_width
        positions = torch.stack((t_idx, h_idx, w_idx), dim=1)
        freqs_cos, freqs_sin = self._get_nd_rotary_embedder().forward_cuda(positions)
        return freqs_cos.float(), freqs_sin.float()

    def _forward_inference(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        kv_cache: list[SelfAttentionKVCache] = None,
        crossattn_cache: list[CrossAttentionKVCache] = None,
        current_start: int = 0,
        cache_start: int = 0,
        start_frame: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        # Reuse the same request flag as Wan sequence shard so Krea can enable
        # Ulysses without introducing a separate user-facing switch.
        sequence_shard_enabled = _is_krea_sequence_shard_enabled()
        sp_size = get_sp_world_size()
        sp_rank = get_sp_group().rank_in_group if sequence_shard_enabled else 0
        try:
            forward_batch = get_forward_context().forward_batch
        except AssertionError:
            forward_batch = None

        orig_dtype = hidden_states.dtype
        encoder_hidden_states = self._maybe_first_tensor(encoder_hidden_states)
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must not be empty")
        encoder_hidden_states_image = self._maybe_first_tensor(
            encoder_hidden_states_image
        )
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        frame_seq_len = post_patch_height * post_patch_width
        seq_shard_pad = 0
        local_frame_seq_len = frame_seq_len
        valid_frame_seq_len = frame_seq_len
        frame_seq_start = 0

        # Patchify video latents -> token sequence [B, T'*H'*W', C].
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if sequence_shard_enabled:
            hidden_dim = hidden_states.shape[-1]
            # Shard sequence by per-frame spatial tokens instead of full flattened
            # sequence so each rank still owns all frames (needed by causal cache).
            hidden_states = hidden_states.view(
                batch_size,
                post_patch_num_frames,
                frame_seq_len,
                hidden_dim,
            )
            if frame_seq_len % sp_size != 0:
                seq_shard_pad = sp_size - (frame_seq_len % sp_size)
                pad = torch.zeros(
                    (
                        batch_size,
                        post_patch_num_frames,
                        seq_shard_pad,
                        hidden_dim,
                    ),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                hidden_states = torch.cat([hidden_states, pad], dim=2)
            # Per frame, each rank receives a fixed shard size after padding.
            local_frame_seq_len = hidden_states.shape[2] // sp_size
            frame_seq_start = sp_rank * local_frame_seq_len
            frame_seq_end = min(frame_seq_len, frame_seq_start + local_frame_seq_len)
            # Number of non-padding tokens owned by this rank in each frame.
            valid_frame_seq_len = max(frame_seq_end - frame_seq_start, 0)
            hidden_states = hidden_states.view(
                batch_size,
                post_patch_num_frames,
                sp_size,
                local_frame_seq_len,
                hidden_dim,
            )
            hidden_states = hidden_states[:, :, sp_rank].reshape(
                batch_size,
                post_patch_num_frames * local_frame_seq_len,
                hidden_dim,
            )
            if forward_batch is not None:
                # Expose layout for attention/KV cache path in the same forward.
                forward_batch._krea_sp_local_tokens_per_frame = local_frame_seq_len
                forward_batch._krea_sp_valid_tokens_per_frame = valid_frame_seq_len
        elif forward_batch is not None:
            forward_batch._krea_sp_local_tokens_per_frame = None
            forward_batch._krea_sp_valid_tokens_per_frame = None

        if sequence_shard_enabled:
            # Build RoPE on local sharded token coordinates.
            freqs_cos, freqs_sin = self._compute_rope_for_spatial_shard(
                post_patch_num_frames=post_patch_num_frames,
                post_patch_width=post_patch_width,
                frame_seq_start=frame_seq_start,
                frame_seq_len=local_frame_seq_len,
                start_frame=start_frame,
                device=hidden_states.device,
            )
            freqs_cis = (freqs_cos, freqs_sin)
            # KV cache indices are expressed in token offsets. Convert from global
            # per-frame width to this rank's local per-frame width.
            frame_idx = current_start // frame_seq_len if frame_seq_len > 0 else 0
            current_start = frame_idx * local_frame_seq_len
            if cache_start is not None:
                cache_frame_idx = (
                    cache_start // frame_seq_len if frame_seq_len > 0 else 0
                )
                cache_start = cache_frame_idx * local_frame_seq_len
        else:
            # Non-sharded path keeps original global RoPE generation.
            d = self.hidden_size // self.num_attention_heads
            rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
            rope_kwargs = self._get_rope_embed_kwargs(hidden_states)
            freqs_cos, freqs_sin = get_rotary_pos_embed(
                (
                    post_patch_num_frames * get_sp_world_size(),
                    post_patch_height,
                    post_patch_width,
                ),
                self.hidden_size,
                self.num_attention_heads,
                rope_dim_list,
                dtype=(
                    torch.float32
                    if current_platform.is_mps() or current_platform.is_musa()
                    else torch.float64
                ),
                rope_theta=10000,
                start_frame=start_frame,
                **rope_kwargs,
            )
            if freqs_cos is not None and freqs_cos.device != hidden_states.device:
                freqs_cos = freqs_cos.to(hidden_states.device)
                freqs_sin = freqs_sin.to(hidden_states.device)
            freqs_cis = (
                (freqs_cos.float(), freqs_sin.float())
                if freqs_cos is not None
                else None
            )

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep.flatten(), encoder_hidden_states, encoder_hidden_states_image
            )
        )
        timestep_proj = timestep_proj.unflatten(1, (6, self.hidden_size)).unflatten(
            dim=0, sizes=timestep.shape
        )

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )
        encoder_hidden_states = (
            encoder_hidden_states.to(orig_dtype)
            if current_platform.is_mps()
            else encoder_hidden_states
        )
        assert encoder_hidden_states.dtype == orig_dtype

        hidden_states = self._run_inference_blocks(
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            freqs_cis,
            kv_cache,
            crossattn_cache,
            current_start,
            cache_start,
        )

        if sequence_shard_enabled:
            hidden_dim = hidden_states.shape[-1]
            # Gather sharded spatial tokens back to full per-frame sequence before
            # output projection/unpatchify, then drop right padding.
            hidden_states = hidden_states.view(
                batch_size,
                post_patch_num_frames,
                local_frame_seq_len,
                hidden_dim,
            )
            hidden_states = sequence_model_parallel_all_gather(
                hidden_states.contiguous(), dim=2
            )
            if seq_shard_pad > 0:
                hidden_states = hidden_states[:, :, :frame_seq_len, :]
            hidden_states = hidden_states.reshape(
                batch_size,
                post_patch_num_frames * frame_seq_len,
                hidden_dim,
            )

        shape_info = _ForwardShapeInfo(
            batch_size=batch_size,
            num_frames=num_frames,
            post_patch_num_frames=post_patch_num_frames,
            post_patch_height=post_patch_height,
            post_patch_width=post_patch_width,
            p_t=p_t,
            p_h=p_h,
            p_w=p_w,
        )
        return self._project_to_output(hidden_states, temb, timestep, shape_info)

    def forward(self, *args, **kwargs):
        return self._forward_inference(*args, **kwargs)


EntryClass = KreaCausalWanTransformer3DModel
