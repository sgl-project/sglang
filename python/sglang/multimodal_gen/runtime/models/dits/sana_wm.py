# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.dits.sana_wm import SanaWMConfig
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT

# Re-exported for back-compat: callers import these names from this module path.
from sglang.multimodal_gen.runtime.models.dits.sana_wm_components import (  # noqa: F401
    _CACHE_TYPE_CONCAT,
    _CACHE_TYPE_STATE,
    _INT32_SAFE_CONV_ELEMENTS,
    _NUM_STREAM_CACHE_SLOTS,
    _SLOT_CAM_K,
    _SLOT_CAM_V,
    _SLOT_FFN_TCONV,
    _SLOT_K,
    _SLOT_SHORTCONV,
    _SLOT_TYPE_FLAG,
    _SLOT_V,
    BidirectionalGDNUCPESinglePathLiteLA,
    CaptionEmbedder,
    GLUMBConvTemp,
    MultiHeadCrossAttention,
    PatchEmbedMS3D,
    T2IFinalLayer,
    TimestepEmbedder,
    WanRotaryPosEmbed,
    _apply_block_diagonal,
    _apply_complex_rope,
    _apply_ray_projmat,
    _apply_rotary_emb_bhnd,
    _apply_rotary_emb_dn,
    _bidirectional_short_conv,
    _build_ucpe_apply_fns,
    _compute_fov_from_focal,
    _compute_frame_gates,
    _ConvLayer,
    _downscale_to_reference_rms,
    _flip_and_shift,
    _gdn_chunk_scan_forward,
    _gdn_scan_bidirectional,
    _gdn_scan_cached,
    _gdn_scan_forward,
    _gdn_scan_forward_stateful,
    _invert_SE3,
    _log_sana_wm_triton_cam_gdn_fallback,
    _log_sana_wm_triton_gdn_fallback,
    _RMSNorm,
    _sana_wm_chunk_boundaries_for_attention,
    _sana_wm_chunk_index_from_chunk_size,
    _sana_wm_chunked_attention,
    _sana_wm_normalize_chunk_index,
    _sana_wm_padded_scale,
    _sana_wm_sdpa,
    _ShortConvolution,
    _single_path_delta_chunk_scan_forward,
    _single_path_delta_scan_bidirectional,
    _single_path_delta_scan_cached,
    _single_path_delta_scan_forward,
    _single_path_delta_scan_forward_stateful,
    _sinusoidal_timestep_embedding,
    _slice_rope_for_cam,
    _slice_rope_to_current_chunk,
    _temporal_short_conv_cached,
    _tensor_cache_key,
    _UpstreamMlp,
    compute_chunk_plucker,
    process_camera_conditions_ucpe,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SanaWMBlock(nn.Module):
    """One transformer block of SANA-WM."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float,
        t_kernel_size: int,
        qk_norm: bool,
        cross_norm: bool,
        conv_kernel_size: int,
        k_conv_only: bool,
        softmax_main: bool,
        use_chunk_plucker_post_attn: bool,
        chunk_size: Optional[int] = None,
        chunk_split_strategy: str = "uniform",
        update_rule: str = "torch_chunk",
        cam_update_rule: str = "torch_chunk",
        chunk_gdn_chunk_size: int = 21,
        use_chunked_softmax_attention: bool = False,
        gdn_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.softmax_main = softmax_main
        self.chunk_size = chunk_size
        self.chunk_split_strategy = chunk_split_strategy

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = BidirectionalGDNUCPESinglePathLiteLA(
            in_dim=hidden_size,
            heads=num_heads,
            head_dim=head_dim,
            qk_norm=qk_norm,
            conv_kernel_size=conv_kernel_size,
            k_conv_only=k_conv_only,
            softmax_main=softmax_main,
            update_rule=update_rule,
            cam_update_rule=cam_update_rule,
            chunk_gdn_chunk_size=chunk_gdn_chunk_size,
            use_chunked_softmax_attention=use_chunked_softmax_attention,
            gdn_backend=gdn_backend,
        )

        self.cross_attn = MultiHeadCrossAttention(
            d_model=hidden_size,
            num_heads=num_heads,
            qk_norm=cross_norm,
        )

        self.mlp = GLUMBConvTemp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            t_kernel_size=t_kernel_size,
        )

        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size**0.5
        )

        if use_chunk_plucker_post_attn:
            self.plucker_proj = nn.Linear(hidden_size, hidden_size, bias=True)
            nn.init.zeros_(self.plucker_proj.weight)
            nn.init.zeros_(self.plucker_proj.bias)
        else:
            self.plucker_proj = None

    @staticmethod
    def _modulate(
        x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return x * (1 + scale) + shift

    @staticmethod
    def _reshape_framewise_modulation(
        x: torch.Tensor,
        num_frames: int,
    ) -> tuple[torch.Tensor, int]:
        B, N, C = x.shape
        tokens_per_frame = N // num_frames
        return x.reshape(B, num_frames, tokens_per_frame, C), tokens_per_frame

    def forward(
        self,
        x: torch.Tensor,  # (B, N, D)
        y: torch.Tensor,  # (B, L, D) text embeds
        t: torch.Tensor,  # (B, 6*D) AdaLN-single
        HW: Tuple[int, int, int],
        rotary_emb: Optional[torch.Tensor],
        prope_fns: Optional[Tuple[Callable, Callable, Callable]],
        plucker_emb: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        chunk_size: Optional[int] = None,
        chunk_split_strategy: Optional[str] = None,
        chunk_index: Optional[List[int]] = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        if t.dim() == 2:
            num_frames = None
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + t.reshape(B, 6, -1)
            ).chunk(6, dim=1)
        else:
            num_frames = t.reshape(B, -1, 6, t.shape[-1] // 6).shape[1]
            t = t.reshape(B, num_frames, 6, -1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None, None, :, :] + t
            ).chunk(6, dim=2)

        # Self-attention with UCPE camera branch
        if num_frames is None:
            x_in = self._modulate(self.norm1(x), shift_msa, scale_msa)
        else:
            x_norm, tokens_per_frame = self._reshape_framewise_modulation(
                self.norm1(x), num_frames
            )
            x_in = self._modulate(x_norm, shift_msa, scale_msa).reshape_as(x)
        attn_out = self.attn(
            x_in,
            HW=HW,
            rotary_emb=rotary_emb,
            prope_fns=prope_fns,
            chunk_size=self.chunk_size if chunk_size is None else chunk_size,
            chunk_split_strategy=(
                self.chunk_split_strategy
                if chunk_split_strategy is None
                else chunk_split_strategy
            ),
            chunk_index=chunk_index,
        )
        if num_frames is None:
            x = x + gate_msa * attn_out
        else:
            attn_out = attn_out.reshape(B, num_frames, tokens_per_frame, -1)
            x = x + (gate_msa * attn_out).reshape_as(x)

        # Plücker post-attn injection (zero-init linear)
        if self.plucker_proj is not None and plucker_emb is not None:
            x = x + self.plucker_proj(plucker_emb)

        # Cross-attention
        x = x + self.cross_attn(x, y, mask=mask)

        # FFN
        if num_frames is None:
            x_in = self._modulate(self.norm2(x), shift_mlp, scale_mlp)
            x = x + gate_mlp * self.mlp(x_in, HW=HW)
        else:
            x_norm, tokens_per_frame = self._reshape_framewise_modulation(
                self.norm2(x), num_frames
            )
            x_in = self._modulate(x_norm, shift_mlp, scale_mlp).reshape_as(x)
            mlp_out = self.mlp(x_in, HW=HW).reshape(B, num_frames, tokens_per_frame, -1)
            x = x + (gate_mlp * mlp_out).reshape_as(x)
        return x

    def forward_long(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        HW: Tuple[int, int, int],
        rotary_emb: Optional[torch.Tensor],
        prope_fns: Optional[Tuple[Callable, Callable, Callable]],
        plucker_emb: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        *,
        kv_cache: list,
        save_kv_cache: bool,
    ) -> Tuple[torch.Tensor, list]:
        """Streaming counterpart of ``forward``: threads the per-block 10-slot ``kv_cache`` through cached attention + FFN."""
        B = x.shape[0]
        if t.dim() == 2:
            num_frames = None
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + t.reshape(B, 6, -1)
            ).chunk(6, dim=1)
        else:
            num_frames = t.reshape(B, -1, 6, t.shape[-1] // 6).shape[1]
            t = t.reshape(B, num_frames, 6, -1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None, None, :, :] + t
            ).chunk(6, dim=2)

        if num_frames is None:
            x_in = self._modulate(self.norm1(x), shift_msa, scale_msa)
        else:
            x_norm, tokens_per_frame = self._reshape_framewise_modulation(
                self.norm1(x), num_frames
            )
            x_in = self._modulate(x_norm, shift_msa, scale_msa).reshape_as(x)

        attn_out, kv_cache = self.attn.forward_long(
            x_in,
            HW=HW,
            rotary_emb=rotary_emb,
            prope_fns=prope_fns,
            kv_cache=kv_cache,
            save_kv_cache=save_kv_cache,
        )
        if num_frames is None:
            x = x + gate_msa * attn_out
        else:
            attn_out = attn_out.reshape(B, num_frames, tokens_per_frame, -1)
            x = x + (gate_msa * attn_out).reshape_as(x)

        if self.plucker_proj is not None and plucker_emb is not None:
            x = x + self.plucker_proj(plucker_emb)

        x = x + self.cross_attn(x, y, mask=mask)

        if num_frames is None:
            x_in = self._modulate(self.norm2(x), shift_mlp, scale_mlp)
        else:
            x_norm, tokens_per_frame = self._reshape_framewise_modulation(
                self.norm2(x), num_frames
            )
            x_in = self._modulate(x_norm, shift_mlp, scale_mlp).reshape_as(x)

        # GLUMBConvTemp returns a tuple whenever the streaming path is active
        # (ffn_tail set OR save requested); branch on tuple-ness, never on
        # save_kv_cache alone (a read-only pass with a populated slot 9 still
        # returns a tuple).
        mlp_out = self.mlp(
            x_in,
            HW=HW,
            ffn_tail=kv_cache[_SLOT_FFN_TCONV],
            save_ffn_tail=save_kv_cache,
        )
        if isinstance(mlp_out, tuple):
            mlp_out, ffn_tail = mlp_out
            if save_kv_cache:
                kv_cache[_SLOT_FFN_TCONV] = ffn_tail
        if num_frames is None:
            x = x + gate_mlp * mlp_out
        else:
            mlp_out = mlp_out.reshape(B, num_frames, tokens_per_frame, -1)
            x = x + (gate_mlp * mlp_out).reshape_as(x)
        return x, kv_cache


class SanaWMTransformer3DModel(CachableDiT, LayerwiseOffloadableModuleMixin):
    """SANA-WM 2.6B TI2V world model.

    Forward inputs:
        hidden_states:           (B, C, T, H, W)        128-ch LTX-2 latent
        encoder_hidden_states:   (B, L, 2304)           Gemma-2 embeddings
        timestep:                (B,)
        encoder_attention_mask:  (B, L) optional bool
        camera_conditions:       (B, T, 20)             latent-frame raymap:
                                                        16 c2w + (fx,fy,cx,cy)
        chunk_plucker:           (B, 48, T, H, W)       optional, computed
                                                        from camera_conditions
                                                        if absent.

    Returns: ``(B, C, T, H, W)`` predicted velocity / noise.
    """

    _fsdp_shard_conditions = SanaWMConfig()._fsdp_shard_conditions
    _compile_conditions = SanaWMConfig()._compile_conditions
    _supported_attention_backends = SanaWMConfig()._supported_attention_backends
    param_names_mapping = SanaWMConfig().param_names_mapping
    reverse_param_names_mapping = SanaWMConfig().reverse_param_names_mapping
    lora_param_names_mapping: dict = {}

    def __init__(self, config: SanaWMConfig, hf_config=None, **kwargs) -> None:
        super().__init__(config, hf_config=hf_config or {}, **kwargs)
        arch = config.arch_config

        self.patch_size = (arch.patch_size_t, arch.patch_size, arch.patch_size)
        self.inner_dim = arch.num_attention_heads * arch.attention_head_dim
        self.hidden_size = self.inner_dim
        self.num_attention_heads = arch.num_attention_heads
        self.attention_head_dim = arch.attention_head_dim
        self.out_channels = arch.out_channels
        self.num_channels_latents = arch.num_channels_latents
        self.vae_temporal_stride = arch.vae_temporal_stride
        self.timestep_norm_scale_factor = getattr(
            arch, "timestep_norm_scale_factor", 1.0
        )

        # --- Embedders ---
        self.x_embedder = PatchEmbedMS3D(
            self.patch_size,
            arch.in_channels,
            self.inner_dim,
            bias=True,
        )

        self.t_embedder = TimestepEmbedder(self.inner_dim, frequency_embedding_size=256)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.inner_dim, 6 * self.inner_dim, bias=True),
        )

        self.y_embedder = CaptionEmbedder(
            in_channels=arch.caption_channels,
            hidden_size=self.inner_dim,
            token_num=arch.model_max_length,
        )
        self.y_norm = bool(getattr(arch, "y_norm", True))
        self.attention_y_norm = _RMSNorm(
            self.inner_dim,
            scale_factor=getattr(arch, "y_norm_scale_factor", 1.0),
            eps=getattr(arch, "y_norm_eps", 1e-5),
        )

        # 3-channel raymap embedder -- kept for state_dict compatibility but
        # only invoked when ``use_chunk_plucker_post_attn`` is False.
        # When ``True`` (the case for the released checkpoint) the absmap
        # path is skipped entirely.
        self.raymap_embedder = PatchEmbedMS3D(
            self.patch_size,
            3,
            self.inner_dim,
            bias=True,
        )
        # 48-channel plucker embedder (chunk-packed)
        if arch.use_chunk_plucker_post_attn or arch.use_chunk_plucker_input:
            self.plucker_embedder = PatchEmbedMS3D(
                self.patch_size,
                arch.chunk_plucker_channels,
                self.inner_dim,
                bias=True,
            )
            nn.init.zeros_(self.plucker_embedder.proj.weight)
            nn.init.zeros_(self.plucker_embedder.proj.bias)
        else:
            self.plucker_embedder = None
        self.use_chunk_plucker_post_attn = arch.use_chunk_plucker_post_attn
        self.use_chunk_plucker_input = arch.use_chunk_plucker_input
        self.chunk_size = getattr(arch, "chunk_size", None)
        self.chunk_split_strategy = getattr(arch, "chunk_split_strategy", "uniform")

        # --- RoPE ---
        self.rope = WanRotaryPosEmbed(
            attention_head_dim=arch.linear_head_dim,
            patch_size=self.patch_size,
            max_seq_len=1024,
        )

        # --- Transformer blocks ---
        depth = arch.num_layers
        self.softmax_every_n = arch.softmax_every_n
        softmax_idx = set(
            i
            for i in range(depth)
            if arch.softmax_every_n > 0 and (i + 1) % arch.softmax_every_n == 0
        )
        self.softmax_block_indices = tuple(sorted(softmax_idx))

        self.blocks = nn.ModuleList(
            [
                SanaWMBlock(
                    hidden_size=self.inner_dim,
                    num_heads=arch.num_attention_heads,
                    head_dim=arch.linear_head_dim,
                    mlp_ratio=arch.mlp_ratio,
                    t_kernel_size=arch.t_kernel_size,
                    qk_norm=arch.qk_norm,
                    cross_norm=arch.cross_norm,
                    conv_kernel_size=arch.conv_kernel_size,
                    k_conv_only=arch.k_conv_only,
                    softmax_main=(i in softmax_idx),
                    use_chunk_plucker_post_attn=(
                        arch.use_chunk_plucker_post_attn
                        and (
                            arch.chunk_plucker_post_attn_blocks < 0
                            or i < arch.chunk_plucker_post_attn_blocks
                        )
                    ),
                    chunk_size=self.chunk_size,
                    chunk_split_strategy=self.chunk_split_strategy,
                    update_rule=getattr(arch, "update_rule", "torch_chunk"),
                    cam_update_rule=getattr(arch, "cam_update_rule", "torch_chunk"),
                    chunk_gdn_chunk_size=getattr(arch, "chunk_gdn_chunk_size", 21),
                    use_chunked_softmax_attention=getattr(
                        arch, "use_chunked_softmax_attention", False
                    ),
                    gdn_backend=getattr(arch, "gdn_backend", "auto"),
                )
                for i in range(depth)
            ]
        )

        self.final_layer = T2IFinalLayer(
            self.inner_dim, self.patch_size, self.out_channels
        )

        # Cache RoPE freqs per shape -- avoids recomputation across denoising
        # steps with constant latent shapes.
        self._freqs_cache: dict = {}
        self._ucpe_apply_fns_cache: Optional[
            Tuple[Tuple, torch.Tensor, Tuple[Callable, Callable, Callable]]
        ] = None
        self._plucker_emb_cache: Optional[Tuple[Tuple, torch.Tensor, torch.Tensor]] = (
            None
        )

        # FSDP shard targets
        self.layer_names = ["blocks"]

    def post_load_weights(self) -> None:
        # FSDP loader initializes the model on meta and only materializes
        # tensors that appear in the checkpoint. WanRotaryPosEmbed._freqs is a
        # derived, non-persistent constant, so recompute it deterministically.
        for module in self.modules():
            if isinstance(module, WanRotaryPosEmbed):
                if module._freqs.is_meta:
                    module._init_freqs_buffer()

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def _get_freqs(self, T: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (T, H, W, str(device))
        if key not in self._freqs_cache:
            self._freqs_cache[key] = self.rope((T, H, W), device)
        return self._freqs_cache[key]

    def _get_freqs_window(
        self,
        start: int,
        end: int,
        H: int,
        W: int,
        device: torch.device,
        frame_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """RoPE freqs for a streaming chunk at GLOBAL frame positions.

        ``frame_index`` (per-token global positions) overrides ``(start, end)``;
        the count branch is cached, the tensor branch is computed fresh.
        """
        if frame_index is not None:
            return self.rope((end - start, H, W), device, frame_index=frame_index)
        key = ("win", int(start), int(end), H, W, str(device))
        if key not in self._freqs_cache:
            self._freqs_cache[key] = self.rope(((int(start), int(end)), H, W), device)
        return self._freqs_cache[key]

    def _get_ucpe_apply_fns(
        self,
        camera_conditions: torch.Tensor,
        *,
        HW: Tuple[int, int, int],
        freqs: torch.Tensor,
    ) -> Tuple[Callable, Callable, Callable]:
        head_dim = self.attention_head_dim
        if torch.is_grad_enabled():
            raymats = process_camera_conditions_ucpe(
                camera_conditions,
                HW=HW,
                patch_size=self.patch_size,
            )
            raymats_flat = raymats.reshape(camera_conditions.shape[0], -1, 4, 4)
            return _build_ucpe_apply_fns(head_dim, raymats_flat, freqs)

        key = (
            "ucpe",
            HW,
            self.patch_size,
            head_dim,
            _tensor_cache_key(camera_conditions),
            _tensor_cache_key(freqs),
        )
        cached = self._ucpe_apply_fns_cache
        if cached is not None and cached[0] == key:
            return cached[2]

        raymats = process_camera_conditions_ucpe(
            camera_conditions,
            HW=HW,
            patch_size=self.patch_size,
        )
        raymats_flat = raymats.reshape(camera_conditions.shape[0], -1, 4, 4)
        prope_fns = _build_ucpe_apply_fns(head_dim, raymats_flat, freqs)
        self._ucpe_apply_fns_cache = (key, camera_conditions, prope_fns)
        return prope_fns

    def _get_plucker_emb(
        self,
        chunk_plucker: torch.Tensor,
        *,
        latent_token_count: int,
    ) -> torch.Tensor:
        if self.plucker_embedder is None:
            raise ValueError("SANA-WM plucker_embedder is not initialized.")

        weight = self.plucker_embedder.proj.weight
        bias = self.plucker_embedder.proj.bias
        key = (
            "plucker_emb",
            latent_token_count,
            self.patch_size,
            _tensor_cache_key(chunk_plucker),
            _tensor_cache_key(weight),
            None if bias is None else _tensor_cache_key(bias),
        )
        if not torch.is_grad_enabled():
            cached = self._plucker_emb_cache
            if cached is not None and cached[0] == key:
                return cached[2]

        plucker_emb = self.plucker_embedder(chunk_plucker.to(weight.dtype))
        if plucker_emb.shape[1] != latent_token_count:
            raise ValueError(
                f"plucker_emb token count {plucker_emb.shape[1]} != "
                f"latent token count {latent_token_count}; "
                "expected chunk_plucker shape (B, 48, T, H, W)."
            )

        if not torch.is_grad_enabled():
            self._plucker_emb_cache = (key, chunk_plucker, plucker_emb)
        return plucker_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        camera_conditions: Optional[torch.Tensor] = None,
        chunk_plucker: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,  # kept for compat
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            raise ValueError("SANA-WM forward requires encoder_hidden_states.")
        if timestep is None:
            raise ValueError("SANA-WM forward requires timestep.")

        B, C, T_raw, H_raw, W_raw = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        T = T_raw // p_t
        H = H_raw // p_h
        W = W_raw // p_w
        chunk_size = kwargs.get("chunk_size", self.chunk_size)
        chunk_split_strategy = kwargs.get(
            "chunk_split_strategy", self.chunk_split_strategy
        )
        chunk_index = kwargs.get("chunk_index", None)

        # Patch embed: (B, C, T, H, W) -> (B, T*H*W, D)
        x = self.x_embedder(hidden_states.to(dtype=self.x_embedder.proj.weight.dtype))

        # Timestep AdaLN-single. SANA-WM's LTX sampler passes per-frame
        # timesteps shaped (B, 1, T) so the clean first-frame condition can stay
        # at timestep 0 while remaining latent frames denoise. Keep the scalar
        # path for generic scheduler compatibility.
        if self.timestep_norm_scale_factor != 1.0:
            timestep_for_embed = (
                timestep.float() / self.timestep_norm_scale_factor
            ).to(torch.float32)
        else:
            timestep_for_embed = timestep.long().to(torch.float32)

        if timestep_for_embed.dim() == 1:
            t_emb = self.t_embedder(timestep_for_embed)  # (B, D)
            t6 = self.t_block(t_emb)  # (B, 6D)
        else:
            timestep_shape = tuple(timestep_for_embed.shape)
            t_flat = self.t_embedder(timestep_for_embed.flatten())
            t6_flat = self.t_block(t_flat)
            t_emb = t_flat.unflatten(0, timestep_shape)
            t6 = t6_flat.unflatten(0, timestep_shape)

        if isinstance(encoder_attention_mask, (list, tuple)):
            encoder_attention_mask = encoder_attention_mask[0]
        y = encoder_hidden_states
        if y.dim() == 3:
            y = y.unsqueeze(1)
        y = self.y_embedder(y).squeeze(1)  # (B, L, D)
        if y.shape[0] != B:
            y = y.expand(B, -1, -1).contiguous()
        if self.y_norm:
            y = self.attention_y_norm(y)
        if encoder_attention_mask is not None and encoder_attention_mask.shape[0] != B:
            encoder_attention_mask = encoder_attention_mask.expand(B, -1).contiguous()

        freqs = self._get_freqs(T, H, W, x.device)

        # Camera conditioning: UCPE prope_fns + Plücker
        prope_fns = None
        if camera_conditions is not None:
            if camera_conditions.shape[1] != T:
                raise ValueError(
                    "SANA-WM camera_conditions must be sampled at latent "
                    f"frames: got {camera_conditions.shape[1]} frames, "
                    f"expected T={T}."
                )
            prope_fns = self._get_ucpe_apply_fns(
                camera_conditions,
                HW=(T, H, W),
                freqs=freqs,
            )

        # Plücker post-attn embedding (shared across all blocks)
        plucker_emb = None
        needs_plucker_emb = (
            chunk_plucker is not None
            and self.plucker_embedder is not None
            and (self.use_chunk_plucker_post_attn or self.use_chunk_plucker_input)
        )
        if needs_plucker_emb:
            plucker_emb = self._get_plucker_emb(
                chunk_plucker,
                latent_token_count=x.shape[1],
            )  # (B, T*H*W, D)

        if self.use_chunk_plucker_input and plucker_emb is not None:
            x = x + plucker_emb

        if not self.use_chunk_plucker_post_attn:
            plucker_emb = None

        # --- 6. Transformer blocks ---
        HW = (T, H, W)
        for block in self.blocks:
            x = block(
                x,
                y=y,
                t=t6,
                HW=HW,
                rotary_emb=freqs,
                prope_fns=prope_fns,
                plucker_emb=plucker_emb,
                mask=encoder_attention_mask,
                chunk_size=chunk_size,
                chunk_split_strategy=chunk_split_strategy,
                chunk_index=chunk_index,
            )

        x = self.final_layer(x, t_emb)  # (B, N, p_t*p_h*p_w*C_out)

        # Un-patch
        x = x.reshape(B, T, H, W, p_t, p_h, p_w, self.out_channels)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.reshape(B, self.out_channels, T * p_t, H * p_h, W * p_w)
        return x

    def forward_long(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        camera_conditions: Optional[torch.Tensor] = None,
        chunk_plucker: Optional[torch.Tensor] = None,
        *,
        kv_cache: Optional[list] = None,
        save_kv_cache: bool = True,
        start_f: Optional[int] = None,
        end_f: Optional[int] = None,
        frame_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, list]:
        """Streaming autoregressive forward over a chunk of latent frames.

        RoPE / camera / plücker are windowed to the chunk's GLOBAL frame range
        ``[start_f, end_f)``; a per-block 10-slot ``kv_cache`` carries recurrent
        state / concat-windows across chunks. Returns ``(out, new_cache)``.
        """
        if encoder_hidden_states is None:
            raise ValueError("SANA-WM forward_long requires encoder_hidden_states.")
        if timestep is None:
            raise ValueError("SANA-WM forward_long requires timestep.")

        if kv_cache is None:
            kv_cache = [[None] * _NUM_STREAM_CACHE_SLOTS for _ in self.blocks]

        B, C, T_raw, H_raw, W_raw = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        T = T_raw // p_t
        H = H_raw // p_h
        W = W_raw // p_w
        start = 0 if start_f is None else int(start_f)
        end = start + T if end_f is None else int(end_f)

        x = self.x_embedder(hidden_states.to(dtype=self.x_embedder.proj.weight.dtype))

        # Timestep AdaLN-single: force the framewise (B, 1, T) path so blocks
        # always apply per-frame modulation.
        if timestep.dim() == 1:
            timestep = timestep[:, None, None].expand(-1, 1, T)
        elif timestep.dim() == 2:
            timestep = timestep[:, None, :]
        if self.timestep_norm_scale_factor != 1.0:
            timestep_for_embed = (
                timestep.float() / self.timestep_norm_scale_factor
            ).to(torch.float32)
        else:
            timestep_for_embed = timestep.long().to(torch.float32)
        timestep_shape = tuple(timestep_for_embed.shape)
        t_flat = self.t_embedder(timestep_for_embed.flatten())
        t6_flat = self.t_block(t_flat)
        t_emb = t_flat.unflatten(0, timestep_shape)
        t6 = t6_flat.unflatten(0, timestep_shape)

        if isinstance(encoder_attention_mask, (list, tuple)):
            encoder_attention_mask = encoder_attention_mask[0]
        y = encoder_hidden_states
        if y.dim() == 3:
            y = y.unsqueeze(1)
        y = self.y_embedder(y).squeeze(1)
        if y.shape[0] != B:
            y = y.expand(B, -1, -1).contiguous()
        if self.y_norm:
            y = self.attention_y_norm(y)
        if encoder_attention_mask is not None and encoder_attention_mask.shape[0] != B:
            encoder_attention_mask = encoder_attention_mask.expand(B, -1).contiguous()

        # RoPE windowed to global frame positions [start, end)
        freqs = self._get_freqs_window(
            start, end, H, W, x.device, frame_index=frame_index
        )

        # Camera conditioning: slice to the chunk, co-windowed w/ freqs
        prope_fns = None
        if camera_conditions is not None:
            if camera_conditions.shape[1] != T:
                # .contiguous(): canonical layout regardless of how the caller
                # built the full-length tensor, so batch and realtime windows are
                # kernel-level identical (slice offset/stride changes the reduction
                # order otherwise — measured 1e-7 seeds amplifying to %-level drift
                # through the bf16 block stack).
                camera_conditions = camera_conditions[:, start:end].contiguous()
            if camera_conditions.shape[0] != B:
                camera_conditions = camera_conditions.repeat(
                    B // camera_conditions.shape[0], 1, 1
                )
            prope_fns = self._get_ucpe_apply_fns(
                camera_conditions, HW=(T, H, W), freqs=freqs
            )

        # Plücker post-attn / input embedding, sliced to the chunk.
        if chunk_plucker is not None and chunk_plucker.shape[2] != T:
            chunk_plucker = chunk_plucker[
                :, :, start:end
            ].contiguous()  # see camera note
        if chunk_plucker is not None and chunk_plucker.shape[0] != B:
            chunk_plucker = chunk_plucker.repeat(
                B // chunk_plucker.shape[0], 1, 1, 1, 1
            )
        plucker_emb = None
        needs_plucker_emb = (
            chunk_plucker is not None
            and self.plucker_embedder is not None
            and (self.use_chunk_plucker_post_attn or self.use_chunk_plucker_input)
        )
        if needs_plucker_emb:
            plucker_emb = self._get_plucker_emb(
                chunk_plucker, latent_token_count=x.shape[1]
            )
        if self.use_chunk_plucker_input and plucker_emb is not None:
            x = x + plucker_emb
        if not self.use_chunk_plucker_post_attn:
            plucker_emb = None

        # parity harness (env-gated, no-op in prod): on the FIRST sink-path call
        # (frame_index not None), checksum the pre-block tensors and x after every
        # block to localize where the two execution paths first diverge.
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm import (
            parity_probe,
        )

        _probe_path = __import__("os").environ.get(parity_probe.ENV_BLOCK_PROBE)
        _probe = None
        if (
            _probe_path
            and frame_index is not None
            and not getattr(self, "_block_probe_done", False)
        ):
            _ck = parity_probe.checksum
            _probe = {
                "x_embed": _ck(x),
                "t6": _ck(t6),
                "y": _ck(y),
                "freqs": (
                    (
                        tuple(freqs.shape),
                        float(freqs.real.detach().double().sum().item()),
                        float(freqs.imag.detach().double().sum().item()),
                    )
                    if freqs is not None
                    else None
                ),
                "plucker_emb": _ck(plucker_emb),
                "frame_index": frame_index.tolist(),
            }

        HW = (T, H, W)
        new_cache = []
        for i, block in enumerate(self.blocks):
            x, block_cache = block.forward_long(
                x,
                y,
                t6,
                HW,
                freqs,
                prope_fns,
                plucker_emb,
                encoder_attention_mask,
                kv_cache=kv_cache[i],
                save_kv_cache=save_kv_cache,
            )
            new_cache.append(block_cache)
            if _probe is not None:
                _probe[f"x_after_block_{i:02d}"] = parity_probe.checksum(x)
        if _probe is not None:
            torch.save(_probe, _probe_path)
            self._block_probe_done = True

        x = self.final_layer(x, t_emb)
        x = x.reshape(B, T, H, W, p_t, p_h, p_w, self.out_channels)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.reshape(B, self.out_channels, T * p_t, H * p_h, W * p_w)
        return x, new_cache


EntryClass = SanaWMTransformer3DModel
