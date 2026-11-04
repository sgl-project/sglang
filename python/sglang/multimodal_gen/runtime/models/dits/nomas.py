# SPDX-License-Identifier: Apache-2.0
# Adapted from: https://github.com/Tencent-Hunyuan/Hy-Nomas
"""Nomas DiT with explicit segment KV state, tensor and Ulysses sequence parallel.

Consumes precomputed text features and normalized camera extrinsics. Under TP
every linear in the double-stream blocks is head-sharded; under Ulysses SP the
video tokens are sharded and exchanged for heads around attention while the
text stream is replicated, as in the upstream implementation. The KV cache on
each rank holds its local heads only.
"""

import torch
from diffusers.models.embeddings import PixArtAlphaTextProjection
from msgspec import Struct
from torch import nn
from torch.nn import functional as F

from sglang.multimodal_gen.configs.models.dits.nomas import (
    NomasArchConfig,
    NomasConfig,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.layers.visual_embedding import ModulateProjection
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.models.dits.nomas_kv_cache import (
    NomasAttentionPlan,
    NomasCache,
)
from sglang.multimodal_gen.runtime.models.dits.nomas_layers import (
    FinalLayerPerBlock,
    IndividualTokenRefiner,
    ParallelInfo,
    PatchEmbed,
    RMSNorm,
    TimestepEmbedder,
    current_parallel_info,
    gated_residual,
    get_nd_rotary_pos_embed,
    local_token_range,
    modulate_for_linear,
    rms_norm_rotate,
    sp_all_gather,
    ulysses_all_to_all,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)


def _camera_rope_tables(device=None):
    return get_nd_rotary_pos_embed(
        [8, 28, 28], (0, 0, 0), (7500, 1, 1), theta=256, device=device
    )


class NomasSegmentContext(Struct, frozen=True):
    """Request-local metadata shared by the Euler steps of one segment."""

    attention: NomasAttentionPlan
    rope: tuple[torch.Tensor, torch.Tensor]
    shape: tuple[int, int, int]
    history_length: int
    frame_offset: int


class NomasTextRefiner(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        self.input_embedder = nn.Linear(config.text_states_dim, d)
        self.t_embedder = TimestepEmbedder(d)
        self.c_embedder = PixArtAlphaTextProjection(
            config.text_states_dim, d, act_fn="silu"
        )
        self.individual_token_refiner = IndividualTokenRefiner(
            d, config.num_attention_heads, depth=2
        )

    def forward(self, x, t, mask):
        weights = mask.to(x.dtype).unsqueeze(-1)
        pooled = (x * weights).sum(1) / weights.sum(1)
        c = self.t_embedder(t) + self.c_embedder(pooled)
        return self.individual_token_refiner(self.input_embedder(x), c, mask)


class NomasDoubleStreamBlock(nn.Module):
    def __init__(self, config, info: ParallelInfo):
        super().__init__()
        d, heads = config.hidden_size, config.num_attention_heads
        self.info = info
        # Heads after TP; Ulysses splits these again around attention.
        self.heads = heads // info.tp_size
        self.attn_heads = heads // info.head_shards
        for stream in ("img", "txt"):
            setattr(
                self,
                f"{stream}_mod",
                ModulateProjection(d, factor=6, act_layer="silu"),
            )
            for i in (1, 2):
                setattr(
                    self,
                    f"{stream}_norm{i}",
                    nn.LayerNorm(d, eps=1e-6, elementwise_affine=False),
                )
            for name in ("q", "k", "v"):
                setattr(
                    self,
                    f"{stream}_attn_{name}",
                    ColumnParallelLinear(
                        d,
                        d,
                        bias=True,
                        gather_output=False,
                        prefix=f"{stream}_attn_{name}",
                    ),
                )
            setattr(
                self,
                f"{stream}_attn_proj",
                RowParallelLinear(
                    d,
                    d,
                    bias=True,
                    input_is_parallel=True,
                    prefix=f"{stream}_attn_proj",
                ),
            )
            for name in ("q", "k"):
                setattr(self, f"{stream}_attn_{name}_norm", RMSNorm(d // heads))
            setattr(
                self,
                f"{stream}_mlp",
                MLP(d, 4 * d, act_type="gelu_pytorch_tanh", prefix=f"{stream}_mlp"),
            )

    def forward(
        self, img, txt, vec, rope, layer_cache, valid_length, attention, collect_cache
    ):
        qkv, mods = [], []
        for stream, x in (("img", img), ("txt", txt)):
            mod = getattr(self, f"{stream}_mod")(vec).chunk(6, -1)
            mods.append(mod)
            x = modulate_for_linear(
                getattr(self, f"{stream}_norm1")(x), shift=mod[0], scale=mod[1]
            )
            q, k, v = [
                getattr(self, f"{stream}_attn_{name}")(x)[0].unflatten(
                    -1, (self.heads, -1)
                )
                for name in ("q", "k", "v")
            ]
            q_norm = getattr(self, f"{stream}_attn_q_norm")
            k_norm = getattr(self, f"{stream}_attn_k_norm")
            if stream == "img":
                # QK norm, cast and RoPE fused; values are unchanged.
                q = rms_norm_rotate(q_norm, q, rope, v.dtype)
                k = rms_norm_rotate(k_norm, k, rope, v.dtype)
                # Local tokens, all heads -> all tokens, local heads.
                q, k, v = (ulysses_all_to_all(x, self.info, 2, 1) for x in (q, k, v))
            else:
                q, k = q_norm(q).to(v), k_norm(k).to(v)
                # Text is replicated; keep this rank's head slice.
                lo = self.info.sp_rank * self.attn_heads
                q, k, v = (x[:, :, lo : lo + self.attn_heads] for x in (q, k, v))
            qkv.append((q, k, v))
        q, k, v = [torch.cat((qkv[0][i], qkv[1][i]), 1) for i in range(3)]
        entry = {"k": k, "v": v} if collect_cache else None
        video_tokens = q.shape[1] - txt.shape[1]
        img_attn, txt_attn = attention(q, k, v, layer_cache, valid_length).split(
            (video_tokens, txt.shape[1]), 1
        )
        img_attn = ulysses_all_to_all(img_attn, self.info, 1, 2)
        txt_attn = sp_all_gather(txt_attn, self.info, 2)
        outputs = []
        for stream, x, attn, mod in zip(
            ("img", "txt"), (img, txt), (img_attn, txt_attn), mods
        ):
            x = gated_residual(
                x, getattr(self, f"{stream}_attn_proj")(attn.flatten(2))[0], mod[2]
            )
            x = gated_residual(
                x,
                getattr(self, f"{stream}_mlp")(
                    modulate_for_linear(
                        getattr(self, f"{stream}_norm2")(x), shift=mod[3], scale=mod[4]
                    )
                ),
                mod[5],
            )
            outputs.append(x)
        return *outputs, entry


class NomasCameraAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size = config.camera_chunk_size
        self.keyboard_embed = nn.Sequential(
            nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128)
        )
        self.key_attn_q_norm, self.key_attn_k_norm = RMSNorm(64), RMSNorm(64)
        self.img_attn_q = nn.Linear(config.hidden_size, 1024, bias=False)
        self.keyboard_attn_kv = nn.Linear(128, 2048, bias=False)
        self.proj_keyboard = nn.Linear(1024, config.hidden_size, bias=False)
        cos, sin = _camera_rope_tables()
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(self, x, camera, spatial_tokens, has_camera, token_offset=0):
        """``x`` holds video tokens [token_offset, token_offset + n) of the segment."""
        b, n, _ = x.shape
        frames = camera.shape[1]
        q = self.img_attn_q(x).view(b, n, 16, 64)
        k, v = (
            self.keyboard_attn_kv(self.keyboard_embed(camera))
            .view(b, frames, 2, 16, 64)
            .unbind(2)
        )
        tokens = torch.arange(n, device=x.device) + token_offset
        qids = (tokens // spatial_tokens).clamp(max=7499)
        kids = torch.arange(frames, device=x.device).clamp(max=7499)
        q = rms_norm_rotate(
            self.key_attn_q_norm,
            q,
            (self.freqs_cos[qids], self.freqs_sin[qids]),
            v.dtype,
        )
        k = rms_norm_rotate(
            self.key_attn_k_norm,
            k,
            (self.freqs_cos[kids], self.freqs_sin[kids]),
            v.dtype,
        )
        chunks = []
        # Each camera chunk covers a contiguous token range; intersect with ours.
        for start in range(0, frames, self.chunk_size):
            end = min(start + self.chunk_size, frames)
            lo = max(start * spatial_tokens, token_offset) - token_offset
            hi = min(end * spatial_tokens, token_offset + n) - token_offset
            if hi <= lo:
                continue
            out = F.scaled_dot_product_attention(
                q[:, lo:hi].transpose(1, 2),
                k[:, start:end].transpose(1, 2),
                v[:, start:end].transpose(1, 2),
            )
            chunks.append(out.transpose(1, 2).flatten(2))
        out = self.proj_keyboard(torch.cat(chunks, 1))
        if has_camera is not None:
            # Upstream intentionally promotes this residual to float32.
            out = out * has_camera.float()[:, None, None]
        return x + out


class NomasTransformer3DModel(BaseDiT, LayerwiseOffloadableModuleMixin):
    param_names_mapping = NomasArchConfig().param_names_mapping
    reverse_param_names_mapping = {}
    # Segment attention (FA3 / Flex / SDPA reference) is model-specific, not a
    # general SGL attention backend.
    _supported_attention_backends = {AttentionBackendEnum.TORCH_SDPA}

    def __init__(
        self,
        config: NomasConfig,
        hf_config=None,
        *,
        quant_config=None,
        attention_backend="auto",
    ):
        if quant_config is not None:
            raise ValueError("Nomas does not support quantized weights yet")
        if config.quant_config is not None:
            raise NotImplementedError("Nomas v1 does not implement quantized loading")
        if attention_backend == "auto":
            # FA3 + log-sum-exp merge is about 2x FlexAttention on Hopper.
            attention_backend = "fa3" if current_platform.is_hopper() else "flex"
        if attention_backend not in ("fa3", "flex", "sdpa"):
            raise ValueError("Nomas attention backend must be auto, fa3, flex or sdpa")
        super().__init__(config, hf_config or {})
        c = self.config
        c.__post_init__()
        self.parallel = current_parallel_info()
        if c.num_attention_heads % self.parallel.head_shards:
            raise ValueError(
                f"{c.num_attention_heads} attention heads are not divisible by "
                f"tp_size * sequence_parallel_degree = {self.parallel.head_shards}"
            )
        self.hidden_size = c.hidden_size
        self.num_attention_heads = c.num_attention_heads
        self.local_attention_heads = c.num_attention_heads // self.parallel.head_shards
        self.num_channels_latents = c.in_channels
        self.attention_backend = attention_backend
        # Framework layerwise offload hooks this ModuleList; action_modules stay resident.
        self.layer_names = ["double_blocks"]
        self.img_in = PatchEmbed(list(c.patch_size), c.in_channels, c.hidden_size)
        self.txt_in = NomasTextRefiner(c)
        self.time_in = TimestepEmbedder(c.hidden_size)
        self.double_blocks = nn.ModuleList(
            [NomasDoubleStreamBlock(c, self.parallel) for _ in range(c.num_layers)]
        )
        self.final_layer = FinalLayerPerBlock(
            c.hidden_size, list(c.patch_size), c.out_channels
        )
        self.action_modules = nn.ModuleList(
            [NomasCameraAttention(c) for _ in range(c.num_layers)]
        )
        self.extrinsics_embedding_camctrl = nn.Linear(64, 64)
        self.c2ws_hidden_states_layer1 = nn.Linear(64, 64)
        self.c2ws_hidden_states_layer2 = nn.Linear(64, 64)

    def post_load_weights(self):
        # Meta initialization / to_empty() does not restore nonpersistent buffers
        # from a checkpoint. Build these on CPU, matching the source model.
        cos, sin = _camera_rope_tables(torch.device("cpu"))
        for module in self.action_modules:
            p = module.img_attn_q.weight
            module.freqs_cos = cos.to(device=p.device, dtype=p.dtype)
            module.freqs_sin = sin.to(device=p.device, dtype=p.dtype)

    def allocate_cache(self, capacity):
        p = next(self.parameters())
        return NomasCache.allocate(
            self.config,
            capacity,
            device=p.device,
            dtype=p.dtype,
            num_heads=self.local_attention_heads,
        )

    def embed_camera(self, norm_extrinsics):
        """Already-normalized [1, F, 4, 4] extrinsics; pad the front to 4 frames."""
        if (
            norm_extrinsics.ndim != 4
            or norm_extrinsics.shape[0] != 1
            or norm_extrinsics.shape[2:] != (4, 4)
            or norm_extrinsics.shape[1] == 0
        ):
            raise ValueError("normalized extrinsics must have shape [1, F>0, 4, 4]")
        p = self.extrinsics_embedding_camctrl.weight
        x = norm_extrinsics.to(device=p.device, dtype=p.dtype)
        pad = (-x.shape[1]) % 4
        if pad:
            x = torch.cat((x[:, :1].expand(-1, pad, -1, -1), x), 1)
        x = self.extrinsics_embedding_camctrl(x.reshape(1, -1, 64))
        return x + self.c2ws_hidden_states_layer2(
            F.silu(self.c2ws_hidden_states_layer1(x))
        )

    def prepare_segment_context(
        self,
        latent,
        text_mask,
        cache,
        *,
        text_offsets=None,
        video_windows=None,
        hide_cached_action_tails=False,
        rope=None,
    ):
        """Build timestep-independent metadata; rebuild attention after eviction.

        The caller owns this context for one generate_segment call. RoPE can
        survive eviction because the absolute frame offset does not change.
        """
        t, h, w = latent.shape[2:]
        c = self.config
        attention = NomasAttentionPlan(
            text_mask,
            cache.history_mask(hide_cached_action_tails),
            t,
            h * w,
            text_offsets,
            video_windows,
            self.attention_backend,
        )
        if rope is None:
            offset = cache.frame_offset
            # Upstream builds offset tables on the CPU; keep its construction.
            rope = get_nd_rotary_pos_embed(
                c.rope_dim_list,
                (offset, 0, 0),
                (offset + t, h, w),
                c.rope_theta,
                device=latent.device if offset == 0 else None,
            )
            rope = tuple(x.to(latent.device) for x in rope)
            lo, hi = local_token_range(t * h * w, self.parallel)
            rope = tuple(x[lo:hi] for x in rope)
        return NomasSegmentContext(
            attention, rope, (t, h, w), cache.valid_length, cache.frame_offset
        )

    def predict(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        text_mask,
        *,
        cache,
        camera_embedding=None,
        camera_has_mask=None,
        text_offsets=None,
        video_windows=None,
        hide_cached_action_tails=False,
        collect_cache=False,
        _context: NomasSegmentContext | None = None,
    ):
        """Read-only prediction; optional new K/V are returned, never committed here."""
        if self.training or torch.is_grad_enabled():
            raise RuntimeError(
                "Nomas cache inference requires eval() and inference_mode()"
            )
        c = self.config
        self._check_inputs(
            hidden_states, timestep, encoder_hidden_states, text_mask, cache
        )
        _, _, t, h, w = hidden_states.shape
        self._check_camera(camera_embedding, camera_has_mask, t, hidden_states.dtype)
        if _context is None:
            _context = self.prepare_segment_context(
                hidden_states,
                text_mask,
                cache,
                text_offsets=text_offsets,
                video_windows=video_windows,
                hide_cached_action_tails=hide_cached_action_tails,
            )
        if (
            _context.shape != (t, h, w)
            or _context.history_length != cache.valid_length
            or _context.frame_offset != cache.frame_offset
        ):
            raise ValueError("stale Nomas segment context; rebuild after cache changes")
        attention, rope = _context.attention, _context.rope
        # Sequence parallel: this rank owns video tokens [lo, hi); text is replicated.
        lo, hi = local_token_range(t * h * w, self.parallel)
        img = self.img_in(hidden_states)[:, lo:hi]
        txt = self.txt_in(encoder_hidden_states, timestep, text_mask)
        vec = self.time_in(timestep)
        entries = []
        for index, block in enumerate(self.double_blocks):
            img, txt, entry = block(
                img,
                txt,
                vec,
                rope,
                cache.layers[index],
                cache.valid_length,
                attention,
                collect_cache,
            )
            if collect_cache:
                entries.append(entry)
            if camera_embedding is not None:
                img = self.action_modules[index](
                    img,
                    camera_embedding,
                    h * w,
                    camera_has_mask,
                    token_offset=lo,
                )
        img = self.final_layer(img, vec)
        img = sp_all_gather(img, self.parallel, 1)
        return img.view(1, t, h, w, c.out_channels).permute(
            0, 4, 1, 2, 3
        ).contiguous(), entries

    def _check_inputs(
        self, hidden_states, timestep, encoder_hidden_states, text_mask, cache
    ):
        c = self.config
        dtype = next(self.parameters()).dtype
        if (
            hidden_states.ndim != 5
            or hidden_states.shape[:2] != (1, c.in_channels)
            or min(hidden_states.shape[2:]) <= 0
        ):
            raise ValueError("latents must have shape [1, in_channels, T>0, H>0, W>0]")
        if text_mask.ndim != 2 or text_mask.shape[0] != 1:
            raise ValueError("text mask must have shape [1, S]")
        if encoder_hidden_states.shape != (1, text_mask.shape[-1], c.text_states_dim):
            raise ValueError("text feature shape does not match mask/model")
        if timestep.shape != (1,):
            raise ValueError("timestep must have shape [1]")
        if (
            len(cache.layers) != c.num_layers
            or cache.visible.device != hidden_states.device
        ):
            raise ValueError("cache layer count or device mismatch")
        for tensor in (hidden_states, encoder_hidden_states):
            if tensor.device != self.device or tensor.dtype != dtype:
                raise ValueError(
                    "latent/text dtype and device must match model weights"
                )
        if text_mask.device != self.device or timestep.device != self.device:
            raise ValueError("mask and timestep device must match model weights")

    def _check_camera(self, camera_embedding, camera_has_mask, frames, dtype):
        if camera_embedding is not None:
            if camera_embedding.shape != (1, frames, 64):
                raise ValueError("camera/latent alignment mismatch")
            if (
                camera_embedding.device != self.device
                or camera_embedding.dtype != dtype
            ):
                raise ValueError("camera embedding dtype/device mismatch")
        if camera_has_mask is not None and (
            camera_has_mask.shape != (1,) or camera_has_mask.device != self.device
        ):
            raise ValueError("camera mask must have shape [1] on model device")

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        *,
        text_mask,
        cache,
        **kwargs,
    ):
        return self.predict(
            hidden_states,
            timestep,
            encoder_hidden_states,
            text_mask,
            cache=cache,
            **kwargs,
        )[0]


EntryClass = NomasTransformer3DModel
