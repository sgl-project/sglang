# SPDX-License-Identifier: Apache-2.0
"""OmniDreams pre-processing + autoregressive denoising stages.

Two model-specific stages implement the OmniDreams (FlashDreams port) inference
contract (decoding uses the standard single-pass ``DecodingStage``):

* :class:`OmniDreamsBeforeDenoisingStage` -- text encoding (Cosmos-Reason1-7B
  ``full_concat`` -> 100352), i2v reference-frame VAE latent + condition mask,
  2-step self-forcing schedule, and all ``DenoisingStage.verify_input`` fields.
  AR geometry/conditioning are stashed in ``batch.extra["omnidreams"]``.

* :class:`OmniDreamsDenoisingStage` -- a full ``forward()`` override (the base
  ``DenoisingStage`` loop is single-pass and stateless) that drives the
  autoregressive rollout: per-chunk ``BlockKVCache`` lifecycle, ``shift_t``
  RoPE, the 2-step self-forcing denoise, and the context-noise re-forward that
  writes the *clean* chunk's K/V into the cache (mirroring FlashDreams
  ``DiffusionModel.finalize`` and SGLang ``CausalDMDDenoisingStage``).

Tensors flow in PATCHIFIED token space ``[B, L, D]`` during the denoise loop
(matching the DiT forward), and each chunk's clean latent is unpatchified to
``[B, C, T, H, W]``. The chunks are concatenated into ``batch.latents`` for the
standard single-pass decode, whose Wan VAE causal feature cache flows across
chunk boundaries (correct continuity + FlashDreams frame counts).

A remaining HD-map VAE-encode numerics note is flagged inline with ``TODO(gpu)``.
"""

from __future__ import annotations

import PIL.Image
import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
)
from sglang.multimodal_gen.runtime.models.dits.omnidreams_rope import (
    RotaryPositionEmbedding3D,
)
from sglang.multimodal_gen.runtime.models.encoders.omnidreams_text import (
    full_concat_embeddings,
)
from sglang.multimodal_gen.runtime.models.vision_utils import (
    load_image,
    normalize,
    numpy_to_pt,
    pil_to_numpy,
    resize,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)

# Cosmos-Reason1 prompt template (FlashDreams cosmos_reason1.py).
_SYSTEM_PROMPT = (
    "You are a helpful assistant who will provide prompts to an image generator."
)
_TEXT_MAX_LENGTH = 512
# Upper bound on autoregressive chunks per request. Bounds the rollout loop
# length (and thus GPU memory/compute) against an unbounded ``num_frames`` from
# the HTTP API. ~256 chunks * len_t(2) * 4 = ~2048 pixel frames.
_MAX_AR_CHUNKS = 256


# --------------------------------------------------------------------------- #
# Pre-processing stage                                                        #
# --------------------------------------------------------------------------- #
class OmniDreamsBeforeDenoisingStage(PipelineStage):
    """Consolidated pre-processing for the OmniDreams pipeline.

    Populates every field ``DenoisingStage.verify_input`` checks plus the AR
    rollout state consumed by :class:`OmniDreamsDenoisingStage`.
    """

    def __init__(
        self,
        transformer,
        scheduler=None,
        text_encoder=None,
        tokenizer=None,
        vae=None,
        config=None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vae = vae
        self.config = config

    # ----- helpers ---------------------------------------------------------- #
    @torch.no_grad()
    def _encode_text(self, prompt: str, device: torch.device) -> torch.Tensor:
        """Cosmos-Reason1-7B text -> ``[1, L, 100352]`` full_concat embedding."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        # Newer transformers return a BatchEncoding (dict-like) rather than a
        # bare tensor; normalize to the input_ids tensor.
        if not isinstance(input_ids, torch.Tensor):
            input_ids = input_ids["input_ids"]
        # Pad/truncate to a fixed context length (FlashDreams uses 512). Build
        # an attention mask so the encoder does not attend to padding tokens
        # (otherwise short prompts get corrupted embeddings).
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or 0
        valid_len = input_ids.shape[1]
        if valid_len < _TEXT_MAX_LENGTH:
            pad = input_ids.new_full(
                (input_ids.shape[0], _TEXT_MAX_LENGTH - valid_len), pad_id
            )
            input_ids = torch.cat([input_ids, pad], dim=1)
            attention_mask = torch.cat(
                [
                    input_ids.new_ones((input_ids.shape[0], valid_len)),
                    input_ids.new_zeros(
                        (input_ids.shape[0], _TEXT_MAX_LENGTH - valid_len)
                    ),
                ],
                dim=1,
            )
        else:
            input_ids = input_ids[:, :_TEXT_MAX_LENGTH]
            attention_mask = input_ids.new_ones(input_ids.shape)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return full_concat_embeddings(out.hidden_states)

    @staticmethod
    def _resolve_input_image(batch: Req):
        """First non-None image-like input on the request, in priority order.

        Accepts a preprocessed tensor, raw pixels, the standard
        ``condition_image``/``vae_image`` fields, or an ``image_path`` to load.
        Returns ``None`` when the request carries no reference image (text-only /
        unconditioned). Uses explicit None-checks -- ``a or b`` chaining raises on
        tensor operands.
        """
        for attr in (
            "preprocessed_image",
            "pixel_values",
            "condition_image",
            "vae_image",
        ):
            val = getattr(batch, attr, None)
            if val is not None:
                return val
        image_path = getattr(batch, "image_path", None)
        if image_path is None:
            return None
        if isinstance(image_path, (list, tuple)):
            return image_path[0] if len(image_path) > 0 else None
        return image_path

    @staticmethod
    def _preprocess_pixels(
        image, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor | None:
        """Pixels (PIL / path / tensor) -> ``[1, 3, 1, H, W]`` in ``[-1, 1]``.

        (The batch dim is synthesized when the input lacks it; the Wan VAE encodes
        a single image at a time.)

        Mirrors ``ImageVAEEncodingStage.preprocess`` (``image_encoding.py``):
        PIL/path -> ``resize(h, w)`` -> ``pil_to_numpy`` -> ``numpy_to_pt`` ->
        ``normalize``. Already-tensor inputs pass through (assumed pre-sized) and
        skip the ``[0,1]->[-1,1]`` normalize when already signed (``min() < 0``),
        matching the standard stage's ``do_normalize`` heuristic. A temporal axis
        is added so the Wan VAE sees a single-frame clip.
        """
        if isinstance(image, (str, bytes)):
            image = load_image(image)
        if isinstance(image, PIL.Image.Image):
            image = resize(image, height, width)
            x = numpy_to_pt(pil_to_numpy(image))  # [1,3,H,W] in [0,1]
            x = normalize(x)  # -> [-1,1]
        elif torch.is_tensor(image):
            x = image
            if x.dim() == 3:  # [3,H,W] -> [1,3,H,W]
                x = x.unsqueeze(0)
            if x.min() >= 0:  # assume [0,1] pixels -> [-1,1]
                x = normalize(x)
        else:
            return None
        x = x.to(device=device, dtype=dtype)
        if x.dim() == 4:  # [B,3,H,W] -> [B,3,1,H,W]
            x = x.unsqueeze(2)
        return x

    @torch.no_grad()
    def _encode_reference_image(
        self,
        batch: Req,
        device: torch.device,
        vae_dtype: torch.dtype,
        height: int,
        width: int,
    ) -> torch.Tensor | None:
        """VAE-encode the i2v reference frame -> normalized latent ``[B,16,1,h,w]``.

        Returns ``None`` (text-only / unconditioned smoke) when no reference
        image is present on the request.
        """
        image = self._resolve_input_image(batch)
        if image is None:
            logger.warning(
                "OmniDreams: no reference image on request; running unconditioned. "
                "Provide condition_image/image_path (or pixel_values) for i2v."
            )
            return None

        x = self._preprocess_pixels(image, height, width, device, vae_dtype)
        if x is None:
            return None
        return self._vae_encode_normalized(x)

    @torch.no_grad()
    def _vae_encode_normalized(self, x: torch.Tensor) -> torch.Tensor:
        """``[B,3,T,H,W]`` pixels -> latent normalized into the DiT space.

        Encode, take the distribution mode (deterministic), then ``(z-mean)/std``.
        Shared by the i2v reference frame and the HD-map conditioning encode.
        """
        latent_dist = self.vae.encode(x)
        latent = (
            latent_dist.mode()
            if hasattr(latent_dist, "mode")
            else latent_dist.sample() if hasattr(latent_dist, "sample") else latent_dist
        )
        mean = torch.tensor(
            self.vae.latents_mean, device=latent.device, dtype=latent.dtype
        ).view(1, -1, 1, 1, 1)
        std = torch.tensor(
            self.vae.latents_std, device=latent.device, dtype=latent.dtype
        ).view(1, -1, 1, 1, 1)
        return (latent - mean) / std

    @torch.no_grad()
    def _encode_hdmap(
        self,
        batch: Req,
        device: torch.device,
        vae_dtype: torch.dtype,
        dit_dtype: torch.dtype,
        num_chunks: int,
        height: int,
        width: int,
    ) -> list[torch.Tensor] | None:
        """Per-chunk HD-map conditioning -> ``list[num_chunks]`` of patchified tokens.

        HD-map is a *per-chunk* driving condition (each latent chunk has its own
        HD-map frames), so this returns a list indexed by chunk -- not one tensor
        shared across chunks. Returns ``None`` when the request carries no HD-map
        input, in which case the AR stage falls back to zeros (HDMap disabled).

        Accepts ``batch.hdmap_path`` / ``batch.hdmap_pixels`` as either a single
        input (broadcast to every chunk) or a per-chunk list. Each entry runs the
        same pixel-preprocess + VAE-encode + patchify path as the reference image.

        TODO(gpu): the HD-map pixel -> 16ch-latent VAE numerics are validated on
        GPU; this encode path only runs when real HD-map input is supplied.
        """
        hdmap = getattr(batch, "hdmap_path", None)
        if hdmap is None:
            hdmap = getattr(batch, "hdmap_pixels", None)
        if hdmap is None:
            return None

        per_chunk = list(hdmap) if isinstance(hdmap, (list, tuple)) else None
        tokens: list[torch.Tensor] = []
        for ci in range(num_chunks):
            if per_chunk is not None:
                src = per_chunk[ci] if ci < len(per_chunk) else per_chunk[-1]
            else:
                src = hdmap
            x = self._preprocess_pixels(src, height, width, device, vae_dtype)
            if x is None:
                logger.warning(
                    "OmniDreams: HD-map chunk %d preprocessed to None; disabling "
                    "HDMap (all chunks fall back to zeros). Check hdmap input.",
                    ci,
                )
                return None
            latent = self._vae_encode_normalized(x).to(dit_dtype)
            # [B,16,t,h,w] -> [B, L, additional_concat_ch*pdim] via the DiT patchify.
            tokens.append(self.transformer.patchify(latent))
        return tokens

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        config = server_args.pipeline_config
        device = get_local_torch_device()
        dit_dtype = PRECISION_TO_TYPE[config.dit_precision]
        vae_dtype = PRECISION_TO_TYPE[config.vae_precision]
        arch = config.dit_config.arch_config

        # --- geometry (patchified token grid) ---
        height = int(batch.height)
        width = int(batch.width)
        sp = config.vae_config.arch_config.scale_factor_spatial  # 8
        latent_h = height // sp
        latent_w = width // sp
        hp = latent_h // arch.patch_spatial
        wp = latent_w // arch.patch_spatial
        len_t = int(getattr(batch, "len_t", getattr(config, "len_t", 2)))
        tokens_per_frame = hp * wp
        chunk_tokens = len_t * tokens_per_frame

        # --- text conditioning (100352) ---
        prompt = batch.prompt if isinstance(batch.prompt, str) else str(batch.prompt)
        text_embeds = self._encode_text(prompt, device).to(dit_dtype)
        batch.prompt_embeds = [text_embeds]
        batch.negative_prompt_embeds = None
        batch.image_embeds = []
        batch.do_classifier_free_guidance = False

        # --- i2v reference latent -> patchified frame-0 token block ---
        image_latent = self._encode_reference_image(
            batch, device, vae_dtype, height, width
        )
        if image_latent is not None:
            image_latent = image_latent.to(dit_dtype)
            # [B,16,1,h,w] -> [B, hp*wp, 16*pdim] via the DiT patchify.
            image_token = self.transformer.patchify(image_latent)
        else:
            image_token = None
        batch.image_latent = None  # AR stage assembles batch.latents itself.

        # --- schedule (2-step self-forcing) ---
        scheduler = self.scheduler
        batch.scheduler = scheduler
        batch.timesteps = scheduler.denoising_step_list.to(device)
        batch.sigmas = scheduler.denoising_sigmas.tolist()
        batch.num_inference_steps = int(scheduler.denoising_step_list.shape[0])
        batch.guidance_scale = 1.0
        batch.eta = 0.0
        if batch.generator is None:
            seed = int(getattr(batch, "seed", None) or 0)
            batch.generator = torch.Generator(device=device).manual_seed(seed)

        # --- AR rollout state for the denoising stage ---
        num_chunks = self._compute_num_chunks(batch, len_t)
        # Per-chunk HD-map tokens (None -> AR stage uses zeros / HDMap disabled).
        hdmap_tokens = self._encode_hdmap(
            batch, device, vae_dtype, dit_dtype, num_chunks, height, width
        )
        batch.extra["omnidreams"] = {
            "hp": hp,
            "wp": wp,
            "len_t": len_t,
            "tokens_per_frame": tokens_per_frame,
            "chunk_tokens": chunk_tokens,
            "latent_h": latent_h,
            "latent_w": latent_w,
            "num_chunks": num_chunks,
            "window_size_t": int(getattr(batch, "window_size_t", 6)),
            "sink_size_t": int(getattr(batch, "sink_size_t", 0)),
            "context_noise": float(getattr(batch, "context_noise", 128)),
            "image_token": image_token,  # [B, hp*wp, in*pdim] or None
            # Per-chunk HD-map tokens: None, or list[num_chunks] of
            # [B, chunk_tokens, additional_concat_ch*pdim].
            "hdmap_tokens": hdmap_tokens,
        }
        # raw_latent_shape lets SDPA-path attn metadata stay a no-op.
        batch.raw_latent_shape = (
            text_embeds.shape[0],
            chunk_tokens,
            arch.out_channels,
            latent_h,
            latent_w,
        )
        return batch

    @staticmethod
    def _compute_num_chunks(batch: Req, len_t: int) -> int:
        """Latent chunks needed for the requested pixel-frame count.

        FlashDreams decode: chunk 0 -> ``1 + (len_t-1)*4`` frames, chunk>=1 ->
        ``len_t*4`` frames (temporal compression 4).
        """
        num_frames = int(getattr(batch, "num_frames", None) or 0)
        tc = 4
        if num_frames <= 0:
            n = max(1, int(getattr(batch, "num_chunks", 1)))
        else:
            first = 1 + (len_t - 1) * tc
            if num_frames <= first:
                n = 1
            else:
                n = 1 + -(-(num_frames - first) // (len_t * tc))  # ceil division
        if n > _MAX_AR_CHUNKS:
            logger.warning(
                "OmniDreams: requested %d AR chunks exceeds the cap %d; clamping "
                "(num_frames=%d). Raise _MAX_AR_CHUNKS if longer rollouts are needed.",
                n,
                _MAX_AR_CHUNKS,
                num_frames,
            )
            n = _MAX_AR_CHUNKS
        return n

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, V.string_or_list_strings)
        return result


# --------------------------------------------------------------------------- #
# Autoregressive denoising stage                                              #
# --------------------------------------------------------------------------- #
class OmniDreamsDenoisingStage(DenoisingStage):
    """Autoregressive rollout (full ``forward()`` override).

    The base ``DenoisingStage`` runs a single diffusers-style ``scheduler.step``
    loop with no cross-chunk state. OmniDreams instead rolls over latent chunks,
    each chunk: roll the per-block KV window, run a 2-step self-forcing denoise,
    then re-forward the *clean* chunk at the context-noise timestep to write the
    authoritative (clean) K/V into the cache.
    """

    def __init__(self, transformer, scheduler, vae=None) -> None:
        super().__init__(transformer, scheduler, vae=vae)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # Phase 6 guard: TP is supported via column/row parallel layers in the
        # DiT, but SP (ulysses/ring) is not yet supported for the AR chunk loop.
        # Guarded with try/except since SP is not initialized in CPU tests.
        try:
            from sglang.multimodal_gen.runtime.distributed.parallel_state import (
                get_sp_world_size,
            )

            assert get_sp_world_size() <= 1, (
                "Sequence parallelism (SP) is not yet supported for OmniDreams. "
                "Run with --ulysses-degree 1 --ring-degree 1."
            )
        except AssertionError as e:
            if "not initialized" in str(e):
                pass  # SP not booted — single-GPU (or CPU test), safe.
            else:
                raise

        # The downstream single-pass decode of the concatenated AR latents
        # relies on the Wan VAE's causal temporal feature cache flowing across
        # frames within one decode() call. A VAE without feature caching would
        # change the frame-count semantics, so fail loudly rather than silently.
        if self.vae is not None and not getattr(self.vae, "use_feature_cache", True):
            raise RuntimeError(
                "OmniDreams AR rollout requires a Wan VAE with "
                "use_feature_cache=True for correct streaming frame counts."
            )

        config = server_args.pipeline_config
        device = get_local_torch_device()
        dit_dtype = PRECISION_TO_TYPE[config.dit_precision]
        arch = config.dit_config.arch_config
        st = batch.extra["omnidreams"]

        hp, wp, len_t = st["hp"], st["wp"], st["len_t"]
        tokens_per_frame = st["tokens_per_frame"]
        chunk_tokens = st["chunk_tokens"]
        num_chunks = st["num_chunks"]
        context_noise = st["context_noise"]
        head_dim = arch.model_channels // arch.num_heads
        in_d = arch.in_channels * arch.patch_temporal * arch.patch_spatial**2
        hdmap_d = (
            arch.additional_concat_ch * arch.patch_temporal * arch.patch_spatial**2
        )
        mask_d = arch.patch_temporal * arch.patch_spatial**2

        scheduler = batch.scheduler if batch.scheduler is not None else self.scheduler
        scheduler = scheduler.to(device)
        text = batch.prompt_embeds[0].to(device=device, dtype=dit_dtype)
        B = text.shape[0]
        gen = batch.generator
        # ``.normal_(generator=)`` requires the generator to live on the noise
        # tensor's device; re-seed onto ``device`` if the caller passed a CPU one.
        if gen is not None and gen.device != device:
            gen = torch.Generator(device=device).manual_seed(gen.initial_seed())

        # 3D RoPE for one chunk (NeoX 44:42:42; h/w extrapolate 3.0, t 1.0).
        rope = RotaryPositionEmbedding3D(
            head_dim=head_dim,
            len_h=hp,
            len_w=wp,
            len_t=len_t,
            h_extrapolation_ratio=3.0,
            w_extrapolation_ratio=3.0,
            t_extrapolation_ratio=1.0,
            device=device,
        )

        # One BlockKVCache per transformer block (token counts = frames * hp*wp).
        caches = self.transformer.init_kv_caches(
            batch_size=B,
            chunk_tokens=chunk_tokens,
            window_tokens=st["window_size_t"] * tokens_per_frame,
            sink_tokens=st["sink_size_t"] * tokens_per_frame,
            device=device,
            dtype=dit_dtype,
        )

        # Frame-0 conditioning (i2v): channel mask (into the DiT) + inject mask
        # (pins the clean reference latent) + the reference token block.
        frame0 = tokens_per_frame
        image_token = st["image_token"]
        if image_token is not None:
            image_full = torch.zeros(
                B, chunk_tokens, in_d, device=device, dtype=dit_dtype
            )
            image_full[:, :frame0, :] = image_token.to(device=device, dtype=dit_dtype)
            inject_mask = torch.zeros(
                B, chunk_tokens, 1, device=device, dtype=dit_dtype
            )
            inject_mask[:, :frame0, :] = 1.0
            cond_mask_c0 = torch.zeros(
                B, chunk_tokens, mask_d, device=device, dtype=dit_dtype
            )
            cond_mask_c0[:, :frame0, :] = 1.0
        else:
            image_full = inject_mask = None
            cond_mask_c0 = torch.zeros(
                B, chunk_tokens, mask_d, device=device, dtype=dit_dtype
            )
        cond_mask_zero = torch.zeros(
            B, chunk_tokens, mask_d, device=device, dtype=dit_dtype
        )
        hdmap_zero = torch.zeros(
            B, chunk_tokens, hdmap_d, device=device, dtype=dit_dtype
        )

        # Phase 6: precompute cross-attn K/V once per prompt (text context is
        # static; avoids redundant k_proj/v_proj in every forward call).
        # Feed the projected context (crossattn_proj applied) so the cached K/V
        # match the per-block input dimensions.
        cross_attn_kv = self.transformer.precompute_cross_attn_kv(
            self.transformer.crossattn_proj(text)
        )

        # Phase 5: compute view_indices for cross-view attention (optional).
        # Default: single-view (V=1). Multi-view is gated by
        # arch.enable_cross_view_attn and num_views on the request.
        view_count = int(getattr(batch, "num_views", 1) or 1)
        # Bound num_views to the camera-embedding table to prevent an
        # out-of-range index (and reject nonsensical/abusive values).
        n_cameras = int(getattr(arch, "n_cameras_emb", 1))
        if view_count < 1 or view_count > n_cameras:
            raise ValueError(
                f"num_views={view_count} out of range [1, {n_cameras}] "
                "(n_cameras_emb)."
            )
        view_indices: torch.Tensor | None = None
        if view_count > 1 and self.transformer.adaln_view_embedder is not None:
            view_indices = (
                torch.arange(view_count, device=device, dtype=torch.long)
                .unsqueeze(0)
                .expand(B, -1)
            )  # [B, V]

        # Loop-invariant context-noise timestep tensor (same scalar every chunk).
        ctx_noise_t = torch.tensor(context_noise, device=device, dtype=dit_dtype)

        latent_chunks: list[torch.Tensor] = []
        for chunk_idx in range(num_chunks):
            rope_freqs = rope.shift_t(chunk_idx)
            is_first = chunk_idx == 0
            cond_mask = cond_mask_c0 if is_first else cond_mask_zero
            # HD-map is per-chunk: index this chunk's tokens (None -> zeros, i.e.
            # HDMap disabled). Explicit None-check -- a tensor in ``or`` raises.
            if st["hdmap_tokens"] is None:
                hdmap_chunk = hdmap_zero
            else:
                hdmap_chunk = st["hdmap_tokens"][chunk_idx].to(
                    device=device, dtype=dit_dtype
                )
            pin = is_first and image_full is not None

            def predict_flow(noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                if pin:
                    noisy = noisy * (1.0 - inject_mask) + image_full * inject_mask
                return self.transformer(
                    hidden_states=noisy,
                    encoder_hidden_states=text,
                    timestep=t,
                    condition_video_input_mask=cond_mask,
                    rope_freqs=rope_freqs,
                    hdmap_condition=hdmap_chunk,
                    kv_caches=caches,
                    cross_attn_kv=cross_attn_kv,
                    view_indices=view_indices,
                )

            for c in caches:
                c.before_update(chunk_idx)

            noise = torch.empty(
                B, chunk_tokens, in_d, device=device, dtype=dit_dtype
            ).normal_(generator=gen)
            clean = scheduler.sample(noise, predict_flow=predict_flow, rng=gen)
            if pin:
                clean = clean * (1.0 - inject_mask) + image_full * inject_mask

            # Authoritative cache write: re-forward the CLEAN chunk at the
            # context-noise timestep so the cache holds in-distribution K/V.
            ctx_latent = scheduler.add_noise(
                clean,
                ctx_noise_t,
                rng=gen,
            )
            self.transformer(
                hidden_states=ctx_latent,
                encoder_hidden_states=text,
                timestep=ctx_noise_t,
                condition_video_input_mask=cond_mask,
                rope_freqs=rope_freqs,
                hdmap_condition=hdmap_chunk,
                kv_caches=caches,
                cross_attn_kv=cross_attn_kv,
                view_indices=view_indices,
            )

            for c in caches:
                c.after_update(chunk_idx)

            # [B, L, out*pdim] -> [B, out, len_t, h, w].
            latent_chunks.append(self.transformer.unpatchify(clean, len_t, hp, wp))

        # Concatenate the AR chunks into the full latent sequence. The standard
        # DecodingStage decodes this in a single pass; the Wan VAE's causal
        # temporal feature cache flows across chunk boundaries, yielding correct
        # continuity and FlashDreams frame counts.
        batch.latents = torch.cat(latent_chunks, dim=2)

        # Phase 6: SP post-process — latents may need gathering when SP is
        # eventually supported. Currently a no-op (SP is guarded at entry).
        batch.latents = self._postprocess_sp_latents(batch, server_args)

        return batch

    def _postprocess_sp_latents(
        self, batch: Req, server_args: ServerArgs
    ) -> torch.Tensor:
        """Gather sharded latents when SP is active (future, currently no-op).

        When SP is enabled, each rank outputs partial-sequence latents that
        must be all-gathered along the time dimension. This is a placeholder
        that returns the latents as-is when SP is not active.
        """
        try:
            from sglang.multimodal_gen.runtime.distributed import (
                get_sp_world_size,
                sequence_model_parallel_all_gather,
            )

            if get_sp_world_size() > 1 and getattr(
                batch, "did_sp_shard_latents", False
            ):
                return sequence_model_parallel_all_gather(batch.latents, dim=2)
        except (ImportError, AssertionError):
            pass
        return batch.latents

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.min_dims(1)])
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check(
            "omnidreams_extra",
            batch.extra.get("omnidreams"),
            lambda x: isinstance(x, dict),
        )
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result
