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

from collections import OrderedDict

import PIL.Image
import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.dits.omnidreams_rope import (
    RotaryPositionEmbedding3D,
)
from sglang.multimodal_gen.runtime.models.encoders.omnidreams_text import (
    full_concat_embeddings,
)
from sglang.multimodal_gen.runtime.models.vision_utils import (
    load_image,
    load_video,
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
# LRU cache for text embeddings (key = prompt string, value = [1, L, 100352] on CPU).
# Avoids re-running the 14 GB Cosmos-Reason1-7B for repeated prompts in serving.
_TEXT_EMBED_CACHE_MAX_SIZE = 32
# HD-map inputs ending in one of these are decoded as a per-frame raster video;
# any other single string is treated as one image (degenerate broadcast).
_HDMAP_VIDEO_EXTS = (".mp4", ".gif", ".webm", ".mov", ".mkv", ".avi")

# --------------------------------------------------------------------------- #
# Diagnostics helpers                                                         #
# --------------------------------------------------------------------------- #
_OMNIDREAMS_DIAG_ENVS = ("SGLANG_OMNIDREAMS_DIAGNOSTICS",)


def _omnidreams_diag_enabled() -> bool:
    import os as _os

    return any(
        _os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
        for name in _OMNIDREAMS_DIAG_ENVS
    )


def _log_omnidreams_stats(label: str, tensor: torch.Tensor) -> None:
    """Log per-frame tensor statistics when ``SGLANG_OMNIDREAMS_DIAGNOSTICS``
    is set (default off). Stats include global + per-frame (ndim==5) mean/std.
    """
    if not _omnidreams_diag_enabled():
        return
    t = tensor.float()
    lines = [
        f"[omnidreams diag] {label}: shape={tuple(t.shape)} "
        f"mean={t.mean():.4f} std={t.std():.4f} min={t.min():.3f} max={t.max():.3f}"
    ]
    if t.ndim == 5:
        for f_idx in range(t.shape[2]):
            ff = t[:, :, f_idx]
            lines.append(f"  frame {f_idx:2d}: mean={ff.mean():.4f} std={ff.std():.4f}")
    logger.info("\n".join(lines))


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
        # Per-instance LRU cache: prompt string -> text embedding on CPU.
        self._text_embed_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        """Declare the text encoder + VAE so the residency manager can stage
        them on the GPU only around their use-sites (and offload them again
        afterwards) when ``--text-encoder-cpu-offload`` / ``--vae-cpu-offload``
        are set. The DiT is only used here via the weightless ``patchify``
        rearrange, so it is not declared (its weights stay wherever loaded).
        """
        stage_name = self._component_stage_name(stage_name)
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        return [
            ComponentUse(stage_name=stage_name, component_name="text_encoder"),
            ComponentUse(
                stage_name=stage_name,
                component_name="vae",
                target_dtype=vae_dtype,
            ),
        ]

    # ----- helpers ---------------------------------------------------------- #
    @torch.no_grad()
    def _encode_text(self, prompt: str, device: torch.device) -> torch.Tensor:
        """Cosmos-Reason1-7B text -> ``[1, L, 100352]`` full_concat embedding.

        CRITICAL: no ``attention_mask`` is passed to the encoder. The checkpoint
        was trained with FlashDreams' ``CosmosReason1TextEncoder``, which runs the
        LM on the full padded sequence WITHOUT a mask. The DiT cross-attends over
        all ``_TEXT_MAX_LENGTH`` token embeddings (valid + padding), so the padding
        embeddings are part of the trained conditioning distribution. Passing a
        mask changes the padding-token hidden states drastically (abs diff up to
        ~99 vs the no-mask path), pushing the conditioning out of distribution and
        producing washed-out / blurry rollouts. Message format and ``add_vision_id``
        mirror FlashDreams exactly so the token sequence is identical.
        """
        # LRU cache: skip the 14 GB encoder for repeated prompts (serving).
        cached = self._text_embed_cache.get(prompt)
        if cached is not None:
            # Move to front (LRU hit); return pinned tensor on device.
            self._text_embed_cache.move_to_end(prompt)
            return cached.to(device=device, non_blocking=True)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            add_vision_id=False,
            return_tensors="pt",
        )
        # Newer transformers return a BatchEncoding (dict-like) rather than a
        # bare tensor; normalize to the input_ids tensor.
        if not isinstance(input_ids, torch.Tensor):
            input_ids = input_ids["input_ids"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        # Pad/truncate to a fixed context length (FlashDreams uses 512). NO
        # attention mask -- see the docstring: the model is trained on the
        # unmasked padded sequence.
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or 0
        valid_len = input_ids.shape[1]
        if valid_len < _TEXT_MAX_LENGTH:
            pad = input_ids.new_full(
                (input_ids.shape[0], _TEXT_MAX_LENGTH - valid_len), pad_id
            )
            input_ids = torch.cat([input_ids, pad], dim=1)
        else:
            input_ids = input_ids[:, :_TEXT_MAX_LENGTH]
        input_ids = input_ids.to(device)
        out = self.text_encoder(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        embeds = full_concat_embeddings(out.hidden_states)
        # Store on CPU to avoid consuming GPU VRAM in the cache.
        if len(self._text_embed_cache) >= _TEXT_EMBED_CACHE_MAX_SIZE:
            self._text_embed_cache.popitem(last=False)
        self._text_embed_cache[prompt] = embeds.detach().cpu()
        return embeds

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

    def _preprocess_hdmap_clip(
        self,
        frames: list,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Per-frame HD-map rasters -> a single clip ``[B, 3, T, H, W]`` in ``[-1, 1]``.

        Each frame runs the same single-frame ``_preprocess_pixels`` path
        (PIL/path/tensor -> resize -> normalize) and the results are stacked on
        the temporal axis. Returns ``None`` if any frame fails to preprocess.
        """
        per_frame: list[torch.Tensor] = []
        for f in frames:
            x = self._preprocess_pixels(f, height, width, device, dtype)
            if x is None:
                return None
            per_frame.append(x)  # [B,3,1,H,W]
        if not per_frame:
            return None
        return torch.cat(per_frame, dim=2)  # [B,3,T,H,W]

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
        normed = (latent - mean) / std
        _log_omnidreams_stats("vae_encode_normed", normed)
        return normed

    @staticmethod
    def _resolve_hdmap_frames(hdmap) -> list | None:
        """Resolve an HD-map input to a per-frame raster list, or ``None`` for the
        degenerate single-image case (caller broadcasts one raster).

        * video path (``_HDMAP_VIDEO_EXTS``) -> decoded to per-frame PIL images;
        * ``list`` / ``tuple`` -> per-frame sequence as-is, **except** a single
          video-path element is expanded via ``load_video`` (CLI ``--hdmap-path``
          always passes a list, even for a single arg);
        * single image (path / PIL / tensor) -> ``None``.
        """
        if isinstance(hdmap, (list, tuple)):
            items = list(hdmap)
            if len(items) == 0:
                return None
            # CLI passes --hdmap-path as a list even for a single arg; expand a
            # lone video-path element via load_video so it becomes per-frame rasters.
            if (
                len(items) == 1
                and isinstance(items[0], str)
                and items[0].lower().endswith(_HDMAP_VIDEO_EXTS)
            ):
                return load_video(items[0])
            return items
        if isinstance(hdmap, str) and hdmap.lower().endswith(_HDMAP_VIDEO_EXTS):
            return load_video(hdmap)
        return None

    @torch.no_grad()
    def _encode_hdmap(
        self,
        batch: Req,
        device: torch.device,
        vae_dtype: torch.dtype,
        dit_dtype: torch.dtype,
        num_chunks: int,
        len_t: int,
        height: int,
        width: int,
    ) -> list[torch.Tensor] | None:
        """Per-frame HD-map conditioning -> ``list[num_chunks]`` of patchified tokens.

        HD-map is OmniDreams' central per-frame control signal (lane lines + actor
        boxes rendered at the ego pose); the generated viewpoint changes *because*
        the raster shifts frame-to-frame. The full per-frame raster sequence is
        VAE-encoded as one causal clip -- matching the output latent temporal
        layout -- then sliced into ``num_chunks`` groups of ``len_t`` latent
        frames. Returns ``None`` when the request carries no HD-map input, in
        which case the AR stage falls back to zeros (HDMap disabled).

        Accepts ``batch.hdmap_path`` / ``batch.hdmap_pixels`` as a video path
        (decoded per-frame), a per-frame list of rasters, or -- as a degenerate
        back-compat / smoke fallback -- a single image broadcast across every
        latent frame (no temporal motion).

        TODO(gpu): the HD-map pixel -> 16ch-latent VAE numerics and the causal
        multi-frame temporal compression are validated on GPU; this encode path
        only runs when real HD-map input is supplied.
        """
        hdmap = getattr(batch, "hdmap_path", None)
        if hdmap is None:
            hdmap = getattr(batch, "hdmap_pixels", None)
        if hdmap is None:
            return None

        # L latent frames total -> 1 + (L-1)*4 pixel frames (causal VAE, tc=4),
        # matching the output chunk math (chunk0=1+(len_t-1)*4, later=len_t*4).
        num_latent = num_chunks * len_t
        total_pixel = 1 + (num_latent - 1) * 4

        frames = self._resolve_hdmap_frames(hdmap)
        if frames is None:
            # Degenerate single-image fallback: one raster broadcast across all
            # latent frames (no temporal motion -- back-compat / smoke only).
            x = self._preprocess_pixels(hdmap, height, width, device, vae_dtype)
            if x is None:
                logger.warning(
                    "OmniDreams: HD-map preprocessed to None; disabling HDMap "
                    "(all chunks fall back to zeros). Check hdmap input."
                )
                return None
            latent = self._vae_encode_normalized(x).to(dit_dtype)  # [B,16,1,h,w]
            if num_latent > 1 and latent.ndim == 5 and latent.shape[2] == 1:
                latent = latent.repeat(1, 1, num_latent, 1, 1)
        else:
            # Per-frame path: clamp/truncate to total_pixel, then encode the whole
            # clip causally -> num_latent distinct latent frames.
            if len(frames) < total_pixel:
                logger.warning(
                    "OmniDreams: HD-map has %d frames but %d are needed for "
                    "%d chunks x len_t=%d; clamping (repeating last frame).",
                    len(frames),
                    total_pixel,
                    num_chunks,
                    len_t,
                )
                frames = list(frames) + [frames[-1]] * (total_pixel - len(frames))
            else:
                frames = list(frames[:total_pixel])
            clip = self._preprocess_hdmap_clip(frames, height, width, device, vae_dtype)
            if clip is None:
                logger.warning(
                    "OmniDreams: HD-map clip preprocessed to None; disabling "
                    "HDMap (all chunks fall back to zeros). Check hdmap input."
                )
                return None
            latent = self._vae_encode_normalized(clip).to(dit_dtype)  # [B,16,L,h,w]

        if latent.shape[2] != num_latent:
            logger.warning(
                "OmniDreams: HD-map encoded to %d latent frames, expected %d "
                "(num_chunks=%d, len_t=%d). Check VAE temporal compression.",
                latent.shape[2],
                num_latent,
                num_chunks,
                len_t,
            )
        # Slice into per-chunk groups of len_t latent frames, patchify each:
        # [B,16,len_t,h,w] -> [B, chunk_tokens, additional_concat_ch*pdim].
        tokens: list[torch.Tensor] = []
        for ci in range(num_chunks):
            chunk_latent = latent[:, :, ci * len_t : (ci + 1) * len_t]
            tokens.append(self.transformer.patchify(chunk_latent))
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
        with self.use_declared_component(
            component_name="text_encoder", module=self.text_encoder
        ):
            text_embeds = self._encode_text(prompt, device).to(dit_dtype)
        batch.prompt_embeds = [text_embeds]
        batch.negative_prompt_embeds = None
        batch.image_embeds = []
        batch.do_classifier_free_guidance = False

        # --- i2v reference latent -> patchified frame-0 token block ---
        with self.use_declared_component(component_name="vae", module=self.vae):
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
        with self.use_declared_component(component_name="vae", module=self.vae):
            hdmap_tokens = self._encode_hdmap(
                batch, device, vae_dtype, dit_dtype, num_chunks, len_t, height, width
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

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        """Declare only the DiT for residency scheduling. The base
        ``DenoisingStage`` would also declare the VAE (this stage receives one
        to assert ``use_feature_cache``), but the AR rollout never runs the VAE
        here, so declaring it would needlessly hold it on the GPU through the
        denoise loop. The VAE encode/decode use-sites live in the before- and
        decoding-stages instead.
        """
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(
                stage_name=stage_name,
                component_name="transformer",
                phase="transformer",
                preferred_ready_after_request=True,
                memory_intensive=True,
            )
        ]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # Phase 6: SP is supported via USPAttention (self-attn) and
        # LocalAttention (cross-attn). Each rank processes its local
        # sequence shard; USPAttention handles all-to-all internally.
        # KV-cache and RoPE are sized per-rank via local chunk/window sizes.
        sp_size: int = 1
        sp_rank: int = 0
        try:
            sp_size = max(1, get_sp_world_size())
            from sglang.multimodal_gen.runtime.distributed.parallel_state import (
                get_sp_parallel_rank,
            )
            sp_rank = get_sp_parallel_rank()
        except (ImportError, AssertionError, RuntimeError):
            pass  # SP not initialized — single-GPU or CPU test.

        # The downstream single-pass decode of the concatenated AR latents
        # relies on the Wan VAE's causal temporal feature cache flowing across
        # frames within one decode() call. A VAE without feature caching would
        # change the frame-count semantics, so fail loudly rather than silently.
        if self.vae is not None and not getattr(self.vae, "use_feature_cache", True):
            raise RuntimeError(
                "OmniDreams AR rollout requires a Wan VAE with "
                "use_feature_cache=True for correct streaming frame counts."
            )

        # Bring the DiT onto the GPU for the whole AR rollout when it is being
        # CPU-offloaded (no-op when it is already resident). The manager keeps a
        # single DiT resident afterwards; on the error path the request-level
        # finish_request still releases it, so an explicit try/finally is not
        # needed here.
        residency_manager = self._component_residency_manager
        transformer_use = None
        if residency_manager is not None:
            transformer_use = self._declared_component_use(component_name="transformer")
            residency_manager.begin_use(transformer_use, self.transformer)

        config = server_args.pipeline_config
        device = get_local_torch_device()
        dit_dtype = PRECISION_TO_TYPE[config.dit_precision]
        arch = config.dit_config.arch_config
        st = batch.extra["omnidreams"]

        hp, wp, len_t = st["hp"], st["wp"], st["len_t"]
        # SP: each rank processes a local sub-chunk of the full sequence.
        # USPAttention handles all-to-all internally; the DiT forward
        # expects the local shard [B, S_local, D].
        local_chunk_tokens = st["chunk_tokens"] // sp_size
        local_window_tokens = st["window_size_t"] * st["tokens_per_frame"] // sp_size
        local_sink_tokens = st["sink_size_t"] * st["tokens_per_frame"] // sp_size
        tokens_per_frame = st["tokens_per_frame"]
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
            chunk_tokens=local_chunk_tokens,
            window_tokens=local_window_tokens,
            sink_tokens=local_sink_tokens,
            device=device,
            dtype=dit_dtype,
        )

        # Frame-0 conditioning (i2v): channel mask (into the DiT) + inject mask
        # (pins the clean reference latent) + the reference token block.
        frame0 = tokens_per_frame // sp_size
        image_token = st["image_token"]
        if image_token is not None:
            image_full = torch.zeros(
                B, local_chunk_tokens, in_d, device=device, dtype=dit_dtype
            )
            # Image pin: the first frame is split across SP ranks.
            # Each rank writes its local portion of the reference latent.
            image_full[:, :frame0, :] = (
                image_token[:, :frame0, :].to(device=device, dtype=dit_dtype)
            )
            inject_mask = torch.zeros(
                B, local_chunk_tokens, 1, device=device, dtype=dit_dtype
            )
            inject_mask[:, :frame0, :] = 1.0
            cond_mask_c0 = torch.zeros(
                B, local_chunk_tokens, mask_d, device=device, dtype=dit_dtype
            )
            cond_mask_c0[:, :frame0, :] = 1.0
        else:
            image_full = inject_mask = None
            cond_mask_c0 = torch.zeros(
                B, local_chunk_tokens, mask_d, device=device, dtype=dit_dtype
            )
        cond_mask_zero = torch.zeros(
            B, local_chunk_tokens, mask_d, device=device, dtype=dit_dtype
        )
        hdmap_zero = torch.zeros(
            B, local_chunk_tokens, hdmap_d, device=device, dtype=dit_dtype
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
            if sp_size > 1:
                rope_freqs = rope_freqs[
                    sp_rank * local_chunk_tokens : (sp_rank + 1) * local_chunk_tokens
                ]
            is_first = chunk_idx == 0
            cond_mask = cond_mask_c0 if is_first else cond_mask_zero
            # HD-map is per-chunk: index this chunk's tokens (None -> zeros, i.e.
            # HDMap disabled). Explicit None-check -- a tensor in ``or`` raises.
            if st["hdmap_tokens"] is None:
                hdmap_chunk = hdmap_zero
            else:
                full_hdmap = st["hdmap_tokens"][chunk_idx]
                if sp_size > 1:
                    full_hdmap = full_hdmap[
                        :, sp_rank * local_chunk_tokens : (sp_rank + 1) * local_chunk_tokens, :
                    ]
                hdmap_chunk = full_hdmap.to(device=device, dtype=dit_dtype)
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
                B, local_chunk_tokens, in_d, device=device, dtype=dit_dtype
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
            if pin:
                ctx_latent = ctx_latent * (1.0 - inject_mask) + image_full * inject_mask
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

        # SP post-process: gather sharded latents from all ranks.
        # Each rank produced its local portion of the AR rollout;
        # all-gather along the time dimension so the downstream decode
        # sees the full latent sequence.
        if sp_size > 1:
            batch.latents = sequence_model_parallel_all_gather(
                batch.latents, dim=2
            )

        _log_omnidreams_stats("ar_concat_latents", batch.latents)

        if residency_manager is not None and transformer_use is not None:
            residency_manager.end_use(transformer_use, self.transformer)

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
