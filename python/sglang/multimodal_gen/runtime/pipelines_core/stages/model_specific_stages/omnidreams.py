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

import inspect
import os
from collections import OrderedDict
from typing import Any

import PIL.Image
import torch
import torch.nn as nn

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.dits.omnidreams_cuda_graph import (
    CUDAGraphWrapper,
)
from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import build_fp8_dit
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
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams_hdmap_decode import (
    decode_hdmap_ab,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
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
_MAX_AR_CHUNKS = 320
# LRU cache for text embeddings (key = prompt string, value = [1, L, 100352] on CPU).
# Avoids re-running the 14 GB Cosmos-Reason1-7B for repeated prompts in serving.
_TEXT_EMBED_CACHE_MAX_SIZE = 32
# HD-map inputs ending in one of these are decoded as a per-frame raster video;
# any other single string is treated as one image (degenerate broadcast).
_HDMAP_VIDEO_EXTS = (".mp4", ".gif", ".webm", ".mov", ".mkv", ".avi")

# Cache for pre-built latent normalization tensors (keyed by VAE id, device, dtype).
_latent_norm_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
# Cache for whether vae.encode() accepts a `cache` kwarg (keyed by VAE type name).
_encode_accepts_cache: dict[str, bool] = {}

# --------------------------------------------------------------------------- #
# Diagnostics helpers                                                         #
# --------------------------------------------------------------------------- #
_OMNIDREAMS_DIAG_ENVS = ("SGLANG_OMNIDREAMS_DIAGNOSTICS",)


def _omnidreams_diag_enabled() -> bool:
    return any(
        os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
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


@torch.no_grad()
def _vae_encode_normalized(
    x: torch.Tensor,
    vae: nn.Module,
    cache: Any = None,
    is_first_chunk: bool = True,
) -> torch.Tensor:
    """``[B,3,T,H,W]`` pixels -> latent normalized into the DiT space.

    Module-level so both the before-stage (i2v reference + HD-map broadcast)
    and the denoising stage (per-chunk HD-map stream encode) share one path.
    Encode, take the distribution mode (deterministic), then ``(z-mean)/std``.

    Args:
        x: Input pixels in [-1, 1] range.
        vae: VAE encoder module (image_encoder or encoder).
        cache: Optional persistent streaming encode cache (LightVAE AR
            per-chunk path). Passed straight through to ``vae.encode`` when
            the encoder supports it; otherwise ignored (one-shot encode).
        is_first_chunk: Only meaningful with ``cache``: the first chunk seeds
            the causal left-context, later chunks continue the stream.
    """
    encode_kwargs: dict[str, Any] = {}
    if cache is not None:
        vae_type = type(vae).__name__
        if vae_type not in _encode_accepts_cache:
            _encode_accepts_cache[vae_type] = (
                "cache" in inspect.signature(vae.encode).parameters
            )
        if _encode_accepts_cache[vae_type]:
            encode_kwargs["cache"] = cache
            encode_kwargs["is_first_chunk"] = is_first_chunk
    latent_dist = vae.encode(x, **encode_kwargs)
    latent = (
        latent_dist.mode()
        if hasattr(latent_dist, "mode")
        else latent_dist.sample() if hasattr(latent_dist, "sample") else latent_dist
    )
    cache_key = (id(vae), latent.device, latent.dtype)
    cached = _latent_norm_cache.get(cache_key)
    if cached is None:
        mean = torch.tensor(
            vae.latents_mean, device=latent.device, dtype=latent.dtype
        ).view(1, -1, 1, 1, 1)
        std = torch.tensor(
            vae.latents_std, device=latent.device, dtype=latent.dtype
        ).view(1, -1, 1, 1, 1)
        _latent_norm_cache[cache_key] = (mean, std)
    else:
        mean, std = cached
    normed = (latent - mean) / std
    _log_omnidreams_stats("vae_encode_normed", normed)
    return normed


def _hdmap_chunk_pixel_bounds(
    chunk_idx: int, len_t: int, tc: int = 4
) -> tuple[int, int]:
    """Pixel-frame ``[start, end)`` slice for HD-map chunk ``chunk_idx``.

    Mirrors the causal VAE chunk math (temporal compression ``tc=4``): chunk 0
    covers ``1 + (len_t-1)*tc`` pixel frames, every later chunk covers
    ``len_t*tc``. The concatenation of all chunks equals
    ``total_pixel = 1 + (num_chunks*len_t - 1) * tc`` (closed under the AR
    geometry), so independent per-chunk encodes tile the full decoded clip
    without gaps or overlaps -- matching the FlashDreams per-step slice encode.
    """
    if chunk_idx == 0:
        return 0, 1 + (len_t - 1) * tc
    base = 1 + (len_t - 1) * tc + (chunk_idx - 1) * len_t * tc
    return base, base + len_t * tc


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
        image_encoder=None,
        encoder=None,
        config=None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.image_encoder = image_encoder  # one-shot first-frame I2V conditioning
        self.encoder = encoder  # per-AR-step HDMap conditioning
        self.config = config
        # Per-instance LRU cache: prompt string -> text embedding on CPU.
        self._text_embed_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        """Declare text_encoder + image_encoder + encoder for residency manager."""
        stage_name = self._component_stage_name(stage_name)
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        uses = [ComponentUse(stage_name=stage_name, component_name="text_encoder")]

        # image_encoder (may be None for T2V tasks without I2V conditioning)
        if self.image_encoder is not None:
            uses.append(
                ComponentUse(
                    stage_name=stage_name,
                    component_name="image_encoder",
                    target_dtype=vae_dtype,
                )
            )

        # encoder (HDMap, always present)
        uses.append(
            ComponentUse(
                stage_name=stage_name,
                component_name="encoder",
                target_dtype=vae_dtype,
            )
        )
        return uses

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
        return _vae_encode_normalized(x, self.image_encoder)

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

    @staticmethod
    def _resolve_hdmap_video_path(hdmap) -> str | None:
        """Return the on-disk video path if ``hdmap`` is (or wraps) a single video
        file, else ``None``. Used to route the per-frame video path through the
        A+B decode (``decode_hdmap_ab``): early-stop ffmpeg at ``total_pixel``
        frames + numpy->torch preprocess (5-9x faster than decoding the whole
        clip via ``load_video`` then truncating, near-zero drift). ``None`` for
        in-memory frame lists / single images, which keep the legacy path.
        """
        if isinstance(hdmap, (list, tuple)):
            items = list(hdmap)
            if len(items) == 1 and isinstance(items[0], str) and items[0].lower().endswith(
                _HDMAP_VIDEO_EXTS
            ):
                return items[0]
            return None
        if isinstance(hdmap, str) and hdmap.lower().endswith(_HDMAP_VIDEO_EXTS):
            return hdmap
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
    ) -> tuple[list[torch.Tensor] | None, torch.Tensor | None]:
        """Prepare HD-map conditioning for the AR loop.

        Returns ``(hdmap_tokens, hdmap_pixel)``:

        * No HD-map input -> ``(None, None)`` (AR stage falls back to zeros).
        * Degenerate single-image fallback -> ``(tokens, None)``: the single
          raster is VAE-encoded once here and broadcast across every latent
          frame, then sliced into ``num_chunks`` precomputed patchified tokens
          (no temporal motion -- back-compat / smoke only).
        * Per-frame (video) path -> ``(None, hdmap_pixel)``: the full per-frame
          raster sequence is decoded once (``load_video`` / A+B fast path) and
          preprocessed into one causal clip ``hdmap_pixel``
          ``[B, 3, total_pixel, H, W]`` on ``device``. The per-chunk VAE encode
          + patchify is **deferred to the AR loop** (see
          :class:`OmniDreamsDenoisingStage`), aligning with the FlashDreams
          replay path (one-shot decode + per-step slice encode) and enabling
          per-step closed-loop conditioning where each chunk's pixels arrive at
          runtime.

        HD-map is OmniDreams' central per-frame control signal (lane lines +
        actor boxes rendered at the ego pose); the generated viewpoint changes
        *because* the raster shifts frame-to-frame.
        """
        hdmap = getattr(batch, "hdmap_path", None)
        if hdmap is None:
            hdmap = getattr(batch, "hdmap_pixels", None)
        if hdmap is None:
            return None, None

        # L latent frames total -> 1 + (L-1)*4 pixel frames (causal VAE, tc=4),
        # matching the output chunk math (chunk0=1+(len_t-1)*4, later=len_t*4).
        num_latent = num_chunks * len_t
        total_pixel = 1 + (num_latent - 1) * 4

        # A+B fast path (per-frame video): when the HD-map input is an on-disk
        # video file, decode only ``total_pixel`` frames (early-stop ffmpeg) and
        # preprocess via numpy->torch (cv2 resize) -- 5-9x faster than the legacy
        # ``load_video``-decodes-all-frames + PIL path, near-zero drift (bit-
        # identical at native res; cv2-vs-PIL lanczos ~0.07 at resize). See
        # ``omnidreams_hdmap_decode.py`` + the A+B benchmark in the progress doc.
        video_path = self._resolve_hdmap_video_path(hdmap)
        if video_path is not None:
            try:
                clip = decode_hdmap_ab(
                    video_path, total_pixel, height, width, device, vae_dtype
                )  # [1, 3, total_pixel, H, W] in [-1, 1]
            except Exception:
                logger.exception(
                    "OmniDreams: A+B HD-map decode failed for %s; falling back to "
                    "legacy load_video path.", video_path
                )
                clip = None
            if clip is not None:
                return None, clip
            # Fall through to the legacy path below on failure.

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
                return None, None
            latent = _vae_encode_normalized(x, self.encoder).to(dit_dtype)  # [B,16,1,h,w]
            if num_latent > 1 and latent.ndim == 5 and latent.shape[2] == 1:
                latent = latent.repeat(1, 1, num_latent, 1, 1)
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
            return tokens, None

        # Per-frame (video) path: decode + preprocess the full causal clip once
        # here (one-shot decode, matching FlashDreams ``_load_video``); defer the
        # per-chunk VAE encode + patchify to the AR loop. Clamp/truncate to
        # total_pixel (repeating the last frame when short).
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
            return None, None
        return None, clip

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
        # When the text encoder is CPU-offloaded, skip the residency manager's
        # enter()/exit() — it would forcefully move the 14 GB Cosmos-Reason1-7B onto
        # the GPU, which OOMs on cards < 16 GB (e.g. RTX 5070 12 GB).  Run directly
        # on CPU instead; the single forward is fast enough for a 7B LM and the
        # embedding is cached on CPU for reuse across requests.
        _te_offloaded = getattr(server_args, "text_encoder_cpu_offload", False)
        if _te_offloaded:
            text_embeds = self._encode_text(prompt, torch.device("cpu")).to(
                device=device, dtype=dit_dtype
            )
        else:
            with self.use_declared_component(
                component_name="text_encoder", module=self.text_encoder
            ):
                text_embeds = self._encode_text(prompt, device).to(dit_dtype)
        batch.prompt_embeds = [text_embeds]
        batch.negative_prompt_embeds = None
        batch.image_embeds = []
        batch.do_classifier_free_guidance = False

        # --- i2v reference latent -> patchified frame-0 token block ---
        with self.use_declared_component(
            component_name="image_encoder", module=self.image_encoder
        ):
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
        # HD-map prep. The per-frame video path returns a preprocessed pixel clip
        # (``hdmap_pixel``) whose per-chunk VAE encode + patchify is deferred to
        # the AR loop (FlashDreams replay: one-shot decode + per-step encode);
        # the degenerate single-image fallback returns precomputed per-chunk
        # tokens. No HD-map input -> (None, None) (AR stage uses zeros).
        with self.use_declared_component(
            component_name="encoder", module=self.encoder
        ):
            hdmap_tokens, hdmap_pixel = self._encode_hdmap(
                batch, device, vae_dtype, dit_dtype, num_chunks, len_t, height, width
            )
            # Keep the full hdmap pixel clip on CPU; the AR loop brings
            # each small per-chunk slice to GPU for encode (~13GB saved
            # for long videos; otherwise resident for the whole rollout).
            if hdmap_pixel is not None:
                hdmap_pixel = hdmap_pixel.cpu()
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
            # HD-map, in two mutually-exclusive shapes (both None = disabled):
            #  * ``hdmap_tokens``: None, or list[num_chunks] of precomputed
            #    [B, chunk_tokens, additional_concat_ch*pdim] (single-image
            #    broadcast fallback only);
            #  * ``hdmap_pixel``: None, or a full preprocessed clip
            #    [B, 3, total_pixel, H, W] -- per-chunk VAE-encoded in the AR
            #    loop (the per-frame video path).
            "hdmap_tokens": hdmap_tokens,
            "hdmap_pixel": hdmap_pixel,
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

    def __init__(self, transformer, scheduler, decoder=None, encoder=None) -> None:
        super().__init__(transformer, scheduler, vae=decoder)
        # HD-map VAE encoder: per-chunk VAE-encoded inside the AR loop (the
        # per-frame video path defers its encode here from the before-stage).
        self.encoder = encoder

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        """Declare the DiT (and HD-map VAE encoder, when present) for residency.

        The base ``DenoisingStage`` would also declare the *decoder* VAE (this
        stage receives one only to assert ``use_feature_cache``); the AR rollout
        never runs the decoder here, so it is deliberately NOT declared -- the
        decode use-site lives in the decoding stage. The HD-map *encoder*, by
        contrast, IS now used per-chunk inside the AR loop (the per-frame video
        path defers its VAE encode here), so it is declared and held resident
        alongside the DiT when present.
        """
        stage_name = self._component_stage_name(stage_name)
        uses = [
            ComponentUse(
                stage_name=stage_name,
                component_name="transformer",
                phase="transformer",
                preferred_ready_after_request=True,
                memory_intensive=True,
            )
        ]
        if self.encoder is not None:
            uses.append(
                ComponentUse(
                    stage_name=stage_name,
                    component_name="encoder",
                    phase="encoder",
                    preferred_ready_after_request=True,
                    memory_intensive=False,
                )
            )
        return uses

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

        # Bring the DiT onto the GPU for the whole AR rollout when it is being
        # CPU-offloaded (no-op when it is already resident). The manager keeps a
        # single DiT resident afterwards; on the error path the request-level
        # finish_request still releases it, so an explicit try/finally is not
        # needed here.
        residency_manager = self._component_residency_manager
        transformer_use = None
        encoder_use = None
        if residency_manager is not None:
            transformer_use = self._declared_component_use(component_name="transformer")
            residency_manager.begin_use(transformer_use, self.transformer)
            # The HD-map encoder VAE is per-chunk VAE-encoded in the AR loop
            # (per-frame video path). Hold it resident alongside the DiT.
            if self.encoder is not None:
                encoder_use = self._declared_component_use(component_name="encoder")
                residency_manager.begin_use(encoder_use, self.encoder)

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

        # CUDA-graph capture of the steady-state DiT calls. The 3 calls per
        # chunk (2 self-forcing denoise steps + 1 context re-forward) share
        # identical tensor shapes once the KV window is steady, so a single
        # captured graph replays for all of them. ``_dit_call`` binds the
        # loop-invariant args (text/caches/cross_attn_kv/view_indices); the
        # per-call-varying tensors (noisy, timestep, cond_mask, rope, hdmap) are
        # passed positionally so the wrapper stages them into static buffers and
        # copies them in per replay. Fill-phase chunks run eager because
        # BlockKVCache.cached_k() returns a variable-length slice until the
        # window is full (a graph captured then would bake the wrong shape).
        use_cuda_graph = (
            getattr(config, "enable_cuda_graph", False)
            or envs.SGLANG_OMNIDREAMS_CUDA_GRAPH
        ) and device.type == "cuda"

        def _dit_call(hidden_states, timestep, cond_mask_t, rope_t, hdmap_t):
            # Eager path: rope_t is the [L, D] cos|sin cache (shift_t).
            return self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=text,
                timestep=timestep,
                condition_video_input_mask=cond_mask_t,
                rope_cos_sin=rope_t,
                hdmap_condition=hdmap_t,
                kv_caches=caches,
                cross_attn_kv=cross_attn_kv,
                view_indices=view_indices,
            )

        cuda_graph_runner = (
            CUDAGraphWrapper(
                _dit_call,
                warmup_iters=getattr(config, "cuda_graph_warmup_iters", 2),
            )
            if use_cuda_graph
            else None
        )

        # Native FP8 DiT (optimized_dit_forward). Mutually exclusive with
        # the CUDA graph (the native op manages its own graph). GPU/sm_120
        # only -- build_fp8_dit returns None when the native ext is unavailable,
        # so this transparently falls back to the eager DiT.
        fp8_dit = None
        mode = getattr(config, "native_dit_acceleration", "disabled")
        if envs.SGLANG_OMNIDREAMS_FP8_DIT:
            mode = "required"  # env forces required mode

        if mode != "disabled":
            if cuda_graph_runner is not None:
                logger.warning(
                    "OmniDreams: native_dit_acceleration overrides enable_cuda_graph "
                    "(the native FP8 op manages its own CUDA graph)."
                )
                cuda_graph_runner = None

            # Resolve fp8_prepared_path: explicit config wins, else infer
            # alongside the raw checkpoint.
            model_path = server_args.model_path
            fp8_prepared_path = getattr(config, "native_dit_fp8_prepared_path", None)
            if fp8_prepared_path is None:
                if os.path.isfile(model_path):
                    ckpt_dir = os.path.dirname(model_path)
                else:
                    ckpt_dir = model_path
                fp8_prepared_path = os.path.join(ckpt_dir, "omnidreams_fp8_dit.pt")

            # Lightweight fingerprint check: if the pre-quantized file exists
            # but the raw checkpoint has a newer mtime, warn and fall back
            # (or error for required mode) — avoids silently using stale
            # quantized weights after a checkpoint upgrade.
            if os.path.exists(fp8_prepared_path):
                try:
                    payload = torch.load(
                        fp8_prepared_path, map_location="cpu", weights_only=True
                    )
                    meta = payload.get("meta", {})
                    fp = meta.get("checkpoint_fingerprint", {})
                    # Resolve raw checkpoint path for fingerprint comparison
                    _DEFAULT_CKPT_RELPATH = "single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt"
                    if os.path.isfile(model_path):
                        raw_path = model_path
                    else:
                        raw_path = os.path.join(model_path, _DEFAULT_CKPT_RELPATH)
                    if os.path.exists(raw_path):
                        ckpt_stat = os.stat(raw_path)
                        if (fp.get("file_size") != ckpt_stat.st_size
                                or fp.get("mtime") != ckpt_stat.st_mtime):
                            logger.warning(
                                "OmniDreams: FP8 prepared weights fingerprint "
                                "mismatch (checkpoint may have been updated). "
                                "Re-run: python -m sglang.multimodal_gen.tools."
                                "export_omnidreams_fp8_dit_weights "
                                "--checkpoint %s --output %s",
                                raw_path, fp8_prepared_path,
                            )
                            if mode == "required":
                                raise RuntimeError(
                                    "FP8 prepared weights fingerprint mismatch. "
                                    "Re-run the offline exporter."
                                )
                            fp8_prepared_path = None  # force fallback / skip
                except Exception:
                    # Corrupt or old-format file — treat as missing.
                    logger.warning(
                        "OmniDreams: failed to read FP8 prepared weights at %s; "
                        "treating as missing.", fp8_prepared_path
                    )
                    if mode == "required":
                        raise
                    fp8_prepared_path = None

            fp8_dit = build_fp8_dit(
                self.transformer,
                arch,
                mode=mode,
                fp8_prepared_path=fp8_prepared_path,
                attention_backend=getattr(config, "native_dit_backend", "auto"),
            )
            if fp8_dit is None and mode == "required":
                raise RuntimeError(
                    "native_dit_acceleration='required' but native FP8 DiT unavailable. "
                    "Check sm_120 native ext build."
                )
            elif fp8_dit is None:
                logger.info("OmniDreams: native FP8 DiT unavailable; using eager DiT.")

        latent_chunks: list[torch.Tensor] = []
        # Persistent streaming VAE-encode cache for the per-frame HD-map path.
        # The LightVAE causal conv left-context must flow across AR chunks
        # (chunk 0 seeds, chunk 1+ continues) rather than re-seeding every
        # call — otherwise the short tail of a later chunk underflows
        # ``time_conv`` (kernel=3) at the deepest downsample. Allocated once
        # per rollout when the encoder supports the streaming contract.
        hdmap_encode_cache: Any = None
        if st["hdmap_pixel"] is not None and hasattr(
            self.encoder, "initialize_ar_encode_cache"
        ):
            hdmap_encode_cache = self.encoder.initialize_ar_encode_cache()
        for chunk_idx in range(num_chunks):
            # Eager/CUDA-graph paths consume the [L, D] cos|sin cache; the native
            # FP8 path needs the raw-angle [L,1,1,head_dim] tensor (it takes
            # cos/sin itself). Only build the latter when native is active.
            rope_cos_sin = rope.shift_t(chunk_idx)
            rope_freqs = (
                rope.shift_t_freqs(chunk_idx) if fp8_dit is not None else None
            )
            is_first = chunk_idx == 0
            cond_mask = cond_mask_c0 if is_first else cond_mask_zero
            # HD-map conditioning for this chunk, in three mutually-exclusive
            # shapes (set in the before-stage's ``_encode_hdmap``):
            #  * ``hdmap_pixel``: per-frame video path -> VAE-encode this chunk's
            #    pixel slice here (FlashDreams replay: one-shot decode + per-step
            #    slice encode). The causal VAE conv no longer crosses chunk
            #    boundaries, so boundary latents differ from a one-shot encode --
            #    by design, matching FlashDreams.
            #  * ``hdmap_tokens``: single-image broadcast fallback -> precomputed.
            #  * both None -> HDMap disabled (zeros).
            if st["hdmap_pixel"] is not None:
                s, e = _hdmap_chunk_pixel_bounds(chunk_idx, len_t)
                chunk_clip = st["hdmap_pixel"][:, :, s:e].to(
                    device=device
                )  # [B,3,T_chunk,H,W]
                chunk_latent = _vae_encode_normalized(
                    chunk_clip,
                    self.encoder,
                    cache=hdmap_encode_cache,
                    is_first_chunk=is_first,
                ).to(dit_dtype)  # [B,16,len_t,h,w]
                hdmap_chunk = self.transformer.patchify(chunk_latent)
            elif st["hdmap_tokens"] is not None:
                hdmap_chunk = st["hdmap_tokens"][chunk_idx].to(
                    device=device, dtype=dit_dtype
                )
            else:
                hdmap_chunk = hdmap_zero
            pin = is_first and image_full is not None

            # Roll the per-block KV window BEFORE deciding capture eligibility:
            # is_steady_state() then reflects whether this chunk reads a full
            # (fixed-shape) window -- the capture precondition. Once steady it
            # stays steady, so the graph captured on the first steady chunk
            # replays for every chunk after.
            for c in caches:
                c.before_update(chunk_idx)
            steady_now = (
                cuda_graph_runner is not None and caches[0].is_steady_state()
            )

            def _call_dit(hidden_states, timestep):
                if fp8_dit is not None:
                    # Native FP8 path (owns its own KV write at write_start).
                    return fp8_dit(
                        hidden_states=hidden_states,
                        encoder_hidden_states=text,
                        timestep=timestep,
                        condition_video_input_mask=cond_mask,
                        rope_freqs=rope_freqs,
                        hdmap_condition=hdmap_chunk,
                        kv_caches=caches,
                        cross_attn_kv=cross_attn_kv,
                        view_indices=view_indices,
                        ar_idx=chunk_idx,
                        len_t=len_t,
                        hp=hp,
                        wp=wp,
                    )
                if steady_now:
                    return cuda_graph_runner(
                        hidden_states, timestep, cond_mask, rope_cos_sin, hdmap_chunk
                    )
                return _dit_call(
                    hidden_states, timestep, cond_mask, rope_cos_sin, hdmap_chunk
                )

            def predict_flow(noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                # The first-frame ``pin`` injection stays EAGER (outside the
                # captured graph) so ``image_full`` is not baked in. pin is only
                # true on chunk 0, which is always fill-phase (eager) anyway.
                if pin:
                    noisy = noisy * (1.0 - inject_mask) + image_full * inject_mask
                return _call_dit(noisy, t)

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
            if pin:
                ctx_latent = ctx_latent * (1.0 - inject_mask) + image_full * inject_mask
            # Return ignored: this forward exists for its in-cache K/V write
            # side effect (captured inside the graph at steady-state).
            _call_dit(ctx_latent, ctx_noise_t)

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

        # Release the hdmap pixel clip now the rollout is done; the decode
        # stage only needs batch.latents. The clip lives on CPU (~13GB for
        # long hdmap videos) and holding it through the decode .float()
        # cast pushes peak CPU past the cgroup limit (kernel OOM-kill).
        st["hdmap_pixel"] = None

        _log_omnidreams_stats("ar_concat_latents", batch.latents)

        if residency_manager is not None and transformer_use is not None:
            residency_manager.end_use(transformer_use, self.transformer)
        if residency_manager is not None and encoder_use is not None:
            residency_manager.end_use(encoder_use, self.encoder)

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


class OmniDreamsLightTAEDecodingStage(DecodingStage):
    """Single-pass decode via the LightTAE (TAEHV) tiny decoder.

    Differs from the standard ``DecodingStage`` in two ways:

    * ``scale_and_shift`` is a no-op. :class:`LightTAEDecoder` un-normalizes the
      DiT latent with the LightTAE per-channel mean/std INSIDE its ``decode``;
      applying the Wan ``scale_and_shift`` here too would double-normalize.
    * ``load_model`` is a no-op. The (small ~45 MB) LightTAE decoder is custom-
      loaded in ``OmniDreamsPipeline.load_modules`` and passed in pre-built, so
      the lazy ``VAELoader`` path (keyed on ``model_loaded``) must be skipped.

    The decoder is registered under its own ``vae_decoder`` component so the
    Wan VAE (still used for encode) keeps the ``vae`` slot.
    """

    def scale_and_shift(self, latents: torch.Tensor, server_args):
        return latents

    def load_model(self):
        return None
