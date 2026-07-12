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

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.dits.omnidreams import (
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
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
    DecodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime.vae import (
    CausalVaeDecodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams_hdmap_decode import (
    decode_hdmap_ab,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.realtime.states import (
    RealtimeCausalDiTState,
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
    mean = torch.tensor(
        vae.latents_mean, device=latent.device, dtype=latent.dtype
    ).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.latents_std, device=latent.device, dtype=latent.dtype).view(
        1, -1, 1, 1, 1
    )
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


def _is_realtime(batch: Req) -> bool:
    """True when this request is part of a realtime (streaming) session.

    Mirrors the detection contract used by the other realtime stages
    (``realtime_vae.py``, ``causal_denoising.py``): a non-empty
    ``realtime_session_id`` plus an attached ``session`` object.
    """
    return (
        bool(getattr(batch, "realtime_session_id", None)) and batch.session is not None
    )


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
            if (
                len(items) == 1
                and isinstance(items[0], str)
                and items[0].lower().endswith(_HDMAP_VIDEO_EXTS)
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
                    "legacy load_video path.",
                    video_path,
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
            latent = _vae_encode_normalized(x, self.encoder).to(
                dit_dtype
            )  # [B,16,1,h,w]
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

        # --- realtime session path ----------------------------------------- #
        # In realtime mode each forward() processes exactly one AR chunk driven
        # by ``batch.block_idx``. block_idx==0 runs the full one-shot prep (text
        # encode, i2v reference, schedule, hdmap) and stashes the result into a
        # persistent ``RealtimeCausalDiTState``; block_idx>0 short-circuits and
        # only refreshes the per-chunk hdmap pixels from ``condition_inputs``.
        # The offline path below is bit-identical and untouched.
        is_realtime = _is_realtime(batch)
        if is_realtime and batch.block_idx > 0:
            return self._realtime_before_subsequent_chunk(batch, server_args)

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
        with self.use_declared_component(component_name="encoder", module=self.encoder):
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

        # In realtime mode, stash the one-shot prep into the persistent DiT
        # cache state so subsequent chunks (block_idx>0) skip the text/image
        # encoders entirely. The denoise stage reads these back via
        # ``cache_state.runtime_cache``. num_chunks is forced to 1: the
        # realtime denoise stage processes exactly one chunk per forward().
        if is_realtime and batch.block_idx == 0:
            self._realtime_stash_initial_state(
                batch,
                server_args,
                rope=None,  # built lazily in the denoise stage (needs head_dim)
                text_embeds=text_embeds,
                image_full=None,  # assembled in the denoise stage (needs device)
                inject_mask=None,
                cond_mask_c0=None,
                cond_mask_zero=None,
                hdmap_zero=None,
                cross_attn_kv=None,  # precomputed in the denoise stage
                scheduler=scheduler,
                generator=batch.generator,
                hdmap_encode_cache=None,
                arch_constants={
                    "hp": hp,
                    "wp": wp,
                    "len_t": len_t,
                    "tokens_per_frame": tokens_per_frame,
                    "chunk_tokens": chunk_tokens,
                    "head_dim": arch.model_channels // arch.num_heads,
                    "in_d": arch.in_channels
                    * arch.patch_temporal
                    * arch.patch_spatial**2,
                    "hdmap_d": arch.additional_concat_ch
                    * arch.patch_temporal
                    * arch.patch_spatial**2,
                    "mask_d": arch.patch_temporal * arch.patch_spatial**2,
                    "context_noise": float(getattr(batch, "context_noise", 128)),
                    "window_size_t": int(getattr(batch, "window_size_t", 6)),
                    "sink_size_t": int(getattr(batch, "sink_size_t", 0)),
                },
                hdmap_tokens=hdmap_tokens,
                hdmap_pixel=hdmap_pixel,
                image_token=image_token,
            )
            # Realtime per-call: the denoise stage runs exactly one chunk.
            batch.extra["omnidreams"]["num_chunks"] = 1

        return batch

    def _realtime_stash_initial_state(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        rope,
        text_embeds: torch.Tensor,
        image_full,
        inject_mask,
        cond_mask_c0,
        cond_mask_zero,
        hdmap_zero,
        cross_attn_kv,
        scheduler,
        generator,
        hdmap_encode_cache,
        arch_constants: dict,
        hdmap_tokens,
        hdmap_pixel,
        image_token,
    ) -> None:
        """Stash one-shot prep into ``RealtimeCausalDiTState.runtime_cache``.

        The denoise stage reads these back on every realtime forward() call.
        Tensors that need device/dit_dtype assembly (rope, masks, cross_attn_kv,
        image_full) are built lazily in the denoise stage on the first chunk to
        avoid duplicating the device-aware construction logic here; only the
        device-independent inputs (text embeds, image_token, hdmap, scheduler,
        generator, arch constants) are stashed here.
        """
        cache_state = batch.session.get_or_create_state(RealtimeCausalDiTState)
        rc = cache_state.runtime_cache
        rc.clear()
        rc["rope"] = rope
        rc["text_embeds"] = text_embeds.detach()
        rc["image_token"] = image_token.detach() if image_token is not None else None
        rc["image_full"] = image_full
        rc["inject_mask"] = inject_mask
        rc["cond_mask_c0"] = cond_mask_c0
        rc["cond_mask_zero"] = cond_mask_zero
        rc["hdmap_zero"] = hdmap_zero
        rc["cross_attn_kv"] = cross_attn_kv
        rc["scheduler"] = scheduler
        rc["generator"] = generator
        rc["hdmap_encode_cache"] = hdmap_encode_cache
        rc["arch_constants"] = dict(arch_constants)
        # HD-map conditioning (one of these is non-None when HDMap is enabled):
        #  * ``hdmap_tokens``: list[num_chunks] precomputed per-chunk tokens
        #    (single-image broadcast fallback);
        #  * ``hdmap_pixel``: full preprocessed clip on CPU, per-chunk slices
        #    VAE-encoded in the denoise loop (per-frame video path).
        rc["hdmap_tokens"] = hdmap_tokens
        rc["hdmap_pixel"] = hdmap_pixel
        cache_state.kv_cache = None  # initialized in the denoise stage
        cache_state.chunk_idx = 0

    @torch.no_grad()
    def _realtime_before_subsequent_chunk(
        self, batch: Req, server_args: ServerArgs
    ) -> Req:
        """block_idx>0 realtime path: skip encoders, refresh per-chunk hdmap.

        Loads the stashed ``RealtimeCausalDiTState`` and rebuilds a minimal
        ``batch.extra["omnidreams"]`` for the single-chunk denoise call. The
        per-chunk HD-map pixels arrive via ``batch.condition_inputs["hdmap"]``
        (closed-loop); when absent the AR stage falls back to the stashed
        ``hdmap_pixel`` clip slice or zeros (open-loop).
        """
        cache_state = batch.session.get_or_create_state(RealtimeCausalDiTState)
        rc = cache_state.runtime_cache
        ac = rc["arch_constants"]
        hp = ac["hp"]
        wp = ac["wp"]
        len_t = ac["len_t"]
        tokens_per_frame = ac["tokens_per_frame"]
        chunk_tokens = ac["chunk_tokens"]

        # Per-chunk closed-loop HDMap: the adapter injects this chunk's pixels
        # via condition_inputs. When present, stash as ``hdmap_pixel_chunk`` so
        # the denoise stage VAE-encodes just this slice (mirroring the offline
        # per-frame video path). Otherwise fall back to the stashed clip slice
        # or zeros (open-loop).
        hdmap_chunk_pixels = None
        cond_inputs = getattr(batch, "condition_inputs", None) or {}
        hdmap_in = cond_inputs.get("hdmap")
        if hdmap_in is not None and torch.is_tensor(hdmap_in):
            # [B, 3, T_chunk, H, W] in [-1, 1], already preprocessed by the
            # adapter. Kept on CPU; the denoise stage moves the slice to GPU.
            hdmap_chunk_pixels = hdmap_in.detach().cpu()

        batch.extra["omnidreams"] = {
            "hp": hp,
            "wp": wp,
            "len_t": len_t,
            "tokens_per_frame": tokens_per_frame,
            "chunk_tokens": chunk_tokens,
            "latent_h": None,  # not used by the realtime denoise path
            "latent_w": None,
            "num_chunks": 1,  # realtime: one chunk per forward()
            "window_size_t": ac["window_size_t"],
            "sink_size_t": ac["sink_size_t"],
            "context_noise": ac["context_noise"],
            "image_token": rc["image_token"],
            "hdmap_tokens": rc["hdmap_tokens"],
            "hdmap_pixel": rc["hdmap_pixel"],
            # Per-chunk closed-loop override (highest priority in the denoise
            # stage's HDMap resolution).
            "hdmap_pixel_chunk": hdmap_chunk_pixels,
            # Signal to the denoise stage that this is a subsequent realtime
            # chunk (state already initialized).
            "realtime_subsequent": True,
        }
        # raw_latent_shape lets SDPA-path attn metadata stay a no-op.
        arch = server_args.pipeline_config.dit_config.arch_config
        batch.raw_latent_shape = (
            rc["text_embeds"].shape[0],
            chunk_tokens,
            arch.out_channels,
            None,
            None,
        )
        # Re-hydrate the standard fields the denoise stage reads. The schedule
        # fields (timesteps / sigmas / num_inference_steps / guidance_scale /
        # eta) mirror the block_idx==0 Before path (``batch.timesteps =
        # scheduler.denoising_step_list.to(device)`` above) so
        # ``DenoisingStage.verify_input`` passes and the AR loop sees the same
        # 2-step self-forcing schedule on every chunk. The stashed scheduler is
        # the same object used on chunk 0, so its denoising_step_list is valid.
        scheduler = rc["scheduler"]
        device = get_local_torch_device()
        batch.prompt_embeds = [rc["text_embeds"]]
        batch.negative_prompt_embeds = None
        batch.image_embeds = []
        batch.do_classifier_free_guidance = False
        batch.scheduler = scheduler
        batch.timesteps = scheduler.denoising_step_list.to(device)
        batch.sigmas = scheduler.denoising_sigmas.tolist()
        batch.num_inference_steps = int(scheduler.denoising_step_list.shape[0])
        batch.guidance_scale = 1.0
        batch.eta = 0.0
        batch.generator = rc["generator"]
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

        # --- realtime session path ----------------------------------------- #
        # One forward() = one AR chunk, driven by ``batch.block_idx``. State
        # (KV caches, rope, text embeds, masks, cross-attn KV, scheduler,
        # generator, hdmap cache) persists in ``RealtimeCausalDiTState`` across
        # calls. The per-chunk math is IDENTICAL to the offline loop body; only
        # the loop boundary changes (one iteration per call) plus state
        # persistence + per-chunk streaming decode. The offline path below is
        # bit-identical and untouched.
        if _is_realtime(batch):
            return self._realtime_denoise_forward(batch, server_args)

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

        # Eager DiT call binding the loop-invariant args (text/caches/
        # cross_attn_kv/view_indices); the per-call-varying tensors (noisy,
        # timestep, cond_mask, rope, hdmap) are passed positionally.
        def _dit_call(hidden_states, timestep, cond_mask_t, rope_t, hdmap_t):
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

        # FP8 modes (Phase 1): ``weight_only_fp8`` dequantizes pre-quantized FP8
        # weights to bf16 and runs the standard eager PyTorch DiT; ``disabled``
        # runs the raw bf16 checkpoint. The native FP8 DiT
        # (optimized_dit_forward) was removed in Phase 1; a PyTorch-native
        # ``fp8_compute`` mode may be added in Phase 2. ``auto``/``required`` are
        # accepted as inert back-compat aliases (mapped in
        # ``OmniDreamsPipelineConfig.__post_init__``).
        mode = getattr(config, "native_dit_acceleration", "disabled")

        if mode == "weight_only_fp8":
            # ---- Weight-only FP8: dequantize FP8→bf16, use eager PyTorch path ----
            # Resolve fp8_prepared_path
            model_path = server_args.model_path
            fp8_prepared_path = getattr(config, "native_dit_fp8_prepared_path", None)
            if fp8_prepared_path is None:
                if os.path.isfile(model_path):
                    ckpt_dir = os.path.dirname(model_path)
                else:
                    ckpt_dir = model_path
                fp8_prepared_path = os.path.join(ckpt_dir, "omnidreams_fp8_dit.pt")
            # Cache: skip reload if weights already dequantized on a prior call.
            already_loaded = getattr(
                self.transformer, "_weight_only_fp8_applied", False
            )
            if already_loaded:
                logger.debug(
                    "OmniDreams: weight_only_fp8 weights already loaded, skipping."
                )
            elif fp8_prepared_path and os.path.exists(fp8_prepared_path):
                from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import (
                    dequantize_fp8_weights_to_bf16,
                )

                payload = torch.load(
                    fp8_prepared_path, map_location="cpu", weights_only=True
                )
                bf16_weights = dequantize_fp8_weights_to_bf16(payload["weights"])
                del payload  # free the 5.7GB FP8 dict immediately
                # Filter to only keys the model actually has.
                model_keys = set(self.transformer.state_dict().keys())
                matched = {k: v for k, v in bf16_weights.items() if k in model_keys}
                del bf16_weights
                device = next(self.transformer.parameters()).device
                self.transformer.load_state_dict(
                    {k: v.to(device=device) for k, v in matched.items()},
                    strict=False,
                )
                n = len(matched)
                del matched
                self.transformer._weight_only_fp8_applied = True
                logger.info(
                    "OmniDreams: loaded dequantized FP8 weights into DiT "
                    "(weight_only_fp8 mode, %d keys). Caching for reuse.",
                    n,
                )
            else:
                logger.warning(
                    "OmniDreams: weight_only_fp8 mode but FP8 prepared weights "
                    "not found at %s; using raw bf16 checkpoint.",
                    fp8_prepared_path,
                )

        elif mode == "fp8_compute":
            # ---- Phase 2: FP8-compute linears (torch._scaled_mm) ----
            # Swap the DiT linears to FP8-compute in place (post-load). On non-FP8
            # HW (CPU) install_fp8_compute_on_dit is a no-op -> eager bf16. The AR
            # loop below runs unchanged; the swapped quant_method/linears make the
            # matmuls FP8-compute. Idempotent (guarded by _fp8_compute_applied).
            from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import (
                install_fp8_compute_on_dit,
            )

            installed = install_fp8_compute_on_dit(self.transformer)
            if installed:
                logger.info("OmniDreams: fp8_compute active.")
            elif device.type == "cuda":
                logger.info(
                    "OmniDreams: fp8_compute requested but unavailable; eager bf16."
                )

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
            # Eager/CUDA-graph paths consume the [L, D] cos|sin cache (shift_t).
            rope_cos_sin = rope.shift_t(chunk_idx)
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
                ).to(
                    dit_dtype
                )  # [B,16,len_t,h,w]
                hdmap_chunk = self.transformer.patchify(chunk_latent)
            elif st["hdmap_tokens"] is not None:
                hdmap_chunk = st["hdmap_tokens"][chunk_idx].to(
                    device=device, dtype=dit_dtype
                )
            else:
                hdmap_chunk = hdmap_zero
            pin = is_first and image_full is not None

            # Roll the per-block KV window before the forward.
            for c in caches:
                c.before_update(chunk_idx)

            def predict_flow(noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                # The first-frame ``pin`` injection stays eager so ``image_full``
                # is not baked in. pin is only true on chunk 0.
                if pin:
                    noisy = noisy * (1.0 - inject_mask) + image_full * inject_mask
                return _dit_call(noisy, t, cond_mask, rope_cos_sin, hdmap_chunk)

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
            # side effect.
            _dit_call(ctx_latent, ctx_noise_t, cond_mask, rope_cos_sin, hdmap_chunk)

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

    # ------------------------------------------------------------------ #
    # Realtime (streaming) path                                          #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _realtime_denoise_forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Process exactly one AR chunk per call using persistent session state.

        Numerical parity invariant: the per-chunk body (BlockKVCache
        before/after_update, shift_t RoPE, 2-step self-forcing denoise via
        ``scheduler.sample``, context-noise re-forward at ``ctx_noise_t``) is
        IDENTICAL to the offline loop body. Only the loop boundary differs
        (one iteration per call) plus state persistence in
        ``RealtimeCausalDiTState`` and per-chunk streaming VAE decode.
        """
        config = server_args.pipeline_config
        device = get_local_torch_device()
        dit_dtype = PRECISION_TO_TYPE[config.dit_precision]
        arch = config.dit_config.arch_config

        # Hold the DiT (and HD-map encoder, when present) resident for this
        # chunk's forward. Same pattern as the offline path.
        residency_manager = self._component_residency_manager
        transformer_use = None
        encoder_use = None
        if residency_manager is not None:
            transformer_use = self._declared_component_use(component_name="transformer")
            residency_manager.begin_use(transformer_use, self.transformer)
            if self.encoder is not None:
                encoder_use = self._declared_component_use(component_name="encoder")
                residency_manager.begin_use(encoder_use, self.encoder)

        try:
            return self._realtime_denoise_chunk(
                batch, server_args, device, dit_dtype, arch
            )
        finally:
            if residency_manager is not None and transformer_use is not None:
                residency_manager.end_use(transformer_use, self.transformer)
            if residency_manager is not None and encoder_use is not None:
                residency_manager.end_use(encoder_use, self.encoder)

    def _realtime_denoise_chunk(
        self,
        batch: Req,
        server_args: ServerArgs,
        device: torch.device,
        dit_dtype: torch.dtype,
        arch,
    ) -> Req:
        """Run one AR chunk + streaming decode for realtime mode."""
        config = server_args.pipeline_config
        cache_state = batch.session.get_or_create_state(RealtimeCausalDiTState)
        rc = cache_state.runtime_cache
        st = batch.extra["omnidreams"]

        hp = st["hp"]
        wp = st["wp"]
        len_t = st["len_t"]
        tokens_per_frame = st["tokens_per_frame"]
        chunk_tokens = st["chunk_tokens"]
        context_noise = st["context_noise"]
        block_idx = batch.block_idx
        # chunk_idx is the AR chunk index (cache_state.chunk_idx == block_idx
        # by construction; the before-stage inits to 0 and we increment after
        # each chunk).
        chunk_idx = cache_state.chunk_idx

        head_dim = rc["arch_constants"]["head_dim"]
        in_d = rc["arch_constants"]["in_d"]
        hdmap_d = rc["arch_constants"]["hdmap_d"]
        mask_d = rc["arch_constants"]["mask_d"]

        scheduler = rc["scheduler"].to(device)
        text = rc["text_embeds"].to(device=device, dtype=dit_dtype)
        B = text.shape[0]
        gen = rc["generator"]
        if gen is not None and gen.device != device:
            gen = torch.Generator(device=device).manual_seed(gen.initial_seed())
            rc["generator"] = gen

        # ---- lazy one-shot assembly on chunk 0 (needs device/dit_dtype) ---- #
        if cache_state.kv_cache is None or rc["rope"] is None:
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
            rc["rope"] = rope
            caches = self.transformer.init_kv_caches(
                batch_size=B,
                chunk_tokens=chunk_tokens,
                window_tokens=st["window_size_t"] * tokens_per_frame,
                sink_tokens=st["sink_size_t"] * tokens_per_frame,
                device=device,
                dtype=dit_dtype,
            )
            cache_state.kv_cache = caches

            frame0 = tokens_per_frame
            image_token = rc["image_token"]
            if image_token is not None:
                image_full = torch.zeros(
                    B, chunk_tokens, in_d, device=device, dtype=dit_dtype
                )
                image_full[:, :frame0, :] = image_token.to(
                    device=device, dtype=dit_dtype
                )
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
            rc["image_full"] = image_full
            rc["inject_mask"] = inject_mask
            rc["cond_mask_c0"] = cond_mask_c0
            rc["cond_mask_zero"] = cond_mask_zero
            rc["hdmap_zero"] = hdmap_zero

            # Precompute cross-attn K/V once (text context is static).
            rc["cross_attn_kv"] = self.transformer.precompute_cross_attn_kv(
                self.transformer.crossattn_proj(text)
            )

            # Initialize the persistent HD-map streaming encode cache (per-frame
            # video path only) once per session.
            if st.get("hdmap_pixel") is not None and hasattr(
                self.encoder, "initialize_ar_encode_cache"
            ):
                rc["hdmap_encode_cache"] = self.encoder.initialize_ar_encode_cache()

            # FP8 weight-only mode: dequantize once on the first chunk (same
            # logic as the offline path; the result persists on self.transformer
            # across calls via the _weight_only_fp8_applied flag).
            mode = getattr(config, "native_dit_acceleration", "disabled")
            if mode == "weight_only_fp8":
                self._maybe_load_weight_only_fp8(batch, server_args)
            elif mode == "fp8_compute":
                self._maybe_install_fp8_compute(config, device)

        rope = rc["rope"]
        caches = cache_state.kv_cache
        image_full = rc["image_full"]
        inject_mask = rc["inject_mask"]
        cond_mask_c0 = rc["cond_mask_c0"]
        cond_mask_zero = rc["cond_mask_zero"]
        hdmap_zero = rc["hdmap_zero"]
        cross_attn_kv = rc["cross_attn_kv"]
        hdmap_encode_cache = rc["hdmap_encode_cache"]

        # View indices (optional, multi-view cross-view attn).
        view_count = int(getattr(batch, "num_views", 1) or 1)
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
            )

        ctx_noise_t = torch.tensor(context_noise, device=device, dtype=dit_dtype)

        # Eager DiT call binding the session-persistent args.
        def _dit_call(hidden_states, timestep, cond_mask_t, rope_t, hdmap_t):
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

        # ---- per-chunk body (IDENTICAL math to the offline loop body) ---- #
        rope_cos_sin = rope.shift_t(chunk_idx)
        is_first = chunk_idx == 0
        cond_mask = cond_mask_c0 if is_first else cond_mask_zero

        # HD-map conditioning for this chunk. Resolution order (mirrors the
        # offline loop + closed-loop override):
        #  1. ``hdmap_pixel_chunk``: closed-loop per-chunk pixels from
        #     ``condition_inputs["hdmap"]`` (set by the before-stage for
        #     block_idx>0). VAE-encoded here (per-chunk slice).
        #  2. ``hdmap_pixel``: stashed full clip -> slice this chunk's frames
        #     and VAE-encode (open-loop per-frame video path, same as offline).
        #  3. ``hdmap_tokens``: precomputed per-chunk tokens (single-image
        #     broadcast fallback).
        #  4. both None -> zeros (HDMap disabled).
        hdmap_pixel_chunk = st.get("hdmap_pixel_chunk")
        if hdmap_pixel_chunk is not None:
            # Closed-loop pixels arrive as fp32 from the realtime adapter's
            # _decode_hdmap_chunk; the offline path pre-casts to the VAE dtype in
            # _preprocess_pixels, so cast here to match the encoder's conv dtype.
            chunk_clip = hdmap_pixel_chunk.to(
                device=device, dtype=next(self.encoder.parameters()).dtype
            )
            chunk_latent = _vae_encode_normalized(
                chunk_clip,
                self.encoder,
                cache=hdmap_encode_cache,
                is_first_chunk=is_first,
            ).to(dit_dtype)
            hdmap_chunk = self.transformer.patchify(chunk_latent)
        elif st["hdmap_pixel"] is not None:
            s, e = _hdmap_chunk_pixel_bounds(chunk_idx, len_t)
            chunk_clip = st["hdmap_pixel"][:, :, s:e].to(device=device)
            chunk_latent = _vae_encode_normalized(
                chunk_clip,
                self.encoder,
                cache=hdmap_encode_cache,
                is_first_chunk=is_first,
            ).to(dit_dtype)
            hdmap_chunk = self.transformer.patchify(chunk_latent)
        elif st["hdmap_tokens"] is not None:
            hdmap_chunk = st["hdmap_tokens"][chunk_idx].to(
                device=device, dtype=dit_dtype
            )
        else:
            hdmap_chunk = hdmap_zero

        pin = is_first and image_full is not None

        # Roll the per-block KV window before the forward.
        for c in caches:
            c.before_update(chunk_idx)

        def predict_flow(noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            if pin:
                noisy = noisy * (1.0 - inject_mask) + image_full * inject_mask
            return _dit_call(noisy, t, cond_mask, rope_cos_sin, hdmap_chunk)

        noise = torch.empty(
            B, chunk_tokens, in_d, device=device, dtype=dit_dtype
        ).normal_(generator=gen)
        clean = scheduler.sample(noise, predict_flow=predict_flow, rng=gen)
        if pin:
            clean = clean * (1.0 - inject_mask) + image_full * inject_mask

        # Authoritative cache write: re-forward the CLEAN chunk at the
        # context-noise timestep (identical to the offline loop body).
        ctx_latent = scheduler.add_noise(clean, ctx_noise_t, rng=gen)
        if pin:
            ctx_latent = ctx_latent * (1.0 - inject_mask) + image_full * inject_mask
        _dit_call(ctx_latent, ctx_noise_t, cond_mask, rope_cos_sin, hdmap_chunk)

        for c in caches:
            c.after_update(chunk_idx)

        # [B, L, out*pdim] -> [B, out, len_t, h, w].
        chunk_latent_btchw = self.transformer.unpatchify(clean, len_t, hp, wp)

        # Advance the persistent chunk index for the next call.
        cache_state.chunk_idx += 1

        # This chunk's latent (for any downstream that reads batch.latents).
        batch.latents = chunk_latent_btchw
        _log_omnidreams_stats("realtime_chunk_latents", batch.latents)

        # Per-chunk streaming VAE decode is owned by the downstream
        # ``omnidreams_decoding`` stage (``OmniDreamsCausalDecodingStage``),
        # which is the authoritative path whose output ``gpu_worker`` ships over
        # the WebSocket. Do NOT decode here: a second decode would be discarded
        # (its ``batch.raw_frame_batches`` is overwritten by the decoding stage's
        # ``output_batch.output``) and would double the VAE cost per chunk.

        return batch

    def _maybe_load_weight_only_fp8(self, batch: Req, server_args: ServerArgs) -> None:
        """Dequantize FP8 weights into the DiT once (weight_only_fp8 mode).

        Extracted from the offline forward; idempotent via the
        ``_weight_only_fp8_applied`` flag on the transformer.
        """
        already_loaded = getattr(self.transformer, "_weight_only_fp8_applied", False)
        if already_loaded:
            return
        config = server_args.pipeline_config
        model_path = server_args.model_path
        fp8_prepared_path = getattr(config, "native_dit_fp8_prepared_path", None)
        if fp8_prepared_path is None:
            if os.path.isfile(model_path):
                ckpt_dir = os.path.dirname(model_path)
            else:
                ckpt_dir = model_path
            fp8_prepared_path = os.path.join(ckpt_dir, "omnidreams_fp8_dit.pt")
        if fp8_prepared_path and os.path.exists(fp8_prepared_path):
            from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import (
                dequantize_fp8_weights_to_bf16,
            )

            payload = torch.load(
                fp8_prepared_path, map_location="cpu", weights_only=True
            )
            bf16_weights = dequantize_fp8_weights_to_bf16(payload["weights"])
            del payload
            model_keys = set(self.transformer.state_dict().keys())
            matched = {k: v for k, v in bf16_weights.items() if k in model_keys}
            del bf16_weights
            device = next(self.transformer.parameters()).device
            self.transformer.load_state_dict(
                {k: v.to(device=device) for k, v in matched.items()},
                strict=False,
            )
            n = len(matched)
            del matched
            self.transformer._weight_only_fp8_applied = True
            logger.info(
                "OmniDreams: loaded dequantized FP8 weights into DiT "
                "(weight_only_fp8 mode, %d keys). Caching for reuse.",
                n,
            )
        else:
            logger.warning(
                "OmniDreams: weight_only_fp8 mode but FP8 prepared weights "
                "not found at %s; using raw bf16 checkpoint.",
                fp8_prepared_path,
            )

    def _maybe_install_fp8_compute(self, config, device) -> None:
        """Install FP8-compute linears on the DiT once (fp8_compute mode).

        Extracted from the offline forward; idempotent.
        """
        if getattr(self.transformer, "_fp8_compute_applied", False):
            return
        from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import (
            install_fp8_compute_on_dit,
        )

        installed = install_fp8_compute_on_dit(self.transformer)
        if installed:
            logger.info("OmniDreams: fp8_compute active.")
        elif device.type == "cuda":
            logger.info(
                "OmniDreams: fp8_compute requested but unavailable; eager bf16."
            )

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


class OmniDreamsCausalDecodingStage(CausalVaeDecodingStage):
    """Realtime-aware Wan VAE decode for OmniDreams.

    Reuses :class:`CausalVaeDecodingStage` verbatim: offline (no session) falls
    through to the base :class:`DecodingStage` single-pass decode; realtime
    (session present) uses the persistent per-chunk causal streaming decode with
    a conv ``_feat_map`` carried across AR chunks, so a steady chunk of ``len_t``
    latents emits the full ``len_t * temporal_compression`` pixel frames.

    This subclass exists only to be the OmniDreams decoding stage identity; the
    behaviour lives in the base class. The correctness of steady-chunk frame
    counts depends on the decoder's persistent ``_feat_map`` surviving across
    chunks -- see ``WanVAE.clear_encode_cache``: because OmniDreams shares one
    cached WanVAE instance for hdmap-encode and latent-decode, ``encode()`` must
    NOT run a full ``clear_cache()`` (which would wipe the live decode cache and
    collapse each steady chunk to the causal-anchor 1-frame path).
    """


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
