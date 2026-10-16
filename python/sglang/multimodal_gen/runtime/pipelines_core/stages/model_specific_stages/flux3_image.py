# SPDX-License-Identifier: Apache-2.0
"""Served stages for the FLUX.3 Image Klein packed pipeline."""

from __future__ import annotations

import hashlib
import math
import os
from contextlib import contextmanager, nullcontext
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.distributed.cfg_parallel_utils import (
    run_cfg_parallel,
)
from sglang.multimodal_gen.runtime.distributed.cfg_policy import (
    CFGBranch,
    CFGPolicy,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context,
    set_forward_context,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.dits.flux3_image import TEXT, Packed, pack
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import use_vae_fast_path
from sglang.multimodal_gen.runtime.pipelines.flux3_image_package import (
    Flux3ImagePackageConfig,
)
from sglang.multimodal_gen.runtime.pipelines_core.request_utils import (
    normalize_output_seeds,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.condition_encoding import (
    ConditionEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import precision_to_dtype
from sglang.srt.environ import envs

logger = init_logger(__name__)

DOWNSAMPLING_RATIO = 16

MAX_REFERENCES = 10

MATCH_MAX_PIXELS = 4 * 1024 * 1024  # BFL limit: one reference, padded canvas included

MULTI_REFERENCE_MAX_PIXELS = 1024 * 1024  # BFL limit: each of several references

MIN_SIDE = 64  # BFL limit: a reference canvas side

MAX_SIDE = 4096  # BFL limit: a reference canvas side

_CFG_HIDDEN = 3072

_CFG_ACT_STACK = 32


def latent_shape(height: int, width: int) -> tuple[int, int]:
    if height % DOWNSAMPLING_RATIO or width % DOWNSAMPLING_RATIO:
        raise ValueError(
            f"width and height must be multiples of {DOWNSAMPLING_RATIO}, "
            f"got {width}x{height}"
        )
    return height // DOWNSAMPLING_RATIO, width // DOWNSAMPLING_RATIO


def latent_ids(height: int, width: int, device: torch.device | str = "cpu") -> Tensor:
    rows, columns = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    zeros = torch.zeros_like(rows)
    return torch.stack([zeros, rows, columns, zeros], dim=-1).reshape(height * width, 4)


def scaled_ids(ids: Tensor, scale: float) -> Tensor:
    ids = ids.float()
    ids[:, 1:3] *= scale
    return ids


def noise_tokens(
    height: int,
    width: int,
    seed: int,
    config: Flux3ImagePackageConfig,
    device: torch.device,
) -> tuple[tuple[Tensor, Tensor], Tensor]:
    latent_height, latent_width = latent_shape(height, width)
    noise = torch.randn(
        (
            latent_height * latent_width,
            config.dit.in_channels["image"],
        ),
        generator=torch.Generator().manual_seed(seed),
    )
    # bf16 values in fp32 storage: as in the reference, the Euler steps
    # accumulate in fp32 (the DiT casts its input to bf16).
    noise = noise.to(torch.bfloat16).to(device=device, dtype=torch.float32)
    ids = latent_ids(latent_height, latent_width, device)
    return (noise, scaled_ids(ids, config.dit.image_rope_scale)), ids


def timeshift(alpha: float, timestep: Tensor) -> Tensor:
    return alpha * timestep / (1.0 + (alpha - 1.0) * timestep)


def build_timesteps(steps: int, alpha: float) -> Tensor:
    return timeshift(alpha, torch.linspace(1.0, 0.0, steps + 1))


def resolve_sequential_cfg(
    packed: Packed,
    guidance: float,
    hidden_size: int = _CFG_HIDDEN,
) -> bool:
    """Choose two batch-1 forwards when packed CFG activations may not fit."""
    if guidance <= 1.0:
        return False
    if not torch.cuda.is_available():
        return False
    tokens_per_sample = [
        sum(
            packed.mode_indices[mode][sample][1] - packed.mode_indices[mode][sample][0]
            for mode in packed.mode_indices
        )
        for sample in range(packed.num_samples)
    ]
    free, _ = torch.cuda.mem_get_info(packed.data["image"].device)
    activation = (
        packed.num_samples
        * max(tokens_per_sample)
        * int(hidden_size)
        * torch.bfloat16.itemsize
    )
    return activation * _CFG_ACT_STACK >= free


@contextmanager
def _dit_step(step_index: int, callback: Callable[[], None] | None = None):
    try:
        forward_batch = get_forward_context().forward_batch
    except AssertionError:
        forward_batch = None
    with set_forward_context(
        current_timestep=step_index,
        attn_metadata=None,
        forward_batch=forward_batch,
    ):
        yield
    if callback is not None:
        callback()


def denoise_packed(
    dit,
    packed: Packed,
    schedule: Tensor,
    guidance: float,
    *,
    step_callback: Callable[[], None] | None = None,
    mode: str = "image",
    sequential_cfg: bool = False,
) -> Tensor:
    if packed.num_samples % 2:
        raise ValueError(
            "packed CFG requires an unconditional and conditional sample per output"
        )
    device_type = packed.data[mode].device.type
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=device_type == "cuda",
    ):
        for step in range(len(schedule) - 1):
            timestep = schedule[step].item()
            delta = (schedule[step + 1] - schedule[step]).item()
            packed.timesteps[mode].fill_(timestep)
            with _dit_step(step, step_callback):
                if sequential_cfg:
                    half = packed.num_samples // 2
                    unconditional = dit(packed=packed.slice_samples(0, half))[mode]
                    conditional = dit(
                        packed=packed.slice_samples(half, packed.num_samples)
                    )[mode]
                else:
                    unconditional, conditional = dit(packed=packed)[mode].chunk(2)
            velocity = unconditional + guidance * (conditional - unconditional)
            image = packed.data[mode].chunk(2)[0] + delta * velocity
            packed.data[mode] = torch.cat((image, image))
    return packed.data[mode].chunk(2)[0]


def denoise_flow_map(
    dit,
    packed: Packed,
    schedule: Tensor,
    *,
    step_callback: Callable[[], None] | None = None,
    mode: str = "image",
) -> Tensor:
    """Flow-map sampling: one forward per step jumps from ``t`` to the next time.

    The DiT is a step-distilled flow map, so there is no CFG pair.
    """
    device = packed.data[mode].device
    packed.timesteps_ref = torch.zeros(
        packed.num_samples, device=device, dtype=torch.float32
    )
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        for step in range(len(schedule) - 1):
            timestep = schedule[step].item()
            delta = (schedule[step + 1] - schedule[step]).item()
            packed.timesteps[mode].fill_(timestep)
            packed.timesteps_ref.fill_(schedule[step + 1].item())
            with _dit_step(step, step_callback):
                velocity = dit(packed=packed)[mode]
            packed.data[mode] = packed.data[mode] + delta * velocity
    return packed.data[mode]


def vae_encode(vae, image: Tensor) -> tuple[Tensor, Tensor]:
    latent = vae.encode(image.unsqueeze(0).to(next(vae.parameters()).dtype))
    height, width = latent.shape[2:]
    tokens = latent[0].permute(1, 2, 0).reshape(height * width, latent.shape[1])
    return tokens, latent_ids(height, width, latent.device)


def vae_decode(vae, tokens: Tensor, ids: Tensor, *, batched: bool = False) -> Tensor:
    height = int(ids[:, 1].max()) + 1
    width = int(ids[:, 2].max()) + 1
    latent = (
        tokens.float()
        .reshape(-1, height, width, tokens.shape[-1])
        .permute(0, 3, 1, 2)
        .to(next(vae.parameters()).dtype)
    )
    decoded = vae.decode(latent)
    return decoded if batched else decoded[0]


def encode_references(
    vae,
    references: list[Tensor],
    rope_scale: float,
    device: torch.device,
) -> tuple[Tensor, Tensor] | None:
    if not references:
        return None
    tokens = []
    ids = []
    for index, reference in enumerate(references):
        reference_tokens, reference_ids = vae_encode(vae, reference.to(device))
        reference_ids = scaled_ids(reference_ids, rope_scale)
        reference_ids[:, 3] = -(1 + index)
        tokens.append(reference_tokens.to(torch.bfloat16))
        ids.append(reference_ids)
    return torch.cat(tokens), torch.cat(ids)


def _ceil16(value: int) -> int:
    return -(-value // DOWNSAMPLING_RATIO) * DOWNSAMPLING_RATIO


def reference_canvas(
    width: int, height: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """(fitted size, canvas) of a single reference: the canvas is the fitted size
    rounded up to 16, at most ``MATCH_MAX_PIXELS`` with sides in 64..4096."""
    scale = min(
        1.0,
        math.sqrt(MATCH_MAX_PIXELS / (width * height)),
        MAX_SIDE / max(width, height),
    )
    while True:
        fit = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = tuple(min(MAX_SIDE, max(MIN_SIDE, _ceil16(v))) for v in fit)
        if canvas[0] * canvas[1] <= MATCH_MAX_PIXELS:
            return fit, canvas
        scale *= 0.995


def prepare_reference(
    image: Image.Image,
) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    """A single reference, fitted by :func:`reference_canvas` and reflect-padded
    to its canvas, centred. Also returns the ``(left, top, right, bottom)`` box of
    the fitted image inside a padded canvas, which the output is cropped to."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    fit, (width, height) = reference_canvas(image.width, image.height)
    if fit != image.size:
        image = image.resize(fit, Image.Resampling.LANCZOS)
    if (width, height) == image.size:
        return image, None
    left = (width - image.width) // 2
    top = (height - image.height) // 2
    padding = (
        (top, height - image.height - top),
        (left, width - image.width - left),
        (0, 0),
    )
    # reflect needs a pad shorter than the side
    mode = "reflect" if min(image.size) > DOWNSAMPLING_RATIO else "symmetric"
    padded = Image.fromarray(np.pad(np.asarray(image), padding, mode=mode))
    return padded, (left, top, left + image.width, top + image.height)


def prepare_multi_reference(image: Image.Image) -> Image.Image:
    """Each reference of a multi-reference edit: at most 1 MP (sides floored),
    centre-cropped to the 16-px grid."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width * height > MULTI_REFERENCE_MAX_PIXELS:
        scale = math.sqrt(MULTI_REFERENCE_MAX_PIXELS / (width * height))
        width, height = int(width * scale), int(height * scale)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    grid_width = width // DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO
    grid_height = height // DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO
    left = (width - grid_width) // 2
    top = (height - grid_height) // 2
    return image.crop((left, top, left + grid_width, top + grid_height))


def _open_reference(path: str | bytes | Image.Image) -> Image.Image:
    from sglang.srt.utils.common import get_image_bytes

    if isinstance(path, Image.Image):
        return path
    if isinstance(path, str) and os.path.isfile(path):
        return Image.open(path)
    return Image.open(BytesIO(get_image_bytes(path)))


def _to_tensor(image: Image.Image) -> Tensor:
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0


def load_references(
    paths: list[str | bytes | Image.Image],
) -> tuple[list[Tensor], tuple[int, int], tuple[int, int, int, int] | None]:
    """References in ``[-1, 1]``, the first one's ``(width, height)`` (the
    default output size) and, for a single padded reference, its crop box."""
    if not 1 <= len(paths) <= MAX_REFERENCES:
        raise ValueError(
            f"FLUX.3 takes 1 to {MAX_REFERENCES} reference images, got {len(paths)}"
        )
    images = [_open_reference(path) for path in paths]
    if len(images) == 1:
        image, crop = prepare_reference(images[0])
        prepared = [image]
    else:
        prepared = [prepare_multi_reference(image) for image in images]
        crop = None
    return [_to_tensor(image) for image in prepared], prepared[0].size, crop


def output_size(
    width: int | None,
    height: int | None,
    reference_size: tuple[int, int] | None,
    crop: tuple[int, int, int, int] | None,
    default: tuple[int, int],
) -> tuple[int, int, tuple[int, int, int, int] | None]:
    """The reference's sizing rule: an unset side takes the first reference's
    (else ``default``); a padded single reference is cropped back only when
    neither side was set."""
    fallback = reference_size or default
    keep_crop = crop if width is None and height is None else None
    return width or fallback[0], height or fallback[1], keep_crop


def crop_output(image: Tensor, crop: tuple[int, int, int, int] | None) -> Tensor:
    """Drop the reflect padding of a single reference from a ``(C, H, W)`` image."""
    if crop is None:
        return image
    left, top, right, bottom = crop
    return image[..., top:bottom, left:right]


def validate_generation(
    mode: str,
    *,
    prompt: str | list[str] | None,
    image_path: str | bytes | list | None,
) -> bool:
    if mode not in {"t2i", "i2i"}:
        raise ValueError(f"FLUX.3 mode must be 't2i' or 'i2i', got {mode!r}")
    if isinstance(prompt, (list, tuple)) and len(prompt) != 1:
        raise ValueError("FLUX.3 processes exactly one prompt per request")
    if mode == "t2i" and image_path is not None:
        raise ValueError("FLUX.3 mode 't2i' does not accept conditioning input")
    if mode == "i2i" and image_path is None:
        raise ValueError("FLUX.3 mode 'i2i' requires image_path")
    return mode == "i2i"


def _flux3_prims_ts_autotune_context(server_args: ServerArgs):
    """Autotune FlashInfer PrimsTS GEMMs during FLUX.3 request warmup."""
    from flashinfer.autotuner import autotune

    cache_root = Path(envs.SGLANG_CACHE_DIR.get())
    model_key = hashlib.sha256(str(server_args.model_path).encode()).hexdigest()[:16]
    rank = int(os.environ.get("RANK", "0"))
    device = torch.cuda.get_device_name(torch.cuda.current_device()).replace(" ", "_")
    cache_path = (
        cache_root
        / "flashinfer"
        / "autotune"
        / "flux3_prims_ts"
        / device
        / model_key
        / f"rank{rank}.json"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    raw_buckets = envs.SGLANG_OPT_FLUX3_PRIMS_TS_TUNING_BUCKETS.get()
    buckets = tuple(int(value) for value in raw_buckets) or None
    return autotune(True, cache=str(cache_path), tuning_buckets=buckets, round_up=True)


def _mode(batch: Req, pipeline_config) -> str:
    # One server serves both modes: a request with references is an edit.
    requested = batch.flux3_mode
    mode = requested or ("i2i" if batch.image_path else "t2i")
    if mode not in pipeline_config.served_modes():
        raise ValueError(
            f"this server accepts FLUX.3 modes "
            f"{sorted(pipeline_config.served_modes())}; got {mode!r}"
        )
    return mode


def _image_paths(batch: Req) -> list[str] | None:
    """The request's reference image paths, or ``None`` without any."""
    value = batch.image_path
    if value is None:
        return None
    paths = list(value) if isinstance(value, (list, tuple)) else [value]
    if not paths:
        return None
    if not all(isinstance(path, str) and path for path in paths):
        raise ValueError("FLUX.3 input image paths must be non-empty strings")
    return paths


def _steps(batch: Req, package_config) -> int:
    """The request's step count, else the package's (16 for the flow map)."""
    return int(batch.num_inference_steps or package_config.sampler.steps)


def _cfg_scale(batch: Req, package_config) -> float | None:
    """The request's CFG scale, else the package's; ``None`` for a flow map."""
    value = batch.true_cfg_scale
    if value is None:
        value = batch.guidance_scale
    if package_config.dit.flow_map:
        # As in the reference, any scale other than 1 asks for CFG.
        if value is not None and float(value) != 1.0:
            raise ValueError(
                "the FLUX.3 flow-map (distilled) DiT samples without CFG; "
                "drop guidance_scale"
            )
        return None
    return float(value) if value is not None else package_config.sampler.guidance


class Flux3ImageTextEncodeStage(PipelineStage):
    def __init__(self, text_encoder, package_config) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.package_config = package_config

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "text_encoder",
                phase="text_encode",
                memory_intensive=True,
                skip_empty_cache_on_release=True,
            )
        ]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        return self.run_grouped_requests([batch], server_args)[0]

    @torch.no_grad()
    def run_grouped_requests(self, batches, server_args):
        prompts = []
        for batch in batches:
            validate_generation(
                _mode(batch, server_args.pipeline_config),
                prompt=batch.prompt,
                image_path=_image_paths(batch),
            )
            prompt = (
                batch.prompt[0]
                if isinstance(batch.prompt, (list, tuple))
                else batch.prompt
            )
            if self.package_config.dit.flow_map:
                if batch.negative_prompt:
                    raise ValueError(
                        "the FLUX.3 flow-map DiT samples without CFG; drop negative_prompt"
                    )
                prompts.append(prompt or "")
            else:
                prompts.extend([batch.negative_prompt or "", prompt or ""])
        with self.use_declared_component(
            component_name="text_encoder",
            module=self.text_encoder,
            phase="text_encode",
        ):
            with torch.inference_mode():
                encoded = self.text_encoder.encode_batch(prompts)
        branches = 1 if self.package_config.dit.flow_map else 2
        for index, batch in enumerate(batches):
            batch.flux3_text = encoded[index * branches : (index + 1) * branches]
        return batches


class Flux3ImageConditionEncodeStage(ConditionEncodingStage):
    def __init__(self, vae, package_config) -> None:
        super().__init__()
        self.vae = vae
        self.package_config = package_config

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "vae",
                phase="condition_encode",
                target_dtype=precision_to_dtype(self.package_config.vae.dtype),
                memory_intensive=True,
                skip_empty_cache_on_release=True,
            )
        ]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        pipeline_config = server_args.pipeline_config
        mode = _mode(batch, pipeline_config)
        # The padding of a single reference that the decoded image drops.
        batch.flux3_crop = None
        if mode == "t2i":
            batch.flux3_condition = None
            return batch
        references, reference_size, crop = load_references(_image_paths(batch))
        # As in the reference, an unset side takes the first reference's.
        explicit = batch.sampling_params._explicit_fields
        width, height, batch.flux3_crop = output_size(
            batch.width if "width" in explicit else None,
            batch.height if "height" in explicit else None,
            reference_size,
            crop,
            tuple(self.package_config.sampler.size),
        )
        batch.width, batch.height = width, height
        with (
            self.use_declared_component(
                component_name="vae",
                module=self.vae,
                phase="condition_encode",
            ),
            use_vae_fast_path(self.vae, batch.quality != "exact"),
        ):
            with torch.inference_mode():
                batch.flux3_condition = encode_references(
                    self.vae,
                    references,
                    self.package_config.dit.image_rope_scale,
                    get_local_torch_device(),
                )
        return batch


class _CFGParallelModel:
    """Run one packed CFG sample per rank and restore pack order."""

    def __init__(self, model):
        self.model = model
        self.policy = CFGPolicy(
            branches=[
                CFGBranch("unconditional", False, {"index": 0}),
                CFGBranch("conditional", True, {"index": 1}),
            ]
        )

    def __call__(self, *, packed: Packed):
        half = packed.num_samples // 2
        predictions = run_cfg_parallel(
            self.policy,
            lambda branch: self.model(
                packed=packed.slice_samples(
                    branch.kwargs["index"] * half, (branch.kwargs["index"] + 1) * half
                )
            )["image"],
        )
        return {"image": torch.cat(predictions)}


class Flux3ImageDenoiseStage(DenoisingStage):
    def __init__(self, transformer, package_config) -> None:
        super().__init__(transformer=transformer, scheduler=None)
        self.package_config = package_config
        # Set per request in forward, before cache-dit decides to mount.
        self._sequential_cfg_active = False

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        # The CFG pair, or the prompt alone for a flow map.
        samples = 1 if self.package_config.dit.flow_map else 2
        result.add_check(
            "flux3_text", batch.flux3_text, lambda value: len(value) == samples
        )
        result.add_check(
            "flux3_text[0]",
            batch.flux3_text[0][0],
            [V.is_tensor, V.min_dims(2)],
        )
        return result

    # Cache-DiT caches the joint single-block trunk, which runs once per step:
    # packed CFG and the flow map are one forward. Ulysses SP runs each block
    # once per sample stream, which the cached block loop cannot intercept.
    def _cache_dit_requested(self) -> bool:
        return super()._cache_dit_requested() and self._cache_dit_fits_parallelism()

    def _cache_dit_requested_for_batch(self, batch: Req) -> bool:
        if not super()._cache_dit_requested_for_batch(batch):
            return False
        if not self._cache_dit_fits_parallelism():
            return False
        if self._sequential_cfg_active:
            logger.warning(
                "Cache-DiT is skipped for this FLUX.3 request: its CFG pair runs "
                "as two forwards to fit in device memory."
            )
            return False
        return True

    def _cache_dit_has_separate_cfg(self, batch: Req) -> bool:
        return False

    def _cache_dit_joint_cfg_decision(self) -> bool:
        return True

    @staticmethod
    def _cache_dit_fits_parallelism() -> bool:
        if get_sp_world_size() == 1:
            return True
        logger.warning_once(
            "Cache-DiT is not supported with FLUX.3 Image Klein under Ulysses "
            "sequence parallelism; running without it."
        )
        return False

    def _denoise_model(self, server_args: ServerArgs):
        model = self._maybe_get_bcg_runner(self.transformer) or self.transformer
        if server_args.enable_cfg_parallel:
            model = _CFGParallelModel(model)
        return model

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "transformer",
                phase="denoise",
                target_dtype=torch.bfloat16,
                memory_intensive=True,
                release_after_use=bool(server_args.dit_layerwise_offload),
                prefetch_after_request=True,
            )
        ]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        return self._forward_batched([batch], server_args)[0]

    def run_grouped_requests(self, batches, server_args):
        return self._forward_batched(batches, server_args)

    @torch.no_grad()
    def _forward_batched(
        self, batches: list[Req], server_args: ServerArgs
    ) -> list[Req]:
        batch = batches[0]
        pipeline_config = server_args.pipeline_config
        width = int(batch.width or self.package_config.sampler.size[0])
        height = int(batch.height or self.package_config.sampler.size[1])
        steps = _steps(batch, self.package_config)
        alpha = float(pipeline_config.flux3_alpha or self.package_config.sampler.alpha)
        guidance = _cfg_scale(batch, self.package_config)
        images = [
            [
                noise_tokens(
                    int(req.height or height),
                    int(req.width or width),
                    seed,
                    self.package_config,
                    get_local_torch_device(),
                )
                for seed in normalize_output_seeds(
                    req.seed or 0, num_outputs_per_prompt=req.num_outputs_per_prompt
                )
            ]
            for req in batches
        ]
        flow_map = self.package_config.dit.flow_map
        packed = pack(
            [
                {
                    TEXT: req.flux3_text[branch],
                    "image_cond": req.flux3_condition,
                    "image": image,
                }
                for branch in range(1 if flow_map else 2)
                for req, request_images in zip(batches, images)
                for image, _ in request_images
            ],
            dict(self.package_config.dit.in_channels),
        )
        packed.rope = {
            mode: self.transformer.rope(ids) for mode, ids in packed.ids.items()
        }
        if not flow_map and batch.quality != "exact":
            # Each mode has one timestep. Project it once, then broadcast the
            # modulation; exact keeps the reference GEMM shapes and rounding.
            packed.timesteps = {
                mode: time[:1] for mode, time in packed.timesteps.items()
            }
        num_steps = max(0, steps)
        schedule = build_timesteps(steps, alpha)
        batch.timesteps = schedule
        with self.use_declared_component(
            component_name="transformer",
            module=self.transformer,
            phase="denoise",
        ):
            sequential_cfg = (
                not flow_map
                and not server_args.enable_cfg_parallel
                and resolve_sequential_cfg(packed, guidance)
            )
            self._sequential_cfg_active = sequential_cfg
            if (
                not flow_map
                and not server_args.enable_breakable_cuda_graph
                and not server_args.enable_torch_compile
                and not server_args.enable_attention_backend_autotune
                and not server_args.dit_layerwise_offload
                and not server_args.enable_cfg_parallel
                and not sequential_cfg
                and batch.skip_softmax_params is None
                and self.transformer.can_reuse_mode_blocks()
            ):
                packed.static_hidden = {}
            self._maybe_enable_cache_dit_and_torch_compile(num_steps, batch)
            with (
                torch.inference_mode(),
                set_forward_context(
                    current_timestep=0,
                    attn_metadata=None,
                    forward_batch=batch,
                ),
                self.progress_bar(total=num_steps, batch=batch) as progress,
            ):
                tune_prims_ts = (
                    batch.is_warmup and envs.SGLANG_ENABLE_FLUX3_PRIMS_TS_AUTOTUNE.get()
                )
                tune_context = (
                    _flux3_prims_ts_autotune_context(server_args)
                    if tune_prims_ts
                    else nullcontext()
                )
                with tune_context:
                    if flow_map:
                        latents = denoise_flow_map(
                            self._denoise_model(server_args),
                            packed,
                            schedule,
                            step_callback=progress.update,
                        )
                    else:
                        latents = denoise_packed(
                            self._denoise_model(server_args),
                            packed,
                            schedule,
                            guidance,
                            step_callback=progress.update,
                            sequential_cfg=sequential_cfg,
                        )
        counts = [sum(image[0].shape[0] for image, _ in group) for group in images]
        for req, latent, group in zip(batches, latents.split(counts), images):
            req.latents, req.latent_ids = latent, group[0][1]
        return batches


class Flux3ImageDecodeStage(PipelineStage):
    def __init__(self, vae, package_config) -> None:
        super().__init__()
        self.vae = vae
        self.package_config = package_config

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DECODER

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        keep = bool(server_args.pipeline_config.flux3_keep_visual_vae_resident)
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "vae",
                phase="decode",
                target_dtype=precision_to_dtype(self.package_config.vae.dtype),
                memory_intensive=True,
                keep_ready_after_warmup=True,
                preferred_ready_after_request=keep,
                keep_ready_after_request=keep,
            )
        ]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        with (
            self.use_declared_component(
                component_name="vae",
                module=self.vae,
                phase="decode",
            ),
            use_vae_fast_path(self.vae, batch.quality != "exact"),
        ):
            with torch.inference_mode():
                image = crop_output(
                    vae_decode(self.vae, batch.latents, batch.latent_ids, batched=True),
                    batch.flux3_crop,
                )
        return OutputBatch(output=self._to_display(image), metrics=batch.metrics)

    @torch.no_grad()
    def run_grouped_requests(self, batches, server_args):
        groups = {}
        for index, batch in enumerate(batches):
            groups.setdefault((batch.width, batch.height), []).append((index, batch))
        outputs = [None] * len(batches)
        with (
            self.use_declared_component(
                component_name="vae", module=self.vae, phase="decode"
            ),
            use_vae_fast_path(self.vae, batches[0].quality != "exact"),
        ):
            for group in groups.values():
                decoded = vae_decode(
                    self.vae,
                    torch.cat([req.latents for _, req in group]),
                    group[0][1].latent_ids,
                    batched=True,
                )
                decoded_groups = decoded.split(
                    [req.num_outputs_per_prompt for _, req in group]
                )
                for (index, req), images in zip(group, decoded_groups):
                    outputs[index] = OutputBatch(
                        output=list(
                            self._to_display(crop_output(images, req.flux3_crop))
                        ),
                        metrics=req.metrics,
                    )
        return outputs

    @staticmethod
    def _to_display(image: Tensor) -> Tensor:
        return ((image.clamp(-1, 1) + 1) * 0.5).reshape(-1, *image.shape[-3:])
