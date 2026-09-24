# SPDX-License-Identifier: Apache-2.0

import gc
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import constants, snapshot_download
from PIL import Image
from transformers import AutoProcessor

from sglang.multimodal_gen.runtime.models.dits.qwen_image21_mlx import (
    build_layout,
    load_transformer,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl_mlx import load_encoders
from sglang.multimodal_gen.runtime.models.vaes.qwen_image21_mlx import load_vae
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import MemorySnapshot

logger = init_logger(__name__)


SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."


def resize_images(images, width, height):
    resized = []
    area = width * height
    for image in images:
        image_width = max(
            32, round(math.sqrt(area * image.width / image.height) / 32) * 32
        )
        image_height = max(
            32, round(math.sqrt(area * image.height / image.width) / 32) * 32
        )
        resized.append(
            image.convert("RGBA").resize(
                (image_width, image_height), Image.Resampling.LANCZOS
            )
        )
    return resized


def image_position_ids(input_ids, image_grid_thw, image_token_id, merge_size):
    """Build Qwen3-VL positions for one unpadded text/image prompt."""
    positions = []
    cursor = offset = 0
    tokens = input_ids.tolist()
    for frames, height, width in image_grid_thw:
        start = tokens.index(image_token_id, cursor)
        length = start - cursor
        positions.append(np.broadcast_to(np.arange(length) + offset, (3, length)))
        grid = np.indices((frames, height // merge_size, width // merge_size)).reshape(
            3, -1
        )
        positions.append(grid + offset + length)
        offset = int(positions[-1].max()) + 1
        cursor = start + grid.shape[1]
    length = len(tokens) - cursor
    positions.append(np.broadcast_to(np.arange(length) + offset, (3, length)))
    return mx.array(np.concatenate(positions, axis=1)[:, None], dtype=mx.int32)


def encode_prompt(processor, text_encoder, vision_encoder, prompt, images):
    """Return pre-norm conditioning with each image run collapsed to one slot."""
    prefix = " ".join(
        f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>"
        for i in range(len(images))
    )
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prefix}{prompt or ' '}<|im_end|>\n<|im_start|>assistant\n"
    )
    kwargs = dict(text=[text], padding=True, padding_side="left", return_tensors="pt")
    if images:
        vision_images = []
        for image in images:
            white = Image.new("RGB", image.size, (255, 255, 255))
            white.paste(image, mask=image.getchannel("A"))
            vision_images.append(white)
        kwargs["images"] = vision_images
    inputs = processor(**kwargs)
    valid = inputs.attention_mask[0].bool()
    ids = inputs.input_ids[0, valid].numpy()
    input_ids = mx.array(ids[None])
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    grid = inputs.image_grid_thw.tolist() if images else []
    positions = image_position_ids(
        ids, grid, image_token_id, processor.image_processor.merge_size
    )
    encoder_kwargs = dict(input_ids=input_ids, position_ids=positions)
    if images:
        pixels = mx.array(inputs.pixel_values.numpy()).astype(mx.bfloat16)
        pooled, deepstack = vision_encoder(pixels, grid)
        visual_positions = mx.array(np.flatnonzero(ids == image_token_id))
        embeddings = text_encoder.embed_tokens(input_ids)
        embeddings[0, visual_positions] = pooled
        encoder_kwargs = dict(
            inputs_embeds=embeddings,
            position_ids=positions,
            visual_positions=visual_positions,
            deepstack_visual_embeds=deepstack,
        )
    hidden = text_encoder(**encoder_kwargs)
    system_message = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
    ]
    drop = len(
        processor.apply_chat_template(system_message, tokenize=True, return_dict=False)[
            0
        ]
    )
    hidden, ids = hidden[:, drop:], ids[drop:]
    image_mask = ids == image_token_id
    keep = ~image_mask
    keep[0] = True
    keep[1:] |= image_mask[1:] & ~image_mask[:-1]
    return hidden[:, mx.array(np.flatnonzero(keep))], image_mask[keep].tolist()


def flow_schedule(config, steps, image_seq_len):
    if steps < 1:
        raise ValueError("num_inference_steps must be positive")
    if (
        not config["use_dynamic_shifting"]
        or config.get("time_shift_type", "exponential") != "exponential"
    ):
        raise ValueError("Qwen-Image 2.1 MLX requires dynamic exponential shifting")
    if any(
        config.get(key, False)
        for key in (
            "invert_sigmas",
            "stochastic_sampling",
            "use_beta_sigmas",
            "use_exponential_sigmas",
            "use_karras_sigmas",
        )
    ):
        raise ValueError("unsupported Qwen-Image 2.1 MLX scheduler configuration")
    slope = (config["max_shift"] - config["base_shift"]) / (
        config["max_image_seq_len"] - config["base_image_seq_len"]
    )
    mu = (
        image_seq_len * slope
        + config["base_shift"]
        - config["base_image_seq_len"] * slope
    )
    sigmas = np.linspace(1, 1 / steps, steps).astype(np.float32)
    sigmas = (math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))).astype(np.float32)
    terminal = config.get("shift_terminal")
    if terminal is not None and sigmas[-1] != 1:
        one_minus = 1 - sigmas
        sigmas = 1 - one_minus / (one_minus[-1] / (1 - terminal))
    timesteps = sigmas * config["num_train_timesteps"]
    return mx.array(np.append(sigmas, np.float32(0))), mx.array(timesteps)


def flow_step(latents, prediction, delta):
    # torch's zero-dimensional FP32 delta is converted to the BF16 output dtype
    update = delta.astype(prediction.dtype) * prediction
    return (latents.astype(mx.float32) + update.astype(mx.float32)).astype(
        prediction.dtype
    )


def decode_latents(latents, config):
    mean = mx.array(config["latents_mean"]).astype(latents.dtype)
    std = mx.array(config["latents_std"]).astype(latents.dtype)
    scale = mx.reciprocal(std.astype(mx.float32)).astype(latents.dtype)
    return latents / scale + mean


class QwenImage21MLXGenerator:
    """Load one model component at a time; no request retains model weights."""

    def __init__(self, model_path, revision=None):
        self.model_path = Path(model_path)
        if not self.model_path.is_dir():
            self.model_path = Path(
                snapshot_download(
                    model_path,
                    revision=revision,
                    local_files_only=constants.HF_HUB_OFFLINE,
                    allow_patterns=[
                        "configs/*.json",
                        "quantization_config.json",
                        "processor/*",
                        "scheduler/*",
                        "text_encoders/qwen3vl_8b_mlx_q4.safetensors",
                        "diffusion_models/qwen_image_2.1_mlx_q4.safetensors",
                        "vae/qwen_image_2.1_vae_bf16.safetensors",
                    ],
                )
            )
        self.configs = {
            name: json.loads(
                (self.model_path / "configs" / f"{name}_config.json").read_text()
            )
            for name in ("text_encoder", "transformer", "vae")
        }
        self.quantization = json.loads(
            (self.model_path / "quantization_config.json").read_text()
        )
        self.scheduler = json.loads(
            (self.model_path / "scheduler/scheduler_config.json").read_text()
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path / "processor")

    def encode_images(self, images):
        vae = load_vae(
            str(self.model_path / "vae/qwen_image_2.1_vae_bf16.safetensors"),
            self.configs["vae"],
        )
        conditions = []
        for image in images:
            pixels = mx.array(np.array(image).astype(np.float32))[None] / 255
            pixels = (2 * pixels - 1).astype(mx.bfloat16)
            latent = vae.encode(pixels)
            mean = mx.array(self.configs["vae"]["latents_mean"]).astype(latent.dtype)
            std = mx.array(self.configs["vae"]["latents_std"]).astype(latent.dtype)
            conditions.append(((latent - mean) / std).reshape(1, -1, latent.shape[-1]))
            mx.eval(conditions[-1])
        return mx.concatenate(conditions, axis=1)

    def encode_prompts(self, prompts, images):
        text_encoder, vision_encoder = load_encoders(
            str(self.model_path / "text_encoders/qwen3vl_8b_mlx_q4.safetensors"),
            self.configs["text_encoder"],
            self.quantization,
            bool(images),
        )
        encoded = []
        for prompt in prompts:
            embeddings, slots = encode_prompt(
                self.processor, text_encoder, vision_encoder, prompt, images
            )
            mx.eval(embeddings)
            encoded.append((embeddings, slots))
        return encoded

    def denoise(
        self,
        latents,
        conditioning,
        conditions,
        shapes,
        steps,
        guidance_scale,
        progress=None,
    ):
        model = load_transformer(
            str(self.model_path / "diffusion_models/qwen_image_2.1_mlx_q4.safetensors"),
            self.configs["transformer"],
            self.quantization,
        )
        layouts, caches = [], []
        for embeddings, slots in conditioning:
            layout = build_layout(
                slots, shapes, self.configs["transformer"]["axes_dims_rope"]
            )
            layouts.append(layout)
            caches.append(model.prepare_conditioning(embeddings, layout, conditions))
            mx.eval(caches[-1])
        sigmas, timesteps = flow_schedule(self.scheduler, steps, latents.shape[1])
        denoise = model.compile_denoise()
        for index in range(steps):
            start = time.perf_counter()
            prediction = denoise(
                latents, timesteps[index : index + 1], layouts[0].target_rope, caches[0]
            )
            if guidance_scale > 1:
                negative = denoise(
                    latents,
                    timesteps[index : index + 1],
                    layouts[1].target_rope,
                    caches[1],
                )
                delta = prediction - negative
                scaled = (delta.astype(mx.float32) * guidance_scale).astype(delta.dtype)
                prediction = negative + scaled
            latents = flow_step(latents, prediction, sigmas[index + 1] - sigmas[index])
            mx.eval(latents)
            if progress is not None:
                progress(index + 1, steps, time.perf_counter() - start)
        return latents

    def decode(self, latents, width, height):
        vae = load_vae(
            str(self.model_path / "vae/qwen_image_2.1_vae_bf16.safetensors"),
            self.configs["vae"],
        )
        latents = decode_latents(
            latents.reshape(1, height // 16, width // 16, -1), self.configs["vae"]
        )
        pixels = vae.compile_decode()(latents)
        mx.eval(pixels)
        pixels = mx.clip(pixels / 2 + 0.5, 0, 1)
        return Image.fromarray(np.array((pixels[0] * 255).astype(mx.uint8)))

    def generate(
        self,
        prompt,
        width=1024,
        height=1024,
        num_inference_steps=40,
        seed=0,
        images=(),
        negative_prompt="",
        guidance_scale=1.0,
        latents=None,
        progress=None,
    ):
        if width < 32 or height < 32 or width % 32 or height % 32:
            raise ValueError(
                "Qwen-Image 2.1 dimensions must be positive multiples of 32"
            )
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        images = resize_images(images, width, height)
        shapes = [(1, image.height // 16, image.width // 16) for image in images]
        shapes.append((1, height // 16, width // 16))
        prompts = [prompt, negative_prompt] if guidance_scale > 1 else [prompt]
        timings = {}
        mx.reset_peak_memory()
        start = time.perf_counter()
        conditions = self.encode_images(images) if images else None
        if conditions is not None:
            mx.eval(conditions)
        gc.collect()
        mx.clear_cache()
        timings["image_encoding"] = time.perf_counter() - start
        start = time.perf_counter()
        conditioning = self.encode_prompts(prompts, images)
        gc.collect()
        mx.clear_cache()
        timings["text_encoding"] = time.perf_counter() - start
        if latents is None:
            # preserve SGLang's CPU-generator seed and channel-major noise layout
            noise = torch.randn(
                (
                    1,
                    1,
                    self.configs["transformer"]["in_channels"],
                    height // 16,
                    width // 16,
                ),
                generator=torch.Generator(device="cpu").manual_seed(seed),
                dtype=torch.bfloat16,
            )
            latents = mx.array(noise.float().numpy()).astype(mx.bfloat16)
            latents = latents.reshape(
                1, self.configs["transformer"]["in_channels"], -1
            ).transpose(0, 2, 1)
        start = time.perf_counter()
        latents = self.denoise(
            latents,
            conditioning,
            conditions,
            shapes,
            num_inference_steps,
            guidance_scale,
            progress,
        )
        del conditioning, conditions
        gc.collect()
        mx.clear_cache()
        timings["denoising"] = time.perf_counter() - start
        start = time.perf_counter()
        image = self.decode(latents, width, height)
        gc.collect()
        mx.clear_cache()
        timings["decoding"] = time.perf_counter() - start
        timings["peak_memory_bytes"] = mx.get_peak_memory()
        return image, timings


class QwenImage21MLXGenerationStage(PipelineStage):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, batch, server_args):
        prompts = batch.prompt if isinstance(batch.prompt, list) else [batch.prompt]
        negatives = (
            batch.negative_prompt
            if isinstance(batch.negative_prompt, list)
            else [batch.negative_prompt] * len(prompts)
        )
        images = batch.condition_image
        images = (
            [] if images is None else images if isinstance(images, list) else [images]
        )
        guidance = server_args.pipeline_config.get_classifier_free_guidance_scale(
            batch, batch.guidance_scale
        )
        outputs = []
        peak_memory = 0

        def progress(index, total, seconds):
            if batch.metrics is not None:
                batch.metrics.record_step(seconds)
            if not batch.is_warmup and not batch.suppress_logs:
                logger.info("MLX denoising step %d/%d: %.3fs", index, total, seconds)

        for prompt_index, (prompt, negative) in enumerate(
            zip(prompts, negatives, strict=True)
        ):
            for output_index in range(batch.num_outputs_per_prompt):
                seed_index = prompt_index * batch.num_outputs_per_prompt + output_index
                image, timings = self.generator.generate(
                    prompt=prompt,
                    negative_prompt=negative or "",
                    images=images,
                    width=batch.width,
                    height=batch.height,
                    num_inference_steps=batch.num_inference_steps,
                    seed=batch.seeds[seed_index],
                    guidance_scale=guidance,
                    progress=progress,
                )
                outputs.append(np.array(image))
                peak_memory = max(peak_memory, timings["peak_memory_bytes"])
                logger.debug(
                    "MLX phase timings (including component loading): %s", timings
                )
        if batch.metrics is not None:
            batch.metrics.record_memory_snapshot(
                "mlx",
                MemorySnapshot(
                    allocated_mb=mx.get_active_memory() / 1024**2,
                    reserved_mb=(mx.get_active_memory() + mx.get_cache_memory())
                    / 1024**2,
                    peak_allocated_mb=peak_memory / 1024**2,
                    peak_reserved_mb=0.0,
                ),
            )
        return OutputBatch(
            output=outputs,
            metrics=batch.metrics,
            usage=batch.usage,
            peak_memory_mb=peak_memory / 1024**2,
        )
