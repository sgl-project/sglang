# SPDX-License-Identifier: Apache-2.0

import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import constants, snapshot_download
from PIL import Image
from transformers import AutoProcessor

from .qwen_image21 import build_layout
from .qwen_image21_processing import (
    decode_latents,
    encode_prompt,
    flow_schedule,
    flow_step,
    resize_images,
)
from .weights import load_encoders, load_transformer, load_vae


class QwenImage21MLXPipeline:
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
