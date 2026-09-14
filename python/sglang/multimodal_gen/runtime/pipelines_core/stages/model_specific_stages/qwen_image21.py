# SPDX-License-Identifier: Apache-2.0
import math

import torch
from PIL import Image

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import build_layout
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    calculate_linear_shift,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.utils.vision import load_image

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
SYSTEM_TEMPLATE = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"


def collapse_image_slots(hidden, input_ids, image_token_id):
    image_mask = input_ids == image_token_id
    keep = ~image_mask
    keep[0] = True
    keep[1:] |= image_mask[1:] & ~image_mask[:-1]
    return hidden[keep], image_mask[keep]


class QwenImage21InputValidationStage(InputValidationStage):
    def preprocess_condition_image(
        self, batch, server_args, condition_image_width, condition_image_height
    ):
        # one model-owned resize is shared by the VLM and VAE in the encoding stage
        return None

    def forward(self, batch, server_args):
        if batch.prompt is None:
            raise ValueError("Qwen-Image 2.1 requires a prompt to build image-token positions")
        batch = super().forward(batch, server_args)
        if batch.height % 32 or batch.width % 32:
            raise ValueError("Qwen-Image 2.1 height and width must be divisible by 32")
        return batch


class QwenImage21EncodingStage(PipelineStage):
    def __init__(self, text_encoder, processor, vae, scheduler):
        super().__init__()
        self.text_encoder, self.processor, self.vae, self.scheduler = (
            text_encoder,
            processor,
            vae,
            scheduler,
        )
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.drop_idx = len(
            processor.tokenizer.encode(SYSTEM_TEMPLATE, add_special_tokens=False)
        )

    def component_uses(self, server_args, stage_name=None):
        name = self._component_stage_name(stage_name)
        return [
            ComponentUse(name, "text_encoder", target_dtype=torch.bfloat16),
            ComponentUse(name, "vae", target_dtype=torch.bfloat16),
        ]

    def encode_prompt(self, prompt, images, device):
        prefix = " ".join(
            f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>"
            for i in range(len(images))
        )
        text = (
            SYSTEM_TEMPLATE
            + f"<|im_start|>user\n{prefix}{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
        kwargs = dict(text=[text], padding=True, return_tensors="pt")
        if images:
            kwargs["images"] = images
        inputs = self.processor(**kwargs).to(device)
        with self.use_declared_component(
            component_name="text_encoder", module=self.text_encoder
        ) as encoder:
            outputs = encoder(
                **inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1
            )
            # native Qwen3-VL exposes pre-final-norm intermediate states
            final_hidden = encoder.model.language_model.norm(outputs.hidden_states[-1])
        valid = inputs.attention_mask[0].bool()
        hidden = final_hidden[0, valid][self.drop_idx :]
        ids = inputs.input_ids[0, valid][self.drop_idx :]
        return collapse_image_slots(hidden, ids, self.image_token_id)

    def forward(self, batch, server_args):
        config = server_args.pipeline_config
        ac = config.vae_config.arch_config
        device = get_local_torch_device()
        images = batch.condition_image
        images = (
            [] if images is None else images if isinstance(images, list) else [images]
        )
        resized, shapes, conditions = [], [], []
        area = batch.height * batch.width
        for image in images:
            if not isinstance(image, Image.Image):
                image = load_image(image)
            width = max(
                32, round(math.sqrt(area * image.width / image.height) / 32) * 32
            )
            height = max(
                32, round(math.sqrt(area * image.height / image.width) / 32) * 32
            )
            resized.append(
                image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            )
            shapes.append((1, height // 16, width // 16))
        if resized:
            with self.use_declared_component(
                component_name="vae", module=self.vae
            ) as vae:
                for image in resized:
                    pixels = torch.frombuffer(
                        bytearray(image.tobytes()), dtype=torch.uint8
                    ).reshape(image.height, image.width, 3)
                    pixels = (
                        pixels.permute(2, 0, 1)[None, :, None].to(
                            device=device, dtype=torch.float32
                        )
                        / 127.5
                        - 1
                    )
                    latent = vae.encode(pixels.to(torch.bfloat16)).mode()
                    mean = latent.new_tensor(ac.latents_mean).view(1, ac.z_dim, 1, 1, 1)
                    std = latent.new_tensor(ac.latents_std).view(1, ac.z_dim, 1, 1, 1)
                    conditions.append(
                        ((latent - mean) / std).flatten(2).transpose(1, 2)
                    )
        shapes.append((1, batch.height // 16, batch.width // 16))
        prompts = batch.prompt if isinstance(batch.prompt, list) else [batch.prompt]
        negatives = (
            batch.negative_prompt
            if isinstance(batch.negative_prompt, list)
            else [batch.negative_prompt] * len(prompts)
        )
        sample_count = len(prompts) * batch.num_outputs_per_prompt
        condition_latents = (
            torch.cat(conditions, dim=1).expand(sample_count, -1, -1)
            if conditions
            else None
        )
        for negative in (
            [False, True] if batch.do_classifier_free_guidance else [False]
        ):
            embeds, masks, layouts = [], [], []
            for prompt in negatives if negative else prompts:
                with set_forward_context(
                    current_timestep=None, attn_metadata=None, forward_batch=batch
                ):
                    hidden, slots = self.encode_prompt(prompt, resized, device)
                layout = build_layout(
                    slots.tolist(), shapes, config.dit_config.axes_dims_rope, device
                )
                for _ in range(batch.num_outputs_per_prompt):
                    embeds.append(hidden)
                    layouts.append(layout)
            max_length = max(x.shape[0] for x in embeds)
            for x in embeds:
                masks.append(torch.arange(max_length, device=device) < x.shape[0])
            packed = torch.stack(
                [
                    torch.nn.functional.pad(x, (0, 0, 0, max_length - x.shape[0]))
                    for x in embeds
                ]
            )
            mask = torch.stack(masks)
            if negative:
                batch.negative_prompt_embeds = [packed]
                batch.negative_prompt_embeds_mask = [mask]
                batch.negative_prompt_seq_lens = [mask.sum(1).tolist()]
            else:
                batch.prompt_embeds = [packed]
                batch.prompt_embeds_mask = [mask]
                batch.prompt_seq_lens = [mask.sum(1).tolist()]
            batch.extra["qwen21_negative" if negative else "qwen21_positive"] = dict(
                layouts=layouts,
                condition_latents=condition_latents,
                prefix_caches=[
                    [{} for _ in range(config.dit_config.num_layers)]
                    for _ in range(sample_count)
                ],
            )
        sched = self.scheduler.config
        batch.extra["qwen21_mu"] = calculate_linear_shift(
            (batch.height // 16) * (batch.width // 16),
            base_seq_len=sched.get("base_image_seq_len", 256),
            max_seq_len=sched.get("max_image_seq_len", 4096),
            base_shift=sched.get("base_shift", 0.5),
            max_shift=sched.get("max_shift", 1.15),
        )
        return batch


def prepare_qwen21_mu(batch, server_args):
    return "mu", batch.extra["qwen21_mu"]


class QwenImage21DenoisingStage(DenoisingStage):
    def _predict_noise(
        self,
        current_model,
        latent_model_input,
        timestep,
        target_dtype,
        guidance,
        **kwargs,
    ):
        caches = kwargs["prefix_caches"]
        if caches is not None and not caches[0][0]:
            # prefill is request-specific; graph replay must only see populated cache tensors
            return current_model(
                hidden_states=latent_model_input, timestep=timestep, **kwargs
            )
        return super()._predict_noise(
            current_model,
            latent_model_input,
            timestep,
            target_dtype,
            guidance,
            **kwargs,
        )
