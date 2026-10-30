# SPDX-License-Identifier: Apache-2.0
import math

import torch
from PIL import Image

from sglang.multimodal_gen.runtime.cache.conditioning import cached_encoder_call
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_group,
    get_tp_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import build_layout
from sglang.multimodal_gen.runtime.pipelines_core.comfyui_mode import (
    get_or_create_run_state,
)
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
    def load_condition_image(self, image):
        return load_image(image, convert_method=lambda image: image.convert("RGBA"))

    def preprocess_condition_image(
        self, batch, server_args, condition_image_width, condition_image_height
    ):
        # one model-owned resize is shared by the VLM and VAE in the encoding stage
        return None

    def forward(self, batch, server_args):
        if batch.prompt is None:
            raise ValueError(
                "Qwen-Image 2.1 requires a prompt to build image-token positions"
            )
        batch = super().forward(batch, server_args)
        if batch.height % 32 or batch.width % 32:
            raise ValueError("Qwen-Image 2.1 height and width must be divisible by 32")
        batch.sigmas = server_args.pipeline_config.prepare_sigmas(
            batch.sigmas, batch.num_inference_steps
        )
        if not batch.sigmas:
            raise ValueError("Qwen-Image 2.1 requires a non-empty sigma schedule")
        # the preset or explicit sigma grid determines the denoising step count
        batch.num_inference_steps = len(batch.sigmas)
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
        self.text_encoder.model.visual.fp32_position_interpolation = False
        self.text_encoder.model.visual.rotary_pos_emb.recompute_on_device_change = True
        # (dtype, device) -> latent mean/std, uploaded once: a per-request upload
        # blocks the host on the VAE encode it should overlap with prompt prep
        self.latent_stats = {}
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        system_message = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
        ]
        self.drop_idx = len(
            processor.apply_chat_template(
                system_message, tokenize=True, return_dict=False
            )[0]
        )

    def component_uses(self, server_args, stage_name=None):
        name = self._component_stage_name(stage_name)
        return [
            # preserve the loader's mixed weight and rotary buffer dtypes
            ComponentUse(name, "text_encoder"),
            ComponentUse(name, "vae", target_dtype=torch.bfloat16),
        ]

    def encode_prompt(self, prompt, images, device):
        cache_group = self.text_encoder._encoder_tp_group
        if cache_group is None and model_parallel_is_initialized():
            cache_group = get_tp_group()
        conditioning = cached_encoder_call(
            self.text_encoder,
            (prompt, images),
            {"device": str(device)},
            lambda: self._encode_prompt(prompt, images, device),
            cache_group,
            namespace=self,
            share_in_group=True,
        )
        self.finish_unused_declared_component(
            component_name="text_encoder", module=self.text_encoder
        )
        return conditioning

    def _encode_prompt(self, prompt, images, device):
        prefix = " ".join(
            f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>"
            for i in range(len(images))
        )
        text = (
            SYSTEM_TEMPLATE
            + f"<|im_start|>user\n{prefix}{prompt or ' '}<|im_end|>\n<|im_start|>assistant\n"
        )
        kwargs = dict(
            text=[text], padding=True, padding_side="left", return_tensors="pt"
        )
        if images:
            vision_images = []
            for image in images:
                if image.mode == "RGBA":
                    # vision conditioning uses white compositing; the VAE keeps RGBA
                    white = Image.new("RGB", image.size, (255, 255, 255))
                    white.paste(image, mask=image.getchannel("A"))
                    image = white
                vision_images.append(image)
            kwargs["images"] = vision_images
        inputs = self.processor(**kwargs).to(device)
        with self.use_declared_component(
            component_name="text_encoder", module=self.text_encoder
        ) as encoder:
            outputs = encoder(
                **inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1
            )
            # the checkpoint expects Transformers 4.57's pre-final-norm hidden state
            final_hidden = outputs.hidden_states[-1]
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
        image_mode = "RGBA" if ac.in_channels == 4 else "RGB"
        for image in images:
            if not isinstance(image, Image.Image):
                image = load_image(
                    image, convert_method=lambda image: image.convert(image_mode)
                )
            width = max(
                32, round(math.sqrt(area * image.width / image.height) / 32) * 32
            )
            height = max(
                32, round(math.sqrt(area * image.height / image.width) / 32) * 32
            )
            resized.append(
                image.convert(image_mode).resize(
                    (width, height), Image.Resampling.LANCZOS
                )
            )
            shapes.append((1, height // 16, width // 16))
        if resized:
            with self.use_declared_component(
                component_name="vae", module=self.vae
            ) as vae:
                vae.use_tiling = config.vae_tiling
                for image in resized:
                    pixels = torch.frombuffer(
                        bytearray(image.tobytes()), dtype=torch.uint8
                    ).reshape(image.height, image.width, ac.in_channels)
                    # preserve the reference's batch stride for identical cuDNN convolution rounding
                    pixels = (
                        pixels[None].permute(0, 3, 1, 2).unsqueeze(2).float() / 255.0
                    )
                    pixels = (2 * pixels - 1).to(device=device, dtype=torch.bfloat16)
                    latent = vae.encode(pixels).mode()
                    stats_key = (latent.dtype, latent.device)
                    if stats_key not in self.latent_stats:
                        self.latent_stats[stats_key] = tuple(
                            latent.new_tensor(values).view(1, ac.z_dim, 1, 1, 1)
                            for values in (ac.latents_mean, ac.latents_std)
                        )
                    mean, std = self.latent_stats[stats_key]
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
        for negative in [False, True] if batch.do_classifier_free_guidance else [False]:
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


# ComfyUI integrated mode. Must match COND_EXTRA_KEY in the plugin's
# executors/qwen_image21.py.
COMFYUI_COND_EXTRA_KEY = "qwen21_cond"


def insert_image_placeholders(context, image_slots, num_refs):
    """Context with a zero token at each ComfyUI image slot, and the native flags."""
    text_len = context.shape[1]
    # Same slot padding as ComfyUI's build_sequence.
    slots = (list(image_slots) + [text_len] * num_refs)[:num_refs]
    flags = torch.zeros(text_len + num_refs, dtype=torch.bool, device=context.device)
    flags[[slot + i for i, slot in enumerate(slots)]] = True
    out = context.new_zeros(context.shape[0], len(flags), context.shape[2])
    out[:, ~flags] = context
    return out, flags.tolist()


def comfyui_layout(image_slots, image_shapes, axes_dims, device):
    """Native ``build_layout`` plus ComfyUI's half-token centring of references.

    ComfyUI shifts a reference by ``0.5 * (h % 2 - target_h % 2)`` per axis; native
    sizes are multiples of 32 px so parity never differs there.
    """
    layout = build_layout(image_slots, image_shapes, axes_dims, device)
    _, target_h, target_w = image_shapes[-1]
    refs = [(s, e) for s, e, is_image in layout["segments"] if is_image]
    for (start, stop), (_, h, w) in zip(refs, image_shapes[:-1]):
        shifts = (0.0, 0.5 * (h % 2 - target_h % 2), 0.5 * (w % 2 - target_w % 2))
        if not any(shifts):
            continue
        delta = torch.cat(
            [
                shift
                * (10000.0 ** (-torch.arange(0, dim, 2, device=device).float() / dim))
                for shift, dim in zip(shifts, axes_dims)
            ]
        )
        layout["prefix_rope"][start:stop] *= torch.polar(torch.ones_like(delta), delta)
    return layout


def _cache_fits(need_bytes, device) -> bool:
    free = 0
    if device.type == "cuda":
        # Blocks freed by an earlier run stay in PyTorch's cache, not in mem_get_info.
        free = torch.cuda.mem_get_info(device)[0]
        free += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    free = torch.tensor([free], dtype=torch.int64)
    # Ranks running one call must agree: under TP an uncached prefix runs
    # extra all-reduces, so a split decision would hang.
    for group in (get_tp_group(), get_sp_group()):
        if group.world_size > 1:
            torch.distributed.all_reduce(
                free, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
            )
    return int(free) > 2 * need_bytes


class QwenImage21ComfyUIConditionStage(PipelineStage):
    """Turn one single-row ComfyUI ``apply_model`` call into native denoising inputs.

    ComfyUI's text encoder drops the vision tokens and reports ``image_slots``:
    the text positions of the reference latents. A placeholder token at each
    slot gives the native layout input; the DiT overwrites those rows with
    ``img_in(condition_latents)``. Conditioning and its prefix K/V cache are
    built once per cond for the sampler run.
    """

    def forward(self, batch, server_args):
        _, _, height, width = batch.latents.shape
        batch.latents = batch.latents.flatten(2).transpose(1, 2).contiguous()
        payload = batch.extra[COMFYUI_COND_EXTRA_KEY]
        conds = get_or_create_run_state(batch, dict)
        key = (batch.extra["comfyui_cond_key"], height, width)
        if key not in conds:
            conds[key] = self._new_cond(batch, server_args, payload, height, width)
        batch.prompt_embeds, batch.extra["qwen21_positive"] = conds[key]
        # Drop the cond at the sampler's last call so its K/V cache (GiBs) is freed
        # with this request, not at the next run; timesteps carry sigma * 1000.
        final_sigma = payload["final_sigma"]
        if (
            final_sigma is not None
            and batch.timesteps.item() <= final_sigma * 1e3 + 1e-3
        ):
            del conds[key]
        return batch

    def _new_cond(self, batch, server_args, payload, height, width):
        arch = server_args.pipeline_config.dit_config.arch_config
        device, dtype = batch.latents.device, torch.bfloat16
        refs = [r.to(device=device, dtype=dtype) for r in payload["ref_latents"]]
        context, flags = insert_image_placeholders(
            batch.prompt_embeds[0].to(device=device, dtype=dtype),
            payload["image_slots"],
            len(refs),
        )
        shapes = [(1, r.shape[-2], r.shape[-1]) for r in refs] + [(1, height, width)]
        layout = comfyui_layout(flags, shapes, arch.axes_dims_rope, device)
        condition_latents = None
        if refs:
            condition_latents = torch.cat(
                [r.flatten(2).transpose(1, 2) for r in refs], dim=1
            )
        # K and V of every layer, bf16.
        need = 4 * arch.num_layers * layout["prefix_rope"].shape[0] * arch.hidden_size
        caches = None
        if _cache_fits(need, device):
            caches = [[{} for _ in range(arch.num_layers)]]
        return [context], dict(
            layouts=[layout], condition_latents=condition_latents, prefix_caches=caches
        )
