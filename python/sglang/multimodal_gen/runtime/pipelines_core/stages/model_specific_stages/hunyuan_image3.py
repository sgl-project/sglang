"""HunyuanImage-3 AR stage — diffusion loop via sglang backbone."""

from functools import partial
from typing import Any

import torch

from sglang.multimodal_gen.configs.sample.hunyuan_image3 import (
    align_hunyuan_image3_resolution,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_tp_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


_DEFAULT_NUM_INFERENCE_STEPS = 50
_DEFAULT_GUIDANCE_SCALE = 2.5

_SYSTEM_PROMPTS = {
    "en_unified": """You are an advanced multimodal model whose core mission is to analyze user intent and generate high-quality text and images.

#### Four Core Capabilities
1.  **Text-to-Text (T2T):** Generate coherent text responses from text prompts.
2.  **Text-to-Image (T2I):** Generate high-quality images from text prompts.
3.  **Text & Image to Text (TI2T):** Generate accurate text responses based on a combination of images and text.
4.  **Text & Image to Image (TI2I):** Generate modified images based on a reference image and editing instructions.

---
### Image Generation Protocol (for T2I & TI2I)
You will operate in one of two modes, determined by the user's starting tag:
#### **<recaption> Mode (Prompt Rewriting)**:
*   **Trigger:** Input begins with `<recaption>`.
*   **Task:** Immediately rewrite the user's text into a structured, objective, and detail-rich professional-grade prompt.
*   **Output:** Output only the rewritten prompt within `<recaption>` tags: `<recaption>Rewritten professional-grade prompt</recaption>`

#### **<think> Mode (Think + Rewrite)**:
*   **Trigger:** Input begins with `<think>`.
*   **Task:** First, conduct a structured analysis of the request within `<think>` tags. Then, output the professional prompt, rewritten based on the analysis, within `<recaption>` tags.
*   **Output:** Strictly adhere to the format: `<think>Analysis process</think><recaption>Rewritten prompt</recaption>`

---
### Execution Standards and Guidelines
#### **`<think>` Phase: Analysis Guidelines**
**For T2I (New Image Generation):**
Deconstruct the user's request into the following core visual components:
*   **Subject:** Key features of the main character/object, including appearance, pose, expression, and emotion.
*   **Composition:** Camera angle, lens type, and layout.
*   **Environment/Background:** The setting, time of day, weather, and background elements.
*   **Lighting:** Technical details such as light source type, direction, and quality.
*   **Color Palette:** The dominant hues and overall color scheme.
*   **Style/Quality:** The artistic style, clarity, depth of field, and other technical details.
*   **Text:** Identify any text to be rendered in the image, including its content, style, and position.
*   **Details:** Small elements that add narrative depth and realism.

**For TI2I (Image Editing):**
Adopt a task-diagnostic approach:
1.  **Diagnose Task:** Identify the edit type and analyze key requirements.
2.  **Prioritize Analysis:**
    *   **Adding:** Analyze the new element's position and appearance, ensuring seamless integration with the original image's lighting, shadows, and style.
    *   **Removing:** Identify the target for removal and determine how to logically fill the resulting space using surrounding textures and lighting.
    *   **Modifying:** Analyze what to change and what it should become, while emphasizing which elements must remain unchanged.
    *   **Style Transfer:** Deconstruct the target style into specific features (e.g., brushstrokes, color palette) and apply them to the original image.
    *   **Text Editing:** Ensure correct content and format. Consider the text's visual style (e.g., font, color, material) and how it adapts to the surface's perspective, curvature, and lighting.
    *   **Reference Editing:** Extract specific visual elements (e.g., appearance, posture, composition, lines, depth) from the reference image to generate an image that aligns with the text description while also incorporating the referenced content.
    *   **Inferential Editing:** Identify vague requests (e.g., "make it more professional") and translate them into concrete visual descriptions.

#### `<recaption>` Phase: Professional-Grade Prompt Generation Rules
**General Rewriting Principles (for T2I & TI2I):**
1.  **Structure & Logic:** Start with a global description. Use positional words (e.g., "foreground", "background") to define the layout.
2.  **Absolute Objectivity:** Avoid subjective terms. Convey aesthetics through precise descriptions of color, light, shadow, and materials.
3.  **Physical & Logical Consistency:** Ensure all descriptions adhere to the laws of physics and common sense.
4.  **Fidelity to User Intent:** Preserve the user's core concepts, subjects, and attributes. Text to be rendered in the image **must be enclosed in double quotes ("")**.
5.  **Camera & Resolution:** Translate camera parameters into descriptions of visual effects. Convert resolution information into natural language.

**T2I-Specific Guidelines:**
*   **Style Adherence & Inference:** Strictly follow the specified style. If none is given, infer the most appropriate style and detail it using professional terminology.
*   **Style Detailing:**
    *   **Photography/Realism:** Use professional photography terms to describe lighting, lens effects, and material textures.
    *   **Painting/Illustration:** Specify the art movement or medium's characteristics.
    *   **UI/Design:** Objectively describe the final product. Define layout, elements, and typography. Text content must be specific and unambiguous.

**TI2I-Specific Guidelines:**
*   **Preserve Unchanged Elements:** Emphasize elements that **remain unchanged**. Unless explicitly instructed, never alter a character's identity/appearance, the core background, camera angle, or overall style.
*   **Clear Editing Instructions:**
    *   **Replacement:** Use the logic "**replace B with A**," and provide a detailed description of A.
    *   **Addition:** Clearly state what to add, where, and what it looks like.
*   **Unambiguous Referencing:** Avoid vague references (e.g., "that person"). Use specific descriptions of appearance.
""",
    "en_vanilla": "You are a helpful assistant to generate an image from user's description.",
}


def _get_system_prompt(sys_type: str) -> str | None:
    if sys_type == "none":
        return None
    elif sys_type in _SYSTEM_PROMPTS:
        return _SYSTEM_PROMPTS[sys_type]
    elif sys_type == "auto":
        return _SYSTEM_PROMPTS["en_unified"]
    else:
        return _SYSTEM_PROMPTS["en_unified"]


def _build_causal_attention_mask(
    batch_size: int,
    seq_len: int,
    image_slices: list[list[slice]],
    device: torch.device,
) -> torch.Tensor:
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril(0).repeat(batch_size, 1, 1)
    for i in range(batch_size):
        for image_slice in image_slices[i]:
            mask[i, image_slice, image_slice] = True
            start = image_slice.start if image_slice.start is not None else 0
            stop = image_slice.stop if image_slice.stop is not None else seq_len
            assert start < stop, f"Invalid image slice: {image_slice}"

    return mask.unsqueeze(1)


def _build_rope_image_info(
    tokenizer_output: Any,
    batch_size: int,
    token_h: int,
    token_w: int,
    image_info: Any = None,
) -> list[list[tuple[slice, tuple[int, int]]]]:
    gen_slices = getattr(tokenizer_output, "gen_image_slices", None)

    if image_info is not None:
        th = getattr(image_info, "token_height", token_h)
        tw = getattr(image_info, "token_width", token_w)
    else:
        th, tw = token_h, token_w

    rope_image_info: list[list[tuple[slice, tuple[int, int]]]] = []
    for b in range(batch_size):
        batch_info: list[tuple[slice, tuple[int, int]]] = []
        if gen_slices is not None:
            slices = gen_slices[b] if isinstance(gen_slices[0], list) else gen_slices
            for s in slices:
                batch_info.append((s, (th, tw)))
        rope_image_info.append(batch_info)
    return rope_image_info


class HunyuanImage3AR(PipelineStage):
    """Native AR stage for HunyuanImage-3 text-to-image generation."""

    def __init__(
        self,
        ar_model,
        vae=None,
        tokenizer=None,
        processor=None,
        scheduler=None,
        model_path: str = "",
    ):
        super().__init__()
        self.ar_model = ar_model
        self._vae = vae
        self._tokenizer = tokenizer
        self._processor = processor
        self._scheduler = scheduler
        self._model_path = model_path
        self._custom_tokenizer = None
        self._sequence_template: str | None = None
        self._drop_think: bool = False
        self._gen_config_steps: int | None = None
        self._gen_config_guidance_scale: float | None = None

    def _get_sequence_template(self) -> str:
        if self._sequence_template is not None:
            return self._sequence_template
        try:
            from transformers.generation.configuration_utils import GenerationConfig
            gen_cfg = GenerationConfig.from_pretrained(self._model_path)
            self._sequence_template = getattr(gen_cfg, "sequence_template", "pretrain")
            self._drop_think = getattr(gen_cfg, "drop_think", False)
        except Exception:
            self._sequence_template = "pretrain"
            self._drop_think = False
        return self._sequence_template

    def _read_gen_config(self) -> dict:
        if not hasattr(self, "_gen_config_cache"):
            try:
                from transformers.generation.configuration_utils import GenerationConfig
                gen_cfg = GenerationConfig.from_pretrained(self._model_path)
                self._gen_config_cache = gen_cfg
            except Exception:
                self._gen_config_cache = None
        return self._gen_config_cache

    def _read_num_inference_steps(self) -> int:
        if self._gen_config_steps is not None:
            return self._gen_config_steps
        gen_cfg = self._read_gen_config()
        if gen_cfg is not None:
            val = getattr(gen_cfg, "diff_infer_steps", None)
            if val is not None:
                self._gen_config_steps = int(val)
                return self._gen_config_steps
        self._gen_config_steps = _DEFAULT_NUM_INFERENCE_STEPS
        return self._gen_config_steps

    def _read_guidance_scale(self) -> float:
        if self._gen_config_guidance_scale is not None:
            return self._gen_config_guidance_scale
        gen_cfg = self._read_gen_config()
        if gen_cfg is not None:
            val = getattr(gen_cfg, "diff_guidance_scale", None)
            if val is not None:
                self._gen_config_guidance_scale = float(val)
                return self._gen_config_guidance_scale
        self._gen_config_guidance_scale = _DEFAULT_GUIDANCE_SCALE
        return self._gen_config_guidance_scale

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        if not isinstance(self.ar_model, torch.nn.Module):
            return []
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "transformer",
                memory_intensive=True,
            )
        ]

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.REPLICATED

    def _resolve_custom_tokenizer(self, server_args: ServerArgs):
        if self._custom_tokenizer is not None:
            return self._custom_tokenizer

        model_path = self._model_path
        if not model_path:
            raise ValueError(
                "HunyuanImage3AR requires a model_path to load the tokenizer."
            )

        from transformers import AutoTokenizer

        from .hunyuan_image3_tokenizer import HunyuanImage3TokenizerWrapper

        base_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            revision=server_args.revision,
        )

        self._custom_tokenizer = HunyuanImage3TokenizerWrapper(base_tokenizer)
        return self._custom_tokenizer

    def _get_image_info_class(self, tokenizer):
        from .hunyuan_image3_tokenizer import ImageInfo as WrapperImageInfo
        return WrapperImageInfo

    def _rebuild_image_info(self, image_info, ImageInfoCls):
        if isinstance(image_info, ImageInfoCls):
            return image_info
        new_info = ImageInfoCls.__new__(ImageInfoCls)
        new_info.__dict__.update(image_info.__dict__)
        return new_info

    def _resolve_processor(self, server_args: ServerArgs):
        if self._processor is not None:
            return self._processor

        model_path = self._model_path
        if not model_path or not server_args.trust_remote_code:
            return None

        try:
            from transformers.dynamic_module_utils import (
                get_class_from_dynamic_module,
            )

            hf_config_obj = server_args.hf_config if hasattr(server_args, "hf_config") else None
            if hf_config_obj is None:
                from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
                    get_hf_config,
                )
                hf_config_obj = get_hf_config(
                    model_path,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
            processor_cls = get_class_from_dynamic_module(
                "image_processor.HunyuanImage3ImageProcessor",
                model_path,
                revision=server_args.revision,
            )
            self._processor = processor_cls(hf_config_obj)
        except Exception as e:
            pass
        return self._processor

    def _backbone_forward(
        self,
        num_image_tokens: int,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        custom_pos_emb: tuple[torch.Tensor, torch.Tensor],
        first_step: bool,
    ) -> torch.Tensor:

        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size).contiguous()
        attention_mask = attention_mask.contiguous()
        cos, sin = custom_pos_emb
        cos = cos.contiguous()
        sin = sin.contiguous()

        if model_parallel_is_initialized():
            tp_group = get_tp_group()
            if tp_group.world_size > 1:
                hidden_states = tp_group.broadcast(hidden_states, src=0)
                attention_mask = tp_group.broadcast(attention_mask, src=0)
                cos = tp_group.broadcast(cos, src=0)
                sin = tp_group.broadcast(sin, src=0)

        output = self.ar_model.forward_block(
            hidden_states,
            attention_mask,
            (cos, sin),
            num_image_tokens=num_image_tokens,
            first_step=first_step,
        )
        actual_batch = attention_mask.shape[0]
        actual_seq_len = output.shape[0] // actual_batch
        return output.view(actual_batch, actual_seq_len, hidden_size)

    def _instantiate_vae_tokens_first_step(
        self,
        hidden_states: torch.Tensor,
        images: torch.Tensor,
        timesteps: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seqlen, n_embd = hidden_states.shape
        t_emb = self.ar_model.time_embed(timesteps)
        image_seq, token_h, token_w = self.ar_model.patch_embed(images, t_emb)
        image_scatter_index = (
            torch.arange(seqlen, device=hidden_states.device)
            .unsqueeze(0)
            .expand(bsz, -1)
        )
        image_scatter_index = image_scatter_index.masked_select(image_mask.bool()).reshape(bsz, -1)
        hidden_states = hidden_states.clone()
        hidden_states.scatter_(
            dim=1,
            index=image_scatter_index.unsqueeze(-1).expand(-1, -1, n_embd),
            src=image_seq,
        )
        return hidden_states

    def _instantiate_timestep_tokens(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        timestep_index: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seqlen, n_embd = hidden_states.shape
        timestep_emb = self.ar_model.timestep_emb(timesteps).reshape(bsz, -1, n_embd)

        if timestep_index.dtype == torch.bool:
            index = (
                torch.arange(seqlen, device=hidden_states.device)
                .unsqueeze(0)
                .expand(bsz, -1)
            )
            ts_scatter_index = index.masked_select(timestep_index).reshape(bsz, -1)
        else:
            ts_scatter_index = timestep_index.long()

        num_positions = ts_scatter_index.shape[1]
        timestep_emb = timestep_emb.expand(-1, num_positions, -1)
        hidden_states = hidden_states.clone()
        hidden_states.scatter_(
            dim=1,
            index=ts_scatter_index.unsqueeze(-1).expand(-1, -1, n_embd),
            src=timestep_emb,
        )
        return hidden_states

    def _extract_diffusion_pred(
        self, hidden_states, timesteps, image_mask, token_h, token_w,
        first_step, num_special_tokens,
    ):
        n_embd = hidden_states.size(-1)
        t_emb = self.ar_model.time_embed_2(timesteps)
        if first_step:
            image_output = hidden_states.masked_select(
                image_mask.unsqueeze(-1).bool()
            ).reshape(-1, token_h * token_w, n_embd)
        else:
            image_output = hidden_states[:, 1:, :]
        return self.ar_model.final_layer(image_output, t_emb, token_h, token_w)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        tokenizer = self._resolve_custom_tokenizer(server_args)
        processor = self._resolve_processor(server_args)

        width, height = align_hunyuan_image3_resolution(batch.width, batch.height)
        if processor is not None:
            image_info = processor.build_gen_image_info(f"{height}x{width}")
            height = image_info.image_height
            width = image_info.image_width
            token_h = image_info.token_height
            token_w = image_info.token_width
            ImageInfoCls = self._get_image_info_class(tokenizer)
            if ImageInfoCls is not None:
                image_info = self._rebuild_image_info(image_info, ImageInfoCls)
        else:
            vae_factor = getattr(
                self.ar_model.hf_config, "vae_downsample_factor", [16, 16]
            )
            if isinstance(vae_factor, (list, tuple)):
                vae_h = vae_factor[0]
                vae_w = vae_factor[1] if len(vae_factor) > 1 else vae_factor[0]
            else:
                vae_h = vae_w = int(vae_factor)
            token_h = height // vae_h
            token_w = width // vae_w
            image_info = None

        num_image_tokens = token_h * token_w

        device = get_local_torch_device()
        model_device = self.ar_model.model.embed_tokens.weight.device
        if model_device.type == "cpu":
            self.ar_model.to(device)
        else:
            device = model_device

        batch_size = 1

        sp = batch.sampling_params
        user_explicit_fields = getattr(sp, "_explicit_fields", set()) if sp else set()
        if "guidance_scale" in user_explicit_fields:
            guidance_scale = float(sp.guidance_scale)
        else:
            guidance_scale = self._read_guidance_scale()
        do_cfg = guidance_scale > 1.0
        cfg_factor = 2 if do_cfg else 1

        bot_task = getattr(batch, "bot_task", "image")
        sys_type = getattr(batch, "sys_type", "en_unified")
        if bot_task == "none":
            bot_task = "image"

        tokenizer_kwargs: dict[str, Any] = dict(
            batch_prompt=[batch.prompt],
            mode="gen_image",
            bot_task=bot_task,
            sequence_template=self._get_sequence_template(),
            drop_think=self._drop_think,
            cfg_factor=cfg_factor,
            image_base_size=getattr(
                processor, "vae_reso_group", None
            ) and processor.vae_reso_group.base_size,
        )

        system_prompt = _get_system_prompt(sys_type)
        if system_prompt is not None:
            tokenizer_kwargs["batch_system_prompt"] = [system_prompt]

        if image_info is not None:
            tokenizer_kwargs["batch_gen_image_info"] = [image_info]

        tokenizer_output_dict = tokenizer.apply_chat_template(**tokenizer_kwargs)
        if isinstance(tokenizer_output_dict, dict):
            tokenizer_output = tokenizer_output_dict.get("output", tokenizer_output_dict)
        else:
            tokenizer_output = tokenizer_output_dict

        if hasattr(tokenizer_output, "tokens"):
            input_ids = tokenizer_output.tokens.to(device)
        elif isinstance(tokenizer_output, torch.Tensor):
            input_ids = tokenizer_output.to(device)
        else:
            input_ids = tokenizer_output["tokens"].to(device)

        actual_batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        if hasattr(tokenizer_output, "gen_image_mask"):
            image_mask = tokenizer_output.gen_image_mask.to(device)
        else:
            image_mask = tokenizer_output.get("gen_image_mask")
            if image_mask is not None:
                image_mask = image_mask.to(device)

        if hasattr(tokenizer_output, "gen_timestep_scatter_index"):
            timestep_index = tokenizer_output.gen_timestep_scatter_index.to(device)
        else:
            timestep_index = tokenizer_output.get("gen_timestep_scatter_index")
            if timestep_index is not None:
                timestep_index = timestep_index.to(device)

        gen_slices = getattr(tokenizer_output, "gen_image_slices", [[] for _ in range(actual_batch_size)])
        joint_slices = getattr(tokenizer_output, "joint_image_slices", [[] for _ in range(actual_batch_size)])
        if not isinstance(gen_slices[0], list):
            gen_slices = [gen_slices]
        if not isinstance(joint_slices[0], list):
            joint_slices = [joint_slices]
        image_slices = [joint_slices[i] + gen_slices[i] for i in range(actual_batch_size)]
        attention_mask = _build_causal_attention_mask(
            actual_batch_size, seq_len, image_slices, device
        )

        non_first_seq_len = 1 + num_image_tokens

        rope_image_info = _build_rope_image_info(
            tokenizer_output, actual_batch_size, token_h, token_w, image_info
        )
        cos, sin = self.ar_model.cached_rope(seq_len, device, rope_image_info=rope_image_info)

        non_first_rope_info: list[list[tuple[slice, tuple[int, int]]]] = [
            [(slice(1, non_first_seq_len), (token_h, token_w))]
            for _ in range(actual_batch_size)
        ]
        non_first_cos, non_first_sin = self.ar_model.cached_rope(
            non_first_seq_len, device, rope_image_info=non_first_rope_info
        )

        if timestep_index is not None:
            if timestep_index.dtype == torch.bool:
                _ts_pos = int(timestep_index[0].float().argmax().item())
            else:
                _ts_pos = int(timestep_index[0, 0].item())
            _ts_rope_cos = cos[0, _ts_pos : _ts_pos + 1].clone()
            _ts_rope_sin = sin[0, _ts_pos : _ts_pos + 1].clone()
            non_first_cos = non_first_cos.clone()
            non_first_sin = non_first_sin.clone()
            non_first_cos[:, 0:1] = _ts_rope_cos
            non_first_sin[:, 0:1] = _ts_rope_sin

        num_inference_steps = int(
            getattr(batch, "num_inference_steps", None) or _DEFAULT_NUM_INFERENCE_STEPS
        )
        scheduler = self._scheduler
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps

        hf_config = self.ar_model.hf_config
        if hasattr(hf_config, "vae") and isinstance(hf_config.vae, dict):
            latent_channels = hf_config.vae["latent_channels"]
        else:
            latent_channels = getattr(hf_config, "latent_channels", 32)

        vae_factor = getattr(hf_config, "vae_downsample_factor", [16, 16])
        if isinstance(vae_factor, (list, tuple)):
            vae_h = vae_factor[0]
            vae_w = vae_factor[1] if len(vae_factor) > 1 else vae_factor[0]
        else:
            vae_h = vae_w = int(vae_factor)

        latent_h = height // vae_h
        latent_w = width // vae_w

        generator = torch.Generator(device=device)
        if batch.seed is not None:
            generator.manual_seed(batch.seed)

        latents = torch.randn(
            1,
            latent_channels,
            latent_h,
            latent_w,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )

        backbone_fn = partial(self._backbone_forward, num_image_tokens)

        first_step_input_ids = input_ids

        for step_idx, t in enumerate(timesteps):
            first_step = step_idx == 0

            latent_model_input = scheduler.scale_model_input(latents, t)
            if do_cfg:
                latent_model_input = torch.cat([latent_model_input] * cfg_factor, dim=0)

            latent_bs = latent_model_input.shape[0]
            t_expand = t.repeat(latent_bs).to(device)

            with torch.autocast(device_type=current_platform.device_type, dtype=torch.bfloat16, enabled=True):
                hidden_states = self.ar_model.model.get_input_embeddings(
                    first_step_input_ids,
                ).expand(latent_bs, -1, -1)
                hidden_states = self._instantiate_vae_tokens_first_step(
                    hidden_states, latent_model_input, t_expand, image_mask,
                )
                if timestep_index is not None:
                    hidden_states = self._instantiate_timestep_tokens(
                        hidden_states, t_expand, timestep_index,
                    )

                backbone_out = backbone_fn(
                    hidden_states, attention_mask, (cos, sin), True,
                )

                pred = self._extract_diffusion_pred(
                    backbone_out, t_expand, image_mask,
                    token_h, token_w, first_step=True,
                    num_special_tokens=seq_len - num_image_tokens,
                )

            pred = pred.float()

            if do_cfg:
                pred_cond, pred_uncond = pred.chunk(2)
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            latent_dtype = latents.dtype
            latents = scheduler.step(pred, t, latents, return_dict=False)[0].to(dtype=latent_dtype)

        batch.latents = latents.to(torch.bfloat16).unsqueeze(2)
        return batch
