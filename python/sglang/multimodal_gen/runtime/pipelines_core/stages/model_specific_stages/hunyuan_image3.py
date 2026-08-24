"""Native AR stage for HunyuanImage-3 text-to-image generation.

Implements the diffusion sampling loop directly using the sglang backbone's
``forward_block`` interface, without relying on the official HF shell model.
"""

import os
from functools import partial
from typing import Any, Optional

import torch
from einops import rearrange

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
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# Default sampling parameters (from generation_config.json)
_DEFAULT_NUM_INFERENCE_STEPS = 50
_DEFAULT_GUIDANCE_SCALE = 2.5

# System prompts for HunyuanImage-3 (from vllm_omni system_prompt.py)
_SYSTEM_PROMPTS = {
    "en_vanilla": """\
You are an advanced AI text-to-image generation system. Given a detailed text prompt, your task is to create a high-quality, visually compelling image that accurately represents the described scene, characters, or objects. Pay careful attention to style, color, lighting, perspective, and any specific instructions provided.
""",
    "en_recaption": """\
You are a world-class image generation prompt expert. Your task is to rewrite a user's simple description into a **structured, objective, and detail-rich** professional-level prompt.

The final output must be wrapped in `<recaption>` tags.

### **Universal Core Principles**

When rewriting the prompt (inside the `<recaption>` tags), you must adhere to the following principles:

1.  **Absolute Objectivity**: Describe only what is visually present. Avoid subjective words like "beautiful" or "sad". Convey aesthetic qualities through specific descriptions of color, light, shadow, and composition.
2.  **Physical and Logical Consistency**: All scene elements (e.g., gravity, light, shadows, reflections, spatial relationships, object proportions) must strictly adhere to real-world physics and common sense. For example, tennis players must be on opposite sides of the net; objects cannot float without a cause.
3.  **Structured Description**: Strictly follow a logical order: from general to specific, background to foreground, and primary to secondary elements. Use directional terms like "foreground," "mid-ground," "background," and "left side of the frame" to clearly define the spatial layout.
4.  **Use Present Tense**: Describe the scene from an observer's perspective using the present tense, such as "A man stands..." or "Light shines on..."
5.  **Use Rich and Specific Descriptive Language**: Use precise adjectives to describe the quantity, size, shape, color, and other attributes of objects, subjects, and text. Vague expressions are strictly prohibited.

If the user specifies a style (e.g., oil painting, anime, UI design, text rendering), strictly adhere to that style. Otherwise, first infer a suitable style from the user's input. If there is no clear stylistic preference, default to an **ultra-realistic photographic style**. Then, generate the detailed rewritten prompt according to the **Style-Specific Creation Guide** below:

### **Style-Specific Creation Guide**

Based on the determined artistic style, apply the corresponding professional knowledge.

**1. Photography and Realism Style**
*   Utilize professional photography terms (e.g., lighting, lens, composition) and meticulously detail material textures, physical attributes of subjects, and environmental details.

**2. Illustration and Painting Style**
*   Clearly specify the artistic school (e.g., Japanese Cel Shading, Impasto Oil Painting) and focus on describing its unique medium characteristics, such as line quality, brushstroke texture, or paint properties.

**3. Graphic/UI/APP Design Style**
*   Objectively describe the final product, clearly defining the layout, elements, and color palette. All text on the interface must be enclosed in double quotes `""` to specify its exact content (e.g., "Login"). Vague descriptions are strictly forbidden.

**4. Typographic Art**
*   The text must be described as a complete physical object. The description must begin with the text itself. Use a straightforward front-on or top-down perspective to ensure the entire text is visible without cropping.

### **Final Output Requirements**

1.  **Output the Final Prompt Only**: Do not show any thought process, Markdown formatting, or line breaks.
2.  **Adhere to the Input**: You must retain the core concepts, attributes, and any specified text from the user's input.
3.  **Style Reinforcement**: Mention the core style 3-5 times within the prompt and conclude with a style declaration sentence.
4.  **Avoid Self-Reference**: Describe the image content directly. Remove redundant phrases like "This image shows..." or "The scene depicts..."
5.  **The final output must be wrapped in `<recaption>xxxx</recaption>` tags.**

The user will now provide an input prompt. You will provide the expanded prompt.
""",
    "en_think_recaption": """\
You will act as a top-tier Text-to-Image AI. Your core task is to deeply analyze the user's text input and transform it into a detailed, artistic, and fully user-intent-compliant image.

Your workflow is divided into two phases:

1. Thinking Phase (<think>): In the <think> tag, you need to conduct a structured thinking process, progressively breaking down and enriching the constituent elements of the image. This process must include, but is not limited to, the following dimensions:

Subject: Clearly define the core character(s) or object(s) in the scene, including their appearance, posture, expression, and emotion.
Composition: Set the camera angle and layout, such as close-up, long shot, bird's-eye view, golden ratio composition, etc.
Environment/Background: Describe the scene where the subject is located, including the location, time of day, weather, and other elements in the background.
Lighting: Define the type, direction, and quality of the light source, such as soft afternoon sunlight, cool tones of neon lights, dramatic Rembrandt lighting, etc., to create a specific atmosphere.
Color Palette: Set the main color tone and color scheme of the image, such as vibrant and saturated, low-saturation Morandi colors, black and white, etc.
Quality/Style: Determine the artistic style and technical details of the image. This includes user-specified styles (e.g., anime, oil painting) or the default realistic style, as well as camera parameters (e.g., focal length, aperture, depth of field).
Details: Add minute elements that enhance the realism and narrative quality of the image, such as a character's accessories, the texture of a surface, dust particles in the air, etc.


2. Recaption Phase (<recaption>): In the <recaption> tag, merge all the key details from the thinking process into a coherent, precise, and visually evocative final description. This description is the direct instruction for generating the image, so it must be clear, unambiguous, and organized in a way that is most suitable for an image generation engine to understand.

Absolutely Objective: Describe only what is visually present. Avoid subjective words like "beautiful" or "sad." Convey aesthetic sense through concrete descriptions of colors, light, shadow, and composition.

Physical and Logical Consistency: All scene elements (e.g., gravity, light and shadow, reflections, spatial relationships, object proportions) must strictly adhere to the physical laws of the real world and common sense. For example, in a tennis match, players must be on opposite sides of the net; objects cannot float without reason.

Structured Description: Strictly follow a logical order: from whole to part, background to foreground, and primary to secondary. Use directional words like "foreground," "mid-ground," "background," "left side of the frame" to clearly define the spatial layout.

Use Present Tense: Describe from an observer's perspective using the present tense, such as "a man stands," "light shines on..."
Use Rich and Specific Descriptive Language: Use precise adjectives to describe the quantity, size, shape, color, and other attributes of objects/characters/text. Absolutely avoid any vague expressions.


Output Format:
<think>Thinking process</think><recaption>Refined image description</recaption>Generate Image


You must strictly adhere to the following rules:

1. Faithful to Intent, Reasonable Expansion: You can creatively add details to the user's description to enhance the image's realism and artistic quality. However, all additions must be highly consistent with the user's core intent and never introduce irrelevant or conflicting elements.
2. Style Handling: When the user does not specify a style, you must default to an "Ultra-realistic, Photorealistic" style. If the user explicitly specifies a style (e.g., anime, watercolor, oil painting, cyberpunk, etc.), both your thinking process and final description must strictly follow and reflect that specified style.
3. Text Rendering: If specific text needs to appear in the image (such as words on a sign, a book title), you must enclose this text in English double quotes (""). Descriptive text must not use double quotes.
4. Design-related Images: You need to specify all text and graphical elements that appear in the image and clearly describe their design details, including font, color, size, position, arrangement, visual effects, etc.
""",
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
    "en_vanilla_short": "You are a helpful assistant to generate an image from user's description.",
}


def _resolve_system_prompt(
    system_prompt: str | None,
    bot_task: str = "image",
) -> str | None:
    """Resolve system prompt: preset name → prompt text, or raw text as-is."""
    if system_prompt is None or system_prompt == "none":
        return None
    if system_prompt in _SYSTEM_PROMPTS:
        return _SYSTEM_PROMPTS[system_prompt]
    if system_prompt == "dynamic":
        if bot_task == "think":
            return _SYSTEM_PROMPTS["en_think_recaption"]
        elif bot_task == "recaption":
            return _SYSTEM_PROMPTS["en_recaption"]
        elif bot_task == "image":
            return _SYSTEM_PROMPTS["en_vanilla_short"]
        return None
    if system_prompt == "auto":
        return _SYSTEM_PROMPTS["en_unified"]
    # Treat as raw custom text
    return system_prompt



def _build_causal_attention_mask(
    batch_size: int,
    seq_len: int,
    image_slices: list[list[slice]],
    device: torch.device,
) -> tuple[torch.Tensor, list[list[tuple[int, int]]]]:
    """Build 4D causal attention mask with full attention at image positions.

    Matches vllm-omni's ``_prepare_attention_mask_for_generation`` exactly:
    per-batch mask rows, combined joint + gen image slices, and
    ``full_attn_spans`` tracking for downstream use.

    Args:
        batch_size: batch size (may be doubled for CFG).
        seq_len: total sequence length.
        image_slices: per-batch list of slice objects marking image token
            positions that should use full (non-causal) attention.
        device: target device.

    Returns:
        ``(attention_mask, full_attn_spans)`` where *attention_mask* has shape
        ``[batch_size, 1, seq_len, seq_len]`` and *full_attn_spans* is a
        per-batch list of ``(start, stop)`` tuples for image regions.
    """
    # Causal (lower-triangular) mask — per-batch rows (matches vllm-omni)
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril(0).repeat(batch_size, 1, 1)

    full_attn_spans: list[list[tuple[int, int]]] = [[] for _ in range(batch_size)]
    for i in range(batch_size):
        for image_slice in image_slices[i]:
            mask[i, image_slice, image_slice] = True
            start = image_slice.start if image_slice.start is not None else 0
            stop = image_slice.stop if image_slice.stop is not None else seq_len
            assert start < stop, f"Invalid image slice: {image_slice}"
            full_attn_spans[i].append((int(start), int(stop)))
        if full_attn_spans[i]:
            full_attn_spans[i].sort(key=lambda x: x[0])

    mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
    return mask, full_attn_spans


def _build_rope_image_info(
    tokenizer_output: Any,
    batch_size: int,
    token_h: int,
    token_w: int,
    image_info: Any = None,
    sections: list | None = None,
) -> list[list[tuple[slice, tuple[int, int]]]]:
    """Build 2D-RoPE image info from tokenizer output and image dimensions.

    Returns a per-batch list of ``[(slice, (token_h, token_w)), ...]`` tuples
    describing where image tokens sit in the sequence and their spatial layout.

    When *sections* is provided (from the tokenizer's output), per-image
    spatial dims are read from each section's ``token_height`` /
    ``token_width`` (which may be lists for ``joint_image`` sections).
    Otherwise falls back to the explicit *token_h* / *token_w*.
    """
    gen_slices = getattr(tokenizer_output, "gen_image_slices", None)
    joint_slices = getattr(tokenizer_output, "joint_image_slices", None)
    cond_vae_slices = getattr(tokenizer_output, "cond_vae_image_slices", None)
    cond_vit_slices = getattr(tokenizer_output, "cond_vit_image_slices", None)

    # Resolve spatial dims for gen image: prefer image_info attrs, fall back to args
    if image_info is not None:
        th = getattr(image_info, "token_height", token_h)
        tw = getattr(image_info, "token_width", token_w)
    else:
        th, tw = token_h, token_w

    # Build per-section shape lookup from tokenizer sections
    section_shapes: list[tuple[int, int]] = []
    if sections is not None:
        for section in sections:
            stype = section.get("type", "")
            if "image" in stype:
                t_h = section.get("token_height", th)
                t_w = section.get("token_width", tw)
                if isinstance(t_h, list):
                    # joint_image: list of [vae, vit] dims
                    for h_i, w_i in zip(t_h, t_w):
                        section_shapes.append((int(h_i), int(w_i)))
                else:
                    section_shapes.append((int(t_h), int(t_w)))

    rope_image_info: list[list[tuple[slice, tuple[int, int]]]] = []
    for b in range(batch_size):
        batch_info: list[tuple[slice, tuple[int, int]]] = []
        # Use section shapes if available, otherwise fall back to gen dims
        shape_idx = 0

        # Add joint (cond) image slices first (they appear first in the sequence)
        if cond_vae_slices is not None:
            slices = cond_vae_slices[b] if isinstance(cond_vae_slices[0], list) else cond_vae_slices
            for s in slices:
                if shape_idx < len(section_shapes):
                    batch_info.append((s, section_shapes[shape_idx]))
                    shape_idx += 1
                else:
                    batch_info.append((s, (token_h, token_w)))
        if cond_vit_slices is not None:
            slices = cond_vit_slices[b] if isinstance(cond_vit_slices[0], list) else cond_vit_slices
            for s in slices:
                if shape_idx < len(section_shapes):
                    batch_info.append((s, section_shapes[shape_idx]))
                    shape_idx += 1
                else:
                    batch_info.append((s, (token_h, token_w)))

        # Add gen image slices
        if gen_slices is not None:
            slices = gen_slices[b] if isinstance(gen_slices[0], list) else gen_slices
            for s in slices:
                if shape_idx < len(section_shapes):
                    batch_info.append((s, section_shapes[shape_idx]))
                    shape_idx += 1
                else:
                    batch_info.append((s, (th, tw)))

        rope_image_info.append(batch_info)
    return rope_image_info


class HunyuanImage3AR(PipelineStage):
    """Native AR stage for HunyuanImage-3 text-to-image generation.

    Runs the flow-matching diffusion loop directly using the sglang backbone
    (``forward_block``) and the diffusion I/O modules (``patch_embed``,
    ``timestep_emb``, ``time_embed``, ``final_layer``, ``time_embed_2``)
    that live on the AR model.

    Only direct image generation (text-to-image) is supported.

    Args:
        ar_model: The sglang-loaded HunyuanImage-3 backbone with diffusion
            I/O modules, providing ``forward_block``.
        vae: The pipeline-loaded VAE module (used for config only; decode
            happens in the decoding stage).
        tokenizer: Standard HF tokenizer (may be unused if we load the
            custom tokenizer ourselves).
        processor: The repo's HunyuanImage3ImageProcessor.
        scheduler: Flow-matching Euler scheduler.
        model_path: Path to the model repository (for loading the custom
            tokenizer).
    """

    def __init__(
        self,
        ar_model,
        vae=None,
        tokenizer=None,
        processor=None,
        scheduler=None,
        model_path: str = "",
        vision_model=None,
        vision_aligner=None,
    ):
        super().__init__()
        self.ar_model = ar_model
        self._vae = vae
        self._tokenizer = tokenizer
        self._processor = processor
        self._scheduler = scheduler
        self._model_path = model_path
        self._vision_model = vision_model
        self._vision_aligner = vision_aligner
        self._custom_tokenizer = None
        self._sequence_template: str | None = None
        self._drop_think: bool = False
        self._gen_config_steps: int | None = None
        self._gen_config_guidance_scale: float | None = None

    def _get_sequence_template(self) -> str:
        """Read sequence_template from model's generation_config, default 'pretrain'.

        Matches vllm-omni behaviour: ``GenerationConfig.from_pretrained(model_path)``
        then ``getattr(generation_config, 'sequence_template', 'pretrain')``.
        """
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
        logger.info(
            "Using sequence_template='%s', drop_think=%s (from model generation_config)",
            self._sequence_template, self._drop_think,
        )
        return self._sequence_template

    def _read_gen_config(self) -> dict:
        """Lazily read and cache the model's generation_config.json."""
        if not hasattr(self, "_gen_config_cache"):
            try:
                from transformers.generation.configuration_utils import GenerationConfig
                gen_cfg = GenerationConfig.from_pretrained(self._model_path)
                self._gen_config_cache = gen_cfg
            except Exception:
                self._gen_config_cache = None
        return self._gen_config_cache

    def _read_num_inference_steps(self) -> int:
        """Read diff_infer_steps from generation_config.json, fallback to _DEFAULT_NUM_INFERENCE_STEPS."""
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
        """Read diff_guidance_scale from generation_config.json, fallback to _DEFAULT_GUIDANCE_SCALE."""
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

    # ------------------------------------------------------------------
    # Tokenizer / processor resolution
    # ------------------------------------------------------------------

    def _resolve_custom_tokenizer(self, server_args: ServerArgs):
        """Load base tokenizer + sglang-native HunyuanImage3 wrapper.

        Uses ``AutoTokenizer.from_pretrained`` (without ``trust_remote_code``)
        to get the base ``PreTrainedTokenizerFast``, then wraps it in
        ``HunyuanImage3TokenizerWrapper`` which provides the multimodal
        ``apply_chat_template`` entry point.
        """
        if self._custom_tokenizer is not None:
            return self._custom_tokenizer

        model_path = self._model_path
        if not model_path:
            raise ValueError(
                "HunyuanImage3AR requires a model_path to load the tokenizer."
            )

        from transformers import AutoTokenizer

        from .hunyuan_image3_tokenizer import HunyuanImage3TokenizerWrapper

        # Load the base tokenizer (no trust_remote_code)
        base_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            revision=server_args.revision,
        )

        # Wrap in our sglang-native tokenizer wrapper
        self._custom_tokenizer = HunyuanImage3TokenizerWrapper(base_tokenizer)
        logger.info(
            "Loaded base tokenizer + HunyuanImage3TokenizerWrapper from %s",
            model_path,
        )
        return self._custom_tokenizer

    def _get_image_info_class(self, tokenizer):
        """Return the ``ImageInfo`` class from our tokenizer wrapper module.

        The ``HunyuanImage3TokenizerWrapper`` performs ``isinstance`` checks
        against its own ``ImageInfo`` class.  We return that class so
        ``_rebuild_image_info`` can convert the processor's ImageInfo
        into the correct type.
        """
        from .hunyuan_image3_tokenizer import ImageInfo as WrapperImageInfo
        return WrapperImageInfo

    def _rebuild_image_info(self, image_info, ImageInfoCls):
        """Re-create *image_info* as an instance of *ImageInfoCls*.

        Copies all instance attributes so the tokenizer's ``isinstance`` check
        succeeds even when the processor and tokenizer loaded
        ``tokenization_hunyuan_image_3.py`` from different cache directories.
        """
        if isinstance(image_info, ImageInfoCls):
            return image_info
        # Create a bare instance and copy all attributes from the source.
        new_info = ImageInfoCls.__new__(ImageInfoCls)
        new_info.__dict__.update(image_info.__dict__)
        return new_info

    def _resolve_processor(self, server_args: ServerArgs):
        """Return the image processor, loading it lazily if needed."""
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
            logger.warning("Failed to load image processor: %s", e)
        return self._processor

    # ------------------------------------------------------------------
    # Backbone forward (with TP broadcast for determinism)
    # ------------------------------------------------------------------

    def _backbone_forward(
        self,
        num_image_tokens: int,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        custom_pos_emb: tuple[torch.Tensor, torch.Tensor],
        first_step: bool,
    ) -> torch.Tensor:
        """Run one backbone pass through the sglang forward_block."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size).contiguous()
        attention_mask = attention_mask.contiguous()
        cos, sin = custom_pos_emb
        cos = cos.contiguous()
        sin = sin.contiguous()

        # Broadcast from rank 0 for deterministic TP collectives
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
        # Derive reshape dims from actual output (may differ from input
        # batch_size after TP broadcast).
        actual_batch = attention_mask.shape[0]
        actual_seq_len = output.shape[0] // actual_batch
        return output.view(actual_batch, actual_seq_len, hidden_size)

    # ------------------------------------------------------------------
    # Diffusion I/O helpers
    # ------------------------------------------------------------------

    def _instantiate_vae_tokens_first_step(
        self,
        hidden_states: torch.Tensor,
        images: torch.Tensor,
        timesteps: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter VAE image embeddings + timestep embeddings into text hidden states.

        Used on the first diffusion step when hidden_states contains text
        token embeddings.
        """
        bsz, seqlen, n_embd = hidden_states.shape
        # Timestep conditioning for patch_embed
        t_emb = self.ar_model.time_embed(timesteps)
        # VAE latent → sequence embedding
        image_seq, token_h, token_w = self.ar_model.patch_embed(images, t_emb)
        # Scatter image embeddings at image_mask positions
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
        """Scatter timestep embeddings into hidden_states at timestep_index positions.

        The ``timestep_emb`` module produces one embedding per batch element,
        but ``timestep_index`` may mark multiple sequence positions that should
        all receive the *same* embedding vector.

        ``timestep_index`` may be either:
          - A boolean mask of shape ``(bsz, seqlen)`` — True marks scatter positions.
          - Position indices of shape ``(bsz, K)`` — each value is a column index.
        """
        bsz, seqlen, n_embd = hidden_states.shape
        # One embedding per batch element → [bsz, 1, n_embd]
        timestep_emb = self.ar_model.timestep_emb(timesteps).reshape(bsz, -1, n_embd)

        # Determine scatter indices based on timestep_index format
        if timestep_index.dtype == torch.bool:
            # Boolean mask: (bsz, seqlen) with True at scatter positions
            index = (
                torch.arange(seqlen, device=hidden_states.device)
                .unsqueeze(0)
                .expand(bsz, -1)
            )
            ts_scatter_index = index.masked_select(timestep_index).reshape(bsz, -1)
        else:
            # Position indices: (bsz, K) containing column indices directly
            ts_scatter_index = timestep_index.long()

        num_positions = ts_scatter_index.shape[1]
        # Expand the single embedding to fill all marked positions
        timestep_emb = timestep_emb.expand(-1, num_positions, -1)
        hidden_states = hidden_states.clone()
        hidden_states.scatter_(
            dim=1,
            index=ts_scatter_index.unsqueeze(-1).expand(-1, -1, n_embd),
            src=timestep_emb,
        )
        return hidden_states

    def _build_non_first_step_input(
        self, timesteps: torch.Tensor, images: torch.Tensor, batch_size: int,
    ) -> torch.Tensor:
        """Build hidden states for non-first diffusion steps (no text tokens).

        Concatenates [timestep_emb, patch_embed(latents, time_embed(t))].
        """
        t_emb = self.ar_model.time_embed(timesteps)
        image_emb, _, _ = self.ar_model.patch_embed(images, t_emb)
        timestep_emb = self.ar_model.timestep_emb(timesteps).reshape(
            batch_size, -1, self.ar_model.config.hidden_size
            if hasattr(self.ar_model.config, "hidden_size")
            else image_emb.shape[-1]
        )
        return torch.cat([timestep_emb, image_emb], dim=1)

    def _extract_diffusion_pred(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        image_mask: torch.Tensor,
        token_h: int,
        token_w: int,
        first_step: bool,
        num_special_tokens: int,
    ) -> torch.Tensor:
        """Extract the noise prediction from backbone output via final_layer."""
        n_embd = hidden_states.size(-1)
        t_emb = self.ar_model.time_embed_2(timesteps)

        if first_step:
            # Select image positions using the mask
            image_output = hidden_states.masked_select(
                image_mask.unsqueeze(-1).bool()
            ).reshape(-1, token_h * token_w, n_embd)
        else:
            # Non-first step: skip the timestep token (position 0)
            image_output = hidden_states[:, 1:, :]

        pred = self.ar_model.final_layer(image_output, t_emb, token_h, token_w)

        return pred

    # ------------------------------------------------------------------
    # Conditional image processing helpers (TI2I / I2I)
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_and_crop_center(image, target_width: int, target_height: int):
        """Resize with aspect-ratio preservation then center-crop."""
        from PIL import Image as PILImage
        tw, th = target_width, target_height
        w, h = image.size
        tr = th / tw
        r = h / w
        if r < tr:
            resize_height = th
            resize_width = int(round(th / h * w))
        else:
            resize_width = tw
            resize_height = int(round(tw / w * h))
        resized = image.resize((resize_width, resize_height), PILImage.Resampling.LANCZOS)
        crop_left = int(round((resize_width - tw) / 2.0))
        crop_top = int(round((resize_height - th) / 2.0))
        return resized.crop((crop_left, crop_top, crop_left + tw, crop_top + th))

    def _preprocess_cond_image(self, pil_image, processor):
        """Preprocess cond image → JointImageInfo with dual VAE+ViT tensors."""
        from .hunyuan_image3_tokenizer import ImageInfo, JointImageInfo

        pil_image = pil_image.convert("RGB")
        orig_width, orig_height = pil_image.size

        # Resolution lookup
        hf_config = self.ar_model.hf_config
        vae_factor = getattr(hf_config, "vae_downsample_factor", [16, 16])
        if isinstance(vae_factor, (list, tuple)):
            vae_h = vae_factor[0]
            vae_w = vae_factor[1] if len(vae_factor) > 1 else vae_factor[0]
        else:
            vae_h = vae_w = int(vae_factor)
        vae_w_factor = vae_w
        vae_h_factor = vae_h

        if processor is not None and hasattr(processor, "reso_group"):
            base_size, ratio_idx = processor.reso_group.get_base_size_and_ratio_index(
                orig_width, orig_height
            )
            base_size = int(base_size)
            ratio_idx = int(ratio_idx)
            reso = processor.reso_group[ratio_idx]
            target_width = int(reso.width)
            target_height = int(reso.height)
        else:
            base_size = 1024
            ratio_idx = 0
            target_width = (orig_width // vae_w_factor) * vae_w_factor
            target_height = (orig_height // vae_h_factor) * vae_h_factor

        vae_input = self._resize_and_crop_center(pil_image, target_width, target_height)
        if processor is not None and hasattr(processor, "vae_processor"):
            vae_tensor = processor.vae_processor(vae_input)
        else:
            import torchvision.transforms as T
            # Match vllm-omni HunyuanImage3ImageProcessor: normalize to [-1, 1]
            vae_tensor = T.Compose([
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ])(vae_input).unsqueeze(0)

        vae_info = ImageInfo(
            image_type="vae",
            image_width=target_width,
            image_height=target_height,
            token_width=target_width // vae_w_factor,
            token_height=target_height // vae_h_factor,
            base_size=base_size,
            ratio_index=ratio_idx,
        )

        vit_patch_size = 1
        if processor is not None and hasattr(processor, "vision_encoder_processor"):
            vit_inputs = processor.vision_encoder_processor(pil_image, return_tensors="pt")
            vit_tensor = vit_inputs["pixel_values"].squeeze(0)
            spatial_shapes = vit_inputs["spatial_shapes"].squeeze(0)
            pixel_attention_mask = vit_inputs["pixel_attention_mask"].squeeze(0)
            vit_token_h = int(spatial_shapes[0].item())
            vit_token_w = int(spatial_shapes[1].item())
            vit_patch_size = getattr(processor.vision_encoder_processor, "patch_size", 1)
            if isinstance(vit_patch_size, (tuple, list)):
                vit_patch_size = int(vit_patch_size[0])
        else:
            vit_config = getattr(hf_config, "vit", None)
            if vit_config is None:
                vit_config = {}
            vit_num_channels = vit_config.get("num_channels", 3) if isinstance(vit_config, dict) else getattr(vit_config, "num_channels", 3)
            vit_hidden_patch_size = vit_config.get("patch_size", 14) if isinstance(vit_config, dict) else getattr(vit_config, "patch_size", 14)
            vit_feat_dim = vit_num_channels * vit_hidden_patch_size * vit_hidden_patch_size
            vit_patch_size = vit_hidden_patch_size

            fallback_w = (orig_width // vit_patch_size) * vit_patch_size
            fallback_h = (orig_height // vit_patch_size) * vit_patch_size
            fallback_w = max(fallback_w, vit_patch_size)
            fallback_h = max(fallback_h, vit_patch_size)
            resized = self._resize_and_crop_center(pil_image, fallback_w, fallback_h)

            import torchvision.transforms as T
            resized_tensor = T.ToTensor()(resized)
            vit_token_h = fallback_h // vit_patch_size
            vit_token_w = fallback_w // vit_patch_size

            patches = resized_tensor.unfold(1, vit_patch_size, vit_patch_size).unfold(2, vit_patch_size, vit_patch_size)
            patches = patches.permute(1, 2, 0, 3, 4).reshape(vit_token_h * vit_token_w, vit_feat_dim)
            vit_tensor = patches
            spatial_shapes = torch.tensor([vit_token_h, vit_token_w])
            pixel_attention_mask = torch.ones(vit_token_h * vit_token_w, dtype=torch.long)

        vit_info = ImageInfo(
            image_type="siglip2",
            image_width=vit_token_w * vit_patch_size,
            image_height=vit_token_h * vit_patch_size,
            token_width=vit_token_w,
            token_height=vit_token_h,
            image_token_length=int(vit_tensor.shape[0]),
        )

        joint_info = JointImageInfo(
            vae_image_info=vae_info,
            vision_image_info=vit_info,
            vision_encoder_kwargs={
                "spatial_shapes": spatial_shapes,
                "pixel_attention_mask": pixel_attention_mask,
            },
        )
        vae_info.image_tensor = vae_tensor
        vit_info.image_tensor = vit_tensor
        return joint_info, vae_tensor, vit_tensor, joint_info.vision_encoder_kwargs

    def _vae_encode_cond_image(self, vae_tensor, device):
        """VAE-encode cond image → (t=0, latents)."""
        if vae_tensor.ndim == 3:
            vae_tensor = vae_tensor.unsqueeze(0)
        if vae_tensor.ndim == 4:
            vae_tensor = vae_tensor.unsqueeze(2)

        vae = self._vae
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            result = vae.encode(vae_tensor.to(device))
            if isinstance(result, torch.Tensor):
                latents = result
            else:
                latents = result.latent_dist.sample()
            config = vae.config
            if hasattr(config, "shift_factor") and config.shift_factor:
                latents.sub_(config.shift_factor)
            if hasattr(config, "scaling_factor") and config.scaling_factor:
                latents.mul_(config.scaling_factor)

        if hasattr(vae, "ffactor_temporal"):
            latents = latents.squeeze(2)

        t = torch.zeros((latents.shape[0],))
        return t, latents.squeeze(0)

    def _encode_cond_images(self, cond_image_infos, cfg_factor, device, generator=None):
        """Encode cond images through VAE+ViT, return tensors and vit_kwargs."""
        cond_vae_list, cond_t_list, cond_vit_list = [], [], []
        for info in cond_image_infos:
            t, latents = self._vae_encode_cond_image(info.vae_image_info.image_tensor, device)
            cond_vit_list.append(info.vision_image_info.image_tensor)
            cond_vae_list.append(latents)
            cond_t_list.append(t)

        cond_t = torch.cat(cond_t_list, dim=0)
        cond_vit_images = torch.stack(cond_vit_list, dim=0)

        if all(v.shape == cond_vae_list[0].shape for v in cond_vae_list):
            cond_vae_images = torch.stack(cond_vae_list, dim=0)
        else:
            cond_vae_images = cond_vae_list

        if cfg_factor > 1:
            cond_t = cond_t.repeat(cfg_factor)
            if isinstance(cond_vae_images, torch.Tensor):
                cond_vae_images = cond_vae_images.repeat(cfg_factor, 1, 1, 1)
            else:
                cond_vae_images = cond_vae_images * cfg_factor
            cond_vit_images = cond_vit_images.repeat(cfg_factor, 1, 1)

        vit_kwargs = {"spatial_shapes": [], "attention_mask": []}
        for info in cond_image_infos:
            vit_kwargs["spatial_shapes"].append(info.vision_encoder_kwargs["spatial_shapes"])
            vit_kwargs["attention_mask"].append(info.vision_encoder_kwargs["pixel_attention_mask"])
        vit_kwargs["spatial_shapes"] = torch.stack(vit_kwargs["spatial_shapes"])
        vit_kwargs["attention_mask"] = torch.stack(vit_kwargs["attention_mask"])
        if cfg_factor > 1:
            vit_kwargs["spatial_shapes"] = vit_kwargs["spatial_shapes"].repeat(cfg_factor, *([1] * (vit_kwargs["spatial_shapes"].ndim - 1)))
            vit_kwargs["attention_mask"] = vit_kwargs["attention_mask"].repeat(cfg_factor, *([1] * (vit_kwargs["attention_mask"].ndim - 1)))

        return cond_vae_images, cond_t, cond_vit_images, vit_kwargs

    def _instantiate_cond_vae_tokens(
        self, hidden_states, cond_vae_images, cond_timesteps, cond_vae_image_mask,
    ):
        """Scatter VAE cond image embeddings at cond_vae_image_mask positions."""
        bsz, seq_len, n_embd = hidden_states.shape
        index = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)

        if isinstance(cond_vae_images, list):
            t_emb = []
            for image_i, t_i in zip(cond_vae_images, cond_timesteps):
                t_i_emb = self.ar_model.time_embed(t_i.unsqueeze(0).to(hidden_states.device))
                if image_i.dim() == 3:
                    image_i = image_i.unsqueeze(0)
                image_i_seq, _, _ = self.ar_model.patch_embed(image_i.to(hidden_states.device), t_i_emb)
                scatter_idx = index[0:1].masked_select(cond_vae_image_mask[0:1].bool()).reshape(1, -1)
                hidden_states = hidden_states.clone()
                hidden_states[0:1].scatter_(
                    dim=1,
                    index=scatter_idx.unsqueeze(-1).repeat(1, 1, n_embd),
                    src=image_i_seq.reshape(1, -1, n_embd),
                )
                t_emb.append(t_i_emb)
        else:
            t_emb = self.ar_model.time_embed(cond_timesteps.to(hidden_states.device))
            image_seq, _, _ = self.ar_model.patch_embed(cond_vae_images.to(hidden_states.device), t_emb)
            scatter_idx = index.masked_select(cond_vae_image_mask.bool()).reshape(bsz, -1)
            hidden_states = hidden_states.clone()
            hidden_states.scatter_(
                dim=1,
                index=scatter_idx.unsqueeze(-1).repeat(1, 1, n_embd),
                src=image_seq,
            )
        return hidden_states

    def _instantiate_cond_vit_tokens(
        self, hidden_states, cond_vit_images, cond_vit_image_mask, vit_kwargs,
    ):
        """Run ViT+aligner, scatter at cond_vit_image_mask positions."""
        bsz, seq_len, n_embd = hidden_states.shape
        index = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)

        cond_vit_embeds = []
        for batch_idx, image in enumerate(cond_vit_images):
            cur_kwargs = {k: v[batch_idx] for k, v in vit_kwargs.items()}
            if cur_kwargs["spatial_shapes"].ndim == 1:
                cur_kwargs["spatial_shapes"] = cur_kwargs["spatial_shapes"].unsqueeze(0)
            if cur_kwargs["attention_mask"].ndim == 1:
                cur_kwargs["attention_mask"] = cur_kwargs["attention_mask"].unsqueeze(0)
            image_embed = self._vision_model(
                image.unsqueeze(0).to(hidden_states.device),
                **cur_kwargs,
            )
            image_embed = self._vision_aligner(image_embed)
            n, sl, dim = image_embed.shape
            image_embed = image_embed.reshape(n * sl, dim)
            cond_vit_embeds.append(image_embed)

        for i, (embed, mask) in enumerate(zip(cond_vit_embeds, cond_vit_image_mask)):
            scatter_idx = index[i:i+1].masked_select(mask.bool()).reshape(1, -1)
            hidden_states = hidden_states.clone()
            hidden_states[i:i+1].scatter_(
                dim=1,
                index=scatter_idx.unsqueeze(-1).repeat(1, 1, n_embd),
                src=embed.reshape(1, -1, n_embd),
            )
        return hidden_states

    def _instantiate_cond_timestep_tokens(
        self, hidden_states, cond_timesteps, cond_timestep_scatter_index,
    ):
        """Scatter cond timestep embeddings."""
        return self._instantiate_timestep_tokens(
            hidden_states, cond_timesteps, cond_timestep_scatter_index,
        )

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Run the native diffusion loop and store the final latents."""
        # 1. Resolve tokenizer and processor
        tokenizer = self._resolve_custom_tokenizer(server_args)
        processor = self._resolve_processor(server_args)

        # 2. Determine image resolution
        width, height = align_hunyuan_image3_resolution(batch.width, batch.height)
        if processor is not None:
            image_info = processor.build_gen_image_info(f"{height}x{width}")
            height = image_info.image_height
            width = image_info.image_width
            token_h = image_info.token_height
            token_w = image_info.token_width
            # Ensure ImageInfo uses the tokenizer's module class so that
            # isinstance checks inside the tokenizer succeed.
            ImageInfoCls = self._get_image_info_class(tokenizer)
            if ImageInfoCls is not None:
                image_info = self._rebuild_image_info(image_info, ImageInfoCls)
        else:
            # Fallback: compute from VAE downsample factor
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

        # 2b. Ensure the AR model lives on the compute device.
        # When cpu_offload is enabled the pipeline loads weights on CPU;
        # we must move them to the accelerator before running inference.
        device = get_local_torch_device()
        model_device = self.ar_model.model.embed_tokens.weight.device
        if model_device.type == "cpu":
            logger.info("Moving AR model from CPU to %s", device)
            self.ar_model.to(device)
        else:
            device = model_device

        # 3. Build input sequence using the custom tokenizer
        batch_size = 1

        # Resolve guidance_scale with priority:
        # 1. User-explicit value from sampling params
        # 2. Model generation_config default (diff_guidance_scale)
        # 3. Hardcoded fallback (_DEFAULT_GUIDANCE_SCALE)
        sp = batch.sampling_params
        user_explicit_fields = getattr(sp, "_explicit_fields", set()) if sp else set()
        if "guidance_scale" in user_explicit_fields:
            guidance_scale = float(sp.guidance_scale)
            _guidance_source = "user"
        else:
            guidance_scale = self._read_guidance_scale()
            _guidance_source = "generation_config"
        do_cfg = guidance_scale > 1.0
        cfg_factor = 2 if do_cfg else 1

        # Get bot_task and system_prompt from batch (with defaults)
        bot_task = getattr(batch, "bot_task", "image")
        system_prompt = getattr(batch, "system_prompt", "en_unified")
        cot_text = getattr(batch, "cot_text", None)

        # Handle "none" bot_task (convert to "image" for tokenizer compatibility)
        if bot_task == "none":
            bot_task = "image"

        # Normalize bot_task for the tokenizer (matches vllm-omni)
        tokenizer_bot_task = bot_task
        if tokenizer_bot_task == "think_recaption":
            tokenizer_bot_task = "think"
        elif tokenizer_bot_task == "vanilla":
            tokenizer_bot_task = "image"

        # Build tokenizer inputs
        tokenizer_kwargs: dict[str, Any] = dict(
            batch_prompt=[batch.prompt],
            mode="gen_image",
            bot_task=tokenizer_bot_task,
            sequence_template=self._get_sequence_template(),
            drop_think=self._drop_think,
            cfg_factor=cfg_factor,
            image_base_size=getattr(
                processor, "vae_reso_group", None
            ) and processor.vae_reso_group.base_size,
        )

        # Resolve system prompt (preset name → text, or raw text as-is)
        resolved_prompt = _resolve_system_prompt(system_prompt, bot_task=tokenizer_bot_task)
        if resolved_prompt is not None:
            tokenizer_kwargs["batch_system_prompt"] = [resolved_prompt.strip()]

        # Pass CoT text if provided (think/recaption output)
        if cot_text is not None:
            tokenizer_kwargs["batch_cot_text"] = [cot_text]

        # Provide gen image info if the tokenizer supports it
        if image_info is not None:
            tokenizer_kwargs["batch_gen_image_info"] = [image_info]

        # --- Conditional image handling (TI2I / I2I) ---
        cond_image_infos_list = None
        raw_cond_images = getattr(batch, "condition_image", None)
        # Fall back to image_path if condition_image is absent
        if raw_cond_images is None:
            image_path = getattr(batch, "image_path", None)
            if image_path is not None:
                if isinstance(image_path, list):
                    raw_cond_images = image_path
                else:
                    raw_cond_images = [image_path]
        if raw_cond_images is not None:
            if not isinstance(raw_cond_images, (list, tuple)):
                raw_cond_images = [raw_cond_images]
            cond_joint_infos = []
            for raw_img in raw_cond_images:
                from PIL import Image as PILImage
                if not isinstance(raw_img, PILImage.Image):
                    if isinstance(raw_img, str):
                        raw_img = PILImage.open(raw_img)
                    elif isinstance(raw_img, torch.Tensor):
                        raw_img = PILImage.fromarray(
                            raw_img.cpu().permute(1, 2, 0).numpy()
                        )
                joint_info, _, _, _ = self._preprocess_cond_image(raw_img, processor)
                cond_joint_infos.append(joint_info)
            cond_image_infos_list = [cond_joint_infos]
            tokenizer_kwargs["batch_cond_image_info"] = cond_image_infos_list

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

        # --- Conditional image masks / indices from tokenizer ---
        cond_vae_image_mask = None
        cond_vit_image_mask = None
        cond_timestep_scatter_index = None
        if cond_image_infos_list is not None:
            if hasattr(tokenizer_output, "cond_vae_image_mask"):
                cond_vae_image_mask = tokenizer_output.cond_vae_image_mask.to(device)
            if hasattr(tokenizer_output, "cond_vit_image_mask"):
                cond_vit_image_mask = tokenizer_output.cond_vit_image_mask.to(device)
            if hasattr(tokenizer_output, "cond_timestep_scatter_index"):
                cond_timestep_scatter_index = tokenizer_output.cond_timestep_scatter_index.to(device)

            # Expand cond masks for CFG (tokenizer produces per-sample masks)
            if do_cfg and actual_batch_size > 1:
                if cond_vae_image_mask is not None:
                    if cond_vae_image_mask.ndim == 1:
                        cond_vae_image_mask = cond_vae_image_mask.unsqueeze(0).expand(actual_batch_size, -1)
                    elif cond_vae_image_mask.shape[0] == 1:
                        cond_vae_image_mask = cond_vae_image_mask.expand(actual_batch_size, -1)
                if cond_vit_image_mask is not None:
                    if cond_vit_image_mask.ndim == 1:
                        cond_vit_image_mask = cond_vit_image_mask.unsqueeze(0).expand(actual_batch_size, -1)
                    elif cond_vit_image_mask.shape[0] == 1:
                        cond_vit_image_mask = cond_vit_image_mask.expand(actual_batch_size, -1)
                if cond_timestep_scatter_index is not None:
                    if cond_timestep_scatter_index.ndim == 1:
                        cond_timestep_scatter_index = cond_timestep_scatter_index.unsqueeze(0).expand(actual_batch_size, -1)
                    elif cond_timestep_scatter_index.shape[0] == 1:
                        cond_timestep_scatter_index = cond_timestep_scatter_index.expand(actual_batch_size, -1)

        # 4. Build attention mask (4D causal + full attn at image positions)
        # Matches vllm-omni: combine joint_image_slices + gen_image_slices
        gen_slices = getattr(tokenizer_output, "gen_image_slices", [[] for _ in range(actual_batch_size)])
        joint_slices = getattr(tokenizer_output, "joint_image_slices", [[] for _ in range(actual_batch_size)])
        if not isinstance(gen_slices[0], list):
            gen_slices = [gen_slices]
        if not isinstance(joint_slices[0], list):
            joint_slices = [joint_slices]
        image_slices = [joint_slices[i] + gen_slices[i] for i in range(actual_batch_size)]
        attention_mask, full_attn_spans = _build_causal_attention_mask(
            actual_batch_size, seq_len, image_slices, device
        )

        # Non-first-step attention mask: shorter sequence [timestep_tok, image_toks...]
        non_first_seq_len = 1 + num_image_tokens
        non_first_image_slices = [
            [slice(1, non_first_seq_len)]
            for _ in range(actual_batch_size)
        ]
        non_first_attention_mask, _ = _build_causal_attention_mask(
            actual_batch_size, non_first_seq_len, non_first_image_slices, device
        )

        # 5. Build 2D RoPE image info and compute cached cos/sin
        # Extract sections from tokenizer output for per-image RoPE dims
        tokenizer_sections = None
        if isinstance(tokenizer_output_dict, dict):
            tokenizer_sections = tokenizer_output_dict.get("sections")
            # CFG batching may produce a list-of-lists; flatten to a single list
            if tokenizer_sections and isinstance(tokenizer_sections[0], list):
                tokenizer_sections = tokenizer_sections[0]
        rope_image_info = _build_rope_image_info(
            tokenizer_output, actual_batch_size, token_h, token_w, image_info,
            sections=tokenizer_sections,
        )
        cos, sin = self.ar_model.cached_rope(seq_len, device, rope_image_info=rope_image_info)

        # Pre-build RoPE for non-first steps (shorter sequence: 1 timestep + image tokens).
        non_first_rope_info: list[list[tuple[slice, tuple[int, int]]]] = [
            [(slice(1, non_first_seq_len), (token_h, token_w))]
            for _ in range(actual_batch_size)
        ]
        non_first_cos, non_first_sin = self.ar_model.cached_rope(
            non_first_seq_len, device, rope_image_info=non_first_rope_info
        )

        # Fix non-first-step RoPE for the timestep token at position 0.
        # In build_2d_rope, position 0 (before the first image slice) gets
        # y=x=0, producing zero RoPE (cos=1, sin=0).
        # In vLLM-omni, the timestep token stays at its text-prefix position
        # and receives text-position RoPE.  Override position 0 with the
        # timestep token's original RoPE from the first-step tensors.
        if timestep_index is not None:
            if timestep_index.dtype == torch.bool:
                _ts_pos = int(timestep_index[0].float().argmax().item())
            else:
                _ts_pos = int(timestep_index[0, 0].item())
            # cos shape: [batch_size, seq_len, n_elem//2]
            _ts_rope_cos = cos[0, _ts_pos : _ts_pos + 1].clone()
            _ts_rope_sin = sin[0, _ts_pos : _ts_pos + 1].clone()
            non_first_cos = non_first_cos.clone()
            non_first_sin = non_first_sin.clone()
            non_first_cos[:, 0:1] = _ts_rope_cos
            non_first_sin[:, 0:1] = _ts_rope_sin

        # 6. Set up the diffusion scheduler
        num_inference_steps = int(
            getattr(batch, "num_inference_steps", None) or _DEFAULT_NUM_INFERENCE_STEPS
        )


        scheduler = self._scheduler
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps

        # 7. Prepare noise latents
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

        # Generate base noise with batch_size=1, then duplicate for CFG
        # (matching vllm-omni which uses the SAME noise for cond/uncond)
        latents = torch.randn(
            1,
            latent_channels,
            latent_h,
            latent_w,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )

        # Backbone forward agent (bound to num_image_tokens for KV cache)
        backbone_fn = partial(self._backbone_forward, num_image_tokens)

        # 7b. Encode conditional images (TI2I / I2I)
        cond_vae_images = None
        cond_vit_images = None
        cond_t = None
        vit_kwargs = None
        if cond_image_infos_list is not None and cond_image_infos_list[0]:
            if self._vision_model is not None:
                if not isinstance(self._vision_model, torch.nn.Module):
                    self._vision_model.to(device)
                self._vision_model.eval()
            if self._vision_aligner is not None:
                if not isinstance(self._vision_aligner, torch.nn.Module):
                    self._vision_aligner.to(device)
                self._vision_aligner.eval()

            cond_vae_images, cond_t, cond_vit_images, vit_kwargs = self._encode_cond_images(
                cond_image_infos_list[0], cfg_factor, device, generator,
            )

        # 8. Diffusion sampling loop
        # Keep a reference to the original input_ids so that every denoising
        # step can rebuild the full sequence (text + image + special tokens).
        first_step_input_ids = input_ids

        for step_idx, t in enumerate(timesteps):
            first_step = step_idx == 0

            # Scale model input for scheduler
            latent_model_input = scheduler.scale_model_input(latents, t)
            # Duplicate latents for CFG (same noise for both branches)
            if do_cfg:
                latent_model_input = torch.cat([latent_model_input] * cfg_factor, dim=0)

            # Prepare timestep tensor – match latent batch size
            latent_bs = latent_model_input.shape[0]
            t_expand = t.repeat(latent_bs).to(device)

            with torch.autocast(device_type=current_platform.device_type, dtype=torch.bfloat16, enabled=True):
                # Build hidden_states with the SAME sequence structure on every
                # step.  The model was trained with the full special-token
                # layout (<boi>, timestep, <eoi>, shape tokens, <sep>) and
                # produces garbage when the sequence is shortened.
                # Re-embed the original input_ids to get a fresh base, then
                # scatter the updated image + timestep embeddings on top.
                hidden_states = self.ar_model.model.get_input_embeddings(
                    first_step_input_ids,
                ).expand(latent_bs, -1, -1)
                # Scatter VAE image embeddings at image positions
                hidden_states = self._instantiate_vae_tokens_first_step(
                    hidden_states, latent_model_input, t_expand, image_mask,
                )
                # Scatter timestep embedding
                if timestep_index is not None:
                    hidden_states = self._instantiate_timestep_tokens(
                        hidden_states, t_expand, timestep_index,
                    )

                # --- Scatter conditional image embeddings (TI2I / I2I) ---
                if cond_vae_images is not None and cond_vae_image_mask is not None:
                    hidden_states = self._instantiate_cond_vae_tokens(
                        hidden_states, cond_vae_images, cond_t, cond_vae_image_mask,
                    )
                if cond_vit_images is not None and cond_vit_image_mask is not None:
                    hidden_states = self._instantiate_cond_vit_tokens(
                        hidden_states, cond_vit_images, cond_vit_image_mask, vit_kwargs,
                    )
                if cond_timestep_scatter_index is not None and cond_t is not None:
                    hidden_states = self._instantiate_cond_timestep_tokens(
                        hidden_states, cond_t.to(hidden_states.device), cond_timestep_scatter_index,
                    )

                # Use the same RoPE and attention mask on every step
                step_cos, step_sin = cos, sin
                step_attn_mask = attention_mask

                # Run backbone (always use first_step=True so the attention
                # meta matches the full-sequence layout)
                backbone_out = backbone_fn(
                    hidden_states, step_attn_mask, (step_cos, step_sin), True,
                )

                # Extract diffusion prediction via image_mask (same for all steps)
                pred = self._extract_diffusion_pred(
                    backbone_out, t_expand, image_mask,
                    token_h, token_w, first_step=True,
                    num_special_tokens=seq_len - num_image_tokens,
                )

            pred = pred.float()

            # Classifier-free guidance
            if do_cfg:
                pred_cond, pred_uncond = pred.chunk(2)
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # Scheduler step (latents is always batch_size=1)
            latent_dtype = latents.dtype
            latents = scheduler.step(pred, t, latents, return_dict=False)[0].to(dtype=latent_dtype)

        # 9. Store latents for the decoding stage.
        # The denoising loop produces latents in the VAE-encoded space.
        # The decoding stage's ``scale_and_shift`` will convert them to
        # raw VAE space (``latents / scaling_factor + shift_factor``)
        # before calling ``vae.decode``.  We only need to add the temporal
        # dimension expected by the 3D VAE: [B, C, H, W] -> [B, C, 1, H, W].
        batch.latents = latents.to(torch.bfloat16).unsqueeze(2)

        logger.info(
            "HunyuanImage3AR produced latents %s for %dx%d image",
            tuple(batch.latents.shape),
            height,
            width,
        )
        return batch
