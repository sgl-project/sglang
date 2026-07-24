# SPDX-License-Identifier: Apache-2.0
# Ported from the LingBot-Video reference pipeline:
#   /vllm-workspace/lingbot-video/lingbot_video/pipeline_lingbot_video.py
#
# Parity-critical: PROMPT_TEMPLATE / DEFAULT_NEGATIVE_PROMPT / tokenization
# arguments / `_compute_crop_start` are copied VERBATIM from upstream. The
# DiT is trained on structured-JSON captions carried in `batch.prompt`; raw
# free-text prompts are out-of-distribution and produce garbage.


import torch
from diffusers.utils.torch_utils import randn_tensor
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

TOKEN_LENGTH = 37698
HIDDEN_STATE_SKIP_LAYER = 0

PROMPT_TEMPLATE = (
    "<|im_start|>system\nGiven a user input that may include a text prompt alone, "
    "a text prompt with an image reference, or a text prompt with a video reference "
    'or a video reference alone, generate an "Enhanced prompt" that provides detailed '
    "visual descriptions suitable for video generation. Evaluate the level of detail "
    "in the user's input: if it is simple, enrich it by adding specifics about colors, "
    "shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal "
    "progression, and spatial relationships to create vivid, concrete, and temporally "
    "coherent scenes to create vivid and concrete scenes. Please generate only the "
    "enhanced description for the prompt below and avoid including any additional "
    "commentary or evaluations:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
IMG_PROMPT_TEMPLATE = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PROMPT_TEMPLATE = "<|vision_start|><|video_pad|><|vision_end|>"

DEFAULT_NEGATIVE_PROMPT = '{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution", "unstable color", "color flicker", "underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", "signature", "logo", "subtitles", "pillarboxed", "side bars", "portrait image in landscape frame"], "temporal_and_motion_stability": ["flickering", "jittery", "motion blur", "temporal inconsistency", "warping", "morphing", "incoherent motion", "unnatural movement", "static object with sudden jump", "frame-to-frame inconsistency"], "material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "liquid freezing improperly", "distorted reflections"]}}'


class LingBotVideoBeforeDenoisingStage(PipelineStage):
    """Monolithic pre-denoising stage for LingBot-Video MoE (T2V, base).

    Consolidates all upstream pre-processing into a single stage that produces a
    ``Req`` batch with every field the standard ``DenoisingStage`` expects:
      * Qwen3-VL prompt/negative encoding (system-template wrap -> tokenize ->
        hidden_states[-1] -> strip system prefix -> trim to mask for B=1)
      * fp32 noise latents [1, 16, (num_frames-1)//4+1, H//8, W//8]
      * FlowUniPC timesteps/sigmas at ``shift=flow_shift`` (sigmas as a Python list)

    The ``processor`` constructor argument holds the Qwen3-VL processor (the
    model_index.json module is named ``processor``, not ``tokenizer``).
    """

    def __init__(self, vae, text_encoder, processor, transformer, scheduler):
        super().__init__()
        self.vae = vae
        self.text_encoder = text_encoder
        self.processor = processor
        self.transformer = transformer
        self.scheduler = scheduler

        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 8
        self.token_length = TOKEN_LENGTH
        self.hidden_state_skip_layer = HIDDEN_STATE_SKIP_LAYER
        self.prompt_template = PROMPT_TEMPLATE
        self._crop_start: int | None = None

    # ------------------------------------------------------------------ utils

    @staticmethod
    def check_inputs(height: int, width: int, num_frames: int) -> None:
        if num_frames != 1 and (num_frames - 1) % 4 != 0:
            raise ValueError(f"`num_frames` must be 1 or 4n+1, got {num_frames}.")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`height` and `width` must be multiples of 16, got {height}x{width}."
            )

    @staticmethod
    def apply_text_to_template(text: str, template: str = PROMPT_TEMPLATE) -> str:
        return template.format(text)

    def _compute_crop_start(self) -> int:
        if self._crop_start is None:
            marker = "<|USER_INPUT_MARKER|>"
            marked = self.prompt_template.format(marker)
            marker_pos = marked.find(marker)
            if marker_pos < 0:
                self._crop_start = 0
            else:
                prefix = self.processor(
                    text=marked[:marker_pos],
                    images=None,
                    videos=None,
                    return_tensors="pt",
                )
                self._crop_start = int(prefix["input_ids"].shape[1])
        return self._crop_start

    def _build_prompt_inputs(self, prompt: str | list[str]):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        # T2V base: no image/video reference, so the visual template is empty.
        texts = [
            self.apply_text_to_template(text, self.prompt_template) for text in prompts
        ]
        return self.processor(
            text=texts,
            images=None,
            videos=None,
            video_metadata=None,
            do_resize=False,
            truncation=True,
            max_length=self.token_length,
            padding="longest",
            return_tensors="pt",
        )

    @torch.no_grad()
    def _encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Encode a prompt into Qwen3-VL hidden states + attention mask.

        Mirrors upstream ``LingBotVideoPipeline.encode_prompt``: wrap in the
        system template, tokenize with max_length=37698 + padding="longest",
        run the Qwen3-VL encoder with ``output_hidden_states=True``, take
        ``hidden_states[-1]``, strip the system-template prefix via
        ``_compute_crop_start``, and (for B=1) drop right padding.
        """
        if self.text_encoder is None or self.processor is None:
            raise ValueError(
                "`text_encoder` and `processor` are required for encode_prompt()."
            )

        inputs = self._build_prompt_inputs(prompt)
        inputs = inputs.to(device)
        outputs = self.text_encoder(
            **inputs,
            output_hidden_states=self.hidden_state_skip_layer is not None,
        )
        if self.hidden_state_skip_layer is not None:
            prompt_embeds = outputs.hidden_states[-(self.hidden_state_skip_layer + 1)]
        else:
            prompt_embeds = outputs.last_hidden_state

        prompt_mask = inputs["attention_mask"]
        crop_start = self._compute_crop_start()
        if crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_mask = prompt_mask[:, crop_start:]

        # Batch=1 can drop right padding before DiT inference.
        if prompt_embeds.shape[0] == 1:
            true_len = int(prompt_mask[0].sum().item())
            prompt_embeds = prompt_embeds[:, :true_len]
            prompt_mask = prompt_mask[:, :true_len]

        return prompt_embeds.to(dtype=dtype), prompt_mask

    def _prepare_latents(
        self,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        in_channels = self.transformer.config.in_channels
        shape = (1, in_channels, latent_frames, latent_height, latent_width)
        return randn_tensor(
            shape, generator=generator, device=device, dtype=torch.float32
        )

    def _prepare_timesteps(
        self, num_inference_steps: int, device: torch.device, shift: float
    ):
        self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
        timesteps = self.scheduler.timesteps
        sigmas = self.scheduler.sigmas.tolist()  # MUST be a Python list
        return timesteps, sigmas

    # --------------------------------------------------------------- forward

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        # DiT compute dtype (bf16). Upstream casts prompt embeds to the
        # transformer dtype before each forward, so we pre-cast here.
        dtype = next(self.transformer.parameters(), torch.tensor([])).dtype
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            dtype = torch.bfloat16

        height = int(batch.height)
        width = int(batch.width)
        num_frames = int(batch.num_frames)
        self.check_inputs(height, width, num_frames)

        guidance_scale = batch.guidance_scale
        flow_shift = (
            batch.flow_shift
            if getattr(batch, "flow_shift", None) is not None
            else server_args.pipeline_config.flow_shift
        )
        num_inference_steps = int(batch.num_inference_steps)

        generator = torch.Generator(device=device).manual_seed(int(batch.seed))

        # 1. Encode prompt + negative prompt (structured-JSON captions).
        prompt_embeds, prompt_mask = self._encode_prompt(batch.prompt, device, dtype)
        negative_prompt = (
            batch.negative_prompt
            if getattr(batch, "negative_prompt", None) is not None
            else DEFAULT_NEGATIVE_PROMPT
        )
        negative_embeds, negative_mask = self._encode_prompt(
            negative_prompt, device, dtype
        )

        # 2. Prepare fp32 noise latents.
        latents = self._prepare_latents(num_frames, height, width, device, generator)

        # 3. Prepare timesteps / sigmas.
        timesteps, sigmas = self._prepare_timesteps(
            num_inference_steps, device, flow_shift
        )

        # 4. Populate the batch contract expected by the standard DenoisingStage.
        batch.prompt_embeds = [prompt_embeds]
        batch.negative_prompt_embeds = [negative_embeds]
        # Tensors (not lists): the DenoisingStage forwards these verbatim as
        # `encoder_attention_mask`, and the LingBotVideo DiT derives text_lens
        # from the mask.
        batch.prompt_attention_mask = prompt_mask
        batch.negative_attention_mask = negative_mask
        batch.latents = latents
        batch.timesteps = timesteps
        batch.num_inference_steps = len(timesteps)
        batch.sigmas = sigmas
        batch.raw_latent_shape = latents.shape
        batch.generator = generator
        batch.scheduler = self.scheduler
        batch.do_classifier_free_guidance = guidance_scale > 1.0

        return batch
