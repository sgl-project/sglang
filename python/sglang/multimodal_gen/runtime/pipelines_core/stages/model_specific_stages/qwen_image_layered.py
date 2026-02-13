import inspect
import math
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vision_utils import load_image
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus.calculate_dimensions
def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if sample_mode == "sample":
        return encoder_output.sample(generator)
    elif sample_mode == "argmax":
        return encoder_output.mode()
    else:
        return encoder_output


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class QwenImageLayeredBeforeDenoisingStage(PipelineStage):
    def __init__(
        self, vae, tokenizer, processor, transformer, scheduler, model_path
    ) -> None:
        super().__init__()
        self.vae = vae.to(torch.bfloat16)
        from transformers import Qwen2_5_VLForConditionalGeneration

        self.text_encoder = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, subfolder="text_encoder"
            )
            .to(get_local_torch_device())
            .to(torch.bfloat16)
        )
        self.tokenizer = tokenizer
        self.processor = processor
        self.transformer = transformer
        self.scheduler = scheduler

        self.vae_scale_factor = (
            2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        )
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self.vl_processor = processor
        self.tokenizer_max_length = 1024
        self.latent_channels = self.vae.z_dim if getattr(self, "vae", None) else 16

        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.image_caption_prompt_cn = """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n# 图像标注器\n你是一个专业的图像标注器。请基于输入图像，撰写图注:\n1.
使用自然、描述性的语言撰写图注，不要使用结构化形式或富文本形式。\n2. 通过加入以下内容，丰富图注细节：\n - 对象的属性：如数量、颜色、形状、大小、位置、材质、状态、动作等\n -
对象间的视觉关系：如空间关系、功能关系、动作关系、从属关系、比较关系、因果关系等\n - 环境细节：例如天气、光照、颜色、纹理、气氛等\n - 文字内容：识别图像中清晰可见的文字，不做翻译和解释，用引号在图注中强调\n3.
保持真实性与准确性：\n - 不要使用笼统的描述\n -
描述图像中所有可见的信息，但不要加入没有在图像中出现的内容\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"""
        self.image_caption_prompt_en = """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n# Image Annotator\nYou are a professional
image annotator. Please write an image caption based on the input image:\n1. Write the caption using natural,
descriptive language without structured formats or rich text.\n2. Enrich caption details by including: \n - Object
attributes, such as quantity, color, shape, size, material, state, position, actions, and so on\n - Vision Relations
between objects, such as spatial relations, functional relations, possessive relations, attachment relations, action
relations, comparative relations, causal relations, and so on\n - Environmental details, such as weather, lighting,
colors, textures, atmosphere, and so on\n - Identify the text clearly visible in the image, without translation or
explanation, and highlight it in the caption with quotation marks\n3. Maintain authenticity and accuracy:\n - Avoid
generalizations\n - Describe all visible information in the image, while do not add information not explicitly shown in
the image\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"""
        self.default_sample_size = 128

    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._extract_masked_hidden
    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def get_image_caption(self, prompt_image, use_en_prompt=True, device=None):
        if use_en_prompt:
            prompt = self.image_caption_prompt_en
        else:
            prompt = self.image_caption_prompt_cn
        model_inputs = self.vl_processor(
            text=prompt,
            images=prompt_image,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            generated_ids = self.text_encoder.generate(
                **model_inputs, max_new_tokens=512
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            output_text = self.vl_processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return output_text.strip()

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt,
            padding=True,
            return_tensors="pt",
        ).to(device)
        encoder_hidden_states = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(
            hidden_states, txt_tokens.attention_mask
        )
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [
            torch.ones(e.size(0), dtype=torch.long, device=e.device)
            for e in split_hidden_states
        ]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
                for u in split_hidden_states
            ]
        )
        encoder_attention_mask = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
                for u in attn_mask_list
            ]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width, layers):
        latents = latents.view(
            batch_size, layers, num_channels_latents, height // 2, 2, width // 2, 2
        )
        latents = latents.permute(0, 1, 3, 5, 2, 4, 6)
        latents = latents.reshape(
            batch_size, layers * (height // 2) * (width // 2), num_channels_latents * 4
        )

        return latents

    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
    ):
        r"""
        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt, device
            )

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask

    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit.QwenImageEditPipeline._encode_vae_image
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        self.vae = self.vae.to(get_local_torch_device())
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(
                    self.vae.encode(image[i : i + 1]),
                    generator=generator[i],
                    sample_mode="argmax",
                )
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(
                self.vae.encode(image), generator=generator, sample_mode="argmax"
            )
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std
        self.vae.to("cpu")
        return image_latents

    def prepare_latents(
        self,
        image,
        batch_size,
        num_channels_latents,
        height,
        width,
        layers,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (
            batch_size,
            layers + 1,
            num_channels_latents,
            height,
            width,
        )  ### the generated first image is combined image

        image_latents = None
        if image is not None:
            image = image.to(device=device, dtype=dtype)
            if image.shape[1] != self.latent_channels:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            else:
                image_latents = image
            if (
                batch_size > image_latents.shape[0]
                and batch_size % image_latents.shape[0] == 0
            ):
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat(
                    [image_latents] * additional_image_per_prompt, dim=0
                )
            elif (
                batch_size > image_latents.shape[0]
                and batch_size % image_latents.shape[0] != 0
            ):
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                image_latents = torch.cat([image_latents], dim=0)

            image_latent_height, image_latent_width = image_latents.shape[3:]
            image_latents = image_latents.permute(
                0, 2, 1, 3, 4
            )  # (b, c, f, h, w) -> (b, f, c, h, w)
            image_latents = self._pack_latents(
                image_latents,
                batch_size,
                num_channels_latents,
                image_latent_height,
                image_latent_width,
                1,
            )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
            latents = self._pack_latents(
                latents, batch_size, num_channels_latents, height, width, layers + 1
            )
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        use_en_prompt = True
        device = get_local_torch_device()
        layers = batch.num_frames
        num_inference_steps = batch.num_inference_steps
        generator = batch.generator

        assert batch.image_path is not None
        image = load_image(batch.image_path[0])
        image = image.convert("RGBA")
        image_size = image.size
        resolution = 640  # TODO: support user-specified resolution
        calculated_width, calculated_height = calculate_dimensions(
            resolution * resolution, image_size[0] / image_size[1]
        )

        height = calculated_height
        width = calculated_width

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        # if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
        image = self.image_processor.resize(image, calculated_height, calculated_width)
        prompt_image = image
        image = self.image_processor.preprocess(
            image, calculated_height, calculated_width
        )
        image = image.unsqueeze(2)
        image = image.to(dtype=torch.bfloat16)

        prompt = self.get_image_caption(
            prompt_image, use_en_prompt=use_en_prompt, device=device
        )

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            device=device,
        )

        negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
            prompt=batch.negative_prompt,
            device=device,
        )

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
            image,
            1,
            num_channels_latents,
            height,
            width,
            layers,
            prompt_embeds.dtype,
            device,
            generator,
        )
        img_shapes = [
            [
                *[
                    (
                        1,
                        height // self.vae_scale_factor // 2,
                        width // self.vae_scale_factor // 2,
                    )
                    for _ in range(layers + 1)
                ],
                (
                    1,
                    calculated_height // self.vae_scale_factor // 2,
                    calculated_width // self.vae_scale_factor // 2,
                ),
            ]
        ]

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 0, num_inference_steps + 1)[:-1]
        image_seq_len = latents.shape[1]
        base_seqlen = 256 * 256 / 16 / 16
        mu = (image_latents.shape[1] / base_seqlen) ** 0.5
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        txt_seq_lens = (
            prompt_embeds_mask.sum(dim=1).tolist()
            if prompt_embeds_mask is not None
            else None
        )
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist()
            if negative_prompt_embeds_mask is not None
            else None
        )
        is_rgb = torch.tensor([0]).to(device=device, dtype=torch.long)

        batch.prompt_embeds = [prompt_embeds]
        batch.prompt_embeds_mask = [prompt_embeds_mask]
        batch.negative_prompt_embeds = [negative_prompt_embeds]
        batch.negative_prompt_embeds_mask = [negative_prompt_embeds_mask]
        batch.latents = latents
        batch.image_latent = image_latents
        batch.num_inference_steps = num_inference_steps
        batch.sigmas = sigmas.tolist()  # Convert numpy array to list for validation
        batch.generator = torch.manual_seed(0)
        batch.original_condition_image_size = image_size
        batch.raw_latent_shape = latents.shape
        batch.txt_seq_lens = txt_seq_lens
        batch.img_shapes = img_shapes

        return batch
