# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from typing import Any, Dict, Optional
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers

import numpy
import torch
import torch.utils.checkpoint
import torch.distributed
import numpy as np
import transformers
from PIL import Image
from einops import rearrange
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import diffusers
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.image_processor import VaeImageProcessor

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    retrieve_timesteps,
    rescale_noise_cfg,
)

from diffusers.utils import deprecate
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from .unet.modules import UNet2p5DConditionModel
from .unet.attn_processor import SelfAttnProcessor2_0, RefAttnProcessor2_0, PoseRoPEAttnProcessor2_0

__all__ = [
    "HunyuanPaintPipeline",
    "UNet2p5DConditionModel",
    "SelfAttnProcessor2_0",
    "RefAttnProcessor2_0",
    "PoseRoPEAttnProcessor2_0",
]


def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == "RGB":
        return maybe_rgba
    elif maybe_rgba.mode == "RGBA":
        rgba = maybe_rgba
        img = numpy.random.randint(127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=numpy.uint8)
        img = Image.fromarray(img, "RGB")
        img.paste(rgba, mask=rgba.getchannel("A"))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


class HunyuanPaintPipeline(StableDiffusionPipeline):

    """Custom pipeline for multiview PBR texture generation.
    
    Extends Stable Diffusion with:
    - Material-specific conditioning
    - Multiview processing
    - Position-aware attention
    - 2.5D UNet integration
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        feature_extractor: CLIPImageProcessor,
        safety_checker=None,
        use_torch_compile=False,
    ):
        DiffusionPipeline.__init__(self)

        safety_checker = None
        self.register_modules(
            vae=torch.compile(vae) if use_torch_compile else vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=torch.compile(feature_extractor) if use_torch_compile else feature_extractor,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        if isinstance(self.unet, UNet2DConditionModel):
            self.unet = UNet2p5DConditionModel(self.unet, None, self.scheduler)

    def eval(self):
        self.unet.eval()
        self.vae.eval()

    def set_pbr_settings(self, pbr_settings: List[str]):
        self.pbr_settings = pbr_settings

    def set_learned_parameters(self):

        """Configures parameter freezing strategy.
        
        Freezes:
        - Standard attention layers
        - Dual-stream reference UNet
        
        Unfreezes:
        - Material-specific parameters
        - DINO integration components
        """

        freezed_names = ["attn1", "unet_dual"]
        added_learned_names = ["albedo", "mr", "dino"]

        for name, params in self.unet.named_parameters():
            if any(freeze_name in name for freeze_name in freezed_names) and all(
                learned_name not in name for learned_name in added_learned_names
            ):
                params.requires_grad = False
            else:
                params.requires_grad = True

    def prepare(self):
        if isinstance(self.unet, UNet2DConditionModel):
            self.unet = UNet2p5DConditionModel(self.unet, None, self.scheduler).eval()

    @torch.no_grad()
    def encode_images(self, images):

        """Encodes multiview image batches into latent space.
        
        Args:
            images: Input images [B, N_views, C, H, W]
            
        Returns:
            torch.Tensor: Latent representations [B, N_views, C, H_latent, W_latent]
        """

        B = images.shape[0]
        images = rearrange(images, "b n c h w -> (b n) c h w")

        dtype = next(self.vae.parameters()).dtype
        images = (images - 0.5) * 2.0
        posterior = self.vae.encode(images.to(dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        latents = rearrange(latents, "(b n) c h w -> b n c h w", b=B)
        return latents

    @torch.no_grad()
    def __call__(
        self,
        images=None,
        prompt=None,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        *args,
        num_images_per_prompt: Optional[int] = 1,
        guidance_scale=3.0,
        output_type: Optional[str] = "pil",
        width=512,
        height=512,
        num_inference_steps=15,
        return_dict=True,
        sync_condition=None,
        **cached_condition,
    ):

        """Main generation method for multiview PBR textures.
        
        Steps:
        1. Input validation and preparation
        2. Reference image encoding
        3. Condition processing (normal/position maps)
        4. Prompt embedding setup
        5. Classifier-free guidance preparation
        6. Diffusion sampling loop
        
        Args:
            images: List of reference PIL images
            prompt: Text prompt (overridden by learned embeddings)
            cached_condition: Dictionary containing:
                - images_normal: Normal maps (PIL or tensor)
                - images_position: Position maps (PIL or tensor)
            
        Returns:
            List[PIL.Image]: Generated multiview PBR textures
        """

        self.prepare()
        if images is None:
            raise ValueError("Inputting embeddings not supported for this pipeline. Please pass an image.")
        assert not isinstance(images, torch.Tensor)

        if not isinstance(images, List):
            images = [images]

        images = [to_rgb_image(image) for image in images]
        images_vae = [torch.tensor(np.array(image) / 255.0) for image in images]
        images_vae = [image_vae.unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(0) for image_vae in images_vae]
        images_vae = torch.cat(images_vae, dim=1)
        images_vae = images_vae.to(device=self.vae.device, dtype=self.unet.dtype)

        batch_size = images_vae.shape[0]
        N_ref = images_vae.shape[1]

        assert batch_size == 1
        assert num_images_per_prompt == 1

        if self.unet.use_ra:
            ref_latents = self.encode_images(images_vae)
            cached_condition["ref_latents"] = ref_latents

        def convert_pil_list_to_tensor(images):
            bg_c = [1.0, 1.0, 1.0]
            images_tensor = []
            for batch_imgs in images:
                view_imgs = []
                for pil_img in batch_imgs:
                    img = numpy.asarray(pil_img, dtype=numpy.float32) / 255.0
                    if img.shape[2] > 3:
                        alpha = img[:, :, 3:]
                        img = img[:, :, :3] * alpha + bg_c * (1 - alpha)
                    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous().half().to("cuda")
                    view_imgs.append(img)
                view_imgs = torch.cat(view_imgs, dim=0)
                images_tensor.append(view_imgs.unsqueeze(0))

            images_tensor = torch.cat(images_tensor, dim=0)
            return images_tensor

        if "images_normal" in cached_condition:
            if isinstance(cached_condition["images_normal"], List):
                cached_condition["images_normal"] = convert_pil_list_to_tensor(cached_condition["images_normal"])

            cached_condition["embeds_normal"] = self.encode_images(cached_condition["images_normal"])

        if "images_position" in cached_condition:

            if isinstance(cached_condition["images_position"], List):
                cached_condition["images_position"] = convert_pil_list_to_tensor(cached_condition["images_position"])

            cached_condition["position_maps"] = cached_condition["images_position"]
            cached_condition["embeds_position"] = self.encode_images(cached_condition["images_position"])

        if self.unet.use_learned_text_clip:

            all_shading_tokens = []
            for token in self.unet.pbr_setting:
                all_shading_tokens.append(
                    getattr(self.unet, f"learned_text_clip_{token}").unsqueeze(dim=0).repeat(batch_size, 1, 1)
                )
            prompt_embeds = torch.stack(all_shading_tokens, dim=1)
            negative_prompt_embeds = torch.stack(all_shading_tokens, dim=1)
            # negative_prompt_embeds = torch.zeros_like(prompt_embeds)

        else:
            if prompt is None:
                prompt = "high quality"
            if isinstance(prompt, str):
                prompt = [prompt for _ in range(batch_size)]
            device = self._execution_device
            prompt_embeds, _ = self.encode_prompt(
                prompt, device=device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=False
            )

            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt for _ in range(batch_size)]
            if negative_prompt is not None:
                negative_prompt_embeds, _ = self.encode_prompt(
                    negative_prompt,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    do_classifier_free_guidance=False,
                )
            else:
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)

        if guidance_scale > 1:
            if self.unet.use_ra:
                cached_condition["ref_latents"] = cached_condition["ref_latents"].repeat(
                    3, *([1] * (cached_condition["ref_latents"].dim() - 1))
                )
                cached_condition["ref_scale"] = torch.as_tensor([0.0, 1.0, 1.0]).to(cached_condition["ref_latents"])

            if self.unet.use_dino:
                zero_states = torch.zeros_like(cached_condition["dino_hidden_states"])
                cached_condition["dino_hidden_states"] = torch.cat(
                    [zero_states, zero_states, cached_condition["dino_hidden_states"]]
                )

                del zero_states
            if "embeds_normal" in cached_condition:
                cached_condition["embeds_normal"] = cached_condition["embeds_normal"].repeat(
                    3, *([1] * (cached_condition["embeds_normal"].dim() - 1))
                )

            if "embeds_position" in cached_condition:
                cached_condition["embeds_position"] = cached_condition["embeds_position"].repeat(
                    3, *([1] * (cached_condition["embeds_position"].dim() - 1))
                )

            if "position_maps" in cached_condition:
                cached_condition["position_maps"] = cached_condition["position_maps"].repeat(
                    3, *([1] * (cached_condition["position_maps"].dim() - 1))
                )

        images = self.denoise(
            None,
            *args,
            cross_attention_kwargs=None,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
            width=width,
            height=height,
            return_dict=return_dict,
            **cached_condition,
        )

        return images

    def denoise(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`]
                (https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
                
        Core denoising procedure for multiview PBR texture generation.
        
        Handles the complete diffusion process including:
        - Input validation and preparation
        - Timestep scheduling
        - Latent noise initialization
        - Iterative denoising with specialized guidance
        - Output decoding and post-processing
        
        Key innovations:
        1. Triple-batch classifier-free guidance:
           - Negative (unconditional)
           - Reference-conditioned
           - Full-conditioned
        2. View-dependent guidance scaling:
           - Adjusts influence based on camera azimuth
        3. PBR-aware latent organization:
           - Maintains material/view separation throughout
        4. Optimized VRAM management:
           - Selective tensor reshaping
        
        Processing Stages:
        1. Setup & Validation: Configures pipeline components and validates inputs
        2. Prompt Encoding: Processes text/material conditioning
        3. Latent Initialization: Prepares noise for denoising process
        4. Iterative Denoising: 
            a) Scales and organizes latent variables
            b) Predicts noise at current timestep
            c) Applies view-dependent guidance
            d) Computes previous latent state
        5. Output Decoding: Converts latents to final images
        6. Cleanup: Releases resources and formats output
        
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        # open cache
        kwargs["cache"] = {}

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated,"
                "consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated,"
                "consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None

        """
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )'
        """

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        assert num_images_per_prompt == 1
        # 5. Prepare latent variables
        n_pbr = len(self.unet.pbr_setting)
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * kwargs["num_in_batch"] * n_pbr,  # num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latents = rearrange(
                    latents, "(b n_pbr n) c h w -> b n_pbr n c h w", n=kwargs["num_in_batch"], n_pbr=n_pbr
                )
                # latent_model_input = torch.cat([latents] * 3) if self.do_classifier_free_guidance else latents
                latent_model_input = latents.repeat(3, 1, 1, 1, 1, 1) if self.do_classifier_free_guidance else latents
                latent_model_input = rearrange(latent_model_input, "b n_pbr n c h w -> (b n_pbr n) c h w")
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = rearrange(
                    latent_model_input, "(b n_pbr n) c h w ->b n_pbr n c h w", n=kwargs["num_in_batch"], n_pbr=n_pbr
                )

                # predict the noise residual

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                    **kwargs,
                )[0]
                latents = rearrange(latents, "b n_pbr n c h w -> (b n_pbr n) c h w")
                # perform guidance
                if self.do_classifier_free_guidance:
                    # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    # noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_pred_uncond, noise_pred_ref, noise_pred_full = noise_pred.chunk(3)

                    if "camera_azims" in kwargs.keys():
                        camera_azims = kwargs["camera_azims"]
                    else:
                        camera_azims = [0] * kwargs["num_in_batch"]

                    def cam_mapping(azim):
                        if azim < 90 and azim >= 0:
                            return float(azim) / 90.0 + 1
                        elif azim >= 90 and azim < 330:
                            return 2.0
                        else:
                            return -float(azim) / 90.0 + 5.0

                    view_scale_tensor = (
                        torch.from_numpy(np.asarray([cam_mapping(azim) for azim in camera_azims]))
                        .unsqueeze(0)
                        .repeat(n_pbr, 1)
                        .view(-1)
                        .to(noise_pred_uncond)[:, None, None, None]
                    )
                    noise_pred = noise_pred_uncond + self.guidance_scale * view_scale_tensor * (
                        noise_pred_ref - noise_pred_uncond
                    )
                    noise_pred += self.guidance_scale * view_scale_tensor * (noise_pred_full - noise_pred_ref)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_ref, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents[:, :num_channels_latents, :, :], **extra_step_kwargs, return_dict=False
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
