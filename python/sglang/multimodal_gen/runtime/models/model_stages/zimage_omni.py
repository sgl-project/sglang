from typing import List, Optional

import torch
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vision_utils import load_image
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class ZImageOmniBeforeDenoisingStage(PipelineStage):
    def __init__(
        self,
        vae,
        vae_scale_factor,
        siglip,
        siglip_processor,
        tokenizer,
        text_encoder,
    ):
        self.vae = vae

        # Copied from diffusers.pipeline_z_image_omni
        self.image_processor = Flux2ImageProcessor(
            vae_scale_factor=vae_scale_factor * 2
        )

        self.vae_scale_factor = vae_scale_factor
        self.siglip = siglip
        self.siglip_processor = siglip_processor
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

    def encode_text(
        self,
        text: str | list[str],
        server_args: ServerArgs,
        num_condition_images: int = 0,
    ):
        # Normalize input to list[str]
        assert isinstance(text, str | list)
        if isinstance(text, str):
            texts: list[str] = [text]
        else:
            texts = text

        target_device = get_local_torch_device()

        assert (
            len(server_args.pipeline_config.text_encoder_configs) == 1
        ), "Should be single encoder."
        preprocess_func = server_args.pipeline_config.preprocess_text_funcs[0]
        postprocess_func = server_args.pipeline_config.postprocess_text_funcs[0]
        encoder_config = server_args.pipeline_config.text_encoder_configs[0]
        text_encoder_extra_arg = server_args.pipeline_config.text_encoder_extra_args[0]

        processed_text_list: list[str] = []
        for prompt_str in texts:
            preprocessed = preprocess_func(prompt_str)
            processed_text_list.append(preprocessed)

        tok_kwargs = {
            **encoder_config.tokenizer_kwargs,
            **text_encoder_extra_arg,
            "num_condition_images": num_condition_images,
        }

        text_inputs: dict = server_args.pipeline_config.tokenize_prompt(
            processed_text_list, self.tokenizer, tok_kwargs
        ).to(target_device)

        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]

        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs: BaseEncoderOutput = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        prompt_embeds = postprocess_func(outputs, text_inputs)

        return [prompt_embeds]

    def _preprocess_image(
        self,
        image_path: Optional[List[str]],
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
    ):
        default_height, default_width = 1024, 1024
        if image_path is None:
            # if no image input
            # use default h,w
            return None, None, (default_height, default_width)

        condition_image = []
        resized_images = []

        image_height, image_width = None, None
        for path in image_path:
            if path.endswith(".mp4"):
                raise NotImplementedError("unimplemented")
                # img = load_video(path)[0]
            else:
                img = load_image(path)

            # check img input
            self.image_processor.check_image_input(img)

            image_width, image_height = img.size
            # resize to (1024, 1024) if (height, width) is not define
            if image_width * image_height > 1024 * 1024:
                if target_width is not None and target_width is not None:
                    img = self.image_processor._resize_to_target_area(
                        img, target_height * target_width
                    )
                else:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                image_width, image_height = img.size
            resized_images.append(img)

            multiple_of = self.vae_scale_factor * 2
            image_width = (image_width // multiple_of) * multiple_of
            image_height = (image_height // multiple_of) * multiple_of
            img = self.image_processor.preprocess(
                img, height=image_height, width=image_width, resize_mode="crop"
            )
            condition_image.append(img)

        # if no specific h,w, use input image size
        target_height = target_height or image_height
        target_width = target_width or image_width

        vae_scale = self.vae_scale_factor * 2
        if target_height % vae_scale != 0:
            raise ValueError(
                f"Height must be divisible by {vae_scale} (got {target_height}). "
                f"Please adjust the height to a multiple of {vae_scale}."
            )
        if target_width % vae_scale != 0:
            raise ValueError(
                f"Width must be divisible by {vae_scale} (got {target_width}). "
                f"Please adjust the width to a multiple of {vae_scale}."
            )

        return condition_image, resized_images, (target_height, target_width)

    def _prepare_siglip(
        self,
        resized_images: Optional[List[torch.Tensor]],
        do_classifier_free_guidance: bool = False,
        dtype=None,
    ):
        """
        Args:
            resized_images (list[tensor]): processed image
            do_classifier_free_guidance (bool)
            dtype (dtype): target dtype
        Returns
            pos_latents (tensor)
            neg_latents (tensor)
        """
        if resized_images is None:
            return None, None

        # TODO: hard code
        batch_size = 1
        device = get_local_torch_device()

        siglip_embeds = []

        self.siglip = self.siglip.to(device)

        for image in resized_images:
            siglip_inputs = (
                self.siglip_processor(images=[image], return_tensors="pt")
                .to(device)
                .to(dtype)
            )
            shape = siglip_inputs.spatial_shapes[0]
            hidden_state = self.siglip(**siglip_inputs).last_hidden_state
            B, N, C = hidden_state.shape
            hidden_state = hidden_state[:, : shape[0] * shape[1]]
            hidden_state = hidden_state.view(shape[0], shape[1], C)
            siglip_embeds.append(hidden_state.to(dtype))

        condition_siglip_embeds = [siglip_embeds for _ in range(batch_size)]

        condition_siglip_embeds = [
            None if sels == [] else sels + [None] for sels in condition_siglip_embeds
        ]

        negative_condition_siglip_embeds = None
        if do_classifier_free_guidance:
            negative_condition_siglip_embeds = condition_siglip_embeds

        return condition_siglip_embeds, negative_condition_siglip_embeds

    def _prepare_image_latents(
        self,
        condition_images,
        do_classifier_free_guidance=False,
        dtype=None,
    ):
        """
        Args:
            condition_images (list[tensor]): processed image
            do_classifier_free_guidance (bool)
            dtype (dtype): target dtype
        Returns
            pos_latents (tensor)
            neg_latents (tensor)
        """
        if condition_images is None:
            return None, None

        image_latents = []

        # TODO: hard code
        batch_size = 1
        device = get_local_torch_device()

        self.vae = self.vae.to(device)

        for image in condition_images:
            # TODO: hard code vae fp32
            image = image.to(device=device, dtype=torch.float32)
            image_latent = (
                self.vae.encode(image).latent_dist.mode()[0]
                - self.vae.config.shift_factor
            ) * self.vae.config.scaling_factor
            image_latent = image_latent.unsqueeze(1).to(dtype)
            image_latents.append(image_latent)  # (16, 128, 128)

        condition_latents = [image_latents for _ in range(batch_size)]

        negative_condition_latents = None
        if do_classifier_free_guidance:
            negative_condition_latents = condition_latents

        return condition_latents, negative_condition_latents

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        # Copied from diffusers.pipeline_z_image_omni
        do_classifier_free_guidance = batch.guidance_scale > 1
        batch.do_classifier_free_guidance = do_classifier_free_guidance
        device = get_local_torch_device()
        batch.generator = torch.Generator(device=device).manual_seed(batch.seed)

        # Encode positive prompt with all available encoders
        assert batch.prompt is not None
        assert isinstance(batch.prompt_embeds, list)

        num_condition_images: int = (
            len(batch.image_path) if batch.image_path is not None else 0
        )

        pos_prompt_embeds_list = self.encode_text(
            batch.prompt,
            server_args,
            num_condition_images=num_condition_images,
        )
        for pe in pos_prompt_embeds_list:
            batch.prompt_embeds.append(pe)

        # Encode negative prompt if CFG is enabled
        if batch.do_classifier_free_guidance:
            # Copied from diffusers.pipeline_z_image_omni
            if batch.negative_prompt is None:
                batch.negative_prompt = ""

            neg_prompt_embeds_list = self.encode_text(
                batch.negative_prompt,
                server_args,
                num_condition_images=num_condition_images,
            )
            assert isinstance(batch.negative_prompt_embeds, list)
            for ne in neg_prompt_embeds_list:
                batch.negative_prompt_embeds.append(ne)

        # 3. Process condition images. Copied from diffusers.pipelines.flux2.pipeline_flux2
        condition_image, resized_images, (target_height, target_width) = (
            self._preprocess_image(
                batch.image_path,
                batch.height,
                batch.width,
            )
        )

        # 4. Prepare latent variables
        # prepare siglip
        condition_siglip_embeds, negative_condition_siglip_embeds = (
            self._prepare_siglip(
                resized_images, do_classifier_free_guidance, dtype=torch.float32
            )
        )

        # prepare image latent
        condition_latents, negative_condition_latents = self._prepare_image_latents(
            condition_image, do_classifier_free_guidance, dtype=torch.float32
        )

        batch.height = target_height
        batch.width = target_width

        # siglip
        batch.condition_siglip_embeds = condition_siglip_embeds
        batch.negative_condition_siglip_embeds = negative_condition_siglip_embeds

        # image iatent
        batch.condition_latents = condition_latents
        batch.negative_condition_latents = negative_condition_latents

        return batch
