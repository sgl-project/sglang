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
        """ """

        # Normalize input to list[str]
        assert isinstance(text, str | list)
        if isinstance(text, str):
            texts: list[str] = [text]
        else:
            texts = text

        target_device = get_local_torch_device()

        # TODO: review in init?
        preprocess_funcs = server_args.pipeline_config.preprocess_text_funcs
        postprocess_funcs = server_args.pipeline_config.postprocess_text_funcs
        text_encoder_extra_args = server_args.pipeline_config.text_encoder_extra_args
        encoder_cfgs = server_args.pipeline_config.text_encoder_configs
        assert (
            len(server_args.pipeline_config.text_encoder_configs) == 1
        ), "Should be single encoder."
        preprocess_func = preprocess_funcs[0]
        postprocess_func = postprocess_funcs[0]
        encoder_config = encoder_cfgs[0]
        text_encoder_extra_arg = text_encoder_extra_args[0]

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

        # batch.height = height
        # batch.width = width

        # # TODO: hard code debug
        # # should be condition_images(a list)
        # # condition_images -> prepare_image_latents
        # batch.condition_image = condition_image
        # # # prepare for
        # # # resized_images -> prepare_siglip_embeds
        # # batch.resized_images = resized_images

    def _prepare_siglip(
        self,
        resized_images: Optional[List[torch.Tensor]],
        do_classifier_free_guidance: bool = False,
    ):
        if resized_images is None:
            return None, None

        # TODO: hard code
        batch_size = 1
        device = get_local_torch_device()
        dtype = torch.bfloat16

        siglip_embeds = []

        # TODO: review?
        # self.siglip_processor = self.siglip_processor.to(device)
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
            siglip_embeds.append(hidden_state)

        # siglip_embeds = [siglip_embeds] * batch_size
        # TODO: 2D list.
        # where
        #  len(l) == batch size
        #  len(l[0]) == image nums
        # TODO: review why?
        condition_siglip_embeds = [siglip_embeds.copy() for _ in range(batch_size)]

        # TODO: review
        ## ====================
        # dtype cast?
        condition_siglip_embeds = [
            [se for se in sels] for sels in condition_siglip_embeds
        ]

        # TODO: placeholder and end with None
        # pad None at the end
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
            batch.condition_image
        Returns
            pos_latents (tensor)
            neg_latents (tensor)
        """
        # TODO: hard code debug
        # do nothing and skip
        if condition_images is None:
            return None, None

        image_latents = []

        # TODO: hard code
        batch_size = 1
        dtype = dtype or torch.bfloat16
        device = get_local_torch_device()

        self.vae = self.vae.to(device)

        for image in condition_images:
            image = image.to(device=device, dtype=dtype)
            image_latent = (
                # TODO: hard code to vae(fp32) dtype. reivew
                self.vae.encode(image.to(torch.float32)).latent_dist.mode()[0]
                - self.vae.config.shift_factor
            ) * self.vae.config.scaling_factor
            image_latent = image_latent.unsqueeze(1).to(dtype)
            image_latents.append(image_latent)  # (16, 128, 128)

        # image_latents = [image_latents] * batch_size
        # TODO: review no copy?
        # num_images_per_prompt * batch_size
        # a 2d list
        # dim0 = batch_size
        # dim1 = num_images
        # condition_latents = [image_latents.copy() for _ in range(batch_size)]
        condition_latents = [image_latents for _ in range(batch_size)]
        # TODO: review
        # casting
        condition_latents = [
            [lat.to(dtype) for lat in lats] for lats in condition_latents
        ]

        # negative_condition_latents = [
        #     [None for lat in lats] for lats in condition_latents
        # ]
        # TODO: review, flag by None
        negative_condition_latents = None
        if do_classifier_free_guidance:
            # negative_condition_latents = [
            #     [lat.clone() for lat in batch] for batch in condition_latents
            # ]
            # TODO: review, readonly assert?
            negative_condition_latents = condition_latents

        # # TODO: comment usage
        # # for condition_latents_model_input = condition_latents + negative_condition_latents
        # # in denoising loop
        # batch.condition_latents = condition_latents
        # batch.negative_condition_latents = negative_condition_latents
        return condition_latents, negative_condition_latents

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        # Copied from diffusers.pipeline_z_image_omni
        do_classifier_free_guidance = batch.guidance_scale > 1
        batch.do_classifier_free_guidance = do_classifier_free_guidance

        # # TODO:
        # (prompt_embeds, negative_prompt_embeds,) = self.encode_prompt()

        # Encode positive prompt with all available encoders
        assert batch.prompt is not None
        num_condition_images: int = (
            len(batch.image_path) if batch.image_path is not None else 0
        )

        # TODO: review
        pos_prompt_embeds_list = self.encode_text(
            batch.prompt,
            server_args,
            num_condition_images=num_condition_images,
        )
        assert isinstance(batch.prompt_embeds, list)
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
            self._prepare_siglip(resized_images, do_classifier_free_guidance)
        )

        # prepare image latent
        condition_latents, negative_condition_latents = self._prepare_image_latents(
            condition_image,
            do_classifier_free_guidance,
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
