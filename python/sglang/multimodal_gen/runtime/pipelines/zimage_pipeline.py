# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.vision_utils import load_image, load_video
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline, Req
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    ConditioningStage,
    DecodingStage,
    DenoisingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def prepare_mu(batch: Req, server_args: ServerArgs):
    height = batch.height
    width = batch.width
    vae_scale_factor = server_args.pipeline_config.vae_config.vae_scale_factor
    image_seq_len = ((int(height) // vae_scale_factor) // 2) * (
        (int(width) // vae_scale_factor) // 2
    )
    mu = calculate_shift(
        image_seq_len,
        # hard code, since scheduler_config is not in PipelineConfig now
        256,
        4096,
        0.5,
        1.15,
    )
    return "mu", mu


class ZImagePipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "ZImagePipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(
            stage_name="input_validation_stage", stage=InputValidationStage()
        )

        self.add_stage(
            stage_name="prompt_encoding_stage_primary",
            stage=TextEncodingStage(
                text_encoders=[
                    self.get_module("text_encoder"),
                ],
                tokenizers=[
                    self.get_module("tokenizer"),
                ],
            ),
        )

        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler"),
                prepare_extra_set_timesteps_kwargs=[prepare_mu],
            ),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae"))
        )


class _ImageProcessStage(PipelineStage):
    def __init__(
        self,
        image_processor,
        vae_scale_factor,
    ) -> None:
        self.image_processor = image_processor
        self.vae_scale_factor = vae_scale_factor

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Args:
            batch.height
            batch.width
            batch.image_path
        Returns:
            batch.condition_image
            batch.original_condition_image_size
        """
        # 3. Process condition images. Copied from diffusers.pipelines.flux2.pipeline_flux2
        condition_image = []
        resized_images = []
        height = batch.height
        width = batch.width

        if batch.image_path is not None:
            for path in batch.image_path:
                if path.endswith(".mp4"):
                    raise NotImplementedError("unimplemented")
                    img = load_video(path)[0]
                else:
                    img = load_image(path)

                # check img input
                self.image_processor.check_image_input(img)

                image_width, image_height = img.size
                # resize to (1024, 1024) if (height, width) is not define
                if image_width * image_height > 1024 * 1024:
                    if height is not None and width is not None:
                        img = self.image_processor._resize_to_target_area(
                            img, height * width
                        )
                    else:
                        img = self.image_processor._resize_to_target_area(
                            img, 1024 * 1024
                        )
                    image_width, image_height = img.size
                resized_images.append(img)

                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(
                    img, height=image_height, width=image_width, resize_mode="crop"
                )
                condition_image.append(img)

            if len(condition_image) > 0:
                height = height or image_height
                width = width or image_width

        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0:
            raise ValueError(
                f"Height must be divisible by {vae_scale} (got {height}). "
                f"Please adjust the height to a multiple of {vae_scale}."
            )
        if width % vae_scale != 0:
            raise ValueError(
                f"Width must be divisible by {vae_scale} (got {width}). "
                f"Please adjust the width to a multiple of {vae_scale}."
            )

        # TODO:
        # hard code with image size
        # dulplicate with input_validation_stage
        batch.height = image_height
        batch.width = image_width

        # TODO: hard code debug
        # should be condition_images(a list)
        # condition_images -> prepare_image_latents
        batch.condition_image = condition_image
        # prepare for
        # resized_images -> prepare_siglip_embeds
        batch.resized_images = resized_images
        return batch


class _PrepareImageLatentsStage(PipelineStage):
    def __init__(self, vae) -> None:
        self.vae = vae

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Args:
            batch.condition_image
        Returns
            TODO
        """
        image_latents = []
        images = batch.condition_image
        num_images_per_prompt = (
            len(batch.image_path) if batch.image_path is not None else 0
        )
        assert batch.batch_size == 1, f"under test: {batch.batch_size=}"

        # TODO: skip
        # review
        if num_images_per_prompt == 0:
            return batch

        batch_size = batch.batch_size * num_images_per_prompt
        # TODO: hardcode debug
        # a graceful way to do that?
        dtype = (
            batch.prompt_embeds[0].dtype
            if not isinstance(batch.prompt_embeds[0], list)
            else batch.prompt_embeds[0][0].dtype
        )
        device = get_local_torch_device()

        # TODO: hard code debug
        # do nothing and skip
        if images is None:
            return batch

        self.vae = self.vae.to(device)

        for image in images:
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
        if batch.do_classifier_free_guidance:
            # negative_condition_latents = [
            #     [lat.clone() for lat in batch] for batch in condition_latents
            # ]
            # TODO: review, readonly assert?
            negative_condition_latents = condition_latents

        # TODO: comment usage
        # for condition_latents_model_input = condition_latents + negative_condition_latents
        # in denoising loop
        batch.condition_latents = condition_latents
        batch.negative_condition_latents = negative_condition_latents

        return batch


class _PrepareSiglipStage(PipelineStage):
    def __init__(self, transformer, siglip, siglip_processor) -> None:
        self.transformer = transformer
        self.siglip = siglip
        self.siglip_processor = siglip_processor

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Args:
            batch.resized_images
        Returns:
            TODO
        """
        siglip_embeds = []
        images = batch.resized_images
        # TODO: hard code
        if images is None:
            return batch

        # TODO: hard code
        device = get_local_torch_device()
        # TODO: review?
        # self.siglip_processor = self.siglip_processor.to(device)
        self.siglip = self.siglip.to(device)

        num_images_per_prompt = (
            len(batch.image_path) if batch.image_path is not None else 0
        )
        batch_size = batch.batch_size * num_images_per_prompt
        # TODO: hard code
        dtype = torch.bfloat16
        do_classifier_free_guidance = batch.do_classifier_free_guidance

        for image in images:
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
        condition_siglip_embeds = [siglip_embeds.copy() for _ in range(batch_size)]

        # TODO: review
        ## ====================
        # dtype cast
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
            # negative_condition_siglip_embeds = [
            #     [se.clone() for se in batch] for batch in condition_siglip_embeds
            # ]
            # # TODO: debug remove
            # negative_condition_siglip_embeds = [
            #     None if sels == [] else sels + [None]
            #     for sels in negative_condition_siglip_embeds
            # ]

            # TODO: review, readonly modify assert
            negative_condition_siglip_embeds = condition_siglip_embeds

        # TODO: for siglip_feats = pos + neg
        # in denoising loop
        batch.condition_siglip_embeds = condition_siglip_embeds
        # TODO: debug remove
        batch.negative_condition_siglip_embeds = negative_condition_siglip_embeds
        return batch


class ZImageOmniPipeline(ZImagePipeline):
    pipeline_name = "ZOmniImagePipeline"

    # TODO: review how to add extra component?
    _extra_config_module_map = {
        "siglip": "image_encoder",
        "siglip_processor": "processor",
    }
    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
        "siglip",
        "siglip_processor",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Set up pipeline stages with proper dependency injection."""

        # copy from diffusers
        from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

        vae_scale_factor = server_args.pipeline_config.vae_config.vae_scale_factor
        # NOTE: replace vae with Flux in zimage-omni
        self.image_processor = Flux2ImageProcessor(
            vae_scale_factor=vae_scale_factor * 2
        )

        self.add_stage(
            stage_name="input_validation_stage", stage=InputValidationStage()
        )

        self.add_stage(
            stage_name="prompt_encoding_stage_primary",
            stage=TextEncodingStage(
                text_encoders=[
                    self.get_module("text_encoder"),
                ],
                tokenizers=[
                    self.get_module("tokenizer"),
                ],
            ),
        )

        # TODO: dulplicate with InputValidationStage:229
        # refactory later
        self.add_stage(
            stage_name="image_process",
            stage=_ImageProcessStage(
                image_processor=self.image_processor,
                vae_scale_factor=vae_scale_factor,
            ),
        )

        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler"),
                prepare_extra_set_timesteps_kwargs=[prepare_mu],
            ),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )

        self.add_stage(
            stage_name="image_siglip_preparation_stage",
            stage=_PrepareSiglipStage(
                transformer=self.get_module("transformer"),
                siglip=self.get_module("siglip"),
                siglip_processor=self.get_module("siglip_processor"),
            ),
        )

        self.add_stage(
            stage_name="image_latent_preparation_stage",
            stage=_PrepareImageLatentsStage(
                vae=self.get_module("vae"),
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae"))
        )


EntryClass = [
    ZImagePipeline,
    ZImageOmniPipeline,
]
