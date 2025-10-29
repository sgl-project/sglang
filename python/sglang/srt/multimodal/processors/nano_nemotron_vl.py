import numpy.typing as npt
import torch
from PIL import Image

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.nano_nemotron_vl import NemotronH_Nano_VL_V2
from sglang.srt.multimodal.internvl_utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    image_to_pixel_values,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

Image.MAX_IMAGE_PIXELS = None
DEFAULT_NUM_TILES = 12


def video_to_pixel_values(
    video: npt.NDArray,
    *,
    input_size: int,
    max_num_tiles: int = 1,
    use_thumbnail: bool,
) -> torch.Tensor:
    assert max_num_tiles == 1, "Video modality always uses one tile"

    # Convert each frame to a single resized tile tensor consistent
    # with image path
    frames_tensors: list[torch.Tensor] = []
    for frame in video:
        pil_frame = image_to_pixel_values(
            Image.fromarray(frame, mode="RGB"),
            input_size=input_size,
            max_num_tiles=max_num_tiles,
            use_thumbnail=use_thumbnail,
            normalize=False,
        )
        # dynamic_preprocess returns tensors already; take the single tile
        assert len(pil_frame) >= 1
        frames_tensors.append(pil_frame[-1])

    return torch.stack(frames_tensors)


def input_conditioner(x, norm_mean, norm_std):
    return (x - norm_mean) / norm_std


class NanoNemotronVLImageProcessor(BaseMultimodalProcessor):
    models = [NemotronH_Nano_VL_V2]

    def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _image_processor, *args, **kwargs)
        image_size = (
            getattr(hf_config, "force_image_size", None)
            or hf_config.vision_config.image_size
        )
        patch_size = hf_config.vision_config.patch_size
        if isinstance(image_size, list):
            image_size = image_size[0]
        if isinstance(patch_size, list):
            patch_size = patch_size[0]

        self.image_size = image_size

        self.IMG_CONTEXT_TOKEN = "<image>"
        self.IMG_START_TOKEN = "<img>"
        self.IMG_END_TOKEN = "</img>"
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (hf_config.downsample_ratio**2)
        )
        if hasattr(self._processor, "tokenizer"):
            tokenizer = self._processor.tokenizer
        else:
            tokenizer = self._processor
        self.tokenizer = tokenizer

        self.img_start_token_id = tokenizer.convert_tokens_to_ids(self.IMG_START_TOKEN)
        self.img_end_token_id = tokenizer.convert_tokens_to_ids(self.IMG_END_TOKEN)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMG_CONTEXT_TOKEN,
            image_token_id=tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN),
        ).build(_image_processor)

        # Normalization config (mean/std) and tiling behavior
        self.norm_mean = torch.tensor(
            getattr(hf_config, "norm_mean", IMAGENET_MEAN)
        ).reshape(1, 3, 1, 1)
        self.norm_std = torch.tensor(
            getattr(hf_config, "norm_std", IMAGENET_STD)
        ).reshape(1, 3, 1, 1)
        self.use_thumbnail = getattr(hf_config, "use_thumbnail", True)
        self.max_num_tiles = DEFAULT_NUM_TILES

    async def process_mm_data_async(
        self, image_data, input_text, request_obj, **kwargs
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
            discard_alpha_channel=True,
        )

        num_patches_list = []
        pixel_values_list = []

        # Process each input with allocated frames
        for image_index, image in enumerate(base_output.images):
            try:
                # TODO: video input
                # Convert PIL to GPU tensor
                tiles = image_to_pixel_values(
                    image,
                    input_size=self.image_size,
                    max_num_tiles=self.max_num_tiles,
                    use_thumbnail=self.use_thumbnail,
                    normalize=False,
                )
                tiles = input_conditioner(tiles, self.norm_mean, self.norm_std)
                tiles = tiles.to(dtype=torch.bfloat16)
                pixel_values_list.append(tiles)
                num_patches_list.append(tiles.shape[0])

            except Exception as e:
                print(f"[Error] Failed to process image {image_index}: {e}")
                return None

        # Concatenate all
        pixel_values = torch.cat(pixel_values_list, dim=0)
        original_placeholder = "<<<__CONTEXT_PLACEHOLDER__>>>"
        input_text_updated = input_text.replace(
            self.IMG_CONTEXT_TOKEN, original_placeholder
        )
        for num_patches in num_patches_list:
            image_tokens = (
                self.IMG_START_TOKEN
                + self.IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + self.IMG_END_TOKEN
            )
            input_text_updated = input_text_updated.replace(
                original_placeholder, image_tokens, 1
            )

        input_text_updated = input_text_updated.replace(
            original_placeholder, self.IMG_CONTEXT_TOKEN
        )

        # Tokenize
        input_ids_tensor = self.tokenizer(input_text_updated, return_tensors="pt")[
            "input_ids"
        ].flatten()
        input_ids = input_ids_tensor.tolist()

        # Get image token offsets
        image_offsets = self.get_mm_items_offset(
            input_ids=input_ids_tensor.to("cuda"),
            mm_token_id=self.mm_tokens.image_token_id,
        )

        items = [
            MultimodalDataItem(
                feature=pixel_values,
                modality=Modality.IMAGE,
                offsets=image_offsets,
            )
        ]

        return {
            "input_ids": input_ids,
            "mm_items": items,
            "im_start_id": self.img_start_token_id,
            "im_end_id": self.img_end_token_id,
            "im_token_id": self.mm_tokens.image_token_id,
        }
