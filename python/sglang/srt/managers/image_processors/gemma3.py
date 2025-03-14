import asyncio
import itertools
import math
import re
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import PIL
import transformers
from transformers import AutoImageProcessor, AutoProcessor, BatchFeature
from transformers.image_processing_utils import BaseImageProcessor, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_pil_image,
    is_scaled_image,
    is_valid_image,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from transformers.processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import (
    TensorType,
    filter_out_non_signature_kwargs,
    logging,
    to_py_obj,
)

from sglang.srt.configs import Gemma3Config
from sglang.srt.managers.image_processor import (
    BaseImageProcessor as SGLangBaseImageProcessor,
)
from sglang.srt.managers.image_processors.base_image_processor import (
    get_global_processor,
)
from sglang.srt.models.gemma3_mm import Gemma3ForConditionalGeneration

# Copied from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/image_processing_gemma3_fast.py
# will be removed in the future

logger = logging.get_logger(__name__)


def rescale(
    image: np.ndarray,
    scale: float,
    data_format: Optional[Union[str, ChannelDimension]] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
    **kwargs,
) -> np.ndarray:
    """
    Rescale an image by a scale factor. image = image * scale.

    Args:
        image (`np.ndarray`):
            Image to rescale.
        scale (`float`):
            The scaling factor to rescale pixel values by.
        data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format for the output image. If unset, the channel dimension format of the input
            image is used. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input image. If unset, the channel dimension format is inferred
            from the input image. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

    Returns:
        `np.ndarray`: The rescaled image.
    """
    return transformers.image_transforms.rescale(
        image,
        scale=scale,
        data_format=data_format,
        input_data_format=input_data_format,
        **kwargs,
    )


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
    data_format: Optional[Union[str, ChannelDimension]] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
    **kwargs,
) -> np.ndarray:
    """
    Normalize an image. image = (image - image_mean) / image_std.

    Args:
        image (`np.ndarray`):
            Image to normalize.
        mean (`float` or `Iterable[float]`):
            Image mean to use for normalization.
        std (`float` or `Iterable[float]`):
            Image standard deviation to use for normalization.
        data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format for the output image. If unset, the channel dimension format of the input
            image is used. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input image. If unset, the channel dimension format is inferred
            from the input image. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

    Returns:
        `np.ndarray`: The normalized image.
    """
    return transformers.image_transforms.normalize(
        image,
        mean=mean,
        std=std,
        data_format=data_format,
        input_data_format=input_data_format,
        **kwargs,
    )


class Gemma3ImagesKwargs(ImagesKwargs):
    do_pan_and_scan: Optional[bool]
    pan_and_scan_min_crop_size: Optional[int]
    pan_and_scan_max_num_crops: Optional[int]
    pan_and_scan_min_ratio_to_activate: Optional[float]
    do_convert_rgb: Optional[bool]


def is_valid_list_of_images(images: List):
    return images and all(is_valid_image(image) for image in images)


# copied from transformer
def make_nested_list_of_images(
    images: Union[List[ImageInput], ImageInput],
) -> ImageInput:
    """
    Ensure that the output is a nested list of images.
    Args:
        images (`Union[List[ImageInput], ImageInput]`):
            The input image.
    Returns:
        list: A list of list of images or a list of 4d array of images.
    """
    # If it's a list of batches, it's already in the right format
    if (
        isinstance(images, (list, tuple))
        and all(isinstance(images_i, (list, tuple)) for images_i in images)
        and all(is_valid_list_of_images(images_i) for images_i in images)
    ):
        return images

    # If it's a list of images, it's a single batch, so convert it to a list of lists
    if isinstance(images, (list, tuple)) and is_valid_list_of_images(images):
        if is_pil_image(images[0]) or images[0].ndim == 3:
            return [images]
        if images[0].ndim == 4:
            return [list(image) for image in images]

    # If it's a single image, convert it to a list of lists
    if is_valid_image(images):
        if is_pil_image(images) or images.ndim == 3:
            return [[images]]
        if images.ndim == 4:
            return [list(images)]

    raise ValueError(
        "Invalid input type. Must be a single image, a list of images, or a list of batches of images."
    )


class Gemma3ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Gemma3ImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            "do_pan_and_scan": False,
            "pan_and_scan_min_crop_size": 256,
            "pan_and_scan_max_num_crops": 4,
            "pan_and_scan_min_ratio_to_activate": 1.2,
        },
    }


class Gemma3Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "image_seq_length"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template=None,
        image_seq_length: int = 256,
        **kwargs,
    ):
        self.image_seq_length = image_seq_length
        self.image_token_id = tokenizer.image_token_id
        self.boi_token = tokenizer.boi_token
        image_tokens_expanded = "".join([tokenizer.image_token] * image_seq_length)
        self.full_image_sequence = (
            f"\n\n{tokenizer.boi_token}{image_tokens_expanded}{tokenizer.eoi_token}\n\n"
        )

        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        videos=None,
        audio=None,
        **kwargs: Unpack[Gemma3ProcessorKwargs],
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("Provide at least one of `text` or `images`.")

        output_kwargs = self._merge_kwargs(
            Gemma3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        image_inputs = {}
        if images is not None:
            batched_images = make_nested_list_of_images(images)
            image_inputs = self.image_processor(
                batched_images, **output_kwargs["images_kwargs"]
            )

            # Create empty text to be replaced with placeholders
            if not text:
                text = [
                    " ".join([self.boi_token] * len(images))
                    for images in batched_images
                ]

            if len(batched_images) != len(text):
                raise ValueError(
                    f"Received inconsistently sized batches of images ({len(batched_images)}) and text ({len(text)})."
                )

            # Replace image tokens by the full expanded sequence
            batch_num_crops = to_py_obj(image_inputs.pop("num_crops"))
            text_with_crops = text

            for batch_idx, (prompt, images, num_crops) in enumerate(
                zip(text, batched_images, batch_num_crops)
            ):

                image_indexes = [m.start() for m in re.finditer(self.boi_token, prompt)]

                if len(images) != len(image_indexes):
                    raise ValueError(
                        f"Prompt contained {len(image_indexes)} image tokens but received {len(images)} images."
                    )

                # Insert additional image tokens for Pan-and-Scan crops
                for num, idx in reversed(list(zip(num_crops, image_indexes))):
                    if num:
                        formatted_image_text = (
                            f"Here is the original image {self.boi_token} and here are some crops to help you see better "
                            + " ".join([self.boi_token] * num)
                        )
                        prompt = (
                            prompt[:idx]
                            + formatted_image_text
                            + prompt[idx + len(self.boi_token) :]
                        )
                        text_with_crops[batch_idx] = prompt

            # Expand placeholder image tokens to the full image token sequence
            text = [
                prompt.replace(self.boi_token, self.full_image_sequence)
                for prompt in text
            ]

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(
            text=text, **output_kwargs["text_kwargs"], return_tensors="np"
        )

        # Add token type ids manually, as tokenizer can't do arbitrary position token types
        array_ids = np.array(text_inputs["input_ids"])
        mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
        mm_token_type_ids[array_ids == self.image_token_id] = 1
        text_inputs = {
            k: v.tolist() for k, v in text_inputs.items()
        }  # in case user requested list inputs
        text_inputs["token_type_ids"] = mm_token_type_ids.tolist()
        return BatchFeature(
            data={**text_inputs, **image_inputs}, tensor_type=return_tensors
        )

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Gemma
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Gemma
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names + ["token_type_ids"]
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


class Gemma3ImageProcessor(BaseImageProcessor):
    r"""
    Constructs a SigLIP image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the image after resizing. Can be overridden by `size` in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image by the specified mean and standard deviation. Can be overridden by
            `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        do_pan_and_scan (`bool`, *optional*):
            Whether to apply `pan_and_scan` to images.
        pan_and_scan_min_crop_size (`int`, *optional*):
            Minimum size of each crop in pan and scan.
        pan_and_scan_max_num_crops (`int`, *optional*):
            Maximum number of crops per image in pan and scan.
        pan_and_scan_min_ratio_to_activate (`float`, *optional*):
            Minimum aspect ratio to activate pan and scan.
    """

    model_input_names = ["pixel_values", "num_crops"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        do_pan_and_scan: bool = None,
        pan_and_scan_min_crop_size: int = None,
        pan_and_scan_max_num_crops: int = None,
        pan_and_scan_min_ratio_to_activate: float = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 224, "width": 224}
        size = get_size_dict(size, default_to_square=True)
        image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb
        self.do_pan_and_scan = do_pan_and_scan
        self.pan_and_scan_min_crop_size = pan_and_scan_min_crop_size
        self.pan_and_scan_max_num_crops = pan_and_scan_max_num_crops
        self.pan_and_scan_min_ratio_to_activate = pan_and_scan_min_ratio_to_activate

    def pan_and_scan(
        self,
        image: np.ndarray,
        pan_and_scan_min_crop_size: int,
        pan_and_scan_max_num_crops: int,
        pan_and_scan_min_ratio_to_activate: float,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Pan and Scan and image, by cropping into smaller images when the aspect ratio exceeds
        minumum allowed ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            pan_and_scan_min_crop_size (`int`, *optional*):
                Minimum size of each crop in pan and scan.
            pan_and_scan_max_num_crops (`int`, *optional*):
                Maximum number of crops per image in pan and scan.
            pan_and_scan_min_ratio_to_activate (`float`, *optional*):
                Minimum aspect ratio to activate pan and scan.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        height, width = get_image_size(image)

        # Square or landscape image.
        if width >= height:
            # Only apply PaS if the image is sufficiently exaggerated
            if width / height < pan_and_scan_min_ratio_to_activate:
                return []

            # Select ideal number of crops close to the image aspect ratio and such that crop_size > min_crop_size.
            num_crops_w = int(
                math.floor(width / height + 0.5)
            )  # Half round up rounding.
            num_crops_w = min(
                int(math.floor(width / pan_and_scan_min_crop_size)), num_crops_w
            )

            # Make sure the number of crops is in range [2, pan_and_scan_max_num_crops].
            num_crops_w = max(2, num_crops_w)
            num_crops_w = min(pan_and_scan_max_num_crops, num_crops_w)
            num_crops_h = 1

        # Portrait image.
        else:
            # Only apply PaS if the image is sufficiently exaggerated
            if height / width < pan_and_scan_min_ratio_to_activate:
                return []

            # Select ideal number of crops close to the image aspect ratio and such that crop_size > min_crop_size.
            num_crops_h = int(math.floor(height / width + 0.5))
            num_crops_h = min(
                int(math.floor(height / pan_and_scan_min_crop_size)), num_crops_h
            )

            # Make sure the number of crops is in range [2, pan_and_scan_max_num_crops].
            num_crops_h = max(2, num_crops_h)
            num_crops_h = min(pan_and_scan_max_num_crops, num_crops_h)
            num_crops_w = 1

        crop_size_w = int(math.ceil(width / num_crops_w))
        crop_size_h = int(math.ceil(height / num_crops_h))

        # Don't apply PaS if crop size is too small.
        if min(crop_size_w, crop_size_h) < pan_and_scan_min_crop_size:
            return []

        crop_positions_w = [crop_size_w * i for i in range(num_crops_w)]
        crop_positions_h = [crop_size_h * i for i in range(num_crops_h)]

        if input_data_format == ChannelDimension.LAST:
            image_crops = [
                image[pos_h : pos_h + crop_size_h, pos_w : pos_w + crop_size_w]
                for pos_h, pos_w in itertools.product(
                    crop_positions_h, crop_positions_w
                )
            ]
        else:
            image_crops = [
                image[:, pos_h : pos_h + crop_size_h, pos_w : pos_w + crop_size_w]
                for pos_h, pos_w in itertools.product(
                    crop_positions_h, crop_positions_w
                )
            ]

        return image_crops

    def _process_images_for_pan_and_scan(
        self,
        images: List[np.ndarray],
        do_pan_and_scan: bool,
        pan_and_scan_min_crop_size: int,
        pan_and_scan_max_num_crops: int,
        pan_and_scan_min_ratio_to_activate: float,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        pas_images_list = []
        num_crops = []
        for image in images:
            pas_images = self.pan_and_scan(
                image=image,
                pan_and_scan_min_crop_size=pan_and_scan_min_crop_size,
                pan_and_scan_max_num_crops=pan_and_scan_max_num_crops,
                pan_and_scan_min_ratio_to_activate=pan_and_scan_min_ratio_to_activate,
                data_format=data_format,
                input_data_format=input_data_format,
            )
            pas_images_list.extend([image] + pas_images)
            num_crops.append(len(pas_images))
        return pas_images_list, num_crops

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        do_convert_rgb: bool = None,
        do_pan_and_scan: bool = None,
        pan_and_scan_min_crop_size: int = None,
        pan_and_scan_max_num_crops: int = None,
        pan_and_scan_min_ratio_to_activate: float = None,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            do_pan_and_scan (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to apply `pan_and_scan` to images.
            pan_and_scan_min_crop_size (`int`, *optional*, defaults to `self.pan_and_scan_min_crop_size`):
                Minimum size of each crop in pan and scan.
            pan_and_scan_max_num_crops (`int`, *optional*, defaults to `self.pan_and_scan_max_num_crops`):
                Maximum number of crops per image in pan and scan.
            pan_and_scan_min_ratio_to_activate (`float`, *optional*, defaults to `self.pan_and_scan_min_ratio_to_activate`):
                Minimum aspect ratio to activate pan and scan.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, param_name="size", default_to_square=False)
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = (
            rescale_factor if rescale_factor is not None else self.rescale_factor
        )
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = (
            do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        )
        do_pan_and_scan = (
            do_pan_and_scan if do_pan_and_scan is not None else self.do_pan_and_scan
        )
        pan_and_scan_min_crop_size = (
            pan_and_scan_min_crop_size
            if pan_and_scan_min_crop_size is not None
            else self.pan_and_scan_min_crop_size
        )
        pan_and_scan_max_num_crops = (
            pan_and_scan_max_num_crops
            if pan_and_scan_max_num_crops is not None
            else self.pan_and_scan_max_num_crops
        )
        pan_and_scan_min_ratio_to_activate = (
            pan_and_scan_min_ratio_to_activate
            if pan_and_scan_min_ratio_to_activate is not None
            else self.pan_and_scan_min_ratio_to_activate
        )

        images_list = make_nested_list_of_images(images)

        if not valid_images(images_list[0]):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )
        if do_convert_rgb:
            images_list = [
                [convert_to_rgb(image) for image in images] for images in images_list
            ]

        # All transformations expect numpy arrays.
        images_list = [
            [to_numpy_array(image) for image in images] for images in images_list
        ]

        if do_rescale and is_scaled_image(images_list[0][0]):
            logger.warning(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images_list[0][0])

        if do_pan_and_scan:
            images_list_and_num_crops = [
                self._process_images_for_pan_and_scan(
                    images=images,
                    do_pan_and_scan=do_pan_and_scan,
                    pan_and_scan_min_crop_size=pan_and_scan_min_crop_size,
                    pan_and_scan_max_num_crops=pan_and_scan_max_num_crops,
                    pan_and_scan_min_ratio_to_activate=pan_and_scan_min_ratio_to_activate,
                    data_format=data_format,
                    input_data_format=input_data_format,
                )
                for images in images_list
            ]
            images_list = [images for images, _ in images_list_and_num_crops]
            num_crops = [num_crops for _, num_crops in images_list_and_num_crops]
        else:
            num_crops = [[0] for images in images_list]

        processed_images = []
        for images in images_list:
            for image in images:
                if do_resize:
                    height, width = size["height"], size["width"]
                    image = resize(
                        image=image,
                        size=(height, width),
                        resample=resample,
                        input_data_format=input_data_format,
                    )

                if do_rescale:
                    image = rescale(
                        image=image,
                        scale=rescale_factor,
                        input_data_format=input_data_format,
                    )

                if do_normalize:
                    image = normalize(
                        image=image,
                        mean=image_mean,
                        std=image_std,
                        input_data_format=input_data_format,
                    )

                image = to_channel_dimension_format(
                    image, data_format, input_channel_dim=input_data_format
                )
                processed_images.append(image)

        data = {"pixel_values": processed_images, "num_crops": num_crops}
        return BatchFeature(data=data, tensor_type=return_tensors)


class Gemma3SGLangImageProcessor(SGLangBaseImageProcessor):
    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.IMAGE_TOKEN = "<start_of_image>"
        self.IM_START_TOKEN_ID = hf_config.boi_token_index
        self.IM_END_TOKEN_ID = hf_config.eoi_token_index

    @staticmethod
    def _process_images_task(images, input_text, _hf_config):
        if isinstance(images, list) and len(images) == 0:
            images = None
        processor = get_global_processor()
        result = processor.__call__(
            text=[input_text],
            images=images,
            padding=True,
            return_tensors="pt",
            # images_kwargs={
            #     "input_data_format": ChannelDimension.FIRST
            # }
        )

        pixel_values = getattr(result, "pixel_values", None)

        return {
            "input_ids": result.input_ids,
            "pixel_values": pixel_values,
        }

    async def _process_images(self, images, input_text) -> dict:
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                Gemma3SGLangImageProcessor._process_images_task,
                images,
                input_text,
                self.hf_config,
            )
        else:
            return self._process_images_task(images, input_text, self.hf_config)

    async def process_images_async(
        self,
        image_data: List[Union[str, bytes]],
        input_ids,
        request_obj,
        max_req_input_len,
        *args,
        **kwargs,
    ):
        if not image_data:
            return None
        if isinstance(image_data, str):
            image_data = [image_data]

        image_token = self.IMAGE_TOKEN
        base_output = self.load_images(
            input_ids=input_ids,
            image_data=image_data,
            image_token=image_token,
            max_req_input_len=max_req_input_len,
            discard_alpha_channel=True,
        )

        ret = await self._process_images(
            input_text=base_output.input_text, images=base_output.all_frames
        )
        return {
            "input_ids": ret["input_ids"].flatten().tolist(),
            "pixel_values": ret["pixel_values"],
            "image_hashes": base_output.image_hashes,
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
        }


ImageProcessorMapping = {
    Gemma3ForConditionalGeneration: Gemma3SGLangImageProcessor,
}

AutoImageProcessor.register(
    config_class=Gemma3Config,
    image_processor_class=None,
    slow_image_processor_class=Gemma3ImageProcessor,
    fast_image_processor_class=None,
    exist_ok=True,
)
AutoProcessor.register(
    config_class=Gemma3Config, processor_class=Gemma3Processor, exist_ok=True
)
