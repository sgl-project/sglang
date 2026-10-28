import asyncio
import io
import json
import logging
import math
import re
from itertools import product
from typing import List, Optional, Union
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from transformers import BatchFeature, ProcessorMixin, TensorType

from sglang.srt.configs.step5 import Step5VLConfig
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.step3_vl import Step3VLForConditionalGeneration
from sglang.srt.models.step3_vl_10b import StepVLForConditionalGeneration
from sglang.srt.models.step3p7 import Step3p7ForConditionalGeneration
from sglang.srt.models.step5_vl import Step5ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.runtime_context import get_mm
from sglang.srt.utils.common import CLIENT_MEDIA_EXCEPTIONS

logger = logging.getLogger(__name__)

LOW_DETAIL_MAX_SIDE = 728
LOW_DETAIL_MAX_RATIO = 1.5

Step3Image = Union[Image.Image, torch.Tensor]
ImageWithPatches = tuple[Step3Image, list[Step3Image], list[int] | None]


class CPUToTensor(torch.nn.Module):
    def forward(
        self, raw_image: Union[np.ndarray, Image.Image, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(raw_image, torch.Tensor):
            image_tensor = raw_image.cpu()
            if image_tensor.ndim != 3:
                raise TypeError(
                    f"Expected CHW image tensor, got shape {tuple(image_tensor.shape)}"
                )
            if image_tensor.shape[0] == 1:
                image_tensor = image_tensor.repeat(3, 1, 1)
            elif image_tensor.shape[0] != 3:
                raise TypeError(
                    f"Expected CHW image tensor with 1 or 3 channels, got shape {tuple(image_tensor.shape)}"
                )
            if image_tensor.dtype == torch.uint8:
                image_tensor = image_tensor.to(torch.float32).div(255)
            elif not image_tensor.is_floating_point():
                image_tensor = image_tensor.to(torch.float32)
            return image_tensor.contiguous()
        if isinstance(raw_image, Image.Image):
            return transforms.ToTensor()(raw_image)
        if raw_image.ndim == 2:
            raw_image = raw_image[:, :, None].repeat(3, -1)
        image_tensor = torch.from_numpy(raw_image)
        image_tensor = torch.permute(image_tensor, (2, 0, 1)).contiguous()
        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.to(torch.float32).div(255)
        return image_tensor


class Step3VisionProcessor:
    def __init__(self, size, interpolation_mode="bicubic", patch_size=None):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        patch_size = patch_size if patch_size is not None else size

        self.transform = transforms.Compose(
            [
                CPUToTensor(),
                transforms.Normalize(mean, std),
                transforms.Resize(
                    (size, size),
                    interpolation=(
                        InterpolationMode.BICUBIC
                        if interpolation_mode == "bicubic"
                        else InterpolationMode.BILINEAR
                    ),
                    antialias=True,
                ),
            ]
        )

        self.patch_transform = (
            transforms.Compose(
                [
                    CPUToTensor(),
                    transforms.Normalize(mean, std),
                    transforms.Resize(
                        (patch_size, patch_size),
                        interpolation=(
                            InterpolationMode.BICUBIC
                            if interpolation_mode == "bicubic"
                            else InterpolationMode.BILINEAR
                        ),
                        antialias=True,
                    ),
                ]
            )
            if patch_size is not None
            else None
        )

    def __call__(self, image, is_patch=False):
        if is_patch:
            return {"pixel_values": self.patch_transform(image).unsqueeze(0)}
        else:
            return {"pixel_values": self.transform(image).unsqueeze(0)}


class ImagePatcher:
    def get_image_size(self, img: Step3Image) -> tuple[int, int]:
        if isinstance(img, Image.Image):
            return img.size
        if isinstance(img, torch.Tensor):
            if img.ndim != 3:
                raise TypeError(
                    f"Expected CHW image tensor, got shape {tuple(img.shape)}"
                )
            return int(img.shape[-1]), int(img.shape[-2])
        raise TypeError(f"Unsupported image type: {type(img)}")

    def determine_window_size(self, long: int, short: int) -> int:
        if long <= 728:
            return short if long / short > 1.5 else 0
        return min(short, 504) if long / short > 4 else 504

    def slide_window(
        self,
        width: int,
        height: int,
        sizes: list[tuple[int, int]],
        steps: list[tuple[int, int]],
        img_rate_thr: float = 0.6,
    ) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
        assert 1 >= img_rate_thr >= 0, "The `img_rate_thr` should lie in 0~1"
        windows = []
        # Sliding windows.
        for size, step in zip(sizes, steps):
            size_w, size_h = size
            step_w, step_h = step

            x_num = 1 if width <= size_w else math.ceil((width - size_w) / step_w + 1)
            x_start = [step_w * i for i in range(x_num)]
            if len(x_start) > 1 and x_start[-1] + size_w > width:
                x_start[-1] = width - size_w

            y_num = 1 if height <= size_h else math.ceil((height - size_h) / step_h + 1)
            y_start = [step_h * i for i in range(y_num)]
            if len(y_start) > 1 and y_start[-1] + size_h > height:
                y_start[-1] = height - size_h

            start = np.array(list(product(y_start, x_start)), dtype=int)
            start[:, [0, 1]] = start[:, [1, 0]]
            windows.append(np.concatenate([start, start + size], axis=1))
        windows = np.concatenate(windows, axis=0)

        return [
            (int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1]))
            for box in windows
        ], (x_num, y_num)

    def square_pad(self, img: Step3Image) -> Step3Image:
        w, h = self.get_image_size(img)
        if w == h:
            return img
        size = max(w, h)
        if isinstance(img, Image.Image):
            padded = Image.new(img.mode, (size, size), 0)
            padded.paste(img, (0, 0))
            return padded
        return torch.nn.functional.pad(img, (0, size - w, 0, size - h), value=0)

    def get_image_size_for_padding(
        self, img_width: int, img_height: int
    ) -> tuple[int, int]:
        ratio = img_width / img_height
        if min(img_height, img_width) < 32 and (ratio > 4 or ratio < 1 / 4):
            new_size = max(img_height, img_width)
            return new_size, new_size
        return img_width, img_height

    def get_image_size_for_preprocess(
        self, img_width: int, img_height: int
    ) -> tuple[int, int]:

        if max(img_height, img_width) > 3024:
            scale_factor = 3024 / max(img_height, img_width)
            img_width = int(img_width * scale_factor)
            img_height = int(img_height * scale_factor)
            return img_width, img_height
        else:
            return img_width, img_height

    def get_image_size_for_crop(
        self, img_width: int, img_height: int, window_size: int
    ):
        w_ratio = img_width / window_size
        h_ratio = img_height / window_size

        if w_ratio < 1:
            width_new = img_width
        else:
            decimal_w = w_ratio - img_width // window_size
            w_ratio = int(w_ratio) + 1 if decimal_w > 0.2 else int(w_ratio)
            width_new = window_size * w_ratio
        if h_ratio < 1:
            height_new = img_height
        else:
            decimal_h = h_ratio - img_height // window_size
            h_ratio = int(h_ratio) + 1 if decimal_h > 0.2 else int(h_ratio)
            height_new = window_size * h_ratio
        return int(width_new), int(height_new)

    def resize(self, img: Step3Image, size: tuple[int, int]) -> Step3Image:
        if isinstance(img, Image.Image):
            return img.resize(size, Image.Resampling.BILINEAR)
        return F.resize(
            img,
            [size[1], size[0]],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ).contiguous()

    def patch_crop(
        self, img: Step3Image, i: int, j: int, th: int, tw: int
    ) -> Step3Image:
        if isinstance(img, Image.Image):
            return img.crop((j, i, j + tw, i + th))
        return img[:, i : i + th, j : j + tw].contiguous()

    def get_num_patches(self, img_width: int, img_height: int) -> tuple[int, int]:
        img_width, img_height = self.get_image_size_for_padding(img_width, img_height)
        img_width, img_height = self.get_image_size_for_preprocess(
            img_width, img_height
        )
        window_size = self.determine_window_size(
            max(img_height, img_width), min(img_height, img_width)
        )
        if window_size == 0:
            return 0, 0
        else:
            img_width, img_height = self.get_image_size_for_crop(
                img_width, img_height, window_size
            )
            center_list, (x_num, y_num) = self.slide_window(
                img_width,
                img_height,
                [(window_size, window_size)],
                [(window_size, window_size)],
            )
            full_rows = (len(center_list) - 1) // x_num + 1
            if len(center_list) > 0 and len(center_list) % x_num == 0:
                full_rows -= 1
            return len(center_list), full_rows

    def __call__(
        self, img: Step3Image
    ) -> tuple[Step3Image, list[Step3Image], list[bool] | None]:
        img_width, img_height = self.get_image_size(img)
        new_img_width, new_img_height = self.get_image_size_for_padding(
            img_width, img_height
        )
        if new_img_width != img_width or new_img_height != img_height:
            img = self.square_pad(img)
            img_width, img_height = self.get_image_size(img)

        new_img_width, new_img_height = self.get_image_size_for_preprocess(
            img_width, img_height
        )
        img = self.resize(img, (new_img_width, new_img_height))
        window_size = self.determine_window_size(
            max(new_img_height, new_img_width), min(new_img_height, new_img_width)
        )
        if window_size == 0:
            return img, [], None
        else:
            new_img_width, new_img_height = self.get_image_size_for_crop(
                new_img_width, new_img_height, window_size
            )
            if (new_img_width, new_img_height) != (img_width, img_height):
                img_for_crop = self.resize(img, (new_img_width, new_img_height))
            else:
                img_for_crop = img

            patches = []
            newlines = []
            center_list, (x_num, y_num) = self.slide_window(
                new_img_width,
                new_img_height,
                [(window_size, window_size)],
                [(window_size, window_size)],
            )
            for patch_id, center_lf_point in enumerate(center_list):
                x, y, patch_w, patch_h = center_lf_point
                big_patch = self.patch_crop(img_for_crop, y, x, patch_h, patch_w)
                patches.append(big_patch)
                if (patch_id + 1) % x_num == 0:
                    newlines.append(patch_id)

            if newlines and newlines[-1] == len(patches) - 1:
                newlines.pop()

            return (
                img,
                patches,
                (
                    [i in newlines for i in range(len(patches))]
                    if len(patches) > 0
                    else None
                ),
            )


class Step3VLProcessor:
    def __init__(
        self,
        config,
        tokenizer,
    ) -> None:
        super().__init__()

        self.config = config
        if isinstance(tokenizer, ProcessorMixin):
            tokenizer = tokenizer.tokenizer
        self.tokenizer = tokenizer

        self.image_size = 728
        self.patch_size = 504
        self.image_preprocessor = Step3VisionProcessor(
            self.image_size, "bilinear", self.patch_size
        )

        self.num_image_feature_size = 169
        self.num_patch_feature_size = 81
        self.image_token = "<im_patch>"
        self.image_feature_placeholder = self.image_token * self.num_image_feature_size
        self.patch_feature_placeholder = self.image_token * self.num_patch_feature_size

        self.patcher = ImagePatcher()

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[self.image_token]

    def get_num_image_tokens(self, img_width: int, img_height: int) -> int:
        num_patches, num_newlines = self.patcher.get_num_patches(img_width, img_height)

        return (
            num_patches * (self.num_patch_feature_size + 2)
            + self.num_image_feature_size
            + 2
            + num_newlines
        )

    def _split_images(self, images: list[Image.Image]) -> list[ImageWithPatches]:
        result = []
        for img in images:
            result.append(self.patcher(img))
        return result

    def _convert_images_to_pixel_values(
        self,
        images: list[Step3Image],
        is_patch: bool = False,
    ) -> list[torch.Tensor]:
        return [
            self.image_preprocessor(img, is_patch=is_patch)["pixel_values"]
            for img in images
        ]

    def _get_patch_repl(
        self,
        num_patches: int,
        patch_newline_mask: list[bool] | None,
    ) -> tuple[str, list[int]]:
        text = ""
        token_ids = []
        for i in range(num_patches):
            assert len(patch_newline_mask) == num_patches
            text += f"<patch_start>{self.patch_feature_placeholder}<patch_end>"
            token_ids.extend(
                [self.tokenizer.convert_tokens_to_ids("<patch_start>")]
                + [self.image_token_id] * self.num_patch_feature_size
                + [self.tokenizer.convert_tokens_to_ids("<patch_end>")]
            )
            if patch_newline_mask and patch_newline_mask[i]:
                text += "<patch_newline>"
                token_ids.append(
                    self.tokenizer.convert_tokens_to_ids("<patch_newline>")
                )
        return text, token_ids

    def _get_image_repl(
        self,
        num_images: int,
    ) -> tuple[str, list[int]]:
        text = f"<im_start>{self.image_feature_placeholder}<im_end>"
        token_ids = (
            [self.tokenizer.convert_tokens_to_ids("<im_start>")]
            + [self.image_token_id] * self.num_image_feature_size
            + [self.tokenizer.convert_tokens_to_ids("<im_end>")]
        )
        return text * num_images, token_ids * num_images

    def _get_image_repl_features(
        self,
        num_images: int,
        num_patches: int,
        patch_new_line_idx: Optional[list[bool]],
    ) -> tuple[str, list[int]]:
        if num_patches > 0:
            patch_repl, patch_repl_ids = self._get_patch_repl(
                num_patches, patch_new_line_idx
            )
        else:
            patch_repl = ""
            patch_repl_ids = []
        image_repl, image_repl_ids = self._get_image_repl(num_images)
        return patch_repl + image_repl, patch_repl_ids + image_repl_ids

    def replace_placeholder(self, text: str, placeholder: str, repls: list[str]) -> str:
        parts = text.split(placeholder)

        if len(parts) - 1 != len(repls):
            raise ValueError(
                "The number of placeholders does not match the number of replacements."  # noqa: E501
            )

        result = [parts[0]]
        for i, repl in enumerate(repls):
            result.append(repl)
            result.append(parts[i + 1])

        return "".join(result)

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        *args,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> BatchFeature:
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]

        if len(images) == 0:
            image_inputs = {}
            text_inputs = self.tokenizer(text, add_special_tokens=add_special_tokens)
        else:
            splitted_images_data = self._split_images(images)
            pixel_values_lst = []
            patch_pixel_values_lst = []
            patch_newline_mask_lst = []
            image_repl_str_lst = []
            image_repl_ids_lst = []
            num_patches = []
            for (
                raw_img,
                img_patches,
                patch_newline_mask,
            ) in splitted_images_data:  # noqa: E501
                pixel_values_lst.extend(self._convert_images_to_pixel_values([raw_img]))

                if len(img_patches) > 0:
                    patch_pixel_values_lst.extend(
                        self._convert_images_to_pixel_values(img_patches, is_patch=True)
                    )
                num_patches.append(len(img_patches))

                image_repl_str, image_repl_ids = self._get_image_repl_features(
                    1, len(img_patches), patch_newline_mask
                )
                image_repl_str_lst.append(image_repl_str)
                image_repl_ids_lst.extend(image_repl_ids)

                if patch_newline_mask is not None:
                    patch_newline_mask_lst.extend(patch_newline_mask)

            image_inputs = {
                "pixel_values": torch.cat(pixel_values_lst),
                "num_patches": num_patches,
            }
            if patch_pixel_values_lst:
                image_inputs["patch_pixel_values"] = torch.cat(patch_pixel_values_lst)
            if patch_newline_mask_lst:
                image_inputs["patch_newline_mask"] = torch.tensor(
                    patch_newline_mask_lst, dtype=torch.bool
                )

            text = [
                self.replace_placeholder(t, self.image_token, image_repl_str_lst)
                for t in text
            ]
            text_inputs = self.tokenizer(text, add_special_tokens=add_special_tokens)

        return BatchFeature(
            {
                **text_inputs,
                **image_inputs,
            },
            tensor_type=return_tensors,
        )


def step3_vl_rust_spec(processor: Step3VLProcessor) -> Optional[dict]:
    preprocessor = processor.image_preprocessor
    resizes = (
        preprocessor.transform.transforms[2],
        preprocessor.patch_transform.transforms[2],
    )
    if any(
        t.interpolation != InterpolationMode.BILINEAR or not t.antialias
        for t in resizes
    ):
        return None
    tokenizer = processor.tokenizer
    names = {
        "image_start_id": "<im_start>",
        "image_end_id": "<im_end>",
        "patch_start_id": "<patch_start>",
        "patch_end_id": "<patch_end>",
        "patch_newline_id": "<patch_newline>",
    }
    ids = {key: tokenizer.convert_tokens_to_ids(tok) for key, tok in names.items()}
    unk = getattr(tokenizer, "unk_token_id", None)
    if any(v is None or (unk is not None and v == unk) for v in ids.values()):
        return None
    normalize = preprocessor.transform.transforms[1]
    return dict(
        image_token_id=processor.image_token_id,
        image_size=processor.image_size,
        patch_size=processor.patch_size,
        image_feature_tokens=processor.num_image_feature_size,
        patch_feature_tokens=processor.num_patch_feature_size,
        image_mean=[float(x) for x in normalize.mean],
        image_std=[float(x) for x in normalize.std],
        **ids,
    )


################################################


class Step3VLImageProcessor(SGLangBaseProcessor):
    models = [
        Step3VLForConditionalGeneration,
        StepVLForConditionalGeneration,
        Step3p7ForConditionalGeneration,
        Step5ForConditionalGeneration,
    ]

    # Preprocessing runs in tokenizer workers, which must not compete with TP0.
    mm_feature_transport_override = "cpu"
    gpu_image_decode = False
    video_preprocessing_device = "cpu"
    per_image_mm_items = False

    def _fast_image_processor_device(self, processor):
        return "cpu"

    def _preprocessing_competes_with_the_scheduler(self):
        return False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        # TODO, check _processor is tokenizer or processor.
        processor = Step3VLProcessor(hf_config, _processor)
        self._rust, self._rust_spec_json = self._load_rust(processor)
        super().__init__(hf_config, server_args, processor, *args, **kwargs)
        is_step5 = isinstance(hf_config, Step5VLConfig)
        self.prefer_tokenized_input = is_step5
        self.accepts_chat_prompt_ids = is_step5
        self.per_image_mm_items = is_step5
        self.IM_TOKEN = "<im_patch>"
        self.IM_TOKEN_ID = self._processor.tokenizer.get_vocab()[self.IM_TOKEN]
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IM_TOKEN,
            image_token_id=self.IM_TOKEN_ID,
            image_token_regex=re.compile(r"(?:<im_patch>)"),
        ).build(_processor)

        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

    @staticmethod
    def _load_rust(processor: Step3VLProcessor):
        mm = get_mm()
        if mm.image_processor_backend != "auto" or mm.disable_fast_image_processor:
            return None, None
        fields = step3_vl_rust_spec(processor)
        if fields is None:
            return None, None
        try:
            from sglang.srt.rust_extensions import load_rust_extension

            module = load_rust_extension(
                "sglang.srt.rust_extensions._multimodal"
            ).step3_vl
        except (ImportError, OSError, RuntimeError, AttributeError) as error:
            logger.warning(
                "Step3-VL Rust image processor unavailable; using Python: %s", error
            )
            return None, None
        return module, json.dumps({"family": "step3_vl", **fields})

    def preprocess_fingerprint_payload(self):
        payload = super().preprocess_fingerprint_payload()
        payload["step3_vl_preprocess_backend"] = (
            "rust" if self._rust is not None else "python"
        )
        return payload

    @staticmethod
    def _rust_image_source(item) -> Optional[Union[str, bytes, Image.Image]]:
        if isinstance(item, (str, bytes, Image.Image)):
            return item
        url = getattr(item, "url", None)
        if isinstance(url, str) and not getattr(item, "preprocess_kwargs", None):
            return url
        return None

    @staticmethod
    def _read_image_bytes(source: Union[str, bytes, Image.Image]) -> bytes:
        # detail=low has already been resized/padded using PIL. Lossless encoding
        # preserves those exact pixels for the Rust normalization/patch pipeline.
        if isinstance(source, Image.Image):
            buffer = io.BytesIO()
            source.save(buffer, format="PNG")
            return buffer.getvalue()
        from sglang.srt.utils.common import get_image_bytes, image_extension_names

        if isinstance(source, str):
            if source.startswith("file://"):
                source = unquote(urlparse(source).path)
            elif not source.startswith(
                ("http://", "https://", "data:", "/")
            ) and source.lower().endswith(image_extension_names):
                with open(source, "rb") as f:
                    return f.read()
        return get_image_bytes(source)

    async def _process_mm_data_rust(self, sources, input_text):
        loop = asyncio.get_running_loop()
        datas = await asyncio.gather(
            *(
                loop.run_in_executor(self.io_executor, self._read_image_bytes, s)
                for s in sources
            )
        )
        semantics = ["pil"] * len(datas)
        images = await asyncio.to_thread(
            self._rust.preprocess_batch, list(datas), self._rust_spec_json, semantics
        )
        if isinstance(input_text, list):
            input_ids = list(input_text)
        else:
            tokenizer = self._processor.tokenizer
            input_ids = tokenizer(
                input_text,
                add_special_tokens=not self._template_wrote_bos(tokenizer, input_text),
            )["input_ids"]
        newlines = [list(nl) for _, _, nl in images]
        input_ids, runs = self._rust.expand_input_ids(
            input_ids, newlines, self._rust_spec_json
        )
        size, patch = self._processor.image_size, self._processor.patch_size
        model_specific_data = {
            "num_patches": torch.tensor([len(nl) for nl in newlines], dtype=torch.long)
        }
        patches = [ppv for _, ppv, _ in images if ppv.size]
        if patches:
            model_specific_data["patch_pixel_values"] = torch.from_numpy(
                np.concatenate(patches)
            ).view(-1, 3, patch, patch)
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=torch.from_numpy(np.concatenate([pv for pv, _, _ in images])).view(
                -1, 3, size, size
            ),
            offsets=[tuple(r) for per_image in runs for r in per_image],
            model_specific_data=model_specific_data,
        )
        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=self._split_per_image(item),
            im_token_id=self.mm_tokens.image_token_id,
        )

    _PER_IMAGE_KEYS = frozenset(
        {"num_patches", "patch_pixel_values", "patch_newline_mask"}
    )

    def _split_per_image(self, item: MultimodalDataItem) -> List[MultimodalDataItem]:
        if not self.per_image_mm_items or not item.is_image():
            return [item]
        data = item.model_specific_data
        num_patches = data.get("num_patches")
        if (
            item.feature is None
            or item.offsets is None
            or num_patches is None
            or not set(data).issubset(self._PER_IMAGE_KEYS)
        ):
            return [item]
        if isinstance(num_patches, torch.Tensor):
            counts = [int(x) for x in num_patches.flatten().tolist()]
        else:
            counts = [int(x) for x in num_patches]
        if (
            len(counts) <= 1
            or item.feature.shape[0] != len(counts)
            or len(item.offsets) != sum(n + 1 for n in counts)
        ):
            return [item]
        patch_pixel_values = data.get("patch_pixel_values")
        patch_newline_mask = data.get("patch_newline_mask")
        if patch_pixel_values is not None and patch_pixel_values.shape[0] != sum(
            counts
        ):
            return [item]
        items, run, patch = [], 0, 0
        for i, n in enumerate(counts):
            image_data = {"num_patches": torch.tensor([n], dtype=torch.long)}
            if n and patch_pixel_values is not None:
                image_data["patch_pixel_values"] = patch_pixel_values[patch : patch + n]
            if n and patch_newline_mask is not None:
                image_data["patch_newline_mask"] = patch_newline_mask[patch : patch + n]
            items.append(
                MultimodalDataItem(
                    modality=item.modality,
                    feature=item.feature[i : i + 1],
                    offsets=list(item.offsets[run : run + n + 1]),
                    model_specific_data=image_data,
                )
            )
            run += n + 1
            patch += n
        return items

    def _postprocess_mm_items_before_transport(self, mm_items, *, images):
        return [split for item in mm_items for split in self._split_per_image(item)]

    def preprocess(self, image):
        return {"pixel_values": self.transform(image).unsqueeze(0)}

    def __call__(self, image):
        return self.preprocess(image)

    @staticmethod
    def _is_low_detail(item) -> bool:
        return getattr(item, "detail", None) == "low"

    @classmethod
    def _load_low_detail_image(cls, item):
        from sglang.srt.utils.common import load_image

        img, _ = load_image(item.url, gpu_image_decode=False)
        if img.mode != "RGB":
            img = img.convert("RGB")
        long_side = max(img.size)
        if long_side > LOW_DETAIL_MAX_SIDE:
            scale = LOW_DETAIL_MAX_SIDE / long_side
            img = img.resize(
                (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                Image.Resampling.BILINEAR,
            )
        w, h = img.size
        if max(w, h) > min(w, h) * LOW_DETAIL_MAX_RATIO:
            need = math.ceil(max(w, h) / LOW_DETAIL_MAX_RATIO)
            pad_w, pad_h = (w, max(h, need)) if w >= h else (max(w, need), h)
            padded = Image.new("RGB", (pad_w, pad_h), (0, 0, 0))
            padded.paste(img, (0, 0))
            img = padded
        return img

    async def _resolve_low_detail(self, image_data):
        if not image_data or not any(self._is_low_detail(i) for i in image_data):
            return image_data
        loop = asyncio.get_running_loop()
        resolved = []
        for item in image_data:
            if self._is_low_detail(item):
                resolved.append(
                    await loop.run_in_executor(
                        self.io_executor, self._load_low_detail_image, item
                    )
                )
            else:
                resolved.append(item)
        return resolved

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text: str | List[int],
        request_obj,
        *args,
        **kwargs,
    ):
        image_data = await self._resolve_low_detail(image_data)
        if isinstance(input_text, list) and not request_obj.video_data:
            placeholders = input_text.count(self.IM_TOKEN_ID)
            if placeholders != len(image_data or []):
                raise ValueError(
                    f"prompt has {placeholders} image placeholder(s) "
                    f"but {len(image_data or [])} image(s)"
                )
        if self._rust is not None and image_data and not request_obj.video_data:
            sources = [self._rust_image_source(item) for item in image_data]
            if all(s is not None for s in sources):
                try:
                    return await self._process_mm_data_rust(sources, input_text)
                except CLIENT_MEDIA_EXCEPTIONS:
                    pass
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            multimodal_tokens=self.mm_tokens,
        )

        mm_items, input_ids, ret = await self.process_and_combine_mm_data_async(
            base_output, self.mm_tokens
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
        )
