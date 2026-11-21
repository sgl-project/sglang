import json
from typing import List, Optional, Union

import torch
from transformers import AutoImageProcessor, AutoTokenizer, BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.utils import cached_file

IMAGE_PLACEHOLDER = "<|image|>"
IMAGE_PATCH_TOKEN = "<im_patch>"
IMAGE_COL_TOKEN = "<im_col>"
IMAGE_START_TOKEN = "<im_start>"
IMAGE_END_TOKEN = "<im_end>"
LOW_RES_IMAGE_START_TOKEN = "<low_res_im_start>"

_LAYOUT_KEYS = (
    "image_use_col_tokens",
    "use_single_crop_col_tokens",
    "use_single_crop_start_token",
)


class Molmo2Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template: Optional[str] = None,
        image_use_col_tokens: bool = True,
        use_single_crop_col_tokens: Optional[bool] = None,
        use_single_crop_start_token: bool = True,
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.image_use_col_tokens = image_use_col_tokens
        self.use_single_crop_col_tokens = (
            image_use_col_tokens
            if use_single_crop_col_tokens is None
            else use_single_crop_col_tokens
        )
        self.use_single_crop_start_token = use_single_crop_start_token
        self.bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *args,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        **kwargs,
    ):
        kwargs.pop("_from_auto", None)
        image_processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        layout = _load_layout(pretrained_model_name_or_path, revision=revision)
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=tokenizer.chat_template,
            **layout,
        )

    def __call__(
        self,
        text: Union[str, List[str]],
        images=None,
        images_kwargs: Optional[dict] = None,
        return_tensors: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> BatchFeature:
        texts = [text] if isinstance(text, str) else list(text)
        if len(texts) != 1:
            raise ValueError("Molmo2Processor expects exactly one prompt")
        prompt = texts[0]

        image_inputs = {}
        if images:
            image_kwargs = dict(images_kwargs or {})
            if device is not None:
                image_kwargs["device"] = device
            image_inputs = _process_images(
                image_processor=self.image_processor,
                images=images,
                image_kwargs=image_kwargs,
            )
            prompt = self._expand_image_placeholders(
                prompt, image_inputs["image_grids"]
            )

        kwargs.pop("padding", None)
        input_ids = self.tokenizer(prompt, **kwargs)["input_ids"]
        if not input_ids or input_ids[0] != self.bos_token_id:
            input_ids = [self.bos_token_id] + input_ids

        return BatchFeature(
            data={
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
                **image_inputs,
            },
            tensor_type=return_tensors,
        )

    def _expand_image_placeholders(self, prompt: str, image_grids: torch.Tensor) -> str:
        num_placeholders = prompt.count(IMAGE_PLACEHOLDER)
        if num_placeholders != len(image_grids):
            raise ValueError(
                f"Prompt has {num_placeholders} {IMAGE_PLACEHOLDER} placeholders "
                f"for {len(image_grids)} images"
            )
        pieces = prompt.split(IMAGE_PLACEHOLDER)
        expanded = [pieces[0]]
        for image_grid, piece in zip(image_grids, pieces[1:]):
            expanded.append(
                _image_string(
                    image_grid.tolist(),
                    image_use_col_tokens=self.image_use_col_tokens,
                    use_single_crop_col_tokens=self.use_single_crop_col_tokens,
                    use_single_crop_start_token=self.use_single_crop_start_token,
                )
            )
            expanded.append(piece)
        return "".join(expanded)


def _image_string(
    image_grid: List[int],
    *,
    image_use_col_tokens: bool,
    use_single_crop_col_tokens: bool,
    use_single_crop_start_token: bool,
) -> str:
    low_res_h, low_res_w, high_res_h, high_res_w = image_grid
    low_res_row = IMAGE_PATCH_TOKEN * low_res_w
    if use_single_crop_col_tokens:
        low_res_row += IMAGE_COL_TOKEN
    high_res_row = IMAGE_PATCH_TOKEN * high_res_w
    if image_use_col_tokens:
        high_res_row += IMAGE_COL_TOKEN
    low_res_start = (
        LOW_RES_IMAGE_START_TOKEN if use_single_crop_start_token else IMAGE_START_TOKEN
    )
    return (
        low_res_start
        + low_res_row * low_res_h
        + IMAGE_END_TOKEN
        + IMAGE_START_TOKEN
        + high_res_row * high_res_h
        + IMAGE_END_TOKEN
    )


def _process_images(*, image_processor, images, image_kwargs: dict) -> dict:
    per_image = [
        image_processor([image], return_tensors="pt", **image_kwargs)
        for image in images
    ]
    poolings = []
    patch_offset = 0
    for out in per_image:
        pooling = out["image_token_pooling"]
        poolings.append(torch.where(pooling >= 0, pooling + patch_offset, pooling))
        patch_offset += out["pixel_values"].shape[0] * out["pixel_values"].shape[1]
    return {
        "pixel_values": torch.cat([out["pixel_values"] for out in per_image]),
        "image_token_pooling": torch.cat(poolings),
        "image_grids": torch.cat([out["image_grids"] for out in per_image]),
        "image_num_crops": torch.cat([out["image_num_crops"] for out in per_image]),
    }


def _load_layout(pretrained_model_name_or_path: str, revision: Optional[str]) -> dict:
    path = cached_file(
        pretrained_model_name_or_path,
        "processor_config.json",
        revision=revision,
        _raise_exceptions_for_missing_entries=False,
    )
    if path is None:
        return {}
    with open(path) as f:
        config = json.load(f)
    return {key: config[key] for key in _LAYOUT_KEYS if key in config}
