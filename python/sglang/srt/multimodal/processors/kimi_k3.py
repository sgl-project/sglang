"""Kimi K3 multimodal processor.

GPU image preprocessing dedicated to K3: unlike the K2.5 wrapper it keeps
the alpha channel through the bicubic resize and then composites RGBA
images onto the checkpoint-configured background
(``transparent_bg_config`` with ``transparent_bg_fill_stage ==
"after_resize"`` in preprocessor_config.json), instead of dropping alpha
at load time.
"""

import re
from typing import Dict, List, Union

import numpy as np
import torch
from PIL import Image

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration
from sglang.srt.multimodal.kimi_k3_image_processing import (
    DEFERRED_PREPROCESSING_KEY,
)
from sglang.srt.multimodal.kimi_k3_image_processing import (
    fill_transparent_bg as _fill_transparent_bg,
)
from sglang.srt.multimodal.kimi_k3_image_processing import (
    to_chw_uint8,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.kimi_common import KimiGridMMDataMixin
from sglang.srt.multimodal.processors.kimi_k25 import (
    KimiGPUProcessorWrapper,
    _get_image_dimensions,
    _gpu_preprocess_images,
    _grid_thw_from_resize_config,
    navit_resize_config,
)
from sglang.srt.utils import is_cuda
from sglang.srt.utils.cuda_ipc_transport_utils import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
)


def _encode_k3_special_tokens(tokenizer, text: str) -> list[int]:
    """Encode K3 control tokens without allowing them to be BPE-split."""
    try:
        return list(tokenizer.encode(text, allowed_special="all"))
    except TypeError:
        # Keep the helper usable with lightweight tokenizer stubs in CPU tests.
        return list(tokenizer.encode(text))


def _expand_k3_image_prompt_token_ids(
    input_ids: Union[List[int], torch.Tensor],
    image_token_id: int,
    image_token_counts: List[int],
    image_sizes: List[tuple[int, int]],
    tokenizer,
) -> torch.Tensor:
    """Expand K3 image placeholders into the checkpoint's media contract.

    K3 requires each image feature span to be enclosed by its original uploaded
    dimensions.  The chat template deliberately emits one ``media_pad`` per
    image; after decode, insert the surrounding control tokens and expand that
    one placeholder to the NaViT feature count.
    """
    if len(image_token_counts) != len(image_sizes):
        raise ValueError("Expected one original size for each K3 image.")

    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.detach().flatten().cpu().numpy()
    input_ids = np.asarray(input_ids, dtype=np.int64)

    placeholder_count = np.count_nonzero(input_ids == image_token_id)
    if placeholder_count != len(image_token_counts):
        raise ValueError(
            f"Expected {len(image_token_counts)} image placeholder token(s), "
            f"found {placeholder_count}."
        )

    output = []
    image_index = 0
    for token_id in input_ids:
        if token_id != image_token_id:
            output.append(int(token_id))
            continue

        width, height = image_sizes[image_index]
        output.extend(
            _encode_k3_special_tokens(
                tokenizer,
                f"<|media_begin|>image {width}x{height}<|media_content|>",
            )
        )
        output.extend([image_token_id] * image_token_counts[image_index])
        output.extend(_encode_k3_special_tokens(tokenizer, "<|media_end|>"))
        image_index += 1

    return torch.tensor(output, dtype=torch.long).unsqueeze(0)


def _expand_k3_image_prompt_text(
    input_text: str,
    image_token: str,
    image_token_counts: List[int],
    image_sizes: List[tuple[int, int]],
) -> str:
    """Render the K3 media framing for the CPU HF-processor fallback."""
    parts = input_text.split(image_token)
    if len(parts) - 1 != len(image_token_counts):
        raise ValueError(
            f"Expected {len(image_token_counts)} image placeholder(s), "
            f"found {len(parts) - 1}."
        )

    output = [parts[0]]
    for image_token_count, (width, height), suffix in zip(
        image_token_counts, image_sizes, parts[1:]
    ):
        output.extend(
            (
                f"<|media_begin|>image {width}x{height}<|media_content|>",
                image_token * image_token_count,
                "<|media_end|>",
                suffix,
            )
        )
    return "".join(output)


def _k3_to_cuda_chw(image: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
    if isinstance(image, Image.Image):
        return to_chw_uint8(image, device="cuda")

    image = image.cuda()
    if image.dim() == 2:
        image = image.unsqueeze(0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return image


class KimiK3GPUProcessorWrapper(KimiGPUProcessorWrapper):
    def __init__(self, *args, transparent_bg_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._transparent_bg_config = transparent_bg_config

    def _prepare_input_ids(
        self, input_text, resize_configs, original_input_ids, image_sizes
    ):
        image_token_counts = [config["num_tokens"] for config in resize_configs]
        if original_input_ids is None:
            original_input_ids = _encode_k3_special_tokens(
                self._hf_processor.tokenizer, input_text
            )
        return _expand_k3_image_prompt_token_ids(
            original_input_ids,
            self._image_token_id,
            image_token_counts,
            image_sizes,
            self._hf_processor.tokenizer,
        )

    def __call__(self, text=None, images=None, **kwargs):
        images = images or kwargs.pop("images", None)
        original_input_ids = kwargs.pop("sglang_original_input_ids", None)
        if images and torch.cuda.is_available():
            return self._gpu_call(text, images, original_input_ids)
        return self._cpu_call(text, images, original_input_ids, **kwargs)

    def _gpu_call(self, text, images, original_input_ids=None):
        input_text = text[0] if isinstance(text, list) else text

        resize_configs = []
        image_sizes = []
        for image in images:
            w, h = _get_image_dimensions(image)
            image_sizes.append((w, h))
            resize_configs.append(
                navit_resize_config(
                    w,
                    h,
                    self._patch_size,
                    self._merge_kernel_size,
                    self._in_patch_limit,
                    self._patch_limit_on_one_side,
                    self._fixed_output_tokens,
                )
            )

        input_ids = self._prepare_input_ids(
            input_text, resize_configs, original_input_ids, image_sizes
        )

        image_scale, image_bias = self._get_gpu_norm_tensors()
        # Shared source-compatible batched pipeline (same as K2.5): RGBA
        # inputs land in their own source-shape groups, and the
        # transparent-background compositing runs on each resized batch
        # before patchify -- identical order to the previous per-image path.
        pixel_values, grid_thws = _gpu_preprocess_images(
            images,
            resize_configs,
            image_scale,
            image_bias,
            self._patch_size,
            to_chw=_k3_to_cuda_chw,
            post_resize=lambda x: _fill_transparent_bg(x, self._transparent_bg_config),
        )

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thws,
        }

    def _cpu_call(self, text, images, original_input_ids=None, **kwargs):
        """HF fallback with the same K3 media framing as the GPU path."""
        input_text = text[0] if isinstance(text, list) else text
        if not images:
            return self._hf_processor(text=[input_text], **kwargs)

        image_sizes = [_get_image_dimensions(image) for image in images]
        image_token_counts = [
            self._hf_processor.media_processor.media_tokens_calculator(
                {"type": "image", "image": image}
            )
            for image in images
        ]
        expanded_text = _expand_k3_image_prompt_text(
            input_text,
            self._image_token,
            image_token_counts,
            image_sizes,
        )
        kwargs["medias"] = [{"type": "image", "image": image} for image in images]
        out = self._hf_processor(text=[expanded_text], **kwargs)
        out["input_ids"] = self._prepare_input_ids(
            input_text,
            [{"num_tokens": count} for count in image_token_counts],
            original_input_ids,
            image_sizes,
        )
        grid_thws = out.pop("grid_thws", None)
        if grid_thws is not None:
            out["image_grid_thw"] = grid_thws
        return out

    def prepare_deferred(self, text, images, original_input_ids=None):
        input_text = text[0] if isinstance(text, list) else text
        image_sizes = [_get_image_dimensions(image) for image in images]
        resize_configs = [
            navit_resize_config(
                width,
                height,
                self._patch_size,
                self._merge_kernel_size,
                self._in_patch_limit,
                self._patch_limit_on_one_side,
                self._fixed_output_tokens,
            )
            for width, height in image_sizes
        ]
        input_ids = self._prepare_input_ids(
            input_text, resize_configs, original_input_ids, image_sizes
        )
        deferred_config = {
            "image_mean": list(self._image_mean),
            "image_std": list(self._image_std),
            "transparent_bg_config": self._transparent_bg_config,
        }
        return input_ids, resize_configs, deferred_config


class KimiK3ImageProcessor(KimiGridMMDataMixin, SGLangBaseProcessor):
    models = [KimiK3ForConditionalGeneration]
    # K3 accuracy is sensitive to the chroma upsampling used for common 4:2:0
    # JPEG inputs. This mode uses interpolated nvJPEG upsampling when the K3
    # image dependency is installed and otherwise falls back to PIL.
    gpu_image_decode = "nvjpeg_fancy"
    prefer_tokenized_input = True
    precompute_hash_before_cpu_transfer = True
    auto_mm_processor_worker_num = 2
    auto_mm_io_worker_num = 16
    supports_mm_processor_concurrency = True
    preserve_processor_input_ids = True

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        mm_tokens = MultimodalSpecialTokens(
            image_token="<|media_pad|>",
            image_token_id=hf_config.media_placeholder_token_id,
            image_token_regex=re.compile(r"(?:<\|media_pad\|>)+"),
        ).build(_processor)

        media_proc_cfg = _processor.media_processor.media_proc_cfg

        processor = KimiK3GPUProcessorWrapper(
            _processor,
            image_token=mm_tokens.image_token,
            image_token_id=mm_tokens.image_token_id,
            patch_size=media_proc_cfg["patch_size"],
            merge_kernel_size=media_proc_cfg["merge_kernel_size"],
            in_patch_limit=media_proc_cfg["in_patch_limit"],
            patch_limit_on_one_side=media_proc_cfg["patch_limit_on_one_side"],
            fixed_output_tokens=media_proc_cfg.get("fixed_output_tokens"),
            image_mean=media_proc_cfg["image_mean"],
            image_std=media_proc_cfg["image_std"],
            transparent_bg_config=media_proc_cfg.get("transparent_bg_config"),
        )
        super().__init__(hf_config, server_args, processor, *args, **kwargs)
        self.mm_tokens = mm_tokens

    def _should_defer_gpu_preprocessing(self, images) -> bool:
        if (
            not images
            or self.mm_feature_transport != "cpu"
            or not is_cuda()
            or not all(
                isinstance(image, Image.Image)
                or (isinstance(image, torch.Tensor) and image.dtype == torch.uint8)
                for image in images
            )
        ):
            return False

        raw_bytes = 0
        processed_bytes = 0
        patch_size = self._processor._patch_size
        for image in images:
            width, height = _get_image_dimensions(image)
            resize_config = navit_resize_config(
                width,
                height,
                patch_size,
                self._processor._merge_kernel_size,
                self._processor._in_patch_limit,
                self._processor._patch_limit_on_one_side,
                self._processor._fixed_output_tokens,
            )
            if isinstance(image, torch.Tensor):
                channels = (
                    3 if image.dim() == 2 or image.shape[0] == 1 else image.shape[0]
                )
            else:
                channels = (
                    4
                    if image.mode != "RGB"
                    and ("A" in image.getbands() or "transparency" in image.info)
                    else 3
                )
            raw_bytes += channels * width * height
            padded_width = resize_config["new_width"] + resize_config["pad_width"]
            padded_height = resize_config["new_height"] + resize_config["pad_height"]
            processed_bytes += 3 * padded_width * padded_height * torch.float32.itemsize

        return raw_bytes <= processed_bytes

    def _build_deferred_output(self, base_output):
        input_ids, resize_configs, deferred_config = self._processor.prepare_deferred(
            base_output.input_text,
            base_output.images,
            base_output.input_ids,
        )
        offsets = self.get_mm_items_offset(
            input_ids.flatten(), self.mm_tokens.image_token_id
        )
        if len(offsets) != len(base_output.images):
            raise ValueError("Expected one Kimi-K3 image span for each image")

        items = []
        for image, resize_config, offset in zip(
            base_output.images, resize_configs, offsets
        ):
            grid_thw = _grid_thw_from_resize_config(
                resize_config, self._processor._patch_size
            )
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=to_chw_uint8(image),
                offsets=[offset],
                model_specific_data={
                    "image_grid_thw": torch.tensor([grid_thw], dtype=torch.int64),
                    DEFERRED_PREPROCESSING_KEY: {
                        **deferred_config,
                        "resize_config": resize_config,
                    },
                },
            )
            items.append(item)

        self._precompute_hashes_before_cpu_transfer(items)
        return MultimodalProcessorOutput(
            input_ids=input_ids.flatten().tolist(),
            mm_items=items,
            im_token_id=self.mm_tokens.image_token_id,
        )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        if getattr(request_obj, "video_data", None) or kwargs.get("audio_data"):
            raise ValueError("Kimi-K3 supports image input only")

        expected_image_count = len(image_data or [])
        placeholder_count = self.count_image_placeholders(
            input_text, self.mm_tokens.image_token_id
        )
        if placeholder_count is not None:
            if placeholder_count != expected_image_count:
                raise ValueError(
                    "Kimi image placeholders must map one-to-one to image data: "
                    f"expected {expected_image_count}, found {placeholder_count} token(s)"
                )
            # Keep structural media tokens distinct from user text that happens to
            # spell ``<|media_pad|>``. Decoding the whole prompt and matching the
            # resulting string would lose that distinction and could bind an image
            # to user-provided text instead of the renderer-inserted token.
            base_output = await self.fast_load_mm_data(
                prompt=input_text,
                image_data=image_data,
                multimodal_tokens=self.mm_tokens,
                discard_alpha_channel=False,
                # Unlike load_mm_data, fast_load_mm_data does not derive
                # input_ids from the prompt. Without this the wrapper falls
                # back to re-encoding the decoded string, which is the loss of
                # the structural/user distinction described above.
                input_ids=input_text,
            )
        else:
            base_output = await self.load_mm_data(
                prompt=input_text,
                image_data=image_data,
                multimodal_tokens=self.mm_tokens,
                discard_alpha_channel=False,
            )

        if len(base_output.images) != expected_image_count:
            raise ValueError(
                "Kimi image placeholders must map one-to-one to image data: "
                f"expected {expected_image_count}, loaded {len(base_output.images)}"
            )

        if self._should_defer_gpu_preprocessing(base_output.images):
            return self._build_deferred_output(base_output)

        mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
            base_output,
            self.mm_tokens,
            sglang_original_input_ids=base_output.input_ids,
        )

        # K3's tower is unconditionally image-wise data-parallel (each image
        # is consumed by exactly one TP rank), so keep IPC proxies lazy until
        # that assignment is known: one tokenizer/scheduler crossing per
        # image instead of one per rank. K2.5 gates this on
        # --mm-enable-dp-encoder; K3 needs no flag.
        if self.keep_mm_features_on_device:
            for item in mm_items:
                item.model_specific_data[DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY] = (
                    True
                )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
        )

    def get_mm_data(self, prompt, embeddings, **kwargs):
        img_grid_thw = kwargs.get("img_grid_thw", None)
        output = self._build_kimi_mm_data_from_grids(
            prompt=prompt,
            embeddings=embeddings,
            image_token_id=self.mm_tokens.image_token_id,
            img_grid_thw=img_grid_thw,
        )
        image_sizes = kwargs.get("original_image_sizes")
        if image_sizes is None:
            return output

        counts = [self._num_image_tokens_from_grid(grid) for grid in img_grid_thw]
        if len(image_sizes) != len(counts):
            raise ValueError(
                "Expected one original image size for each K3 encoder grid."
            )
        output.input_ids = (
            _expand_k3_image_prompt_token_ids(
                prompt,
                self.mm_tokens.image_token_id,
                counts,
                [tuple(size) for size in image_sizes],
                self._tokenizer,
            )
            .flatten()
            .tolist()
        )

        search_start = 0
        for item, count in zip(output.mm_items, counts):
            start = output.input_ids.index(self.mm_tokens.image_token_id, search_start)
            item.offsets = [(start, start + count - 1)]
            search_start = start + count
        return output
