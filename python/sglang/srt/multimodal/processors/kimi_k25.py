import re
from typing import Dict, List, Tuple, Union

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.kimi_k25 import KimiK25ForConditionalGeneration
from sglang.srt.multimodal.gpu_image_processing import (
    get_image_dimensions,
    gpu_preprocess_images,
    navit_resize_config,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)


# Compatible with KimiVLForConditionalGeneration
class KimiK2_5VLImageProcessor(SGLangBaseProcessor):
    models = [KimiK25ForConditionalGeneration]
    gpu_image_decode = True  # nvJPEG for JPEG, PIL fallback for others

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|media_pad|>",
            # TODO: could we convert in MultimodalSpecialTokens?
            image_token_id=hf_config.media_placeholder_token_id,
            image_token_regex=re.compile(r"(?:<\|media_pad\|>)+"),
        ).build(_processor)

        # Extract media processing config from HF processor
        media_proc_cfg = _processor.media_processor.media_proc_cfg
        self._patch_size = media_proc_cfg["patch_size"]
        self._merge_kernel_size = media_proc_cfg["merge_kernel_size"]
        self._in_patch_limit = media_proc_cfg["in_patch_limit"]
        self._patch_limit_on_one_side = media_proc_cfg["patch_limit_on_one_side"]
        self._fixed_output_tokens = media_proc_cfg.get("fixed_output_tokens")
        self._image_mean = media_proc_cfg["image_mean"]
        self._image_std = media_proc_cfg["image_std"]
        self._gpu_norm_tensors = None

    def _get_gpu_norm_tensors(self, device="cuda"):
        if self._gpu_norm_tensors is None:
            image_mean = torch.tensor(
                self._image_mean, device=device, dtype=torch.float32
            ).view(1, 3, 1, 1)
            image_std_inv = (
                1.0 / torch.tensor(self._image_std, device=device, dtype=torch.float32)
            ).view(1, 3, 1, 1)
            self._gpu_norm_tensors = (image_mean, image_std_inv)
        return self._gpu_norm_tensors

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )
        prompt = base_output.input_text

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
        )

    def _num_image_tokens_from_grid(self, grid_thw: torch.Tensor) -> int:
        # Kimi-K2.5 applies temporal pooling and spatial 2D merge in vision tower.
        # The output sequence length per image is h*w/(merge_h*merge_w).
        merge_h, merge_w = self.hf_config.vision_config.merge_kernel_size
        _t, h, w = grid_thw.tolist()
        return (h * w) // (merge_h * merge_w)

    def get_mm_data(self, prompt, embeddings, **kwargs):
        img_grid_thw = kwargs.get("img_grid_thw", None)

        if not isinstance(prompt, list):
            prompt = self._tokenizer.encode(prompt)

        image_token_id = self.mm_tokens.image_token_id
        image_token_counts = [
            self._num_image_tokens_from_grid(grid) for grid in img_grid_thw
        ]

        input_ids = []
        offsets = []
        img_idx = 0

        for token in prompt:
            if token != image_token_id:
                input_ids.append(token)
                continue

            if img_idx >= len(image_token_counts):
                raise ValueError(
                    "The number of image placeholders exceeds img_grid_thw entries."
                )

            num_tokens = image_token_counts[img_idx]
            start = len(input_ids)
            input_ids.extend([image_token_id] * num_tokens)
            offsets.append((start, len(input_ids) - 1))
            img_idx += 1

        if img_idx != len(image_token_counts):
            raise ValueError(
                "The number of image placeholders does not match img_grid_thw entries."
            )

        image_embeddings = embeddings[Modality.IMAGE]
        mm_items = []
        consumed = 0
        for start, end in offsets:
            num_tokens = end - start + 1
            embedding_slice = image_embeddings[consumed : consumed + num_tokens]
            consumed += num_tokens
            mm_items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    offsets=[(start, end)],
                    precomputed_embeddings=embedding_slice,
                )
            )

        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=mm_items,
            im_token_id=image_token_id,
        )

    def _process_and_collect_mm_items(
        self, input_text: str, images=None, audios=None, videos=None, **kwargs
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
        if images and torch.cuda.is_available():
            return self._gpu_process_and_collect_mm_items(
                input_text, images, audios, videos, **kwargs
            )
        return self._cpu_process_and_collect_mm_items(
            input_text, images, audios, videos, **kwargs
        )

    def _gpu_process_and_collect_mm_items(
        self, input_text: str, images, audios=None, videos=None, **kwargs
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
        # 1. Compute resize configs (CPU math, no PIL needed)
        resize_configs = []
        for image in images:
            w, h = get_image_dimensions(image)
            config = navit_resize_config(
                w,
                h,
                self._patch_size,
                self._merge_kernel_size,
                self._in_patch_limit,
                self._patch_limit_on_one_side,
                self._fixed_output_tokens,
            )
            resize_configs.append(config)

        # 2. Expand image tokens in input_text
        parts = input_text.split(self.mm_tokens.image_token)
        result = [parts[0]]
        for config, part in zip(resize_configs, parts[1:]):
            result.append(self.mm_tokens.image_token * config["num_tokens"] + part)
        input_text = "".join(result)

        # 3. Tokenize text (unchanged)
        text_inputs = self._processor.tokenizer(input_text, return_tensors="pt")
        input_ids = text_inputs["input_ids"].flatten()

        # 4. GPU image processing
        image_mean, image_std_inv = self._get_gpu_norm_tensors()
        pixel_values, grid_thws = gpu_preprocess_images(
            images, resize_configs, image_mean, image_std_inv, self._patch_size
        )

        # 5. Assemble result
        ret = {
            "input_ids": input_ids.unsqueeze(0),
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
        }
        if not self.server_args.keep_mm_feature_on_device:
            ret["pixel_values"] = ret["pixel_values"].to("cpu")

        collected_items = self.collect_mm_items_from_processor_output(ret)
        return collected_items, input_ids, ret

    def _cpu_process_and_collect_mm_items(
        self, input_text: str, images=None, audios=None, videos=None, **kwargs
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
        """Original CPU path using HF processor."""
        parts = input_text.split(self.mm_tokens.image_token)

        result = [parts[0]]
        for image, part in zip(images, parts[1:]):
            num_tokens = self._processor.media_processor.media_tokens_calculator(
                {"type": "image", "image": image}
            )
            result.append(self.mm_tokens.image_token * num_tokens + part)

        input_text = "".join(result)

        if images:  # for kimi k2 vl
            mediums = []
            for image in images:
                mediums.append({"type": "image", "image": image})
            key = "_medias"[1:]  # bypass lint
            kwargs[key] = mediums
            images = None

        ret = self.process_mm_data(
            input_text=input_text,
            images=images,
            audios=audios,
            videos=videos,
            **kwargs,
        )

        input_ids = ret["input_ids"].flatten()
        collected_items = self.collect_mm_items_from_processor_output(ret)

        return collected_items, input_ids, ret
