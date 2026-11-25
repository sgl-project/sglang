"""DeepSeek-V4.1 image preprocessing, preserving raw token IDs for Engram."""

import asyncio
import logging
from functools import partial

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalProcessorOutput
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.multimodal.deepseek_v41_image_processing import (
    GPU_PLAN_KEY,
    image_token_types,
    patchify_image,
    patchify_image_rust,
    prepare_image_gpu,
)
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.runtime_context import get_mm
from sglang.srt.rust_extensions import load_rust_extension

logger = logging.getLogger(__name__)


class DeepseekV41ImageProcessor(BaseMultimodalProcessor):
    models = [DeepseekV4ForCausalLM]
    preserve_processor_input_ids = True
    prefer_tokenized_input = True
    gpu_image_decode = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        self.preprocess_backend = "cpu"
        self.cpu_patchify = patchify_image
        backend = get_mm().image_processor_backend
        if backend == "pil":
            self.preprocess_backend = "cpu"
        elif envs.SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU.get():
            self.preprocess_backend = "gpu"
        elif backend == "auto":
            # Resolved before super().__init__(), which fingerprints the backend;
            # a load failure falls back to PIL, later image errors do not.
            try:
                extension = load_rust_extension(
                    "sglang.srt.rust_extensions._multimodal"
                )
                resize_patchify = extension.dsv41.resize_patchify
            except (ImportError, OSError, RuntimeError, AttributeError) as error:
                logger.warning(
                    "V4.1 Rust image processor unavailable; using PIL: %s", error
                )
            else:
                self.preprocess_backend = "rust"
                self.cpu_patchify = partial(
                    patchify_image_rust, resize_patchify=resize_patchify
                )
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.image_token_id = hf_config.image_token_id
        # Every image-span position uses the same token id: only the span
        # length distinguishes the start, feature and newline rows.
        self.IM_TOKEN_ID = self.image_token_id
        self.IM_START_TOKEN_ID = self.image_token_id
        self.IM_END_TOKEN_ID = self.image_token_id
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=_processor.convert_ids_to_tokens(self.image_token_id),
            image_token_id=self.image_token_id,
        ).build(_processor)

    def get_mm_data(self, prompt, embeddings, **kwargs):
        """Rebuild V4.1 image spans around embeddings received from an encoder."""
        img_grid_thw = kwargs.get("img_grid_thw")
        if img_grid_thw is None:
            return super().get_mm_data(prompt, embeddings, **kwargs)

        tokens = prompt if isinstance(prompt, list) else self._processor.encode(prompt)
        grids = (
            img_grid_thw.tolist()
            if isinstance(img_grid_thw, torch.Tensor)
            else img_grid_thw
        )

        input_ids, spans = [], []
        grid_idx = 0
        for token in tokens:
            if token == self.image_token_id and grid_idx < len(grids):
                _, n_llm_h, n_llm_w = (int(value) for value in grids[grid_idx])
                count = 2 + (n_llm_w + 1) * n_llm_h
                start = len(input_ids)
                input_ids.extend([self.image_token_id] * count)
                spans.append((start, len(input_ids) - 1))
                grid_idx += 1
            else:
                input_ids.append(token)

        # pad_value stays unset: the scheduler derives the MM_PAD_SHIFT
        # sentinels from the item hash, and Engram detects image tokens by it.
        image_embeddings = embeddings[Modality.IMAGE]
        mm_items, consumed = [], 0
        for start, end in spans:
            num_rows = end - start + 1
            mm_items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    offsets=[(start, end)],
                    precomputed_embeddings=image_embeddings[
                        consumed : consumed + num_rows
                    ],
                )
            )
            consumed += num_rows

        if grid_idx != len(grids):
            raise ValueError(
                f"Image placeholders ({grid_idx}) do not match grids ({len(grids)})"
            )
        if consumed != image_embeddings.shape[0]:
            raise ValueError(
                f"Encoder returned {image_embeddings.shape[0]} rows for "
                f"{consumed} image-span positions"
            )

        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=mm_items,
            im_token_id=self.image_token_id,
            im_start_id=self.image_token_id,
            im_end_id=self.image_token_id,
        )

    def preprocess_fingerprint_payload(self):
        payload = super().preprocess_fingerprint_payload()
        payload["dsv41_preprocess_backend"] = self.preprocess_backend
        return payload

    async def process_mm_data_async(
        self, image_data, input_text, request_obj, *args, **kwargs
    ):
        if isinstance(input_text, list):
            # Decoded text spells image-token text like a placeholder; the IDs
            # tell them apart, so the count below is the alignment check.
            ids = input_text
            self.validate_mm_data(image_data)
            base = await self.fast_load_mm_data(
                input_text,
                image_data=image_data,
                multimodal_tokens=self.mm_tokens,
                input_ids=ids,
            )
        else:
            ids = self._processor.encode(input_text)
            base = await self.load_mm_data(
                input_text, image_data=image_data, multimodal_tokens=self.mm_tokens
            )
        if ids.count(self.image_token_id) != len(base.images):
            raise ValueError("Image placeholders and images must match")
        images = iter(base.images)
        tokens, items = [], []
        for token in ids:
            if token != self.image_token_id:
                tokens.append(token)
                continue
            image = next(images)
            plan = None
            if self.preprocess_backend == "gpu":
                patches, plan, lh, lw = await asyncio.to_thread(
                    prepare_image_gpu, image, self.hf_config
                )
                h = plan["height"] // plan["patch_size"]
                w = plan["width"] // plan["patch_size"]
            else:
                patches, h, w, lh, lw = await asyncio.to_thread(
                    self.cpu_patchify, image, self.hf_config
                )
            if self.keep_mm_features_on_device:
                patches = patches.to(torch.device("cuda", self.server_args.base_gpu_id))
            metadata = {"n_vit_h": h, "n_vit_w": w}
            if plan is not None:
                metadata[GPU_PLAN_KEY] = plan
            count = len(image_token_types(lh, lw))
            start = len(tokens)
            tokens.extend([self.image_token_id] * count)
            items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=patches,
                    offsets=[(start, start + count - 1)],
                    model_specific_data=metadata,
                )
            )
        return MultimodalProcessorOutput(
            input_ids=tokens,
            mm_items=self._prepare_mm_items_for_transport(items),
            im_token_id=self.image_token_id,
        )
