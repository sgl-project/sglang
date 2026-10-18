"""DeepSeek-V4.1 image preprocessing, preserving raw token IDs for Engram."""

import asyncio
import logging
from functools import partial

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    MM_PAD_SHIFT_VALUE,
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.multimodal.deepseek_v41_image_processing import (
    GPU_PLAN_KEY,
    image_token_types,
    num_image_tokens,
    patchify_image,
    patchify_image_rust,
    prepare_image_gpu,
)
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
        if backend == "pil" or get_mm().disable_fast_image_processor:
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
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=_processor.convert_ids_to_tokens(self.image_token_id),
            image_token_id=self.image_token_id,
        ).build(_processor)

    def preprocess_fingerprint_payload(self):
        payload = super().preprocess_fingerprint_payload()
        payload["dsv41_preprocess_backend"] = self.preprocess_backend
        return payload

    async def process_mm_data_async(
        self, image_data, input_text, request_obj, *args, **kwargs
    ):
        # EPD/PD decode instances receive the already-expanded input_ids, where
        # image positions hold MM_PAD_SHIFT_VALUE+hash sentinels that are out of
        # vocab. The image embeddings were produced during Prefill, so
        # re-processing here is redundant (and decoding the sentinel ids back to
        # text would fail). A pad sentinel is the reliable signal that the ids
        # are already expanded.
        req_input_ids = getattr(request_obj, "input_ids", None)
        if req_input_ids is not None and any(
            int(t) >= MM_PAD_SHIFT_VALUE for t in req_input_ids
        ):
            return MultimodalProcessorOutput(
                input_ids=list(req_input_ids),
                mm_items=[],
                im_token_id=self.image_token_id,
            )
        base = await self.load_mm_data(
            input_text, image_data=image_data, multimodal_tokens=self.mm_tokens
        )
        ids = (
            input_text
            if isinstance(input_text, list)
            else self._processor.encode(input_text)
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

    def get_mm_data(self, prompt, embeddings, *, img_grid_thw=None, **kwargs):
        """EPD P-side rebuild: expand each image placeholder into its span.

        The encoder ships one embedding block per image — the complete sentinel
        span ([START] + rows + [END], ``num_image_tokens(lh, lw)`` rows) built by
        get_image_feature. V4.1 spans are position-independent, so the blocks
        are spliced verbatim at the placeholder positions; ``img_grid_thw``
        carries [t, n_llm_h, n_llm_w] per image (see preprocess_mm_for_encoder).
        """
        image_embeddings = (
            embeddings.get(Modality.IMAGE)
            if isinstance(embeddings, dict)
            else embeddings
        )
        if image_embeddings is None:
            raise ValueError(
                "DeepSeek-V4.1 EPD get_mm_data received no image embeddings"
            )
        if img_grid_thw is None:
            raise ValueError("DeepSeek-V4.1 EPD get_mm_data requires img_grid_thw")

        ids = prompt if isinstance(prompt, list) else self._processor.encode(prompt)
        grid = torch.as_tensor(img_grid_thw).reshape(-1, 3)
        num_placeholders = sum(1 for t in ids if t == self.image_token_id)
        if num_placeholders != grid.shape[0]:
            raise ValueError(
                f"Found {num_placeholders} image placeholders but got "
                f"{grid.shape[0]} image grids"
            )

        tokens, items = [], []
        image_index = 0
        cursor = 0
        for token in ids:
            if token != self.image_token_id:
                tokens.append(token)
                continue
            lh, lw = int(grid[image_index][-2]), int(grid[image_index][-1])
            count = num_image_tokens(lh, lw)
            block = image_embeddings[cursor : cursor + count]
            cursor += count
            if block.shape[0] != count:
                raise ValueError(
                    f"DSV4.1 EPD block length mismatch: expected {count} rows "
                    f"for grid ({lh}, {lw}), got {block.shape[0]}"
                )
            start = len(tokens)
            tokens.extend([self.image_token_id] * count)
            items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    offsets=[(start, start + count - 1)],
                    precomputed_embeddings=block,
                )
            )
            image_index += 1
        if cursor != image_embeddings.shape[0]:
            raise ValueError(
                f"DSV4.1 EPD embeddings not fully consumed: used {cursor} of "
                f"{image_embeddings.shape[0]} rows"
            )

        return MultimodalProcessorOutput(
            input_ids=tokens,
            mm_items=items,
            im_token_id=self.image_token_id,
        )
