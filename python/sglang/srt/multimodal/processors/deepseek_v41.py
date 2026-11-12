"""DeepSeek-V4.1 image preprocessing, preserving raw token IDs for Engram."""

import asyncio
import logging
from functools import partial

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalProcessorOutput
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.multimodal.cache import resolve_multimodal_item_hash
from sglang.srt.multimodal.deepseek_v41_image_processing import (
    GPU_PLAN_KEY,
    image_token_types,
    patchify_image,
    patchify_image_rust,
    prepare_image_gpu,
)
from sglang.srt.multimodal.media_artifacts.base import (
    MediaArtifactCacheMixin,
    MediaArtifactInput,
)
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.runtime_context import get_mm
from sglang.srt.rust_extensions import load_rust_extension

logger = logging.getLogger(__name__)


class DeepseekV41ImagePreprocessArtifact(msgspec.Struct, frozen=True):
    content_digest: str
    artifact_key: str
    feature_hash: int
    feature: torch.Tensor
    n_vit_h: int
    n_vit_w: int
    token_count: int

    @property
    def has_feature(self) -> bool:
        return True

    def cache_value(self) -> "DeepseekV41ImagePreprocessArtifact":
        return self

    def cache_size_items(self) -> tuple:
        return msgspec.structs.astuple(self)


class DeepseekV41ImageProcessor(MediaArtifactCacheMixin, BaseMultimodalProcessor):
    models = [DeepseekV4ForCausalLM]
    preserve_processor_input_ids = True
    prefer_tokenized_input = True
    gpu_image_decode = False
    artifact_modality = Modality.IMAGE
    auto_mm_preprocess_cache_size_mb = 256

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

    async def _run_preprocess_and_build_artifact_batch(self, entries):
        # patchify must not block the tokenizer event loop
        return await asyncio.get_running_loop().run_in_executor(
            self.io_executor, self.prepare_artifact_batch, entries
        )

    def prepare_artifact_batch(
        self, entries: list[MediaArtifactInput]
    ) -> list[DeepseekV41ImagePreprocessArtifact]:
        artifacts = []
        for entry in entries:
            feature, n_vit_h, n_vit_w, lh, lw = self.cpu_patchify(
                entry.media, self.hf_config
            )
            artifacts.append(
                DeepseekV41ImagePreprocessArtifact(
                    content_digest=entry.content_digest,
                    artifact_key=entry.artifact_key,
                    # Use the same feature-hash contract as MultimodalDataItem.
                    feature_hash=resolve_multimodal_item_hash(
                        feature=feature, namespace=entry.artifact_key
                    ),
                    feature=feature,
                    n_vit_h=n_vit_h,
                    n_vit_w=n_vit_w,
                    token_count=len(image_token_types(lh, lw)),
                )
            )
        return artifacts

    async def process_mm_data_async(
        self, image_data, input_text, request_obj, *args, **kwargs
    ):
        if (
            not image_data
            or any(self._is_preprocessed_input(item) for item in image_data)
            or not self.mm_preprocess_cache.enabled
            or self.preprocess_backend == "gpu"
        ):
            return await self._process_mm_data_uncached(
                image_data, input_text, request_obj, *args, **kwargs
            )
        artifacts = await self.prepare_media_artifacts(
            image_data,
            content_hashes=request_obj.mm_content_hashes,
        )
        return self.compose_request(input_text, artifacts)

    def compose_request(
        self,
        input_text,
        artifacts: list[DeepseekV41ImagePreprocessArtifact],
    ) -> MultimodalProcessorOutput:
        ids = (
            input_text
            if isinstance(input_text, list)
            else self._processor.encode(input_text)
        )

        if ids.count(self.image_token_id) != len(artifacts):
            raise ValueError("Image placeholders and images must match")

        artifacts = iter(artifacts)
        tokens, items = [], []

        for token in ids:
            if token != self.image_token_id:
                tokens.append(token)
                continue

            artifact = next(artifacts)
            patches = artifact.feature

            if self.keep_mm_features_on_device:
                patches = patches.to(torch.device("cuda", self.server_args.base_gpu_id))

            start = len(tokens)
            tokens.extend([self.image_token_id] * artifact.token_count)
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=patches,
                offsets=[(start, start + artifact.token_count - 1)],
                model_specific_data={
                    "n_vit_h": artifact.n_vit_h,
                    "n_vit_w": artifact.n_vit_w,
                },
            )
            # Adopt the artifact digest; set_hash would rehash the patches per request.
            item.set_identity(artifact.feature_hash.to_bytes(32, "big"))
            items.append(item)

        return MultimodalProcessorOutput(
            input_ids=tokens,
            mm_items=self._prepare_mm_items_for_transport(items),
            im_token_id=self.image_token_id,
        )

    async def _process_mm_data_uncached(
        self, image_data, input_text, request_obj, *args, **kwargs
    ):
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
