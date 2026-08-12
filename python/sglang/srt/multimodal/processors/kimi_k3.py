"""Kimi K3 multimodal processor.

GPU image preprocessing dedicated to K3: unlike the K2.5 wrapper it keeps
the alpha channel through the bicubic resize and then composites RGBA
images onto the checkpoint-configured background
(``transparent_bg_config`` with ``transparent_bg_fill_stage ==
"after_resize"`` in preprocessor_config.json), instead of dropping alpha
at load time.
"""

import asyncio
import math
import re
import time
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.mem_cache.multimodal_cache import (
    MM_EMBEDDING_CACHE_IDENTITY_KEY,
    MM_EMBEDDING_CACHE_LEASE_ID_KEY,
)
from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration
from sglang.srt.multimodal.cache import (
    CacheLookup,
    CacheReservation,
    MediaSnapshot,
    PreprocessCacheLookup,
    build_artifact_key,
    build_feature_hash,
    parse_content_hash,
    snapshot_media,
)
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
from sglang.srt.multimodal.processors.kimi_k3_artifact import (
    KimiK3DeferredConfig,
    KimiK3ImageArtifact,
    KimiK3MediaLookup,
    KimiK3ResizeConfig,
)
from sglang.srt.multimodal.processors.kimi_k25 import (
    KimiGPUProcessorWrapper,
    _get_image_dimensions,
    _gpu_preprocess_images,
    _grid_thw_from_resize_config,
    navit_resize_config,
)
from sglang.srt.multimodal.transport.cuda_ipc import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
)
from sglang.srt.utils import ImageData, is_cuda, load_image


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

    def prepare_image_features(self, images):
        """Prepare prompt-independent, per-image features in one processor call."""
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

        if images and torch.cuda.is_available():
            image_scale, image_bias = self._get_gpu_norm_tensors()
            pixel_values, grid_thws = _gpu_preprocess_images(
                images,
                resize_configs,
                image_scale,
                image_bias,
                self._patch_size,
                to_chw=_k3_to_cuda_chw,
                post_resize=lambda x: _fill_transparent_bg(
                    x, self._transparent_bg_config
                ),
            )
        else:
            # The checkpoint CPU processor couples prompt composition with media
            # preprocessing. A synthetic prompt keeps that API but is discarded;
            # image features and grids are independent of its text.
            output = self._cpu_call(self._image_token * len(images), images)
            pixel_values = output["pixel_values"]
            grid_thws = output["image_grid_thw"]

        grids = [tuple(int(value) for value in grid) for grid in grid_thws.tolist()]
        patch_counts = [math.prod(grid) for grid in grids]
        if sum(patch_counts) != pixel_values.shape[0]:
            raise ValueError(
                "Kimi-K3 processor feature length does not match image grids: "
                f"{pixel_values.shape[0]} != {sum(patch_counts)}"
            )
        return (
            list(pixel_values.split(patch_counts)),
            image_sizes,
            resize_configs,
            grids,
        )


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
    auto_mm_preprocess_cache_size_mb = 256
    supports_early_mm_cache = True
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

    def preprocess_fingerprint_payload(self):
        payload = super().preprocess_fingerprint_payload()
        payload["kimi_k3"] = {
            "patch_size": self._processor._patch_size,
            "merge_kernel_size": self._processor._merge_kernel_size,
            "in_patch_limit": self._processor._in_patch_limit,
            "patch_limit_on_one_side": self._processor._patch_limit_on_one_side,
            "fixed_output_tokens": self._processor._fixed_output_tokens,
            "image_mean": self._processor._image_mean,
            "image_std": self._processor._image_std,
            "transparent_bg_config": self._processor._transparent_bg_config,
            "preprocessing_backend": "gpu" if is_cuda() else "cpu",
            "feature_transport": self.mm_feature_transport,
            "resize_antialias": True,
        }
        return payload

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

    @staticmethod
    def _artifact_preprocess_kwargs(source) -> dict:
        if isinstance(source, ImageData):
            detail = source.detail
            max_dynamic_patch = source.max_dynamic_patch
            preprocess_kwargs = source.preprocess_kwargs
        elif isinstance(source, dict):
            detail = source.get("detail")
            max_dynamic_patch = source.get("max_dynamic_patch")
            preprocess_kwargs = source.get("preprocess_kwargs")
        else:
            return {}
        return {
            key: value
            for key, value in {
                "detail": None if detail == "auto" else detail,
                "max_dynamic_patch": max_dynamic_patch,
                "preprocess_kwargs": preprocess_kwargs,
            }.items()
            if value not in (None, {})
        }

    def _artifact_key(self, content_digest: str, source) -> str:
        return build_artifact_key(
            content_digest,
            modality="image",
            processor_fingerprint=self.processor_fingerprint,
            preprocess_kwargs=self._artifact_preprocess_kwargs(source),
        )

    def _decode_media_snapshot(self, snapshot: MediaSnapshot):
        start = time.perf_counter()
        try:
            data = snapshot.data
            if isinstance(data, torch.Tensor):
                return data
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data)
            if isinstance(data, Image.Image):
                data.load()
                return data
            image, _ = load_image(data, self.gpu_image_decode)
            if isinstance(image, Image.Image):
                image.load()
            return image
        finally:
            self.observe_preprocess_phase("decode", time.perf_counter() - start)

    def _make_artifact(
        self,
        *,
        content_digest: str,
        artifact_key: str,
        original_size: tuple[int, int],
        resize_config: dict,
        grid_thw: tuple[int, int, int],
        feature: torch.Tensor,
        deferred: Optional[KimiK3DeferredConfig] = None,
    ) -> KimiK3ImageArtifact:
        item = MultimodalDataItem(modality=Modality.IMAGE, feature=feature)
        item.set_pad_value()
        feature_hash = build_feature_hash(artifact_key, item.hash)
        if not self.keep_mm_features_on_device and feature.device.type != "cpu":
            feature = feature.cpu()
        return KimiK3ImageArtifact(
            content_digest=content_digest,
            artifact_key=artifact_key,
            feature_hash=feature_hash,
            original_size=original_size,
            resize_config=KimiK3ResizeConfig.from_dict(resize_config),
            grid_thw=grid_thw,
            feature=feature,
            deferred=deferred,
        )

    def _prepare_artifact_batch(
        self,
        entries: list[tuple[str, str, object]],
        *,
        processor=None,
    ) -> list[KimiK3ImageArtifact]:
        """Process cache misses as one batch while preserving image order."""
        processor = processor or self._processor
        artifacts: list[Optional[KimiK3ImageArtifact]] = [None] * len(entries)
        eager_indices = []
        eager_images = []

        for index, (content_digest, artifact_key, image) in enumerate(entries):
            if not self._should_defer_gpu_preprocessing([image]):
                eager_indices.append(index)
                eager_images.append(image)
                continue

            width, height = _get_image_dimensions(image)
            resize_config = navit_resize_config(
                width,
                height,
                processor._patch_size,
                processor._merge_kernel_size,
                processor._in_patch_limit,
                processor._patch_limit_on_one_side,
                processor._fixed_output_tokens,
            )
            grid_thw = _grid_thw_from_resize_config(
                resize_config, processor._patch_size
            )
            feature = to_chw_uint8(image).cpu().contiguous()
            artifacts[index] = self._make_artifact(
                content_digest=content_digest,
                artifact_key=artifact_key,
                original_size=(width, height),
                resize_config=resize_config,
                grid_thw=grid_thw,
                feature=feature,
                deferred=KimiK3DeferredConfig(
                    backend="gpu",
                    feature_layout="chw",
                    image_mean=tuple(processor._image_mean),
                    image_std=tuple(processor._image_std),
                    transparent_bg_config=processor._transparent_bg_config,
                ),
            )

        if eager_images:
            features, sizes, configs, grids = processor.prepare_image_features(
                eager_images
            )
            for index, feature, size, config, grid in zip(
                eager_indices, features, sizes, configs, grids
            ):
                content_digest, artifact_key, _ = entries[index]
                artifacts[index] = self._make_artifact(
                    content_digest=content_digest,
                    artifact_key=artifact_key,
                    original_size=size,
                    resize_config=config,
                    grid_thw=grid,
                    feature=feature,
                )

        if any(artifact is None for artifact in artifacts):
            raise RuntimeError("Kimi-K3 artifact batch did not produce every image")
        return artifacts

    async def _run_artifact_batch(
        self, entries: list[tuple[str, str, object]]
    ) -> list[KimiK3ImageArtifact]:
        start = time.perf_counter()
        try:
            if self.mm_processor_executor is None:
                return self._prepare_artifact_batch(entries)
            return await self.mm_processor_executor.run(
                self._prepare_artifact_batch, entries
            )
        finally:
            self.observe_preprocess_phase("processor", time.perf_counter() - start)

    @staticmethod
    def _artifact_usable(
        artifact: KimiK3ImageArtifact, allow_featureless: bool
    ) -> bool:
        return artifact.has_feature or allow_featureless

    async def prepare_media_artifacts(
        self,
        image_data,
        request_obj,
        *,
        featureless_hit_mask: Optional[list[bool]] = None,
        media_lookups: Optional[list[KimiK3MediaLookup]] = None,
    ) -> list[KimiK3ImageArtifact]:
        """Resolve identities and preprocess only per-image cache misses."""
        image_count = len(image_data)
        if featureless_hit_mask is None:
            featureless_hit_mask = [False] * image_count
        if len(featureless_hit_mask) != image_count:
            raise ValueError("featureless_hit_mask must align with image_data")

        if media_lookups is None:
            lookup = await self.lookup_preprocess_cache(image_data, request_obj)
            media_lookups = lookup.processor_state
        if len(media_lookups) != image_count:
            raise ValueError("media_lookups must align with image_data")

        artifacts: list[Optional[KimiK3ImageArtifact]] = [None] * image_count
        snapshots = [lookup.snapshot for lookup in media_lookups]
        keys = [lookup.artifact_key for lookup in media_lookups]
        load_indices = []
        for index, (lookup, allow_featureless) in enumerate(
            zip(media_lookups, featureless_hit_mask)
        ):
            cached = lookup.cached_artifact
            if cached is not None and self._artifact_usable(cached, allow_featureless):
                # Keep the lookup's strong reference across the scheduler lease
                # round trip; get() here only refreshes LRU recency when present.
                self.mm_preprocess_cache.get(lookup.artifact_key)
                artifacts[index] = cached
            else:
                load_indices.append(index)

        # A trusted metadata hit can skip the first read. If the scheduler does
        # not have its embedding, read and verify now before recomputing features.
        caller_hashes = (
            getattr(request_obj, "mm_content_hashes", None) or [None] * image_count
        )
        for index in load_indices:
            if snapshots[index] is not None:
                continue
            snapshot = await asyncio.wrap_future(
                self.io_executor.submit(snapshot_media, image_data[index])
            )
            expected = parse_content_hash(caller_hashes[index])
            if expected is not None and expected != snapshot.content_digest:
                raise ValueError(
                    f"content hash mismatch for image_data[{index}]: "
                    f"expected {expected}, got {snapshot.content_digest}"
                )
            snapshots[index] = snapshot

        # Deduplicate misses before decode and reserve them against other requests.
        first_index_by_key = {}
        previous_metadata = {}
        for index in load_indices:
            if artifacts[index] is not None:
                continue
            key = keys[index]
            if key not in first_index_by_key:
                first_index_by_key[key] = index
                previous_metadata[key] = self.mm_preprocess_cache.pop(key)

        unique_keys = list(first_index_by_key)
        reservations = self.mm_preprocess_cache.reserve_many(unique_keys)
        resolved_by_key = {}
        owners = []
        for key, reservation in zip(unique_keys, reservations):
            if isinstance(reservation, CacheLookup):
                resolved_by_key[key] = reservation.value
            elif reservation.owner:
                owners.append(reservation)

        if owners:
            owner_task = self.mm_preprocess_cache.create_background_task(
                self._fulfill_artifact_reservations(
                    owners,
                    first_index_by_key,
                    snapshots,
                    previous_metadata,
                    resolved_by_key,
                )
            )
            # The cache owns this shared work. Cancelling the request that won
            # a reservation must not fail another request already joining it.
            await asyncio.shield(owner_task)

        for key, reservation in zip(unique_keys, reservations):
            if isinstance(reservation, CacheReservation) and not reservation.owner:
                resolved_by_key[key] = await self.mm_preprocess_cache.wait(reservation)

        for index in range(image_count):
            if artifacts[index] is None:
                artifacts[index] = resolved_by_key[keys[index]]
        return artifacts

    async def _fulfill_artifact_reservations(
        self,
        owners: list[CacheReservation[str, KimiK3ImageArtifact]],
        first_index_by_key: dict[str, int],
        snapshots: list[Optional[MediaSnapshot]],
        previous_metadata: dict[str, KimiK3ImageArtifact],
        resolved_by_key: dict[str, KimiK3ImageArtifact],
    ) -> None:
        try:
            owner_entries = []
            for reservation in owners:
                index = first_index_by_key[reservation.key]
                snapshot = snapshots[index]
                image = await asyncio.wrap_future(
                    self.io_executor.submit(self._decode_media_snapshot, snapshot)
                )
                owner_entries.append((snapshot.content_digest, reservation.key, image))
            owner_artifacts = await self._run_artifact_batch(owner_entries)
            for reservation, artifact in zip(owners, owner_artifacts):
                old = previous_metadata.get(reservation.key)
                if old is not None and old.feature_hash != artifact.feature_hash:
                    raise ValueError(
                        "Kimi-K3 cached artifact feature hash changed for identical "
                        f"identity {reservation.key}"
                    )
                self.mm_preprocess_cache.fulfill(
                    reservation,
                    artifact,
                    cache_value=artifact.cache_value(),
                )
                resolved_by_key[reservation.key] = artifact
        except BaseException as error:
            for reservation in owners:
                if not reservation.future.done():
                    self.mm_preprocess_cache.fail(reservation, error)
            raise

    async def lookup_preprocess_cache(
        self, image_data, request_obj
    ) -> Optional[PreprocessCacheLookup]:
        """Look up prompt-independent metadata before processor dispatch."""
        if (
            not self.mm_preprocess_cache.enabled
            or not image_data
            or any(self._is_preprocessed_input(item) for item in image_data)
        ):
            return None
        image_count = len(image_data)
        caller_hashes = getattr(request_obj, "mm_content_hashes", None)
        if caller_hashes is None:
            caller_hashes = [None] * image_count
        if len(caller_hashes) != image_count:
            raise ValueError(
                f"mm_content_hashes has {len(caller_hashes)} entries for "
                f"{image_count} images"
            )
        caller_hashes = [parse_content_hash(value) for value in caller_hashes]

        lookups: list[Optional[KimiK3MediaLookup]] = [None] * image_count
        read_indices = []
        for index, (source, caller_hash) in enumerate(zip(image_data, caller_hashes)):
            if self.trust_mm_content_hashes and caller_hash is not None:
                key = self._artifact_key(caller_hash, source)
                cached = self.mm_preprocess_cache.peek(key)
                if cached is not None:
                    lookups[index] = KimiK3MediaLookup(
                        artifact_key=key,
                        content_digest=caller_hash,
                        snapshot=None,
                        cached_artifact=cached,
                    )
                    continue
            read_indices.append(index)

        futures = {
            index: self.io_executor.submit(snapshot_media, image_data[index])
            for index in read_indices
        }
        for index, future in futures.items():
            snapshot = await asyncio.wrap_future(future)
            expected = caller_hashes[index]
            if expected is not None and expected != snapshot.content_digest:
                raise ValueError(
                    f"content hash mismatch for image_data[{index}]: "
                    f"expected {expected}, got {snapshot.content_digest}"
                )
            key = self._artifact_key(snapshot.content_digest, image_data[index])
            lookups[index] = KimiK3MediaLookup(
                artifact_key=key,
                content_digest=snapshot.content_digest,
                snapshot=snapshot,
                cached_artifact=self.mm_preprocess_cache.peek(key),
            )
        resolved = tuple(lookups)
        return PreprocessCacheLookup(
            processor_state=resolved,
            feature_hashes=tuple(
                (
                    lookup.cached_artifact.feature_hash
                    if lookup.cached_artifact is not None
                    else None
                )
                for lookup in resolved
            ),
            feature_identities=tuple(
                (
                    lookup.cached_artifact.artifact_key
                    if lookup.cached_artifact is not None
                    else None
                )
                for lookup in resolved
            ),
            identity_sources=tuple(
                "trusted" if lookup.snapshot is None else "server_computed"
                for lookup in resolved
            ),
        )

    def compose_request(
        self,
        input_text,
        artifacts: list[KimiK3ImageArtifact],
        *,
        featureless_hit_mask: Optional[list[bool]] = None,
        embedding_lease_id: Optional[str] = None,
    ) -> MultimodalProcessorOutput:
        if featureless_hit_mask is None:
            featureless_hit_mask = [False] * len(artifacts)
        if len(featureless_hit_mask) != len(artifacts):
            raise ValueError("featureless_hit_mask must align with artifacts")
        if any(featureless_hit_mask) and embedding_lease_id is None:
            raise ValueError("featureless cache hits require an embedding lease")
        original_ids = (
            input_text
            if isinstance(input_text, (list, torch.Tensor))
            else _encode_k3_special_tokens(self._tokenizer, input_text)
        )
        input_ids = _expand_k3_image_prompt_token_ids(
            original_ids,
            self.mm_tokens.image_token_id,
            [artifact.resize_config.num_tokens for artifact in artifacts],
            [artifact.original_size for artifact in artifacts],
            self._tokenizer,
        ).flatten()
        offsets = self.get_mm_items_offset(input_ids, self.mm_tokens.image_token_id)
        if len(offsets) != len(artifacts):
            raise ValueError("Expected one Kimi-K3 image span for each image")

        items = []
        for artifact, offset, featureless in zip(
            artifacts, offsets, featureless_hit_mask
        ):
            model_specific_data = {
                "image_grid_thw": torch.tensor([artifact.grid_thw], dtype=torch.int64)
            }
            if artifact.deferred is not None:
                model_specific_data[DEFERRED_PREPROCESSING_KEY] = (
                    artifact.deferred.as_dict(artifact.resize_config)
                )
            if featureless:
                model_specific_data[MM_EMBEDDING_CACHE_LEASE_ID_KEY] = (
                    embedding_lease_id
                )
            model_specific_data[MM_EMBEDDING_CACHE_IDENTITY_KEY] = artifact.artifact_key
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=None if featureless else artifact.feature,
                offsets=[offset],
                model_specific_data=model_specific_data,
            )
            item.set_hash(artifact.feature_hash)
            if self.use_cuda_ipc and isinstance(item.feature, torch.Tensor):
                item.feature = self._wrap_tensor_for_cuda_ipc(item.feature)
            if self.keep_mm_features_on_device and item.feature is not None:
                item.model_specific_data[DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY] = (
                    True
                )
            items.append(item)

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=items,
            im_token_id=self.mm_tokens.image_token_id,
        )

    async def _process_mm_data_uncached(
        self, image_data, input_text, request_obj, **kwargs
    ):
        """Compatibility path for precomputed inputs and lightweight test stubs."""
        expected_image_count = len(image_data or [])
        placeholder_count = self.count_image_placeholders(
            input_text, self.mm_tokens.image_token_id
        )
        if placeholder_count is not None:
            base_output = await self.fast_load_mm_data(
                prompt=input_text,
                image_data=image_data,
                multimodal_tokens=self.mm_tokens,
                discard_alpha_channel=False,
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
        if (
            not hasattr(self, "mm_preprocess_cache")
            or any(self._is_preprocessed_input(item) for item in image_data)
            or not self.mm_preprocess_cache.enabled
        ):
            return await self._process_mm_data_uncached(
                image_data, input_text, request_obj, **kwargs
            )

        featureless_hit_mask = kwargs.get("featureless_hit_mask")
        artifacts = await self.prepare_media_artifacts(
            image_data,
            request_obj,
            featureless_hit_mask=featureless_hit_mask,
            media_lookups=kwargs.get("preprocess_cache_lookups"),
        )
        return self.compose_request(
            input_text,
            artifacts,
            featureless_hit_mask=featureless_hit_mask,
            embedding_lease_id=kwargs.get("embedding_lease_id"),
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
