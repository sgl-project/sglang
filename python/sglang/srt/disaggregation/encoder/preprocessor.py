"""CPU-bound multimodal preprocessing for the EPD encoder.

This module is designed to be replaceable by a Rust implementation.
It handles all CPU-bound work: media I/O (image/video/audio loading),
HF processor calls, config validation, and related helper computations.
GPU tensor operations remain in :mod:`server.MMEncoder`.
"""

import asyncio
import concurrent.futures
import functools
import logging
from dataclasses import dataclass
from http import HTTPStatus
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoProcessor

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import (
    get_attn_tensor_model_parallel_rank,
    get_attn_tensor_model_parallel_world_size,
    get_attn_tp_group,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.cache import parse_content_hash, snapshot_media
from sglang.srt.multimodal.encoder_preprocessing import (
    EncoderMediaProcessorConfig,
    EncoderPreprocessOutput,
    invoke_encoder_preprocessor,
)
from sglang.srt.multimodal.processors.glm4v import (
    _glm_effective_presize_budget,
    glm_budget_kwargs,
    glm_decode_frames_at,
    glm_max_image_tokens_from_configs,
    glm_processor_video_config,
    glm_sample_and_decode_sync,
    glm_sample_frame_indices,
    preprocess_video_frames_sync,
    split_glm_video_items,
)
from sglang.srt.multimodal.processors.qwen_vl import preprocess_video
from sglang.srt.runtime_context import (
    get_device,
    get_mm,
    get_model,
    get_parallel,
    get_serving,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    CLIENT_MEDIA_EXCEPTIONS,
    load_audio,
    load_image,
    load_video,
)
from sglang.srt.utils.hf_transformers_utils import resolve_image_processor_backend

logger = logging.getLogger(__name__)


_mm_grid_attrs = {
    # Kimi K2.5/K3 HF processors use grid_thws (see base_processor.ATTR_NAME_TO_MODALITY).
    Modality.IMAGE: ("image_grid_thw", "image_grid_hws", "grid_thws"),
    Modality.VIDEO: ("video_grid_thw",),
    Modality.AUDIO: ("audio_feature_lens_raw",),
}


def _convert(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.tensor(data)
    elif isinstance(data, list) and isinstance(data[0], np.ndarray):
        return torch.tensor(np.array(data))
    elif isinstance(data, list) and isinstance(data[0], (int, float)):
        return torch.tensor(data)
    else:
        return data


def _get_original_image_size(image):
    """Return an image's original (width, height) before encoder preprocessing."""
    if isinstance(image, dict):
        image = image.get("image")
    if isinstance(image, torch.Tensor):
        if image.ndim < 2:
            raise ValueError(f"Invalid image tensor shape: {tuple(image.shape)}")
        return [int(image.shape[-1]), int(image.shape[-2])]
    if hasattr(image, "size"):
        width, height = image.size
        return [int(width), int(height)]
    raise TypeError(f"Cannot determine original image size from {type(image)}")


@dataclass
class EncoderPreprocessResult:
    mm_inputs: dict
    grid_thw: Union[torch.Tensor, List]
    token_counts: List[int]


class EncoderPreprocessor:
    """CPU-bound multimodal preprocessing pipeline.

    Takes raw media URLs / base64 data and produces HF processor output dicts
    (CPU tensors).  The GPU model is never touched here — only the HF
    image/video/audio processors are invoked.

    Parameters
    ----------
    server_args : ServerArgs
        Server configuration (model path, processor flags, etc.).
    model_config : ModelConfig
        Model configuration (hf_config, hidden_size, etc.).
    model_preprocessor : callable, optional
        Optional model-specific preprocessor (``model.preprocess_mm_for_encoder``).
        When provided, overrides the default HF processor path for the given
        modality.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
        encoder_media_processor_config: EncoderMediaProcessorConfig,
        model_preprocessor: Optional[Callable] = None,
    ):
        self.server_args = server_args
        self.model_config = model_config
        self._model_preprocessor = model_preprocessor
        self.encoder_media_processor_config = encoder_media_processor_config
        self.model_type = getattr(
            model_config.hf_config, "model_type", "unknown"
        ).lower()

        self.device = get_device().device

        use_image_processor_gpu = envs.SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU.get()
        self.use_image_processor_gpu = (
            use_image_processor_gpu
            and resolve_image_processor_backend(get_mm()) != "pil"
        )

        self._load_mm_processor(server_args)
        self._supported_modalities = frozenset(
            modality
            for modality, processor in (
                (Modality.IMAGE, self.image_processor),
                (Modality.VIDEO, self.video_processor),
                (Modality.AUDIO, self.audio_processor),
            )
            if processor is not None or self._model_preprocessor is not None
        )
        self._build_vision_config(get_mm().mm_process_config)
        self.model_audio_sr = self._resolve_audio_sr()
        logger.info(f"Resolved model audio sample rate: {self.model_audio_sr} Hz")

        self.preproc_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=envs.SGLANG_ENCODER_PREPROC_WORKERS.get()
        )
        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=envs.SGLANG_ENCODER_MM_LOAD_WORKERS.get()
        )

    # ------------------------------------------------------------------
    # HF Processor Loading
    # ------------------------------------------------------------------

    def _load_mm_processor(self, server_args: ServerArgs):
        from transformers import AutoImageProcessor, AutoVideoProcessor

        image_processor_backend = resolve_image_processor_backend(get_mm())
        image_processor_kwargs = (
            {}
            if image_processor_backend == "auto"
            else {"backend": image_processor_backend}
        )
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                get_serving().tokenizer_path or get_model().model_path,
                trust_remote_code=get_model().trust_remote_code,
                revision=server_args.revision,
                **image_processor_kwargs,
            )
        except Exception as e:
            logger.warning(f"Failed to load image processor: {e}")
            self.image_processor = None

        try:
            self.video_processor = AutoVideoProcessor.from_pretrained(
                get_serving().tokenizer_path or get_model().model_path,
                trust_remote_code=get_model().trust_remote_code,
                revision=server_args.revision,
            )
        except Exception as e:
            logger.warning(f"Failed to load video processor: {e}")
            self.video_processor = None

        try:
            _audio_proc = AutoProcessor.from_pretrained(
                get_serving().tokenizer_path or get_model().model_path,
                trust_remote_code=get_model().trust_remote_code,
                revision=server_args.revision,
            )
            if not hasattr(_audio_proc, "feature_extractor"):
                logger.warning(
                    "Loaded AutoProcessor has no feature_extractor attribute, "
                    "audio processing will be unavailable."
                )
                self.audio_processor = None
            else:
                self.audio_processor = _audio_proc
        except Exception as e:
            logger.warning(f"Failed to load audio processor: {e}")
            self.audio_processor = None

    # ------------------------------------------------------------------
    # Config Validation
    # ------------------------------------------------------------------

    def _build_vision_config(self, mm_process_config):
        self.vision_config = (
            mm_process_config.get("vision_config", {})
            if mm_process_config is not None
            else {}
        )
        for modality_str in ["image", "video", "audio"]:
            if not self.vision_config.get(modality_str, None):
                self.vision_config[modality_str] = {}
            if self.use_image_processor_gpu:
                self.vision_config[modality_str]["device"] = self.device

            if modality_str == "video":
                # GLM reads its own defaults from the HF video processor
                # (max_frames=2048); applying the Qwen values here would clobber them.
                if "glm" not in self.model_type:
                    video_defaults = {"fps": 2.0, "max_frames": 768, "min_frames": 4}
                    for k, v in video_defaults.items():
                        self.vision_config["video"].setdefault(k, v)

            if modality_str == "audio":
                if "return_attention_mask" not in self.vision_config["audio"]:
                    self.vision_config["audio"]["return_attention_mask"] = True
                if "padding" not in self.vision_config["audio"]:
                    if self.model_type == "qwen2_audio":
                        self.vision_config["audio"]["padding"] = "max_length"
                    else:
                        self.vision_config["audio"]["padding"] = True
                if "truncation" not in self.vision_config["audio"]:
                    if (
                        hasattr(self, "audio_processor")
                        and self.audio_processor is not None
                    ):
                        if self.audio_processor.__class__.__name__ in {
                            "Gemma3nProcessor",
                            "GlmAsrProcessor",
                            "Qwen2AudioProcessor",
                            "Qwen3OmniMoeProcessor",
                        }:
                            self.vision_config["audio"]["truncation"] = False

    def _resolve_audio_sr(self) -> int:
        def _read(obj, attr):
            if obj is None:
                return None
            if isinstance(obj, dict):
                return obj.get(attr)
            return getattr(obj, attr, None)

        audio_cfg = self.vision_config.get("audio", {})
        sr = audio_cfg.get("audio_sampling_rate")
        if sr:
            return int(sr)

        hf_cfg = self.model_config.hf_config
        thinker_cfg = _read(hf_cfg, "thinker_config")
        pc = _read(thinker_cfg, "processor_config") or _read(hf_cfg, "processor_config")
        sr = _read(pc, "audio_sampling_rate")
        if sr:
            return int(sr)
        ac = _read(thinker_cfg, "audio_config") or _read(hf_cfg, "audio_config")
        for attr in ("sampling_rate", "sample_rate"):
            sr = _read(ac, attr)
            if sr:
                return int(sr)

        sr = audio_cfg.get("sampling_rate")
        if sr:
            return int(sr)
        logger.warning(
            "No audio sampling rate found in mm_config or hf_config; "
            "falling back to 16000 Hz. If the model expects a different SR "
            "(e.g. MiMo-V2 defaults to 24000), audio will be warped."
        )
        return 16000

    # ------------------------------------------------------------------
    # Media I/O
    # ------------------------------------------------------------------

    def _load_single_item(
        self,
        data,
        modality: Modality,
        frame_count_limit=None,
        discard_alpha_channel=True,
    ):
        from sglang.srt.disaggregation.encoder.server import BadRequestError, MMError

        media_metadata = {}
        content_hash = None
        if isinstance(data, dict):
            if "url" not in data:
                return data
            media_metadata = {key: value for key, value in data.items() if key != "url"}
            content_hash = parse_content_hash(data.get("content_hash"))
            data = data["url"]
        try:
            if modality == Modality.IMAGE:
                if content_hash is not None:
                    snapshot = snapshot_media(data)
                    if snapshot.content_digest != content_hash:
                        raise BadRequestError(
                            "Encoder media content hash mismatch: "
                            f"expected {content_hash}, got {snapshot.content_digest}"
                        )
                    data = snapshot.data
                gpu_image_decode = (
                    self.encoder_media_processor_config.image_decode_mode
                    if self.use_image_processor_gpu
                    else False
                )
                img, _ = load_image(data, gpu_image_decode)
                if (
                    discard_alpha_channel
                    and not isinstance(img, torch.Tensor)
                    and img.mode != "RGB"
                ):
                    img = img.convert("RGB")
                if (
                    media_metadata
                    and self.encoder_media_processor_config.preserve_media_metadata
                ):
                    return {
                        "type": "image",
                        "image": img,
                        **media_metadata,
                    }
                return img
            elif modality == Modality.VIDEO:
                vid = load_video(data, frame_count_limit)
                if (
                    media_metadata
                    and self.encoder_media_processor_config.preserve_media_metadata
                ):
                    return {
                        "type": "video",
                        "video": vid,
                        **media_metadata,
                    }
                return vid
            elif modality == Modality.AUDIO:
                return load_audio(data, self.model_audio_sr)

        except MMError:
            raise
        except CLIENT_MEDIA_EXCEPTIONS as e:
            # Not ValueError: the DP envelope classifies by `.code`, which only
            # MMError carries.
            raise BadRequestError(f"Error while loading data {data}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Error while loading data {data}: {e}")

    def _submit_data_loading_tasks(self, items, modalities):
        futures = []
        task_info = []

        for data, modality in zip(items, modalities):
            if modality is not None:
                futures.append(
                    self.io_executor.submit(
                        self._load_single_item,
                        data,
                        modality,
                    )
                )
                task_info.append((modality, data))
        return futures, task_info

    async def _flatten_and_load_data_by_modality(self, mm_items, modality):
        if not isinstance(mm_items, (list, tuple)):
            futures, _ = self._submit_data_loading_tasks([mm_items], [modality])
            return await asyncio.wrap_future(futures[0])

        if len(mm_items) > 0 and isinstance(mm_items[0], (list, tuple)):
            flat_data = []
            flat_indices = []
            for group_idx, item_group in enumerate(mm_items):
                for item in item_group:
                    flat_data.append(item)
                    flat_indices.append(group_idx)

            futures, _ = self._submit_data_loading_tasks(
                flat_data, [modality] * len(flat_data)
            )

            async_futures = [asyncio.wrap_future(f) for f in futures]
            results = await asyncio.gather(*async_futures)

            nested_results = [[] for _ in range(len(mm_items))]
            for idx, result in zip(flat_indices, results):
                nested_results[idx].append(result)

            return nested_results

        else:
            futures, _ = self._submit_data_loading_tasks(
                mm_items, [modality] * len(mm_items)
            )
            async_futures = [asyncio.wrap_future(f) for f in futures]
            return await asyncio.gather(*async_futures)

    async def _flatten_and_load_images(self, mm_items):
        return await self._flatten_and_load_data_by_modality(mm_items, Modality.IMAGE)

    @staticmethod
    def _close_video_decoders(video_items) -> None:
        for video in video_items or []:
            close = getattr(video, "close", None)
            if callable(close):
                close()

    async def _dp_sharded_decode_single_video(
        self,
        vr,
        video_config,
        *,
        tp_rank: int,
        tp_size: int,
        video_processor_kwargs: dict,
        precomputed_indices: Optional[List[int]] = None,
    ):
        video_config = video_config or {}
        video_fps = vr.avg_fps
        duration = len(vr) / video_fps if video_fps else 0
        global_indices = precomputed_indices or glm_sample_frame_indices(
            len(vr),
            video_fps,
            duration,
            target_fps=video_config.get("fps"),
            max_frame_count=video_config.get("max_frames"),
        )
        n_units = len(global_indices) // 2
        base, remainder = divmod(n_units, tp_size)
        gpu_sample_counts = [
            base + (1 if rank < remainder else 0) for rank in range(tp_size)
        ]
        start = sum(gpu_sample_counts[:tp_rank])
        count = gpu_sample_counts[tp_rank]
        local_indices = global_indices[2 * start : 2 * (start + count)]

        local_error = None
        frames = None
        try:
            frames = await asyncio.get_running_loop().run_in_executor(
                self.io_executor,
                glm_decode_frames_at,
                vr,
                local_indices,
                video_config,
            )
        except Exception as exc:
            local_error = exc

        # All ranks must either enter the later ViT all-gather or fail before
        # it.  A rank-local decoder error must therefore be agreed globally.
        ok = torch.tensor([0 if local_error else 1], dtype=torch.int32)
        if tp_size > 1:
            torch.distributed.all_reduce(
                ok,
                op=torch.distributed.ReduceOp.MIN,
                group=get_attn_tp_group().cpu_group,
            )
        if not int(ok.item()):
            if local_error is not None:
                raise local_error
            from sglang.srt.disaggregation.encoder.server import MMError

            raise MMError(
                "peer encoder rank failed during sharded video decode",
                code=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        if frames is None:
            height, width = vr.frame_shape
            frames = np.zeros((0, height, width, 3), dtype=np.uint8)

        video_processor_kwargs["do_sample_frames"] = False
        video_processor_kwargs["return_metadata"] = True
        # Preserve the same per-frame spatial budget as the unsharded request.
        if global_indices and local_indices:
            budget = video_config.get("max_image_tokens")
            if budget is None:
                budget = getattr(self.video_processor, "max_image_tokens", None)
            if budget is not None:
                video_processor_kwargs["max_image_tokens"] = max(
                    1, int(int(budget) * len(local_indices) / len(global_indices))
                )
        video_processor_kwargs["_dp_meta"] = {
            "global_indices": list(global_indices),
            "fps": video_fps,
            "n_units": n_units,
            "gpu_sample_counts": gpu_sample_counts,
        }
        return [frames], video_processor_kwargs

    async def _flatten_and_load_videos(self, mm_items):
        if not isinstance(mm_items, (list, tuple)):
            mm_items = [mm_items]

        video_configs = [{} for _ in mm_items]
        if "glm" in self.model_type:
            mm_items, video_configs = split_glm_video_items(mm_items)
            defaults = glm_processor_video_config(self.video_processor)
            defaults.update(self.vision_config.get("video", {}))
            video_configs = [
                {**defaults, **dict(config or {})} for config in video_configs
            ]

        futures, _ = self._submit_data_loading_tasks(
            mm_items, [Modality.VIDEO] * len(mm_items)
        )
        async_futures = [asyncio.wrap_future(f) for f in futures]
        video_items = await asyncio.gather(*async_futures)

        video_processor_kwargs = {}
        if "qwen" in self.model_type:
            video_processed = [
                await preprocess_video(
                    video, video_config=self.vision_config.get("video", {})
                )
                for video in video_items
            ]
            videos, video_metadata = map(list, zip(*video_processed))
            video_processor_kwargs["do_sample_frames"] = False
            if video_metadata:
                video_processor_kwargs["video_metadata"] = video_metadata
            return videos, video_processor_kwargs

        if "glm" in self.model_type:
            budget_kwargs = glm_budget_kwargs(
                self.video_processor,
                user_max_image_tokens=glm_max_image_tokens_from_configs(video_configs),
                count=len(video_items),
                split=True,
            )
            if budget_kwargs is not None:
                video_processor_kwargs.update(budget_kwargs)
                video_configs = [
                    _glm_effective_presize_budget(
                        config, budget_kwargs.get("max_image_tokens")
                    )
                    for config in video_configs
                ]

            framed = any(isinstance(video, list) for video in video_items)
            if framed:
                processed = await asyncio.gather(
                    *[
                        asyncio.get_running_loop().run_in_executor(
                            self.io_executor, preprocess_video_frames_sync, video
                        )
                        for video in video_items
                    ]
                )
            else:
                tp_size = get_attn_tensor_model_parallel_world_size()
                sampled = None
                if len(video_items) == 1:
                    vr = video_items[0]
                    config = video_configs[0]
                    sampled = glm_sample_frame_indices(
                        len(vr),
                        vr.avg_fps,
                        len(vr) / vr.avg_fps if vr.avg_fps else 0,
                        target_fps=config.get("fps"),
                        max_frame_count=config.get("max_frames"),
                    )
                if (
                    self.server_args.mm_enable_dp_encoder
                    and tp_size > 1
                    and sampled is not None
                    and len(sampled) >= max(32, tp_size * 2)
                ):
                    result = await self._dp_sharded_decode_single_video(
                        video_items[0],
                        video_configs[0],
                        tp_rank=get_attn_tensor_model_parallel_rank(),
                        tp_size=tp_size,
                        video_processor_kwargs=video_processor_kwargs,
                        precomputed_indices=sampled,
                    )
                    self._close_video_decoders(video_items)
                    return result

                processed = await asyncio.gather(
                    *[
                        asyncio.get_running_loop().run_in_executor(
                            self.io_executor,
                            glm_sample_and_decode_sync,
                            video,
                            video_configs[index],
                        )
                        for index, video in enumerate(video_items)
                    ]
                )
            videos, video_metadata = map(list, zip(*processed))
            video_processor_kwargs["do_sample_frames"] = False
            video_processor_kwargs["return_metadata"] = True
            if video_metadata:
                video_processor_kwargs["video_metadata"] = video_metadata
            self._close_video_decoders(video_items)
            return videos, video_processor_kwargs

        self._close_video_decoders(video_items)
        raise NotImplementedError(
            f"Video processing is not supported for {self.model_type} model."
        )

    async def _flatten_and_load_audios(self, mm_items):
        return await self._flatten_and_load_data_by_modality(mm_items, Modality.AUDIO)

    # ------------------------------------------------------------------
    # HF Processor Calls
    # ------------------------------------------------------------------

    async def process_mm_items(
        self, mm_items, modality: Modality
    ) -> EncoderPreprocessResult:
        """Process multimodal items through the HF processor pipeline.

        Returns the ``mm_inputs`` dict produced by the HF image/video/audio
        processor, its normalized grid metadata, and one output token count per
        grid entry. Does not look up ``get_feature_fn``; that stays in
        :class:`MMEncoder`.
        """
        if modality == Modality.IMAGE:
            mm_inputs = await self._process_image_items(
                mm_items, self._model_preprocessor
            )
        elif modality == Modality.VIDEO:
            mm_inputs = await self._process_video_items(
                mm_items, self._model_preprocessor
            )
        elif modality == Modality.AUDIO:
            mm_inputs = await self._process_audio_items(
                mm_items, self._model_preprocessor
            )
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        grid_thw = self._get_mm_grid_dim(mm_inputs, modality)
        token_counts = [self.get_num_tokens(grid, modality) for grid in grid_thw]
        return EncoderPreprocessResult(
            mm_inputs=mm_inputs,
            grid_thw=grid_thw,
            token_counts=token_counts,
        )

    def supports_modality(self, modality: Modality) -> bool:
        return modality in self._supported_modalities

    async def process_batch_mm_items(
        self, requests: List[dict], modality: Modality
    ) -> tuple[EncoderPreprocessResult, List[int]]:
        """Flatten requests, run the processor once, and return batch layout."""
        flat_items, items_per_req = self._flatten_batch_requests(requests, modality)
        result = await self.process_mm_items(flat_items, modality)
        return result, items_per_req

    def _flatten_batch_requests(
        self, requests: List[dict], modality: Modality
    ) -> tuple[List, List[int]]:
        # items_per_req counts grid entries (post-expansion) so per-request
        # slicing of grid_dim/final_slices stays aligned for processors that
        # expand one leaf into multiple grids (e.g. Kimi-VL/K2.5/K3 dict-of-images).
        flat_items = []
        items_per_req = []
        for req in requests:
            leaves = self._flatten_nested_items(req["mm_items"])
            flat_items.extend(leaves)
            items_per_req.append(sum(self._grid_count_per_leaf(leaves, modality)))
        return flat_items, items_per_req

    async def _process_image_items(self, mm_items, model_preprocessor):
        if not (self.image_processor or model_preprocessor):
            raise ValueError("No image processor available")
        images = await self._flatten_and_load_images(mm_items)
        if self.model_type in ["kimi_k25", "kimi_k3", "kimi_vl"]:
            images = self._normalize_kimi_encoder_images(images)
        original_image_sizes = [_get_original_image_size(item) for item in images]
        if model_preprocessor:
            processor_output = invoke_encoder_preprocessor(
                model_preprocessor,
                images,
                Modality.IMAGE,
                self.vision_config,
                image_processor=self.image_processor,
                use_gpu_preprocessing=self.use_image_processor_gpu,
            )
            if (
                isinstance(processor_output, EncoderPreprocessOutput)
                and processor_output.materialize_local_items is not None
            ):
                parallel = get_parallel()
                await asyncio.get_running_loop().run_in_executor(
                    self.preproc_executor,
                    processor_output.materialize_for_rank,
                    parallel.attn_tp_rank,
                    parallel.attn_tp_size,
                )
            return processor_output
        image_config = self.vision_config.get("image", {})
        processor_input = await asyncio.get_running_loop().run_in_executor(
            self.preproc_executor,
            functools.partial(self.image_processor, images=images, **image_config),
        )
        if self.model_type == "kimi_k3":
            processor_input["original_image_sizes"] = original_image_sizes
        return processor_input

    async def _process_video_items(self, mm_items, model_preprocessor):
        if model_preprocessor:
            return model_preprocessor(mm_items, Modality.VIDEO, self.vision_config)
        if not self.video_processor:
            raise ValueError("No video processor available")

        videos, video_processor_kwargs = await self._flatten_and_load_videos(mm_items)
        processor_input = await asyncio.get_running_loop().run_in_executor(
            self.preproc_executor,
            functools.partial(
                self.video_processor, videos=videos, **video_processor_kwargs
            ),
        )

        if (
            self.model_type
            in [
                "qwen3_vl",
                "qwen3_vl_moe",
                "qwen3_5",
                "qwen3_5_moe",
                "intern_s2_preview",
            ]
            and video_processor_kwargs.get("video_metadata", None) is not None
        ):
            video_metadata = video_processor_kwargs["video_metadata"]
            try:
                merge_size = (
                    self.model_config.hf_config.vision_config.spatial_merge_size
                )
            except (AttributeError, KeyError):
                merge_size = 2
            video_timestamps = []
            for metadata in video_metadata:
                video_fps = metadata.get("fps", None) or 24
                frames_indices = metadata.get("frames_indices", None)
                timestamps = self._calculate_timestamps(
                    frames_indices, video_fps, merge_size
                )
                video_timestamps.append(timestamps)
            processor_input["video_timestamps"] = video_timestamps
        elif (
            self.model_type in ["qwen2_5_vl", "qwen2_5_omni", "qwen3_omni_moe"]
            and processor_input.get("video_grid_thw", None) is not None
        ):
            video_grid_thw = processor_input["video_grid_thw"]
            try:
                temporal_patch_size = self.video_processor.temporal_patch_size
            except AttributeError:
                temporal_patch_size = 2
            fps_list = [
                self.vision_config.get("video", {}).get("fps", None) or 2
            ] * len(video_grid_thw)
            second_per_grid_ts = [(temporal_patch_size / fps) for fps in fps_list]
            second_per_grid_ts_tensor = torch.tensor(
                second_per_grid_ts, dtype=torch.float32
            )
            processor_input["second_per_grid_ts"] = second_per_grid_ts_tensor

        return processor_input

    async def _process_audio_items(self, mm_items, model_preprocessor):
        audios = await self._flatten_and_load_audios(mm_items)

        if model_preprocessor:
            return model_preprocessor(audios, Modality.AUDIO, self.vision_config)

        if not self.audio_processor:
            raise ValueError("No audio processor available")

        audio_config = self.vision_config.get("audio", {})
        processor_input = await asyncio.get_running_loop().run_in_executor(
            self.preproc_executor,
            functools.partial(
                self.audio_processor.feature_extractor, audios, **audio_config
            ),
        )
        processor_input["feature_attention_mask"] = processor_input.pop(
            "attention_mask"
        )
        input_lengths = torch.tensor(
            processor_input["feature_attention_mask"].sum(-1), dtype=torch.long
        )
        processor_input["audio_feature_lens_raw"] = input_lengths
        output_lengths = self._get_feat_extract_output_lengths(input_lengths)
        processor_input["audio_feature_lens"] = output_lengths
        return processor_input

    # ------------------------------------------------------------------
    # Audio Feature Length Computation
    # ------------------------------------------------------------------

    def _get_feat_extract_output_lengths(self, feature_lens):
        if self.model_type in ["qwen2_audio", "qwen2_5_omni"]:
            input_length = (feature_lens - 1) // 2 + 1
            return (input_length - 2) // 2 + 1
        elif self.model_type in ["qwen3_asr", "qwen3_omni_moe"]:
            input_lengths_leave = feature_lens % 100
            feat_lengths = (input_lengths_leave - 1) // 2 + 1
            output_lengths = (
                ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (feature_lens // 100) * 13
            )
            return output_lengths
        elif self.model_type == "mimo_v2":
            return feature_lens
        else:
            logger.warning(
                f"Fallback to original HF audio sample logic for {self.model_type}"
            )
            input_length = (feature_lens - 1) // 2 + 1
            return (input_length - 2) // 2 + 1

    def _get_mm_grid_dim(self, mm_inputs: dict, modality: Modality):
        # Kimi K2.5/K3 vision processors only emit `grid_thws`; prefer it over generic keys
        # so we never pick a mis-typed or stale `image_grid_hws` field from kwargs.
        attrs = _mm_grid_attrs[modality]
        model_type = (self.model_type or "").lower()
        if modality == Modality.IMAGE:
            # Kimi K2.5/K3 emit grid_thws, while Kimi-VL emits image_grid_hws.
            # Other model types keep the generic attr order above.
            if model_type in ("kimi_k25", "kimi_k3"):
                attrs = ("grid_thws", "image_grid_thw", "image_grid_hws")
            elif model_type == "kimi_vl":
                attrs = ("image_grid_hws", "image_grid_thw", "grid_thws")

        for attr in attrs:
            if attr in mm_inputs and mm_inputs[attr] is not None:
                return _convert(mm_inputs[attr])
        raise ValueError(
            f"Grid dim ({_mm_grid_attrs[modality]}) not found in {mm_inputs}"
        )

    def get_num_patches(
        self, grid: Union[torch.Tensor, List[int]], modality: Modality
    ) -> int:
        """Calculate number of raw patches (before merge/sampling). Used for pixel_values slicing."""
        if modality == Modality.AUDIO:
            return int(grid.item())
        if self.model_type == "kimi_vl" and modality == Modality.IMAGE:
            h, w = self._kimi_hw_from_patch_grid(grid)
            return h * w
        return int(grid[0] * grid[1] * grid[2])

    @staticmethod
    def _kimi_hw_from_patch_grid(
        grid: Union[torch.Tensor, np.ndarray, List[int], Tuple[int, ...]],
    ) -> Tuple[int, int]:
        """Extract (height, width) from Kimi 2D or 3D patch-grid metadata."""
        if isinstance(grid, torch.Tensor):
            values = grid.flatten().tolist()
        elif isinstance(grid, np.ndarray):
            values = grid.reshape(-1).tolist()
        else:
            values = np.asarray(grid).reshape(-1).tolist()

        if len(values) not in (2, 3):
            raise ValueError(
                f"Invalid Kimi image grid metadata: {values}; "
                "expected [h, w] or [t, h, w]"
            )
        return int(values[-2]), int(values[-1])

    def _kimi_tokens_from_patch_grid(self, grid: Union[torch.Tensor, List[int]]) -> int:
        """Calculate Kimi image tokens from either 2D or 3D patch metadata."""
        h, w = self._kimi_hw_from_patch_grid(grid)
        merge_h, merge_w = self.model_config.hf_config.vision_config.merge_kernel_size
        return (h * w) // (merge_h * merge_w)

    def get_num_tokens(
        self, grid: Union[torch.Tensor, List[int]], modality: Modality
    ) -> int:
        """Compatibility helper for callers that still provide patch grids."""
        if modality == Modality.AUDIO:
            input_length = self.get_num_patches(grid, modality)
            return self._get_feat_extract_output_lengths(input_length)
        else:
            if (
                self.model_type in ["kimi_k25", "kimi_k3", "kimi_vl"]
                and modality == Modality.IMAGE
            ):
                return self._kimi_tokens_from_patch_grid(grid)
            merge_size = getattr(self.image_processor, "merge_size", 2)
            return self.get_num_patches(grid, modality) // (merge_size**2)

    # ------------------------------------------------------------------
    # Video Timestamp Computation
    # ------------------------------------------------------------------

    def _calculate_timestamps(self, indices, video_fps: float, merge_size: int = 2):
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend(
                indices[-1] for _ in range(merge_size - len(indices) % merge_size)
            )
        timestamps = [idx / video_fps for idx in indices]
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2
            for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps

    # ------------------------------------------------------------------
    # Kimi Normalization
    # ------------------------------------------------------------------

    def _normalize_kimi_encoder_images(self, images):
        """Normalize Kimi image inputs for the image processor call."""
        from PIL import Image as PILImage

        def wrap_one(img):
            if isinstance(img, dict) and img.get("type") in ("image", "video_chunk"):
                return [img]
            if isinstance(img, PILImage.Image):
                return [{"type": "image", "image": img}]
            return [img]

        if not images:
            return images

        # Disagg may supply nested lists from grouped routing.
        images = self._flatten_nested_items(images)

        if self.model_type == "kimi_vl":
            normalized = []
            for img in images:
                if (
                    isinstance(img, dict)
                    and img.get("type") == "image"
                    and "image" in img
                ):
                    inner = img["image"]
                    if isinstance(inner, (list, tuple)):
                        normalized.extend(self._flatten_nested_items(inner))
                    else:
                        normalized.append(inner)
                else:
                    normalized.append(img)
            return normalized

        # Kimi-K2.5/K3 vision processors expect media dicts.
        normalized = []
        for img in images:
            wrapped = wrap_one(img)
            for media in wrapped:
                if (
                    isinstance(media, dict)
                    and media.get("type") == "image"
                    and isinstance(media.get("image"), (list, tuple))
                ):
                    for inner in self._flatten_nested_items(media["image"]):
                        normalized.append({**media, "image": inner})
                else:
                    normalized.append(media)

        return normalized

    # ------------------------------------------------------------------
    # Utility Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_nested_items(items):
        if not isinstance(items, (list, tuple)):
            return [items]

        flat = []
        for item in items:
            if isinstance(item, (list, tuple)):
                flat.extend(EncoderPreprocessor._flatten_nested_items(item))
            else:
                flat.append(item)
        return flat

    def _grid_count_per_leaf(self, leaves: List, modality: Modality) -> List[int]:
        """Number of grid entries each leaf produces under the model's processor.

        Most processors map 1 leaf -> 1 grid. Kimi-VL/K2.5/K3 image processors expand
        a leaf shaped {"type": "image", "image": [pil1, pil2, ...]} into N grids.
        """
        if (
            self.model_type not in ("kimi_k25", "kimi_k3", "kimi_vl")
            or modality != Modality.IMAGE
        ):
            return [1] * len(leaves)

        def count(leaf):
            if (
                isinstance(leaf, dict)
                and leaf.get("type") == "image"
                and isinstance(leaf.get("image"), (list, tuple))
            ):
                return len(self._flatten_nested_items(leaf["image"]))
            return 1

        return [count(leaf) for leaf in leaves]
