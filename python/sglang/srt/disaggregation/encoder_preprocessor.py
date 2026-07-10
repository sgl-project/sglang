"""CPU-bound multimodal preprocessing for the EPD encoder.

This module is designed to be replaceable by a Rust implementation.
It handles all CPU-bound work: media I/O (image/video/audio loading),
HF processor calls, config validation, and related helper computations.
GPU tensor operations remain in :mod:`encode_server.MMEncoder`.
"""

import asyncio
import concurrent.futures
import functools
import logging
import os
from typing import Callable, List, Optional

import torch
from transformers import AutoProcessor

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.qwen_vl import preprocess_video
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import load_audio, load_image, load_video

logger = logging.getLogger(__name__)


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
        model_preprocessor: Optional[Callable] = None,
    ):
        self.server_args = server_args
        self.model_config = model_config
        self._model_preprocessor = model_preprocessor
        self.model_type = getattr(
            model_config.hf_config, "model_type", "unknown"
        ).lower()

        self.device = server_args.device

        use_image_processor_gpu = envs.SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU.get()
        self.use_image_processor_gpu = (
            use_image_processor_gpu and not server_args.disable_fast_image_processor
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
        self._build_vision_config(server_args.mm_process_config)
        self.model_audio_sr = self._resolve_audio_sr()
        logger.info(f"Resolved model audio sample rate: {self.model_audio_sr} Hz")

        self.preproc_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=envs.SGLANG_ENCODER_PREPROC_WORKERS.get()
        )
        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("SGLANG_ENCODER_MM_LOAD_WORKERS", 4))
        )

    # ------------------------------------------------------------------
    # HF Processor Loading
    # ------------------------------------------------------------------

    def _load_mm_processor(self, server_args: ServerArgs):
        from transformers import AutoImageProcessor, AutoVideoProcessor

        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                server_args.tokenizer_path or server_args.model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=not server_args.disable_fast_image_processor,
            )
        except Exception as e:
            logger.warning(f"Failed to load image processor: {e}")
            self.image_processor = None

        try:
            self.video_processor = AutoVideoProcessor.from_pretrained(
                server_args.tokenizer_path or server_args.model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=not server_args.disable_fast_image_processor,
            )
        except Exception as e:
            logger.warning(f"Failed to load video processor: {e}")
            self.video_processor = None

        try:
            _audio_proc = AutoProcessor.from_pretrained(
                server_args.tokenizer_path or server_args.model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=not server_args.disable_fast_image_processor,
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
        if isinstance(data, dict):
            return data
        try:
            if modality == Modality.IMAGE:
                img, _ = load_image(data, False)
                if (
                    discard_alpha_channel
                    and not isinstance(img, torch.Tensor)
                    and img.mode != "RGB"
                ):
                    img = img.convert("RGB")
                return img
            elif modality == Modality.VIDEO:
                return load_video(data, frame_count_limit)
            elif modality == Modality.AUDIO:
                return load_audio(data, self.model_audio_sr)

        except Exception as e:
            raise RuntimeError(f"Error while loading data {data}: {e}")

    def submit_data_loading_tasks(self, items, modalities):
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
            futures, _ = self.submit_data_loading_tasks([mm_items], [modality])
            return await asyncio.wrap_future(futures[0])

        if len(mm_items) > 0 and isinstance(mm_items[0], (list, tuple)):
            flat_data = []
            flat_indices = []
            for group_idx, item_group in enumerate(mm_items):
                for item in item_group:
                    flat_data.append(item)
                    flat_indices.append(group_idx)

            futures, _ = self.submit_data_loading_tasks(
                flat_data, [modality] * len(flat_data)
            )

            async_futures = [asyncio.wrap_future(f) for f in futures]
            results = await asyncio.gather(*async_futures)

            nested_results = [[] for _ in range(len(mm_items))]
            for idx, result in zip(flat_indices, results):
                nested_results[idx].append(result)

            return nested_results

        else:
            futures, _ = self.submit_data_loading_tasks(
                mm_items, [modality] * len(mm_items)
            )
            async_futures = [asyncio.wrap_future(f) for f in futures]
            return await asyncio.gather(*async_futures)

    async def _flatten_and_load_images(self, mm_items):
        return await self._flatten_and_load_data_by_modality(mm_items, Modality.IMAGE)

    async def _flatten_and_load_videos(self, mm_items):
        if not isinstance(mm_items, (list, tuple)):
            mm_items = [mm_items]

        futures, _ = self.submit_data_loading_tasks(
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
        else:
            raise NotImplementedError(
                f"Video processing is not supported for {self.model_type} model."
            )

    async def _flatten_and_load_audios(self, mm_items):
        return await self._flatten_and_load_data_by_modality(mm_items, Modality.AUDIO)

    # ------------------------------------------------------------------
    # HF Processor Calls
    # ------------------------------------------------------------------

    async def process_mm_items(self, mm_items, modality: Modality) -> dict:
        """Process multimodal items through the HF processor pipeline.

        Returns the ``mm_inputs`` dict produced by the HF image/video/audio
        processor.  Does NOT look up ``get_feature_fn`` — that stays in
        :class:`MMEncoder`.
        """
        if modality == Modality.IMAGE:
            return await self._process_image_items(mm_items, self._model_preprocessor)
        elif modality == Modality.VIDEO:
            return await self._process_video_items(mm_items, self._model_preprocessor)
        elif modality == Modality.AUDIO:
            return await self._process_audio_items(mm_items, self._model_preprocessor)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    def supports_modality(self, modality: Modality) -> bool:
        return modality in self._supported_modalities

    async def _process_image_items(self, mm_items, model_preprocessor):
        if not (self.image_processor or model_preprocessor):
            raise ValueError("No image processor available")
        images = await self._flatten_and_load_images(mm_items)
        if model_preprocessor:
            return model_preprocessor(images, Modality.IMAGE, self.vision_config)
        image_config = self.vision_config.get("image", {})
        if self.model_type in ["kimi_k25", "kimi_vl"]:
            images = self._normalize_kimi_encoder_images(images)
        return await asyncio.get_running_loop().run_in_executor(
            self.preproc_executor,
            functools.partial(self.image_processor, images=images, **image_config),
        )

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
        from PIL import Image as PILImage

        def wrap_one(img):
            if isinstance(img, dict) and img.get("type") in ("image", "video_chunk"):
                return [img]
            if isinstance(img, PILImage.Image):
                return [{"type": "image", "image": img}]
            return [img]

        if not images:
            return images

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

        Most processors map 1 leaf -> 1 grid. Kimi-VL/K25 image processors expand
        a leaf shaped {"type": "image", "image": [pil1, pil2, ...]} into N grids.
        """
        if self.model_type not in ("kimi_k25", "kimi_vl") or modality != Modality.IMAGE:
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
