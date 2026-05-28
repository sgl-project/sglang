import asyncio
import concurrent.futures
import contextlib
import ctypes
import logging
import multiprocessing as mp
import os
import pickle
import time
import traceback
from collections import defaultdict
from http import HTTPStatus
from typing import Dict, List, Optional, Set, Tuple, Union

import aiohttp
import numpy as np
import torch
import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse, Response
from transformers import AutoProcessor

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.encode_receiver import (
    EmbeddingData,
    video_meta_attrs_for,
)
from sglang.srt.distributed.parallel_state import (
    get_default_distributed_backend,
    get_mooncake_transfer_engine,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.managers.io_struct import ProfileReq, ProfileReqInput, ProfileReqType
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.model_loader import get_model
from sglang.srt.multimodal.processors.qwen_vl import preprocess_video
from sglang.srt.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import (
    configure_logger,
    load_audio,
    load_image,
    load_video,
    random_uuid,
)
from sglang.srt.utils.network import (
    NetworkAddress,
    config_socket,
    get_local_ip_auto,
    get_zmq_socket,
)

logger = logging.getLogger(__name__)

HEALTH_CHECK_TIMEOUT = 10

# Minimal 32x32 black PNG for health check dummy encode
MINIMUM_PNG_PICTURE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="

# Minimal WAV: 16kHz mono 16-bit PCM, 160 samples (0.01s) of silence
MINIMUM_WAV_SILENCE_BASE64 = "UklGRmQBAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YUABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="

rid_lock = asyncio.Lock()
rid_to_receive_endpoint: Dict[str, List[str]] = dict()
rid_to_receive_count: Dict[str, int] = dict()
rid_to_err_msg: Dict[str, str] = dict()
cond_dict_lock = asyncio.Lock()
rid_to_cond: Dict[str, asyncio.Condition] = {}

use_image_processor_gpu = envs.SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU.get()

ENCODER_MAX_BATCH_SIZE = envs.SGLANG_ENCODER_MAX_BATCH_SIZE.get()
# Watchdog: max time to wait for a batched /encode result. Bounds HTTP latency
# if the batch worker stalls (NCCL hang, dead worker proc, etc.).
ENCODER_REQ_TIMEOUT = envs.SGLANG_ENCODER_REQ_TIMEOUT.get()


class MMError(Exception):
    def __init__(self, message, code=HTTPStatus.INTERNAL_SERVER_ERROR):
        self.message = message
        self.code = code
        super().__init__(self.message)


class BadRequestError(MMError):
    def __init__(self, message):
        super().__init__(message, code=HTTPStatus.BAD_REQUEST)


class InternalError(MMError):
    def __init__(self, message):
        super().__init__(message, code=HTTPStatus.INTERNAL_SERVER_ERROR)


class TensorWrapper:
    """Wrapper to keep tensor alive while exposing buffer for zero-copy."""

    def __init__(self, tensor):
        # Ensure tensor is on CPU and contiguous
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Keep tensor reference
        self.tensor = tensor
        self.shape = list(tensor.shape)
        self.dtype = tensor.dtype

    def __buffer__(self):
        data_ptr = self.tensor.data_ptr()
        total_bytes = self.tensor.numel() * self.tensor.element_size()
        c_obj = (ctypes.c_char * total_bytes).from_address(data_ptr)
        c_obj._keep_alive_ref = self
        return memoryview(c_obj)


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


_mm_grid_attrs = {
    # Kimi K2.5 HF processor uses grid_thws (see base_processor.ATTR_NAME_TO_MODALITY).
    Modality.IMAGE: ["image_grid_thw", "image_grid_hws", "grid_thws"],
    Modality.VIDEO: ["video_grid_thw"],
    Modality.AUDIO: ["audio_feature_lens_raw"],
}

_mm_feature_attrs = {
    Modality.IMAGE: ["pixel_values"],
    Modality.VIDEO: ["pixel_values_videos"],
    Modality.AUDIO: ["input_features"],
}


def _get_mm_grid_dim(mm_inputs, modality, model_type: Optional[str] = None):
    # Kimi K2.5 vision processor only emits `grid_thws`; prefer it over generic keys
    # so we never pick a mis-typed or stale `image_grid_hws` field from kwargs.
    attrs = _mm_grid_attrs[modality]
    if (model_type or "").lower() in [
        "kimi_k25",
        "kimi_vl",
    ] and modality == Modality.IMAGE:
        attrs = ("grid_thws", "image_grid_thw", "image_grid_hws")
    for attr in attrs:
        if attr in mm_inputs and mm_inputs[attr] is not None:
            return mm_inputs[attr]
    raise ValueError(f"Grid dim ({_mm_grid_attrs[modality]}) not found in {mm_inputs}")


def _get_mm_feature(mm_inputs, modality):
    for attr in _mm_feature_attrs[modality]:
        if attr in mm_inputs:
            return mm_inputs[attr]
    raise ValueError(
        f"Feature attrs ({_mm_feature_attrs[modality]}) not found in {mm_inputs}"
    )


def _build_mm_aux_data(mm_inputs, model_type=None):
    # Video aux metadata, scoped to model_type's video-meta attrs.
    return {attr: mm_inputs.get(attr) for attr in video_meta_attrs_for(model_type)}


class MMEncoder:
    def __init__(
        self,
        server_args: ServerArgs,
        schedule_path=None,
        dist_init_method=None,
        rank: int = 0,
    ):
        logger.info(f"init MMEncoder {rank}/{server_args.tp_size}")
        self.server_args = server_args
        set_global_server_args_for_scheduler(server_args)
        self.rank = rank
        self.profiler = EncoderProfiler(rank)
        self._load_mm_processor(server_args)

        self.model_config = ModelConfig.from_server_args(
            server_args,
        )
        self.load_config = LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
            model_loader_extra_config=server_args.model_loader_extra_config,
            remote_instance_weight_loader_seed_instance_ip=server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=server_args.remote_instance_weight_loader_send_weights_group_ports,
        )
        self.model_type = getattr(
            self.model_config.hf_config, "model_type", "unknown"
        ).lower()

        self.device = server_args.device
        self.gpu_id = server_args.base_gpu_id + rank

        self.device_config = DeviceConfig(
            device=self.device,
            gpu_id=self.gpu_id,
        )

        torch.get_device_module(self.device).set_device(self.gpu_id)

        self.use_image_processor_gpu = (
            use_image_processor_gpu and not server_args.disable_fast_image_processor
        )
        self._build_vision_config(server_args.mm_process_config)
        self.model_audio_sr = self._resolve_audio_sr()
        logger.info(f"Resolved model audio sample rate: {self.model_audio_sr} Hz")

        init_distributed_environment(
            backend=get_default_distributed_backend(self.device),
            world_size=server_args.tp_size,
            rank=rank,
            distributed_init_method=dist_init_method,
            local_rank=rank,
        )
        initialize_model_parallel(tensor_model_parallel_size=server_args.tp_size)
        initialize_dp_attention(server_args, self.model_config)

        self.model = get_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=self.device_config,
        )

        self.context = zmq.asyncio.Context(2)
        self.sync_context = zmq.Context()  # Reuse sync context for thread pool
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

        embedding_cache_size = int(os.environ.get("SGLANG_VLM_CACHE_SIZE_MB", "4096"))
        self.mm_cache = MultiModalStaticCache(embedding_cache_size * 1024 * 1024)
        self.mm_cache_lock = asyncio.Lock()

        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("SGLANG_ENCODER_MM_LOAD_WORKERS", 4))
        )
        self.send_timeout = envs.SGLANG_ENCODER_SEND_TIMEOUT.get()

        if schedule_path is not None:
            self.schedule_socket = get_zmq_socket(
                self.context, zmq.PULL, schedule_path, True
            )
        self.background_tasks: Set[asyncio.Task] = set()

        if self.server_args.enable_mm_global_cache:
            from sglang.srt.mem_cache.storage.mooncake_store.embedding_cache_controller import (
                EmbeddingCacheController,
            )

            hidden_dims = self._infer_embedding_dims()
            self.mm_global_cache = EmbeddingCacheController(
                rank,
                server_args.tp_size,
                hidden_dims=hidden_dims,
                tp_group=get_tp_group().cpu_group,
                all_rank_get=False,
            )
        else:
            self.mm_global_cache = None

        if self.rank == 0:
            logger.info(
                f"Using transfer backend: {self.server_args.encoder_transfer_backend}"
            )

            if self.server_args.encoder_transfer_backend == "mooncake":
                self.local_ip = get_local_ip_auto()

                self.engine = get_mooncake_transfer_engine()
                if self.engine is None:
                    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                        init_mooncake_transfer_engine,
                    )

                    self.engine = init_mooncake_transfer_engine(
                        hostname=self.local_ip,
                        gpu_id=self.gpu_id,
                        ib_device=(
                            self.server_args.disaggregation_ib_device
                            or self.server_args.mooncake_ib_device
                        ),
                    )

            self.embedding_to_send = dict()

        logger.info(f"rank {rank} init finish ")

    def _infer_embedding_dims(self) -> dict:
        """Infer per-modality embedding dimensions from hf_config at init time."""
        default = self.model_config.hidden_size
        hf_cfg = self.model_config.hf_config
        thinker_cfg = getattr(hf_cfg, "thinker_config", None)
        dims = {
            Modality.IMAGE: default,
            Modality.VIDEO: default,
            Modality.AUDIO: default,
        }

        vision_cfg = getattr(thinker_cfg, "vision_config", None) or getattr(
            hf_cfg, "vision_config", None
        )
        if vision_cfg is not None:
            out_hs = getattr(vision_cfg, "out_hidden_size", None)
            if out_hs is not None:
                ds = getattr(vision_cfg, "deepstack_visual_indexes", None)
                vis_dim = (
                    out_hs * (1 + len(ds))
                    if isinstance(ds, (list, tuple)) and ds
                    else out_hs
                )
                dims[Modality.IMAGE] = vis_dim
                dims[Modality.VIDEO] = vis_dim

        audio_cfg = getattr(thinker_cfg, "audio_config", None) or getattr(
            hf_cfg, "audio_config", None
        )
        if audio_cfg is not None:
            for attr in ("output_dim", "d_model"):
                val = getattr(audio_cfg, attr, None)
                if val and int(val) > 0:
                    dims[Modality.AUDIO] = int(val)
                    break

        logger.info(f"Global cache embedding dims: {dims}")
        return dims

    def _resolve_audio_sr(self) -> int:
        # Must match MiMoProcessor.from_hf_config — on drift, mimo tags the
        # ndarray with its own audio_sampling_rate and skips resample, so the
        # waveform is interpreted at the wrong rate and warped.
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

    def _build_vision_config(self, mm_process_config):
        """
        Validate vision config, used for image/video/audio.
        If not provided, keep default values.
        """
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
                        # For Qwen2Audio, use padding="max_length"
                        # (same as https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_audio/processing_qwen2_audio.py#L93)
                        self.vision_config["audio"]["padding"] = "max_length"
                    else:
                        self.vision_config["audio"]["padding"] = True
                if "truncation" not in self.vision_config["audio"]:
                    # keep same logic as base_processor.py
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

    def _load_mm_processor(self, server_args: ServerArgs):
        """
        Load image/video/audio processor separately,
        avoid issues with AutoProcessor not recognizing certain models
        """
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
            # Note: AutoProcessor is used for audio processor
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

    def _load_single_item(
        self,
        data,
        modality: Modality,
        frame_count_limit=None,
        discard_alpha_channel=True,
    ):
        """
        Load a single multimodal data.
        If data is precomputed, returns directly.
        Static method that can be pickled for multiprocessing"""
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
                    # Needed only when `img` is a PIL image
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

    def _get_feat_extract_output_lengths(self, feature_lens):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        # qwen2_audio/qwen2.5_omni
        if self.model_type in ["qwen2_audio", "qwen2_5_omni"]:
            input_length = (feature_lens - 1) // 2 + 1
            return (input_length - 2) // 2 + 1
        # qwen3_asr / qwen3_omni_moe (same audio encoder architecture)
        elif self.model_type in ["qwen3_asr", "qwen3_omni_moe"]:
            input_lengths_leave = feature_lens % 100
            feat_lengths = (input_lengths_leave - 1) // 2 + 1
            output_lengths = (
                ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (feature_lens // 100) * 13
            )
            return output_lengths
        elif self.model_type == "mimo_v2":
            # MiMo-V2's preprocess_audio returns audio_token_len (already
            # post-encoder/avg-pooler/group-size). Stored in audio_feature_lens_raw,
            # so no further reduction here.
            return feature_lens
        else:
            # fallback to original HF audio sample logic for other models
            logger.warning(
                f"Fallback to original HF audio sample logic for {self.model_type}"
            )
            input_length = (feature_lens - 1) // 2 + 1
            return (input_length - 2) // 2 + 1

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
            # for qwen-series model, do sample frames before preprocess
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

    async def _flatten_and_load_data_by_modality(self, mm_items, modality):
        """
        Flatten mm_items structure, load multimodal data concurrently, and restore original structure.

        Returns:
            Same structure as load_mm_items would return, support for image/audio
        """
        # Handle single mm_item (not a list)
        if not isinstance(mm_items, (list, tuple)):
            futures, _ = self.submit_data_loading_tasks([mm_items], [modality])
            return await asyncio.wrap_future(futures[0])

        # Handle nested list (list of lists)
        if len(mm_items) > 0 and isinstance(mm_items[0], (list, tuple)):
            # Flatten nested structure
            flat_data = []
            flat_indices = []  # Track which group each item belongs to
            for group_idx, item_group in enumerate(mm_items):
                for item in item_group:
                    flat_data.append(item)
                    flat_indices.append(group_idx)

            # Submit all tasks concurrently
            futures, _ = self.submit_data_loading_tasks(
                flat_data, [modality] * len(flat_data)
            )

            # Wait for all tasks to complete asynchronously
            async_futures = [asyncio.wrap_future(f) for f in futures]
            results = await asyncio.gather(*async_futures)

            # Restore nested structure
            nested_results = [[] for _ in range(len(mm_items))]
            for idx, result in zip(flat_indices, results):
                nested_results[idx].append(result)

            return nested_results

        # Handle simple list
        else:
            futures, _ = self.submit_data_loading_tasks(
                mm_items, [modality] * len(mm_items)
            )
            # Wait for all tasks to complete asynchronously
            async_futures = [asyncio.wrap_future(f) for f in futures]
            return await asyncio.gather(*async_futures)

    def get_num_patches(
        self, grid: Union[torch.Tensor, List[int]], modality: Modality
    ) -> int:
        """Calculate number of raw patches (before merge/sampling). Used for pixel_values slicing."""
        if modality == Modality.AUDIO:
            return int(grid.item())
        else:
            return int(grid[0] * grid[1] * grid[2])

    def _kimi_tokens_from_patch_grid(self, grid: Union[torch.Tensor, List[int]]) -> int:
        """MoonViT + tpool: output len is (h//mh)*(w//mw); temporal dim is pooled (not t*h*w/merge^2)."""
        if isinstance(grid, torch.Tensor):
            flat = grid.flatten()
            _t, h, w = (int(x) for x in flat[:3].tolist())
        else:
            _t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
        merge_h, merge_w = self.model_config.hf_config.vision_config.merge_kernel_size
        return (h * w) // (merge_h * merge_w)

    def get_num_tokens(
        self, grid: Union[torch.Tensor, List[int]], modality: Modality
    ) -> int:
        """Calculate number of tokens (after 2x2 merge). Used for mm_embedding slicing."""
        if modality == Modality.AUDIO:
            input_length = self.get_num_patches(grid, modality)
            return self._get_feat_extract_output_lengths(input_length)
        else:
            if (
                self.model_type in ["kimi_k25", "kimi_vl"]
                and modality == Modality.IMAGE
            ):
                return self._kimi_tokens_from_patch_grid(grid)
            merge_size = getattr(self.image_processor, "merge_size", 2)
            return self.get_num_patches(grid, modality) // (merge_size**2)

    def slice_embedding(
        self, mm_embedding: torch.Tensor, grid_thw: List, modality: Modality
    ) -> List[torch.Tensor]:
        """Slice a concatenated embedding tensor into individual image embeddings."""
        slices, offset = [], 0
        for grid in grid_thw:
            count = self.get_num_tokens(grid, modality)
            slices.append(mm_embedding[offset : offset + count])
            offset += count
        return slices

    def _calculate_hashes_from_features(
        self, mm_feature, grid_thw: List, modality: Modality
    ) -> List[int]:
        """CPU Task: Compute hashes based on processed feature patches."""
        hashes = []
        if modality == Modality.AUDIO and isinstance(mm_feature, list):
            for feature in mm_feature:
                tmp_item = MultimodalDataItem(modality=modality, feature=feature)
                tmp_item.set_pad_value()
                hashes.append(tmp_item.hash)
            return hashes

        offset = 0
        logger.info(f"{mm_feature.shape=} with {modality=}")
        for grid in grid_thw:
            num_patches = self.get_num_patches(grid, modality)
            feature_slice = mm_feature[offset : offset + num_patches]
            tmp_item = MultimodalDataItem(modality=modality, feature=feature_slice)
            tmp_item.set_pad_value()
            hashes.append(tmp_item.hash)
            offset += num_patches
        return hashes

    async def _encode_missing(
        self,
        mm_feature,
        mm_inputs: dict,
        indices: List[int],
        modality: Modality = Modality.IMAGE,
        get_feature_fn=None,
    ) -> List[torch.Tensor]:
        """
        GPU Task: Run ViT inference ONLY on the subset of mm items missing from the cache.
        """
        grid_thw = _get_mm_grid_dim(mm_inputs, modality, self.model_type)

        # Audio features are per-item (list of mels for mimo_v2, or batched
        # N x n_mels x T_max for qwen2_audio); slice by item index and keep
        # per-item shape. Image/video features are concatenated along the
        # patch dim; slice by cumulative patch offsets and cat.
        if modality == Modality.AUDIO:
            if isinstance(mm_feature, list):
                sub_feature = [mm_feature[i] for i in indices]
            else:
                sub_feature = mm_feature[list(indices)]
        else:
            sub_feature_list = []
            offsets = [0]
            curr = 0
            for g in grid_thw:
                curr += self.get_num_patches(g, modality)
                offsets.append(curr)
            for idx in indices:
                sub_feature_list.append(mm_feature[offsets[idx] : offsets[idx + 1]])
            sub_feature = torch.cat(sub_feature_list, dim=0)

        mm_item = MultimodalDataItem.from_dict(
            {
                "modality": modality,
                "feature": (
                    sub_feature
                    if isinstance(sub_feature, list)
                    else _convert(sub_feature)
                ),
            }
        )

        for k, v in mm_inputs.items():
            if k in _mm_feature_attrs.get(modality, []):
                continue
            val = _convert(v)
            if k in _mm_grid_attrs.get(modality, []):
                mm_item.set(k, val[indices])
            else:
                mm_item.set(k, val)

        with torch.inference_mode():
            new_embeddings = get_feature_fn([mm_item]).cpu()
            if new_embeddings.ndim != 2:
                new_embeddings = new_embeddings.reshape(-1, new_embeddings.shape[-1])

        sub_grids = [grid_thw[i] for i in indices]
        return self.slice_embedding(new_embeddings, sub_grids, modality)

    async def encode_with_global_cache(
        self,
        mm_items,
        modality: Modality,
        req_id: str,
        num_parts: int,
        part_idx: int,
        hashes: Optional[List[str]] = None,
    ) -> torch.Tensor:
        # mm_inputs: dict
        mm_inputs, get_feature_fn = await self._process_mm_items(mm_items, modality)
        grid_thw = _get_mm_grid_dim(mm_inputs, modality, self.model_type)
        mm_feature = _convert(_get_mm_feature(mm_inputs, modality))
        num_items = len(grid_thw)

        # Hashes must be grid-space; a leaf-space list would size-mismatch
        # rank>0's mask (zeros(num_items)) and deadlock TP.
        if hashes is not None and len(hashes) != num_items:
            raise BadRequestError(
                f"User-supplied hashes length {len(hashes)} != grid count "
                f"{num_items} for {self.model_type}/{modality.name}; hashes "
                f"must be in grid space (1 per encoder grid entry)."
            )

        # Step 1: Rank 0 checks global cache and broadcasts hit/miss mask to all ranks.
        if self.rank == 0:
            if hashes is None:
                mm_hashes = self._calculate_hashes_from_features(
                    mm_feature, grid_thw, modality
                )
            else:
                mm_hashes = hashes
            # Convert hashes to strings (L2 cache expects string keys for Mooncake)
            str_mm_hashes = [str(h) for h in mm_hashes]
            exist_mask = await self.mm_global_cache.batch_is_exist(str_mm_hashes)
            mask_tensor = torch.tensor(
                [1 if e else 0 for e in exist_mask], dtype=torch.int32
            )
        else:
            mm_hashes = None
            mask_tensor = torch.zeros(num_items, dtype=torch.int32)

        if self.server_args.tp_size > 1:
            torch.distributed.broadcast(
                mask_tensor,
                src=0,
                group=self.mm_global_cache.prefetch_tp_group,
            )

        exist_mask = [m.item() == 1 for m in mask_tensor]
        missing_indices = [i for i, e in enumerate(exist_mask) if not e]
        hit_indices = [i for i, e in enumerate(exist_mask) if e]

        # Step 2: All ranks run ViT together on cache-miss images.
        new_slices = []
        if missing_indices:
            new_slices = await self._encode_missing(
                mm_feature, mm_inputs, missing_indices, modality, get_feature_fn
            )

        # Step 3: Rank 0 prefetches cache-hit embeddings from global cache.
        prefetch_status = torch.tensor([1], dtype=torch.int32)

        if self.rank == 0:
            if hit_indices:
                hit_hashes = [str_mm_hashes[i] for i in hit_indices]
                hit_tokens = [
                    self.get_num_tokens(grid_thw[i], modality) for i in hit_indices
                ]
                self.mm_global_cache.prefetch(req_id, hit_hashes, hit_tokens, modality)

                try:

                    async def _wait_prefetch():
                        while not self.mm_global_cache.check_prefetch_progress(req_id):
                            await asyncio.sleep(0.005)

                    await asyncio.wait_for(_wait_prefetch(), timeout=60.0)
                except (asyncio.TimeoutError, Exception) as e:
                    logger.error(
                        f"Prefetch failed for req {req_id}: {e}. "
                        f"Falling back to ViT for {len(hit_indices)} hit items."
                    )
                    prefetch_status[0] = 0

        # Step 4: Broadcast prefetch result to all ranks so they stay in sync.
        if self.server_args.tp_size > 1:
            torch.distributed.broadcast(
                prefetch_status,
                src=0,
                group=self.mm_global_cache.prefetch_tp_group,
            )

        # Step 5: If prefetch failed, all ranks fallback to ViT for the hit mm items.
        if prefetch_status.item() == 0 and hit_indices:
            logger.info(
                f"Req {req_id}: Prefetch failed, all ranks running ViT fallback "
                f"for {len(hit_indices)} mm items."
            )
            fallback_slices = await self._encode_missing(
                mm_feature, mm_inputs, hit_indices, modality, get_feature_fn
            )
        else:
            fallback_slices = None

        # Step 6: Rank 0 assembles final embedding and prepares for sending.
        if self.rank == 0:
            final_slices = [None] * num_items

            for i, idx in enumerate(missing_indices):
                final_slices[idx] = new_slices[i]

            # Fill in cache-hit embeddings (from prefetch or fallback)
            if prefetch_status.item() == 1 and hit_indices:
                cached_slices = self.mm_global_cache.get_embeddings(
                    [str_mm_hashes[i] for i in hit_indices]
                )
                for i, idx in enumerate(hit_indices):
                    final_slices[idx] = cached_slices[i]
            elif fallback_slices is not None:
                for i, idx in enumerate(hit_indices):
                    final_slices[idx] = fallback_slices[i]

            mm_embedding = torch.cat(final_slices, dim=0)

            # Background insert: store newly computed embeddings into global cache.
            # Includes both original misses and fallback-recomputed hits.
            all_new_hashes = [str_mm_hashes[i] for i in missing_indices]
            all_new_slices = list(new_slices)
            if fallback_slices is not None:
                all_new_hashes += [str_mm_hashes[i] for i in hit_indices]
                all_new_slices += list(fallback_slices)

            if all_new_hashes:

                async def _background_insert():
                    await asyncio.to_thread(
                        self.mm_global_cache.insert_batch,
                        all_new_hashes,
                        all_new_slices,
                    )

                task = asyncio.create_task(_background_insert())
                self.background_tasks.add(task)
                task.add_done_callback(self.background_tasks.discard)

            aux_data = _build_mm_aux_data(mm_inputs, self.model_type)
            self.embedding_to_send[req_id] = EmbeddingData(
                req_id,
                num_parts,
                part_idx,
                grid_thw,
                modality,
                mm_embedding,
                **aux_data,
            )
            return (
                mm_embedding.nbytes,
                mm_embedding.shape[0],
                mm_embedding.shape[1],
                None,
                None,
            )
        else:
            return (0, 0, 0, None, None)

    async def _flatten_and_load_audios(self, mm_items):
        """
        Flatten mm_items, load audios concurrently as np.ndarray at
        self.model_audio_sr, restore original structure.
        """
        return await self._flatten_and_load_data_by_modality(mm_items, Modality.AUDIO)

    async def _flatten_and_load_images(self, mm_items):
        """
        Flatten mm_items structure, load images concurrently, and restore original structure.
        """
        return await self._flatten_and_load_data_by_modality(mm_items, Modality.IMAGE)

    def _calculate_timestamps(self, indices, video_fps: float, merge_size: int = 2):
        """Calculate timestamps for video frames, used for qwen3_vl models."""
        # refer to https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/processing_qwen3_vl.py#L255
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend(
                indices[-1] for _ in range(merge_size - len(indices) % merge_size)
            )
        timestamps = [idx / video_fps for idx in indices]
        # Frames are merged by merge_size, so we need to average the timestamps
        # between the first/last frame within the temporal patch
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2
            for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps

    @staticmethod
    def _flatten_nested_items(items):
        if not isinstance(items, (list, tuple)):
            return [items]

        flat = []
        for item in items:
            if isinstance(item, (list, tuple)):
                flat.extend(MMEncoder._flatten_nested_items(item))
            else:
                flat.append(item)
        return flat

    def _grid_count_per_leaf(self, leaves: List, modality: Modality) -> List[int]:
        """Number of grid entries each leaf produces under the model's processor.

        Most processors map 1 leaf → 1 grid. Kimi-VL/K25 image processors expand
        a leaf shaped {"type": "image", "image": [pil1, pil2, ...]} into N grids
        (see _normalize_kimi_encoder_images). Cross-request batching needs these
        counts to keep per-request boundaries aligned with grid_dim.
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

        # Kimi-VL image processor expects a flat list of concrete images.
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

        # Kimi-K2.5 vision processor expects media dicts.
        normalized = []
        for img in images:
            wrapped = wrap_one(img)
            for media in wrapped:
                # Some pipelines may produce {"type": "image", "image": [PIL]}.
                # Split it into one media item per concrete image object.
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

    async def _process_mm_items(self, mm_items, modality):
        model_preprocessor = getattr(self.model, "preprocess_mm_for_encoder", None)

        if modality == Modality.IMAGE:
            processor_input = await self._process_image_items(
                mm_items, model_preprocessor
            )
        elif modality == Modality.VIDEO:
            processor_input = await self._process_video_items(
                mm_items, model_preprocessor
            )
        elif modality == Modality.AUDIO:
            processor_input = await self._process_audio_items(
                mm_items, model_preprocessor
            )
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        target = self.model.thinker if hasattr(self.model, "thinker") else self.model
        get_feature_method = getattr(target, f"get_{modality.name.lower()}_feature")
        return processor_input, get_feature_method

    async def _process_image_items(self, mm_items, model_preprocessor):
        if not (self.image_processor or model_preprocessor):
            raise ValueError("No image processor available")
        images = await self._flatten_and_load_images(mm_items)
        if model_preprocessor:
            return model_preprocessor(images, Modality.IMAGE, self.vision_config)
        image_config = self.vision_config.get("image", {})
        if self.model_type in ["kimi_k25", "kimi_vl"]:
            images = self._normalize_kimi_encoder_images(images)
        return self.image_processor(images=images, **image_config)

    async def _process_video_items(self, mm_items, model_preprocessor):
        if model_preprocessor:
            return model_preprocessor(mm_items, Modality.VIDEO, self.vision_config)
        if not self.video_processor:
            raise ValueError("No video processor available")

        videos, video_processor_kwargs = await self._flatten_and_load_videos(mm_items)
        processor_input = self.video_processor(videos=videos, **video_processor_kwargs)

        # Get additional video metadata
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
                merge_size = 2  # Default merge_size

            video_timestamps = []
            for metadata in video_metadata:
                video_fps = metadata.get("fps", None) or 24  # original video fps
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
                temporal_patch_size = 2  # Default temporal_patch_size
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
        # Await off the event loop so EncoderScheduler can accumulate
        # cross-request batches during download.
        audios = await self._flatten_and_load_audios(mm_items)

        if model_preprocessor:
            return model_preprocessor(audios, Modality.AUDIO, self.vision_config)

        if not self.audio_processor:
            raise ValueError("No audio processor available")

        audio_config = self.vision_config.get("audio", {})
        processor_input = self.audio_processor.feature_extractor(audios, **audio_config)
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

    async def _encode(self, mm_items, modality: Modality) -> torch.Tensor:
        try:
            mm_inputs, get_feature_fn = await self._process_mm_items(mm_items, modality)
        except NotImplementedError as e:
            raise InternalError(f"Not implemented error: {str(e)}")
        except Exception as e:
            raise BadRequestError(f"Failed to process mm items: {str(e)}")
        try:
            # support mm_cache
            mm_embedding = None
            mm_hash = None

            mm_item = MultimodalDataItem.from_dict(
                {
                    "modality": modality,
                    "feature": _convert(_get_mm_feature(mm_inputs, modality)),
                }
            )
            for k, v in mm_inputs.items():
                if k in _mm_feature_attrs[modality]:
                    continue
                mm_item.set(k, _convert(v))

            if self.server_args.enable_prefix_mm_cache:
                mm_item.set_pad_value()
                mm_hash = MultiModalStaticCache.combine_hashes([mm_item.hash])
                async with self.mm_cache_lock:
                    mm_cache = self.mm_cache.get([mm_item.hash])
                    if mm_cache is not None:
                        mm_embedding = mm_cache.embedding

            if mm_embedding is None:
                with torch.inference_mode():
                    mm_embedding: torch.Tensor = get_feature_fn([mm_item])
                    mm_embedding = mm_embedding.cpu()
                if len(mm_embedding.shape) != 2:
                    mm_embedding = mm_embedding.reshape(-1, mm_embedding.shape[-1])

            if self.server_args.enable_prefix_mm_cache:
                async with self.mm_cache_lock:
                    self.mm_cache.set(mm_hash, EmbeddingResult(embedding=mm_embedding))
            if self.profiler is not None:
                self.profiler.step()

            aux_data = _build_mm_aux_data(mm_inputs, self.model_type)

            if modality == Modality.VIDEO and mm_inputs.get("video_audio_features"):
                target = (
                    self.model.thinker if hasattr(self.model, "thinker") else self.model
                )
                encode_video_audio_fn = getattr(target, "encode_video_audio", None)
                if encode_video_audio_fn is not None:
                    audio_embedding = encode_video_audio_fn(mm_inputs)
                    if audio_embedding is not None:
                        aux_data["video_audio_embedding"] = audio_embedding
                else:
                    logger.warning(
                        "Videos carry audio tracks but model has no "
                        "encode_video_audio; dropping audio for EPD encoding."
                    )

            return (
                _get_mm_grid_dim(mm_inputs, modality, self.model_type),
                mm_embedding,
                aux_data,
            )
        except BadRequestError as e:
            raise BadRequestError(f"Bad request error: {str(e)}")
        except Exception as e:
            raise InternalError(f"Internal encoding error: {str(e)}")

    async def _send(
        self,
        embedding: torch.Tensor,
        mm_data: EmbeddingData,
        session_id=None,
        buffer_address=None,
        prefill_host=None,
        embedding_port=None,
        url=None,
    ):
        if self.server_args.encoder_transfer_backend == "mooncake":
            self.engine.register(embedding.data_ptr(), embedding.nbytes)
            self.engine.transfer_sync(
                session_id, embedding.data_ptr(), buffer_address, embedding.nbytes
            )
            self.engine.deregister(embedding.data_ptr())

            mm_data.embedding = None

        # Send ack/data
        if url is not None:
            endpoint = NetworkAddress.parse(url).to_tcp()
        else:
            endpoint = NetworkAddress(prefill_host, embedding_port).to_tcp()
        logger.info(f"{endpoint = }")

        # Serialize data
        if self.server_args.encoder_transfer_backend == "mooncake":
            serialized_data = pickle.dumps(mm_data)
            buffer = None
        else:
            new_mm_data = mm_data.copy_without_embedding()
            if new_mm_data.error_msg is not None:
                buffer = None
                serialized_data = pickle.dumps(new_mm_data)
            else:
                embedding_tensor = TensorWrapper(mm_data.embedding)
                serialized_data = pickle.dumps(new_mm_data)
                buffer = embedding_tensor.__buffer__()

        # Use thread pool executor for parallel ZMQ send operations
        def send_with_socket():
            sock = self.sync_context.socket(zmq.PUSH)
            config_socket(sock, zmq.PUSH)
            try:
                sock.connect(endpoint)
                if buffer is not None:
                    sock.send_multipart([serialized_data, buffer], copy=False)
                else:
                    sock.send_multipart([serialized_data], copy=False)
            finally:
                sock.close()

        await asyncio.get_event_loop().run_in_executor(self.executor, send_with_socket)

    async def encode(self, mm_items, modality: Modality, req_id, num_parts, part_idx):
        try:
            grid_dim, mm_embedding, aux_data = await self._encode(mm_items, modality)

            if self.rank == 0:
                mm_data = EmbeddingData(
                    req_id,
                    num_parts,
                    part_idx,
                    grid_dim,
                    modality,
                    mm_embedding,
                    **aux_data,
                )
                self.embedding_to_send[req_id] = mm_data
            return (
                mm_embedding.nbytes,
                mm_embedding.shape[0],
                mm_embedding.shape[1],
                None,
                None,
            )
        except Exception as e:
            error_code = getattr(e, "code", HTTPStatus.INTERNAL_SERVER_ERROR)
            error_msg = str(e)
            logger.error(f"Rank {self.rank} encode failed: {error_msg} {error_code = }")
            if self.rank == 0:
                mm_data = EmbeddingData(
                    req_id,
                    num_parts,
                    part_idx,
                    None,
                    modality,
                    error_msg=error_msg,
                    error_code=error_code,
                )
                self.embedding_to_send[req_id] = mm_data
                logger.debug(f"Created error EmbeddingData: {mm_data}")
            return 0, 0, 0, error_msg, error_code

    async def encode_request(self, req: dict, modality: Modality):
        """Single-request encode dispatcher: picks cache vs no-cache path."""
        if self.mm_global_cache is not None:
            return await self.encode_with_global_cache(
                mm_items=req["mm_items"],
                modality=modality,
                req_id=req["req_id"],
                num_parts=req["num_parts"],
                part_idx=req["part_idx"],
                hashes=req.get("hashes"),
            )
        return await self.encode(
            mm_items=req["mm_items"],
            modality=modality,
            req_id=req["req_id"],
            num_parts=req["num_parts"],
            part_idx=req["part_idx"],
        )

    async def batch_encode(
        self, requests: List[dict], modality: Modality
    ) -> List[Tuple[int, int, int, Optional[str], Optional[int]]]:
        """Cross-request encoder fusion (image/audio). No cache path."""
        # items_per_req counts grid entries (post-expansion) so per-request
        # slicing of grid_dim/final_slices stays aligned for processors that
        # expand one leaf into multiple grids (e.g. Kimi-VL/K25 dict-of-images).
        flat_items, items_per_req = [], []
        for req in requests:
            leaves = MMEncoder._flatten_nested_items(req["mm_items"])
            flat_items.extend(leaves)
            items_per_req.append(sum(self._grid_count_per_leaf(leaves, modality)))
        total = sum(items_per_req)

        try:
            mm_inputs, get_feat = await self._process_mm_items(flat_items, modality)
        except NotImplementedError as e:
            return self._batch_set_error(
                requests, modality, InternalError(f"Not implemented error: {e}")
            )
        except Exception as e:
            return self._batch_set_error(
                requests, modality, BadRequestError(f"Failed to process mm items: {e}")
            )

        try:
            mm_feature = _convert(_get_mm_feature(mm_inputs, modality))
            grid_dim = _get_mm_grid_dim(mm_inputs, modality, self.model_type)
            if len(grid_dim) != total:
                return self._batch_set_error(
                    requests,
                    modality,
                    InternalError(
                        f"Grid count mismatch for {self.model_type}/"
                        f"{modality.name}: {len(flat_items)} leaves across "
                        f"{len(requests)} requests → expected {total} grids "
                        f"(per-req {items_per_req}), but processor produced "
                        f"{len(grid_dim)}. Add tile-expansion handling in "
                        f"_grid_count_per_leaf."
                    ),
                )

            final_slices = await self._encode_missing(
                mm_feature,
                mm_inputs,
                list(range(total)),
                modality,
                get_feat,
            )

            if self.profiler is not None:
                for _ in requests:
                    self.profiler.step()
            # No aux_data here: batch_encode only handles IMAGE/AUDIO
            # (_BATCHABLE_MODALITIES), and _build_mm_aux_data only extracts
            # video-meta fields — which never appear in image/audio mm_inputs.
            results = []
            offset = 0
            for req, n in zip(requests, items_per_req):
                slices = final_slices[offset : offset + n]
                emb = slices[0] if n == 1 else torch.cat(slices, dim=0)
                if self.rank == 0:
                    self.embedding_to_send[req["req_id"]] = EmbeddingData(
                        req["req_id"],
                        req["num_parts"],
                        req["part_idx"],
                        grid_dim[offset : offset + n],
                        modality,
                        emb,
                    )
                results.append((emb.nbytes, emb.shape[0], emb.shape[1], None, None))
                offset += n
            return results
        except Exception as e:
            return self._batch_set_error(
                requests, modality, InternalError(f"Internal encoding error: {e}")
            )

    def _batch_set_error(
        self, requests: List[dict], modality: Modality, exc: Exception
    ) -> List[Tuple[int, int, int, str, int]]:
        code = getattr(exc, "code", HTTPStatus.INTERNAL_SERVER_ERROR)
        msg = str(exc)
        logger.error(f"Rank {self.rank} batch_encode failed: {msg} {code = }")
        if self.rank == 0:
            for req in requests:
                self.embedding_to_send[req["req_id"]] = EmbeddingData(
                    req["req_id"],
                    req["num_parts"],
                    req["part_idx"],
                    None,
                    modality,
                    error_msg=msg,
                    error_code=code,
                )
        return [(0, 0, 0, msg, code)] * len(requests)

    # For zmq_to_tokenizer zmq_to_scheduler and mooncake
    async def send(
        self, req_id, prefill_host, embedding_port, session_id=None, buffer_address=None
    ):
        mm_data: EmbeddingData = self.embedding_to_send[req_id]
        await self._send(
            mm_data.embedding,
            mm_data,
            session_id=session_id,
            buffer_address=buffer_address,
            prefill_host=prefill_host,
            embedding_port=embedding_port,
        )

    # For zmq_to_scheduler
    async def send_with_url(
        self,
        req_id,
    ):
        mm_data = self.embedding_to_send.get(req_id)
        if not mm_data:
            return
        sent_urls: Set[str] = set()
        all_tasks: List[Tuple[asyncio.Task, str]] = []
        start_time = asyncio.get_running_loop().time()
        timeout = self.send_timeout
        cond = await get_condition(req_id)

        try:
            while True:
                async with rid_lock:
                    current_targets = rid_to_receive_endpoint.get(req_id, set()).copy()
                    expected_count = rid_to_receive_count.get(req_id)

                new_targets = current_targets - sent_urls

                if new_targets:
                    logger.info(
                        f"Found {len(new_targets)} new endpoints for {req_id}. Starting tasks..."
                    )
                    for url in new_targets:
                        task = asyncio.create_task(
                            self._send(
                                mm_data.embedding,
                                mm_data,
                                url=url,
                            )
                        )
                        all_tasks.append((task, url))
                        sent_urls.add(url)  # Mark as handled immediately
                if expected_count is not None and len(sent_urls) >= expected_count:
                    logger.info(
                        f"All {expected_count} endpoints initiated for {req_id}. Breaking loop."
                    )
                    break
                remaining = timeout - (asyncio.get_running_loop().time() - start_time)
                if remaining <= 0:
                    logger.error(
                        f"[{req_id}] Timeout! Sent {len(sent_urls)}/{expected_count}"
                    )
                    break

                async with cond:
                    try:
                        await asyncio.wait_for(cond.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        continue

            if all_tasks:
                logger.info(
                    f"Loop finished. Awaiting completion of {len(all_tasks)} sending tasks..."
                )
                tasks_only = [t[0] for t in all_tasks]
                results = await asyncio.gather(*tasks_only, return_exceptions=True)

                # Process results and log errors
                for i, result in enumerate(results):
                    url = all_tasks[i][1]  # Retrieve URL associated with the task
                    if isinstance(result, Exception):
                        logger.error(f"Failed to send to {url}: {result}")
                    else:
                        logger.debug(f"Successfully sent to {url}")

            logger.info(f"All tasks completed for req_id: {req_id}")

        finally:
            logger.info(f"Cleaning up resources for req_id {req_id}")
            async with rid_lock:
                rid_to_receive_endpoint.pop(req_id, None)
                rid_to_receive_count.pop(req_id, None)
            async with cond_dict_lock:
                rid_to_cond.pop(req_id, None)
            self.embedding_to_send.pop(req_id, None)

    async def get_embedding_port(self, prefill_url):
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=1800)
        ) as session:
            response = await session.post(
                f"{prefill_url}/embedding_bootstrap",
                json={"embedding_port": None},
            )
            response_json = await response.json()
            return response_json["embedding_port"]


class EncoderProfiler:
    def __init__(self, rank: int):
        self.rank = rank
        self.profiler = None
        self.steps_left = None
        self.output_dir = None
        self.prefix = None
        self.profile_id = None

    def start(self, obj: ProfileReq):
        if self.profiler is not None:
            return False, "profiling already running"

        output_dir = obj.output_dir or os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.prefix = obj.profile_prefix or "encoder"
        self.profile_id = str(time.time())

        activities = obj.activities or ["CPU", "GPU"]
        torch_activities = []
        if "CPU" in activities:
            torch_activities.append(torch.profiler.ProfilerActivity.CPU)
        if "GPU" in activities:
            torch_activities.append(torch.profiler.ProfilerActivity.CUDA)

        profile_memory = "MEM" in activities
        if not torch_activities and not profile_memory:
            return False, "no supported activities"

        self.profiler = torch.profiler.profile(
            activities=torch_activities,
            with_stack=True if obj.with_stack is None else obj.with_stack,
            record_shapes=False if obj.record_shapes is None else obj.record_shapes,
            profile_memory=profile_memory,
        )
        self.profiler.start()
        self.steps_left = obj.num_steps
        logger.info(
            f"Encoder profiling started. output_dir={self.output_dir} profile_id={self.profile_id}"
        )
        return True, None

    def step(self):
        if self.profiler is None:
            return
        self.profiler.step()
        if self.steps_left is not None:
            self.steps_left -= 1
            if self.steps_left <= 0:
                self.stop()

    def stop(self):
        if self.profiler is None:
            return False, "profiling not running"
        self.profiler.stop()
        filename = f"{self.prefix}-rank{self.rank}-{self.profile_id}.trace.json"
        trace_path = os.path.join(self.output_dir, filename)
        self.profiler.export_chrome_trace(trace_path)
        logger.info("Encoder profiling saved to: %s", trace_path)
        self.profiler = None
        self.steps_left = None
        return True, None


class PendingRequest:
    __slots__ = ("request", "future", "submit_time")

    def __init__(self, request: dict, loop: asyncio.AbstractEventLoop):
        self.request = request
        self.future: asyncio.Future = loop.create_future()
        self.submit_time = time.time()


# VIDEO excluded: per-video preprocess kwargs (do_sample_frames, video_metadata)
# vary per request and can't merge into one HF processor call.
_BATCHABLE_MODALITIES = {Modality.IMAGE, Modality.AUDIO}


class EncoderScheduler:
    """Aggregate concurrent /encode requests into bounded image/audio batches."""

    def __init__(
        self,
        encoder: "MMEncoder",
        send_sockets: List[zmq.Socket],
        max_batch_size: int,
        request_timeout: float = ENCODER_REQ_TIMEOUT,
    ):
        self.encoder = encoder
        self.send_sockets = send_sockets
        self.max_batch_size = max(1, int(max_batch_size))
        self.request_timeout = max(1.0, float(request_timeout))
        self.pending_queue: "asyncio.Queue[PendingRequest]" = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._batch_worker())
            logger.info(
                f"EncoderScheduler started with max_batch_size={self.max_batch_size}"
            )

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None
        # Reject any requests still queued so their HTTP handlers don't hang.
        while True:
            try:
                pending = self.pending_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not pending.future.done():
                pending.future.set_exception(RuntimeError("EncoderScheduler stopped"))

    async def submit(self, request: dict) -> Tuple:
        pending = PendingRequest(request, asyncio.get_running_loop())
        await self.pending_queue.put(pending)
        try:
            return await asyncio.wait_for(pending.future, timeout=self.request_timeout)
        except asyncio.TimeoutError:
            if not pending.future.done():
                pending.future.cancel()
            req_id = request.get("req_id")
            logger.error(
                f"EncoderScheduler.submit timed out after {self.request_timeout}s "
                f"for req_id={req_id}"
            )
            raise

    async def _collect_batch(self) -> List[PendingRequest]:
        batch = [await self.pending_queue.get()]
        while len(batch) < self.max_batch_size:
            try:
                batch.append(self.pending_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return batch

    async def _batch_worker(self) -> None:
        while True:
            batch: List[PendingRequest] = []
            try:
                batch = await self._collect_batch()
                groups: Dict[Modality, List[PendingRequest]] = defaultdict(list)
                for p in batch:
                    groups[
                        Modality.from_str(p.request.get("modality", "image"))
                    ].append(p)
                for modality, group in groups.items():
                    await self._dispatch_group(group, modality)
            except asyncio.CancelledError:
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(RuntimeError("EncoderScheduler stopped"))
                raise
            except Exception as e:
                logger.error(
                    f"Error in EncoderScheduler batch worker: {e}", exc_info=True
                )
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(e)

    @staticmethod
    def _validate_request_shape(req: dict) -> Optional[str]:
        # Cheap pre-broadcast checks: shape errors that don't require running
        # the HF processor. Once a request reaches TP workers they enter
        # batch_encode and expect to join its collectives — a malformed batch
        # that makes rank-0 bail mid-flight would deadlock the workers.
        if not isinstance(req, dict):
            return f"request is not a dict: {type(req).__name__}"
        if not req.get("req_id"):
            return "missing req_id"
        if not req.get("mm_items"):
            return "missing or empty mm_items"
        if "num_parts" not in req or "part_idx" not in req:
            return "missing num_parts / part_idx"
        h = req.get("hashes")
        if h is not None and not isinstance(h, (list, tuple, str, int, bytes)):
            return f"hashes must be list/scalar, got {type(h).__name__}"
        return None

    async def _dispatch_group(
        self, group: List[PendingRequest], modality: Modality
    ) -> None:
        # Video can't fuse (per-video preprocess kwargs vary).
        if modality not in _BATCHABLE_MODALITIES:
            await self._dispatch_per_request(group, modality)
            return

        # Drop structurally-bad requests before broadcasting; otherwise TP
        # workers would join batch_encode collectives that rank-0 has already
        # abandoned.
        valid: List[PendingRequest] = []
        for p in group:
            err = self._validate_request_shape(p.request)
            if err is None:
                valid.append(p)
                continue
            logger.error(f"Dropping req_id={p.request.get('req_id')} from batch: {err}")
            if not p.future.done():
                p.future.set_exception(BadRequestError(err))
        if not valid:
            return
        group = valid

        requests = [p.request for p in group]
        start = time.time()
        for sock in self.send_sockets:
            sock.send_pyobj(
                {
                    "type": "batch_encode",
                    "modality": modality.name,
                    "requests": requests,
                    "enter_time": start,
                }
            )

        logger.info(f"Dispatching batch of {len(group)} {modality.name} requests")

        try:
            results = await self.encoder.batch_encode(requests, modality)
            if len(group) > 1:
                logger.info(
                    f"Batch of {len(group)} {modality.name} requests completed in "
                    f"{(time.time() - start) * 1000:.1f}ms"
                )
        except Exception as e:
            # batch_encode normally catches and returns errors via _batch_set_error.
            # If it raised, rank-0 may have skipped a collective broadcast, leaving
            # TP workers stuck. Don't try to recover — fail every pending future
            # and let the client retry. Re-broadcasting would risk a deadlock.
            logger.error(f"batch_encode raised: {e}", exc_info=True)
            for p in group:
                if not p.future.done():
                    p.future.set_exception(e)
            return

        if len(results) != len(group):
            err = RuntimeError(
                f"batch_encode returned {len(results)} results for {len(group)} requests"
            )
            logger.error(str(err))
            for p in group:
                if not p.future.done():
                    p.future.set_exception(err)
            return

        for p, result in zip(group, results):
            if not p.future.done():
                p.future.set_result(result)

    async def _dispatch_per_request(
        self,
        group: List[PendingRequest],
        modality: Modality,
    ) -> None:
        for p in group:
            req = p.request
            try:
                for sock in self.send_sockets:
                    sock.send_pyobj(req)
                result = await self.encoder.encode_request(req, modality)
                if not p.future.done():
                    p.future.set_result(result)
            except Exception as e:
                logger.error(
                    f"Per-request encode failed for req_id={req.get('req_id')}: {e}"
                )
                if not p.future.done():
                    p.future.set_exception(e)


encoder: Optional[MMEncoder] = None
send_sockets: List[zmq.Socket] = []
encoder_scheduler: Optional[EncoderScheduler] = None


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    global encoder_scheduler
    if encoder is not None:
        encoder_scheduler = EncoderScheduler(
            encoder, send_sockets, max_batch_size=ENCODER_MAX_BATCH_SIZE
        )
        encoder_scheduler.start()
    try:
        yield
    finally:
        if encoder_scheduler is not None:
            await encoder_scheduler.stop()


app = FastAPI(lifespan=_lifespan)


async def run_encoder(
    server_args: ServerArgs, schedule_path, dist_init_method, rank: int
):
    encoder = MMEncoder(server_args, schedule_path, dist_init_method, rank)
    while True:
        request = await encoder.schedule_socket.recv_pyobj()
        if isinstance(request, ProfileReq):
            if request.type == ProfileReqType.START_PROFILE:
                if encoder.profiler is None:
                    encoder.profiler = EncoderProfiler(encoder.rank)
                encoder.profiler.start(request)
            else:
                encoder.profiler.stop()
        elif isinstance(request, dict) and request.get("type") == "batch_encode":
            await encoder.batch_encode(
                request["requests"],
                Modality.from_str(request["modality"]),
            )
        else:
            await encoder.encode_request(
                request, Modality.from_str(request["modality"])
            )


def launch_encoder(server_args, schedule_path, dist_init_method, rank):
    try:
        asyncio.run(run_encoder(server_args, schedule_path, dist_init_method, rank))
    except KeyboardInterrupt:
        logger.info(f"Exit rank {rank}")
    except Exception:
        traceback.print_exc()


def launch_server(server_args: ServerArgs):
    configure_logger(server_args, prefix=" encode_server")
    global encoder
    ctx = mp.get_context("spawn")
    zmq_ctx = zmq.Context(10)
    ipc_path_prefix = random_uuid()
    port_args = PortArgs.init_new(server_args)
    if server_args.dist_init_addr:
        na = NetworkAddress.parse(server_args.dist_init_addr)
        dist_init_method = na.to_tcp()
    else:
        dist_init_method = NetworkAddress(
            server_args.host or "127.0.0.1", port_args.nccl_port
        ).to_tcp()
    for rank in range(1, server_args.tp_size):
        schedule_path = f"ipc:///tmp/{ipc_path_prefix}_schedule_{rank}"
        send_sockets.append(
            get_zmq_socket(zmq_ctx, zmq.PUSH, schedule_path, bind=False)
        )
        ctx.Process(
            target=launch_encoder,
            args=(server_args, schedule_path, dist_init_method, rank),
            daemon=True,
        ).start()
    encoder = MMEncoder(server_args, dist_init_method=dist_init_method)
    uvicorn.run(app, host=server_args.host, port=server_args.port)


async def get_condition(rid):
    async with cond_dict_lock:
        if rid not in rid_to_cond:
            rid_to_cond[rid] = asyncio.Condition()
        return rid_to_cond[rid]


@app.post("/encode")
async def handle_encode_request(request: dict):
    req_id = request["req_id"]
    start_time = time.monotonic()
    try:

        def start_background_send(req_id):
            task = asyncio.create_task(encoder.send_with_url(req_id=req_id))
            encoder.background_tasks.add(task)
            task.add_done_callback(encoder.background_tasks.discard)

        request.update({"enter_time": time.time()})
        modality = Modality.from_str(request["modality"])
        if encoder_scheduler is not None and modality in _BATCHABLE_MODALITIES:
            try:
                nbytes, embedding_len, embedding_dim, error_msg, error_code = (
                    await encoder_scheduler.submit(request)
                )
            except asyncio.TimeoutError:
                return ORJSONResponse(
                    status_code=HTTPStatus.GATEWAY_TIMEOUT,
                    content={
                        "status": "error",
                        "message": "encoder batch timed out",
                        "req_id": req_id,
                    },
                )
        else:
            for socket in send_sockets:
                socket.send_pyobj(request)
            nbytes, embedding_len, embedding_dim, error_msg, error_code = (
                await encoder.encode_request(request, modality)
            )

        if error_msg:
            if encoder.server_args.encoder_transfer_backend == "zmq_to_scheduler":
                if request["embedding_port"] is None:
                    start_background_send(req_id)
                else:
                    for port in request["embedding_port"]:
                        await encoder.send(
                            req_id=req_id,
                            prefill_host=request["prefill_host"],
                            embedding_port=port,
                        )
            return ORJSONResponse(
                status_code=error_code,
                content={"status": "error", "message": error_msg, "req_id": req_id},
            )
        if encoder.server_args.encoder_transfer_backend == "mooncake":
            del request["mm_items"]
            request.update(
                {
                    "embedding_size": nbytes,
                    "embedding_len": embedding_len,
                    "embedding_dim": embedding_dim,
                }
            )
            return ORJSONResponse(content=request)
        elif encoder.server_args.encoder_transfer_backend == "zmq_to_scheduler":
            logger.info(f"{request['embedding_port'] = }")
            if request["embedding_port"] is None:
                await encoder.send_with_url(
                    req_id=request["req_id"],
                )
            else:
                assert type(request["embedding_port"]) == list
                tasks = []
                for embedding_port in request["embedding_port"]:
                    tasks.append(
                        encoder.send(
                            req_id=request["req_id"],
                            prefill_host=request["prefill_host"],
                            embedding_port=embedding_port,
                        )
                    )
                await asyncio.gather(*tasks)
                encoder.embedding_to_send.pop(request["req_id"], None)
            return ORJSONResponse(content=None)
        elif encoder.server_args.encoder_transfer_backend == "zmq_to_tokenizer":
            await encoder.send(
                req_id=request["req_id"],
                prefill_host=request["prefill_host"],
                embedding_port=request["embedding_port"],
            )
            encoder.embedding_to_send.pop(request["req_id"], None)
            elapsed = time.monotonic() - start_time
            logger.info(
                f"[{req_id}] /encode completed in {elapsed:.3f}s, "
                f"modality={request['modality']}, tokens={embedding_len}"
            )
            return ORJSONResponse(content=None)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Unexpected error in encoder logic for {req_id}: {error_msg}")
        rid_to_err_msg[req_id] = error_msg
        return ORJSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": error_msg,
                "req_id": req_id,
            },
        )


@app.post("/send")
async def handle_send_request(request: dict):
    # mooncake backend
    await encoder.send(
        req_id=request["req_id"],
        prefill_host=request["prefill_host"],
        embedding_port=request["embedding_port"],
        session_id=request["session_id"],
        buffer_address=request["buffer_address"],
    )
    encoder.embedding_to_send.pop(request["req_id"], None)
    return ORJSONResponse(content=None)


@app.post("/scheduler_receive_url")
async def handle_scheduler_receive_url_request(request: dict):
    rid = request["req_id"]
    async with rid_lock:
        global rid_to_receive_endpoint
        if rid not in rid_to_receive_endpoint:
            rid_to_receive_endpoint[rid] = set()
            rid_to_receive_count[rid] = request["receive_count"]
        assert rid_to_receive_count[rid] == request["receive_count"]
        rid_to_receive_endpoint[rid].add(request["receive_url"])
    cond = await get_condition(rid)
    async with cond:
        cond.notify_all()


@app.get("/health")
@app.get("/health_generate")
async def health_generate():
    """
    Health check endpoint for the encoder server.
    Performs a dummy encode to verify the encoder is functional.
    Returns 200 if the encoder is healthy, 503 otherwise.
    """
    if encoder is None:
        return Response(status_code=503)

    # Skip the dummy encode when real requests are already in flight — the
    # ongoing traffic already proves liveness, matching the scheduler's
    # `is_fully_idle`-based health-check skip pattern.
    if encoder.embedding_to_send:
        return Response(status_code=200)

    # Pick the first available modality for the dummy encode
    if encoder.image_processor is not None:
        mm_items = [f"data:image/png;base64,{MINIMUM_PNG_PICTURE_BASE64}"]
        modality = Modality.IMAGE
    elif encoder.audio_processor is not None:
        mm_items = [f"data:audio/wav;base64,{MINIMUM_WAV_SILENCE_BASE64}"]
        modality = Modality.AUDIO
    else:
        # No processor available, fall back to liveness check only
        return Response(status_code=200)

    try:
        req_id = f"{HEALTH_CHECK_RID_PREFIX}_{time.time()}"

        dummy_request = {
            "mm_items": mm_items,
            "modality": modality.name,
            "req_id": req_id,
            "num_parts": 1,
            "part_idx": 0,
        }

        # Broadcast to other TP ranks so distributed ops stay in sync
        for socket in send_sockets:
            socket.send_pyobj(dummy_request)

        # Run encode on rank 0 with timeout
        _, _, _, error_msg, _ = await asyncio.wait_for(
            encoder.encode(
                mm_items=mm_items,
                modality=modality,
                req_id=req_id,
                num_parts=1,
                part_idx=0,
            ),
            timeout=HEALTH_CHECK_TIMEOUT,
        )

        # Clean up stored embedding
        encoder.embedding_to_send.pop(req_id, None)

        if error_msg:
            logger.error(f"Encoder health check failed: {error_msg}")
            return Response(status_code=503)

        return Response(status_code=200)

    except asyncio.TimeoutError:
        logger.error(f"Encoder health check timed out after {HEALTH_CHECK_TIMEOUT}s")
        return Response(status_code=503)
    except Exception as e:
        logger.error(f"Encoder health check failed: {e}")
        return Response(status_code=503)


@app.api_route("/start_profile", methods=["GET", "POST"])
async def start_profile_async(obj: Optional[ProfileReqInput] = None):
    if encoder is None:
        return Response(content="encoder not ready\n", status_code=503)
    req = None
    if obj is None:
        req = ProfileReq(ProfileReqType.START_PROFILE)
    else:
        req = ProfileReq(
            type=ProfileReqType.START_PROFILE,
            output_dir=obj.output_dir,
            start_step=obj.start_step,
            num_steps=obj.num_steps,
            activities=obj.activities,
            with_stack=obj.with_stack,
            record_shapes=obj.record_shapes,
            profile_by_stage=obj.profile_by_stage,
            profile_id=str(time.time()),
            merge_profiles=obj.merge_profiles,
            profile_prefix=obj.profile_prefix,
            profile_stages=obj.profile_stages,
        )
    for socket in send_sockets:
        socket.send_pyobj(req)
    if encoder.profiler is None:
        encoder.profiler = EncoderProfiler(encoder.rank)
    ok, msg = encoder.profiler.start(req)
    if ok:
        detail = (
            f"Start profiling. output_dir={encoder.profiler.output_dir} "
            f"profile_id={encoder.profiler.profile_id}\n"
        )
        return Response(content=detail, status_code=200)
    return Response(
        content=(msg or "Start profiling failed.\n"), status_code=HTTPStatus.BAD_REQUEST
    )


@app.api_route("/stop_profile", methods=["GET", "POST"])
async def stop_profile_async():
    if encoder is None:
        return Response(content="encoder not ready\n", status_code=503)
    if encoder.profiler is None:
        return Response(
            content="profiling not initialized\n", status_code=HTTPStatus.BAD_REQUEST
        )
    req = ProfileReq(ProfileReqType.STOP_PROFILE)
    for socket in send_sockets:
        socket.send_pyobj(req)
    ok, msg = encoder.profiler.stop()
    if ok:
        return Response(content="Stop profiling.\n", status_code=200)
    return Response(
        content=(msg or "Stop profiling failed.\n"), status_code=HTTPStatus.BAD_REQUEST
    )
