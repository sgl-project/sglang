import asyncio
import concurrent.futures
import contextlib
import copy
import ctypes
import functools
import logging
import multiprocessing as mp
import os
import pickle
import threading
import time
import traceback
import uuid
from collections import defaultdict
from http import HTTPStatus
from typing import Annotated, Dict, List, Optional, Set, Tuple, Union

import aiohttp
import numpy as np
import requests as http_requests
import torch
import uvicorn
import zmq
import zmq.asyncio
from fastapi import Body, FastAPI
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
from sglang.srt.managers.io_struct import (
    ProfileReq,
    ProfileReqType,
    async_sock_recv,
    async_sock_send,
    sock_send,
    wrap_as_pickle,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.model_loader import get_model
from sglang.srt.multimodal.processors.qwen_vl import preprocess_video
from sglang.srt.observability.metrics_collector import EncoderMetricsCollector
from sglang.srt.observability.req_time_stats import EncoderReqTimeStats
from sglang.srt.observability.trace import (
    process_tracing_init,
    trace_set_thread_info,
)
from sglang.srt.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import (
    add_prometheus_middleware,
    configure_logger,
    load_audio,
    load_image,
    load_video,
    random_uuid,
    set_prometheus_multiproc_dir,
)
from sglang.srt.utils.common import configure_logger, maybe_reindex_device_id
from sglang.srt.utils.network import (
    NetworkAddress,
    config_socket,
    get_free_port,
    get_local_ip_auto,
    get_zmq_socket,
)

logger = logging.getLogger(__name__)

HEALTH_CHECK_TIMEOUT = 30

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


def _normalize_aux_value(val):
    """Normalize aux values to pickle types compatible with safe_pickle_loads.

    HF multimodal processors (e.g. Qwen3-VL/Omni) emit numpy arrays for
    fields like ``video_timestamps`` / ``second_per_grid_ts``. ``numpy.*`` is
    not in SafeUnpickler's allowlist, so the receiver would refuse to load
    those payloads. Convert numpy values to torch tensors (numeric) or plain
    Python lists (object dtype) before pickling.
    """
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        if val.dtype == object:
            return val.tolist()
        return torch.from_numpy(np.ascontiguousarray(val))
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, (list, tuple)):
        return type(val)(_normalize_aux_value(v) for v in val)
    if isinstance(val, dict):
        return {k: _normalize_aux_value(v) for k, v in val.items()}
    return val


def _build_mm_aux_data(mm_inputs, model_type=None):
    # Video aux metadata, scoped to model_type's video-meta attrs.
    return {
        attr: _normalize_aux_value(mm_inputs.get(attr))
        for attr in video_meta_attrs_for(model_type)
    }


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
        # DP rank for metric labels; overridden by run_dp_worker in DP mode.
        # 0 in the single-instance (non-DP) path.
        self.dp_rank = 0
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
        # Dedicated executor for image preprocessing (resize/normalize).
        # Separate from self.executor (ZMQ sends) to avoid contention under high concurrency.
        self.preproc_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=envs.SGLANG_ENCODER_PREPROC_WORKERS.get()
        )

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

        # Pre-compute embedding metadata (needed by all ranks for mooncake)
        if self.server_args.encoder_transfer_backend == "mooncake":
            self._embedding_dims = self._infer_embedding_dims()
            self._embedding_dtype = next(self.model.parameters()).dtype
            self._element_size = torch.tensor(
                [], dtype=self._embedding_dtype
            ).element_size()

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
            # Need to ensure the NCCL launch order on rank0 matches the dispatch order rank>0
            self.encode_dispatch_lock = asyncio.Lock()

            # Async mooncake state: track background VIT forward completion
            if self.server_args.encoder_transfer_backend == "mooncake":
                self._forward_ready_events: Dict[str, asyncio.Event] = {}
                self._forward_results: Dict[str, dict] = {}
                # when multiple decoder TP ranks call
                # POST /encode with the same req_id, only the first triggers
                # _run_forward(); subsequent callers wait on the event and
                # return the cached metadata.
                self._inflight_encode_lock = asyncio.Lock()
                self._inflight_encode_events: Dict[str, asyncio.Event] = {}
                self._inflight_encode_meta: Dict[str, Tuple] = {}
                self._inflight_encode_cleanup_tasks: Dict[str, asyncio.Task] = {}

        # Bind unified encode entry point based on backend and cache config
        if self.mm_global_cache is not None:
            if self.server_args.encoder_transfer_backend == "mooncake":
                self._encode_fn = self.encode_with_global_cache_mooncake
            else:
                self._encode_fn = self.encode_with_global_cache
        else:
            if self.server_args.encoder_transfer_backend == "mooncake":
                self._encode_fn = self.encode_with_mooncake
            else:
                self._encode_fn = self.encode

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

    def _encode_missing(
        self,
        mm_feature,
        mm_inputs: dict,
        indices: List[int],
        modality: Modality = Modality.IMAGE,
        get_feature_fn=None,
        grid_thw: Optional[List] = None,
        keep_on_gpu: bool = False,
    ) -> List[torch.Tensor]:
        """
        GPU Task: Run ViT inference ONLY on the subset of mm items missing from the cache.
        """
        if grid_thw is None:
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

        forward_start = time.perf_counter()
        with torch.inference_mode():
            new_embeddings = get_feature_fn([mm_item])
            if not keep_on_gpu:
                new_embeddings = new_embeddings.cpu()
            if new_embeddings.ndim != 2:
                new_embeddings = new_embeddings.reshape(-1, new_embeddings.shape[-1])
        if encoder_metrics_collector is not None:
            encoder_metrics_collector.observe_model_forward(
                time.perf_counter() - forward_start, modality=modality.name.lower()
            )

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
            new_slices = self._encode_missing(
                mm_feature, mm_inputs, missing_indices, modality, get_feature_fn
            )

        # Step 3: Rank 0 prefetches cache-hit embeddings and builds fallback_mask.
        fallback_mask = torch.zeros(num_items, dtype=torch.int32)
        cached_slices = []

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

                    # Prefetch IO completed; check which items actually loaded.
                    cached_slices = self.mm_global_cache.get_embeddings(hit_hashes)
                    for i, idx in enumerate(hit_indices):
                        if cached_slices[i] is None:
                            fallback_mask[idx] = 1
                    num_partial_fail = int(fallback_mask.sum().item())
                    if num_partial_fail > 0:
                        logger.warning(
                            f"Req {req_id}: {num_partial_fail}/{len(hit_indices)} "
                            f"cache-hit items failed to load (pool full), "
                            f"falling back to ViT"
                        )
                except (asyncio.TimeoutError, Exception) as e:
                    logger.error(
                        f"Prefetch failed for req {req_id}: {e}. "
                        f"Falling back to ViT for {len(hit_indices)} hit items."
                    )
                    for idx in hit_indices:
                        fallback_mask[idx] = 1

        # Step 4: Broadcast fallback_mask to all ranks so they stay in sync.
        if self.server_args.tp_size > 1:
            torch.distributed.broadcast(
                fallback_mask,
                src=0,
                group=self.mm_global_cache.prefetch_tp_group,
            )

        # Step 5: All ranks run ViT for items that need fallback recomputation.
        fallback_indices = [i for i in range(num_items) if fallback_mask[i].item() == 1]
        fallback_slices = None
        if fallback_indices:
            logger.info(
                f"Req {req_id}: All ranks running ViT fallback "
                f"for {len(fallback_indices)} items."
            )
            fallback_slices = self._encode_missing(
                mm_feature, mm_inputs, fallback_indices, modality, get_feature_fn
            )

        # Step 6: Rank 0 assembles final embedding and prepares for sending.
        if self.rank == 0:
            final_slices = [None] * num_items

            for i, idx in enumerate(missing_indices):
                final_slices[idx] = new_slices[i]

            # Fill in successfully loaded cache-hit embeddings
            if cached_slices:
                for i, idx in enumerate(hit_indices):
                    if cached_slices[i] is not None:
                        final_slices[idx] = cached_slices[i]

            # Fill in ViT fallback results for failed items
            if fallback_slices is not None:
                for i, idx in enumerate(fallback_indices):
                    final_slices[idx] = fallback_slices[i]

            mm_embedding = torch.cat(final_slices, dim=0)

            # Release embedding cache references now that torch.cat has
            # copied the data into a new tensor.  This allows the cache
            # entries to be evicted under memory pressure.
            if cached_slices:
                loaded_hashes = [
                    str_mm_hashes[idx]
                    for idx in hit_indices
                    if fallback_mask[idx].item() == 0
                ]
                if loaded_hashes:
                    self.mm_global_cache.release_embeddings(loaded_hashes)

            # Background insert: store newly computed embeddings into global cache.
            # Includes both original misses and fallback-recomputed hits.
            all_new_hashes = [str_mm_hashes[i] for i in missing_indices]
            all_new_slices = list(new_slices)
            if fallback_slices is not None:
                all_new_hashes += [str_mm_hashes[i] for i in fallback_indices]
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
            if self.profiler is not None:
                self.profiler.step()
            return (
                mm_embedding.nbytes,
                mm_embedding.shape[0],
                mm_embedding.shape[1],
                None,
                None,
            )
        else:
            if self.profiler is not None:
                self.profiler.step()
            return (0, 0, 0, None, None)

    async def encode_with_global_cache_mooncake(
        self,
        mm_items,
        modality: Modality,
        req_id: str,
        num_parts: int,
        part_idx: int,
        hashes: Optional[List[str]] = None,
    ):
        """Async encode with global cache for mooncake backend.
        All ranks participate in VIT forward; tp_size > 1 adds broadcasts for sync."""
        try:
            mm_inputs, get_feature_fn = await self._process_mm_items(mm_items, modality)
            grid_thw = _get_mm_grid_dim(mm_inputs, modality, self.model_type)
            mm_feature = _convert(_get_mm_feature(mm_inputs, modality))
            num_items = len(grid_thw)
            aux_data = _build_mm_aux_data(mm_inputs)

            # Setup metadata and event management
            nbytes, total_tokens, embedding_dim, event = (
                self._setup_mooncake_async_encode(
                    req_id, num_parts, part_idx, grid_thw, modality, aux_data
                )
            )

            # Rank 0: compute hashes
            if self.rank == 0:
                if hashes is None:
                    mm_hashes = self._calculate_hashes_from_features(
                        mm_feature, grid_thw, modality
                    )
                else:
                    mm_hashes = hashes
                str_mm_hashes = [str(h) for h in mm_hashes]

            # All ranks: launch background task for cache check + VIT forward.
            # Do NOT use run_in_executor: get_feature_fn relies on a session
            # context (CUDA / SGLang inference session) that is bound to the
            # event-loop main thread and is NOT available inside a
            # ThreadPoolExecutor worker thread.
            async def _run_forward_with_cache():
                try:
                    # Step 1: Rank 0 checks cache, broadcast mask if TP > 1
                    if self.rank == 0:
                        exist_mask = await self.mm_global_cache.batch_is_exist(
                            str_mm_hashes
                        )
                        mask_tensor = torch.tensor(
                            [1 if e else 0 for e in exist_mask],
                            dtype=torch.int32,
                        )
                    else:
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
                    final_slices = [None] * num_items

                    # Step 2: All ranks run VIT forward for cache misses
                    # (runs in event loop to preserve session context)
                    new_slices = []
                    if missing_indices:
                        new_slices = self._encode_missing(
                            mm_feature,
                            mm_inputs,
                            missing_indices,
                            modality,
                            get_feature_fn,
                            grid_thw,
                            keep_on_gpu=True,
                        )

                    # Step 3: Rank 0 prefetches cache-hit embeddings and builds fallback_mask.
                    fallback_mask = torch.zeros(num_items, dtype=torch.int32)
                    cached_slices = []

                    if self.rank == 0 and hit_indices:
                        hit_hashes = [str_mm_hashes[i] for i in hit_indices]
                        hit_tokens = [
                            self.get_num_tokens(grid_thw[i], modality)
                            for i in hit_indices
                        ]
                        self.mm_global_cache.prefetch(
                            req_id, hit_hashes, hit_tokens, modality
                        )
                        try:

                            async def _wait_prefetch():
                                while not self.mm_global_cache.check_prefetch_progress(
                                    req_id
                                ):
                                    await asyncio.sleep(0.005)

                            await asyncio.wait_for(_wait_prefetch(), timeout=60.0)

                            cached_slices = self.mm_global_cache.get_embeddings(
                                hit_hashes
                            )
                            for i, idx in enumerate(hit_indices):
                                if cached_slices[i] is None:
                                    fallback_mask[idx] = 1
                            num_partial_fail = int(fallback_mask.sum().item())
                            if num_partial_fail > 0:
                                logger.warning(
                                    f"Req {req_id}: {num_partial_fail}/{len(hit_indices)} "
                                    f"cache-hit items failed to load (pool full), "
                                    f"falling back to ViT"
                                )
                        except (asyncio.TimeoutError, Exception) as e:
                            logger.error(
                                f"Prefetch failed for {req_id}: {e}. "
                                f"Falling back to ViT for "
                                f"{len(hit_indices)} hit items."
                            )
                            for idx in hit_indices:
                                fallback_mask[idx] = 1

                    # Step 4: Broadcast fallback_mask to all ranks so they stay in sync.
                    if self.server_args.tp_size > 1:
                        torch.distributed.broadcast(
                            fallback_mask,
                            src=0,
                            group=self.mm_global_cache.prefetch_tp_group,
                        )

                    # Step 5: All ranks run ViT for items that need fallback recomputation.
                    fallback_indices = [
                        i for i in range(num_items) if fallback_mask[i].item() == 1
                    ]
                    fallback_slices = None
                    if fallback_indices:
                        logger.info(
                            f"Req {req_id}: All ranks running ViT fallback "
                            f"for {len(fallback_indices)} items."
                        )
                        fallback_slices = self._encode_missing(
                            mm_feature,
                            mm_inputs,
                            fallback_indices,
                            modality,
                            get_feature_fn,
                            grid_thw,
                            keep_on_gpu=True,
                        )

                    # Step 6: Rank 0 assembles final embedding.
                    if self.rank == 0:
                        for i, idx in enumerate(missing_indices):
                            final_slices[idx] = new_slices[i]

                        # Fill in successfully loaded cache-hit embeddings
                        if cached_slices:
                            for i, idx in enumerate(hit_indices):
                                if cached_slices[i] is not None:
                                    final_slices[idx] = cached_slices[i]

                        # Fill in ViT fallback results for failed items
                        if fallback_slices is not None:
                            for i, idx in enumerate(fallback_indices):
                                final_slices[idx] = fallback_slices[i]

                        # Move cached CPU slices to GPU and match model dtype
                        device = torch.device(f"cuda:{self.gpu_id}")
                        final_slices = [
                            (
                                s.to(device=device, dtype=self._embedding_dtype)
                                if s.device.type == "cpu"
                                else s
                            )
                            for s in final_slices
                        ]
                        mm_embedding = torch.cat(final_slices, dim=0)
                        # Wait for any pending VIT / cat kernels to finish
                        # before publishing to /send: mooncake transfer_sync
                        # is a host-side RDMA read that bypasses CUDA streams
                        # and would otherwise race with in-flight kernels.
                        torch.cuda.current_stream(mm_embedding.device).synchronize()

                        # Release cache refs after data is copied to GPU
                        if cached_slices:
                            loaded_hashes = [
                                str_mm_hashes[idx]
                                for idx in hit_indices
                                if fallback_mask[idx].item() == 0
                            ]
                            if loaded_hashes:
                                self.mm_global_cache.release_embeddings(loaded_hashes)

                        # Background insert: store newly computed embeddings into global cache.
                        # Includes both original misses and fallback-recomputed hits.
                        all_new_hashes = [str_mm_hashes[i] for i in missing_indices]
                        all_new_slices = list(new_slices)
                        if fallback_slices is not None:
                            all_new_hashes += [
                                str_mm_hashes[i] for i in fallback_indices
                            ]
                            all_new_slices += list(fallback_slices)
                        if all_new_hashes:

                            async def _background_insert():
                                await asyncio.to_thread(
                                    self.mm_global_cache.insert_batch,
                                    all_new_hashes,
                                    all_new_slices,
                                )

                            insert_task = asyncio.create_task(_background_insert())
                            self.background_tasks.add(insert_task)
                            insert_task.add_done_callback(self.background_tasks.discard)

                        self._forward_results[req_id]["embedding"] = mm_embedding
                        logger.info(
                            f"Global cache + VIT forward completed for "
                            f"{req_id}, shape={mm_embedding.shape}"
                        )
                except Exception as e:
                    logger.error(
                        f"Global cache + VIT forward failed for " f"{req_id}: {e}"
                    )
                    if self.rank == 0:
                        self._forward_results[req_id]["error"] = str(e)
                finally:
                    if self.rank == 0:
                        event.set()
                    if self.profiler is not None:
                        self.profiler.step()

            self._launch_mooncake_background_task(_run_forward_with_cache())

            if self.rank == 0:
                logger.info(
                    f"Returning metadata immediately for {req_id}, "
                    f"global cache + VIT forward running async"
                )

            return (nbytes, total_tokens, embedding_dim, None, None)

        except Exception as e:
            error_code = getattr(e, "code", HTTPStatus.INTERNAL_SERVER_ERROR)
            error_msg = str(e)
            logger.error(
                f"Rank {self.rank} encode_with_global_cache_mooncake "
                f"failed: {error_msg} {error_code = }"
            )
            return self._handle_mooncake_encode_error(
                req_id, num_parts, part_idx, modality, error_msg, error_code
            )

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

        preprocess_start = time.perf_counter()
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
        if encoder_metrics_collector is not None:
            encoder_metrics_collector.observe_preprocess(
                time.perf_counter() - preprocess_start, modality=modality.name.lower()
            )

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

    async def _encode(self, mm_items, modality: Modality) -> torch.Tensor:
        modality_str = modality.name.lower()
        try:
            # preprocess latency is observed inside _process_mm_items so all
            # callers (encode / batch_encode / global-cache) are covered.
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

            cache_hit = False
            if self.server_args.enable_prefix_mm_cache:
                mm_item.set_pad_value()
                mm_hash = MultiModalStaticCache.combine_hashes([mm_item.hash])
                async with self.mm_cache_lock:
                    mm_cache = self.mm_cache.get([mm_item.hash])
                    if mm_cache is not None:
                        mm_embedding = mm_cache.embedding
                        cache_hit = True

            if mm_embedding is None:
                forward_start = time.perf_counter()
                with torch.inference_mode():
                    mm_embedding: torch.Tensor = get_feature_fn([mm_item])
                    mm_embedding = mm_embedding.cpu()
                if len(mm_embedding.shape) != 2:
                    mm_embedding = mm_embedding.reshape(-1, mm_embedding.shape[-1])
                if encoder_metrics_collector is not None:
                    encoder_metrics_collector.observe_model_forward(
                        time.perf_counter() - forward_start, modality=modality_str
                    )

            # Per-request cache hit metrics: tokens = embedding rows, files = 1 item.
            if (
                self.server_args.enable_prefix_mm_cache
                and encoder_metrics_collector is not None
            ):
                total_tokens = int(mm_embedding.shape[0])
                hit_tokens = total_tokens if cache_hit else 0
                encoder_metrics_collector.record_cache_tokens(
                    hit_tokens, total_tokens, modality=modality_str
                )
                encoder_metrics_collector.record_cache_files(
                    1 if cache_hit else 0, 1, modality=modality_str
                )

            if self.server_args.enable_prefix_mm_cache:
                async with self.mm_cache_lock:
                    entries_before = len(self.mm_cache)
                    already_present = self.mm_cache.has(mm_hash)
                    inserted = self.mm_cache.set(
                        mm_hash, EmbeddingResult(embedding=mm_embedding)
                    )
                    entries_after = len(self.mm_cache)
                    if encoder_metrics_collector is not None:
                        added = 0 if already_present else (1 if inserted else 0)
                        evictions = max(0, added - (entries_after - entries_before))
                        if evictions > 0:
                            encoder_metrics_collector.inc_cache_evictions(
                                modality=modality_str, count=evictions
                            )
                        encoder_metrics_collector.set_cache_state(
                            self.mm_cache.current_size, entries_after
                        )
            if self.profiler is not None:
                self.profiler.step()

            aux_data = _build_mm_aux_data(mm_inputs, self.model_type)

            if modality == Modality.VIDEO and mm_inputs.get("video_audio_features"):
                target = (
                    self.model.thinker if hasattr(self.model, "thinker") else self.model
                )
                encode_video_audio_fn = getattr(target, "encode_video_audio", None)
                if encode_video_audio_fn is not None:
                    audio_forward_start = time.perf_counter()
                    audio_embedding = encode_video_audio_fn(mm_inputs)
                    if encoder_metrics_collector is not None:
                        encoder_metrics_collector.observe_model_forward(
                            time.perf_counter() - audio_forward_start, modality="audio"
                        )
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
            # Wait for async VIT forward completion if needed
            req_id = mm_data.req_id
            if req_id in self._forward_ready_events:
                await self._forward_ready_events[req_id].wait()
                result = self._forward_results.get(req_id)
                if result is not None:
                    if "error" in result:
                        raise InternalError(f"VIT forward failed: {result['error']}")
                    embedding = result["embedding"]
                    # Cache the embedding on mm_data so subsequent /send calls
                    # from other decoder TP ranks can reuse it.
                    mm_data.cached_embedding = embedding

            # Retrieve cached embedding for duplicate /send calls from other
            # decoder TP ranks.
            if embedding is None:
                embedding = mm_data.cached_embedding
            if embedding is None:
                raise InternalError(
                    f"No embedding available for Mooncake GPU-direct transfer: {req_id}"
                )

            expected_nbytes = mm_data.shape[0] * mm_data.shape[1] * self._element_size
            assert embedding.nbytes == expected_nbytes, (
                f"Embedding size mismatch for {req_id}: "
                f"actual={embedding.nbytes}, expected={expected_nbytes} "
                f"(shape={mm_data.shape}, element_size={self._element_size})"
            )

            # MR was registered once in _run_forward and is shared across all
            # sibling-TP /send calls;
            mr_already_registered = (
                self._forward_results.get(req_id, {}).get("mr_ptr")
                == embedding.data_ptr()
            )
            if not mr_already_registered:
                self.engine.register(embedding.data_ptr(), embedding.nbytes)
            _t_xfer_start = time.monotonic()
            await asyncio.to_thread(
                self.engine.transfer_sync,
                session_id,
                embedding.data_ptr(),
                buffer_address,
                embedding.nbytes,
            )
            xfer_ms = (time.monotonic() - _t_xfer_start) * 1000.0
            if encoder_metrics_collector is not None:
                encoder_metrics_collector.observe_transfer(
                    xfer_ms / 1000.0, backend="mooncake"
                )
            if not mr_already_registered:
                self.engine.deregister(embedding.data_ptr())
            # Only emit at INFO when transfer is slow or fell back
            # to per-/send register;
            if xfer_ms > 200.0 or not mr_already_registered:
                logger.info(
                    f"[{req_id}] mooncake transfer_sync={xfer_ms:.1f}ms "
                    f"nbytes={embedding.nbytes} shared_mr={mr_already_registered}"
                )

            mm_data.embedding = None

        # Send ack/data
        if url is not None:
            endpoint = NetworkAddress.parse(url).to_tcp()
        else:
            endpoint = NetworkAddress(prefill_host, embedding_port).to_tcp()
        logger.info(f"{endpoint = }")

        # Serialize data
        if self.server_args.encoder_transfer_backend == "mooncake":
            # Mooncake already pushed the embedding via RDMA;
            new_mm_data = mm_data.copy_without_embedding()
            serialized_data = pickle.dumps(new_mm_data)
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
                sock.close(linger=5000)

        _zmq_xfer_start = time.perf_counter()
        await asyncio.get_event_loop().run_in_executor(self.executor, send_with_socket)
        if (
            encoder_metrics_collector is not None
            and self.server_args.encoder_transfer_backend != "mooncake"
        ):
            encoder_metrics_collector.observe_transfer(
                time.perf_counter() - _zmq_xfer_start,
                backend=self.server_args.encoder_transfer_backend,
            )

    async def encode(
        self, mm_items, modality: Modality, req_id, num_parts, part_idx, hashes=None
    ):
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

    def _setup_mooncake_async_encode(
        self,
        req_id: str,
        num_parts: int,
        part_idx: int,
        grid_thw,
        modality: Modality,
        aux_data: dict,
    ):
        """Setup metadata and event management for mooncake async encode.
        Returns (nbytes, total_tokens, embedding_dim, event)."""
        total_tokens = sum(self.get_num_tokens(g, modality) for g in grid_thw)
        embedding_dim = self._embedding_dims[modality]
        nbytes = total_tokens * embedding_dim * self._element_size

        event = None
        if self.rank == 0:
            mm_data = EmbeddingData(
                req_id,
                num_parts,
                part_idx,
                grid_thw,
                modality,
                embedding=None,
                embedding_shape=[total_tokens, embedding_dim],
                **aux_data,
            )
            self.embedding_to_send[req_id] = mm_data
            event = asyncio.Event()
            self._forward_ready_events[req_id] = event
            self._forward_results[req_id] = {}

        return nbytes, total_tokens, embedding_dim, event

    def _handle_mooncake_encode_error(
        self, req_id, num_parts, part_idx, modality, error_msg, error_code
    ):
        """Handle outer exception for mooncake async encode methods."""
        if self.rank == 0:
            if req_id in self._forward_ready_events:
                self._forward_results[req_id] = {"error": error_msg}
                self._forward_ready_events[req_id].set()
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
        return 0, 0, 0, error_msg, error_code

    def _launch_mooncake_background_task(self, coro):
        """Launch an async background task and track it."""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task

    async def _cleanup_inflight_encode_state(self, req_id: str):
        if not hasattr(self, "_inflight_encode_events"):
            return
        async with self._inflight_encode_lock:
            self._inflight_encode_events.pop(req_id, None)
            self._inflight_encode_meta.pop(req_id, None)
            task = self._inflight_encode_cleanup_tasks.pop(req_id, None)
            if task is not None and not task.done():
                task.cancel()
        # Also clean up embedding data and forward state
        mm_data = self.embedding_to_send.pop(req_id, None)
        if mm_data is not None:
            mm_data.cached_embedding = None
        # Release the rkey after all /send calls have completed.
        forward_state = self._forward_results.pop(req_id, None)
        if forward_state is not None:
            mr_ptr = forward_state.get("mr_ptr")
            if mr_ptr is not None:
                try:
                    self.engine.deregister(mr_ptr)
                except Exception as dereg_err:
                    logger.warning(
                        f"Shared-MR deregister failed for {req_id}: {dereg_err}"
                    )
        self._forward_ready_events.pop(req_id, None)

    def _schedule_inflight_encode_cleanup(self, req_id: str):
        if not hasattr(self, "_inflight_encode_events"):
            return

        async def _cleanup_later():
            await asyncio.sleep(self.send_timeout)
            await self._cleanup_inflight_encode_state(req_id)

        old_task = self._inflight_encode_cleanup_tasks.pop(req_id, None)
        if old_task is not None and not old_task.done():
            old_task.cancel()
        task = asyncio.create_task(_cleanup_later())
        self._inflight_encode_cleanup_tasks[req_id] = task
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def encode_with_mooncake(
        self, mm_items, modality: Modality, req_id, num_parts, part_idx, hashes=None
    ):
        """Async encode for mooncake: all ranks participate in VIT forward via background task,
        rank 0 returns metadata immediately."""
        try:
            mm_inputs, get_feature_fn = await self._process_mm_items(mm_items, modality)
            grid_thw = _get_mm_grid_dim(mm_inputs, modality, self.model_type)
            aux_data = _build_mm_aux_data(mm_inputs)

            # Setup metadata and event management
            nbytes, total_tokens, embedding_dim, event = (
                self._setup_mooncake_async_encode(
                    req_id, num_parts, part_idx, grid_thw, modality, aux_data
                )
            )

            # Build mm_item (all ranks)
            mm_item = MultimodalDataItem.from_dict(
                {
                    "modality": modality,
                    "feature": _convert(_get_mm_feature(mm_inputs, modality)),
                }
            )
            for k, v in mm_inputs.items():
                if k in _mm_feature_attrs.get(modality, []):
                    continue
                val = _convert(v)
                mm_item.set(k, val)

            async def _run_forward():
                try:
                    with torch.inference_mode():
                        emb = get_feature_fn([mm_item])
                        if len(emb.shape) != 2:
                            emb = emb.reshape(-1, emb.shape[-1])
                        # mooncake's transfer_sync is a host-side
                        # RDMA read that bypasses the CUDA stream. Without an
                        # explicit sync here, sibling-TP /send handlers can
                        # invoke transfer_sync while VIT kernels are still
                        # writing `emb`, producing partial / garbage data on
                        # the receiver side
                        if emb.is_cuda:
                            torch.cuda.current_stream(emb.device).synchronize()
                    if self.rank == 0:
                        # Register the MR exactly once here so all sibling-TP /send coroutines share a single registration.
                        try:
                            self.engine.register(emb.data_ptr(), emb.nbytes)
                            self._forward_results[req_id]["mr_ptr"] = emb.data_ptr()
                        except Exception as reg_err:
                            logger.warning(
                                f"Shared-MR register failed for {req_id}, "
                                f"falling back to per-/send register: {reg_err}"
                            )
                            self._forward_results[req_id]["mr_ptr"] = None
                        self._forward_results[req_id]["embedding"] = emb
                except Exception as e:
                    logger.error(f"VIT forward failed for {req_id}: {e}")
                    if self.rank == 0:
                        self._forward_results[req_id]["error"] = str(e)
                finally:
                    if self.rank == 0:
                        event.set()
                    if self.profiler is not None:
                        self.profiler.step()

            self._launch_mooncake_background_task(_run_forward())

            if self.rank == 0:
                logger.info(
                    f"Returning metadata immediately for {req_id}, "
                    f"VIT forward running async"
                )

            return (nbytes, total_tokens, embedding_dim, None, None)

        except Exception as e:
            error_code = getattr(e, "code", HTTPStatus.INTERNAL_SERVER_ERROR)
            error_msg = str(e)
            logger.error(
                f"Rank {self.rank} encode_with_mooncake failed: "
                f"{error_msg} {error_code = }",
                exc_info=True,
            )
            return self._handle_mooncake_encode_error(
                req_id, num_parts, part_idx, modality, error_msg, error_code
            )

    async def encode_request(self, req: dict, modality: Modality):
        """Single-request encode dispatcher.

        Delegates to ``self._encode_fn``, which is bound at ``__init__``
        time to the correct variant (cache / no-cache / mooncake).
        """
        return await self._encode_fn(
            mm_items=req["mm_items"],
            modality=modality,
            req_id=req["req_id"],
            num_parts=req["num_parts"],
            part_idx=req["part_idx"],
            hashes=req.get("hashes"),
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

        if encoder_metrics_collector is not None:
            modality_str = modality.name.lower()
            for n in items_per_req:
                encoder_metrics_collector.observe_mm_items_per_request(
                    n, modality=modality_str
                )
            encoder_metrics_collector.observe_mm_items_per_batch(
                total, modality=modality_str
            )

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

            final_slices = self._encode_missing(
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
        self.pending_queue: asyncio.Queue[PendingRequest] = asyncio.Queue()
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
        modality_str = modality.name.lower()
        if encoder_metrics_collector is not None:
            for p in group:
                encoder_metrics_collector.observe_queue_wait(
                    max(0.0, start - p.submit_time), modality=modality_str
                )
        for sock in self.send_sockets:
            sock_send(
                sock,
                wrap_as_pickle(
                    {
                        "type": "batch_encode",
                        "modality": modality.name,
                        "requests": requests,
                        "enter_time": start,
                    }
                ),
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
        modality_str = modality.name.lower()
        for p in group:
            req = p.request
            try:
                start = time.time()
                if encoder_metrics_collector is not None:
                    encoder_metrics_collector.observe_queue_wait(
                        max(0.0, start - p.submit_time), modality=modality_str
                    )
                    # Count like batch_encode: flatten nested items and expand
                    # per-leaf grids so {"type": "image", "image": [p1, p2, ...]}
                    # counts as N, not 1.
                    leaves = MMEncoder._flatten_nested_items(req.get("mm_items", []))
                    mm_count = sum(self.encoder._grid_count_per_leaf(leaves, modality))
                    encoder_metrics_collector.observe_mm_items_per_request(
                        mm_count, modality=modality_str
                    )
                    encoder_metrics_collector.observe_mm_items_per_batch(
                        mm_count, modality=modality_str
                    )
                for sock in self.send_sockets:
                    sock_send(sock, wrap_as_pickle(req))
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

# Per-process encoder metrics collector. Set in launch_server (non-DP) and in
# run_dp_worker (DP mode, with the worker's dp_rank). None when metrics disabled.
encoder_metrics_collector: Optional[EncoderMetricsCollector] = None

# DP mode (--dp-size > 1): each rank runs as a subprocess with its own
# MMEncoder on its own GPU; the main process only routes via ZMQ so the
# asyncio event loop is never blocked by GPU work.
dp_dispatcher: Optional["DPDispatcher"] = None


async def _push_embedding_to_prefill(enc: MMEncoder, request: dict) -> None:
    # No-op for mooncake (its /send is separate). embedding_port=None is
    # rejected upfront, so ports is always a concrete list here.
    req_id = request["req_id"]
    backend = enc.server_args.encoder_transfer_backend

    if backend == "zmq_to_tokenizer":
        await enc.send(
            req_id=req_id,
            prefill_host=request["prefill_host"],
            embedding_port=request["embedding_port"],
        )
        enc.embedding_to_send.pop(req_id, None)
        return

    if backend == "zmq_to_scheduler":
        ports = request["embedding_port"]
        assert isinstance(ports, list)
        await asyncio.gather(
            *(
                enc.send(
                    req_id=req_id,
                    prefill_host=request["prefill_host"],
                    embedding_port=p,
                )
                for p in ports
            )
        )
        enc.embedding_to_send.pop(req_id, None)


async def _dp_worker_encode_and_send(
    enc: MMEncoder,
    sched: Optional[EncoderScheduler],
    request: dict,
) -> Optional[dict]:
    # Mooncake returns metadata for main to forward; zmq inlines the send.
    # Soft errors raise MMError so the dispatcher route maps them to HTTP.
    req_id = request["req_id"]
    time_stats_json = request.pop("time_stats_json", None)
    time_stats = EncoderReqTimeStats()
    if time_stats_json:
        time_stats.decode_json(time_stats_json)
    request["enter_time"] = time.time()
    modality = Modality.from_str(request["modality"])
    time_stats.modality = modality.name.lower()
    time_stats.set_metrics_collector(encoder_metrics_collector)
    backend = enc.server_args.encoder_transfer_backend

    # URL state lives in main process module globals; workers don't see it.
    if backend == "zmq_to_scheduler" and request.get("embedding_port") is None:
        raise MMError(
            "Encoder DP mode does not support zmq_to_scheduler with "
            "embedding_port=None (URL state isn't synchronised to workers). "
            "Provide an explicit embedding_port list, switch to mooncake / "
            "zmq_to_tokenizer, or run without --dp-size.",
            code=HTTPStatus.BAD_REQUEST,
        )

    time_stats.set_mm_encode_start_time()
    encode_coro = (
        sched.submit(request)
        if sched is not None and modality in _BATCHABLE_MODALITIES
        else enc.encode_request(request, modality)
    )
    try:
        nbytes, embedding_len, embedding_dim, error_msg, error_code = await encode_coro
    except asyncio.TimeoutError:
        time_stats.trace_ctx.abort(abort_info={"reason": "encoder batch timed out"})
        raise

    if error_msg:
        time_stats.trace_ctx.abort(abort_info={"reason": error_msg})
        # zmq backends still forward an error EmbeddingData to P so it
        # doesn't block; send failures here are swallowed.
        try:
            await _push_embedding_to_prefill(enc, request)
        except Exception as e:
            logger.error(
                f"DP error-send failed for req_id={req_id}: {e}", exc_info=True
            )
        # Free the error EmbeddingData stored during encode, or it leaks in
        # embedding_to_send and pins /health into "busy" (a non-empty
        # embedding_to_send reads as busy, skipping the probe). Neither path
        # guarantees cleanup on its own: mooncake's _push_embedding_to_prefill
        # is a no-op, and a swallowed zmq send failure above skips its own pop.
        # zmq lacks the inflight attrs so _cleanup_inflight_encode_state would
        # early-return on it — pop directly. Mirrors the non-DP error path.
        if backend == "mooncake":
            await enc._cleanup_inflight_encode_state(req_id)
        else:
            enc.embedding_to_send.pop(req_id, None)
        raise MMError(error_msg, code=error_code or HTTPStatus.INTERNAL_SERVER_ERROR)

    time_stats.set_mm_encode_end_time()

    if backend == "mooncake":
        request.pop("mm_items", None)
        request.update(
            embedding_size=nbytes,
            embedding_len=embedding_len,
            embedding_dim=embedding_dim,
        )
        # Free the held embedding if the follow-up /send never arrives (same
        # send_timeout cleanup the non-DP path uses).
        enc._schedule_inflight_encode_cleanup(req_id)
        return request

    await _push_embedding_to_prefill(enc, request)
    return None


async def _dp_worker_health_encode(enc: MMEncoder) -> None:
    """Functional health probe run on a DP worker.

    Process-liveness (proc.sentinel) can't see a worker that's alive but
    wedged — hung GPU, NCCL deadlock, stalled ZMQ, or a blocked event loop.
    When idle, run a tiny dummy encode to exercise the VIT forward and surface
    those stalls. No prefill destination: the embedding is discarded, mirroring
    the non-DP /health path. Raises on encode failure so the worker envelope
    carries ``_error`` back to the dispatcher.
    """
    # Busy worker: in-flight traffic already proves liveness, so skip the probe
    # and report healthy — same `embedding_to_send` signal the non-DP /health
    # path uses. A wedged-but-busy worker never reaches here (it can't service
    # the recv), so the dispatcher's broadcast still times out → 503.
    if enc.embedding_to_send:
        return None

    if enc.image_processor is not None:
        mm_items = [f"data:image/png;base64,{MINIMUM_PNG_PICTURE_BASE64}"]
        modality = Modality.IMAGE
    elif enc.audio_processor is not None:
        mm_items = [f"data:audio/wav;base64,{MINIMUM_WAV_SILENCE_BASE64}"]
        modality = Modality.AUDIO
    else:
        # No processor → can't functionally probe; liveness alone is healthy.
        return None

    # uuid keeps rids unique across workers; a bare time.time() can collide.
    req_id = f"{HEALTH_CHECK_RID_PREFIX}_{uuid.uuid4().hex}"
    try:
        _, _, _, error_msg, error_code = await enc.encode(
            mm_items=mm_items,
            modality=modality,
            req_id=req_id,
            num_parts=1,
            part_idx=0,
        )
    finally:
        # Never leave the dummy embedding sitting in the send map.
        enc.embedding_to_send.pop(req_id, None)

    if error_msg:
        raise MMError(error_msg, code=error_code or HTTPStatus.INTERNAL_SERVER_ERROR)


class DPDispatcher:
    """Routes encode requests across DP ranks by least-pending count."""

    def __init__(
        self,
        dp_size: int,
        dispatch_sockets: List,
        result_socket,
        worker_processes: List[mp.Process],
        enable_metrics: bool = False,
    ):
        self.dp_size = dp_size
        self.dispatch_sockets = dispatch_sockets
        self.result_socket = result_socket
        self.worker_processes = worker_processes
        # Key = req_id for encode/broadcast, req_id + "_send" for mooncake /send.
        self.pending_futures: List[Dict[str, asyncio.Future]] = [
            {} for _ in range(dp_size)
        ]
        self.req_id_to_rank: Dict[str, int] = {}
        self._rr_counter = 0
        self._broadcast_counter = 0
        self._dead_ranks: Set[int] = set()
        # req_id -> monotonic ts a mooncake mapping has waited for its /send.
        self._pending_send_at: Dict[str, float] = {}
        # Set when _result_listener gives up; makes alive_ranks report empty.
        self._listener_failed = False

        # Prometheus gauge: pending requests per DP rank. Lives in the main
        # process (the dispatcher), unlike the per-worker EncoderMetricsCollector.
        self.pending_gauge = None
        if enable_metrics:
            from prometheus_client import Gauge

            self.pending_gauge = Gauge(
                name="sglang:encoder_dp_pending_requests",
                documentation="Number of pending requests per encoder DP rank.",
                labelnames=["dp_rank"],
                multiprocess_mode="mostrecent",
            )

    @property
    def pending_counts(self) -> List[int]:
        return [len(d) for d in self.pending_futures]

    def _update_pending_gauge(self) -> None:
        """Push current pending counts to the Prometheus gauge (absolute set)."""
        if self.pending_gauge is not None:
            for i, c in enumerate(self.pending_counts):
                self.pending_gauge.labels(dp_rank=str(i)).set(c)

    @property
    def alive_ranks(self) -> List[int]:
        # Empty if the result listener died; else ranks not marked dead.
        if self._listener_failed:
            return []
        return [r for r in range(self.dp_size) if r not in self._dead_ranks]

    @property
    def all_ranks_alive(self) -> bool:
        # Strict health (only /health uses this); routing still degrades.
        return len(self.alive_ranks) == self.dp_size

    def start(self) -> None:
        logger.info(f"DP dispatcher started: {self.dp_size} ranks (all remote)")
        asyncio.create_task(self._result_listener())
        asyncio.create_task(self._worker_watchdog())
        asyncio.create_task(self._cleanup_stale_mappings())

    def _drop_pending_and_mapping(self, rank: int, req_id: str) -> None:
        # dispatch / broadcast failure: no follow-up /send expected.
        self.pending_futures[rank].pop(req_id, None)
        self.req_id_to_rank.pop(req_id, None)
        self._update_pending_gauge()

    def _fail_pending_for_rank(self, rank: int, reason: str, error_type: str) -> None:
        # Resolve a rank's outstanding futures with 503 so awaiters don't hang.
        pending = self.pending_futures[rank]
        for key, future in list(pending.items()):
            if not future.done():
                future.set_result(
                    {
                        "req_id": key.removesuffix("_send"),
                        "_dp_type": "send" if key.endswith("_send") else "encode",
                        "content": None,
                        "_error": reason,
                        "_error_type": error_type,
                        "_error_code": int(HTTPStatus.SERVICE_UNAVAILABLE),
                    }
                )
            pending.pop(key, None)
        self._update_pending_gauge()

    def _fail_all_pending(self, reason: str, error_type: str) -> None:
        for rank in range(self.dp_size):
            self._fail_pending_for_rank(rank, reason, error_type)
        self.req_id_to_rank.clear()
        self._pending_send_at.clear()

    @staticmethod
    def _timeout_envelope(req_id: str, dp_type: str, reason: str) -> dict:
        return {
            "req_id": req_id,
            "_dp_type": dp_type,
            "content": None,
            "_error": reason,
            "_error_type": "TimeoutError",
            "_error_code": int(HTTPStatus.GATEWAY_TIMEOUT),
        }

    async def dispatch(self, request: dict) -> dict:
        counts = self.pending_counts
        # Skip ranks whose worker process has died.
        alive_ranks = self.alive_ranks
        if not alive_ranks:
            raise MMError(
                "All encoder DP workers are dead.",
                code=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        min_p = min(counts[r] for r in alive_ranks)
        candidates = [r for r in alive_ranks if counts[r] == min_p]
        rank = candidates[self._rr_counter % len(candidates)]
        self._rr_counter += 1
        req_id = request["req_id"]
        self.req_id_to_rank[req_id] = rank
        future = asyncio.get_running_loop().create_future()
        self.pending_futures[rank][req_id] = future
        self._update_pending_gauge()
        logger.info(
            f"MM-Encoder DP dispatch: req_id={req_id}, "
            f"modality={request.get('modality', 'image')}, "
            f"dp_rank={rank}, pending={self.pending_counts}"
        )

        try:
            await async_sock_send(self.dispatch_sockets[rank], wrap_as_pickle(request))
            # An alive-but-stuck worker (NCCL deadlock etc.) wouldn't trip
            # the watchdog, so bound the wait explicitly.
            return await asyncio.wait_for(future, timeout=ENCODER_REQ_TIMEOUT)
        except asyncio.TimeoutError:
            self._drop_pending_and_mapping(rank, req_id)
            return self._timeout_envelope(
                req_id,
                "encode",
                f"Encoder DP rank={rank} timed out after {ENCODER_REQ_TIMEOUT}s",
            )
        except BaseException:
            self._drop_pending_and_mapping(rank, req_id)
            raise

    async def dispatch_send(self, request: dict) -> dict:
        req_id = request["req_id"]
        # /send arrived → stop tracking it for stale-mapping GC.
        self._pending_send_at.pop(req_id, None)
        if self._listener_failed:
            return {
                "req_id": req_id,
                "_error": "encoder DP result listener stopped; cannot route /send",
                "_error_code": int(HTTPStatus.SERVICE_UNAVAILABLE),
            }
        rank = self.req_id_to_rank.get(req_id)
        if rank is None:
            logger.warning(
                f"MM-Encoder dispatch_send: unknown req_id={req_id}, "
                f"cannot route to worker"
            )
            return {"req_id": req_id, "_error": f"Unknown req_id: {req_id}"}
        if rank in self._dead_ranks:
            # Worker died between encode and /send; embedding is gone.
            self.req_id_to_rank.pop(req_id, None)
            return {
                "req_id": req_id,
                "_error": f"DP worker rank={rank} died before /send for req_id={req_id}",
                "_error_code": int(HTTPStatus.SERVICE_UNAVAILABLE),
            }
        key = req_id + "_send"
        future = asyncio.get_running_loop().create_future()
        self.pending_futures[rank][key] = future
        request["_dp_type"] = "send"
        logger.info(
            f"MM-Encoder DP dispatch_send: req_id={req_id}, "
            f"dp_rank={rank}, pending={self.pending_counts}"
        )
        try:
            await async_sock_send(self.dispatch_sockets[rank], wrap_as_pickle(request))
            return await asyncio.wait_for(future, timeout=ENCODER_REQ_TIMEOUT)
        except asyncio.TimeoutError:
            self.pending_futures[rank].pop(key, None)
            self.req_id_to_rank.pop(req_id, None)
            return self._timeout_envelope(
                req_id,
                "send",
                f"Encoder DP rank={rank} /send timed out after {ENCODER_REQ_TIMEOUT}s",
            )
        except BaseException:
            self.pending_futures[rank].pop(key, None)
            self.req_id_to_rank.pop(req_id, None)
            raise

    async def broadcast(
        self, request: dict, timeout: Optional[float] = None
    ) -> List[dict]:
        # Skip dead ranks: a PUSH to a gone worker would just buffer and then
        # surface as a spurious per-rank timeout. All dead → 503 (same as
        # dispatch), which the profile endpoints turn into an HTTP error.
        eff_timeout = timeout if timeout is not None else ENCODER_REQ_TIMEOUT
        alive_ranks = self.alive_ranks
        if not alive_ranks:
            raise MMError(
                "All encoder DP workers are dead.",
                code=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        batch_id = self._broadcast_counter
        self._broadcast_counter += 1
        rank_keys: List[Tuple[int, str]] = []
        futures: List[asyncio.Future] = []
        dp_type = request.get("_dp_type", "unknown")
        try:
            for rank in alive_ranks:
                req_id = f"_broadcast_{batch_id}_{rank}"
                future = asyncio.get_running_loop().create_future()
                self.pending_futures[rank][req_id] = future
                self.req_id_to_rank[req_id] = rank
                rank_keys.append((rank, req_id))
                request_copy = {**request, "req_id": req_id}
                await async_sock_send(
                    self.dispatch_sockets[rank], wrap_as_pickle(request_copy)
                )
                futures.append(future)
            # Concurrent wait → total bounded by eff_timeout, not
            # dp_size × eff_timeout.
            outcomes = await asyncio.gather(
                *(asyncio.wait_for(fut, timeout=eff_timeout) for fut in futures),
                return_exceptions=True,
            )
            results: List[dict] = []
            for (rank, req_id), outcome in zip(rank_keys, outcomes):
                if isinstance(outcome, asyncio.TimeoutError):
                    self._drop_pending_and_mapping(rank, req_id)
                    results.append(
                        self._timeout_envelope(
                            req_id,
                            dp_type,
                            f"Encoder DP rank={rank} broadcast timed out "
                            f"after {eff_timeout}s",
                        )
                    )
                elif isinstance(outcome, BaseException):
                    self._drop_pending_and_mapping(rank, req_id)
                    raise outcome
                else:
                    results.append(outcome)
            return results
        except BaseException:
            for rank, req_id in rank_keys:
                self._drop_pending_and_mapping(rank, req_id)
            raise

    async def _worker_watchdog(self) -> None:
        # proc.sentinel becomes readable on process exit; fail this rank's
        # pending futures so awaiters don't hang on a dead worker.
        loop = asyncio.get_running_loop()
        watch: Dict[int, asyncio.Future] = {}
        for rank, proc in enumerate(self.worker_processes):
            fut: asyncio.Future = loop.create_future()

            # add_reader is level-triggered, so remove_reader inside the
            # callback to avoid spinning every loop iteration.
            def _on_exit(r=rank, f=fut, p=proc, lp=loop):
                try:
                    lp.remove_reader(p.sentinel)
                except (ValueError, OSError):
                    pass
                if not f.done():
                    f.set_result(r)

            try:
                loop.add_reader(proc.sentinel, _on_exit)
            except (ValueError, OSError):
                continue
            watch[rank] = fut

        while watch:
            done, _ = await asyncio.wait(
                watch.values(), return_when=asyncio.FIRST_COMPLETED
            )
            for fut in done:
                rank = fut.result()
                proc = self.worker_processes[rank]
                logger.error(
                    f"DP worker rank={rank} (pid={proc.pid}) exited "
                    f"with code={proc.exitcode}; failing pending requests"
                )
                self._dead_ranks.add(rank)
                reason = f"DP worker rank={rank} died (exitcode={proc.exitcode})"
                self._fail_pending_for_rank(rank, reason, "WorkerDied")
                self.req_id_to_rank = {
                    r: rk for r, rk in self.req_id_to_rank.items() if rk != rank
                }
                watch.pop(rank, None)

    async def _result_listener(self) -> None:
        # Bounded back-off + give-up so a torn-down context exits in ~3s
        # rather than spinning forever on recv errors.
        consecutive_errors = 0
        while True:
            try:
                msg = await async_sock_recv(self.result_socket)
                consecutive_errors = 0
            except asyncio.CancelledError:
                raise
            except Exception:
                consecutive_errors += 1
                logger.error("_result_listener recv error", exc_info=True)
                if consecutive_errors >= 30:
                    logger.error(
                        "_result_listener giving up after 30 consecutive errors"
                    )
                    self._listener_failed = True
                    self._fail_all_pending(
                        "encoder DP result listener stopped after repeated "
                        "recv errors",
                        "ResultListenerStopped",
                    )
                    return
                await asyncio.sleep(min(0.1 * consecutive_errors, 1.0))
                continue
            req_id = msg.get("req_id", "")
            dp_type = msg.get("_dp_type", "encode")
            key = (req_id + "_send") if dp_type == "send" else req_id
            rank = self.req_id_to_rank.get(req_id)
            if rank is None or key not in self.pending_futures[rank]:
                logger.warning(
                    f"_result_listener: no pending future for "
                    f"req_id={req_id}, dp_type={dp_type}, dropping"
                )
                continue
            future = self.pending_futures[rank].pop(key)
            self._update_pending_gauge()
            # Only mooncake encode (content=request dict) needs the mapping
            # kept for the follow-up /send.
            keep_mapping = dp_type == "encode" and msg.get("content") is not None
            if keep_mapping:
                self._pending_send_at[req_id] = time.monotonic()
            else:
                self.req_id_to_rank.pop(req_id, None)
            try:
                future.set_result(msg)

            except asyncio.InvalidStateError:
                logger.warning(
                    f"_result_listener: future already done for "
                    f"req_id={req_id}, dp_type={dp_type}"
                )

    async def _cleanup_stale_mappings(self) -> None:
        # Evict req_id->rank mappings whose /send never came. The worker frees
        # its own embedding via the send_timeout cleanup scheduled at encode,
        # so both sides key off the same timeout.
        ttl = envs.SGLANG_ENCODER_SEND_TIMEOUT.get()
        interval = max(ttl / 4, 30)
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            stale = [rid for rid, ts in self._pending_send_at.items() if now - ts > ttl]
            for rid in stale:
                self._pending_send_at.pop(rid, None)
                self.req_id_to_rank.pop(rid, None)
            if stale:
                logger.warning(
                    f"Evicted {len(stale)} stale encoder DP /send mapping(s) "
                    f"with no /send within {ttl}s"
                )


async def _dp_worker_handle_profile(
    enc: MMEncoder, dp_rank: int, dp_type: str, request: dict
) -> dict:
    prefix = f"dp_rank={dp_rank}: "
    if dp_type == "start_profile":
        req = request.get("profile_req") or ProfileReq()
        req.req_type = ProfileReqType.START_PROFILE
        if enc.profiler is None:
            enc.profiler = EncoderProfiler(dp_rank)
        ok, msg = enc.profiler.start(req)
        detail = (
            f"started profiling, output_dir={enc.profiler.output_dir}" if ok else msg
        )
    else:  # stop_profile
        if enc.profiler is None:
            return {"ok": False, "msg": prefix + "profiling not initialized"}
        ok, msg = enc.profiler.stop()
        detail = "stopped profiling" if ok else msg
    return {"ok": ok, "msg": prefix + detail}


async def _dp_worker_handle_request(
    enc: MMEncoder,
    sched: EncoderScheduler,
    send_sock,
    send_lock: asyncio.Lock,
    dp_rank: int,
    request: dict,
    dp_type: str,
) -> None:
    t0 = time.time()
    modality_str = str(request.get("modality", "image")).lower()
    is_encode = dp_type not in (
        "start_profile",
        "stop_profile",
        "health_encode",
        "send",
    )
    if is_encode and encoder_metrics_collector is not None:
        encoder_metrics_collector.inc_requests_received(modality=modality_str)
    try:
        if dp_type in ("start_profile", "stop_profile"):
            content = await _dp_worker_handle_profile(enc, dp_rank, dp_type, request)
        elif dp_type == "health_encode":
            content = await _dp_worker_health_encode(enc)
        elif dp_type == "send":
            req_id = request["req_id"]
            await enc.send(
                req_id=req_id,
                prefill_host=request["prefill_host"],
                embedding_port=request["embedding_port"],
                session_id=request["session_id"],
                buffer_address=request["buffer_address"],
            )
            # cancels the scheduled cleanup + frees embedding/forward state
            await enc._cleanup_inflight_encode_state(req_id)
            content = None
        else:
            content = await _dp_worker_encode_and_send(enc, sched, request)

        logger.info(
            f"MM-Encoder [dp_rank={dp_rank}] {dp_type} done: "
            f"req_id={request.get('req_id', '?')}, "
            f"modality={request.get('modality', 'image')}, "
            f"cost={(time.time() - t0) * 1000:.1f}ms"
        )
        if is_encode and encoder_metrics_collector is not None:
            encoder_metrics_collector.inc_requests_total(
                modality=modality_str, status="success"
            )
        envelope = {
            "req_id": request.get("req_id", ""),
            "_dp_type": dp_type,
            "content": content,
        }
    except Exception as e:
        logger.error(
            f"DP worker {dp_rank} error on {dp_type} "
            f"req_id={request.get('req_id', '?')}: {e}",
            exc_info=True,
        )
        if is_encode and encoder_metrics_collector is not None:
            encoder_metrics_collector.inc_requests_total(
                modality=modality_str, status="error"
            )
        err_code = int(getattr(e, "code", None) or HTTPStatus.INTERNAL_SERVER_ERROR)
        envelope = {
            "req_id": request.get("req_id", ""),
            "_dp_type": dp_type,
            "content": None,
            "_error": str(e),
            "_error_type": type(e).__name__,
            "_error_code": err_code,
        }

    # pyzmq async send isn't safe for concurrent senders.
    try:
        async with send_lock:
            await async_sock_send(send_sock, wrap_as_pickle(envelope))
    except Exception:
        logger.error(
            f"DP worker {dp_rank} failed to send envelope for "
            f"req_id={request.get('req_id', '?')}",
            exc_info=True,
        )


async def run_dp_worker(
    server_args: ServerArgs,
    dp_rank: int,
    gpu_id: int,
    dispatch_path: str,
    result_path: str,
):
    logger.info(
        f"DP worker {dp_rank} starting on gpu_id={gpu_id} "
        f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})"
    )

    # gpu_id is the device chosen by maybe_reindex_device_id in the parent:
    # 0 when CVD is pinned to one GPU, else the absolute id. rank=0, so
    # MMEncoder runs set_device(base_gpu_id).
    args = copy.deepcopy(server_args)
    args.base_gpu_id = gpu_id
    args.tp_size = 1
    enc = MMEncoder(args, dist_init_method=f"tcp://127.0.0.1:{get_free_port()}", rank=0)

    global encoder_metrics_collector
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()
        labels = {
            "model_name": server_args.served_model_name,
            "dp_rank": str(dp_rank),
        }
        if server_args.extra_metric_labels:
            labels.update(server_args.extra_metric_labels)
        encoder_metrics_collector = EncoderMetricsCollector(labels)
        enc.dp_rank = dp_rank

    sched = EncoderScheduler(
        encoder=enc, send_sockets=[], max_batch_size=ENCODER_MAX_BATCH_SIZE
    )

    ctx = zmq.asyncio.Context(2)
    recv_sock = get_zmq_socket(ctx, zmq.PULL, dispatch_path, False)
    send_sock = get_zmq_socket(ctx, zmq.PUSH, result_path, False)
    send_lock = asyncio.Lock()
    inflight: Set[asyncio.Task] = set()
    # Acquire-before-recv → back-pressure propagates to the dispatcher
    # PUSH buffer. Must be ≥ ENCODER_MAX_BATCH_SIZE or batching degrades.
    max_inflight = envs.SGLANG_ENCODER_DP_WORKER_MAX_INFLIGHT.get()
    if max_inflight < ENCODER_MAX_BATCH_SIZE:
        logger.warning(
            f"SGLANG_ENCODER_DP_WORKER_MAX_INFLIGHT={max_inflight} is below "
            f"ENCODER_MAX_BATCH_SIZE={ENCODER_MAX_BATCH_SIZE}; the encoder "
            f"will never assemble a full batch."
        )
    inflight_sem = asyncio.Semaphore(max_inflight)
    sched.start()
    logger.info(f"DP worker {dp_rank} ready")

    # Task-per-request so EncoderScheduler.pending_queue accumulates and
    # actual cross-request batching can happen.
    try:
        while True:
            await inflight_sem.acquire()
            # Released by _run on success or the outer finally if not spawned.
            spawned = False
            try:
                try:
                    request = await async_sock_recv(recv_sock)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.error(f"DP worker {dp_rank} recv error", exc_info=True)
                    continue
                if not isinstance(request, dict):
                    logger.error(
                        f"DP worker {dp_rank} received non-dict request "
                        f"({type(request).__name__}); dropping"
                    )
                    continue
                dp_type = request.pop("_dp_type", "encode")

                async def _run(req=request, t=dp_type):
                    try:
                        await _dp_worker_handle_request(
                            enc, sched, send_sock, send_lock, dp_rank, req, t
                        )
                    finally:
                        inflight_sem.release()

                task = asyncio.create_task(_run())
                # Ownership transferred to _run; mark before any op that could
                # raise (theoretical: set.add / add_done_callback) and cause a
                # double-release.
                spawned = True
                inflight.add(task)
                task.add_done_callback(inflight.discard)
            finally:
                if not spawned:
                    inflight_sem.release()
    finally:
        # Close zmq on exception/cancellation (normal stop is parent SIGKILL).
        for task in inflight:
            task.cancel()
        ctx.destroy(linger=0)


def launch_dp_worker(
    server_args: ServerArgs,
    dp_rank: int,
    gpu_id: int,
    dispatch_path: str,
    result_path: str,
):
    try:
        configure_logger(server_args, prefix=f" encode_dp_worker[{dp_rank}]")
        asyncio.run(
            run_dp_worker(server_args, dp_rank, gpu_id, dispatch_path, result_path)
        )
    except KeyboardInterrupt:
        logger.info(f"DP worker {dp_rank} exiting")
    except Exception:
        traceback.print_exc()


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    global encoder_scheduler
    if dp_dispatcher is not None:
        dp_dispatcher.start()
        yield
        return
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
        request = await async_sock_recv(encoder.schedule_socket)
        if isinstance(request, ProfileReq):
            if request.req_type == ProfileReqType.START_PROFILE:
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


def _register_encoder_url_with_bootstrap(server_args: ServerArgs):
    """Asynchronously register this encoder with each bootstrap URL.

    Spawns a daemon thread that retries each URL independently with bounded
    backoff.  The encoder's own startup is not blocked: if some bootstrap
    server is slow or unreachable, only the background worker waits.

    Inspired by ``_ensure_prefill_info`` in disaggregation/decode.py: each
    target keeps its own retry count and is retried at a fixed interval
    instead of serialising sleeps in a single thread.
    """

    host = server_args.host
    if not host or host in ("0.0.0.0", "::"):
        host = get_local_ip_auto(server_args.host)
    scheme = "https" if server_args.ssl_certfile else "http"
    encoder_url = NetworkAddress(host, server_args.port).to_url(scheme)
    payload = {"url": encoder_url}
    bootstrap_urls = list(server_args.encoder_register_urls)
    if not bootstrap_urls:
        return

    max_retries = 30
    retry_interval = 5.0
    request_timeout = 5.0

    def _try_register_once(bootstrap_url: str) -> bool:
        try:
            resp = http_requests.post(
                f"{bootstrap_url}/register_encoder_url",
                json=payload,
                timeout=request_timeout,
            )
            if resp.status_code == 200:
                logger.info(
                    f"Registered encoder URL '{encoder_url}' with bootstrap "
                    f"at {bootstrap_url}"
                )
                return True
            logger.warning(
                f"Bootstrap {bootstrap_url} returned {resp.status_code}: {resp.text}"
            )
        except Exception as e:
            logger.debug(f"Register attempt to {bootstrap_url} failed: {e}")
        return False

    def _worker():
        pending = list(bootstrap_urls)
        retry_count = {url: 0 for url in pending}
        while pending:
            still_pending = []
            for bootstrap_url in pending:
                if _try_register_once(bootstrap_url):
                    continue
                retry_count[bootstrap_url] += 1
                if retry_count[bootstrap_url] >= max_retries:
                    logger.error(
                        f"Giving up on bootstrap {bootstrap_url} after "
                        f"{max_retries} attempts. Encoder discovery via this "
                        f"bootstrap will be incomplete."
                    )
                    continue
                still_pending.append(bootstrap_url)
            pending = still_pending
            if pending:
                time.sleep(retry_interval)

    threading.Thread(
        target=_worker, daemon=True, name="encoder-bootstrap-register"
    ).start()


def _unregister_encoder_url_from_bootstrap(server_args: ServerArgs):
    host = server_args.host
    if not host or host in ("0.0.0.0", "::"):
        host = get_local_ip_auto(server_args.host)
    scheme = "https" if server_args.ssl_certfile else "http"
    encoder_url = NetworkAddress(host, server_args.port).to_url(scheme)
    payload = {"url": encoder_url}

    for bootstrap_url in server_args.encoder_register_urls:
        try:
            resp = http_requests.delete(
                f"{bootstrap_url}/unregister_encoder_url",
                json=payload,
                timeout=2.0,
            )
            if resp.status_code == 200:
                logger.info(
                    f"Unregistered encoder URL '{encoder_url}' from "
                    f"bootstrap at {bootstrap_url}"
                )
            else:
                logger.warning(
                    f"Bootstrap {bootstrap_url} returned "
                    f"{resp.status_code} on unregister: {resp.text}"
                )
        except Exception as e:
            logger.debug(f"Unregister from {bootstrap_url} failed: {e}")


def launch_server(server_args: ServerArgs):
    configure_logger(server_args, prefix=" encode_server")
    if server_args.dp_size > 1:
        _launch_server_dp(server_args)
        return

    global encoder, encoder_metrics_collector

    # Set up prometheus metrics.
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()
        labels = {
            "model_name": server_args.served_model_name,
            "dp_rank": "0",
        }
        if server_args.extra_metric_labels:
            labels.update(server_args.extra_metric_labels)
        encoder_metrics_collector = EncoderMetricsCollector(labels)
        add_prometheus_middleware(app)

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
    if server_args.enable_trace:
        process_tracing_init(
            server_args.otlp_traces_endpoint,
            "sglang",
            trace_modules=server_args.trace_modules,
        )
        trace_set_thread_info("Encoder")
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

    # Register this encoder's URL with prefill server(s) if configured.
    if server_args.encoder_register_urls:
        import atexit

        _register_encoder_url_with_bootstrap(server_args)
        atexit.register(_unregister_encoder_url_from_bootstrap, server_args)

    uvicorn.run(app, host=server_args.host, port=server_args.port)


def _launch_server_dp(server_args: ServerArgs):
    global dp_dispatcher

    if server_args.dp_size <= 1 or server_args.tp_size != 1:
        raise ValueError(
            "Encoder DP mode requires --dp-size > 1 and --tp-size 1; got "
            f"dp_size={server_args.dp_size}, tp_size={server_args.tp_size}."
        )
    dp_size = server_args.dp_size
    logger.info(f"Launching encoder in DP mode: dp_size={dp_size}")

    # DP mode: workers (subprocesses) write metrics to the shared multiproc dir;
    # the main process exposes the aggregated /metrics endpoint.
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()
        add_prometheus_middleware(app)

    ctx = mp.get_context("spawn")
    ipc_prefix = random_uuid()
    async_zmq_ctx = zmq.asyncio.Context(dp_size + 1)

    result_path = f"ipc:///tmp/{ipc_prefix}_dp_result"
    result_socket = get_zmq_socket(async_zmq_ctx, zmq.PULL, result_path, True)

    dispatch_sockets: List[zmq.asyncio.Socket] = [
        get_zmq_socket(
            async_zmq_ctx, zmq.PUSH, f"ipc:///tmp/{ipc_prefix}_dp_dispatch_{r}", True
        )
        for r in range(dp_size)
    ]

    # Register atexit BEFORE spawn loop so partial spawns get reaped on
    # exception (atexit holds the list ref and reads it at exit time).
    import atexit

    worker_processes: List[mp.Process] = []

    def _kill_workers():
        for p in worker_processes:
            if p.is_alive():
                p.kill()
        for p in worker_processes:
            p.join(timeout=5)

    atexit.register(_kill_workers)

    for dp_rank in range(dp_size):
        gpu_id = server_args.base_gpu_id + dp_rank
        # Pin the device parent-side around spawn (same convention as the
        # scheduler launcher and DP controller) so the child inherits
        # CUDA_VISIBLE_DEVICES from its first instruction, before any import
        # can enumerate CUDA. No-op unless SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS
        # is set, in which case gpu_id is reindexed to 0 and CVD is pinned.
        with maybe_reindex_device_id(gpu_id) as gpu_id:
            proc = ctx.Process(
                target=launch_dp_worker,
                args=(
                    server_args,
                    dp_rank,
                    gpu_id,
                    f"ipc:///tmp/{ipc_prefix}_dp_dispatch_{dp_rank}",
                    result_path,
                ),
                daemon=False,
            )
            proc.start()
        worker_processes.append(proc)

    dp_dispatcher = DPDispatcher(
        dp_size,
        dispatch_sockets,
        result_socket,
        worker_processes,
        enable_metrics=server_args.enable_metrics,
    )

    # Register this encoder's URL with prefill server(s) if configured.
    if server_args.encoder_register_urls:
        import atexit

        _register_encoder_url_with_bootstrap(server_args)
        atexit.register(_unregister_encoder_url_from_bootstrap, server_args)

    uvicorn.run(app, host=server_args.host, port=server_args.port)


def _summarise_dp_broadcast(results: List[dict]) -> Response:
    # Treat missing/None content as failure so a stuck rank doesn't hide
    # behind the others' "ok". Status = the most severe per-rank error code
    # (5xx beats 4xx) rather than a blanket 400, so a worker's 500/503/504
    # isn't misreported as a client error.
    msgs: List[str] = []
    error_codes: List[int] = []
    for r in results:
        content = r.get("content")
        if isinstance(content, dict):
            msgs.append(content.get("msg", ""))
            if not content.get("ok"):
                # Worker ran but reported a logical failure; no transport code,
                # so treat as a bad request (matches the non-DP profile path).
                error_codes.append(int(r.get("_error_code") or HTTPStatus.BAD_REQUEST))
        else:
            msgs.append(r.get("_error", "unknown error"))
            error_codes.append(
                int(r.get("_error_code") or HTTPStatus.INTERNAL_SERVER_ERROR)
            )
    status_code = 200 if not error_codes else max(error_codes)
    return Response(
        content="\n".join(msgs) + "\n",
        status_code=status_code,
    )


async def get_condition(rid):
    async with cond_dict_lock:
        if rid not in rid_to_cond:
            rid_to_cond[rid] = asyncio.Condition()
        return rid_to_cond[rid]


@app.post("/encode")
async def handle_encode_request(request: dict):
    req_id = request["req_id"]
    start_time = time.monotonic()
    time_stats_json = request.pop("time_stats_json", None)
    time_stats = EncoderReqTimeStats()
    if dp_dispatcher is not None:
        if time_stats_json:
            request = dict(request)
            request["time_stats_json"] = time_stats_json
        try:
            result = await dp_dispatcher.dispatch(request)
        except MMError as e:
            # Surface MMError.code (503 when all workers dead) instead of
            # FastAPI's default 500.
            logger.error(f"DP dispatch refused req_id={req_id}: {e}")
            return ORJSONResponse(
                status_code=int(e.code),
                content={"status": "error", "message": str(e), "req_id": req_id},
            )
        if result.get("_error"):
            error_type = result.get("_error_type", "")
            # `or` (not `dict.get(key, default)`) so explicit None falls back too.
            status_code = result.get("_error_code") or (
                HTTPStatus.BAD_REQUEST
                if error_type == "ValueError"
                else HTTPStatus.INTERNAL_SERVER_ERROR
            )
            logger.error(f"DP worker error for req_id={req_id}: {result['_error']}")
            return ORJSONResponse(
                status_code=status_code,
                content={
                    "status": "error",
                    "message": result["_error"],
                    "req_id": req_id,
                },
            )
        elapsed = time.monotonic() - start_time
        logger.info(
            f"[{req_id}] /encode completed in {elapsed:.3f}s, "
            f"modality={request.get('modality', 'image')}"
        )
        return ORJSONResponse(content=result.get("content"))

    modality_str = str(request.get("modality", "image")).lower()
    try:
        # when multiple decoder TP ranks POST /encode
        # with the same req_id, only the first triggers the VIT forward;
        # subsequent callers wait and return the same metadata.
        if encoder.server_args.encoder_transfer_backend == "mooncake":
            async with encoder._inflight_encode_lock:
                if req_id in encoder._inflight_encode_events:
                    event = encoder._inflight_encode_events[req_id]
                    is_duplicate = True
                else:
                    event = asyncio.Event()
                    encoder._inflight_encode_events[req_id] = event
                    is_duplicate = False

            if is_duplicate:
                await event.wait()
                meta = encoder._inflight_encode_meta.get(req_id)
                if meta is None:
                    return ORJSONResponse(
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        content={
                            "status": "error",
                            "message": "Encode failed on the first request",
                            "req_id": req_id,
                        },
                    )
                nbytes, embedding_len, embedding_dim = meta
                # Build the same metadata response as the first request
                resp = dict(request)
                del resp["mm_items"]
                resp.update(
                    {
                        "embedding_size": nbytes,
                        "embedding_len": embedding_len,
                        "embedding_dim": embedding_dim,
                    }
                )
                return ORJSONResponse(content=resp)

        def start_background_send(req_id):
            task = asyncio.create_task(encoder.send_with_url(req_id=req_id))
            encoder.background_tasks.add(task)
            task.add_done_callback(encoder.background_tasks.discard)

        # broadcast request, lock together with rank0 await so NCCL
        # launch order matches the ZMQ dispatch order rank>0 sees.
        async with encoder.encode_dispatch_lock:
            request.update({"enter_time": time.time()})
            modality = Modality.from_str(request["modality"])
            if time_stats_json:
                time_stats.decode_json(time_stats_json)

            modality_str = modality.name.lower()
            time_stats.modality = modality_str
            time_stats.set_metrics_collector(encoder_metrics_collector)
            time_stats.set_mm_encode_start_time()
            if encoder_metrics_collector is not None:
                encoder_metrics_collector.inc_requests_received(modality=modality_str)
            if encoder_scheduler is not None and modality in _BATCHABLE_MODALITIES:
                try:
                    nbytes, embedding_len, embedding_dim, error_msg, error_code = (
                        await encoder_scheduler.submit(request)
                    )
                except asyncio.TimeoutError:
                    time_stats.trace_ctx.abort(
                        abort_info={"reason": "encoder batch timed out"}
                    )
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
                    sock_send(socket, wrap_as_pickle(request))
                nbytes, embedding_len, embedding_dim, error_msg, error_code = (
                    await encoder.encode_request(request, modality)
                )

        if error_msg:
            time_stats.trace_ctx.abort(abort_info={"reason": error_msg})
        else:
            time_stats.set_mm_encode_end_time()

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
            # Signal waiters on failure for mooncake
            if encoder.server_args.encoder_transfer_backend == "mooncake":
                encoder._inflight_encode_meta.pop(req_id, None)
                evt = encoder._inflight_encode_events.pop(req_id, None)
                if evt:
                    evt.set()
                await encoder._cleanup_inflight_encode_state(req_id)
            if encoder_metrics_collector is not None:
                encoder_metrics_collector.inc_requests_total(
                    modality=modality_str, status="error"
                )
            return ORJSONResponse(
                status_code=error_code,
                content={"status": "error", "message": error_msg, "req_id": req_id},
            )
        if encoder.server_args.encoder_transfer_backend == "mooncake":
            # Store metadata for duplicate callers and signal them
            encoder._inflight_encode_meta[req_id] = (
                nbytes,
                embedding_len,
                embedding_dim,
            )
            evt = encoder._inflight_encode_events.get(req_id)
            if evt:
                evt.set()
            encoder._schedule_inflight_encode_cleanup(req_id)
            del request["mm_items"]
            request.update(
                {
                    "embedding_size": nbytes,
                    "embedding_len": embedding_len,
                    "embedding_dim": embedding_dim,
                }
            )
            if encoder_metrics_collector is not None:
                encoder_metrics_collector.inc_requests_total(
                    modality=modality_str, status="success"
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
            if encoder_metrics_collector is not None:
                encoder_metrics_collector.inc_requests_total(
                    modality=modality_str, status="success"
                )
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
            if encoder_metrics_collector is not None:
                encoder_metrics_collector.inc_requests_total(
                    modality=modality_str, status="success"
                )
            return ORJSONResponse(content=None)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Unexpected error in encoder logic for {req_id}: {error_msg}")
        rid_to_err_msg[req_id] = error_msg
        # Ensure inflight waiters are unblocked on unexpected errors
        if encoder.server_args.encoder_transfer_backend == "mooncake":
            encoder._inflight_encode_meta.pop(req_id, None)
            evt = encoder._inflight_encode_events.pop(req_id, None)
            if evt:
                evt.set()
            await encoder._cleanup_inflight_encode_state(req_id)
        if encoder_metrics_collector is not None:
            encoder_metrics_collector.inc_requests_total(
                modality=modality_str, status="error"
            )
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
    if dp_dispatcher is not None:
        try:
            result = await dp_dispatcher.dispatch_send(request)
        except MMError as e:
            req_id = request.get("req_id", "?")
            logger.error(f"DP dispatch_send refused req_id={req_id}: {e}")
            return Response(
                content=f"Encoder DP worker send error: {e}",
                status_code=int(e.code),
            )
        if result.get("_error"):
            req_id = request.get("req_id", "?")
            status_code = result.get("_error_code") or int(
                HTTPStatus.INTERNAL_SERVER_ERROR
            )
            logger.error(
                f"DP worker send error for req_id={req_id}: {result['_error']}"
            )
            return Response(
                content=f"Encoder DP worker send error: {result['_error']}",
                status_code=status_code,
            )
        return ORJSONResponse(content=result.get("content"))
    await encoder.send(
        req_id=request["req_id"],
        prefill_host=request["prefill_host"],
        embedding_port=request["embedding_port"],
        session_id=request["session_id"],
        buffer_address=request["buffer_address"],
    )
    req_id = request["req_id"]
    # Don't pop embedding_to_send here — other decoder TP ranks may still
    # need it for their /send calls. Cleanup is handled by the scheduled
    # timeout task or _cleanup_inflight_encode_state.
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
    if dp_dispatcher is not None:
        # Strict: any dead (exited) rank fails health → orchestrator restarts.
        if not dp_dispatcher.all_ranks_alive:
            return Response(status_code=503)
        # Process-liveness (proc.sentinel) can't see a worker that's alive but
        # wedged (hung GPU / NCCL deadlock / stalled ZMQ). Probe every rank with
        # a tiny dummy encode; each worker runs it only when idle and otherwise
        # reports healthy at once, keeping the probe off the GPU under load.
        try:
            results = await dp_dispatcher.broadcast(
                {"_dp_type": "health_encode"},
                timeout=HEALTH_CHECK_TIMEOUT,
            )
        except MMError:
            return Response(status_code=503)
        if any(r.get("_error") for r in results):
            return Response(status_code=503)
        return Response(status_code=200)
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
        # uuid keeps rids unique across workers; a bare time.time() can collide.
        req_id = f"{HEALTH_CHECK_RID_PREFIX}_{uuid.uuid4().hex}"

        dummy_request = {
            "mm_items": mm_items,
            "modality": modality.name,
            "req_id": req_id,
            "num_parts": 1,
            "part_idx": 0,
        }

        # Broadcast to other TP ranks so distributed ops stay in sync
        for socket in send_sockets:
            sock_send(socket, wrap_as_pickle(dummy_request))

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
async def start_profile_async(obj: Annotated[Optional[ProfileReq], Body()] = None):
    if dp_dispatcher is not None:
        if obj is not None:
            obj.req_type = ProfileReqType.START_PROFILE
        try:
            results = await dp_dispatcher.broadcast(
                {"_dp_type": "start_profile", "profile_req": obj}
            )
        except MMError as e:
            return Response(content=f"{e}\n", status_code=int(e.code))
        return _summarise_dp_broadcast(results)
    if encoder is None:
        return Response(content="encoder not ready\n", status_code=503)
    req = obj or ProfileReq()
    req.req_type = ProfileReqType.START_PROFILE
    for socket in send_sockets:
        sock_send(socket, req)
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
    if dp_dispatcher is not None:
        try:
            results = await dp_dispatcher.broadcast({"_dp_type": "stop_profile"})
        except MMError as e:
            return Response(content=f"{e}\n", status_code=int(e.code))
        return _summarise_dp_broadcast(results)
    if encoder is None:
        return Response(content="encoder not ready\n", status_code=503)
    if encoder.profiler is None:
        return Response(
            content="profiling not initialized\n", status_code=HTTPStatus.BAD_REQUEST
        )
    req = ProfileReq(req_type=ProfileReqType.STOP_PROFILE)
    for socket in send_sockets:
        sock_send(socket, req)
    ok, msg = encoder.profiler.stop()
    if ok:
        return Response(content="Stop profiling.\n", status_code=200)
    return Response(
        content=(msg or "Stop profiling failed.\n"), status_code=HTTPStatus.BAD_REQUEST
    )
