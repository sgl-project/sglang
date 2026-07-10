import asyncio
import concurrent.futures
import contextlib
import ctypes
import logging
import os
import pickle
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import aiohttp
import numpy as np
import torch
import zmq
import zmq.asyncio

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.encode_receiver import (
    EmbeddingData,
    video_meta_attrs_for,
)
from sglang.srt.disaggregation.encoder_preprocessor import EncoderPreprocessor
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
    sock_send,
    wrap_as_pickle,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.model_loader import get_model
from sglang.srt.observability.metrics_collector import EncoderMetricsCollector
from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils.network import (
    NetworkAddress,
    config_socket,
    get_local_ip_auto,
    get_zmq_socket,
)

logger = logging.getLogger(__name__)


def is_health_check_request(rid: Optional[str]) -> bool:
    return isinstance(rid, str) and rid.startswith(HEALTH_CHECK_RID_PREFIX)


rid_lock = asyncio.Lock()
rid_to_receive_endpoint: Dict[str, List[str]] = dict()
rid_to_receive_count: Dict[str, int] = dict()
rid_to_err_msg: Dict[str, str] = dict()
cond_dict_lock = asyncio.Lock()
rid_to_cond: Dict[str, asyncio.Condition] = {}

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


@dataclass
class GlobalCacheEncodeContext:
    req_id: str
    modality: Modality
    mm_inputs: dict
    get_feature_fn: Any
    grid_thw: List
    token_counts: List[int]
    mm_feature: Any
    num_items: int
    aux_data: dict
    str_mm_hashes: Optional[List[str]]


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

        # CPU preprocessing pipeline (Rust-replaceable)
        self.preprocessor = EncoderPreprocessor(
            server_args=server_args,
            model_config=self.model_config,
            model_preprocessor=getattr(self.model, "preprocess_mm_for_encoder", None),
        )

        self.context = zmq.asyncio.Context(2)
        self.sync_context = zmq.Context()  # Reuse sync context for thread pool
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

        embedding_cache_size = int(os.environ.get("SGLANG_VLM_CACHE_SIZE_MB", "4096"))
        self.mm_cache = MultiModalStaticCache(embedding_cache_size * 1024 * 1024)
        self.mm_cache_lock = asyncio.Lock()

        self.send_timeout = envs.SGLANG_ENCODER_SEND_TIMEOUT.get()

        if schedule_path is not None:
            self.schedule_socket = get_zmq_socket(
                self.context, zmq.PULL, schedule_path, True
            )
        self.background_tasks: Set[asyncio.Task] = set()

        # Embedding dtype = model param dtype. Always available (both transfer
        # backends and the global-cache pool rely on it).
        self._embedding_dtype = next(self.model.parameters()).dtype
        self._element_size = torch.tensor(
            [], dtype=self._embedding_dtype
        ).element_size()

        if self.server_args.enable_mm_global_cache:
            from sglang.srt.mem_cache.embedding_cache_controller import (
                EmbeddingCacheController,
            )
            from sglang.srt.mem_cache.embedding_store import EmbeddingStoreFactory

            embedding_store = EmbeddingStoreFactory.create_backend(
                self.server_args.mm_global_cache_backend,
            )
            hidden_dims = self._infer_embedding_dims()
            self.mm_global_cache = EmbeddingCacheController(
                rank,
                server_args.tp_size,
                embedding_store=embedding_store,
                hidden_dims=hidden_dims,
                tp_group=get_tp_group().cpu_group,
                all_rank_get=False,
                dtype=self._embedding_dtype,
            )
        else:
            self.mm_global_cache = None

        # Pre-compute embedding metadata (needed by all ranks for mooncake)
        if self.server_args.encoder_transfer_backend == "mooncake":
            self._embedding_dims = self._infer_embedding_dims()

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

    def supports_modality(self, modality: Modality) -> bool:
        return self.preprocessor.supports_modality(modality)

    def has_pending_embeddings(self) -> bool:
        return bool(getattr(self, "embedding_to_send", None))

    def discard_embedding(self, req_id: str) -> None:
        embedding_to_send = getattr(self, "embedding_to_send", None)
        if embedding_to_send is not None:
            embedding_to_send.pop(req_id, None)

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

    def get_num_patches(
        self, grid: Union[torch.Tensor, List[int]], modality: Modality
    ) -> int:
        """Calculate number of raw patches (before merge/sampling). Used for pixel_values slicing."""
        if modality == Modality.AUDIO:
            return int(grid.item())
        else:
            return int(grid[0] * grid[1] * grid[2])

    def slice_embedding(
        self, mm_embedding: torch.Tensor, token_counts: Iterable[int]
    ) -> List[torch.Tensor]:
        """Slice a concatenated embedding tensor using preprocessor metadata."""
        slices, offset = [], 0
        for count in token_counts:
            slices.append(mm_embedding[offset : offset + count])
            offset += count
        if mm_embedding.shape[0] != offset:
            raise InternalError(
                f"Encoder produced {mm_embedding.shape[0]} tokens, but "
                f"preprocessor metadata expected {offset}"
            )
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
        token_counts: List[int],
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

        return self.slice_embedding(new_embeddings, (token_counts[i] for i in indices))

    async def _process_mm_items(
        self, mm_items, modality: Modality, log_metrics: bool = True
    ):
        preprocess_start = time.perf_counter()
        processor_input, token_counts = await self.preprocessor.process_mm_items(
            mm_items, modality
        )
        if encoder_metrics_collector is not None and log_metrics:
            item_count = len(token_counts)
            encoder_metrics_collector.observe_preprocess(
                time.perf_counter() - preprocess_start,
                modality=modality.name.lower(),
            )
            encoder_metrics_collector.observe_mm_items_per_request(
                item_count, modality=modality.name.lower()
            )
            encoder_metrics_collector.observe_mm_items_per_batch(
                item_count, modality=modality.name.lower()
            )

        target = self.model.thinker if hasattr(self.model, "thinker") else self.model
        get_feature_method = getattr(target, f"get_{modality.name.lower()}_feature")
        return processor_input, token_counts, get_feature_method

    async def _prepare_global_cache_context(
        self,
        mm_items,
        modality: Modality,
        req_id: str,
        hashes: Optional[List[str]] = None,
    ) -> GlobalCacheEncodeContext:
        mm_inputs, token_counts, get_feature_fn = await self._process_mm_items(
            mm_items, modality
        )
        grid_thw = _get_mm_grid_dim(mm_inputs, modality, self.model_type)
        mm_feature = _convert(_get_mm_feature(mm_inputs, modality))
        num_items = len(grid_thw)
        if len(token_counts) != num_items:
            raise InternalError(
                f"Preprocessor returned {len(token_counts)} token counts for "
                f"{num_items} {modality.name.lower()} grid entries"
            )

        # Hashes must be grid-space; a leaf-space list would size-mismatch
        # rank>0's mask (zeros(num_items)) and deadlock TP.
        if hashes is not None and len(hashes) != num_items:
            raise BadRequestError(
                f"User-supplied hashes length {len(hashes)} != grid count "
                f"{num_items} for {self.model_type}/{modality.name}; hashes "
                f"must be in grid space (1 per encoder grid entry)."
            )

        str_mm_hashes = None
        if self.rank == 0:
            if hashes is None:
                mm_hashes = self._calculate_hashes_from_features(
                    mm_feature, grid_thw, modality
                )
            else:
                mm_hashes = hashes
            # L2 cache expects string keys for Mooncake.
            str_mm_hashes = [str(h) for h in mm_hashes]

        return GlobalCacheEncodeContext(
            req_id=req_id,
            modality=modality,
            mm_inputs=mm_inputs,
            get_feature_fn=get_feature_fn,
            grid_thw=grid_thw,
            token_counts=token_counts,
            mm_feature=mm_feature,
            num_items=num_items,
            aux_data=_build_mm_aux_data(mm_inputs, self.model_type),
            str_mm_hashes=str_mm_hashes,
        )

    def _broadcast_global_cache_mask(self, mask_tensor: torch.Tensor):
        if self.server_args.tp_size > 1:
            torch.distributed.broadcast(
                mask_tensor,
                src=0,
                group=self.mm_global_cache.prefetch_tp_group,
            )

    async def _lookup_global_cache(
        self,
        ctx: GlobalCacheEncodeContext,
    ) -> Tuple[List[int], List[int]]:
        if self.rank == 0:
            exist_mask = await self.mm_global_cache.batch_is_exist(ctx.str_mm_hashes)
            mask_tensor = torch.tensor(
                [1 if e else 0 for e in exist_mask], dtype=torch.int32
            )
        else:
            mask_tensor = torch.zeros(ctx.num_items, dtype=torch.int32)

        self._broadcast_global_cache_mask(mask_tensor)

        exist_mask = [m.item() == 1 for m in mask_tensor]
        missing_indices = [i for i, e in enumerate(exist_mask) if not e]
        hit_indices = [i for i, e in enumerate(exist_mask) if e]
        return missing_indices, hit_indices

    def _prefetch_global_cache_hits(
        self,
        ctx: GlobalCacheEncodeContext,
        hit_indices: List[int],
    ) -> List[str]:
        if self.rank != 0 or not hit_indices:
            return []

        hit_hashes = [ctx.str_mm_hashes[i] for i in hit_indices]
        hit_tokens = [ctx.token_counts[i] for i in hit_indices]
        self.mm_global_cache.prefetch(ctx.req_id, hit_hashes, hit_tokens, ctx.modality)
        return hit_hashes

    async def _wait_global_cache_prefetch(
        self,
        ctx: GlobalCacheEncodeContext,
        hit_indices: List[int],
        hit_hashes: List[str],
    ) -> List[int]:
        fallback_mask = torch.zeros(ctx.num_items, dtype=torch.int32)
        if self.rank == 0 and hit_indices:
            try:

                async def _wait_prefetch():
                    while not self.mm_global_cache.check_prefetch_progress(ctx.req_id):
                        await asyncio.sleep(0.005)

                await asyncio.wait_for(_wait_prefetch(), timeout=60.0)

                for i, idx in enumerate(hit_indices):
                    if not self.mm_global_cache.has_local_embedding(hit_hashes[i]):
                        fallback_mask[idx] = 1
                num_partial_fail = int(fallback_mask.sum().item())
                if num_partial_fail > 0:
                    logger.warning(
                        f"Req {ctx.req_id}: {num_partial_fail}/{len(hit_indices)} "
                        f"cache-hit items failed to load, falling back to ViT"
                    )
            except (asyncio.TimeoutError, Exception) as e:
                logger.error(
                    f"Prefetch failed for req {ctx.req_id}: {e}. "
                    f"Falling back to ViT for {len(hit_indices)} hit items."
                )
                for idx in hit_indices:
                    fallback_mask[idx] = 1

        self._broadcast_global_cache_mask(fallback_mask)
        fallback_indices = [
            i for i in range(ctx.num_items) if fallback_mask[i].item() == 1
        ]
        return fallback_indices

    def _launch_global_cache_insert(
        self,
        ctx: GlobalCacheEncodeContext,
        hashes: List[str],
        d2h_handles: List[Any],
    ):
        if not hashes:
            return

        async def _background_insert():
            await asyncio.to_thread(
                self.mm_global_cache.wait_store_to_pool,
                d2h_handles,
            )
            await asyncio.to_thread(
                self.mm_global_cache.insert_batch,
                hashes,
                ctx.modality,
            )

        task = asyncio.create_task(_background_insert())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    @staticmethod
    def _as_2d_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 2:
            tensor = tensor.reshape(-1, tensor.shape[-1])
        return tensor

    def _assemble_global_cache_cpu(
        self,
        ctx: GlobalCacheEncodeContext,
        hit_indices: List[int],
        missing_indices: List[int],
        fallback_indices: List[int],
        new_slices: List[torch.Tensor],
        fallback_slices: List[torch.Tensor],
    ) -> torch.Tensor:
        miss_slice_pos = {idx: pos for pos, idx in enumerate(missing_indices)}
        fallback_slice_pos = {idx: pos for pos, idx in enumerate(fallback_indices)}
        fallback_index_set = set(fallback_indices)
        token_counts = ctx.token_counts
        dim = self.mm_global_cache.get_embedding_dim(ctx.modality)

        mm_embedding = torch.empty(
            (sum(token_counts), dim),
            dtype=self._embedding_dtype,
            pin_memory=True,
        )

        hit_view_hashes = [
            ctx.str_mm_hashes[idx]
            for idx in hit_indices
            if idx not in fallback_index_set
        ]
        hit_views = {}
        try:
            if hit_view_hashes:
                cached_slice_lists = self.mm_global_cache.get_pool_views(
                    hit_view_hashes
                )
                for h, slices in zip(hit_view_hashes, cached_slice_lists):
                    if slices is None:
                        raise InternalError(
                            f"Cached embedding {h} not available for req {ctx.req_id}"
                        )
                    hit_views[h] = slices

            offset = 0
            for idx, num_tokens in enumerate(token_counts):
                if idx in miss_slice_pos:
                    src = self._as_2d_tensor(new_slices[miss_slice_pos[idx]])
                    mm_embedding[offset : offset + num_tokens].copy_(
                        src, non_blocking=True
                    )
                elif idx in fallback_slice_pos:
                    src = self._as_2d_tensor(fallback_slices[fallback_slice_pos[idx]])
                    mm_embedding[offset : offset + num_tokens].copy_(
                        src, non_blocking=True
                    )
                else:
                    copied = 0
                    for view in hit_views[ctx.str_mm_hashes[idx]]:
                        n = view.shape[0]
                        mm_embedding[offset + copied : offset + copied + n].copy_(view)
                        copied += n
                offset += num_tokens

            torch.cuda.current_stream(self.device).synchronize()
            return mm_embedding
        finally:
            if hit_view_hashes:
                self.mm_global_cache.release_pool_views(hit_view_hashes)

    def _assemble_global_cache_gpu(
        self,
        ctx: GlobalCacheEncodeContext,
        missing_indices: List[int],
        fallback_indices: List[int],
        new_slices: List[torch.Tensor],
        fallback_slices: List[torch.Tensor],
    ) -> torch.Tensor:
        miss_slice_pos = {idx: pos for pos, idx in enumerate(missing_indices)}
        fallback_slice_pos = {idx: pos for pos, idx in enumerate(fallback_indices)}
        token_counts = ctx.token_counts
        embedding_dim = self.mm_global_cache.get_embedding_dim(ctx.modality)
        mm_embedding = torch.empty(
            (sum(token_counts), embedding_dim),
            dtype=self._embedding_dtype,
            device=self.device,
        )

        offset = 0
        copy_handles = []
        for idx, num_tokens in enumerate(token_counts):
            if idx in miss_slice_pos:
                mm_embedding[offset : offset + num_tokens].copy_(
                    new_slices[miss_slice_pos[idx]],
                    non_blocking=True,
                )
            elif idx in fallback_slice_pos:
                mm_embedding[offset : offset + num_tokens].copy_(
                    fallback_slices[fallback_slice_pos[idx]],
                    non_blocking=True,
                )
            else:
                handle = self.mm_global_cache.load_to_device_async(
                    ctx.str_mm_hashes[idx], mm_embedding, offset
                )
                if handle is None:
                    raise InternalError(
                        f"Cached embedding {ctx.str_mm_hashes[idx]} disappeared "
                        f"during assembly for req {ctx.req_id}"
                    )
                copy_handles.append(handle)
            offset += num_tokens

        self.mm_global_cache.wait_load_to_device(copy_handles)
        torch.cuda.current_stream(mm_embedding.device).synchronize()
        return mm_embedding

    async def encode_with_global_cache(
        self,
        mm_items,
        modality: Modality,
        req_id: str,
        num_parts: int,
        part_idx: int,
        hashes: Optional[List[str]] = None,
    ) -> torch.Tensor:
        ctx = await self._prepare_global_cache_context(
            mm_items, modality, req_id, hashes
        )

        missing_indices, hit_indices = await self._lookup_global_cache(ctx)
        hit_hashes = self._prefetch_global_cache_hits(ctx, hit_indices)

        new_slices = []
        if missing_indices:
            new_slices = self._encode_missing(
                ctx.mm_feature,
                ctx.mm_inputs,
                missing_indices,
                ctx.token_counts,
                ctx.modality,
                ctx.get_feature_fn,
                ctx.grid_thw,
                keep_on_gpu=True,
            )

        miss_d2h_handles = []
        if self.rank == 0 and new_slices:
            miss_hashes = [ctx.str_mm_hashes[i] for i in missing_indices]
            miss_d2h_handles = self.mm_global_cache.store_to_pool_async(
                miss_hashes, new_slices, ctx.modality
            )

        fallback_indices = await self._wait_global_cache_prefetch(
            ctx, hit_indices, hit_hashes
        )

        fallback_slices = []
        fallback_d2h_handles = []
        if fallback_indices:
            logger.info(
                f"Req {ctx.req_id}: All ranks running ViT fallback "
                f"for {len(fallback_indices)} items."
            )
            fallback_slices = self._encode_missing(
                ctx.mm_feature,
                ctx.mm_inputs,
                fallback_indices,
                ctx.token_counts,
                ctx.modality,
                ctx.get_feature_fn,
                ctx.grid_thw,
                keep_on_gpu=True,
            )
            if self.rank == 0:
                fallback_hashes = [ctx.str_mm_hashes[i] for i in fallback_indices]
                fallback_d2h_handles = self.mm_global_cache.store_to_pool_async(
                    fallback_hashes, fallback_slices, ctx.modality
                )

        if self.rank == 0:
            mm_embedding = self._assemble_global_cache_cpu(
                ctx,
                hit_indices,
                missing_indices,
                fallback_indices,
                new_slices,
                fallback_slices,
            )

            new_hashes = [ctx.str_mm_hashes[i] for i in missing_indices]
            new_hashes += [ctx.str_mm_hashes[i] for i in fallback_indices]
            self._launch_global_cache_insert(
                ctx,
                new_hashes,
                miss_d2h_handles + fallback_d2h_handles,
            )

            self.embedding_to_send[ctx.req_id] = EmbeddingData(
                ctx.req_id,
                num_parts,
                part_idx,
                ctx.grid_thw,
                ctx.modality,
                mm_embedding,
                **ctx.aux_data,
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
            ctx = await self._prepare_global_cache_context(
                mm_items, modality, req_id, hashes
            )

            nbytes, total_tokens, embedding_dim, event = (
                self._setup_mooncake_async_encode(
                    ctx.req_id,
                    num_parts,
                    part_idx,
                    ctx.grid_thw,
                    ctx.token_counts,
                    ctx.modality,
                    ctx.aux_data,
                )
            )

            # All ranks: launch background task for cache check + VIT forward.
            # Do NOT use run_in_executor: get_feature_fn relies on a session
            # context (CUDA / SGLang inference session) that is bound to the
            # event-loop main thread and is NOT available inside a
            # ThreadPoolExecutor worker thread.
            async def _run_forward_with_cache():
                try:
                    missing_indices, hit_indices = await self._lookup_global_cache(ctx)
                    hit_hashes = self._prefetch_global_cache_hits(ctx, hit_indices)

                    new_slices = []
                    if missing_indices:
                        new_slices = self._encode_missing(
                            ctx.mm_feature,
                            ctx.mm_inputs,
                            missing_indices,
                            ctx.token_counts,
                            ctx.modality,
                            ctx.get_feature_fn,
                            ctx.grid_thw,
                            keep_on_gpu=True,
                        )

                    fallback_indices = await self._wait_global_cache_prefetch(
                        ctx, hit_indices, hit_hashes
                    )

                    fallback_slices = []
                    if fallback_indices:
                        logger.info(
                            f"Req {ctx.req_id}: All ranks running ViT fallback "
                            f"for {len(fallback_indices)} items."
                        )
                        fallback_slices = self._encode_missing(
                            ctx.mm_feature,
                            ctx.mm_inputs,
                            fallback_indices,
                            ctx.token_counts,
                            ctx.modality,
                            ctx.get_feature_fn,
                            ctx.grid_thw,
                            keep_on_gpu=True,
                        )

                    if self.rank == 0:
                        d2h_handles = []
                        if new_slices:
                            miss_hashes = [
                                ctx.str_mm_hashes[i] for i in missing_indices
                            ]
                            miss_handles = self.mm_global_cache.store_to_pool_async(
                                miss_hashes, new_slices, ctx.modality
                            )
                            d2h_handles.extend(miss_handles)
                        if fallback_slices:
                            fallback_hashes = [
                                ctx.str_mm_hashes[i] for i in fallback_indices
                            ]
                            fb_handles = self.mm_global_cache.store_to_pool_async(
                                fallback_hashes, fallback_slices, ctx.modality
                            )
                            d2h_handles.extend(fb_handles)

                        mm_embedding = self._assemble_global_cache_gpu(
                            ctx,
                            missing_indices,
                            fallback_indices,
                            new_slices,
                            fallback_slices,
                        )

                        new_hashes = [ctx.str_mm_hashes[i] for i in missing_indices]
                        new_hashes += [ctx.str_mm_hashes[i] for i in fallback_indices]
                        self._launch_global_cache_insert(
                            ctx,
                            new_hashes,
                            d2h_handles,
                        )

                        self._forward_results[ctx.req_id]["embedding"] = mm_embedding
                        logger.info(
                            f"Global cache + VIT forward completed for "
                            f"{ctx.req_id}, shape={mm_embedding.shape}"
                        )
                except Exception as e:
                    logger.error(
                        f"Global cache + VIT forward failed for {ctx.req_id}: {e}"
                    )
                    if self.rank == 0:
                        self._forward_results[ctx.req_id]["error"] = str(e)
                finally:
                    if self.rank == 0:
                        event.set()
                    if self.profiler is not None:
                        self.profiler.step()

            self._launch_mooncake_background_task(_run_forward_with_cache())

            if self.rank == 0:
                logger.info(
                    f"Returning metadata immediately for {ctx.req_id}, "
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

    async def _encode(
        self, mm_items, modality: Modality, log_metrics: bool = True
    ) -> torch.Tensor:
        modality_str = modality.name.lower()
        try:
            # Single-request preprocess metrics are recorded by _process_mm_items;
            # batch_encode records its combined preprocessing separately.
            mm_inputs, _, get_feature_fn = await self._process_mm_items(
                mm_items, modality, log_metrics=log_metrics
            )
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
            use_mm_cache = self.server_args.enable_prefix_mm_cache and log_metrics
            if use_mm_cache:
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
                if encoder_metrics_collector is not None and log_metrics:
                    encoder_metrics_collector.observe_model_forward(
                        time.perf_counter() - forward_start, modality=modality_str
                    )

            # Per-request cache hit metrics: tokens = embedding rows, files = 1 item.
            if use_mm_cache and encoder_metrics_collector is not None:
                total_tokens = int(mm_embedding.shape[0])
                hit_tokens = total_tokens if cache_hit else 0
                encoder_metrics_collector.record_cache_tokens(
                    hit_tokens, total_tokens, modality=modality_str
                )
                encoder_metrics_collector.record_cache_files(
                    1 if cache_hit else 0, 1, modality=modality_str
                )

            if use_mm_cache:
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
                    if encoder_metrics_collector is not None and log_metrics:
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
            log_metrics = not is_health_check_request(req_id)
            grid_dim, mm_embedding, aux_data = await self._encode(
                mm_items, modality, log_metrics=log_metrics
            )

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
        token_counts: List[int],
        modality: Modality,
        aux_data: dict,
    ):
        """Setup metadata and event management for mooncake async encode.
        Returns (nbytes, total_tokens, embedding_dim, event)."""
        total_tokens = sum(token_counts)
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

    async def begin_or_wait_inflight_encode(
        self, req_id: str
    ) -> Tuple[bool, Optional[Tuple[int, int, int]]]:
        """Claim an encode request or wait for its owner's metadata."""
        if not hasattr(self, "_inflight_encode_events"):
            return True, None

        async with self._inflight_encode_lock:
            event = self._inflight_encode_events.get(req_id)
            if event is None:
                self._inflight_encode_events[req_id] = asyncio.Event()
                return True, None

        await event.wait()
        async with self._inflight_encode_lock:
            return False, self._inflight_encode_meta.get(req_id)

    async def complete_inflight_encode(
        self,
        req_id: str,
        metadata: Optional[Tuple[int, int, int]],
    ) -> None:
        """Publish encode metadata, or signal failure when metadata is None."""
        if not hasattr(self, "_inflight_encode_events"):
            return

        async with self._inflight_encode_lock:
            event = self._inflight_encode_events.get(req_id)
            if event is not None:
                if metadata is None:
                    self._inflight_encode_meta.pop(req_id, None)
                else:
                    self._inflight_encode_meta[req_id] = metadata
                event.set()

        if metadata is None:
            await self.release_inflight_encode(req_id)
        else:
            self._schedule_inflight_encode_cleanup(req_id)

    async def release_inflight_encode(self, req_id: str) -> None:
        """Release duplicate-request, embedding, and Mooncake forward state."""
        if not hasattr(self, "_inflight_encode_events"):
            return
        async with self._inflight_encode_lock:
            self._inflight_encode_events.pop(req_id, None)
            self._inflight_encode_meta.pop(req_id, None)
            task = self._inflight_encode_cleanup_tasks.pop(req_id, None)
            if (
                task is not None
                and task is not asyncio.current_task()
                and not task.done()
            ):
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
            await self.release_inflight_encode(req_id)

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
            mm_inputs, token_counts, get_feature_fn = await self._process_mm_items(
                mm_items, modality
            )
            grid_thw = _get_mm_grid_dim(mm_inputs, modality, self.model_type)
            if len(token_counts) != len(grid_thw):
                raise InternalError(
                    f"Preprocessor returned {len(token_counts)} token counts for "
                    f"{len(grid_thw)} {modality.name.lower()} grid entries"
                )
            aux_data = _build_mm_aux_data(mm_inputs)

            # Setup metadata and event management
            nbytes, total_tokens, embedding_dim, event = (
                self._setup_mooncake_async_encode(
                    req_id,
                    num_parts,
                    part_idx,
                    grid_thw,
                    token_counts,
                    modality,
                    aux_data,
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
                        if emb.shape[0] != total_tokens:
                            raise InternalError(
                                f"Encoder produced {emb.shape[0]} tokens, but "
                                f"preprocessor metadata expected {total_tokens}"
                            )
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
        try:
            preprocess_start = time.perf_counter()
            mm_inputs, token_counts, items_per_req = (
                await self.preprocessor.process_batch_mm_items(requests, modality)
            )
            if encoder_metrics_collector is not None:
                encoder_metrics_collector.observe_preprocess(
                    time.perf_counter() - preprocess_start,
                    modality=modality.name.lower(),
                )
            target = (
                self.model.thinker if hasattr(self.model, "thinker") else self.model
            )
            get_feat = getattr(target, f"get_{modality.name.lower()}_feature")
        except NotImplementedError as e:
            return self._batch_set_error(
                requests, modality, InternalError(f"Not implemented error: {e}")
            )
        except Exception as e:
            return self._batch_set_error(
                requests, modality, BadRequestError(f"Failed to process mm items: {e}")
            )

        if len(items_per_req) != len(requests) or any(
            isinstance(n, bool) or not isinstance(n, int) or n <= 0
            for n in items_per_req
        ):
            return self._batch_set_error(
                requests,
                modality,
                InternalError(
                    f"Invalid batch layout {items_per_req} for {len(requests)} requests"
                ),
            )

        total = sum(items_per_req)
        if len(token_counts) != total:
            return self._batch_set_error(
                requests,
                modality,
                InternalError(
                    f"Preprocessor returned {len(token_counts)} token counts for "
                    f"batch layout {items_per_req}"
                ),
            )
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
            mm_feature = _convert(_get_mm_feature(mm_inputs, modality))
            grid_dim = _get_mm_grid_dim(mm_inputs, modality, self.model_type)
            if len(grid_dim) != total:
                return self._batch_set_error(
                    requests,
                    modality,
                    InternalError(
                        f"Grid count mismatch for {self.model_type}/"
                        f"{modality.name}: {len(requests)} requests expected "
                        f"{total} grids "
                        f"(per-req {items_per_req}), but processor produced "
                        f"{len(grid_dim)}. Processor batch layout must report "
                        f"one entry per encoder grid."
                    ),
                )

            final_slices = self._encode_missing(
                mm_feature,
                mm_inputs,
                list(range(total)),
                token_counts,
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


async def run_encoder(
    server_args: ServerArgs, schedule_path, dist_init_method, rank: int
):
    encoder = MMEncoder(server_args, schedule_path, dist_init_method, rank)
    while True:
        request = await async_sock_recv(encoder.schedule_socket)
        await _handle_encoder_worker_request(encoder, request)


async def _handle_encoder_worker_request(encoder: MMEncoder, request):
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
    elif (
        isinstance(request, dict)
        and isinstance(request.get("req_id"), str)
        and request["req_id"].startswith(HEALTH_CHECK_RID_PREFIX)
    ):
        await encoder.encode(
            mm_items=request["mm_items"],
            modality=Modality.from_str(request["modality"]),
            req_id=request["req_id"],
            num_parts=request["num_parts"],
            part_idx=request["part_idx"],
            hashes=request.get("hashes"),
        )
    else:
        await encoder.encode_request(request, Modality.from_str(request["modality"]))


def launch_encoder(server_args, schedule_path, dist_init_method, rank):
    try:
        asyncio.run(run_encoder(server_args, schedule_path, dist_init_method, rank))
    except KeyboardInterrupt:
        logger.info(f"Exit rank {rank}")
    except Exception:
        traceback.print_exc()


# Per-process encoder metrics collector. Set by encode_http_server in
# launch_server (non-DP) and run_dp_worker (DP mode). None when metrics
# disabled. Kept here because MMEncoder GPU methods reference it directly.
encoder_metrics_collector: Optional[EncoderMetricsCollector] = None
