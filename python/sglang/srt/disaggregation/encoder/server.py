import asyncio
import concurrent.futures
import ctypes
import logging
import os
import pickle
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set, Tuple

import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.encoder.preprocessor import (
    EncoderPreprocessor,
    EncoderPreprocessResult,
    _convert,
    _mm_grid_attrs,
)
from sglang.srt.disaggregation.encoder.receiver import (
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
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    maybe_precompile_model_kernels_after_loading,
)
from sglang.srt.model_loader import get_model as load_model
from sglang.srt.multimodal.encoder_preprocessing import (
    get_encoder_preprocessed_items,
    resolve_encoder_media_processor_config,
)
from sglang.srt.observability.metrics_collector import EncoderMetricsCollector
from sglang.srt.runtime_context import (
    assert_published,
    get_device,
    get_disagg,
    get_exec,
    get_mm,
    get_model,
    get_parallel,
    publish,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import configure_media_url_security
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
rid_to_receive_endpoint: Dict[str, Set[str]] = dict()
rid_to_receive_count: Dict[str, int] = dict()
cond_dict_lock = asyncio.Lock()
rid_to_cond: Dict[str, asyncio.Condition] = {}


async def _get_receive_condition(req_id: str) -> asyncio.Condition:
    async with cond_dict_lock:
        if req_id not in rid_to_cond:
            rid_to_cond[req_id] = asyncio.Condition()
        return rid_to_cond[req_id]


ENCODER_MAX_BATCH_SIZE = envs.SGLANG_ENCODER_MAX_BATCH_SIZE.get()
ENCODER_MAX_BATCH_SIZE_EXPLICIT = envs.SGLANG_ENCODER_MAX_BATCH_SIZE.is_set()
# Watchdog: max time to wait for a batched /encode result. Bounds HTTP latency
# if the batch worker stalls (NCCL hang, dead worker proc, etc.).
ENCODER_REQ_TIMEOUT = envs.SGLANG_ENCODER_REQ_TIMEOUT.get()


class EncoderMetaRegistry:
    """Per-part metadata shared by every encoder request lifecycle.

    Mooncake decoder ranks consume it early to allocate landing buffers. ZMQ
    publishes the same state for a uniform pipeline but does not consume it
    before encode/send completes.
    """

    def __init__(self, *, wait_timeout: float, sweep_timeout: float):
        # How long a decoder blocks in /scheduler_receive_meta_data.
        self.wait_timeout = wait_timeout
        # Backstop for state whose /send calls never all land.
        self.sweep_timeout = sweep_timeout
        self._rid_to_meta: Dict[str, dict] = {}
        self._rid_to_send_done: Dict[str, int] = {}
        self._pending_at: Dict[str, float] = {}
        self._sweeper_task: Optional[asyncio.Task] = None
        # Set only where the embedding also lives; None in the DP main process.
        self.on_release: Optional[Callable[[str], Awaitable[None]]] = None

    def _touch(self, req_id: str) -> None:
        self._pending_at[req_id] = time.monotonic()
        self._ensure_sweeper()

    def _ensure_sweeper(self) -> None:
        if self._sweeper_task is not None and not self._sweeper_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._sweeper_task = loop.create_task(self._sweep_loop())

    async def _sweep_loop(self) -> None:
        # Same idiom as DPDispatcher._cleanup_stale_mappings: one eternal
        # scanner; interval re-read each pass so the MMEncoder override applies.
        while True:
            await asyncio.sleep(max(self.sweep_timeout / 4, 0.01))
            now = time.monotonic()
            stale = [
                rid
                for rid, ts in self._pending_at.items()
                if now - ts > self.sweep_timeout
            ]
            for rid in stale:
                await self._release(rid)

    async def publish(
        self,
        req_id: str,
        nbytes: int,
        embedding_len: int,
        embedding_dim: int,
        error: Optional[str] = None,
    ) -> None:
        """Publish per-part metadata (or an error), wake waiters, arm the sweep."""
        meta = (
            {"error": error}
            if error is not None
            else {
                "embedding_size": nbytes,
                "embedding_len": embedding_len,
                "embedding_dim": embedding_dim,
            }
        )
        async with rid_lock:
            self._rid_to_meta[req_id] = meta
            self._touch(req_id)
        cond = await _get_receive_condition(req_id)
        async with cond:
            cond.notify_all()

    async def wait(self, req_id: str) -> Optional[dict]:
        """Block until req_id's metadata is published; TimeoutError past wait_timeout.
        No _touch here: a pull-first timestamp would let the sweeper pop the very
        Condition this waiter holds, stranding it when publish notifies a new one."""
        cond = await _get_receive_condition(req_id)
        async with cond:
            await asyncio.wait_for(
                cond.wait_for(lambda: self._rid_to_meta.get(req_id) is not None),
                timeout=self.wait_timeout,
            )
        return self._rid_to_meta.get(req_id)

    async def note_send_done(self, req_id: str, receive_count: int) -> None:
        """Count one completed ``/send``; release everything at receive_count."""
        async with rid_lock:
            count = self._rid_to_send_done.get(req_id, 0) + 1
            self._rid_to_send_done[req_id] = count
        if count >= receive_count:
            await self._release(req_id)

    async def _release(self, req_id: str) -> None:
        if self.on_release is not None:
            await self.on_release(req_id)
        await self.discard(req_id)

    async def discard(self, req_id: str) -> None:
        """Drop the meta rendezvous state for req_id. Idempotent."""
        async with rid_lock:
            self._rid_to_meta.pop(req_id, None)
            self._rid_to_send_done.pop(req_id, None)
            self._pending_at.pop(req_id, None)
        async with cond_dict_lock:
            rid_to_cond.pop(req_id, None)


meta_registry = EncoderMetaRegistry(
    wait_timeout=ENCODER_REQ_TIMEOUT,
    sweep_timeout=envs.SGLANG_ENCODER_SEND_TIMEOUT.get(),
)


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


class EncodeContext(msgspec.Struct):
    """One flattened encode batch; a single request is the N=1 case."""

    req_id: str  # first request's id, for cache prefetch keys and logs
    modality: Modality
    preprocess_result: EncoderPreprocessResult
    get_feature_fn: Any
    mm_feature: Any
    num_items: int
    items_per_req: List[int]  # grid entries per request, in flatten order
    aux_data: dict
    str_mm_hashes: Optional[List[str]]
    use_global_cache: bool
    is_health_check: bool


@dataclass
class ReqState:
    """The result and in-flight work for one encoder request."""

    req_id: str
    embedding_data: Optional[EmbeddingData] = None
    active_encodes: int = 0
    active_sends: int = 0
    release_requested: bool = False
    preserve_metadata_on_release: bool = False
    embedding_ready: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    lifecycle_condition: asyncio.Condition = field(
        default_factory=asyncio.Condition, repr=False
    )


@dataclass(frozen=True)
class SendDestination:
    """One normalized destination for exactly one transfer."""

    endpoint: str
    session_id: Optional[str] = None
    buffer_address: Optional[int] = None

    @classmethod
    def from_host_port(
        cls,
        prefill_host: str,
        embedding_port: int,
        *,
        session_id: Optional[str] = None,
        buffer_address: Optional[int] = None,
    ) -> "SendDestination":
        return cls(
            endpoint=NetworkAddress(prefill_host, embedding_port).to_host_port_str(),
            session_id=session_id,
            buffer_address=buffer_address,
        )

    @classmethod
    def from_url(cls, url: str) -> "SendDestination":
        return cls(endpoint=NetworkAddress.parse(url).to_host_port_str())


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


class EncoderDelivery(ABC):
    """Transfer backend boundary. Send never releases the request."""

    def __init__(self, encoder: "MMEncoder"):
        self.encoder = encoder

    @abstractmethod
    async def send(
        self,
        state: ReqState,
        destination: SendDestination,
    ) -> None: ...

    @abstractmethod
    async def release(self, state: ReqState) -> None: ...


class MooncakeDelivery(EncoderDelivery):
    async def send(
        self,
        state: ReqState,
        destination: SendDestination,
    ) -> None:
        mm_data = await self.encoder._wait_for_embedding(state)
        await self.encoder._send(
            mm_data.embedding,
            mm_data,
            session_id=destination.session_id,
            buffer_address=destination.buffer_address,
            url=destination.endpoint,
        )

    async def release(self, state: ReqState) -> None:
        mm_data = state.embedding_data
        if mm_data is not None and mm_data._mr_ptr is not None:
            try:
                self.encoder.engine.deregister(mm_data._mr_ptr)
            except Exception as dereg_err:
                logger.warning(
                    f"Shared-MR deregister failed for {state.req_id}: {dereg_err}"
                )
            finally:
                mm_data._mr_ptr = None


class ZmqDelivery(EncoderDelivery):
    def __init__(self, encoder: "MMEncoder", *, cleanup_receive_state: bool) -> None:
        super().__init__(encoder)
        self.cleanup_receive_state = cleanup_receive_state

    async def send(
        self,
        state: ReqState,
        destination: SendDestination,
    ) -> None:
        mm_data = await self.encoder._wait_for_embedding(state)
        await self.encoder._send(mm_data.embedding, mm_data, url=destination.endpoint)

    async def release(self, state: ReqState) -> None:
        if not self.cleanup_receive_state:
            return
        async with rid_lock:
            rid_to_receive_endpoint.pop(state.req_id, None)
            rid_to_receive_count.pop(state.req_id, None)
        async with cond_dict_lock:
            rid_to_cond.pop(state.req_id, None)


_mm_feature_attrs = {
    Modality.IMAGE: ["pixel_values"],
    Modality.VIDEO: ["pixel_values_videos"],
    Modality.AUDIO: ["input_features"],
}


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
    aux = {
        attr: _normalize_aux_value(mm_inputs.get(attr))
        for attr in video_meta_attrs_for(model_type)
    }
    if model_type == "kimi_k3":
        aux["original_image_sizes"] = _normalize_aux_value(
            mm_inputs.get("original_image_sizes")
        )
    return aux


class MMEncoder:
    def __init__(
        self,
        server_args: ServerArgs,
        schedule_path=None,
        dist_init_method=None,
        rank: int = 0,
        gpu_id: Optional[int] = None,
    ):
        """``gpu_id`` pins this encoder to a device other than
        ``base_gpu_id + rank`` — the DP launcher's per-worker placement. It is
        this instance's value, not a config change, so it travels as an
        argument."""
        assert_published(server_args, role="encoder")
        logger.info(f"init MMEncoder {rank}/{get_parallel().tp_size}")
        self.server_args = server_args
        configure_media_url_security(
            get_mm().allowed_media_domains,
            server_args.media_url_max_file_size_mb,
        )
        self.transfer_backend = get_disagg().encoder_transfer_backend
        self.use_mooncake = self.transfer_backend == "mooncake"
        self.rank = rank
        # DP rank for metric labels; overridden by runtime.run_dp_worker.
        # 0 in the single-instance (non-DP) path.
        self.dp_rank = 0
        self.profiler = EncoderProfiler(rank)

        self.model_config = ModelConfig.from_server_args(
            server_args,
        )
        self.load_config = LoadConfig(
            load_format=get_model().load_format,
            download_dir=server_args.download_dir,
            model_loader_extra_config=get_model().model_loader_extra_config,
            remote_instance_weight_loader_seed_instance_ip=server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=server_args.remote_instance_weight_loader_send_weights_group_ports,
        )
        self.model_type = getattr(
            self.model_config.hf_config, "model_type", "unknown"
        ).lower()

        self.device = get_device().device
        self.gpu_id = server_args.base_gpu_id + rank if gpu_id is None else gpu_id

        self.device_config = DeviceConfig(
            device=self.device,
            gpu_id=self.gpu_id,
        )

        torch.get_device_module(self.device).set_device(self.gpu_id)

        init_distributed_environment(
            backend=get_default_distributed_backend(self.device),
            world_size=get_parallel().tp_size,
            rank=rank,
            distributed_init_method=dist_init_method,
            local_rank=rank,
        )
        initialize_model_parallel(tensor_model_parallel_size=get_parallel().tp_size)
        initialize_dp_attention(server_args, self.model_config)

        self.model = load_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=self.device_config,
        )
        encoder_media_processor_config = resolve_encoder_media_processor_config(
            self.model
        )
        maybe_precompile_model_kernels_after_loading(self.model, self.device)

        # CPU preprocessing pipeline (Rust-replaceable)
        self.preprocessor = EncoderPreprocessor(
            server_args=server_args,
            model_config=self.model_config,
            model_preprocessor=getattr(self.model, "preprocess_mm_for_encoder", None),
            encoder_media_processor_config=encoder_media_processor_config,
        )

        self.context = zmq.asyncio.Context(2)
        self.sync_context = zmq.Context()  # Reuse sync context for thread pool
        self.scheduler_send_sockets = {}
        self.scheduler_send_locks = {}
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
        self._embedding_dims = self._infer_embedding_dims()

        if get_mm().enable_mm_global_cache:
            from sglang.srt.mem_cache.embedding_cache_controller import (
                EmbeddingCacheController,
            )
            from sglang.srt.mem_cache.embedding_store import EmbeddingStoreFactory

            embedding_store = EmbeddingStoreFactory.create_backend(
                get_mm().mm_global_cache_backend,
            )
            self.mm_global_cache = EmbeddingCacheController(
                rank,
                get_parallel().tp_size,
                embedding_store=embedding_store,
                hidden_dims=self._embedding_dims,
                tp_group=get_tp_group().cpu_group,
                all_rank_get=False,
                dtype=self._embedding_dtype,
            )
        else:
            self.mm_global_cache = None

        if self.rank == 0:
            logger.info(
                f"Using transfer backend: {get_disagg().encoder_transfer_backend}"
            )

            if get_disagg().encoder_transfer_backend == "mooncake":
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
                            get_disagg().disaggregation_ib_device
                            or get_exec().moe.mooncake_ib_device
                        ),
                    )

            self.req_states: Dict[str, ReqState] = {}
            # Need to ensure the NCCL launch order on rank0 matches the dispatch order rank>0
            self.encode_dispatch_lock = asyncio.Lock()

            if get_disagg().encoder_transfer_backend == "mooncake":
                self.delivery = MooncakeDelivery(self)
                # Embeddings live here, so registry cleanup uses the common release.
                meta_registry.on_release = self.release_request
                meta_registry.sweep_timeout = self.send_timeout
            else:
                self.delivery = ZmqDelivery(
                    self,
                    cleanup_receive_state=(
                        get_disagg().encoder_transfer_backend == "zmq_to_scheduler"
                    ),
                )

        logger.info(f"rank {rank} init finish ")

    def supports_modality(self, modality: Modality) -> bool:
        return self.preprocessor.supports_modality(modality)

    def has_pending_embeddings(self) -> bool:
        return bool(getattr(self, "req_states", None))

    def _require_active_encode_state(self, req_id: str) -> ReqState:
        """Return the state holding an encode ref; never resurrect a request."""
        state = self.req_states.get(req_id)
        if state is None:
            raise InternalError(
                f"No request state exists while encoding request: {req_id}"
            )
        if state.active_encodes <= 0:
            raise InternalError(f"Request state has no active encode work: {req_id}")
        return state

    def _acquire_encode_ref(self, req_id: str) -> Optional[ReqState]:
        """Acquire a rank 0 encode ref before preprocessing can suspend."""
        if self.rank != 0:
            return None
        state = self.req_states.get(req_id)
        if state is None:
            state = ReqState(req_id)
            self.req_states[req_id] = state
        state.active_encodes += 1
        return state

    async def _release_encode_ref(self, state: Optional[ReqState]) -> None:
        if state is None:
            return
        async with state.lifecycle_condition:
            state.active_encodes -= 1
            assert state.active_encodes >= 0
            should_release = state.release_requested and state.active_encodes == 0
            state.lifecycle_condition.notify_all()
        if should_release:
            await self.release_request(state.req_id)

    def _stage_embedding(self, mm_data: EmbeddingData) -> None:
        state = self._require_active_encode_state(mm_data.req_id)
        metadata = state.embedding_data
        if (
            metadata is not None
            and metadata.embedding is None
            and mm_data.embedding is not None
            and (metadata.shape != mm_data.shape or metadata.dtype != mm_data.dtype)
        ):
            raise InternalError(
                f"Embedding metadata mismatch for {mm_data.req_id}: "
                f"expected={metadata.shape}/{metadata.dtype}, "
                f"actual={mm_data.shape}/{mm_data.dtype}"
            )
        state.embedding_data = mm_data
        state.embedding_ready.set()

    async def _wait_for_embedding(self, state: ReqState) -> EmbeddingData:
        await state.embedding_ready.wait()
        if state.embedding_data is None:
            raise InternalError(f"No embedding available for request: {state.req_id}")
        return state.embedding_data

    async def send_to_destination(
        self, state: ReqState, destination: SendDestination
    ) -> None:
        async with state.lifecycle_condition:
            if (
                self.req_states.get(state.req_id) is not state
                or state.release_requested
            ):
                raise InternalError(f"Encoder request was released: {state.req_id}")
            state.active_sends += 1
        try:
            await self.delivery.send(state, destination)
        finally:
            async with state.lifecycle_condition:
                state.active_sends -= 1
                state.lifecycle_condition.notify_all()

    async def release_request(
        self, req_id: str, *, preserve_metadata: bool = False
    ) -> None:
        """Release backend resources, then the embedding, through one path."""
        state = self.req_states.get(req_id)
        if state is None:
            if not preserve_metadata:
                await meta_registry.discard(req_id)
            return
        async with state.lifecycle_condition:
            state.release_requested = True
            state.preserve_metadata_on_release |= preserve_metadata
            if state.active_encodes > 0:
                return
            await state.lifecycle_condition.wait_for(lambda: state.active_sends == 0)
            if self.req_states.get(req_id) is not state:
                return
            self.req_states.pop(req_id, None)
        await self.delivery.release(state)
        state.embedding_data = None
        if not state.preserve_metadata_on_release:
            await meta_registry.discard(req_id)

    async def register_embedding_destinations(
        self,
        req_id: str,
        expected_destination_count: int,
        destination_urls: Iterable[str],
    ) -> None:
        async with rid_lock:
            if req_id not in rid_to_receive_endpoint:
                rid_to_receive_endpoint[req_id] = set()
                rid_to_receive_count[req_id] = expected_destination_count
            registered_count = rid_to_receive_count[req_id]
            if registered_count != expected_destination_count:
                raise BadRequestError(
                    f"Inconsistent receive_count for req_id={req_id}: "
                    f"registered {registered_count}, got {expected_destination_count}"
                )
            rid_to_receive_endpoint[req_id].update(destination_urls)

        cond = await _get_receive_condition(req_id)
        async with cond:
            cond.notify_all()

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

    def slice_embedding(
        self,
        mm_embedding: torch.Tensor,
        token_counts: Iterable[int],
    ) -> List[torch.Tensor]:
        """Slice embeddings using preprocessing-owned token counts."""
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
        self, mm_feature, grid_thw: List, modality: Modality, mm_inputs=None
    ) -> List[int]:
        """CPU Task: Compute hashes based on processed feature patches."""
        preprocessed_items = (
            get_encoder_preprocessed_items(mm_inputs) if mm_inputs is not None else None
        )
        if preprocessed_items is not None:
            if len(preprocessed_items) != len(grid_thw):
                raise ValueError(
                    "Encoder preprocess item/grid mismatch: "
                    f"{len(preprocessed_items)} items != {len(grid_thw)} grids"
                )
            hashes = []
            for item in preprocessed_items:
                item.set_pad_value()
                hashes.append(item.hash)
            return hashes

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
            num_patches = self.preprocessor.get_num_patches(grid, modality)
            feature_slice = mm_feature[offset : offset + num_patches]
            tmp_item = MultimodalDataItem(modality=modality, feature=feature_slice)
            tmp_item.set_pad_value()
            hashes.append(tmp_item.hash)
            offset += num_patches
        return hashes

    def _encode_missing(
        self,
        mm_feature,
        preprocess_result: EncoderPreprocessResult,
        indices: List[int],
        modality: Modality = Modality.IMAGE,
        get_feature_fn=None,
    ) -> List[torch.Tensor]:
        """
        GPU Task: Run ViT inference ONLY on the subset of mm items missing from the cache.
        """
        token_counts = preprocess_result.token_counts
        mm_items = self._build_model_mm_items(
            mm_feature, preprocess_result, indices, modality
        )

        forward_start = time.perf_counter()
        with torch.inference_mode():
            new_embeddings = get_feature_fn(mm_items)
            if new_embeddings.ndim != 2:
                new_embeddings = new_embeddings.reshape(-1, new_embeddings.shape[-1])
        if encoder_metrics_collector is not None:
            encoder_metrics_collector.observe_model_forward(
                time.perf_counter() - forward_start, modality=modality.name.lower()
            )

        return self.slice_embedding(new_embeddings, (token_counts[i] for i in indices))

    def _build_model_mm_items(
        self,
        mm_feature,
        preprocess_result: EncoderPreprocessResult,
        indices: List[int],
        modality: Modality,
    ) -> List[MultimodalDataItem]:
        """Build the model-facing items selected for one encoder forward.

        Model preprocessors can preserve an item-wise representation with
        ``EncoderPreprocessOutput``. This avoids concatenating and re-slicing
        features before encoder-DP knows which rank owns each item. Legacy
        processor outputs retain their aggregate tensor behavior.
        """
        mm_inputs = preprocess_result.mm_inputs
        grid_thw = preprocess_result.grid_thw
        preprocessed_items = get_encoder_preprocessed_items(mm_inputs)
        if preprocessed_items is not None:
            if len(preprocessed_items) != len(grid_thw):
                raise ValueError(
                    "Encoder preprocess item/grid mismatch: "
                    f"{len(preprocessed_items)} items != {len(grid_thw)} grids"
                )
            selected = [preprocessed_items[index] for index in indices]
            if any(item.modality != modality for item in selected):
                raise ValueError("Encoder preprocess output contains wrong modality")
            return selected

        split_kimi_k3_images = (
            self.model_type == "kimi_k3" and modality == Modality.IMAGE
        )

        if modality == Modality.AUDIO:
            if isinstance(mm_feature, list):
                sub_feature = [mm_feature[i] for i in indices]
            else:
                sub_feature = mm_feature[list(indices)]
        else:
            feature_slices = []
            offsets = [0]
            curr = 0
            for grid in grid_thw:
                curr += self.preprocessor.get_num_patches(grid, modality)
                offsets.append(curr)
            for idx in indices:
                feature_slices.append(mm_feature[offsets[idx] : offsets[idx + 1]])
            if not split_kimi_k3_images:
                sub_feature = torch.cat(feature_slices, dim=0)

        if split_kimi_k3_images:
            mm_items = [
                MultimodalDataItem.from_dict(
                    {"modality": modality, "feature": _convert(feature)}
                )
                for feature in feature_slices
            ]
        else:
            mm_items = [
                MultimodalDataItem.from_dict(
                    {
                        "modality": modality,
                        "feature": (
                            sub_feature
                            if isinstance(sub_feature, list)
                            else _convert(sub_feature)
                        ),
                    }
                )
            ]

        for key, value in mm_inputs.items():
            if key in _mm_feature_attrs.get(modality, []):
                continue
            value = _convert(value)
            if key in _mm_grid_attrs.get(modality, []):
                if split_kimi_k3_images:
                    for mm_item, idx in zip(mm_items, indices):
                        mm_item.set(key, value[idx : idx + 1])
                else:
                    mm_items[0].set(key, value[indices])
            else:
                for mm_item in mm_items:
                    mm_item.set(key, value)
        return mm_items

    async def _prepare_encode_context(
        self,
        requests: List[dict],
        modality: Modality,
        *,
        use_global_cache: bool,
        is_health_check: bool = False,
    ) -> EncodeContext:
        """Flatten a batch of requests into one EncodeContext (single = N of 1)."""
        modality_str = modality.name.lower()
        preprocess_start = time.perf_counter()
        try:
            preprocess_result, items_per_req = (
                await self.preprocessor.process_batch_mm_items(requests, modality)
            )
        except NotImplementedError as e:
            raise InternalError(f"Not implemented error: {str(e)}")
        except Exception as e:
            raise BadRequestError(f"Failed to process mm items: {str(e)}")

        if len(items_per_req) != len(requests) or any(n <= 0 for n in items_per_req):
            raise InternalError(
                f"Invalid batch layout {items_per_req} for {len(requests)} requests"
            )

        if encoder_metrics_collector is not None and not is_health_check:
            encoder_metrics_collector.observe_preprocess(
                time.perf_counter() - preprocess_start,
                modality=modality_str,
            )
            for item_count in items_per_req:
                encoder_metrics_collector.observe_mm_items_per_request(
                    item_count, modality=modality_str
                )
            encoder_metrics_collector.observe_mm_items_per_batch(
                sum(items_per_req), modality=modality_str
            )
        target = self.model.thinker if hasattr(self.model, "thinker") else self.model
        get_feature_fn = getattr(target, f"get_{modality_str}_feature")

        mm_inputs = preprocess_result.mm_inputs
        grid_thw = preprocess_result.grid_thw
        token_counts = preprocess_result.token_counts
        mm_feature = _convert(_get_mm_feature(mm_inputs, modality))
        num_items = len(grid_thw)
        if num_items != sum(items_per_req):
            raise InternalError(
                f"Batch layout {items_per_req} expects {sum(items_per_req)} "
                f"grids, but the processor produced {num_items}"
            )
        if len(token_counts) != num_items:
            raise InternalError(
                f"Preprocessor returned {len(token_counts)} token counts for "
                f"{num_items} {modality_str} grid entries"
            )

        str_mm_hashes = None
        if use_global_cache:
            # Hashes must be grid-space per request (a leaf-space list would
            # size-mismatch rank>0's mask and deadlock TP); validate on every
            # rank so a bad request fails symmetrically before any collective.
            per_req_hashes = [req.get("hashes") for req in requests]
            mm_hashes = None
            if all(h is not None for h in per_req_hashes):
                for req, hashes, n in zip(requests, per_req_hashes, items_per_req):
                    if len(hashes) != n:
                        raise BadRequestError(
                            f"User-supplied hashes length {len(hashes)} != grid "
                            f"count {n} for req {req['req_id']}; hashes must be "
                            f"grid-space (1 per encoder grid entry)."
                        )
                mm_hashes = [h for hashes in per_req_hashes for h in hashes]
            if self.rank == 0:
                if mm_hashes is None:
                    mm_hashes = self._calculate_hashes_from_features(
                        mm_feature, grid_thw, modality, mm_inputs
                    )
                # Embedding stores use string cache keys.
                str_mm_hashes = [str(h) for h in mm_hashes]

        return EncodeContext(
            req_id=requests[0]["req_id"],
            modality=modality,
            preprocess_result=preprocess_result,
            get_feature_fn=get_feature_fn,
            mm_feature=mm_feature,
            num_items=num_items,
            items_per_req=items_per_req,
            aux_data=_build_mm_aux_data(mm_inputs, self.model_type),
            str_mm_hashes=str_mm_hashes,
            use_global_cache=use_global_cache,
            is_health_check=is_health_check,
        )

    def _broadcast_global_cache_mask(self, mask_tensor: torch.Tensor):
        if get_parallel().tp_size > 1:
            torch.distributed.broadcast(
                mask_tensor,
                src=0,
                group=self.mm_global_cache.prefetch_tp_group,
            )

    async def _lookup_global_cache(
        self,
        ctx: EncodeContext,
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
        ctx: EncodeContext,
        hit_indices: List[int],
    ) -> List[str]:
        if self.rank != 0 or not hit_indices:
            return []

        hit_hashes = [ctx.str_mm_hashes[i] for i in hit_indices]
        hit_tokens = [ctx.preprocess_result.token_counts[i] for i in hit_indices]
        self.mm_global_cache.prefetch(ctx.req_id, hit_hashes, hit_tokens, ctx.modality)
        return hit_hashes

    async def _wait_global_cache_prefetch(
        self,
        ctx: EncodeContext,
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
        ctx: EncodeContext,
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
        ctx: EncodeContext,
        hit_indices: List[int],
        missing_indices: List[int],
        fallback_indices: List[int],
        new_slices: List[torch.Tensor],
        fallback_slices: List[torch.Tensor],
    ) -> torch.Tensor:
        miss_slice_pos = {idx: pos for pos, idx in enumerate(missing_indices)}
        fallback_slice_pos = {idx: pos for pos, idx in enumerate(fallback_indices)}
        fallback_index_set = set(fallback_indices)
        token_counts = ctx.preprocess_result.token_counts
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
        ctx: EncodeContext,
        missing_indices: List[int],
        fallback_indices: List[int],
        new_slices: List[torch.Tensor],
        fallback_slices: List[torch.Tensor],
    ) -> torch.Tensor:
        miss_slice_pos = {idx: pos for pos, idx in enumerate(missing_indices)}
        fallback_slice_pos = {idx: pos for pos, idx in enumerate(fallback_indices)}
        token_counts = ctx.preprocess_result.token_counts
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

    async def _compute_global_cache_embedding(
        self,
        ctx: EncodeContext,
        *,
        keep_on_gpu: bool,
    ) -> Optional[torch.Tensor]:
        """Resolve cache hits, compute misses, assemble output, and insert misses."""
        missing_indices, hit_indices = await self._lookup_global_cache(ctx)
        hit_hashes = self._prefetch_global_cache_hits(ctx, hit_indices)

        new_slices = []
        if missing_indices:
            new_slices = self._encode_missing(
                ctx.mm_feature,
                ctx.preprocess_result,
                missing_indices,
                ctx.modality,
                ctx.get_feature_fn,
            )

        miss_d2h_handles = []
        # The CPU output path starts D2H staging before waiting for cache-hit loads.
        if self.rank == 0 and new_slices and not keep_on_gpu:
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
                ctx.preprocess_result,
                fallback_indices,
                ctx.modality,
                ctx.get_feature_fn,
            )
            if self.rank == 0 and not keep_on_gpu:
                fallback_hashes = [ctx.str_mm_hashes[i] for i in fallback_indices]
                fallback_d2h_handles = self.mm_global_cache.store_to_pool_async(
                    fallback_hashes, fallback_slices, ctx.modality
                )

        if self.rank == 0:
            if keep_on_gpu:
                # Start staging newly computed GPU slices into the CPU cache
                # pool asynchronously before assembling the GPU output.
                if new_slices:
                    miss_hashes = [ctx.str_mm_hashes[i] for i in missing_indices]
                    miss_d2h_handles = self.mm_global_cache.store_to_pool_async(
                        miss_hashes, new_slices, ctx.modality
                    )
                if fallback_slices:
                    fallback_hashes = [ctx.str_mm_hashes[i] for i in fallback_indices]
                    fallback_d2h_handles = self.mm_global_cache.store_to_pool_async(
                        fallback_hashes, fallback_slices, ctx.modality
                    )
                mm_embedding = self._assemble_global_cache_gpu(
                    ctx,
                    missing_indices,
                    fallback_indices,
                    new_slices,
                    fallback_slices,
                )
            else:
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
            return mm_embedding

        return None

    async def _compute_direct_embedding(
        self,
        ctx: EncodeContext,
        *,
        keep_on_gpu: bool,
    ) -> torch.Tensor:
        """Compute without global cache, optionally using the prefix MM cache."""
        modality = ctx.modality
        modality_str = modality.name.lower()
        try:
            mm_embedding = None
            mm_hash = None

            mm_items = self._build_model_mm_items(
                ctx.mm_feature,
                ctx.preprocess_result,
                list(range(ctx.num_items)),
                modality,
            )

            cache_hit = False
            # The prefix cache hashes the whole request; a fused multi-request
            # batch has no per-request key, so only N=1 contexts use it.
            use_mm_cache = (
                get_mm().enable_prefix_mm_cache
                and not ctx.is_health_check
                and not keep_on_gpu
                and len(ctx.items_per_req) == 1
            )
            if use_mm_cache:
                for mm_item in mm_items:
                    mm_item.set_pad_value()
                mm_hashes = [mm_item.hash for mm_item in mm_items]
                mm_hash = MultiModalStaticCache.combine_hashes(mm_hashes)
                async with self.mm_cache_lock:
                    mm_cache = self.mm_cache.get(mm_hashes)
                    if mm_cache is not None:
                        mm_embedding = mm_cache.embedding
                        cache_hit = True

            if mm_embedding is None:
                forward_start = time.perf_counter()
                with torch.inference_mode():
                    mm_embedding: torch.Tensor = ctx.get_feature_fn(mm_items)
                    if not keep_on_gpu:
                        mm_embedding = mm_embedding.cpu()
                if len(mm_embedding.shape) != 2:
                    mm_embedding = mm_embedding.reshape(-1, mm_embedding.shape[-1])
                if encoder_metrics_collector is not None and not ctx.is_health_check:
                    encoder_metrics_collector.observe_model_forward(
                        time.perf_counter() - forward_start, modality=modality_str
                    )

            # Per-request cache hit metrics: tokens = embedding rows.
            if use_mm_cache and encoder_metrics_collector is not None:
                total_tokens = int(mm_embedding.shape[0])
                hit_tokens = total_tokens if cache_hit else 0
                encoder_metrics_collector.record_cache_tokens(
                    hit_tokens, total_tokens, modality=modality_str
                )
                encoder_metrics_collector.record_cache_files(
                    len(mm_items) if cache_hit else 0,
                    len(mm_items),
                    modality=modality_str,
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

            if (
                not keep_on_gpu
                and modality == Modality.VIDEO
                and ctx.preprocess_result.mm_inputs.get("video_audio_features")
            ):
                target = (
                    self.model.thinker if hasattr(self.model, "thinker") else self.model
                )
                encode_video_audio_fn = getattr(target, "encode_video_audio", None)
                if encode_video_audio_fn is not None:
                    audio_forward_start = time.perf_counter()
                    audio_embedding = encode_video_audio_fn(
                        ctx.preprocess_result.mm_inputs
                    )
                    if (
                        encoder_metrics_collector is not None
                        and not ctx.is_health_check
                    ):
                        encoder_metrics_collector.observe_model_forward(
                            time.perf_counter() - audio_forward_start, modality="audio"
                        )
                    if audio_embedding is not None:
                        ctx.aux_data["video_audio_embedding"] = audio_embedding
                else:
                    logger.warning(
                        "Videos carry audio tracks but model has no "
                        "encode_video_audio; dropping audio for EPD encoding."
                    )

            return mm_embedding
        except BadRequestError as e:
            raise BadRequestError(f"Bad request error: {str(e)}")
        except Exception as e:
            raise InternalError(f"Internal encoding error: {str(e)}")

    async def _compute_embedding(
        self,
        ctx: EncodeContext,
        *,
        keep_on_gpu: bool,
    ) -> Optional[torch.Tensor]:
        """Compute one flattened request with global cache as an optional stage."""
        if ctx.use_global_cache:
            mm_embedding = await self._compute_global_cache_embedding(
                ctx, keep_on_gpu=keep_on_gpu
            )
        else:
            mm_embedding = await self._compute_direct_embedding(
                ctx, keep_on_gpu=keep_on_gpu
            )

        expected_tokens = sum(ctx.preprocess_result.token_counts)
        if mm_embedding is not None and mm_embedding.shape[0] != expected_tokens:
            raise InternalError(
                f"Encoder produced {mm_embedding.shape[0]} tokens, but "
                f"preprocessor metadata expected {expected_tokens}"
            )
        return mm_embedding

    async def _publish_preprocess_metadata(
        self, ctx: EncodeContext, requests: List[dict]
    ) -> None:
        """Publish each request's size after preprocessing, before model forward."""
        if self.rank != 0:
            return
        embedding_dim = self._embedding_dims[ctx.modality]
        item_offset = 0
        for request, item_count in zip(requests, ctx.items_per_req):
            item_end = item_offset + item_count
            token_count = sum(ctx.preprocess_result.token_counts[item_offset:item_end])
            req_id = request["req_id"]
            state = self._require_active_encode_state(req_id)
            state.embedding_data = EmbeddingData(
                req_id,
                request["num_parts"],
                request["part_idx"],
                ctx.preprocess_result.grid_thw[item_offset:item_end],
                ctx.modality,
                embedding_shape=[token_count, embedding_dim],
                dtype=self._embedding_dtype,
            )
            await meta_registry.publish(
                req_id,
                token_count * embedding_dim * self._element_size,
                token_count,
                embedding_dim,
            )
            item_offset = item_end

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
        if get_disagg().encoder_transfer_backend == "mooncake":
            # Encode is synchronous, so mm_data was staged before /encode returned.
            req_id = mm_data.req_id
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

            # Fall back to a per-send registration only if the shared one failed.
            mr_already_registered = mm_data._mr_ptr == embedding.data_ptr()
            if not mr_already_registered:
                self.engine.register(embedding.data_ptr(), embedding.nbytes)
            _t_xfer_start = time.monotonic()
            xfer_ret = await asyncio.to_thread(
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
            if xfer_ret < 0:
                raise InternalError(
                    f"Mooncake transfer_sync failed for {req_id} "
                    f"(session={session_id}, nbytes={embedding.nbytes}, "
                    f"ret={xfer_ret})"
                )
            # Emit at INFO for slow transfers or per-send registrations.
            if xfer_ms > 200.0 or not mr_already_registered:
                logger.info(
                    f"[{req_id}] mooncake transfer_sync={xfer_ms:.1f}ms "
                    f"nbytes={embedding.nbytes} shared_mr={mr_already_registered}"
                )

            # Sibling ranks re-read mm_data here; meta_registry owns the release.

        # Send ack/data
        if url is not None:
            endpoint = NetworkAddress.parse(url).to_tcp()
        else:
            endpoint = NetworkAddress(prefill_host, embedding_port).to_tcp()
        logger.info(f"{endpoint = }")

        # Serialize data
        if get_disagg().encoder_transfer_backend == "mooncake":
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

        transfer_start = time.perf_counter()
        if self.transfer_backend == "zmq_to_scheduler" and url is not None:
            lock = self.scheduler_send_locks.get(endpoint)
            if lock is None:
                lock = asyncio.Lock()
                self.scheduler_send_locks[endpoint] = lock

            async with lock:
                sock = self.scheduler_send_sockets.get(endpoint)
                if sock is None:
                    sock = self.context.socket(zmq.PUSH)
                    config_socket(sock, zmq.PUSH)
                    sock.setsockopt(zmq.IMMEDIATE, 1)
                    sock.setsockopt(zmq.SNDTIMEO, int(self.send_timeout * 1000))
                    sock.connect(endpoint)
                    self.scheduler_send_sockets[endpoint] = sock
                try:
                    frames = (
                        [serialized_data, buffer]
                        if buffer is not None
                        else [serialized_data]
                    )
                    tracker = await sock.send_multipart(frames, copy=False, track=True)
                except Exception:
                    if self.scheduler_send_sockets.get(endpoint) is sock:
                        self.scheduler_send_sockets.pop(endpoint, None)
                    sock.close(linger=0)
                    raise

            # MessageTracker.wait() protects the zero-copy source buffer; it
            # is not a receiver acknowledgement. Waiting under the per-peer
            # lock serialized every large embedding on that TCP connection.
            # Queue sends in order under the lock, then wait for buffer
            # ownership independently so libzmq can pipeline the connection.
            try:
                await asyncio.to_thread(tracker.wait, self.send_timeout)
            except Exception:
                if self.scheduler_send_sockets.get(endpoint) is sock:
                    self.scheduler_send_sockets.pop(endpoint, None)
                    sock.close(linger=0)
                raise

            if encoder_metrics_collector is not None:
                encoder_metrics_collector.observe_transfer(
                    time.perf_counter() - transfer_start,
                    backend=self.transfer_backend,
                )
            return

        # Per-request sockets remain for zmq_to_tokenizer and legacy direct
        # scheduler sends. Scheduler URL sends use persistent sockets above.
        def send_with_socket():
            sock = self.sync_context.socket(zmq.PUSH)
            config_socket(sock, zmq.PUSH)
            sock.setsockopt(zmq.IMMEDIATE, 1)
            sock.setsockopt(zmq.SNDTIMEO, int(self.send_timeout * 1000))
            try:
                sock.connect(endpoint)
                if buffer is not None:
                    tracker = sock.send_multipart(
                        [serialized_data, buffer], copy=False, track=True
                    )
                else:
                    tracker = sock.send_multipart(
                        [serialized_data], copy=False, track=True
                    )
                tracker.wait(timeout=self.send_timeout)
            finally:
                sock.close(linger=5000)

        await asyncio.get_event_loop().run_in_executor(self.executor, send_with_socket)
        if (
            encoder_metrics_collector is not None
            and get_disagg().encoder_transfer_backend != "mooncake"
        ):
            encoder_metrics_collector.observe_transfer(
                time.perf_counter() - transfer_start,
                backend=get_disagg().encoder_transfer_backend,
            )

    def _register_shared_mr(self, mm_data: EmbeddingData, embedding: torch.Tensor):
        """Register one MR shared by every rank's /send; _send re-registers on failure."""
        try:
            self.engine.register(embedding.data_ptr(), embedding.nbytes)
            mm_data._mr_ptr = embedding.data_ptr()
        except Exception as reg_err:
            logger.warning(
                f"Shared-MR register failed for {mm_data.req_id}, "
                f"falling back to per-/send register: {reg_err}"
            )

    def _stage_embeddings(
        self,
        ctx: EncodeContext,
        requests: List[dict],
        mm_embedding: Optional[torch.Tensor],
        *,
        keep_on_gpu: bool,
    ) -> List[Tuple[int, int, int, Optional[str], Optional[int]]]:
        """Split the fused embedding per request and stage one EmbeddingData each.

        Per-request token ranges are contiguous in flatten order, so each
        staged embedding is a slice of the batch tensor.
        """
        if self.rank != 0:
            return [(0, 0, 0, None, None)] * len(requests)
        if mm_embedding is None:
            raise InternalError(f"Rank 0 produced no embedding for {ctx.req_id}")

        results = []
        staged_embeddings = []
        item_offset = 0
        token_offset = 0
        for req, num_items in zip(requests, ctx.items_per_req):
            item_end = item_offset + num_items
            num_tokens = sum(ctx.preprocess_result.token_counts[item_offset:item_end])
            embedding = mm_embedding[token_offset : token_offset + num_tokens]
            if keep_on_gpu and len(requests) > 1:
                # A view would pin the whole batch tensor until the last transfer.
                embedding = embedding.clone()
            req_aux_data = dict(ctx.aux_data)
            if ctx.aux_data.get("original_image_sizes") is not None:
                req_aux_data["original_image_sizes"] = ctx.aux_data[
                    "original_image_sizes"
                ][item_offset:item_end]
            mm_data = EmbeddingData(
                req["req_id"],
                req["num_parts"],
                req["part_idx"],
                ctx.preprocess_result.grid_thw[item_offset:item_end],
                ctx.modality,
                embedding,
                **req_aux_data,
            )
            # Global-cache embeddings keep registering per /send instead.
            if keep_on_gpu and not ctx.use_global_cache:
                self._register_shared_mr(mm_data, embedding)
            staged_embeddings.append(mm_data)
            results.append(
                (embedding.nbytes, embedding.shape[0], embedding.shape[1], None, None)
            )
            item_offset = item_end
            token_offset += num_tokens

        # transfer_sync bypasses CUDA streams, so GPU writes (forward and the
        # per-request clones) must land before /send reads the buffers.
        if keep_on_gpu and mm_embedding.is_cuda:
            torch.cuda.current_stream(mm_embedding.device).synchronize()
        for mm_data in staged_embeddings:
            self._stage_embedding(mm_data)
        return results

    def _stage_errors(
        self, requests: List[dict], modality: Modality, exc: Exception
    ) -> List[Tuple[int, int, int, Optional[str], Optional[int]]]:
        """Stage one error EmbeddingData per request so /send reports the failure."""
        code = (
            exc.code if isinstance(exc, MMError) else HTTPStatus.INTERNAL_SERVER_ERROR
        )
        msg = str(exc)
        logger.error(f"Rank {self.rank} encode failed: {msg} {code = }", exc_info=True)
        if self.rank == 0:
            for req in requests:
                self._stage_embedding(
                    EmbeddingData(
                        req["req_id"],
                        req["num_parts"],
                        req["part_idx"],
                        None,
                        modality,
                        error_msg=msg,
                        error_code=code,
                    )
                )
        return [(0, 0, 0, msg, code)] * len(requests)

    async def batch_encode(
        self, requests: List[dict], modality: Modality
    ) -> List[Tuple[int, int, int, Optional[str], Optional[int]]]:
        """Encode requests through one fused pipeline; encode() is the N=1 case.

        Fuse-or-not is EncoderScheduler policy, not an API fork. Health probes
        bypass caches and stage on CPU so completion confirms a model forward.
        """
        states = [self._acquire_encode_ref(req["req_id"]) for req in requests]
        is_health_check = all(
            is_health_check_request(req["req_id"]) for req in requests
        )
        keep_on_gpu = self.use_mooncake and not is_health_check
        use_global_cache = self.mm_global_cache is not None and not is_health_check
        try:
            ctx = await self._prepare_encode_context(
                requests,
                modality,
                use_global_cache=use_global_cache,
                is_health_check=is_health_check,
            )
            await self._publish_preprocess_metadata(ctx, requests)
            mm_embedding = await self._compute_embedding(ctx, keep_on_gpu=keep_on_gpu)

            if self.profiler is not None:
                for _ in requests:
                    self.profiler.step()

            return self._stage_embeddings(
                ctx, requests, mm_embedding, keep_on_gpu=keep_on_gpu
            )
        except Exception as e:
            return self._stage_errors(requests, modality, e)
        finally:
            for state in states:
                await self._release_encode_ref(state)

    async def encode(
        self, mm_items, modality: Modality, req_id, num_parts, part_idx, hashes=None
    ):
        """Encode one request: the batch-of-1 case of batch_encode."""
        results = await self.batch_encode(
            [
                {
                    "req_id": req_id,
                    "num_parts": num_parts,
                    "part_idx": part_idx,
                    "mm_items": mm_items,
                    "hashes": hashes,
                }
            ],
            modality,
        )
        return results[0]

    async def encode_request(self, req: dict, modality: Modality):
        """Adapt a request dictionary to the single-request encode interface."""
        return await self.encode(
            mm_items=req["mm_items"],
            modality=modality,
            req_id=req["req_id"],
            num_parts=req["num_parts"],
            part_idx=req["part_idx"],
            hashes=req.get("hashes"),
        )

    # For zmq_to_tokenizer zmq_to_scheduler and mooncake
    async def send(
        self, req_id, prefill_host, embedding_port, session_id=None, buffer_address=None
    ):
        state = self.req_states.get(req_id)
        if state is None:
            # False = nothing transferred: callers must not count this send
            # nor report success, or the decoder waits on an ack never coming.
            logger.warning(
                f"MMEncoder.send: no embedding for req_id={req_id} "
                f"(already released or unknown)"
            )
            return False
        await self.send_to_destination(
            state,
            SendDestination.from_host_port(
                prefill_host,
                embedding_port,
                session_id=session_id,
                buffer_address=buffer_address,
            ),
        )
        return True

    # For zmq_to_scheduler
    async def send_with_url(
        self,
        req_id,
    ):
        state = self.req_states.get(req_id)
        if state is None:
            return
        sent_urls: Set[str] = set()
        all_tasks: List[Tuple[asyncio.Task, str]] = []
        start_time = asyncio.get_running_loop().time()
        timeout = self.send_timeout
        cond = await _get_receive_condition(req_id)

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
                            self.send_to_destination(
                                state,
                                SendDestination.from_url(url),
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
            await self.release_request(req_id)


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
    else:
        # Health-check rids need no special routing: batch_encode derives
        # health semantics from the rid prefix itself.
        await encoder.encode_request(request, Modality.from_str(request["modality"]))


def launch_encoder(server_args, schedule_path, dist_init_method, rank):
    publish(server_args, role="encoder")
    try:
        asyncio.run(run_encoder(server_args, schedule_path, dist_init_method, rank))
    except KeyboardInterrupt:
        logger.info(f"Exit rank {rank}")
    except Exception:
        traceback.print_exc()


# Per-process encoder metrics collector. Set by
# runtime.launch_local_runtime (non-DP) and
# runtime.run_dp_worker (DP mode). None when metrics disabled. Kept
# here because MMEncoder GPU methods reference it directly.
encoder_metrics_collector: Optional[EncoderMetricsCollector] = None
