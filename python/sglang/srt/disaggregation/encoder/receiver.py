import asyncio
import itertools
import logging
import random
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from array import array
from collections import OrderedDict, defaultdict
from contextlib import asynccontextmanager
from enum import IntEnum
from http import HTTPStatus
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import torch
import uvicorn
import zmq
import zmq.asyncio
from aiohttp import ClientSession, ClientTimeout
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse, Response
from transformers import PretrainedConfig

from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    get_mooncake_transfer_engine,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.schedule_batch import Modality, Req
from sglang.srt.multimodal.cache import media_preprocess_kwargs
from sglang.srt.multimodal.transport import determine_tensor_transport_mode
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_mm,
    get_model,
    get_parallel,
    get_serving,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import ImageData
from sglang.srt.utils.common import safe_pickle_loads
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    resolve_image_processor_backend,
)
from sglang.srt.utils.network import (
    NetworkAddress,
    get_local_ip_auto,
    get_zmq_socket_on_host,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


def _mark_keep_device_embedding(mm_inputs) -> None:
    """Tell general_mm_embed_routine not to copy embeddings back to CPU."""
    if mm_inputs is None:
        return
    for item in mm_inputs.mm_items:
        item.keep_device_embedding = True


class EncoderBootstrapServer:
    """Lightweight bootstrap server for dynamic encoder discovery.

    Built on FastAPI + uvicorn to match the style of
    :mod:`sglang.srt.entrypoints.http_server`.  Runs in a daemon thread so
    the language-only tokenizer manager's main loop is unblocked.

    The set of registered URLs is exposed as the ``urls`` list passed in at
    construction time.  Callers that want to observe registrations without
    going through HTTP -- typically a co-located :class:`MMReceiver` -- share
    that list by reference: register/unregister mutate it in place under an
    internal lock, and the receiver simply reads ``self.encode_urls`` (the
    same list).  When ``urls`` is ``None`` the server allocates its own list,
    accessible through :meth:`list_urls`.

    Health-check tuning is controlled by env vars
    ``SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL`` (seconds; 0 disables),
    ``SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_TIMEOUT`` (seconds), and
    ``SGLANG_ENCODER_BOOTSTRAP_EVICTED_TTL`` (seconds; 0 keeps probing
    forever). Explicit constructor args take precedence over the env vars.
    """

    def __init__(
        self,
        host: str,
        port: int,
        urls: Optional[List[str]] = None,
        health_check_interval: Optional[float] = None,
        health_check_timeout: Optional[float] = None,
        evicted_ttl: Optional[float] = None,
    ):

        self.host = host
        self.port = port
        self._urls: List[str] = urls if urls is not None else []
        self._lock = threading.Lock()
        self._server: Optional[uvicorn.Server] = None  # set in _run_server
        self._health_check_interval = (
            health_check_interval
            if health_check_interval is not None
            else envs.SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL.get()
        )
        self._health_check_timeout = (
            health_check_timeout
            if health_check_timeout is not None
            else envs.SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_TIMEOUT.get()
        )
        self._evicted_ttl = (
            evicted_ttl
            if evicted_ttl is not None
            else envs.SGLANG_ENCODER_BOOTSTRAP_EVICTED_TTL.get()
        )
        # Evict only after this many consecutive probe failures (a busy
        # encoder can miss a single 2s probe under load), and keep probing
        # evicted URLs so they re-register automatically once healthy.
        # Values are eviction timestamps; URLs older than ``_evicted_ttl``
        # (when > 0) are permanently dropped.
        self._health_fail_threshold = 3
        self._health_fail_counts: Dict[str, int] = {}
        self._evicted_urls: Dict[str, float] = {}

        @asynccontextmanager
        async def lifespan(fast_api_app: FastAPI):
            task: Optional[asyncio.Task] = None
            if self._health_check_interval > 0:
                task = asyncio.create_task(self._health_check_loop())
            try:
                yield
            finally:
                if task is not None:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

        self.app = FastAPI(lifespan=lifespan, openapi_url=None)

        @self.app.get("/health")
        async def _health() -> Response:
            return Response("OK")

        @self.app.post("/register_encoder_url")
        async def _register(data: dict):
            url = data.get("url") if isinstance(data, dict) else None
            if not url:
                return ORJSONResponse(
                    {"error": "Missing or empty 'url' field"}, status_code=400
                )
            self.register(url)
            return Response("OK")

        @self.app.delete("/unregister_encoder_url")
        async def _unregister(data: dict):
            url = data.get("url") if isinstance(data, dict) else None
            if not url:
                return ORJSONResponse(
                    {"error": "Missing or empty 'url' field"}, status_code=400
                )
            self.unregister(url)
            return Response("OK")

        @self.app.get("/list_encoder_urls")
        async def _list():
            return {"encoder_urls": self.list_urls()}

        self.thread = threading.Thread(
            target=self._run_server, daemon=True, name="EncoderBootstrap"
        )
        self.thread.start()

    # ------------------------------------------------------------------ #
    # In-process API (thread-safe; safe to call from any thread)         #
    # ------------------------------------------------------------------ #
    def register(self, url: str) -> bool:
        """Add *url* if not already present.  Returns True if added."""
        with self._lock:
            self._health_fail_counts.pop(url, None)
            self._evicted_urls.pop(url, None)
            if url not in self._urls:
                self._urls.append(url)
                logger.info(f"Registered encoder URL: {url}")
                return True
            logger.debug(f"Encoder URL already registered: {url}")
            return False

    def unregister(self, url: str) -> bool:
        """Remove *url* if present.  Returns True if removed.

        An explicit unregister also drops the URL from the health-check
        revival set so it does not come back automatically.
        """
        with self._lock:
            removed = url in self._urls or url in self._evicted_urls
            if url in self._urls:
                self._urls.remove(url)
            self._evicted_urls.pop(url, None)
            self._health_fail_counts.pop(url, None)
            if removed:
                logger.info(f"Unregistered encoder URL: {url}")
            return removed

    def list_urls(self) -> List[str]:
        """Return a snapshot of all registered encoder URLs."""
        with self._lock:
            return list(self._urls)

    # ------------------------------------------------------------------ #
    # Health check                                                       #
    # ------------------------------------------------------------------ #
    async def _probe(self, session, url: str) -> bool:
        try:
            async with session.get(f"{url}/health") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def _health_check_loop(self):
        """Probe registered (and previously evicted) encoders periodically.

        A URL is evicted only after ``_health_fail_threshold`` consecutive
        probe failures — a busy encoder may miss a single short-timeout probe
        under load. Evicted URLs keep being probed and re-register
        automatically once they respond again. After ``_evicted_ttl`` seconds
        without a successful probe (when > 0), they are permanently dropped
        so a dead encoder does not get probed forever.
        """

        timeout = ClientTimeout(total=self._health_check_timeout)
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                now = time.time()
                with self._lock:
                    expired = []
                    if self._evicted_ttl > 0:
                        expired = [
                            url
                            for url, ts in self._evicted_urls.items()
                            if now - ts >= self._evicted_ttl
                        ]
                        for url in expired:
                            self._evicted_urls.pop(url, None)
                            self._health_fail_counts.pop(url, None)
                    candidates = list(
                        dict.fromkeys(self._urls + list(self._evicted_urls))
                    )
                if expired:
                    logger.warning(
                        f"Health check permanently dropped {len(expired)} "
                        f"encoder(s) after {self._evicted_ttl}s unhealthy: "
                        f"{expired}"
                    )
                if not candidates:
                    continue
                async with ClientSession(timeout=timeout) as session:
                    results = await asyncio.gather(
                        *(self._probe(session, url) for url in candidates),
                        return_exceptions=True,
                    )
                evicted, revived = [], []
                with self._lock:
                    for url, ok in zip(candidates, results):
                        if ok is True:
                            self._health_fail_counts.pop(url, None)
                            if url in self._evicted_urls:
                                self._evicted_urls.pop(url, None)
                                if url not in self._urls:
                                    self._urls.append(url)
                                revived.append(url)
                        else:
                            if url in self._evicted_urls:
                                continue
                            count = self._health_fail_counts.get(url, 0) + 1
                            self._health_fail_counts[url] = count
                            if count >= self._health_fail_threshold:
                                if url in self._urls:
                                    self._urls.remove(url)
                                self._evicted_urls[url] = now
                                self._health_fail_counts.pop(url, None)
                                evicted.append(url)
                if revived:
                    logger.info(
                        f"Health check revived {len(revived)} encoder(s): {revived}"
                    )
                if evicted:
                    logger.warning(
                        f"Health check evicted {len(evicted)} encoder(s) after "
                        f"{self._health_fail_threshold} consecutive failures "
                        f"(will re-add when healthy): {evicted}"
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Health check loop error: {e}", exc_info=True)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def _run_server(self):

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
            loop="auto",
        )
        self._server = uvicorn.Server(config)
        logger.info(
            f"EncoderBootstrapServer starting on {self.host}:{self.port} "
            f"(health_check every {self._health_check_interval}s, "
            f"timeout {self._health_check_timeout}s)"
        )
        try:
            self._server.run()
        except Exception as e:
            logger.error(f"EncoderBootstrapServer error: {e}", exc_info=True)

    def close(self):
        if self._server is not None:
            # uvicorn polls should_exit on its own event loop; thread-safe.
            self._server.should_exit = True
            logger.info("Stopping EncoderBootstrapServer...")
        if self.thread.is_alive():
            self.thread.join(timeout=5)
            logger.info("EncoderBootstrapServer thread stopped")


def _grpc_target(url: str) -> str:
    if url.startswith("grpc://"):
        return url[len("grpc://") :]
    if url.startswith("grpcs://"):
        raise ValueError("grpcs:// is not supported; use grpc://")
    return url


def _normalize_embedding_ports(embedding_port):
    if embedding_port is None:
        return []
    if isinstance(embedding_port, list):
        return embedding_port
    return [embedding_port]


def _grpc_scheduler_receive_url(target, req_id, receive_url, receive_count):
    import grpc
    from smg_grpc_proto import sglang_encoder_pb2, sglang_encoder_pb2_grpc

    timeout_secs = envs.SGLANG_ENCODER_GRPC_TIMEOUT_SECS.get()
    channel = grpc.insecure_channel(target)
    stub = sglang_encoder_pb2_grpc.SglangEncoderStub(channel)
    try:
        stub.SchedulerReceiveUrl(
            sglang_encoder_pb2.SchedulerReceiveUrlRequest(
                req_id=req_id,
                receive_url=receive_url,
                receive_count=receive_count,
            ),
            timeout=timeout_secs,
        )
    finally:
        channel.close()


def _grpc_encode_request(target, encode_request):
    import grpc
    from smg_grpc_proto import sglang_encoder_pb2, sglang_encoder_pb2_grpc

    timeout_secs = envs.SGLANG_ENCODER_GRPC_TIMEOUT_SECS.get()
    channel = grpc.insecure_channel(target)
    stub = sglang_encoder_pb2_grpc.SglangEncoderStub(channel)
    try:
        response = stub.Encode(
            sglang_encoder_pb2.EncodeRequest(
                mm_items=encode_request["mm_items"],
                req_id=encode_request["req_id"],
                num_parts=encode_request["num_parts"],
                part_idx=encode_request["part_idx"],
                prefill_host=encode_request["prefill_host"],
                embedding_port=_normalize_embedding_ports(
                    encode_request["embedding_port"]
                ),
            ),
            timeout=timeout_secs,
        )
        return response
    finally:
        channel.close()


class EmbeddingData:
    def __init__(
        self,
        req_id,
        num_parts,
        part_idx,
        grid_dim,
        modality,
        embedding=None,
        embedding_shape=None,
        error_msg=None,
        error_code=None,
        **kwargs,
    ):
        self.req_id = req_id
        self.num_parts = num_parts
        self.part_idx = part_idx
        self.grid_dim = grid_dim
        self.modality = modality
        self.embedding = embedding
        self.send_time = None
        self.dtype = embedding.dtype if embedding is not None else None
        if embedding_shape is not None:
            self.shape = embedding_shape
        else:
            self.shape = list(embedding.shape) if embedding is not None else None
        # Encoder-side mooncake MR for `embedding`. Underscored so
        # copy_without_embedding drops this process-local address.
        self._mr_ptr: Optional[int] = None
        self.error_msg = error_msg
        # Coerce to plain int: this object crosses process boundaries via
        # safe_pickle_loads, whose allowlist blocks http.HTTPStatus.
        self.error_code = int(error_code) if error_code is not None else None
        # Store additional metadata (e.g., video_timestamps for qwen3_vl)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_grid(self):
        """Get the grid dimension of the embedding, used for image/video/audio."""
        return self.grid_dim

    def get_embedding(self):
        return self.embedding

    def __repr__(self):
        return f"EmbeddingData(req_id={self.req_id}, num_parts={self.num_parts}, part_idx={self.part_idx}) error_msg={self.error_msg}"

    def copy_without_embedding(self):
        new_data = EmbeddingData(
            req_id=self.req_id,
            num_parts=self.num_parts,
            part_idx=self.part_idx,
            grid_dim=self.grid_dim,
            modality=self.modality,
            embedding=None,
            embedding_shape=self.shape,
            error_msg=self.error_msg,
            error_code=self.error_code,
        )
        for key, value in self.__dict__.items():
            if key.startswith("_") or key == "embedding":
                continue
            setattr(new_data, key, value)
        return new_data


# Modality -> (list attr name, whether to flatten grid for that list)
_MODALITY_GRID_ATTRS = {
    Modality.IMAGE: ("img_grid_thw", False),
    Modality.VIDEO: ("video_grid_thw", False),
    Modality.AUDIO: ("audio_feature_lens", True),
}
# Per-part video metadata for EPD. Tensor attrs cat on dim=0 across parts;
# others chain as lists. video_meta_attrs_for(model_type) resolves the active
# set per instance so non-MiMo runs skip the MiMo audio fields entirely.
_GENERAL_VIDEO_META_ATTRS = (
    "video_timestamps",
    "second_per_grid_ts",
)
_GENERAL_IMAGE_META_ATTRS = ("original_image_sizes",)
# MiMo-VL audio-in-video fields; appended only when model_type is MiMo.
_MIMO_VIDEO_AUDIO_META_ATTRS = (
    "video_audio_feature_lens",
    "video_audio_segment_lens_flat",
    "video_audio_per_video_num_units",
    "video_audio_embedding",
)
_VIDEO_META_TENSOR_ATTRS = ("video_audio_feature_lens", "video_audio_embedding")


def video_meta_attrs_for(model_type: Optional[str]) -> tuple:
    """Video-meta attrs for model_type. MiMo appends its audio-in-video fields."""
    attrs = _GENERAL_VIDEO_META_ATTRS
    if model_type and "mimo" in model_type.lower():
        attrs = attrs + _MIMO_VIDEO_AUDIO_META_ATTRS
    return attrs


def _cat_grid(dims, flatten_items=False):
    """Concatenate non-None grid entries; supports tensor/ndarray/list inputs."""

    def _to_tensor(g):
        if isinstance(g, torch.Tensor):
            return g.cpu() if g.is_cuda else g
        if isinstance(g, np.ndarray):
            return torch.from_numpy(g)
        return torch.as_tensor(g)

    valid = []
    for g in dims:
        if g is None:
            continue
        t = _to_tensor(g)
        if flatten_items:
            t = t.flatten()
        elif t.ndim == 0:
            # Keep cat semantics stable for scalar-like metadata.
            t = t.unsqueeze(0)
        valid.append(t)

    return torch.cat(valid, dim=0) if valid else None


class MultiModalEmbeddingData(EmbeddingData):
    def __init__(
        self,
        part_idx,
        num_parts,
        req_id,
        grid_dim,
        modality,
        embedding,
        embedding_shape,
        model_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            req_id,
            num_parts,
            part_idx,
            grid_dim,
            modality,
            embedding,
            embedding_shape,
            **kwargs,
        )
        self.video_meta_attrs = video_meta_attrs_for(model_type)
        self.img_grid_thw = [None] * num_parts
        self.video_grid_thw = [None] * num_parts
        self.audio_feature_lens = [None] * num_parts
        for attr in _GENERAL_IMAGE_META_ATTRS:
            setattr(self, attr, [None] * num_parts)
        self.modality_list = [
            modality if part_idx == i else None for i in range(num_parts)
        ]
        self.ready_list = [i == part_idx for i in range(num_parts)]
        self.embedding_list = [
            embedding if i == part_idx else None for i in range(num_parts)
        ]
        self.embedding_shape_list = [
            embedding_shape if i == part_idx else None for i in range(num_parts)
        ]
        for attr in self.video_meta_attrs:
            setattr(self, attr, [None] * num_parts)

        self._set_part_grid(part_idx, modality, self.get_grid())
        if modality == Modality.VIDEO:
            self._set_video_meta_for_part(part_idx, kwargs)
        if modality == Modality.IMAGE:
            self._set_image_meta_for_part(part_idx, kwargs)

    def _set_image_meta_for_part(self, part_idx, source):
        for attr_name in _GENERAL_IMAGE_META_ATTRS:
            val = (
                source.get(attr_name)
                if isinstance(source, dict)
                else getattr(source, attr_name, None)
            )
            if val is not None:
                getattr(self, attr_name)[part_idx] = val

    def _set_part_grid(self, part_idx, modality, grid):
        """Set the grid for one part according to modality (IMAGE/VIDEO/AUDIO)."""
        spec = _MODALITY_GRID_ATTRS.get(modality)
        if spec is None:
            raise ValueError(f"Invalid modality: {modality}")
        attr_name, flatten = spec
        value = grid.flatten() if flatten else grid
        getattr(self, attr_name)[part_idx] = value

    def _set_video_meta_for_part(self, part_idx, source):
        """Copy video_timestamps and second_per_grid_ts from source (dict or object)."""
        for attr_name in self.video_meta_attrs:
            val = (
                source.get(attr_name)
                if isinstance(source, dict)
                else getattr(source, attr_name, None)
            )
            if val is not None:
                getattr(self, attr_name)[part_idx] = val

    @classmethod
    def from_embedding_data(
        cls,
        embedding_data: EmbeddingData,
        model_type: Optional[str] = None,
    ):
        """Create MultiModalEmbeddingData from an EmbeddingData instance."""
        # Only forward known optional attrs (e.g. video metadata) so they land on the instance
        extra = {}
        for attr in video_meta_attrs_for(model_type):
            val = getattr(embedding_data, attr, None)
            if val is not None:
                extra[attr] = val
        for attr in _GENERAL_IMAGE_META_ATTRS:
            val = getattr(embedding_data, attr, None)
            if val is not None:
                extra[attr] = val
        mm_data = cls(
            part_idx=embedding_data.part_idx,
            num_parts=embedding_data.num_parts,
            req_id=embedding_data.req_id,
            grid_dim=embedding_data.grid_dim,
            modality=embedding_data.modality,
            embedding=embedding_data.embedding,
            embedding_shape=embedding_data.shape,
            model_type=model_type,
            **extra,
        )
        mm_data.send_time = embedding_data.send_time
        return mm_data

    def __repr__(self):
        return f"MultiModalEmbeddingData(req_id={self.req_id}, num_parts={self.num_parts}, part_idx={self.part_idx}, modality={self.modality})"

    def get_embedding(self, is_concat=False):
        if is_concat:
            groups = defaultdict(list)
            for i, e in enumerate(self.embedding_list):
                if e is not None:
                    groups[self.modality_list[i]].append(e)
            return {mod: torch.cat(tensors, dim=0) for mod, tensors in groups.items()}
        return self.embedding_list

    @property
    def ready(self):
        return sum(self.ready_list) == self.num_parts

    def get_mm_extra_meta(self):
        """Build kwargs for mm_processor.get_mm_data() from grid and optional video meta."""
        kwargs = {
            "img_grid_thw": _cat_grid(self.img_grid_thw),
            "video_grid_thw": _cat_grid(self.video_grid_thw),
            "audio_feature_lens": _cat_grid(
                self.audio_feature_lens, flatten_items=True
            ),
        }
        for attr in self.video_meta_attrs:
            lst = getattr(self, attr, None)
            if not lst:
                continue
            valid = [a for a in lst if a is not None]
            if not valid:
                continue
            if attr in _VIDEO_META_TENSOR_ATTRS:
                kwargs[attr] = torch.cat(valid, dim=0)
            else:
                kwargs[attr] = list(itertools.chain(*valid))
        for attr in _GENERAL_IMAGE_META_ATTRS:
            valid = [value for value in getattr(self, attr) if value is not None]
            if valid:
                kwargs[attr] = list(itertools.chain(*valid))
        return kwargs

    def add(self, embedding_data: EmbeddingData):
        if self.req_id != embedding_data.req_id:
            logger.warning(
                f"Dropping embedding data with mismatched req_id: "
                f"expected {self.req_id}, got {embedding_data.req_id}"
            )
            return False
        assert not self.ready_list[embedding_data.part_idx]
        pid = embedding_data.part_idx
        self.ready_list[pid] = True
        self.modality_list[pid] = embedding_data.modality
        self.embedding_list[pid] = embedding_data.get_embedding()
        self.embedding_shape_list[pid] = embedding_data.shape
        self._set_part_grid(pid, embedding_data.modality, embedding_data.get_grid())
        if embedding_data.modality == Modality.VIDEO:
            self._set_video_meta_for_part(pid, embedding_data)
        if embedding_data.modality == Modality.IMAGE:
            self._set_image_meta_for_part(pid, embedding_data)


def _aggregate_embedding_part(current, recv_obj, model_type):
    """Fold one received part into the aggregate (the first part creates it)."""
    if current is None:
        return MultiModalEmbeddingData.from_embedding_data(
            recv_obj, model_type=model_type
        )
    current.add(recv_obj)
    return current


class WaitingMMRequestStatus(IntEnum):
    FAIL = -1
    PENDING = 0
    SUCCESS = 1
    TIMEOUT = -2


def _select_mm_processor_prompt(recv_req, mm_processor):
    """Mirror tokenizer-side prompt selection for scheduler-side EPD rebuilds."""
    if mm_processor.prefer_tokenized_input and recv_req.input_ids is not None:
        return list(recv_req.input_ids)
    return recv_req.input_text or recv_req.input_ids


def create_part_req_id(original_req_id: str, part_idx: int) -> str:
    """Create a unique part request ID by appending part index suffix."""
    return f"{original_req_id}_local_part_{part_idx}"


def extract_original_req_id(part_req_id: str) -> str:
    """Extract the original request ID from a part request ID."""
    if "_local_part_" in part_req_id:
        return part_req_id.rsplit("_local_part_", 1)[0]
    return part_req_id


def _encoder_media_item(mm_item: dict):
    """Keep per-media options aligned while preserving the legacy URL shape."""
    item = {
        key: value
        for key, value in mm_item.items()
        if key != "modality" and value is not None
    }
    return item["url"] if set(item) == {"url"} else item


def calculate_modality_num_parts(modalities, num_items_assigned):
    """
    Calculate total number of parts and number of parts per modality.

    Args:
        modalities: List of modalities in order
        num_items_assigned: Dictionary mapping modality to list of assignment counts per encoder

    Returns:
        Tuple of (total_num_parts, modality_num_parts_dict)
        - total_num_parts: Total number of parts across all modalities
        - modality_num_parts: Dictionary mapping modality to number of parts for that modality
    """
    total_num_parts = 0
    modality_num_parts = {}
    for modality in modalities:
        num_items_assigned_modality = num_items_assigned.get(modality)
        num_parts = sum(1 for x in num_items_assigned_modality if x != 0)
        modality_num_parts[modality] = num_parts
        total_num_parts += num_parts
    return total_num_parts, modality_num_parts


class WaitingMMRequestBase(ABC):
    """One in-flight multimodal request on a scheduler rank, waiting for
    encoder embeddings. Owns the shared machinery: the ZMQ receive loop,
    failure handling (_fail_and_release), pool-slot lifetime, and the
    TP-consistent status. Subclasses bind the transport.
    """

    def __init__(
        self,
        rid: str,
        recv_req: TokenizedGenerateReqInput,
        mm_processor,
        encoder_urls,
        model_type,
        host_name,
        receive_count,
        embedding_pool: Optional["EmbeddingPool"] = None,
        zmq_context=None,
        embedding_port=None,
    ):
        self.rid = rid
        self.recv_req = recv_req
        self.mm_inputs = None
        self.error = None
        self.thread = None
        self.mm_processor = mm_processor
        self.encoder_urls = encoder_urls
        self.model_type = model_type
        self.host_name = host_name
        self.receive_count = receive_count
        self.num_items_assigned = recv_req.num_items_assigned
        self.zmq_context = zmq_context or zmq.Context()
        if embedding_port is None:
            self.embedding_port, self.recv_socket = get_zmq_socket_on_host(
                self.zmq_context, zmq.PULL, host=host_name
            )
        else:
            self.embedding_port = embedding_port
            self.recv_socket = None
        logger.info(f"Waiting for input {self.embedding_port = }")
        self.recv_embedding_data = None
        # ok=1 pending=0 fail=-1
        self.status = WaitingMMRequestStatus.PENDING
        self.error_msg = None
        self.error_code = None
        self.start_time = time.time()
        # Optional GPU pool bounding received embeddings (zmq_to_scheduler):
        # _try_recv_mm_data stages parts into one slot, staying PENDING while
        # the pool is full.
        self.embedding_pool = embedding_pool
        self.embeddings_buffer = None
        self._pool_slot_id: Optional[int] = None
        # Success-path finalizer handle so abort can release the slot early.
        self._mm_finalizer: Optional[weakref.finalize] = None
        self._pool_full_warned = False

    @abstractmethod
    def send_encode_request(self) -> None:
        """Kick off the transport-specific encode / receive flow."""

    def _try_recv_mm_data(self):
        if self.status != WaitingMMRequestStatus.PENDING:
            return

        # A complete request can remain pending while the GPU pool is full.
        # Retry assembly on every scheduler tick, including shared-socket mode.
        if self.recv_embedding_data is not None and self.recv_embedding_data.ready:
            if self._assemble_mm_inputs_from_embeddings():
                self.close_recv_socket()
            return

        if self.recv_socket is None:
            return

        while self.recv_embedding_data is None or not self.recv_embedding_data.ready:
            try:
                parts = self.recv_socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
            except zmq.Again:
                # No data available yet, wait a bit and retry
                return
            except zmq.ZMQError:
                # Socket closed by another path (e.g. the RDMA receive thread
                # after an encoder error); status is already terminal.
                return
            self.consume_parts(parts)
            if self.status != WaitingMMRequestStatus.PENDING:
                return

    def consume_parts(self, parts) -> None:
        """Consume one message from either a per-request or shared ZMQ socket."""
        if self.status != WaitingMMRequestStatus.PENDING:
            return

        try:
            recv_obj: EmbeddingData = safe_pickle_loads(parts[0])
            if getattr(recv_obj, "error_msg", None) is not None:
                logger.warning(
                    f"Received error signal from encoder for {self.rid}: "
                    f"{recv_obj.error_msg} {recv_obj.error_code = }"
                )
                self._fail_and_release(recv_obj.error_msg, recv_obj.error_code)
                return
            if not self._is_valid_embedding_part(recv_obj):
                return
            # ZMQ materializes frame 1; RDMA already wrote the registered buffer.
            self._extract_embedding_from_buffer(recv_obj, parts)
            self.recv_embedding_data = _aggregate_embedding_part(
                self.recv_embedding_data, recv_obj, self.model_type
            )
        except Exception as e:
            # A malformed message must fail this request, not the scheduler loop.
            logger.exception("Failed to decode embedding message for rid=%s", self.rid)
            self._fail_and_release(f"Failed to decode embedding message: {e}")
            return

        if (
            self.recv_embedding_data.ready
            and self._assemble_mm_inputs_from_embeddings()
        ):
            self.close_recv_socket()

    def close_recv_socket(self) -> None:
        if self.recv_socket is not None:
            self.recv_socket.close()
            self.recv_socket = None

    def _fail_and_release(self, error_msg, error_code=None) -> None:
        """Terminal failure: record the error, free buffers, close the socket."""
        self.error_msg = error_msg
        self.error_code = error_code
        self.status = WaitingMMRequestStatus.FAIL
        self._cleanup_gpu_buffer()
        self.close_recv_socket()

    async def _check_encoder_responses(self, responses, endpoint: str) -> bool:
        """Validate gathered encoder responses; on the first error, FAIL the
        request and release its resources. Returns True if all succeeded."""
        msg = await _extract_encoder_error(responses, endpoint, f"rid={self.rid}")
        if msg is None:
            return True
        self._fail_and_release(msg)
        return False

    def _is_valid_embedding_part(self, recv_obj) -> bool:
        """Check for and drop stale or out-of-sync payloads; normalize the part req_id to the original rid."""
        original_req_id = extract_original_req_id(recv_obj.req_id)
        if original_req_id != self.recv_req.rid:
            logger.warning(
                f"Dropping stale embedding data: expected rid={self.recv_req.rid}, "
                f"got rid={recv_obj.req_id} (likely from ZMQ port reuse)"
            )
            return False
        recv_obj.req_id = original_req_id
        return True

    @abstractmethod
    def _extract_embedding_from_buffer(self, recv_obj, parts) -> None:
        """Materialize ``recv_obj.embedding`` from one received part message."""

    @abstractmethod
    def _prepare_embedding_buffer(self) -> bool:
        """Make ``embeddings_buffer`` ready for assembly, or leave it None
        for the CPU-concat path. False = not ready yet, stay PENDING."""

    def _view_dtype(self):
        """dtype of the bytes in ``embeddings_buffer``."""
        return self.recv_embedding_data.dtype

    def _assemble_mm_inputs_from_embeddings(self) -> bool:
        """Assemble mm_inputs from the received embeddings and mark
        SUCCESS/FAIL. Failures are caught so they still reach the TP-wide
        status all-reduce.

        Returns True when done so the caller closes the recv socket; False
        when the buffer is not ready yet (stay PENDING, retry next tick) or
        the request can never fit the pool (already FAILed, socket closed).
        """
        try:
            if not self._prepare_embedding_buffer():
                return False
            if self.embeddings_buffer is not None:
                # Zero-copy per-modality views into the GPU buffer; slot
                # lifetime is bound to mm_inputs GC in _finish_assemble.
                recv_embedding = _view_pool_buffer_by_modality(
                    self.embeddings_buffer,
                    self.recv_embedding_data,
                    self._view_dtype(),
                )
            else:
                recv_embedding = self.recv_embedding_data.get_embedding(is_concat=True)
            self._finish_assemble(recv_embedding)
            # Releases whatever is still attached: no-op once the slot was
            # detached; RDMA's override also deregisters non-pool buffers.
            self._cleanup_gpu_buffer()
        except Exception as e:
            self._fail_assemble(e)
        return True

    def _finish_assemble(self, recv_embedding) -> None:
        """get_mm_data → bind pool slot → publish onto recv_req → SUCCESS."""
        mm_inputs = self.mm_processor.get_mm_data(
            _select_mm_processor_prompt(self.recv_req, self.mm_processor),
            recv_embedding,
            **self.recv_embedding_data.get_mm_extra_meta(),
        )
        self._bind_pool_slot_to_mm_inputs(mm_inputs)
        self.recv_req.mm_inputs = mm_inputs
        self.recv_req.input_ids = array("q", mm_inputs.input_ids)
        self.status = WaitingMMRequestStatus.SUCCESS

    def _fail_assemble(self, e: Exception) -> None:
        logger.exception("Failed to assemble multimodal inputs for rid=%s", self.rid)
        self._fail_and_release(f"Failed to assemble multimodal inputs: {e}")

    def _bind_pool_slot_to_mm_inputs(self, mm_inputs) -> bool:
        """Bind pool-slot release to mm_inputs GC. Returns True if bound."""
        if (
            mm_inputs is None
            or self._pool_slot_id is None
            or self.embedding_pool is None
        ):
            return False
        # Keep the handle so abort can release the slot immediately.
        self._mm_finalizer = self.embedding_pool.release_on_gc(
            mm_inputs, self._pool_slot_id
        )
        _mark_keep_device_embedding(mm_inputs)
        # Detach so _cleanup_gpu_buffer no-ops; finalize now owns release.
        self._pool_slot_id = None
        self.embeddings_buffer = None
        return True

    def _cleanup_gpu_buffer(self):
        if self._pool_slot_id is not None and self.embedding_pool is not None:
            self.embedding_pool.release(self._pool_slot_id)
            self._pool_slot_id = None
        self.embeddings_buffer = None

    def release_resources(self):
        """Free pool/GPU resources on abort/fail/timeout. Idempotent."""
        self._cleanup_gpu_buffer()
        finalizer, self._mm_finalizer = self._mm_finalizer, None
        if finalizer is not None:
            finalizer()  # at-most-once; a later GC call becomes a no-op


# For zmq_to_scheduler: embedding parts arrive as ZMQ payload frames and
# are optionally staged into the GPU EmbeddingPool.
class WaitingZmqRequest(WaitingMMRequestBase):
    def send_encode_request(self):

        async def _send_single_request(session, url, payload):
            try:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    return await response.text()
            except Exception as e:
                logger.error(f"Failed to send request to {url}: {e}")
                raise

        async def send_embedding_port(req_id, receive_count, host_name, embedding_port):
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=envs.SGLANG_ENCODER_HTTP_TIMEOUT.get()
                )
            ) as session:
                tasks = []
                logger.info(f"{self.num_items_assigned = } ")

                # Calculate part_idx_offset similar to encode() method
                modalities = list(self.num_items_assigned.keys())
                _, modality_num_parts = calculate_modality_num_parts(
                    modalities, self.num_items_assigned
                )

                part_idx_offset = 0
                for modality in modalities:
                    assigned_nums = self.num_items_assigned[modality]
                    num_parts = modality_num_parts[modality]
                    cum_idx = 0
                    for idx, assigned_num in enumerate(assigned_nums):
                        if assigned_num == 0:
                            continue
                        part_idx = part_idx_offset + cum_idx
                        part_req_id = create_part_req_id(req_id, part_idx)
                        encoder_url = self.encoder_urls[idx]
                        target_url = f"{encoder_url}/scheduler_receive_url"
                        payload = {
                            "req_id": part_req_id,  # use part_req_id to match encode request
                            "receive_count": receive_count,
                            "receive_url": NetworkAddress(
                                host_name, embedding_port
                            ).to_host_port_str(),
                            "modality": modality.name,
                        }
                        logger.info(
                            f"Preparing to send to {target_url} with part_req_id={part_req_id}"
                        )
                        task = _send_single_request(session, target_url, payload)
                        tasks.append(task)
                        cum_idx += 1
                    part_idx_offset += num_parts

                if not tasks:
                    logger.info("No tasks to send.")
                    return
                logger.info(f"Concurrently sending {len(tasks)} requests...")
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(results):
                    if isinstance(result, asyncio.TimeoutError):
                        timeout_val = envs.SGLANG_ENCODER_HTTP_TIMEOUT.get()
                        logger.error(
                            f"Request {i} to encoder /scheduler_receive_url timed out "
                            f"({timeout_val}s) for req_id={req_id}"
                        )
                    elif isinstance(result, Exception):
                        logger.error(
                            f"Request {i} to encoder /scheduler_receive_url failed for "
                            f"req_id={req_id}: {result}",
                            exc_info=result,
                        )
                    else:
                        logger.debug(f"Request {i} succeeded.")
                failed = [r for r in results if isinstance(r, BaseException)]
                if failed:
                    # A rank without a registered receive URL can never be
                    # pushed to; fail via the normal completion path now
                    # instead of pending until the embedding wait times out.
                    self._fail_and_release(
                        f"Failed to register receive URL with encoder: {failed[0]!r}",
                        int(HTTPStatus.BAD_GATEWAY),
                    )

        asyncio.run(
            send_embedding_port(
                self.recv_req.rid,
                self.receive_count,
                self.host_name,
                self.embedding_port,
            )
        )

    def _extract_embedding_from_buffer(self, recv_obj, parts) -> None:
        """ZMQ transport carries the embedding bytes as frame 1. Clone so we
        don't depend on the ZMQ buffer after the next recv."""
        buffer = parts[1].buffer if hasattr(parts[1], "buffer") else parts[1]
        recv_obj.embedding = (
            torch.frombuffer(buffer, dtype=recv_obj.dtype)
            .reshape(recv_obj.shape)
            .clone()
        )

    def _prepare_embedding_buffer(self) -> bool:
        """Stage the CPU parts into the GPU pool when one is configured;
        without a pool the CPU-concat path is used (buffer stays None)."""
        if self.embedding_pool is None:
            return True
        return self._try_stage_into_pool()

    def _try_stage_into_pool(self) -> bool:
        """Copy the received parts into one pooled GPU slot, packed in part
        order (modality-contiguous by construction, see _extract_url_data —
        _view_pool_buffer_by_modality asserts this).

        Returns True once ``self.embeddings_buffer`` views the slot; False
        when the pool is currently full (retry next tick) or the request can
        never fit (marked FAIL, socket closed).
        """
        if self.embeddings_buffer is not None:
            return True
        parts = self.recv_embedding_data.embedding_list
        total_bytes = sum(p.nbytes for p in parts if p is not None)
        if total_bytes > self.embedding_pool.size_bytes:
            error_msg = (
                f"EmbeddingPool cannot fit {total_bytes // (1024 * 1024)}MB "
                f"(pool is {self.embedding_pool.size_bytes // (1024 * 1024)}MB). "
                f"Raise SGLANG_EMBEDDING_POOL_SIZE_MB."
            )
            logger.error(f"{error_msg} rid={self.rid}")
            self._fail_and_release(error_msg)
            return False
        staged = self.embedding_pool.try_stage([p for p in parts if p is not None])
        if staged is None:
            if not self._pool_full_warned:
                logger.warning(
                    f"EmbeddingPool full; rid={self.rid} pending for "
                    f"{total_bytes // (1024 * 1024)}MB. Raise "
                    f"SGLANG_EMBEDDING_POOL_SIZE_MB if this is frequent."
                )
                self._pool_full_warned = True
            return False
        self.embeddings_buffer, self._pool_slot_id = staged
        # Drop the CPU clones now that they live in the pool.
        for i in range(len(parts)):
            parts[i] = None
        return True


class WaitingZmqRequestGrpc(WaitingZmqRequest):
    def send_encode_request(self):
        async def send_embedding_port(req_id, receive_count, host_name, embedding_port):
            tasks = []
            # gRPC image-only: flatten modality dict to flat list
            assigned = list(self.num_items_assigned.values())[0]
            logger.info(f"num_items_assigned={assigned}")

            for idx, assigned_num in enumerate(assigned):
                if assigned_num == 0:
                    continue
                encoder_url = self.encoder_urls[idx]
                receive_url = f"{host_name}:{embedding_port}"
                target_url = f"{encoder_url}/SchedulerReceiveUrl"
                logger.info(f"Preparing to send to {target_url}")
                tasks.append(
                    asyncio.to_thread(
                        _grpc_scheduler_receive_url,
                        _grpc_target(encoder_url),
                        req_id,
                        receive_url,
                        receive_count,
                    )
                )

            if not tasks:
                logger.info("No tasks to send.")
                return
            logger.info(f"Concurrently sending {len(tasks)} requests...")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Request {i} failed: {result}")
                else:
                    logger.debug(f"Request {i} succeeded.")

        asyncio.run(
            send_embedding_port(
                self.recv_req.rid,
                self.receive_count,
                self.host_name,
                self.embedding_port,
            )
        )


class WaitingRDMARequest(WaitingMMRequestBase):
    def __init__(
        self,
        rid,
        recv_req,
        mm_processor,
        encoder_urls,
        host_name,
        receive_count,
        zmq_context,
        embeddings_engine,
        dtype,
        gpu_id=0,
        model_type: Optional[str] = None,
        embedding_pool=None,
        embedding_port=None,
    ):
        super().__init__(
            rid=rid,
            recv_req=recv_req,
            mm_processor=mm_processor,
            encoder_urls=encoder_urls,
            model_type=model_type,
            host_name=host_name,
            receive_count=receive_count,
            embedding_pool=embedding_pool,
            zmq_context=zmq_context,
            embedding_port=embedding_port,
        )
        self.embeddings_engine = embeddings_engine
        self.dtype = dtype
        self.gpu_id = gpu_id
        # The receive thread owns the buffer while _receive_running; once
        # _terminal latches, it releases the buffer itself on exit so the
        # scheduler thread never has to wait on it.
        self._buffer_lock = threading.Lock()
        self._terminal = False
        self._receive_running = False

    def send_encode_request(self):
        # Base-class hook. The tokenizer owns /encode, so this rank only pulls
        # sizes and drives the RDMA receive.
        self._receive_running = True
        threading.Thread(target=self._run_receive_in_thread, daemon=True).start()

    def _run_receive_in_thread(self):
        try:
            asyncio.run(self._pull_meta_and_receive_embedding())
        except Exception as e:
            logger.error(f"RDMA receive failed for rid={self.rid}: {e}")
            self._fail_and_release(str(e))
        finally:
            with self._buffer_lock:
                self._receive_running = False
                if self._terminal:
                    self._release_buffer_locked()

    async def _pull_meta_and_receive_embedding(self):
        """Pull per-part sizes, allocate the landing buffer, then drive /send.

        The tokenizer owns /encode; part_idx numbering matches it because both
        derive it from the num_items_assigned frozen onto the request.
        """
        modalities = list(self.num_items_assigned.keys())
        _, modality_num_parts = calculate_modality_num_parts(
            modalities, self.num_items_assigned
        )
        encode_requests = []

        total_num_parts = sum(modality_num_parts.values())
        part_idx_offset = 0
        for modality in modalities:
            assigned_nums = self.num_items_assigned[modality]
            cum_idx = 0
            for idx, assigned_num in enumerate(assigned_nums):
                if assigned_num == 0:
                    continue
                part_idx = part_idx_offset + cum_idx
                encode_requests.append(
                    {
                        "encoder_idx": idx,
                        "part_idx": part_idx,
                        "req_id": create_part_req_id(self.recv_req.rid, part_idx),
                    }
                )
                cum_idx += 1
            part_idx_offset += modality_num_parts[modality]

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=envs.SGLANG_ENCODER_HTTP_TIMEOUT.get())
        ) as session:
            # Phase 1: pull per-part sizes, blocking until the encode publishes.
            tasks = [
                session.post(
                    f"{self.encoder_urls[r['encoder_idx']]}/scheduler_receive_meta_data",
                    json={"req_id": r["req_id"], "part_idx": r["part_idx"]},
                )
                for r in encode_requests
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            if not await self._check_encoder_responses(
                responses, "/scheduler_receive_meta_data"
            ):
                return
            response_json_list = [await r.json() for r in responses]

            # Sort by part_idx
            embedding_sizes, response_sorted, total_bytes = (
                _sort_responses_and_compute_total_bytes(
                    response_json_list, total_num_parts
                )
            )

            # Phase 2: Pre-allocate GPU landing buffer.
            # Prefer the pre-registered persistent pool when available; this avoids
            # per-request register/deregister and keeps the encoder's openSegment
            if total_bytes > 0:
                if self.embedding_pool is not None:
                    alloc_result = await asyncio.to_thread(
                        self.embedding_pool.alloc, total_bytes
                    )
                    if alloc_result is None:
                        # Oversize or alloc timeout — fatal for this request.
                        self._fail_and_release(
                            f"EmbeddingPool could not allocate "
                            f"{total_bytes // (1024 * 1024)}MB (oversize or "
                            f"timeout). Raise SGLANG_EMBEDDING_POOL_SIZE_MB."
                        )
                        return
                    pool_view, buffer_address, slot_id = alloc_result
                    with self._buffer_lock:
                        self.embeddings_buffer = pool_view
                        self._pool_slot_id = slot_id
                    logger.info(
                        f"Pool-allocated Mooncake GPU landing buffer: "
                        f"rid={self.rid}, size={total_bytes}, "
                        f"addr={buffer_address}, slot={slot_id}"
                    )
                else:
                    gpu_buffer = torch.empty(
                        total_bytes, dtype=torch.uint8, device=f"cuda:{self.gpu_id}"
                    )
                    self.embeddings_engine.register(
                        gpu_buffer.data_ptr(), gpu_buffer.nbytes
                    )
                    buffer_address = gpu_buffer.data_ptr()
                    with self._buffer_lock:
                        self.embeddings_buffer = gpu_buffer
                    logger.info(
                        f"Per-request registered Mooncake GPU landing buffer "
                        f"(pool disabled): rid={self.rid}, size={total_bytes}, "
                        f"addr={buffer_address}"
                    )
            else:
                self.embeddings_buffer = None
                buffer_address = 0

            # Abort/timeout may have latched _terminal; don't start RDMA into
            # a buffer that will be released when this thread exits.
            with self._buffer_lock:
                if self._terminal:
                    return

            # Phase 2 cont: POST /send. Metadata carries no routing, so the
            # shard comes from our own part map.
            encoder_idx_by_part = {
                r["part_idx"]: r["encoder_idx"] for r in encode_requests
            }
            offset = 0
            send_tasks = []
            for idx in range(total_num_parts):
                rj = response_sorted[idx]
                rj.update(
                    {
                        "prefill_host": self.host_name,
                        "embedding_port": self.embedding_port,
                        "session_id": self.embeddings_engine.session_id,
                        "buffer_address": offset + buffer_address,
                        # Frees the embedding once all of us have taken it.
                        "receive_count": self.receive_count,
                    }
                )
                send_tasks.append(
                    session.post(
                        f"{self.encoder_urls[encoder_idx_by_part[idx]]}/send",
                        json=rj,
                    )
                )
                offset += embedding_sizes[idx]

            # Phase 3: Wait for RDMA transfers to complete
            send_responses = await asyncio.gather(*send_tasks, return_exceptions=True)
            if not await self._check_encoder_responses(send_responses, "/send"):
                return
            logger.info(f"RDMA transfers completed for rid={self.rid}")

    def _extract_embedding_from_buffer(self, recv_obj, parts) -> None:
        # The embedding already landed in the pre-registered GPU buffer via
        # RDMA; the completion message carries no payload, so
        # recv_obj.embedding stays None.
        pass

    def _prepare_embedding_buffer(self) -> bool:
        # The receive thread already landed the embedding via RDMA (or left
        # the buffer None for the zero-byte case).
        return True

    def _view_dtype(self):
        # Parts carry no payload (aggregate dtype is None); use the model
        # dtype the receiver was constructed with.
        return self.dtype

    def _cleanup_gpu_buffer(self):
        """Latch _terminal and release the GPU buffer. While the receive
        thread runs it owns the buffer (RDMA may be in flight), so release
        is deferred to its exit hook instead of blocking here. Idempotent."""
        with self._buffer_lock:
            self._terminal = True
            if not self._receive_running:
                self._release_buffer_locked()

    def _release_buffer_locked(self):
        """Caller must hold _buffer_lock."""
        if self.embeddings_buffer is None:
            return
        if self._pool_slot_id is not None:
            # Pool-backed: the backing tensor stays registered; just free the slot.
            self.embedding_pool.release(self._pool_slot_id)
            self._pool_slot_id = None
        else:
            try:
                self.embeddings_engine.deregister(self.embeddings_buffer.data_ptr())
            except Exception:
                logger.exception("Failed to deregister GPU buffer for rid=%s", self.rid)
        self.embeddings_buffer = None


async def _extract_encoder_error(responses, endpoint, context, encode_requests=None):
    """Return the first error among gathered encoder responses, or None.

    Pure check — logs each error but has no other side effects; the caller
    decides how to react. ``encode_requests`` optionally enriches each log
    line with the matching request's encoder label.
    """
    for i, resp in enumerate(responses):
        ctx = context
        if encode_requests is not None:
            label = encode_requests[i].get(
                "encoder_url", f"idx={encode_requests[i].get('encoder_idx')}"
            )
            ctx = f"{context}, encoder={label}"
        if isinstance(resp, asyncio.TimeoutError):
            timeout_val = envs.SGLANG_ENCODER_HTTP_TIMEOUT.get()
            logger.error(
                f"Encoder {endpoint} timeout ({timeout_val}s) for {ctx} "
                f"(request {i})"
            )
            return f"Encoder {endpoint} timeout ({timeout_val}s)"
        if isinstance(resp, Exception):
            logger.error(
                f"Encoder {endpoint} failed for {ctx} (request {i}): {resp}",
                exc_info=resp,
            )
            return str(resp)
        if resp.status != 200:
            try:
                err = await resp.json()
                msg = err.get("message", "Unknown error")
            except Exception:
                msg = await resp.text()
            logger.error(f"Encoder {endpoint} returned error {resp.status}: {msg}")
            return msg
    return None


def _sort_responses_and_compute_total_bytes(response_json_list, total_num_parts):
    """Sort responses by part_idx and compute total embedding bytes."""
    embedding_sizes = [None] * total_num_parts
    response_sorted = [None] * total_num_parts
    for rj in response_json_list:
        idx = rj["part_idx"]
        embedding_sizes[idx] = rj["embedding_size"]
        response_sorted[idx] = rj
    total_bytes = sum(s for s in embedding_sizes if s is not None)
    return embedding_sizes, response_sorted, total_bytes


class EmbeddingPool:
    """Persistent GPU buffer pool for received multimodal embeddings.

    Allocator: first-fit on a free-segment list with 256-byte alignment.
    `alloc()` blocks on a Condition when the pool is full and resumes once
    a peer `release()`s a slot; `try_alloc()` is the non-blocking variant
    for callers that re-poll (the zmq_to_scheduler tick). Each successful
    alloc returns a slot_id that must be passed back to release() when the
    consumer is done with the buffer. With `engine` set (mooncake), the
    buffer is registered once so encoder RDMA writes land in pool slots.
    """

    _ALIGN = 256

    def __init__(self, gpu_id: int, size_bytes: int, engine=None):
        self.gpu_id = gpu_id
        self.size_bytes = size_bytes
        self.buffer = torch.empty(
            size_bytes, dtype=torch.uint8, device=f"cuda:{gpu_id}"
        )
        self.base = self.buffer.data_ptr()
        self.engine = engine
        if engine is not None:
            engine.register(self.base, self.buffer.nbytes)
        self._segments_free: List[Tuple[int, int]] = [(0, size_bytes)]
        self._inflight: Dict[int, Tuple[int, int]] = {}
        self._next_slot_id = 0
        self._total_inflight = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        logger.info(
            f"EmbeddingPool allocated: gpu={gpu_id}, "
            f"size={size_bytes // (1024 * 1024)}MB, base=0x{self.base:x}, "
            f"rdma_registered={engine is not None}"
        )

    def try_alloc(self, nbytes: int) -> Optional[Tuple[torch.Tensor, int, int]]:
        """Non-blocking alloc: ``(tensor_view, gpu_addr, slot_id)``, or
        ``None`` when the pool is currently full (oversize requests also get
        ``None`` — callers detect those via ``size_bytes``)."""
        aligned = (nbytes + self._ALIGN - 1) & ~(self._ALIGN - 1)
        with self._lock:
            return self._try_alloc_locked(nbytes, aligned)

    def try_stage(
        self, parts: List[torch.Tensor]
    ) -> Optional[Tuple[torch.Tensor, int]]:
        """Copy CPU part tensors into one slot, packed in list order.

        Returns ``(slot_view, slot_id)``, or ``None`` when the pool is
        currently full. Seam for future async staging (copy streams).
        """
        alloc_result = self.try_alloc(sum(p.nbytes for p in parts))
        if alloc_result is None:
            return None
        slot_view, _, slot_id = alloc_result
        offset = 0
        for part in parts:
            nbytes = part.nbytes
            slot_view[offset : offset + nbytes].copy_(part.flatten().view(torch.uint8))
            offset += nbytes
        return slot_view, slot_id

    def release_on_gc(self, obj, slot_id: int) -> weakref.finalize:
        """Release ``slot_id`` when ``obj`` is GC'd; the returned finalizer
        can be called early to release now (at-most-once either way)."""
        return weakref.finalize(obj, self.release, slot_id)

    def alloc(
        self, nbytes: int, timeout: float = 60.0
    ) -> Optional[Tuple[torch.Tensor, int, int]]:
        """Allocate `nbytes` from the pool.

        Returns ``(tensor_view, gpu_addr, slot_id)`` on success, or ``None``
        when (a) the request is bigger than the pool itself or (b) the wait
        for a free slot exceeds ``timeout`` seconds.

        When the pool is full of in-flight slots, this call blocks the
        calling thread on a Condition until a peer ``release()`` opens
        enough contiguous space.

        NOTE: no ordering guarantee — notify_all + lock race means
        large requests can starve behind small ones, plus thundering-herd.
        """
        if nbytes > self.size_bytes:
            logger.error(
                f"EmbeddingPool: requested {nbytes // (1024 * 1024)}MB "
                f"exceeds pool capacity {self.size_bytes // (1024 * 1024)}MB. "
                f"Raise SGLANG_EMBEDDING_POOL_SIZE_MB."
            )
            return None
        aligned = (nbytes + self._ALIGN - 1) & ~(self._ALIGN - 1)
        deadline = time.monotonic() + timeout
        warned = False
        with self._cond:
            while True:
                slot = self._try_alloc_locked(nbytes, aligned)
                if slot is not None:
                    return slot
                if not warned:
                    inflight_mb = self._total_inflight // (1024 * 1024)
                    cap_mb = self.size_bytes // (1024 * 1024)
                    logger.warning(
                        f"EmbeddingPool full: "
                        f"{inflight_mb}/{cap_mb}MB in-flight across "
                        f"{len(self._inflight)} requests; queueing a "
                        f"{nbytes // (1024 * 1024)}MB request. Raise "
                        f"SGLANG_EMBEDDING_POOL_SIZE_MB if this is frequent."
                    )
                    warned = True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.error(
                        f"EmbeddingPool alloc timed out after "
                        f"{timeout}s waiting for {nbytes // (1024 * 1024)}MB."
                    )
                    return None
                self._cond.wait(timeout=remaining)

    def _try_alloc_locked(
        self, nbytes: int, aligned: int
    ) -> Optional[Tuple[torch.Tensor, int, int]]:
        for i, (off, length) in enumerate(self._segments_free):
            if length >= aligned:
                if length == aligned:
                    self._segments_free.pop(i)
                else:
                    self._segments_free[i] = (off + aligned, length - aligned)
                slot_id = self._next_slot_id
                self._next_slot_id += 1
                self._inflight[slot_id] = (off, aligned)
                self._total_inflight += aligned
                view = self.buffer[off : off + nbytes]
                return view, self.base + off, slot_id
        return None

    def release(self, slot_id: int) -> None:
        """Return a previously-allocated slot to the free list and wake any
        blocked alloc() waiters."""
        with self._cond:
            seg = self._inflight.pop(slot_id, None)
            if seg is None:
                return
            off, aligned = seg
            self._total_inflight -= aligned
            self._coalesce_free_locked(off, aligned)
            self._cond.notify_all()

    def _coalesce_free_locked(self, off: int, length: int) -> None:
        self._segments_free.append((off, length))
        self._segments_free.sort()
        merged: List[Tuple[int, int]] = []
        for s_off, s_len in self._segments_free:
            if merged and merged[-1][0] + merged[-1][1] == s_off:
                p_off, p_len = merged[-1]
                merged[-1] = (p_off, p_len + s_len)
            else:
                merged.append((s_off, s_len))
        self._segments_free = merged


def _iter_part_ranges(embedding_data, dtype):
    """Yield ``(part_idx, shape, byte_start, byte_end)`` for each non-None
    part, packed in part order — the buffer layout shared by the encoder's
    RDMA writes and EmbeddingPool.try_stage."""
    elem_size = torch.tensor([], dtype=dtype).element_size()
    offset = 0
    for i in range(embedding_data.num_parts):
        shape = embedding_data.embedding_shape_list[i]
        if shape is None:
            continue
        nbytes = shape[0] * shape[1] * elem_size
        yield i, shape, offset, offset + nbytes
        offset += nbytes


def _view_pool_buffer_by_modality(raw_buffer, embedding_data, dtype):
    """Zero-copy view of raw_buffer as {modality: [total_tokens, hidden]}.

    Parts of the same modality are contiguous in raw_buffer (the encoder
    writes them modality-outer), so each modality is one reshape of the byte
    range — no per-part split, no torch.cat copy.

    Caller must keep raw_buffer's storage alive while the returned views are
    in use. The pool path binds slot release to mm_inputs GC via finalize.
    """
    # mod -> [byte_start, byte_end, total_tokens, hidden]
    mod_info: Dict[Modality, List[int]] = {}
    for i, shape, start, end in _iter_part_ranges(embedding_data, dtype):
        mod = embedding_data.modality_list[i]
        info = mod_info.get(mod)
        if info is None:
            mod_info[mod] = [start, end, shape[0], shape[1]]
        else:
            assert (
                info[3] == shape[1]
            ), f"hidden_dim mismatch in modality {mod}: {info[3]} vs {shape[1]}"
            assert info[1] == start, f"non-contiguous parts in modality {mod}"
            info[1] = end
            info[2] += shape[0]
    return {
        mod: raw_buffer[s:e].view(dtype).reshape(tokens, hidden)
        for mod, (s, e, tokens, hidden) in mod_info.items()
    }


class MMReceiverBase(ABC):
    def __init__(
        self,
        server_args: ServerArgs,
        dtype: Optional[torch.dtype] = None,
        hf_config: Optional[PretrainedConfig] = None,
        pp_rank: Optional[int] = None,
        tp_rank: Optional[int] = None,
        tp_group: Optional[GroupCoordinator] = None,
        scheduler: Optional["Scheduler"] = None,
        encode_urls: Optional[List[str]] = None,
    ):
        self.context = zmq.asyncio.Context(20)
        # Scheduler-side receive is polled synchronously. Keep one regular ZMQ
        # context alive for the process instead of creating a temporary context
        # whose destruction also closes its per-request socket.
        self.scheduler_context = zmq.Context()
        self.encoder_transfer_backend = get_disagg().encoder_transfer_backend
        # When ``encode_urls`` is shared with an :class:`EncoderBootstrapServer`
        # (tokenizer manager process), it grows / shrinks in place as encoders
        # register or unregister; the receiver always sees the current set.
        # When None (e.g. in a scheduler subprocess that has no in-process
        # bootstrap), fall back to a snapshot of the static --encoder-urls.
        self.encode_urls: List[str] = (
            encode_urls if encode_urls is not None else list(get_disagg().encoder_urls)
        )
        self.recv_timeout = envs.SGLANG_ENCODER_RECV_TIMEOUT.get()
        self.host = get_local_ip_auto(get_serving().host)
        self.pp_rank = pp_rank
        self.tp_rank = tp_rank
        self.tp_size = get_parallel().tp_size
        self.tp_group = tp_group
        self.nnodes = get_parallel().nnodes
        self.hostname = get_local_ip_auto()
        self.waiting_list: List[WaitingMMRequestBase] = []
        self.waiting_by_rid: Dict[str, WaitingMMRequestBase] = {}
        self.scheduler_embedding_port = None
        self.scheduler_recv_socket = None
        if (
            self.encoder_transfer_backend == "zmq_to_scheduler"
            and scheduler is not None
        ):
            (
                self.scheduler_embedding_port,
                self.scheduler_recv_socket,
            ) = get_zmq_socket_on_host(
                self.scheduler_context, zmq.PULL, host=self.hostname
            )
            logger.info(
                "Scheduler TP rank %s reuses ZMQ embedding port %s",
                self.tp_rank,
                self.scheduler_embedding_port,
            )
        self.scheduler = scheduler
        self.gpu_id = scheduler.ps.gpu_id if scheduler is not None else 0
        self.wait_timeout = envs.SGLANG_ENCODER_RECV_TIMEOUT.get()
        self.embedding_pool = None

        self.model_type = (
            getattr(hf_config, "model_type", "").lower()
            if hf_config is not None
            else None
        )
        if self.encoder_transfer_backend == "mooncake":
            self.dtype = dtype
            self.embeddings_engine = get_mooncake_transfer_engine()
            if self.embeddings_engine is None:
                from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                    init_mooncake_transfer_engine,
                )

                self.embeddings_engine = init_mooncake_transfer_engine(
                    hostname=self.host,
                    ib_device=(
                        get_disagg().disaggregation_ib_device
                        or get_exec().moe.mooncake_ib_device
                    ),
                )
            pool_mb = envs.SGLANG_EMBEDDING_POOL_SIZE_MB.get()
            if pool_mb and pool_mb > 0 and scheduler is not None:
                try:
                    self.embedding_pool = EmbeddingPool(
                        self.gpu_id,
                        pool_mb * 1024 * 1024,
                        engine=self.embeddings_engine,
                    )
                except Exception:
                    logger.exception(
                        "Failed to allocate EmbeddingPool, "
                        "falling back to per-request register"
                    )
                    self.embedding_pool = None
            if hf_config is not None:
                self._init_mm_processor(server_args, hf_config)
        elif self.encoder_transfer_backend == "zmq_to_scheduler":
            # Unlike mooncake, do NOT apply the default pool size: explicitly
            # set SGLANG_EMBEDDING_POOL_SIZE_MB=<MB> to bound received
            # embeddings on GPU; unset/0 keeps the unpooled CPU receive.
            if envs.SGLANG_EMBEDDING_POOL_SIZE_MB.is_set() and scheduler is not None:
                pool_mb = envs.SGLANG_EMBEDDING_POOL_SIZE_MB.get()
                if pool_mb and pool_mb > 0:
                    try:
                        self.embedding_pool = EmbeddingPool(
                            self.gpu_id, pool_mb * 1024 * 1024
                        )
                    except Exception:
                        logger.exception(
                            "Failed to allocate EmbeddingPool, "
                            "falling back to unpooled receive"
                        )
                        self.embedding_pool = None
            if hf_config is not None:
                self._init_mm_processor(
                    server_args,
                    hf_config,
                    model_config=(
                        getattr(self.scheduler, "model_config", None)
                        if self.scheduler is not None
                        else None
                    ),
                )

    def _init_mm_processor(
        self,
        server_args: "ServerArgs",
        hf_config: "PretrainedConfig",
        model_config=None,
    ):
        """Load processor and initialize mm_processor, shared by all backends."""
        transport_mode = determine_tensor_transport_mode()
        import_processors("sglang.srt.multimodal.processors")

        extra_kwargs = {}
        if getattr(server_args, "tokenizer_backend", None) is not None:
            extra_kwargs["tokenizer_backend"] = get_serving().tokenizer_backend

        _processor = get_processor(
            get_serving().tokenizer_path,
            tokenizer_mode=get_serving().tokenizer_mode,
            trust_remote_code=get_model().trust_remote_code,
            revision=get_model().revision,
            image_processor_backend=resolve_image_processor_backend(get_mm()),
            **extra_kwargs,
        )

        enable_adaptive_dispatch_to_encoder = (
            get_disagg().enable_adaptive_dispatch_to_encoder
        )
        mm_processor_kwargs = {}
        if model_config is not None:
            mm_processor_kwargs["model_config"] = model_config
        self.mm_processor = get_mm_processor(
            hf_config,
            server_args,
            _processor,
            transport_mode,
            skip_mm_pool=not enable_adaptive_dispatch_to_encoder,
            **mm_processor_kwargs,
        )

    @abstractmethod
    def process_waiting_requests(self, recv_reqs):
        pass

    def abort_waiting_requests(self, recv_req) -> None:
        """Mark matching waiting requests FAIL and free their resources; the
        next process_waiting_requests tick reports the abort through the
        existing FAIL channel. AbortReq is broadcast, so every TP rank does
        this and the status all-reduce stays consistent."""
        for waiting_req in self.waiting_list:
            if not (recv_req.abort_all or waiting_req.rid.startswith(recv_req.rid)):
                continue
            if waiting_req.status in (
                WaitingMMRequestStatus.PENDING,
                WaitingMMRequestStatus.SUCCESS,
            ):
                waiting_req._fail_and_release("Aborted by user", error_code=400)
                waiting_req.release_resources()
                logger.info(f"Abort waiting mm request. rid={waiting_req.rid}")

    async def recv_mm_data(
        self, request_obj, mm_processor, prompt, need_wait_for_mm_inputs=True
    ):
        req_id = None
        try:
            # ``self.encode_urls`` is shared by reference with the bootstrap
            # server (when running) so it always reflects the current set.
            # Snapshot once for the duration of this request to avoid races
            # against concurrent register / unregister.
            encode_urls = list(self.encode_urls)

            if len(encode_urls) == 0 or not need_wait_for_mm_inputs:
                return None
            req_id = uuid.uuid4().hex
            embedding_port, recv_socket = get_zmq_socket_on_host(
                self.context, zmq.PULL, host=self.host
            )
            mm_data = self._extract_url_data(request_obj)
            modalities = [m.get("modality") for m in mm_data]
            logger.info(
                f"[{req_id}] Sending encode request to E, "
                f"modalities={modalities}, num_items={len(mm_data)}"
            )
            send_time = time.monotonic()
            encode_task = asyncio.create_task(
                self.encode(
                    req_id,
                    mm_data,
                    embedding_port,
                    "encode",
                    encode_urls=encode_urls,
                )
            )
            # Parts stream onto the socket while other parts are still
            # encoding, so receive concurrently with the dispatch; the dispatch
            # result only matters for failing fast when nothing will arrive.
            recv_task = asyncio.create_task(
                self._recv_mm_data(req_id, recv_socket, mm_processor, prompt)
            )
            done, _ = await asyncio.wait(
                {encode_task, recv_task},
                timeout=self.recv_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if (
                done
                and recv_task not in done
                and (
                    encode_task.exception() is not None or encode_task.result() is False
                )
            ):
                logger.warning(
                    f"[{req_id}] Encoder dispatch failed; skipping embedding wait"
                )
                recv_task.cancel()
                return None
            result = await asyncio.wait_for(
                recv_task,
                timeout=self.recv_timeout - (time.monotonic() - send_time),
            )
            elapsed = time.monotonic() - send_time
            logger.info(f"[{req_id}] Received embedding from E in {elapsed:.3f}s")
            return result
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - send_time
            logger.warning(f"[{req_id}] Embedding recv timeout after {elapsed:.3f}s")
            return None

    async def _recv_mm_data(self, req_id, recv_socket, mm_processor, prompt):
        """zmq_to_tokenizer receive: embedding parts arrive as 2-frame ZMQ
        messages on the tokenizer's PULL socket. (mooncake/zmq_to_scheduler lands on scheduler ranks via
        WaitingMMRequest.)"""
        if req_id is None:
            return None

        recv_embedding_data: MultiModalEmbeddingData = None

        try:
            while recv_embedding_data is None or not recv_embedding_data.ready:
                parts = await recv_socket.recv_multipart(copy=False)
                if not parts:
                    continue
                recv_obj: EmbeddingData = safe_pickle_loads(parts[0])
                if getattr(recv_obj, "error_msg", None) is not None:
                    logger.warning(
                        f"Encoder error for req_id={req_id}: {recv_obj.error_msg} "
                        f"error_code={getattr(recv_obj, 'error_code', None)}"
                    )
                    return None
                logger.debug("recv_obj=%s", recv_obj)
                # Normalize the part req_id to the original for aggregation.
                recv_obj.req_id = extract_original_req_id(recv_obj.req_id)
                if len(parts) < 2:
                    logger.error(
                        "zmq_to_tokenizer expected 2-part message, got %d parts",
                        len(parts),
                    )
                    return None
                buffer = parts[1].buffer if hasattr(parts[1], "buffer") else parts[1]
                # Clone so we don't depend on ZMQ buffer after next recv.
                recv_obj.embedding = (
                    torch.frombuffer(buffer, dtype=recv_obj.dtype)
                    .reshape(recv_obj.shape)
                    .clone()
                )
                recv_embedding_data = _aggregate_embedding_part(
                    recv_embedding_data, recv_obj, self.model_type
                )

            recv_embedding = recv_embedding_data.get_embedding(is_concat=True)
            return mm_processor.get_mm_data(
                prompt,
                recv_embedding,
                **recv_embedding_data.get_mm_extra_meta(),
            )
        finally:
            recv_socket.close()

    def send_encode_request(self, obj, time_stats_json=None):
        self._send_encode_request(obj, time_stats_json=time_stats_json)

    def _send_encode_request(self, obj, time_stats_json=None):
        mm_data = self._extract_url_data(obj)
        if obj.rid is None:
            obj.rid = uuid.uuid4().hex

        # ``self.encode_urls`` is the shared list maintained by the bootstrap
        # server (and pre-populated with --encoder-urls); take a snapshot for
        # the duration of this dispatch.
        encode_urls = list(self.encode_urls)

        if mm_data and encode_urls:
            logger.info(
                f"Dispatching {len(mm_data)} mm items to {len(encode_urls)} "
                f"encoder(s) {encode_urls} for request {obj.rid}"
            )
            obj.need_wait_for_mm_inputs = True

            num_items_assigned = self._assign_items_by_modality(
                mm_data, len(encode_urls)
            )
            obj.num_items_assigned = num_items_assigned
            # Freeze the encoder URL snapshot onto obj so the scheduler
            # subprocess uses the same list when indexing encoder_idx.
            obj.encoder_urls = encode_urls

            encode_thread = threading.Thread(
                target=self._run_encode_in_thread,
                args=(
                    obj.rid,
                    mm_data,
                    "encode",
                    num_items_assigned,
                    encode_urls,
                    time_stats_json,
                ),
                daemon=True,
            )
            encode_thread.start()
        else:
            # No encoder URLs available (bootstrap may not have any registered yet);
            # reset the flag so the scheduler does not wait for embeddings that will
            # never arrive.  A warning is emitted so the user can diagnose why
            # disaggregation is not happening for this request.
            if mm_data:
                logger.warning(
                    f"No encoder URLs available for request {obj.rid}; "
                    "processing without encoder disaggregation."
                )
            obj.need_wait_for_mm_inputs = False

    def _sync_fail_info_across_tp(self, waiting_req: WaitingMMRequestBase) -> None:
        """Share encoder error fields across TP ranks before abort.

        The encoder sends ZMQ error signals to each TP rank's receive socket,
        but they can arrive at different times. ``all_reduce`` on status makes
        every rank enter FAIL together while only some ranks have populated
        ``error_msg`` / ``error_code``. attn_tp_rank 0 streams the abort to the
        client, so merge the best-known payload from all ranks first.
        """
        if self.tp_size <= 1 or self.tp_group is None:
            return

        gathered = self.tp_group.all_gather_object(
            (waiting_req.error_msg, waiting_req.error_code)
        )
        best_msg = waiting_req.error_msg
        best_code = waiting_req.error_code
        for msg, code in gathered:
            if msg is not None:
                best_msg = msg
            if code is not None:
                best_code = code
        waiting_req.error_msg = best_msg
        waiting_req.error_code = best_code

    # For zmq_to_scheduler
    def _drain_scheduler_embeddings(self):
        if self.scheduler_recv_socket is None:
            return

        while True:
            try:
                parts = self.scheduler_recv_socket.recv_multipart(
                    flags=zmq.NOBLOCK, copy=False
                )
            except zmq.Again:
                return

            recv_obj: EmbeddingData = safe_pickle_loads(parts[0])
            rid = extract_original_req_id(recv_obj.req_id)
            waiting_req = self.waiting_by_rid.get(rid)
            if waiting_req is None:
                logger.warning(
                    "Dropping embedding data for inactive request %s", recv_obj.req_id
                )
                continue
            waiting_req.consume_parts(parts)

    def _process_waiting_requests(self, recv_reqs, waiting_cls, **extra_kwargs):
        new_recv_reqs = []
        for recv_req in recv_reqs:
            if (
                isinstance(recv_req, TokenizedGenerateReqInput)
                and recv_req.need_wait_for_mm_inputs is True
            ):
                # Use the URL snapshot frozen by the tokenizer when it
                # computed num_items_assigned -- the encoder_idx values in
                # that assignment must index into this exact list.  Falling
                # back to ``self.encode_urls`` would only matter if the
                # tokenizer never set encoder_urls (legacy / static path).
                encode_urls = recv_req.encoder_urls or list(self.encode_urls)

                waiting_req = waiting_cls(
                    rid=recv_req.rid,
                    recv_req=recv_req,
                    mm_processor=self.mm_processor,
                    encoder_urls=encode_urls,
                    model_type=self.model_type,
                    host_name=self.hostname,
                    receive_count=self.tp_size,
                    zmq_context=(
                        None
                        if self.scheduler_recv_socket is not None
                        else self.scheduler_context
                    ),
                    embedding_port=self.scheduler_embedding_port,
                    **extra_kwargs,
                )
                if self.scheduler_recv_socket is not None:
                    self.waiting_by_rid[waiting_req.rid] = waiting_req
                waiting_req.send_encode_request()
                self.waiting_list.append(waiting_req)
            else:
                new_recv_reqs.append(recv_req)

        if len(self.waiting_list) == 0:
            return new_recv_reqs, []

        self._drain_scheduler_embeddings()
        current_time = time.time()
        local_status = []
        for waiting_req in self.waiting_list:
            # Per-request sockets receive here; shared-socket requests use this
            # tick to retry pool staging after _drain_scheduler_embeddings().
            waiting_req._try_recv_mm_data()
            if current_time - waiting_req.start_time > self.wait_timeout:
                waiting_req.status = WaitingMMRequestStatus.TIMEOUT
                waiting_req.release_resources()
                waiting_req.close_recv_socket()
            local_status.append(waiting_req.status)

        local_status = torch.tensor(local_status, device="cpu", dtype=torch.int32)

        torch.distributed.all_reduce(
            local_status,
            op=torch.distributed.ReduceOp.MIN,
            group=self.tp_group.cpu_group,
        )

        new_waiting = []
        abort_reqs = []
        for i, waiting_req in enumerate(self.waiting_list):
            status_value = local_status[i].item()
            if status_value == WaitingMMRequestStatus.SUCCESS:
                new_recv_reqs.append(waiting_req.recv_req)
            elif status_value == WaitingMMRequestStatus.FAIL:
                self._sync_fail_info_across_tp(waiting_req)
                logger.error(
                    f"Waiting request {waiting_req.rid} failed: {waiting_req.error_msg} {waiting_req.error_code = }"
                )
                # A peer's FAIL can force-abort this locally PENDING/SUCCESS
                # rank, so release any buffer/slot it still holds.
                waiting_req.release_resources()
                abort_reqs.append(
                    (
                        self.create_req(waiting_req.recv_req),
                        waiting_req.error_msg,
                        waiting_req.error_code,
                    )
                )
            elif status_value == WaitingMMRequestStatus.TIMEOUT:
                logger.error(
                    f"Timed out waiting for image embeddings for request {waiting_req.rid}"
                )
                waiting_req.release_resources()
                abort_reqs.append(
                    (
                        self.create_req(waiting_req.recv_req),
                        f"Timeout waiting for image embedding after {self.wait_timeout}s",
                        HTTPStatus.REQUEST_TIMEOUT,
                    )
                )
            else:  # status_value == WaitingMMRequestStatus.PENDING
                new_waiting.append(waiting_req)
                continue
            self.waiting_by_rid.pop(waiting_req.rid, None)

        self.waiting_list = new_waiting
        return new_recv_reqs, abort_reqs

    def _run_encode_in_thread(
        self,
        req_id,
        mm_data,
        endpoint_encode,
        num_items_assigned,
        encode_urls=None,
        time_stats_json=None,
    ):
        # ``embedding_port`` is always None on this path: zmq_to_scheduler /
        # mooncake ranks register their receive ports with the encoder later
        # via /scheduler_receive_url, so the dispatch itself carries no port.
        try:
            asyncio.run(
                self.encode(
                    req_id=req_id,
                    mm_data=mm_data,
                    embedding_port=None,
                    endpoint_encode=endpoint_encode,
                    num_items_assigned=num_items_assigned,
                    encode_urls=encode_urls,
                    time_stats_json=time_stats_json,
                )
            )
        except Exception as e:
            logger.error(f"Encode failed for request {req_id}: {e}", exc_info=True)

    def create_req(self, recv_req: TokenizedGenerateReqInput):
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
            return_logprob=recv_req.return_logprob,
            top_logprobs_num=recv_req.top_logprobs_num,
            token_ids_logprob=recv_req.token_ids_logprob,
            stream=recv_req.stream,
            lora_id=recv_req.lora_id,
            input_embeds=recv_req.input_embeds,
            custom_logit_processor=recv_req.custom_logit_processor,
            require_reasoning=recv_req.require_reasoning,
            return_hidden_states=recv_req.return_hidden_states,
            return_routed_experts=recv_req.return_routed_experts,
            routed_experts_start_len=recv_req.routed_experts_start_len,
            eos_token_ids=self.scheduler.model_config.hf_eos_token_id,
            bootstrap_host=recv_req.bootstrap_host,
            bootstrap_port=recv_req.bootstrap_port,
            bootstrap_room=recv_req.bootstrap_room,
            disagg_mode=self.scheduler.disaggregation_mode,
            routed_dp_rank=recv_req.routed_dp_rank,
            disagg_prefill_dp_rank=recv_req.disagg_prefill_dp_rank,
            vocab_size=self.scheduler.model_config.vocab_size,
            priority=recv_req.priority,
            metrics_collector=(
                self.scheduler.metrics_collector
                if self.scheduler.metrics_reporter.enable_metrics
                else None
            ),
            extra_key=recv_req.extra_key,
            cache_salt=recv_req.cache_salt,
            http_worker_ipc=recv_req.http_worker_ipc,
            dllm_config=self.scheduler.dllm_config,
        )
        req.tokenizer = self.scheduler.tokenizer
        return req

    def _assign_items_by_modality(
        self, mm_data, encoder_num, random_shuffle=True
    ) -> Dict:
        """
        Assign multimodal items across encoders by modality with cross-modality load balancing.

        Args:
            mm_data: List of multimodal data items, each with a "modality" key
            encoder_num: Number of encoders
            random_shuffle: Whether to shuffle the encoder indices

        Returns:
            Dictionary mapping modality to list of assignment counts per encoder
            Format: {modality: [count_for_encoder_0, count_for_encoder_1, ...]}
        """
        encode_idx = list(range(encoder_num))
        if random_shuffle:
            random.shuffle(encode_idx)
        # Get unique modalities with order preserved
        modalities = list(dict.fromkeys(mm_item.get("modality") for mm_item in mm_data))
        # Use OrderedDict to explicitly maintain modality order
        num_items_assigned = OrderedDict()
        current_offset = 0

        for modality in modalities:
            mm_data_modality = [
                mm_item for mm_item in mm_data if mm_item.get("modality") == modality
            ]
            num_items = len(mm_data_modality)
            if num_items == 0:
                continue

            base = num_items // len(encode_idx)
            remainder = num_items % len(encode_idx)
            # Rotate assignments based on current_offset to balance load across modalities
            assignments = [0] * len(encode_idx)
            for i in range(len(encode_idx)):
                # keep shuffle order when assigning items to encoders
                pos_in_shuffled = (current_offset + i) % len(encode_idx)
                actual_encoder_idx = encode_idx[pos_in_shuffled]
                assignments[actual_encoder_idx] = base + (1 if i < remainder else 0)
            num_items_assigned[modality] = assignments
            current_offset = (current_offset + remainder) % len(encode_idx)

        return num_items_assigned

    def _extract_url_data(self, request_obj: GenerateReqInput) -> List[Dict]:
        def flatten_mm_items(items):
            if not isinstance(items, list):
                return [items]

            flat = []
            for item in items:
                if isinstance(item, (list, tuple)):
                    flat.extend(flatten_mm_items(list(item)))
                else:
                    flat.append(item)
            return flat

        def to_raw_url(mm_item):
            if isinstance(mm_item, ImageData):
                return mm_item.url
            if isinstance(mm_item, dict):
                # tolerate {"url": ...} shaped payloads
                return mm_item.get("url", mm_item)
            return mm_item

        mm_data = []
        image_hashes = request_obj.mm_content_hashes
        image_index = 0
        for mm_items, modality in [
            (request_obj.image_data, Modality.IMAGE),
            (request_obj.video_data, Modality.VIDEO),
            (request_obj.audio_data, Modality.AUDIO),
        ]:
            if mm_items:
                mm_items = flatten_mm_items(mm_items)
                for mm_item in mm_items:
                    if mm_item is None:
                        continue
                    raw_url = to_raw_url(mm_item)
                    if raw_url is None:
                        continue
                    entry = {
                        "url": raw_url,
                        "modality": modality,
                    }
                    entry.update(
                        media_preprocess_kwargs(mm_item, defaults={"detail": "auto"})
                    )
                    if modality == Modality.IMAGE:
                        inline_hash = (
                            mm_item.content_hash
                            if isinstance(mm_item, ImageData)
                            else (
                                mm_item.get("content_hash")
                                if isinstance(mm_item, dict)
                                else None
                            )
                        )
                        explicit_hash = (
                            image_hashes[image_index]
                            if image_hashes is not None
                            and image_index < len(image_hashes)
                            else None
                        )
                        entry["content_hash"] = explicit_hash or inline_hash
                        image_index += 1
                    mm_data.append(entry)
        if image_hashes is not None and image_index != len(image_hashes):
            raise ValueError(
                f"mm_content_hashes has {len(image_hashes)} entries for "
                f"{image_index} images"
            )
        return mm_data


class MMReceiverHTTP(MMReceiverBase):
    def __init__(
        self,
        server_args: ServerArgs,
        dtype: Optional[torch.dtype] = None,
        hf_config: Optional[PretrainedConfig] = None,
        pp_rank: Optional[int] = None,
        tp_rank: Optional[int] = None,
        tp_group: Optional[GroupCoordinator] = None,
        scheduler: Optional["Scheduler"] = None,
        encode_urls: Optional[List[str]] = None,
    ):
        super().__init__(
            server_args,
            dtype=dtype,
            hf_config=hf_config,
            pp_rank=pp_rank,
            tp_rank=tp_rank,
            tp_group=tp_group,
            scheduler=scheduler,
            encode_urls=encode_urls,
        )

    # For zmq_to_scheduler and mooncake
    def process_waiting_requests(self, recv_reqs):
        if self.encoder_transfer_backend == "mooncake":
            return self._process_waiting_requests(
                recv_reqs,
                WaitingRDMARequest,
                embeddings_engine=self.embeddings_engine,
                dtype=self.dtype,
                gpu_id=self.gpu_id,
                embedding_pool=self.embedding_pool,
            )
        return self._process_waiting_requests(
            recv_reqs, WaitingZmqRequest, embedding_pool=self.embedding_pool
        )

    async def encode(
        self,
        req_id,
        mm_data,
        embedding_port,
        endpoint_encode,
        num_items_assigned=None,
        encode_urls=None,
        time_stats_json=None,
    ):
        if len(mm_data) == 0:
            return

        effective_urls = encode_urls if encode_urls is not None else self.encode_urls

        # get unique modalities with order preserved
        modalities = [mm_item.get("modality") for mm_item in mm_data]
        modalities = list(dict.fromkeys(modalities))
        encode_requests = []

        if num_items_assigned is None:
            num_items_assigned = self._assign_items_by_modality(
                mm_data, len(effective_urls)
            )

        # Calculate total num_parts across all modalities
        total_num_parts, modality_num_parts = calculate_modality_num_parts(
            modalities, num_items_assigned
        )

        part_idx_offset = 0
        for modality in modalities:
            num_items_assigned_modality = num_items_assigned.get(modality)
            mm_data_modality = [
                mm_item for mm_item in mm_data if mm_item.get("modality") == modality
            ]

            num_parts = modality_num_parts[modality]
            cum_num_items = 0
            cum_idx = 0
            for idx, assigned_num in enumerate(num_items_assigned_modality):
                if assigned_num == 0:
                    continue
                part_idx = part_idx_offset + cum_idx
                part_req_id = create_part_req_id(req_id, part_idx)
                encode_requests.append(
                    {
                        "encoder_idx": idx,
                        "encoder_url": effective_urls[idx],
                        "mm_items": [
                            _encoder_media_item(mm_item)
                            for mm_item in mm_data_modality[
                                cum_num_items : cum_num_items + assigned_num
                            ]
                        ],
                        "num_parts": total_num_parts,
                        "part_idx": part_idx,
                        "req_id": part_req_id,  # use part_req_id to avoid key collision
                        "modality": modality.name,  # convert enum to string for json serialization
                        "prefill_host": self.host,
                        "embedding_port": embedding_port,
                        "time_stats_json": time_stats_json,
                    }
                )
                cum_idx += 1
                cum_num_items += assigned_num
            part_idx_offset += num_parts

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=envs.SGLANG_ENCODER_HTTP_TIMEOUT.get())
        ) as session:
            # Send encode requests

            tasks = [
                session.post(
                    f"{effective_urls[encode_request['encoder_idx']]}/{endpoint_encode}",
                    json=encode_request,
                )
                for encode_request in encode_requests
            ]

            responses = await asyncio.gather(*tasks, return_exceptions=True)
            # Dispatch only. The embedding never comes back through this call:
            # zmq_to_tokenizer is pushed to our PULL socket during /encode,
            # zmq_to_scheduler to the ports its ranks registered, and mooncake
            # by RDMA once those ranks have pulled sizes and driven /send.
            return (
                await _extract_encoder_error(
                    responses, "HTTP request", f"req_id={req_id}", encode_requests
                )
                is None
            )


class MMReceiverGrpc(MMReceiverBase):
    def __init__(
        self,
        server_args: ServerArgs,
        dtype: Optional[torch.dtype] = None,
        hf_config: Optional[PretrainedConfig] = None,
        pp_rank: Optional[int] = None,
        tp_rank: Optional[int] = None,
        tp_group: Optional[GroupCoordinator] = None,
        scheduler: Optional["Scheduler"] = None,
        encode_urls: Optional[List[str]] = None,
    ):
        if get_disagg().encoder_transfer_backend == "mooncake":
            # The RDMA receive path (WaitingRDMARequest + /meta + /send) only
            # exists for HTTP encoders; gRPC has no RDMA-capable receive.
            raise NotImplementedError(
                "mooncake encoder_transfer_backend requires HTTP encoders; "
                "use zmq_to_scheduler / zmq_to_tokenizer with gRPC."
            )
        super().__init__(
            server_args,
            dtype=dtype,
            hf_config=hf_config,
            pp_rank=pp_rank,
            tp_rank=tp_rank,
            tp_group=tp_group,
            scheduler=scheduler,
            encode_urls=encode_urls,
        )

    # For zmq_to_scheduler
    def process_waiting_requests(self, recv_reqs):
        return self._process_waiting_requests(recv_reqs, WaitingZmqRequestGrpc)

    async def encode(
        self,
        req_id,
        mm_data,
        embedding_port,
        endpoint_encode,
        num_items_assigned=None,
        encode_urls=None,
    ):
        if not mm_data:
            return

        effective_urls = encode_urls if encode_urls is not None else self.encode_urls

        # gRPC currently only supports image; flatten new dict formats to simple lists
        if mm_data and isinstance(mm_data[0], dict):
            non_image = [
                item.get("modality")
                for item in mm_data
                if item.get("modality") != Modality.IMAGE
            ]
            if non_image:
                raise NotImplementedError(
                    f"gRPC encode only supports IMAGE modality, got: {non_image}"
                )
            img_data = [item.get("url") for item in mm_data]
        else:
            img_data = mm_data
        if isinstance(num_items_assigned, dict):
            num_items_assigned = list(num_items_assigned.values())[0]

        encode_requests = []
        if num_items_assigned is None:
            encode_idx = list(range(len(effective_urls)))
            random.shuffle(encode_idx)
            num_items_assigned = [
                (idx + len(img_data)) // len(effective_urls) for idx in encode_idx
            ]
        num_parts = sum(1 for x in num_items_assigned if x != 0)
        cum_num_items = 0
        cum_idx = 0
        for idx, assigned_num in enumerate(num_items_assigned):
            if assigned_num == 0:
                continue
            start = cum_num_items
            end = cum_num_items + assigned_num
            encode_requests.append(
                {
                    "encoder_idx": idx,
                    "mm_items": img_data[start:end],
                    "num_parts": num_parts,
                    "part_idx": cum_idx,
                    "req_id": req_id,
                    "prefill_host": self.host,
                    "embedding_port": embedding_port,
                }
            )
            cum_idx += 1
            cum_num_items += assigned_num

        grpc_tasks = [
            asyncio.to_thread(
                _grpc_encode_request,
                _grpc_target(effective_urls[encode_request["encoder_idx"]]),
                encode_request,
            )
            for encode_request in encode_requests
        ]
        await asyncio.gather(*grpc_tasks)


def _validate_transport_mode(transport_mode: str, encoder_urls):
    if transport_mode == "grpc":
        invalid_prefix = "http://"
        error_msg = (
            "EPD MMReceiver: grpc mode requires grpc:// encoder URLs. "
            "Set SGLANG_ENCODER_MM_RECEIVER_MODE=http for http:// URLs."
        )
    elif transport_mode == "http":
        invalid_prefix = "grpc://"
        error_msg = (
            "EPD MMReceiver: http mode requires http:// encoder URLs. "
            "Set SGLANG_ENCODER_MM_RECEIVER_MODE=grpc for grpc:// URLs."
        )
    else:
        return

    if any(url.startswith(invalid_prefix) for url in encoder_urls):
        raise ValueError(error_msg)


_MM_RECEIVER_BY_MODE = {
    "grpc": MMReceiverGrpc,
    "http": MMReceiverHTTP,
}


def create_mm_receiver(
    server_args: ServerArgs,
    dtype: Optional[torch.dtype] = None,
    hf_config: Optional[PretrainedConfig] = None,
    pp_rank: Optional[int] = None,
    tp_rank: Optional[int] = None,
    tp_group: Optional[GroupCoordinator] = None,
    scheduler: Optional["Scheduler"] = None,
    transport_mode: Optional[str] = None,
    encode_urls: Optional[List[str]] = None,
):
    if transport_mode is None:
        transport_mode = envs.SGLANG_ENCODER_MM_RECEIVER_MODE.get()
        logger.debug(f"MMReceiver transport_mode from env: {transport_mode}")

    _validate_transport_mode(transport_mode, encode_urls or get_disagg().encoder_urls)
    logger.info(f"EPD MMReceiver: using transport_mode={transport_mode}")

    receiver_cls = _MM_RECEIVER_BY_MODE.get(transport_mode)
    if receiver_cls is None:
        raise ValueError(f"Unsupported transport_mode: {transport_mode}")
    return receiver_cls(
        server_args,
        dtype=dtype,
        hf_config=hf_config,
        pp_rank=pp_rank,
        tp_rank=tp_rank,
        tp_group=tp_group,
        scheduler=scheduler,
        encode_urls=encode_urls,
    )
