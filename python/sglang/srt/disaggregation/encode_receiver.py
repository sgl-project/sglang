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
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    MooncakeMMUrlItem,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.schedule_batch import Modality, Req
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import ImageData
from sglang.srt.utils.common import safe_pickle_loads
from sglang.srt.utils.hf_transformers_utils import get_processor
from sglang.srt.utils.network import (
    NetworkAddress,
    get_local_ip_auto,
    get_zmq_socket_on_host,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


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
    ``SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL`` (seconds; 0 disables)
    and ``SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_TIMEOUT`` (seconds).  Explicit
    constructor args take precedence over the env vars.
    """

    def __init__(
        self,
        host: str,
        port: int,
        urls: Optional[List[str]] = None,
        health_check_interval: Optional[float] = None,
        health_check_timeout: Optional[float] = None,
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
        self._consecutive_failures: Dict[str, int] = {}
        self._max_consecutive_failures = 3

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
            self._consecutive_failures.pop(url, None)
            if url not in self._urls:
                self._urls.append(url)
                logger.info(f"Registered encoder URL: {url}")
                return True
            logger.debug(f"Encoder URL already registered: {url}")
            return False

    def unregister(self, url: str) -> bool:
        """Remove *url* if present.  Returns True if removed."""
        with self._lock:
            if url in self._urls:
                self._urls.remove(url)
                self._consecutive_failures.pop(url, None)
                logger.info(f"Unregistered encoder URL: {url}")
                return True
            return False

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
        """Probe each registered encoder periodically and evict dead ones."""

        timeout = ClientTimeout(total=self._health_check_timeout)
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                snapshot = self.list_urls()
                if not snapshot:
                    continue
                async with ClientSession(timeout=timeout) as session:
                    results = await asyncio.gather(
                        *(self._probe(session, url) for url in snapshot),
                        return_exceptions=True,
                    )
                evicted = []
                with self._lock:
                    for url, ok in zip(snapshot, results):
                        if ok is True:
                            self._consecutive_failures.pop(url, None)
                        else:
                            self._consecutive_failures[url] = (
                                self._consecutive_failures.get(url, 0) + 1
                            )
                            if (
                                self._consecutive_failures[url]
                                >= self._max_consecutive_failures
                            ):
                                if url in self._urls:
                                    self._urls.remove(url)
                                self._consecutive_failures.pop(url, None)
                                evicted.append(url)
                if evicted:
                    logger.warning(
                        f"Health check evicted {len(evicted)} encoder(s) "
                        f"after {self._max_consecutive_failures} consecutive "
                        f"failures: {evicted}"
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


def _grpc_send_request(target, request_json):
    import grpc
    from smg_grpc_proto import sglang_encoder_pb2, sglang_encoder_pb2_grpc

    timeout_secs = envs.SGLANG_ENCODER_GRPC_TIMEOUT_SECS.get()
    channel = grpc.insecure_channel(target)
    stub = sglang_encoder_pb2_grpc.SglangEncoderStub(channel)
    try:
        stub.Send(
            sglang_encoder_pb2.SendRequest(
                req_id=request_json["req_id"],
                prefill_host=request_json["prefill_host"],
                embedding_port=request_json["embedding_port"],
                session_id=request_json["session_id"],
                buffer_address=request_json["buffer_address"],
            ),
            timeout=timeout_secs,
        )
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
        self.cached_embedding = None
        self.error_msg = error_msg
        self.error_code = error_code
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
            # cached_embedding is a GPU tensor used only by mooncake's in-process
            if key.startswith("_") or key in ("embedding", "cached_embedding"):
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


class WaitingImageRequestStatus(IntEnum):
    FAIL = -1
    PENDING = 0
    SUCCESS = 1
    TIMEOUT = -2


def create_part_req_id(original_req_id: str, part_idx: int) -> str:
    """Create a unique part request ID by appending part index suffix."""
    return f"{original_req_id}_local_part_{part_idx}"


def extract_original_req_id(part_req_id: str) -> str:
    """Extract the original request ID from a part request ID."""
    if "_local_part_" in part_req_id:
        return part_req_id.rsplit("_local_part_", 1)[0]
    return part_req_id


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


# For zmq_to_scheduler
class WaitingImageRequest:
    def __init__(
        self,
        rid: str,
        recv_req: TokenizedGenerateReqInput,
        mm_processor,
        encoder_urls,
        model_type,
        host_name,
        receive_count,
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
        self.embedding_port, self.recv_socket = get_zmq_socket_on_host(
            zmq.Context(), zmq.PULL, host=host_name
        )
        logger.info(f"Waiting for input {self.embedding_port = }")
        self.recv_embedding_data = None
        # ok=1 pending=0 fail=-1
        self.status = WaitingImageRequestStatus.PENDING
        self.error_msg = None
        self.error_code = None
        self.start_time = time.time()

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

        asyncio.run(
            send_embedding_port(
                self.recv_req.rid,
                self.receive_count,
                self.host_name,
                self.embedding_port,
            )
        )

    def _try_recv_mm_data(self):
        if self.status != WaitingImageRequestStatus.PENDING:
            return
        while self.recv_embedding_data is None or not self.recv_embedding_data.ready:
            try:
                parts = self.recv_socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
            except zmq.Again:
                # No data available yet, wait a bit and retry
                return
            recv_obj: EmbeddingData = safe_pickle_loads(parts[0])
            if getattr(recv_obj, "error_msg", None) is not None:
                logger.warning(
                    f"Received error signal from encoder for {self.rid}: {recv_obj.error_msg} {recv_obj.error_code = }"
                )
                self.error_msg = recv_obj.error_msg
                self.error_code = recv_obj.error_code
                self.status = WaitingImageRequestStatus.FAIL
                self.recv_socket.close()
                return

            # Extract original req_id from part_req_id and drop stale payloads
            # that may arrive on a reused ZMQ port after a prior request aborted.
            original_req_id = extract_original_req_id(recv_obj.req_id)
            if original_req_id != self.recv_req.rid:
                logger.warning(
                    f"Dropping stale embedding data: expected rid={self.recv_req.rid}, "
                    f"got rid={recv_obj.req_id} (likely from ZMQ port reuse)"
                )
                continue
            recv_obj.req_id = original_req_id

            buffer = parts[1].buffer if hasattr(parts[1], "buffer") else parts[1]
            recv_obj.embedding = (
                torch.frombuffer(buffer, dtype=recv_obj.dtype)
                .reshape(recv_obj.shape)
                .clone()
            )

            if self.recv_embedding_data is None:
                self.recv_embedding_data = MultiModalEmbeddingData.from_embedding_data(
                    recv_obj, model_type=self.model_type
                )
            else:
                self.recv_embedding_data.add(recv_obj)

        recv_embedding = self.recv_embedding_data.get_embedding(is_concat=True)
        mm_inputs = self.mm_processor.get_mm_data(
            self.recv_req.input_text,
            recv_embedding,
            **self.recv_embedding_data.get_mm_extra_meta(),
        )
        self.recv_req.mm_inputs = mm_inputs
        self.recv_req.input_ids = array("q", mm_inputs.input_ids)
        self.status = WaitingImageRequestStatus.SUCCESS
        self.recv_socket.close()

    def _cleanup_gpu_buffer(self):
        pass


class WaitingImageRequestGrpc(WaitingImageRequest):
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


class WaitingImageRDMARequest(WaitingImageRequest):
    def __init__(
        self,
        rid,
        recv_req,
        mm_processor,
        encoder_urls,
        host_name,
        receive_count,
        embeddings_engine,
        dtype,
        gpu_id=0,
        model_type: Optional[str] = None,
        embedding_pool=None,
    ):
        super().__init__(
            rid=rid,
            recv_req=recv_req,
            mm_processor=mm_processor,
            encoder_urls=encoder_urls,
            model_type=model_type,
            host_name=host_name,
            receive_count=receive_count,
        )
        self.embeddings_engine = embeddings_engine
        self.dtype = dtype
        self.gpu_id = gpu_id
        self.embeddings_buffer = None
        self.embedding_pool = embedding_pool
        self._buffer_from_pool = False
        self._pool_slot_id: Optional[int] = None

    def send_encode_request(self):
        self._encode_thread = threading.Thread(
            target=self._run_encode_in_thread, daemon=True
        )
        self._encode_thread.start()

    def _run_encode_in_thread(self):
        try:
            asyncio.run(self._send_encode_and_rdma_request())
        except Exception as e:
            logger.error(f"RDMA encode request failed for rid={self.rid}: {e}")
            self.status = WaitingImageRequestStatus.FAIL
            self.error_msg = str(e)
            self._cleanup_gpu_buffer()
            self.recv_socket.close()

    async def _send_encode_and_rdma_request(self):
        modalities = list(self.num_items_assigned.keys())
        _, modality_num_parts = calculate_modality_num_parts(
            modalities, self.num_items_assigned
        )
        encode_requests = []
        # Use the URL list captured at tokenizer time.  TokenizedGenerateReqInput
        # has no image_data field, so reading recv_req.image_data here would
        # always return None and produce empty mm_items.
        mm_data_all = self.recv_req.mm_data_mooncake or []

        total_num_parts = sum(modality_num_parts.values())
        part_idx_offset = 0
        for modality in modalities:
            assigned_nums = self.num_items_assigned[modality]
            num_parts = modality_num_parts[modality]
            mm_data_modality = [d for d in mm_data_all if d.modality == modality]
            cum_num_items = 0
            cum_idx = 0
            for idx, assigned_num in enumerate(assigned_nums):
                if assigned_num == 0:
                    continue
                part_idx = part_idx_offset + cum_idx
                part_req_id = create_part_req_id(self.recv_req.rid, part_idx)
                encode_requests.append(
                    {
                        "encoder_idx": idx,
                        "mm_items": [
                            d.url
                            for d in mm_data_modality[
                                cum_num_items : cum_num_items + assigned_num
                            ]
                        ],
                        "num_parts": total_num_parts,
                        "part_idx": part_idx,
                        "req_id": part_req_id,
                        "modality": modality.name,
                        "prefill_host": self.host_name,
                        "embedding_port": self.embedding_port,
                    }
                )
                cum_idx += 1
                cum_num_items += assigned_num
            part_idx_offset += num_parts

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=envs.SGLANG_ENCODER_HTTP_TIMEOUT.get())
        ) as session:
            # Phase 1: POST /encode to all encoder shards in parallel.
            tasks = [
                session.post(
                    f"{self.encoder_urls[r['encoder_idx']]}/encode",
                    json=r,
                )
                for r in encode_requests
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            if not await self._check_encoder_responses(responses, "/encode"):
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
                        # Either the request exceeds pool capacity outright, or
                        # the wait timed out. Both are fatal for this request
                        # — fall through to error handling.
                        self.status = WaitingImageRequestStatus.FAIL
                        self.error_msg = (
                            f"MooncakeEmbeddingPool could not allocate "
                            f"{total_bytes // (1024 * 1024)}MB (oversize or "
                            f"timeout). Raise SGLANG_EMBEDDING_POOL_SIZE_MB."
                        )
                        self.recv_socket.close()
                        return
                    pool_view, buffer_address, slot_id = alloc_result
                    self.embeddings_buffer = pool_view
                    self._buffer_from_pool = True
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
                    self.embeddings_buffer = gpu_buffer
                    buffer_address = gpu_buffer.data_ptr()
                    self._buffer_from_pool = False
                    logger.info(
                        f"Per-request registered Mooncake GPU landing buffer "
                        f"(pool disabled): rid={self.rid}, size={total_bytes}, "
                        f"addr={buffer_address}"
                    )
            else:
                self.embeddings_buffer = None
                buffer_address = 0

            # Phase 2 cont: POST /send with RDMA info.
            offset = 0
            send_tasks = []
            for idx in range(total_num_parts):
                rj = response_sorted[idx]
                encoder_idx = rj.pop("encoder_idx", None)
                rj.update(
                    {
                        "session_id": self.embeddings_engine.session_id,
                        "buffer_address": offset + buffer_address,
                    }
                )
                send_tasks.append(
                    session.post(
                        f"{self.encoder_urls[encoder_idx]}/send",
                        json=rj,
                    )
                )
                offset += embedding_sizes[idx]

            # Phase 3: Wait for RDMA transfers to complete
            send_responses = await asyncio.gather(*send_tasks, return_exceptions=True)
            if not await self._check_encoder_responses(
                send_responses, "/send", on_error=self._cleanup_gpu_buffer
            ):
                return
            logger.info(f"RDMA transfers completed for rid={self.rid}")

    async def _check_encoder_responses(self, responses, endpoint: str, on_error=None):
        """Validate gathered HTTP responses from the encoder.

        Marks the request as FAIL and closes the recv socket on the first error,
        invoking ``on_error`` (e.g. GPU buffer cleanup) before closing.
        Returns True if all responses succeeded.
        """
        for i, resp in enumerate(responses):
            msg = None
            if isinstance(resp, asyncio.TimeoutError):
                timeout_val = envs.SGLANG_ENCODER_HTTP_TIMEOUT.get()
                logger.error(
                    f"Encoder {endpoint} timeout ({timeout_val}s) for rid={self.rid} "
                    f"(request {i})"
                )
                msg = f"Encoder {endpoint} timeout ({timeout_val}s)"
            elif isinstance(resp, Exception):
                logger.error(
                    f"Encoder {endpoint} failed for rid={self.rid} (request {i}): {resp}",
                    exc_info=resp,
                )
                msg = str(resp)
            elif resp.status != 200:
                try:
                    err = await resp.json()
                    msg = err.get("message", "Unknown error")
                except Exception:
                    msg = await resp.text()
                logger.error(f"Encoder {endpoint} returned error {resp.status}: {msg}")

            if msg is not None:
                self.status = WaitingImageRequestStatus.FAIL
                self.error_msg = msg
                if on_error is not None:
                    on_error()
                self.recv_socket.close()
                return False
        return True

    def _try_recv_mm_data(self):
        """Extract embedding from GPU buffer after RDMA transfer."""
        if self.status != WaitingImageRequestStatus.PENDING:
            return
        while self.recv_embedding_data is None or not self.recv_embedding_data.ready:
            try:
                parts = self.recv_socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
            except zmq.Again:
                return

            recv_obj: EmbeddingData = safe_pickle_loads(parts[0])
            if getattr(recv_obj, "error_msg", None) is not None:
                logger.warning(f"Received error for {self.rid}: {recv_obj.error_msg}")
                self.error_msg = recv_obj.error_msg
                self.error_code = recv_obj.error_code
                self.status = WaitingImageRequestStatus.FAIL
                self._cleanup_gpu_buffer()
                self.recv_socket.close()
                return

            # Extract original req_id
            part_req_id = recv_obj.req_id
            original_req_id = extract_original_req_id(part_req_id)
            if original_req_id != self.recv_req.rid:
                logger.warning(
                    f"Dropping stale embedding data: expected rid={self.recv_req.rid}, "
                    f"got rid={recv_obj.req_id} (likely from ZMQ port reuse)"
                )
                continue
            recv_obj.req_id = original_req_id

            # Embedding was written directly into pre-registered GPU buffer by encode server
            # (Mooncake GPU-direct transfer); no ZMQ payload in this message.
            # recv_obj.embedding stays None until we extract from GPU buffer below
            if self.recv_embedding_data is None:
                self.recv_embedding_data = MultiModalEmbeddingData.from_embedding_data(
                    recv_obj
                )
            else:
                self.recv_embedding_data.add(recv_obj)

        # Zero-copy: build per-modality views directly from the pre-registered
        # GPU buffer. Skips the per-part split + torch.cat round-trip — both
        # the extra GPU allocation and the D2D copy — so mm_item.precomputed_
        # embeddings ends up referencing the pool buffer. Slot lifetime is
        # bound to mm_inputs GC via weakref.finalize below.
        if self.embeddings_buffer is not None:
            recv_embedding = _view_pool_buffer_by_modality(
                self.embeddings_buffer, self.recv_embedding_data, self.dtype
            )
        else:
            recv_embedding = self.recv_embedding_data.get_embedding(is_concat=True)
        mm_inputs = self.mm_processor.get_mm_data(
            self.recv_req.input_text,
            recv_embedding,
            **self.recv_embedding_data.get_mm_extra_meta(),
        )
        # Bind slot release to mm_inputs GC
        if self._buffer_from_pool and mm_inputs is not None:
            weakref.finalize(mm_inputs, self.embedding_pool.release, self._pool_slot_id)
            for item in getattr(mm_inputs, "mm_items", []) or []:
                try:
                    setattr(item, "_keep_device_embedding", True)
                except Exception:
                    pass
            # Detach so _cleanup_gpu_buffer no-ops; finalize now owns release.
            self._pool_slot_id = None
            self.embeddings_buffer = None
            self._buffer_from_pool = False
        self.recv_req.mm_inputs = mm_inputs
        self.recv_req.input_ids = array("q", mm_inputs.input_ids)
        self.status = WaitingImageRequestStatus.SUCCESS
        self._cleanup_gpu_buffer()
        self.recv_socket.close()

    def _cleanup_gpu_buffer(self):
        """Deregister and release the GPU buffer."""
        if self.embeddings_buffer is not None:
            # Pool-backed views share the pre-registered backing tensor; just
            # release the slot back to the pool so a queued alloc can proceed.
            if self._buffer_from_pool:
                if self._pool_slot_id is not None and self.embedding_pool is not None:
                    self.embedding_pool.release(self._pool_slot_id)
                    self._pool_slot_id = None
                self.embeddings_buffer = None
                return
            try:
                self.embeddings_engine.deregister(self.embeddings_buffer.data_ptr())
            except Exception:
                logger.exception("Failed to deregister GPU buffer for rid=%s", self.rid)
            self.embeddings_buffer = None


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


class MooncakeEmbeddingPool:
    """Persistent GPU buffer pool registered once with the Mooncake engine.

    Allocator: first-fit on a free-segment list with 256-byte alignment.
    `alloc()` blocks on a Condition when the pool is full and resumes once
    a peer `release()`s a slot. Each successful alloc returns a slot_id
    that must be passed back to release() when the consumer is done with
    the buffer (after RDMA write completes and the data has been read).
    """

    _ALIGN = 256

    def __init__(self, engine, gpu_id: int, size_bytes: int):
        self.engine = engine
        self.gpu_id = gpu_id
        self.size_bytes = size_bytes
        self.buffer = torch.empty(
            size_bytes, dtype=torch.uint8, device=f"cuda:{gpu_id}"
        )
        self.base = self.buffer.data_ptr()
        self.engine.register(self.base, self.buffer.nbytes)
        self._segments_free: List[Tuple[int, int]] = [(0, size_bytes)]
        self._inflight: Dict[int, Tuple[int, int]] = {}
        self._next_slot_id = 0
        self._total_inflight = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        logger.info(
            f"MooncakeEmbeddingPool registered: gpu={gpu_id}, "
            f"size={size_bytes // (1024 * 1024)}MB, base=0x{self.base:x}"
        )

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
                f"MooncakeEmbeddingPool: requested {nbytes // (1024 * 1024)}MB "
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
                        f"MooncakeEmbeddingPool full: "
                        f"{inflight_mb}/{cap_mb}MB in-flight across "
                        f"{len(self._inflight)} requests; queueing a "
                        f"{nbytes // (1024 * 1024)}MB request. Raise "
                        f"SGLANG_EMBEDDING_POOL_SIZE_MB if this is frequent."
                    )
                    warned = True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.error(
                        f"MooncakeEmbeddingPool alloc timed out after "
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


def _slice_embedding_buffer(raw_buffer, embedding_data, dtype):
    """Slice a flat GPU buffer into per-part embedding tensors in-place."""
    elem_size = torch.tensor([], dtype=dtype).element_size()
    byte_offset = 0
    for i in range(embedding_data.num_parts):
        shape = embedding_data.embedding_shape_list[i]
        if shape is None:
            continue
        part_bytes = shape[0] * shape[1] * elem_size
        embedding_data.embedding_list[i] = (
            raw_buffer[byte_offset : byte_offset + part_bytes]
            .view(dtype)
            .reshape(shape)
        )
        byte_offset += part_bytes


def _view_pool_buffer_by_modality(raw_buffer, embedding_data, dtype):
    """Zero-copy view of raw_buffer as {modality: [total_tokens, hidden]}.

    Replaces _slice_embedding_buffer + get_embedding(is_concat=True): parts of
    the same modality are contiguous in raw_buffer (encoder writes them
    modality-outer in _send_encode_and_rdma_request), so we can reshape the
    byte range directly — no per-part split, no torch.cat copy.

    Caller must keep raw_buffer's storage alive while the returned views are
    in use. The pool path binds slot release to mm_inputs GC via finalize.
    """
    elem_size = torch.tensor([], dtype=dtype).element_size()
    # mod -> [byte_start, byte_end, total_tokens, hidden]
    mod_info: Dict[Modality, List[int]] = {}
    off = 0
    for i in range(embedding_data.num_parts):
        shape = embedding_data.embedding_shape_list[i]
        if shape is None:
            continue
        nbytes = shape[0] * shape[1] * elem_size
        mod = embedding_data.modality_list[i]
        info = mod_info.get(mod)
        if info is None:
            mod_info[mod] = [off, off + nbytes, shape[0], shape[1]]
        else:
            assert (
                info[3] == shape[1]
            ), f"hidden_dim mismatch in modality {mod}: {info[3]} vs {shape[1]}"
            assert info[1] == off, f"non-contiguous parts in modality {mod}"
            info[1] = off + nbytes
            info[2] += shape[0]
        off += nbytes
    return {
        mod: raw_buffer[s:e].view(dtype).reshape(tokens, hidden)
        for mod, (s, e, tokens, hidden) in mod_info.items()
    }


def _determine_tensor_transport_mode(server_args):
    is_cross_node = server_args.dist_init_addr

    if is_cross_node:
        # Fallback to default CPU transport for multi-node
        return "default"
    else:
        return "cuda_ipc"


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
        self.encoder_transfer_backend = server_args.encoder_transfer_backend
        # When ``encode_urls`` is shared with an :class:`EncoderBootstrapServer`
        # (tokenizer manager process), it grows / shrinks in place as encoders
        # register or unregister; the receiver always sees the current set.
        # When None (e.g. in a scheduler subprocess that has no in-process
        # bootstrap), fall back to a snapshot of the static --encoder-urls.
        self.encode_urls: List[str] = (
            encode_urls if encode_urls is not None else list(server_args.encoder_urls)
        )
        self.recv_timeout = envs.SGLANG_ENCODER_RECV_TIMEOUT.get()
        self.host = get_local_ip_auto(server_args.host)
        self.pp_rank = pp_rank
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size
        self.tp_group = tp_group
        self.nnodes = server_args.nnodes
        self.hostname = get_local_ip_auto()
        self.waiting_list: List[WaitingImageRequest] = []
        self.scheduler = scheduler
        self.wait_timeout = envs.SGLANG_ENCODER_RECV_TIMEOUT.get()

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
                        server_args.disaggregation_ib_device
                        or server_args.mooncake_ib_device
                    ),
                )
            self.embeddings_buffer = dict()
            self.embedding_pool = None
            pool_mb = envs.SGLANG_EMBEDDING_POOL_SIZE_MB.get()
            if pool_mb and pool_mb > 0 and scheduler is not None:
                gpu_id = getattr(scheduler, "gpu_id", 0)
                try:
                    self.embedding_pool = MooncakeEmbeddingPool(
                        self.embeddings_engine, gpu_id, pool_mb * 1024 * 1024
                    )
                except Exception:
                    logger.exception(
                        "Failed to allocate MooncakeEmbeddingPool, "
                        "falling back to per-request register"
                    )
                    self.embedding_pool = None
            if hf_config is not None:
                self._init_mm_processor(server_args, hf_config)
        elif self.encoder_transfer_backend == "zmq_to_scheduler":
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
        transport_mode = _determine_tensor_transport_mode(server_args)
        import_processors("sglang.srt.multimodal.processors")

        extra_kwargs = {}
        if getattr(server_args, "tokenizer_backend", None) is not None:
            extra_kwargs["tokenizer_backend"] = server_args.tokenizer_backend

        _processor = None
        try:
            _processor = get_processor(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=not server_args.disable_fast_image_processor,
                **extra_kwargs,
            )
        except ValueError as e:
            error_message = str(e)
            if "does not have a slow version" in error_message:
                logger.info(
                    f"Processor {server_args.tokenizer_path} does not have a slow version. Automatically use fast version"
                )
                _processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    use_fast=True,
                    **extra_kwargs,
                )
            else:
                raise e

        enable_adaptive_dispatch_to_encoder = (
            server_args.enable_adaptive_dispatch_to_encoder
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
            modalities = [m.modality for m in mm_data]
            logger.info(
                f"[{req_id}] Sending encode request to E, "
                f"modalities={modalities}, num_items={len(mm_data)}"
            )
            send_time = time.monotonic()
            asyncio.create_task(
                self.encode(
                    req_id,
                    mm_data,
                    embedding_port,
                    "encode",
                    "send",
                    encode_urls=encode_urls,
                )
            )
            result = await asyncio.wait_for(
                self._recv_mm_data(req_id, recv_socket, mm_processor, prompt),
                timeout=self.recv_timeout,
            )
            elapsed = time.monotonic() - send_time
            logger.info(f"[{req_id}] Received embedding from E in {elapsed:.3f}s")
            return result
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - send_time
            logger.warning(f"[{req_id}] Embedding recv timeout after {elapsed:.3f}s")
            if req_id is not None:
                self._cleanup_mooncake_buffer(req_id)
            return None

    def _cleanup_mooncake_buffer(self, req_id):
        if self.encoder_transfer_backend != "mooncake":
            return
        if not hasattr(self, "embeddings_buffer"):
            return
        embeddings = self.embeddings_buffer.pop(req_id, None)
        if embeddings is None:
            return
        try:
            self.embeddings_engine.deregister(embeddings.data_ptr())
        except Exception:
            logger.exception(
                "mooncake: failed to deregister buffer for req_id=%s", req_id
            )

    async def _recv_mm_data(self, req_id, recv_socket, mm_processor, prompt):
        if req_id is None:
            return None

        recv_embedding = None

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
                    self._cleanup_mooncake_buffer(req_id)
                    return None
                logger.debug("recv_obj=%s", recv_obj)
                # Extract original req_id from part_req_id
                part_req_id = recv_obj.req_id
                original_req_id = extract_original_req_id(part_req_id)
                # Update recv_obj.req_id to original for aggregation
                recv_obj.req_id = original_req_id
                if self.encoder_transfer_backend == "zmq_to_tokenizer":
                    if len(parts) < 2:
                        logger.error(
                            "zmq_to_tokenizer expected 2-part message, got %d parts",
                            len(parts),
                        )
                        return None
                    buffer = (
                        parts[1].buffer if hasattr(parts[1], "buffer") else parts[1]
                    )
                    # Clone so we don't depend on ZMQ buffer after next recv.
                    recv_obj.embedding = (
                        torch.frombuffer(buffer, dtype=recv_obj.dtype)
                        .reshape(recv_obj.shape)
                        .clone()
                    )
                if recv_embedding_data is None:
                    recv_embedding_data = MultiModalEmbeddingData.from_embedding_data(
                        recv_obj, model_type=self.model_type
                    )
                else:
                    recv_embedding_data.add(recv_obj)

            if self.encoder_transfer_backend == "mooncake":
                if req_id not in self.embeddings_buffer:
                    logger.error(
                        "mooncake: embeddings_buffer missing req_id=%s", req_id
                    )
                    return None
                raw_buffer = self.embeddings_buffer.pop(req_id)
                self.embeddings_engine.deregister(raw_buffer.data_ptr())
                _slice_embedding_buffer(raw_buffer, recv_embedding_data, self.dtype)

            recv_embedding = recv_embedding_data.get_embedding(is_concat=True)

            mm_inputs = mm_processor.get_mm_data(
                prompt,
                recv_embedding,
                **recv_embedding_data.get_mm_extra_meta(),
            )
            return mm_inputs
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

            # For mooncake, No tokenizer-side thread.
            # Save mm_data (extracted URL list) onto obj so the scheduler-side
            # WaitingImageRDMARequest can use it.  TokenizedGenerateReqInput does
            # NOT carry image_data, so re-reading recv_req.image_data at scheduler
            # time would always return None.
            if self.encoder_transfer_backend == "mooncake":
                obj.mm_data_mooncake = mm_data
                return

            encode_thread = threading.Thread(
                target=self._run_encode_in_thread,
                args=(
                    obj.rid,
                    mm_data,
                    "encode",
                    num_items_assigned,
                    None,
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

    # For zmq_to_scheduler
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
                    **extra_kwargs,
                )
                waiting_req.send_encode_request()
                self.waiting_list.append(waiting_req)
            else:
                new_recv_reqs.append(recv_req)

        if len(self.waiting_list) == 0:
            return new_recv_reqs, []

        current_time = time.time()
        local_status = []
        for waiting_req in self.waiting_list:
            waiting_req._try_recv_mm_data()
            if current_time - waiting_req.start_time > self.wait_timeout:
                waiting_req.status = WaitingImageRequestStatus.TIMEOUT
                waiting_req._cleanup_gpu_buffer()
                waiting_req.recv_socket.close()
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
            if status_value == WaitingImageRequestStatus.SUCCESS:
                new_recv_reqs.append(waiting_req.recv_req)
            elif status_value == WaitingImageRequestStatus.FAIL:
                logger.error(
                    f"Waiting request {waiting_req.rid} failed: {waiting_req.error_msg} {waiting_req.error_code = }"
                )
                abort_reqs.append(
                    (
                        self.create_req(waiting_req.recv_req),
                        waiting_req.error_msg,
                        waiting_req.error_code,
                    )
                )
            elif status_value == WaitingImageRequestStatus.TIMEOUT:
                logger.error(
                    f"Timed out waiting for image embeddings for request {waiting_req.rid}"
                )
                abort_reqs.append(
                    (
                        self.create_req(waiting_req.recv_req),
                        f"Timeout waiting for image embedding after {self.wait_timeout}s",
                        HTTPStatus.REQUEST_TIMEOUT,
                    )
                )
            else:  # status_value == WaitingImageRequestStatus.PENDING
                new_waiting.append(waiting_req)

        self.waiting_list = new_waiting
        return new_recv_reqs, abort_reqs

    def _run_encode_in_thread(
        self,
        req_id,
        mm_data,
        endpoint_encode,
        num_items_assigned,
        embedding_port,
        encode_urls=None,
        time_stats_json=None,
    ):
        try:
            asyncio.run(
                self.encode(
                    req_id=req_id,
                    mm_data=mm_data,
                    embedding_port=embedding_port,
                    endpoint_encode=endpoint_encode,
                    endpoint_send=None,
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
            http_worker_ipc=recv_req.http_worker_ipc,
            dllm_config=self.scheduler.dllm_config,
        )
        req.tokenizer = self.scheduler.tokenizer
        return req

    async def allocate_embedding_buffer(self, req_id, total_bytes):
        logger.info(
            f"Pre-allocating GPU buffer for mooncake RDMA: "
            f"req_id={req_id}, size={total_bytes} bytes"
        )
        gpu_id = getattr(self.scheduler, "gpu_id", 0)
        embeddings = torch.empty(total_bytes, dtype=torch.uint8, device=gpu_id)
        self.embeddings_engine.register(
            embeddings.data_ptr(),
            embeddings.nbytes,
        )
        self.embeddings_buffer[req_id] = embeddings
        return embeddings.data_ptr()

    def _assign_items_by_modality(
        self, mm_data, encoder_num, random_shuffle=True
    ) -> Dict:
        """
        Assign multimodal items across encoders by modality with cross-modality load balancing.

        Args:
            mm_data: List of multimodal data items, each with a modality
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
        modalities = list(dict.fromkeys(mm_item.modality for mm_item in mm_data))
        # Use OrderedDict to explicitly maintain modality order
        num_items_assigned = OrderedDict()
        current_offset = 0

        for modality in modalities:
            mm_data_modality = [
                mm_item for mm_item in mm_data if mm_item.modality == modality
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

    def _extract_url_data(self, request_obj) -> List[MooncakeMMUrlItem]:
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
        for attr, modality in [
            ("image_data", Modality.IMAGE),
            ("video_data", Modality.VIDEO),
            ("audio_data", Modality.AUDIO),
        ]:
            mm_items = getattr(request_obj, attr, None)
            if mm_items:
                mm_items = flatten_mm_items(mm_items)
                for mm_item in mm_items:
                    mm_data.append(
                        MooncakeMMUrlItem(
                            url=to_raw_url(mm_item),
                            modality=modality,
                        )
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
            gpu_id = getattr(self.scheduler, "gpu_id", 0)
            return self._process_waiting_requests(
                recv_reqs,
                WaitingImageRDMARequest,
                embeddings_engine=self.embeddings_engine,
                dtype=self.dtype,
                gpu_id=gpu_id,
                embedding_pool=self.embedding_pool,
            )
        return self._process_waiting_requests(recv_reqs, WaitingImageRequest)

    async def _check_encoder_responses(self, responses, encode_requests, req_id):
        """Validate gathered HTTP responses. Returns True if all OK."""
        for i, response in enumerate(responses):
            if isinstance(response, asyncio.TimeoutError):
                timeout_val = envs.SGLANG_ENCODER_HTTP_TIMEOUT.get()
                encoder_label = encode_requests[i].get(
                    "encoder_url", f"idx={encode_requests[i].get('encoder_idx')}"
                )
                logger.error(
                    f"Encoder HTTP request timeout ({timeout_val}s) for req_id={req_id} "
                    f"(request {i}), "
                    f"encoder={encoder_label}"
                )
                return False
            elif isinstance(response, Exception):
                logger.error(
                    f"Encoder HTTP request failed for req_id={req_id} (request {i}): {response}",
                    exc_info=response,
                )
                return False
        for response in responses:
            if response.status != 200:
                try:
                    err_data = await response.json()
                    msg = err_data.get("message", "Unknown encoder error")
                except Exception:
                    msg = await response.text()
                logger.error(f"Encoder returned error {response.status}: {msg}")
                return False
        return True

    async def encode(
        self,
        req_id,
        mm_data,
        embedding_port,
        endpoint_encode,
        endpoint_send,
        num_items_assigned=None,
        encode_urls=None,
        time_stats_json=None,
    ):
        if len(mm_data) == 0:
            return

        effective_urls = encode_urls if encode_urls is not None else self.encode_urls

        # get unique modalities with order preserved
        modalities = [mm_item.modality for mm_item in mm_data]
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
                mm_item for mm_item in mm_data if mm_item.modality == modality
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
                            mm_item.url
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

            if not await self._check_encoder_responses(
                responses, encode_requests, req_id
            ):
                return
            response_json_list_unsort = [
                await response.json() for response in responses
            ]

            # zmq backend: return is None
            if None in response_json_list_unsort:
                return

            # mooncake backend: send bootstrap info

            embedding_size_list_sort, response_json_list_sort, total_embedding_bytes = (
                _sort_responses_and_compute_total_bytes(
                    response_json_list_unsort, total_num_parts
                )
            )
            offset = 0
            metadata_tasks = []
            buffer_address = await self.allocate_embedding_buffer(
                req_id,
                total_embedding_bytes,
            )
            for idx in range(len(tasks)):
                response_json = response_json_list_sort[idx]
                buffer_address_adjust = offset + buffer_address
                response_json.update(
                    {
                        "session_id": self.embeddings_engine.session_id,
                        "buffer_address": buffer_address_adjust,
                    }
                )
                metadata_tasks.append(
                    session.post(
                        f"{effective_urls[response_json['encoder_idx']]}/{endpoint_send}",
                        json=response_json,
                    )
                )
                offset += embedding_size_list_sort[idx]
            await asyncio.gather(*metadata_tasks)


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

    def build_and_send_encode_request(self, image_urls, rid):
        encode_req = GenerateReqInput(
            image_data=[ImageData(url=url) for url in image_urls],
            rid=rid,
        )
        self.send_encode_request(encode_req)
        return encode_req

    # For zmq_to_scheduler and mooncake
    def process_waiting_requests(self, recv_reqs):
        return self._process_waiting_requests(recv_reqs, WaitingImageRequestGrpc)

    async def encode(
        self,
        req_id,
        mm_data,
        embedding_port,
        endpoint_encode,
        endpoint_send,
        num_items_assigned=None,
        encode_urls=None,
    ):
        if not mm_data:
            return

        effective_urls = encode_urls if encode_urls is not None else self.encode_urls

        # gRPC currently only supports image; flatten typed multimodal items to simple lists
        if mm_data and isinstance(mm_data[0], MooncakeMMUrlItem):
            non_image = [
                item.modality for item in mm_data if item.modality != Modality.IMAGE
            ]
            if non_image:
                raise NotImplementedError(
                    f"gRPC encode only supports IMAGE modality, got: {non_image}"
                )
            img_data = [item.url for item in mm_data]
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
        grpc_responses = await asyncio.gather(*grpc_tasks)
        response_json_unsorted = []
        for encode_request, response in zip(encode_requests, grpc_responses):
            if self.encoder_transfer_backend == "zmq_to_scheduler":
                response_json_unsorted.append(None)
                continue
            response_json_unsorted.append(
                {
                    "req_id": encode_request["req_id"],
                    "prefill_host": encode_request["prefill_host"],
                    "embedding_port": encode_request["embedding_port"],
                    "encoder_idx": encode_request["encoder_idx"],
                    "part_idx": encode_request["part_idx"],
                    "embedding_size": response.embedding_size,
                    "embedding_len": response.embedding_len,
                    "embedding_dim": response.embedding_dim,
                }
            )

        if None in response_json_unsorted:
            return

        embedding_size_by_part, response_json_sorted, total_embedding_bytes = (
            _sort_responses_and_compute_total_bytes(response_json_unsorted, num_parts)
        )
        offset = 0
        buffer_address = await self.allocate_embedding_buffer(
            req_id,
            total_embedding_bytes,
        )
        grpc_metadata_tasks = []
        for response_json in response_json_sorted:
            response_json.update(
                {
                    "session_id": self.embeddings_engine.session_id,
                    "buffer_address": offset + buffer_address,
                }
            )
            grpc_metadata_tasks.append(
                asyncio.to_thread(
                    _grpc_send_request,
                    _grpc_target(effective_urls[response_json["encoder_idx"]]),
                    response_json,
                )
            )
            offset += embedding_size_by_part[response_json["part_idx"]]

        if grpc_metadata_tasks:
            await asyncio.gather(*grpc_metadata_tasks)


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

    _validate_transport_mode(transport_mode, encode_urls or server_args.encoder_urls)
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
