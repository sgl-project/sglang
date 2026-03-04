import asyncio
import logging
import pickle
import random
import threading
import time
import uuid
from abc import ABC, abstractmethod
from enum import IntEnum
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional

import aiohttp
import torch
import zmq
import zmq.asyncio
from transformers import PretrainedConfig

from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    get_mooncake_transfer_engine,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import ImageData, get_local_ip_auto, get_zmq_socket_on_host
from sglang.srt.utils.hf_transformers_utils import get_processor

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


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
        image_grid_dim,
        embedding=None,
        error_msg=None,
        error_code=None,
    ):
        self.req_id = req_id
        self.num_parts = num_parts
        self.part_idx = part_idx
        self.image_grid_dim = image_grid_dim
        self.embedding = embedding
        self.send_time = None
        self.dtype = embedding.dtype if embedding is not None else None
        self.shape = list(embedding.shape) if embedding is not None else None
        # aggregated data
        self.ready_list = [i == self.part_idx for i in range(self.num_parts)]
        self.embedding_list = [
            embedding if i == self.part_idx else None for i in range(self.num_parts)
        ]
        self.image_grid_dim_list = [
            self.image_grid_dim if i == self.part_idx else None
            for i in range(self.num_parts)
        ]
        self.error_msg = error_msg
        self.error_code = error_code

    def add(self, embedding_data):
        assert self.req_id == embedding_data.req_id
        assert not self.ready_list[embedding_data.part_idx]
        self.ready_list[embedding_data.part_idx] = True
        self.image_grid_dim_list[embedding_data.part_idx] = (
            embedding_data.image_grid_dim
        )
        self.embedding_list[embedding_data.part_idx] = embedding_data.embedding

    def get_embedding(self, is_concat=False):
        if is_concat:
            return torch.concat([embedding.cuda() for embedding in self.embedding_list])
        else:
            return self.embedding_list

    def get_img_grid(self):
        return torch.concatenate(self.image_grid_dim_list)

    @property
    def ready(self):
        return sum(self.ready_list) == self.num_parts

    def __repr__(self):
        return f"EmbeddingData(req_id={self.req_id}, num_parts={self.num_parts}, part_idx={self.part_idx}) error_msg={self.error_msg}"

    def copy_without_embedding(self):
        new_data = EmbeddingData(
            req_id=self.req_id,
            num_parts=self.num_parts,
            part_idx=self.part_idx,
            image_grid_dim=self.image_grid_dim,
            error_msg=self.error_msg,
            error_code=self.error_code,
        )
        new_data.send_time = self.send_time
        new_data.dtype = self.dtype
        new_data.shape = self.shape
        return new_data


class WaitingImageRequestStatus(IntEnum):
    FAIL = -1
    PENDING = 0
    SUCCESS = 1
    TIMEOUT = -2


# For zmq_to_scheduler
class WaitingImageRequest:
    def __init__(
        self,
        rid: str,
        recv_req: TokenizedGenerateReqInput,
        mm_processor,
        encoder_urls,
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
        self.host_name = host_name
        self.receive_count = receive_count
        self.num_items_assigned = recv_req.num_items_assigned
        self.embedding_port, self.recv_socket = get_zmq_socket_on_host(
            zmq.Context(), zmq.PULL
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
                timeout=aiohttp.ClientTimeout(total=1800)
            ) as session:
                tasks = []
                logger.info(f"{self.num_items_assigned = } ")
                for idx, assigned_num in enumerate(self.num_items_assigned):
                    if assigned_num == 0:
                        continue
                    encoder_url = self.encoder_urls[idx]
                    target_url = f"{encoder_url}/scheduler_receive_url"
                    payload = {
                        "req_id": req_id,
                        "receive_count": receive_count,
                        "receive_url": f"{host_name}:{embedding_port}",
                    }

                    logger.info(f"Preparing to send  to {target_url}")

                    task = _send_single_request(session, target_url, payload)
                    tasks.append(task)

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

    def _try_recv_mm_data(self):
        if self.status != WaitingImageRequestStatus.PENDING:
            return
        while self.recv_embedding_data is None or not self.recv_embedding_data.ready:
            try:
                parts = self.recv_socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
            except zmq.Again:
                # No data available yet, wait a bit and retry
                return
            recv_obj: EmbeddingData = pickle.loads(parts[0])
            if getattr(recv_obj, "error_msg", None) is not None:
                logger.warning(
                    f"Received error signal from encoder for {self.rid}: {recv_obj.error_msg} {recv_obj.error_code = }"
                )
                self.error_msg = recv_obj.error_msg
                self.error_code = recv_obj.error_code
                self.status = WaitingImageRequestStatus.FAIL
                self.recv_socket.close()
                return

            buffer = parts[1].buffer if hasattr(parts[1], "buffer") else parts[1]
            recv_obj.embedding = torch.frombuffer(buffer, dtype=recv_obj.dtype).reshape(
                recv_obj.shape
            )
            recv_obj.embedding_list[recv_obj.part_idx] = recv_obj.embedding
            if self.recv_embedding_data is None:
                self.recv_embedding_data = recv_obj
            else:
                self.recv_embedding_data.add(recv_obj)

        recv_embedding = self.recv_embedding_data.get_embedding(is_concat=True)
        img_grid_thw = self.recv_embedding_data.get_img_grid()

        mm_inputs = self.mm_processor.get_mm_data(
            self.recv_req.input_text, recv_embedding, img_grid_thw
        )
        self.recv_req.mm_inputs = mm_inputs
        self.recv_req.input_ids = mm_inputs["input_ids"]
        self.status = WaitingImageRequestStatus.SUCCESS
        self.recv_socket.close()


class WaitingImageRequestGrpc(WaitingImageRequest):
    def send_encode_request(self):
        async def send_embedding_port(req_id, receive_count, host_name, embedding_port):
            tasks = []
            logger.info(f"{self.num_items_assigned = } ")
            for idx, assigned_num in enumerate(self.num_items_assigned):
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
    ):
        self.context = zmq.asyncio.Context(20)
        self.encoder_transfer_backend = server_args.encoder_transfer_backend
        self.encode_urls = server_args.encoder_urls
        self.encode_idx = list(range(len(self.encode_urls)))
        self.host = get_local_ip_auto(server_args.host)
        if self.encoder_transfer_backend == "mooncake":
            self.dtype = dtype
            self.embeddings_engine = get_mooncake_transfer_engine()
            self.embeddings_buffer = dict()
        elif self.encoder_transfer_backend == "zmq_to_scheduler":
            self.pp_rank = pp_rank
            self.tp_rank = tp_rank
            self.tp_size = server_args.tp_size
            self.tp_group = tp_group
            self.nnodes = server_args.nnodes
            self.hostname = get_local_ip_auto()
            self.waiting_list: List[WaitingImageRequest] = []
            self.scheduler = scheduler
            self.wait_timeout = envs.SGLANG_ENCODER_RECV_TIMEOUT.get()
            if hf_config is not None:
                transport_mode = _determine_tensor_transport_mode(server_args)
                import_processors("sglang.srt.multimodal.processors")
                _processor = None
                try:
                    _processor = get_processor(
                        server_args.tokenizer_path,
                        tokenizer_mode=server_args.tokenizer_mode,
                        trust_remote_code=server_args.trust_remote_code,
                        revision=server_args.revision,
                        use_fast=not server_args.disable_fast_image_processor,
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
                        )
                    else:
                        raise e
                self.mm_processor = get_mm_processor(
                    hf_config,
                    server_args,
                    _processor,
                    transport_mode,
                    skip_mm_pool=True,
                )

    @abstractmethod
    def process_waiting_requests(self, recv_reqs):
        pass

    async def recv_mm_data(self, img_data, mm_processor, prompt):
        req_id = None
        try:
            if len(self.encode_urls) == 0:
                return None
            req_id = uuid.uuid4().hex
            embedding_port, recv_socket = get_zmq_socket_on_host(self.context, zmq.PULL)
            if not isinstance(img_data, list):
                img_data = [img_data.url]
            else:
                img_data = [img.url for img in img_data]
            asyncio.create_task(
                self.encode(req_id, img_data, embedding_port, "encode", "send")
            )
            return await asyncio.wait_for(
                self._recv_mm_data(req_id, recv_socket, mm_processor, prompt),
                timeout=20,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Embedding recv timeout for request {req_id}")
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

        recv_embedding_data: EmbeddingData = None

        try:
            while recv_embedding_data is None or not recv_embedding_data.ready:
                parts = await recv_socket.recv_multipart(copy=False)
                if not parts:
                    continue
                recv_obj: EmbeddingData = pickle.loads(parts[0])
                if getattr(recv_obj, "error_msg", None) is not None:
                    logger.warning(
                        f"Encoder error for req_id={req_id}: {recv_obj.error_msg} "
                        f"error_code={getattr(recv_obj, 'error_code', None)}"
                    )
                    self._cleanup_mooncake_buffer(req_id)
                    return None
                logger.debug("recv_obj=%s", recv_obj)
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
                    recv_obj.embedding_list[recv_obj.part_idx] = recv_obj.embedding
                    recv_embedding_data = recv_obj
                else:
                    recv_embedding_data.add(recv_obj)

            if self.encoder_transfer_backend == "mooncake":
                if req_id not in self.embeddings_buffer:
                    logger.error(
                        "mooncake: embeddings_buffer missing req_id=%s", req_id
                    )
                    return None
                recv_embedding = self.embeddings_buffer[req_id]
                del self.embeddings_buffer[req_id]
                self.embeddings_engine.deregister(recv_embedding.data_ptr())
            elif self.encoder_transfer_backend == "zmq_to_tokenizer":
                recv_embedding = recv_embedding_data.get_embedding(is_concat=True)

            img_grid_thw = recv_embedding_data.get_img_grid()
            mm_inputs = mm_processor.get_mm_data(prompt, recv_embedding, img_grid_thw)
            return mm_inputs
        finally:
            recv_socket.close()

    def send_encode_request(self, obj):
        self._send_encode_request(obj)

    def _send_encode_request(self, obj):
        if obj.image_data is None:
            image_urls = []
        elif not isinstance(obj.image_data, list):
            image_urls = [obj.image_data.url]
        else:
            image_urls = [img.url for img in obj.image_data]
        if obj.rid is None:
            obj.rid = uuid.uuid4().hex
        if image_urls and self.encode_urls:
            logger.info(f"Processing {len(image_urls)} images for request {obj.rid}")
            obj.need_wait_for_image = True

            encode_idx = list(range(len(self.encode_urls)))
            random.shuffle(encode_idx)
            obj.num_items_assigned = [
                (idx + len(image_urls)) // len(self.encode_urls) for idx in encode_idx
            ]
            encode_thread = threading.Thread(
                target=self._run_encode_in_thread,
                args=(
                    obj.rid,
                    image_urls,
                    "encode",
                    obj.num_items_assigned,
                    None,
                ),
                daemon=True,
            )
            encode_thread.start()

    # For zmq_to_scheduler
    def _process_waiting_requests(self, recv_reqs, waiting_cls):
        new_recv_reqs = []
        for recv_req in recv_reqs:
            if (
                isinstance(recv_req, TokenizedGenerateReqInput)
                and recv_req.need_wait_for_image is True
            ):
                waiting_req = waiting_cls(
                    rid=recv_req.rid,
                    recv_req=recv_req,
                    mm_processor=self.mm_processor,
                    encoder_urls=self.encode_urls,
                    host_name=self.hostname,
                    receive_count=self.tp_size,
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
        self, req_id, img_data, endpoint_encode, num_items_assigned, embedding_port
    ):
        try:
            asyncio.run(
                self.encode(
                    req_id=req_id,
                    img_data=img_data,
                    embedding_port=embedding_port,
                    endpoint_encode=endpoint_encode,
                    endpoint_send=None,
                    num_items_assigned=num_items_assigned,
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
                if self.scheduler.enable_metrics
                else None
            ),
            http_worker_ipc=recv_req.http_worker_ipc,
            dllm_config=self.scheduler.dllm_config,
        )
        req.tokenizer = self.scheduler.tokenizer
        return req

    async def allocate_embedding_buffer(self, req_id, embedding_length, embedding_dim):
        embeddings = torch.zeros(
            (embedding_length, embedding_dim),
            dtype=self.dtype,
        )
        self.embeddings_engine.register(
            embeddings.data_ptr(),
            embeddings.nbytes,
        )
        self.embeddings_buffer[req_id] = embeddings
        return embeddings.data_ptr()


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
    ):
        super().__init__(
            server_args,
            dtype=dtype,
            hf_config=hf_config,
            pp_rank=pp_rank,
            tp_rank=tp_rank,
            tp_group=tp_group,
            scheduler=scheduler,
        )

    # For zmq_to_scheduler
    def process_waiting_requests(self, recv_reqs):
        return self._process_waiting_requests(recv_reqs, WaitingImageRequest)

    async def encode(
        self,
        req_id,
        img_data,
        embedding_port,
        endpoint_encode,
        endpoint_send,
        num_items_assigned=None,
    ):
        if len(img_data) == 0:
            return

        # Split mm_items
        encode_requests = []
        if num_items_assigned is None:
            random.shuffle(self.encode_idx)
            num_items_assigned = [
                (idx + len(img_data)) // len(self.encode_urls)
                for idx in self.encode_idx
            ]
        num_parts = sum(1 for x in num_items_assigned if x != 0)
        cum_num_items = 0
        cum_idx = 0
        for idx, assigned_num in enumerate(num_items_assigned):
            if assigned_num == 0:
                continue
            encode_requests.append(
                {
                    "encoder_idx": idx,
                    "mm_items": img_data[cum_num_items : cum_num_items + assigned_num],
                    "num_parts": num_parts,
                    "part_idx": cum_idx,
                    "req_id": req_id,
                    "prefill_host": self.host,
                    "embedding_port": embedding_port,
                }
            )
            cum_idx += 1
            cum_num_items += assigned_num

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=1800
            )  # Add timeout for request reliability
        ) as session:
            # Send encode requests

            tasks = [
                session.post(
                    f"{self.encode_urls[encode_request['encoder_idx']]}/{endpoint_encode}",
                    json=encode_request,
                )
                for encode_request in encode_requests
            ]

            responses = await asyncio.gather(*tasks)
            for response in responses:
                if response.status != 200:
                    try:
                        err_data = await response.json()
                        msg = err_data.get("message", "Unknown encoder error")
                    except:
                        msg = await response.text()

                    logger.error(f"Encoder returned error {response.status}: {msg}")
                    return
            response_json_list_unsort = [
                await response.json() for response in responses
            ]

            # zmq backend: return is None
            if None in response_json_list_unsort:
                return

            # mooncake backend: send bootstrap info

            embedding_size_list_sort = [None for _ in range(num_parts)]
            embedding_length_tot = 0
            response_json_list_sort = [None for _ in range(num_parts)]
            for response_json in response_json_list_unsort:
                idx = response_json["part_idx"]
                embedding_size_list_sort[idx] = response_json["embedding_size"]
                embedding_length_tot += response_json["embedding_len"]
                response_json_list_sort[idx] = response_json

            offset = 0
            metadata_tasks = []
            buffer_address = await self.allocate_embedding_buffer(
                req_id,
                embedding_length_tot,
                response_json_list_sort[0]["embedding_dim"],
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
                        f"{self.encode_urls[response_json['encoder_idx']]}/{endpoint_send}",
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
    ):
        super().__init__(
            server_args,
            dtype=dtype,
            hf_config=hf_config,
            pp_rank=pp_rank,
            tp_rank=tp_rank,
            tp_group=tp_group,
            scheduler=scheduler,
        )

    def build_and_send_encode_request(self, image_urls, rid):
        encode_req = GenerateReqInput(
            image_data=[ImageData(url=url) for url in image_urls],
            rid=rid,
        )
        self.send_encode_request(encode_req)
        return encode_req

    # For zmq_to_scheduler
    def process_waiting_requests(self, recv_reqs):
        return self._process_waiting_requests(recv_reqs, WaitingImageRequestGrpc)

    async def encode(
        self,
        req_id,
        img_data,
        embedding_port,
        endpoint_encode,
        endpoint_send,
        num_items_assigned=None,
    ):
        if not img_data:
            return

        encode_requests = []
        if num_items_assigned is None:
            random.shuffle(self.encode_idx)
            num_items_assigned = [
                (idx + len(img_data)) // len(self.encode_urls)
                for idx in self.encode_idx
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
                _grpc_target(self.encode_urls[encode_request["encoder_idx"]]),
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

        embedding_size_by_part = [None for _ in range(num_parts)]
        embedding_length_tot = 0
        response_json_sorted = [None for _ in range(num_parts)]
        for response_json in response_json_unsorted:
            idx = response_json["part_idx"]
            embedding_size_by_part[idx] = response_json["embedding_size"]
            embedding_length_tot += response_json["embedding_len"]
            response_json_sorted[idx] = response_json

        offset = 0
        buffer_address = await self.allocate_embedding_buffer(
            req_id,
            embedding_length_tot,
            response_json_sorted[0]["embedding_dim"],
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
                    _grpc_target(self.encode_urls[response_json["encoder_idx"]]),
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
):
    if transport_mode is None:
        transport_mode = envs.SGLANG_ENCODER_MM_RECEIVER_MODE.get()
        logger.debug(f"MMReceiver transport_mode from env: {transport_mode}")

    _validate_transport_mode(transport_mode, server_args.encoder_urls)
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
    )
