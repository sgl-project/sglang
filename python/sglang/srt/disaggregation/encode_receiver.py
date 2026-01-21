import asyncio
import logging
import pickle
import random
import threading
import uuid
from typing import List, Optional

import aiohttp
import torch
import zmq
import zmq.asyncio
from transformers import PretrainedConfig

from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_local_ip_auto, get_zmq_socket_on_host
from sglang.srt.utils.hf_transformers_utils import get_processor

logger = logging.getLogger(__name__)


class EmbeddingData:
    def __init__(self, req_id, num_parts, part_idx, image_grid_dim, embedding=None):
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
        return f"EmbeddingData(req_id={self.req_id}, num_parts={self.num_parts}, part_idx={self.part_idx})"

    def copy_without_embedding(self):
        new_data = EmbeddingData(
            req_id=self.req_id,
            num_parts=self.num_parts,
            part_idx=self.part_idx,
            image_grid_dim=self.image_grid_dim,
        )
        new_data.send_time = self.send_time
        new_data.dtype = self.dtype
        new_data.shape = self.shape
        return new_data


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
        self.ready = False

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
        if self.ready:
            return
        while self.recv_embedding_data is None or not self.recv_embedding_data.ready:
            try:
                parts = self.recv_socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
            except zmq.Again:
                # No data available yet, wait a bit and retry
                return

            recv_obj: EmbeddingData = pickle.loads(parts[0])
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
        self.ready = True
        self.recv_socket.close()


def _determine_tensor_transport_mode(server_args):
    is_cross_node = server_args.dist_init_addr

    if is_cross_node:
        # Fallback to default CPU transport for multi-node
        return "default"
    else:
        return "cuda_ipc"


class MMReceiver:

    def __init__(
        self,
        server_args: ServerArgs,
        dtype: Optional[torch.dtype] = None,
        hf_config: Optional[PretrainedConfig] = None,
        pp_rank: Optional[int] = None,
        tp_rank: Optional[int] = None,
        tp_group: Optional[GroupCoordinator] = None,
    ):
        self.context = zmq.asyncio.Context(20)
        self.encoder_transfer_backend = server_args.encoder_transfer_backend
        self.encode_urls = server_args.encoder_urls
        self.encode_idx = list(range(len(self.encode_urls)))
        self.host = server_args.host
        if self.encoder_transfer_backend == "mooncake":
            self.dtype = dtype
            self.embeddings_engine = MooncakeTransferEngine(
                hostname=get_local_ip_auto(),
                gpu_id=None,
                ib_device=server_args.disaggregation_ib_device,
            )
            self.embeddings_buffer = dict()
        elif self.encoder_transfer_backend == "zmq_to_scheduler":
            self.pp_rank = pp_rank
            self.tp_rank = tp_rank
            self.tp_size = server_args.tp_size
            self.tp_group = tp_group
            self.nnodes = server_args.nnodes
            self.hostname = get_local_ip_auto()
            self.waiting_list: List[WaitingImageRequest] = []
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
                    hf_config, server_args, _processor, transport_mode
                )

    # For zmq_to_scheduler
    def process_waiting_requests(self, recv_reqs):
        new_recv_reqs = []
        for recv_req in recv_reqs:
            if (
                isinstance(recv_req, TokenizedGenerateReqInput)
                and recv_req.need_wait_for_image is True
            ):
                waiting_req = WaitingImageRequest(
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
            return new_recv_reqs

        local_status = []
        for waiting_req in self.waiting_list:
            waiting_req._try_recv_mm_data()
            local_status.append(waiting_req.ready)

        local_status = torch.tensor(local_status, device="cpu", dtype=torch.int32)

        torch.distributed.all_reduce(
            local_status,
            op=torch.distributed.ReduceOp.MIN,
            group=self.tp_group.cpu_group,
        )

        new_waiting = []
        for i, waiting_req in enumerate(self.waiting_list):
            if local_status[i].item():
                new_recv_reqs.append(waiting_req.recv_req)
            else:
                new_waiting.append(waiting_req)

        self.waiting_list = new_waiting
        return new_recv_reqs

    # For zmq_to_scheduler
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

    # For mooncake
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

    # For zmq_to_scheduler
    def send_encode_request(self, obj):
        if type(obj.image_data) != list:
            image_urls = [obj.image_data.url]
        else:
            image_urls = [img.url for img in obj.image_data]
        if obj.rid is None:
            obj.rid = uuid.uuid4().hex
        if image_urls and len(image_urls) > 0:
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

    # For zmq_to_tokenizer and mooncake
    async def recv_mm_data(self, img_data, mm_processor, prompt):
        try:
            if len(self.encode_urls) == 0:
                return None
            req_id = uuid.uuid4().hex
            embedding_port, recv_socket = get_zmq_socket_on_host(self.context, zmq.PULL)
            if type(img_data) != list:
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
            if hasattr(self, "embeddings_buffer") and req_id in self.embeddings_buffer:
                del self.embeddings_buffer[req_id]
            return None

    # For zmq_to_tokenizer and mooncake
    async def _recv_mm_data(self, req_id, recv_socket, mm_processor, prompt):
        # Bypass MMReceiver
        if req_id is None:
            return None

        recv_embedding = None

        recv_embedding_data: EmbeddingData = None

        while recv_embedding_data is None or not recv_embedding_data.ready:
            parts = await recv_socket.recv_multipart(copy=False)

            recv_obj: EmbeddingData = pickle.loads(parts[0])
            logger.info(f"{recv_obj = }")
            if self.encoder_transfer_backend == "zmq_to_tokenizer":
                buffer = parts[1].buffer if hasattr(parts[1], "buffer") else parts[1]
                recv_obj.embedding = torch.frombuffer(
                    buffer, dtype=recv_obj.dtype
                ).reshape(recv_obj.shape)
            if recv_embedding_data is None:
                recv_obj.embedding_list[recv_obj.part_idx] = recv_obj.embedding
                recv_embedding_data = recv_obj
            else:
                recv_embedding_data.add(recv_obj)

        if self.encoder_transfer_backend == "mooncake":
            recv_embedding = self.embeddings_buffer[req_id]
            del self.embeddings_buffer[req_id]
            self.embeddings_engine.deregister(recv_embedding.data_ptr())
        elif self.encoder_transfer_backend == "zmq_to_tokenizer":
            recv_embedding = recv_embedding_data.get_embedding(is_concat=True)

        recv_socket.close()

        img_grid_thw = recv_embedding_data.get_img_grid()

        mm_inputs = mm_processor.get_mm_data(prompt, recv_embedding, img_grid_thw)
        return mm_inputs
