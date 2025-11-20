import asyncio
import logging
import random
from typing import Dict

import aiohttp
import torch
import zmq
import zmq.asyncio

from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.utils import get_free_port, get_local_ip_auto, get_zmq_socket

logger = logging.getLogger(__name__)


class EmbeddingData:
    def __init__(self, req_id, num_parts, part_idx, image_grid_dim, embedding=None):
        self.req_id = req_id
        self.num_parts = num_parts
        self.part_idx = part_idx
        self.image_grid_dim = image_grid_dim
        self.embedding = embedding

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

    def get_embedding(self):
        return torch.concatenate(self.embedding_list)

    def get_img_grid(self):
        return torch.concatenate(self.image_grid_dim_list)

    @property
    def ready(self):
        return sum(self.ready_list) == self.num_parts

    def __repr__(self):
        return f"EmbeddingData(req_id={self.req_id}, num_parts={self.num_parts}, part_idx={self.part_idx})"


def _generate_id():
    req_id = random.randint(0, 2**63 - 1)
    return req_id


class MMReceiver:

    def __init__(
        self, host, encode_urls, mm_transfer_backend, disaggregation_ib_device, dtype
    ):
        context = zmq.asyncio.Context(2)
        self.embedding_port = get_free_port()
        self.recv_from_encoder = get_zmq_socket(
            context, zmq.PULL, f"tcp://*:{self.embedding_port}", True
        )
        self.received_data: Dict[int, EmbeddingData] = dict()
        self.embeddings_lock = asyncio.Lock()
        self.mm_transfer_backend = mm_transfer_backend
        self.dtype = dtype
        self.encode_urls = encode_urls
        self.encode_idx = list(range(len(self.encode_urls)))
        self.host = host
        if self.mm_transfer_backend == "mooncake":
            self.embeddings_engine = MooncakeTransferEngine(
                hostname=get_local_ip_auto(),
                gpu_id=None,
                ib_device=disaggregation_ib_device,
            )
            self.embeddings_buffer = dict()

    async def encode(self, req_id, img_data, endpoint_encode, endpoint_send):
        if len(img_data) == 0:
            return

        # Split mm_items
        encode_requests = []
        random.shuffle(self.encode_idx)
        num_items_assigned = [
            (idx + len(img_data)) // len(self.encode_urls) for idx in self.encode_idx
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
                    "embedding_port": self.embedding_port,
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

    async def handle_embedding(self):
        recv_obj = await self.recv_from_encoder.recv_pyobj()
        if recv_obj.req_id not in self.received_data:
            self.received_data[recv_obj.req_id] = recv_obj
        else:
            self.received_data[recv_obj.req_id].add(recv_obj)

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

    async def recv_mm_data(self, img_data, mm_processor, prompt):
        try:
            if len(self.encode_urls) == 0:
                return None
            req_id = _generate_id()
            if type(img_data) != list:
                img_data = [img_data.url]
            else:
                img_data = [img.url for img in img_data]
            asyncio.create_task(self.encode(req_id, img_data, "encode", "send"))
            return await asyncio.wait_for(
                self._recv_mm_data(req_id, mm_processor, prompt), timeout=20
            )
        except asyncio.TimeoutError:
            logger.warning(f"Embedding recv timeout for request {req_id}")
            if req_id in self.received_data:
                del self.received_data[req_id]
            if hasattr(self, "embeddings_buffer") and req_id in self.embeddings_buffer:
                del self.embeddings_buffer[req_id]
            return None

    async def _recv_mm_data(self, req_id, mm_processor, prompt):
        # Bypass MMReceiver
        if req_id is None:
            return None

        # E Disaggregation
        recv_embedding = None
        img_grid_thw = None

        # Use async lock to avoid race condition
        async with self.embeddings_lock:
            while (
                req_id not in self.received_data or not self.received_data[req_id].ready
            ):
                await self.handle_embedding()

            recv_embedding_data = self.received_data[req_id]
            if self.mm_transfer_backend == "mooncake":
                recv_embedding = self.embeddings_buffer[req_id]
                self.embeddings_engine.deregister(recv_embedding.data_ptr())
            elif self.mm_transfer_backend == "zmq":
                recv_embedding = recv_embedding_data.get_embedding()
            img_grid_thw = recv_embedding_data.get_img_grid()
            del self.received_data[req_id]
            if self.mm_transfer_backend == "mooncake":
                del self.embeddings_buffer[req_id]

        mm_inputs = mm_processor.get_mm_data(prompt, recv_embedding, img_grid_thw)
        return mm_inputs
