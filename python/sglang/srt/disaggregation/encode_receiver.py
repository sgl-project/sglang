import asyncio
import logging
from typing import Dict

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


class MMReceiver:

    def __init__(self, mm_transfer_backend, disaggregation_ib_device, dtype):
        context = zmq.asyncio.Context(2)
        self.embedding_port = get_free_port()
        self.recv_from_encoder = get_zmq_socket(
            context, zmq.PULL, f"tcp://*:{self.embedding_port}", True
        )
        self.received_data: Dict[int, EmbeddingData] = dict()
        self.embeddings_lock = asyncio.Lock()
        self.mm_transfer_backend = mm_transfer_backend
        self.dtype = dtype
        if self.mm_transfer_backend == "mooncake":
            self.embeddings_engine = MooncakeTransferEngine(
                hostname=get_local_ip_auto(),
                gpu_id=None,
                ib_device=disaggregation_ib_device,
            )
            self.embeddings_buffer = dict()

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

    async def recv_mm_data(self, req_id, mm_processor, prompt):
        try:
            return await asyncio.wait_for(
                self._recv_mm_data(req_id, mm_processor, prompt), timeout=10
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
