import logging
from typing import Optional

import aiohttp
import numpy as np
import torch
import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from transformers import AutoImageProcessor
from transformers.image_utils import load_images

from python.sglang.srt.disaggregation.encode_receiver import EmbeddingData
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.model_loader import get_model
from sglang.srt.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import get_local_ip_auto, get_zmq_socket

logger = logging.getLogger(__name__)


def _convert(data):
    if type(data) == torch.Tensor:
        return data
    elif type(data) == np.ndarray:
        return torch.tensor(data)
    elif type(data) == list and type(data[0]) == np.ndarray:
        return torch.tensor(np.array(data))
    elif type(data) == list and type(data[0]) in [int, float]:
        return torch.tensor(data)
    else:
        return data


_image_grid_attrs = ["image_grid_thw", "image_grid_hws"]


def _get_image_grid_dim(images_input):
    for attr in _image_grid_attrs:
        if attr in images_input:
            return images_input[attr]
    raise ValueError(
        f"Image grid dim ({_image_grid_attrs}) not found in {images_input}"
    )


class ImageEncoder:
    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args
        set_global_server_args_for_scheduler(server_args)

        self.image_processor = AutoImageProcessor.from_pretrained(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            use_fast=True,
        )

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

        port_args = PortArgs.init_new(server_args)
        if server_args.dist_init_addr:
            dist_init_method = f"tcp://{server_args.dist_init_addr}"
        else:
            dist_init_method = f"tcp://127.0.0.1:{port_args.nccl_port}"

        init_distributed_environment(
            world_size=1, rank=0, distributed_init_method=dist_init_method
        )
        initialize_model_parallel()
        initialize_dp_attention(server_args, self.model_config)

        self.model = get_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=DeviceConfig(),
        )

        logger.info(f"Using transfer backend: {self.server_args.mm_transfer_backend}")

        if self.server_args.mm_transfer_backend == "mooncake":
            self.local_ip = get_local_ip_auto()

            self.engine = MooncakeTransferEngine(
                hostname=self.local_ip,
                gpu_id=None,
                ib_device=server_args.disaggregation_ib_device,
            )

        self.context = zmq.asyncio.Context(2)
        self.send_to_prefill_sockets = dict()

        self.embedding_to_send = dict()

    async def mm_encode(self, mm_items) -> torch.Tensor:
        images = load_images(mm_items)

        images_input = self.image_processor(images=images)
        feature = images_input["pixel_values"]
        mm_item = MultimodalDataItem.from_dict(
            {
                "modality": Modality.IMAGE,
                "feature": _convert(feature),
            }
        )
        for k, v in images_input.items():
            if k == "pixel_values":
                continue
            mm_item.set(k, _convert(v))
        mm_embedding = self.model.get_image_feature([mm_item])
        if len(mm_embedding.shape) != 2:
            mm_embedding = mm_embedding.reshape(-1, mm_embedding.shape[-1])
        return _get_image_grid_dim(images_input), mm_embedding

    async def mm_send(
        self,
        prefill_host,
        prefill_url,
        embedding: torch.Tensor,
        mm_data: EmbeddingData,
        session_id=None,
        peer_buffer_address=None,
    ):
        if self.server_args.mm_transfer_backend == "mooncake":
            self.engine.register(embedding.data_ptr(), embedding.nbytes)
            self.engine.transfer_sync(
                session_id, embedding.data_ptr(), peer_buffer_address, embedding.nbytes
            )
            self.engine.deregister(embedding.data_ptr())

            mm_data.embedding = None
            mm_data.embedding_list[mm_data.part_idx] = None

        # Send ack/data
        if prefill_url in self.send_to_prefill_sockets:
            socket = self.send_to_prefill_sockets[prefill_url]
        else:
            embedding_port = await self.get_embedding_port(prefill_url)
            socket = get_zmq_socket(
                self.context,
                zmq.PUSH,
                f"tcp://{prefill_host}:{embedding_port}",
                False,
            )
            self.send_to_prefill_sockets[prefill_url] = socket
        socket.send_pyobj(mm_data)

    @torch.inference_mode()
    async def encode(self, mm_items, req_id, num_parts, part_idx):
        image_grid_dim, mm_embedding = await self.mm_encode(mm_items)
        mm_data = EmbeddingData(
            req_id,
            num_parts,
            part_idx,
            image_grid_dim,
            mm_embedding,
        )
        self.embedding_to_send[mm_data.req_id] = mm_data
        return mm_embedding.nbytes, mm_embedding.shape[0], mm_embedding.shape[1]

    async def send(
        self, req_id, prefill_host, prefill_url, session_id=None, buffer_address=None
    ):
        mm_data: EmbeddingData = self.embedding_to_send[req_id]
        await self.mm_send(
            prefill_host,
            prefill_url,
            mm_data.embedding,
            mm_data,
            session_id,
            buffer_address,
        )
        del self.embedding_to_send[req_id]

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


app = FastAPI()
encoder: Optional[ImageEncoder] = None


def launch_server(server_args: ServerArgs):
    global encoder
    encoder = ImageEncoder(server_args)
    uvicorn.run(app, host=server_args.host, port=server_args.port)


@app.post("/encode")
async def handle_encode_request(request: dict):
    nbytes, embedding_len, embedding_dim = await encoder.encode(
        mm_items=request["mm_items"],
        req_id=request["req_id"],
        num_parts=request["num_parts"],
        part_idx=request["part_idx"],
    )
    if encoder.server_args.mm_transfer_backend == "mooncake":
        del request["mm_items"]
        request.update(
            {
                "embedding_size": nbytes,
                "embedding_len": embedding_len,
                "embedding_dim": embedding_dim,
            }
        )
        return ORJSONResponse(content=request)
    elif encoder.server_args.mm_transfer_backend == "zmq":
        await encoder.send(
            req_id=request["req_id"],
            prefill_host=request["bootstrap_host"],
            prefill_url=request["prefill_url"],
        )
        return ORJSONResponse(content=None)


@app.post("/send")
async def handle_send_request(request: dict):
    # mooncake backend
    await encoder.send(
        req_id=request["req_id"],
        prefill_host=request["bootstrap_host"],
        prefill_url=request["prefill_url"],
        session_id=request["session_id"],
        buffer_address=request["buffer_address"],
    )
    return ORJSONResponse(content=None)


@app.get("/health_check")
async def handle_send_request():
    return ORJSONResponse(content={"is_alive": True})
