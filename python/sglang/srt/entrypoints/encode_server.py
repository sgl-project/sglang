import asyncio
from typing import Optional

import torch
import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from transformers import AutoImageProcessor
from transformers.image_utils import load_images

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
from sglang.srt.multimodal.processors.qwen_vl import resize_image_async
from sglang.srt.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import get_local_ip_auto, get_zmq_socket


class EmbeddingData:
    def __init__(self, req_id, num_parts, part_idx, embedding_len):
        self.req_id = req_id
        self.num_parts = num_parts
        self.part_idx = part_idx
        self.embedding_len = embedding_len

        # aggregated data
        self.ready_list = [i == self.part_idx for i in range(self.num_parts)]
        self.embedding_len_tot = self.embedding_len

    def add(self, embedding_data):
        assert self.req_id == embedding_data.req_id
        assert not self.ready_list[embedding_data.part_idx]
        self.ready_list[embedding_data.part_idx] = True
        self.embedding_len_tot += embedding_data.embedding_len

    @property
    def ready(self):
        return sum(self.ready_list) == self.num_parts


class ImageEncoder:
    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args
        set_global_server_args_for_scheduler(server_args)

        self.image_processor = AutoImageProcessor.from_pretrained(
            server_args.model_path, trust_remote_code=server_args.trust_remote_code
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

        self.gpu_id = 0

        init_distributed_environment(
            world_size=1, rank=0, distributed_init_method=dist_init_method
        )
        initialize_model_parallel()
        initialize_dp_attention(server_args, self.model_config)

        self.model = get_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=DeviceConfig(device="cuda", gpu_id=self.gpu_id),
        )

        self.local_ip = get_local_ip_auto()

        self.engine = MooncakeTransferEngine(
            hostname=self.local_ip,
            gpu_id=self.gpu_id,
        )

        self.context = zmq.asyncio.Context(2)
        self.send_to_prefill_sockets = dict()

        self.embedding_to_send = dict()

    async def encode(self, mm_items) -> torch.Tensor:
        images = load_images(mm_items)

        # Qwen-specific: resize images
        resize_tasks = [resize_image_async(image) for image in images]
        images = await asyncio.gather(*resize_tasks)

        images_input = self.image_processor(images=images)
        mm_item = MultimodalDataItem.from_dict(
            {
                "modality": Modality.IMAGE,
                "feature": images_input["pixel_values"],
            }
        )
        mm_item.set("image_grid_thw", images_input["image_grid_thw"])
        mm_embedding = self.model.get_image_feature([mm_item])
        return mm_embedding

    def send(
        self,
        session_id,
        peer_buffer_address,
        embedding: torch.Tensor,
        meta_data: EmbeddingData,
        prefill_ip,
    ):
        self.engine.register(embedding.data_ptr(), embedding.nbytes)
        self.engine.transfer_sync(
            session_id, embedding.data_ptr(), peer_buffer_address, embedding.nbytes
        )
        self.engine.deregister(embedding.data_ptr())

        # Send ack
        if prefill_ip in self.send_to_prefill_sockets:
            socket = self.send_to_prefill_sockets[prefill_ip]
        else:
            socket = get_zmq_socket(
                self.context,
                zmq.PUSH,
                f"tcp://{prefill_ip}:{self.server_args.embedding_port}",
                False,
            )
            self.send_to_prefill_sockets[prefill_ip] = socket
        socket.send_pyobj(meta_data)

    @torch.inference_mode()
    async def step(self, request_data: dict):
        if "mm_items" in request_data:
            mm_embedding = await self.encode(request_data["mm_items"])
            meta_data = EmbeddingData(
                request_data["req_id"],
                request_data["num_parts"],
                request_data["part_idx"],
                mm_embedding.shape[0],
            )
            self.embedding_to_send[meta_data.req_id] = (mm_embedding, meta_data)
            del request_data["mm_items"]
            request_data.update(
                {
                    "embedding_size": mm_embedding.nbytes,
                    "embedding_len": mm_embedding.shape[0],
                    "embedding_dim": mm_embedding.shape[1],
                }
            )
            return request_data
        else:
            mm_embedding, meta_data = self.embedding_to_send[request_data["req_id"]]
            self.send(
                request_data["session_id"],
                request_data["buffer_address"],
                mm_embedding,
                meta_data,
                request_data["bootstrap_host"],
            )
            del self.embedding_to_send[request_data["req_id"]]
            return None


app = FastAPI()
encoder: Optional[ImageEncoder] = None


def launch_server(server_args: ServerArgs):
    global encoder
    encoder = ImageEncoder(server_args)
    uvicorn.run(app, host=server_args.host, port=server_args.port)


@app.post("/encode")
async def handle_encode_request(request_data: dict):
    ret_data = await encoder.step(request_data)
    return ORJSONResponse(content=ret_data)
