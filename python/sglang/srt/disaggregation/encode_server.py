import asyncio
import ctypes
import logging
import multiprocessing as mp
import os
import pickle
import time
import traceback
from typing import Dict, List, Optional, Set, Tuple

import aiohttp
import numpy as np
import torch
import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse, Response
from transformers import AutoImageProcessor
from transformers.image_utils import load_images

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.disaggregation.encode_receiver import EmbeddingData
from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.srt.model_loader import get_model
from sglang.srt.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import get_local_ip_auto, get_zmq_socket, random_uuid

logger = logging.getLogger(__name__)

rid_lock = asyncio.Lock()
rid_to_receive_endpoint: Dict[str, List[str]] = dict()
rid_to_receive_count: Dict[str, int] = dict()


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


_image_grid_attrs = ["image_grid_thw", "image_grid_hws"]


def _get_image_grid_dim(images_input):
    for attr in _image_grid_attrs:
        if attr in images_input:
            return images_input[attr]
    raise ValueError(
        f"Image grid dim ({_image_grid_attrs}) not found in {images_input}"
    )


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

        self.device = server_args.device
        self.gpu_id = server_args.base_gpu_id + rank

        self.device_config = DeviceConfig(
            device=self.device,
            gpu_id=self.gpu_id,
        )

        torch.get_device_module(self.device).set_device(self.gpu_id)

        init_distributed_environment(
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

        self.context = zmq.asyncio.Context(2)

        embedding_cache_size = int(os.environ.get("SGLANG_VLM_CACHE_SIZE_MB", "4096"))
        self.mm_cache = MultiModalStaticCache(embedding_cache_size * 1024 * 1024)
        self.mm_cache_lock = asyncio.Lock()

        if schedule_path is not None:
            self.schedule_socket = get_zmq_socket(
                self.context, zmq.PULL, schedule_path, True
            )

        if self.rank == 0:
            logger.info(
                f"Using transfer backend: {self.server_args.encoder_transfer_backend}"
            )

            if self.server_args.encoder_transfer_backend == "mooncake":
                self.local_ip = get_local_ip_auto()

                self.engine = MooncakeTransferEngine(
                    hostname=self.local_ip,
                    gpu_id=None,
                    ib_device=server_args.disaggregation_ib_device,
                )

            self.embedding_to_send = dict()

        logger.info(f"rank {rank} init finish ")

    async def _encode(self, mm_items) -> torch.Tensor:
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

        # support mm_cache
        mm_embedding = None
        mm_hash = None

        start_time = time.perf_counter()
        if self.server_args.enable_prefix_mm_cache:
            mm_item.set_pad_value()
            mm_hash = MultiModalStaticCache.combine_hashes([mm_item.hash])
            async with self.mm_cache_lock:
                mm_cache = self.mm_cache.get([mm_item.hash])
                if mm_cache is not None:
                    mm_embedding = mm_cache

        if mm_embedding is None:
            with torch.inference_mode():
                mm_embedding: torch.Tensor = self.model.get_image_feature([mm_item])
                mm_embedding = mm_embedding.cpu()
            if len(mm_embedding.shape) != 2:
                mm_embedding = mm_embedding.reshape(-1, mm_embedding.shape[-1])

        if self.server_args.enable_prefix_mm_cache:
            async with self.mm_cache_lock:
                self.mm_cache.set(mm_hash, mm_embedding)
        end_time = time.perf_counter()
        logger.info(
            f"Vit time : {(end_time - start_time)*1000:.2f} ms {mm_embedding.shape = }"
        )

        return _get_image_grid_dim(images_input), mm_embedding

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
            self.engine.register(embedding.data_ptr(), embedding.nbytes)
            self.engine.transfer_sync(
                session_id, embedding.data_ptr(), buffer_address, embedding.nbytes
            )
            self.engine.deregister(embedding.data_ptr())

            mm_data.embedding = None
            mm_data.embedding_list[mm_data.part_idx] = None

        # Send ack/data
        endpoint = (
            f"tcp://{url}"
            if url is not None
            else f"tcp://{prefill_host}:{embedding_port}"
        )
        logger.info(f"{endpoint = }")
        socket = get_zmq_socket(
            self.context,
            zmq.PUSH,
            endpoint,
            False,
        )

        if self.server_args.encoder_transfer_backend == "mooncake":
            socket.send_multipart([pickle.dumps(mm_data)])
        else:
            new_mm_data = mm_data.copy_without_embedding()
            embedding_tensor = TensorWrapper(mm_data.embedding)
            socket.send_multipart(
                [pickle.dumps(new_mm_data), embedding_tensor.__buffer__()]
            )

    async def encode(self, mm_items, req_id, num_parts, part_idx):
        start_time = time.time()
        image_grid_dim, mm_embedding = await self._encode(mm_items)
        end_time = time.time()
        logger.info(f"🕛 encode cost = {(end_time - start_time) * 1000:.2f}ms")
        if self.rank == 0:
            mm_data = EmbeddingData(
                req_id,
                num_parts,
                part_idx,
                image_grid_dim,
                mm_embedding,
            )
            self.embedding_to_send[mm_data.req_id] = mm_data
        return mm_embedding.nbytes, mm_embedding.shape[0], mm_embedding.shape[1]

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
        timeout = 60.0

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

                if asyncio.get_running_loop().time() - start_time > timeout:
                    logger.error(
                        f"Timeout waiting for all endpoints for {req_id}. Initiated {len(sent_urls)}/{expected_count}"
                    )
                    break

                await asyncio.sleep(0.001)

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


app = FastAPI()
encoder: Optional[MMEncoder] = None
send_sockets: List[zmq.Socket] = []


async def run_encoder(
    server_args: ServerArgs, schedule_path, dist_init_method, rank: int
):
    encoder = MMEncoder(server_args, schedule_path, dist_init_method, rank)
    while True:
        request = await encoder.schedule_socket.recv_pyobj()
        await encoder.encode(
            mm_items=request["mm_items"],
            req_id=request["req_id"],
            num_parts=request["num_parts"],
            part_idx=request["part_idx"],
        )


def launch_encoder(server_args, schedule_path, dist_init_method, rank):
    try:
        asyncio.run(run_encoder(server_args, schedule_path, dist_init_method, rank))
    except KeyboardInterrupt:
        logger.info(f"Exit rank {rank}")
    except Exception:
        traceback.print_exc()


def launch_server(server_args: ServerArgs):
    global encoder
    ctx = mp.get_context("spawn")
    zmq_ctx = zmq.Context(10)
    ipc_path_prefix = random_uuid()
    port_args = PortArgs.init_new(server_args)
    if server_args.dist_init_addr:
        dist_init_method = f"tcp://{server_args.dist_init_addr}"
    else:
        dist_init_method = f"tcp://127.0.0.1:{port_args.nccl_port}"
    for rank in range(1, server_args.tp_size):
        schedule_path = f"ipc:///tmp/{ipc_path_prefix}_schedule_{rank}"
        send_sockets.append(
            get_zmq_socket(zmq_ctx, zmq.PUSH, schedule_path, bind=False)
        )
        ctx.Process(
            target=launch_encoder,
            args=(server_args, schedule_path, dist_init_method, rank),
            daemon=True,
        ).start()
    encoder = MMEncoder(server_args, dist_init_method=dist_init_method)
    uvicorn.run(app, host=server_args.host, port=server_args.port)


@app.post("/encode")
async def handle_encode_request(request: dict):
    # broadcast request
    request.update({"enter_time": time.time()})
    for socket in send_sockets:
        socket.send_pyobj(request)

    nbytes, embedding_len, embedding_dim = await encoder.encode(
        mm_items=request["mm_items"],
        req_id=request["req_id"],
        num_parts=request["num_parts"],
        part_idx=request["part_idx"],
    )
    if encoder.server_args.encoder_transfer_backend == "mooncake":
        del request["mm_items"]
        request.update(
            {
                "embedding_size": nbytes,
                "embedding_len": embedding_len,
                "embedding_dim": embedding_dim,
            }
        )
        return ORJSONResponse(content=request)
    elif encoder.server_args.encoder_transfer_backend == "zmq_to_scheduler":
        logger.info(f"{request['embedding_port'] = }")
        if request["embedding_port"] is None:
            await encoder.send_with_url(
                req_id=request["req_id"],
            )
        else:
            assert type(request["embedding_port"]) == list
            tasks = []
            for embedding_port in request["embedding_port"]:
                tasks.append(
                    encoder.send(
                        req_id=request["req_id"],
                        prefill_host=request["prefill_host"],
                        embedding_port=embedding_port,
                    )
                )
            await asyncio.gather(*tasks)
            encoder.embedding_to_send.pop(request["req_id"], None)
        return ORJSONResponse(content=None)
    elif encoder.server_args.encoder_transfer_backend == "zmq_to_tokenizer":
        await encoder.send(
            req_id=request["req_id"],
            prefill_host=request["prefill_host"],
            embedding_port=request["embedding_port"],
        )
        encoder.embedding_to_send.pop(request["req_id"], None)
        return ORJSONResponse(content=None)


@app.post("/send")
async def handle_send_request(request: dict):
    # mooncake backend
    await encoder.send(
        req_id=request["req_id"],
        prefill_host=request["prefill_host"],
        embedding_port=request["embedding_port"],
        session_id=request["session_id"],
        buffer_address=request["buffer_address"],
    )
    encoder.embedding_to_send.pop(request["req_id"], None)
    return ORJSONResponse(content=None)


@app.post("/scheduler_receive_url")
async def handle_scheduler_receive_url_request(request: dict):
    rid = request["req_id"]
    async with rid_lock:
        global rid_to_receive_endpoint
        if rid not in rid_to_receive_endpoint:
            rid_to_receive_endpoint[rid] = set()
            rid_to_receive_count[rid] = request["receive_count"]
        assert rid_to_receive_count[rid] == request["receive_count"]
        rid_to_receive_endpoint[rid].add(request["receive_url"])


@app.get("/health")
@app.get("/health_generate")
async def health_generate():
    """
    Health check endpoint for the encoder server.
    Returns 200 if the encoder is initialized and ready.
    """
    if encoder is None:
        return Response(status_code=503)
    return Response(status_code=200)
