import asyncio
import itertools
import logging
import pickle
import random
import threading
import uuid
from enum import IntEnum
from typing import TYPE_CHECKING, Dict, List, Optional

import aiohttp
import torch
import zmq
import zmq.asyncio

from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.schedule_batch import Modality, Req
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_local_ip_auto, get_zmq_socket_on_host
from sglang.srt.utils.common import ImageData
from sglang.srt.utils.hf_transformers_utils import get_processor
from transformers import PretrainedConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class EmbeddingData:
    # Standard attributes that are default needed
    _STANDARD_ATTRS = {
        "req_id",
        "num_parts",
        "part_idx",
        "grid_dim",
        "modality",
        "embedding",
        "error_msg",
        "error_code",
        "send_time",
        "dtype",
        "shape",
    }

    def __init__(
        self,
        req_id,
        num_parts,
        part_idx,
        grid_dim,
        modality,
        embedding=None,
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
        self.shape = list(embedding.shape) if embedding is not None else None
        self.error_msg = error_msg
        self.error_code = error_code
        # Store additional metadata (e.g., video_timestamps for qwen3_vl)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_grid(self):
        """
        Get the grid dimension of the embedding, used for image/video/audio.
        """
        return self.grid_dim

    def get_embedding(self):
        return self.embedding

    def __repr__(self):
        return f"EmbeddingData(req_id={self.req_id}, num_parts={self.num_parts}, part_idx={self.part_idx}) error_msg={self.error_msg}"

    def copy_without_embedding(self):
        # collect additional kwargs attributes (e.g., video_timestamps, second_per_grid_ts)
        kwargs = {
            key: getattr(self, key)
            for key in dir(self)
            if not key.startswith("_")
            and key not in self._STANDARD_ATTRS
            and not callable(getattr(self, key, None))
        }
        new_data = EmbeddingData(
            req_id=self.req_id,
            num_parts=self.num_parts,
            part_idx=self.part_idx,
            grid_dim=self.grid_dim,
            modality=self.modality,
            error_msg=self.error_msg,
            error_code=self.error_code,
            **kwargs,
        )
        new_data.send_time = self.send_time
        new_data.dtype = self.dtype
        new_data.shape = self.shape
        return new_data


class MultiModalEmbeddingData(EmbeddingData):
    def __init__(
        self, part_idx, num_parts, req_id, grid_dim, modality, embedding, **kwargs
    ):
        super().__init__(
            req_id, num_parts, part_idx, grid_dim, modality, embedding, **kwargs
        )
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
        # additional videometa attributes
        self.video_timestamps = [None] * num_parts
        self.second_per_grid_ts = [None] * num_parts

        if modality == Modality.IMAGE:
            self.img_grid_thw[part_idx] = self.get_grid()
        elif modality == Modality.VIDEO:
            self.video_grid_thw[part_idx] = self.get_grid()
        elif modality == Modality.AUDIO:
            # For audio, grid_dim represents audio_feature_lens; flatten for variable-length
            self.audio_feature_lens[part_idx] = self.get_grid().flatten()

    @classmethod
    def from_embedding_data(cls, embedding_data: EmbeddingData):
        """Create MultiModalEmbeddingData from an EmbeddingData instance."""
        # Extract kwargs from embedding_data attributes (excluding standard attrs)
        kwargs = {
            key: getattr(embedding_data, key)
            for key in dir(embedding_data)
            if not key.startswith("_")
            and key not in embedding_data._STANDARD_ATTRS
            and not callable(getattr(embedding_data, key, None))
        }
        mm_data = cls(
            part_idx=embedding_data.part_idx,
            num_parts=embedding_data.num_parts,
            req_id=embedding_data.req_id,
            grid_dim=embedding_data.grid_dim,
            modality=embedding_data.modality,
            embedding=embedding_data.embedding,
            **kwargs,
        )
        # Set video_timestamps and second_per_grid_ts for the specific part_idx if available
        if embedding_data.modality == Modality.VIDEO:
            for attr_name in ["video_timestamps", "second_per_grid_ts"]:
                if attr_name in kwargs:
                    getattr(mm_data, attr_name)[embedding_data.part_idx] = kwargs[
                        attr_name
                    ]

        # Copy over additional attributes
        mm_data.send_time = embedding_data.send_time
        return mm_data

    def __repr__(self):
        return f"MultiModalEmbeddingData(req_id={self.req_id}, num_parts={self.num_parts}, part_idx={self.part_idx}, modality={self.modality})"

    def _get_mm_grid(self, modality):
        if modality == Modality.IMAGE:
            grid_dims = self.img_grid_thw
        elif modality == Modality.VIDEO:
            grid_dims = self.video_grid_thw
        elif modality == Modality.AUDIO:
            # flatten all parts into one 1D tensor
            flat = [g.flatten() for g in self.audio_feature_lens if g is not None]
            if flat:
                return torch.cat(flat, dim=0)
            else:
                return None
        else:
            grid_dims = []

        valid_grid_dims = []
        for grid_dim in grid_dims:
            if grid_dim is None:
                continue
            if grid_dim.dim() == 1:
                # TODO: check necessary
                valid_grid_dims.append(grid_dim.unsqueeze(0))
            else:
                valid_grid_dims.append(grid_dim)
        if len(valid_grid_dims) == 0:
            return None
        return torch.cat(valid_grid_dims, dim=0)

    def get_attr_list(self, attr_name, flatten=False):
        """
        Get the attribute of the embedding data, mainly for video metadata now.
        Args:
            attr_name: The name of the attribute to get.
            flatten: Whether to flatten the attribute.
        Returns:
            The attribute value, if flatten is True, return a list of all the attribute values.
        """
        _attr = getattr(self, attr_name)
        if _attr is None:
            return None
        valid_attr = []
        for attr in _attr:
            if attr is not None:
                valid_attr.append(attr)
        if len(valid_attr) == 0:
            return None
        if flatten:
            return list(itertools.chain(*valid_attr))
        return valid_attr

    def get_embedding(self, is_concat=False):
        if is_concat:
            return torch.concat([embedding.cuda() for embedding in self.embedding_list])
        else:
            return self.embedding_list

    @property
    def ready(self):
        return sum(self.ready_list) == self.num_parts

    def get_audio_feature_lens(self):
        return self._get_mm_grid(Modality.AUDIO)

    def get_img_grid(self):
        return self._get_mm_grid(Modality.IMAGE)

    def get_video_grid(self):
        return self._get_mm_grid(Modality.VIDEO)

    def add(self, embedding_data: EmbeddingData):
        assert self.req_id == embedding_data.req_id
        assert not self.ready_list[embedding_data.part_idx]
        self.ready_list[embedding_data.part_idx] = True
        self.modality_list[embedding_data.part_idx] = embedding_data.modality
        self.embedding_list[embedding_data.part_idx] = embedding_data.get_embedding()
        if embedding_data.modality == Modality.IMAGE:
            self.img_grid_thw[embedding_data.part_idx] = embedding_data.get_grid()
        elif embedding_data.modality == Modality.VIDEO:
            self.video_grid_thw[embedding_data.part_idx] = embedding_data.get_grid()
            for attr_name in ["video_timestamps", "second_per_grid_ts"]:
                if getattr(embedding_data, attr_name, None) is not None:
                    getattr(self, attr_name)[embedding_data.part_idx] = getattr(
                        embedding_data, attr_name
                    )
        elif embedding_data.modality == Modality.AUDIO:
            self.audio_feature_lens[embedding_data.part_idx] = (
                embedding_data.get_grid().flatten()
            )
        else:
            raise ValueError(f"Invalid modality: {embedding_data.modality}")


class WaitingImageRequestStatus(IntEnum):
    FAIL = -1
    PENDING = 0
    SUCCESS = 1


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

                for modality, assigned_nums in self.num_items_assigned.items():
                    for idx, assigned_num in enumerate(assigned_nums):
                        if assigned_num == 0:
                            continue
                        encoder_url = self.encoder_urls[idx]
                        target_url = f"{encoder_url}/scheduler_receive_url"
                        payload = {
                            "req_id": req_id,
                            "receive_count": receive_count,
                            "receive_url": f"{host_name}:{embedding_port}",
                            "modality": modality.name,
                        }
                        logger.info(f"Preparing to send to {target_url}")
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

            if self.recv_embedding_data is None:
                self.recv_embedding_data = MultiModalEmbeddingData.from_embedding_data(
                    recv_obj
                )
            else:
                self.recv_embedding_data.add(recv_obj)

        recv_embedding = self.recv_embedding_data.get_embedding(is_concat=True)
        img_grid_thw = self.recv_embedding_data.get_img_grid()
        video_grid_thw = self.recv_embedding_data.get_video_grid()
        audio_feature_lens = self.recv_embedding_data.get_audio_feature_lens()
        kwargs = {
            "img_grid_thw": img_grid_thw,
            "video_grid_thw": video_grid_thw,
            "audio_feature_lens": audio_feature_lens,
        }
        video_timestamps = self.recv_embedding_data.get_attr_list(
            "video_timestamps", flatten=True
        )
        second_per_grid_ts = self.recv_embedding_data.get_attr_list(
            "second_per_grid_ts", flatten=False
        )
        if video_timestamps is not None:
            kwargs["video_timestamps"] = video_timestamps
        if second_per_grid_ts is not None:
            kwargs["second_per_grid_ts"] = second_per_grid_ts

        mm_inputs = self.mm_processor.get_mm_data(
            self.recv_req.input_text, recv_embedding, **kwargs
        )
        self.recv_req.mm_inputs = mm_inputs
        self.recv_req.input_ids = mm_inputs["input_ids"]
        self.status = WaitingImageRequestStatus.SUCCESS
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
        scheduler: Optional["Scheduler"] = None,
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
            self.scheduler = scheduler
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

    def create_req(self, recv_req):
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
            data_parallel_rank=recv_req.data_parallel_rank,
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

    # For zmq_to_scheduler
    def process_waiting_requests(self, recv_reqs):
        new_recv_reqs = []
        for recv_req in recv_reqs:
            if (
                isinstance(recv_req, TokenizedGenerateReqInput)
                and recv_req.need_wait_for_mm_inputs is True
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
            return new_recv_reqs, []

        local_status = []
        for waiting_req in self.waiting_list:
            waiting_req._try_recv_mm_data()
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
            else:  # status_value == WaitingImageRequestStatus.PENDING
                new_waiting.append(waiting_req)

        self.waiting_list = new_waiting
        return new_recv_reqs, abort_reqs

    # For zmq_to_scheduler
    def _run_encode_in_thread(
        self, req_id, mm_data, endpoint_encode, num_items_assigned, embedding_port
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
                )
            )
        except Exception as e:
            logger.error(f"Encode failed for request {req_id}: {e}", exc_info=True)

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
        # Get unique modalities
        modalities = list(dict.fromkeys(mm_item.get("modality") for mm_item in mm_data))
        num_items_assigned = {}
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

    async def encode(
        self,
        req_id,
        mm_data,
        embedding_port,
        endpoint_encode,
        endpoint_send,
        num_items_assigned=None,
    ):
        if len(mm_data) == 0:
            return

        # get unique modalities with order preserved
        modalities = [mm_item.get("modality") for mm_item in mm_data]
        modalities = list(dict.fromkeys(modalities))
        encode_requests = []

        if num_items_assigned is None:
            num_items_assigned = self._assign_items_by_modality(
                mm_data, len(self.encode_urls)
            )

        # Calculate total num_parts across all modalities
        total_num_parts = 0
        modality_num_parts = {}
        for modality in modalities:
            num_items_assigned_modality = num_items_assigned.get(modality)
            num_parts = sum(1 for x in num_items_assigned_modality if x != 0)
            modality_num_parts[modality] = num_parts
            total_num_parts += num_parts

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
                encode_requests.append(
                    {
                        "encoder_idx": self.encode_idx[
                            idx
                        ],  # use shuffle-idx to load-balance
                        "mm_items": [
                            mm_item.get("url")
                            for mm_item in mm_data_modality[
                                cum_num_items : cum_num_items + assigned_num
                            ]
                        ],
                        "num_parts": total_num_parts,
                        "part_idx": part_idx_offset + cum_idx,
                        "req_id": req_id,
                        "modality": modality.name,  # convert enum to string for json serialization
                        "prefill_host": self.host,
                        "embedding_port": embedding_port,
                    }
                )
                cum_idx += 1
                cum_num_items += assigned_num
            part_idx_offset += num_parts

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

            embedding_size_list_sort = [None for _ in range(total_num_parts)]
            embedding_length_tot = 0
            response_json_list_sort = [None for _ in range(total_num_parts)]
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
        mm_data = self._extract_url_data(obj)
        if obj.rid is None:
            obj.rid = uuid.uuid4().hex

        if mm_data and len(mm_data) > 0:
            logger.info(f"Processing {len(mm_data)} mm_items for request {obj.rid}")
            obj.need_wait_for_mm_inputs = True
            obj.num_items_assigned = self._assign_items_by_modality(
                mm_data, len(self.encode_urls)
            )
            encode_thread = threading.Thread(
                target=self._run_encode_in_thread,
                args=(
                    obj.rid,
                    mm_data,
                    "encode",
                    obj.num_items_assigned,
                    None,
                ),
                daemon=True,
            )
            encode_thread.start()

    def _extract_url_data(self, request_obj) -> List[Dict]:
        mm_data = []
        for attr, modality in [
            ("image_data", Modality.IMAGE),
            ("video_data", Modality.VIDEO),
            ("audio_data", Modality.AUDIO),
        ]:
            mm_items = getattr(request_obj, attr, None)
            if mm_items:
                if not isinstance(mm_items, list):
                    mm_items = [mm_items]
                for mm_item in mm_items:
                    mm_data.append(
                        {
                            "url": (
                                mm_item.url
                                if isinstance(mm_item, ImageData)
                                else mm_item
                            ),
                            "modality": modality,
                        }
                    )
        return mm_data

    # For zmq_to_tokenizer and mooncake
    async def recv_mm_data(self, request_obj, mm_processor, prompt):
        try:
            if len(self.encode_urls) == 0:
                return None
            req_id = uuid.uuid4().hex
            embedding_port, recv_socket = get_zmq_socket_on_host(self.context, zmq.PULL)
            mm_data = self._extract_url_data(request_obj)
            asyncio.create_task(
                self.encode(req_id, mm_data, embedding_port, "encode", "send")
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

        recv_embedding_data: MultiModalEmbeddingData = None

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
                recv_embedding_data = MultiModalEmbeddingData.from_embedding_data(
                    recv_obj
                )
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
        video_grid_thw = recv_embedding_data.get_video_grid()
        audio_feature_lens = recv_embedding_data.get_audio_feature_lens()
        kwargs = {
            "img_grid_thw": img_grid_thw,
            "video_grid_thw": video_grid_thw,
            "audio_feature_lens": audio_feature_lens,
        }
        video_timestamps = recv_embedding_data.get_attr_list(
            "video_timestamps", flatten=True
        )
        second_per_grid_ts = recv_embedding_data.get_attr_list(
            "second_per_grid_ts", flatten=False
        )
        if video_timestamps is not None:
            kwargs["video_timestamps"] = video_timestamps
        if second_per_grid_ts is not None:
            kwargs["second_per_grid_ts"] = second_per_grid_ts
        mm_inputs = mm_processor.get_mm_data(prompt, recv_embedding, **kwargs)
        return mm_inputs
