import asyncio
import torch
from typing import Union, Optional
import zmq
import logging
import socket
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from sglang.srt.managers.io_struct import PrefilledReqInput, KVTransferFetch, KVTransferAck
from sglang.srt.managers.pd_disaggregation_controller import PD_DISAGGREGATION_PORT
from sglang.srt.server_args import ServerArgs
from sglang.srt.mem_cache.radix_cache import ReqToTokenPool, TokenToKVPoolAllocator
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.utils import broadcast_pyobj
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save
from sglang.srt.utils import get_zmq_socket

logger = logging.getLogger(__name__)


class TransferEngine:
    """Handles the transfer of data using mooncake_vllm_adaptor and ZeroMQ."""

    def __init__(self,
                 local_host: str,
                 metadata_server: str,
                 device_name: str):
        try:
            import mooncake_vllm_adaptor as mva
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector.") from e

        self.engine = mva.mooncake_vllm_adaptor()

        self.initialize(local_host=local_host,
                        metadata_server=metadata_server,
                        protocol="rdma",
                        device_name=device_name,
                        metadata_backend="etcd")

        # Initialize ZeroMQ context and sockets
        self.local_host = local_host
        self.metadata_server = metadata_server
        self.device_name = device_name

        self.buffer_cleaner = ThreadPoolExecutor(max_workers=1)

    def initialize(self, local_host: str, metadata_server: str,
                   protocol: str, device_name: str,
                   metadata_backend: Union[str, None]) -> None:
        """Initialize the mooncake instance."""
        if metadata_backend is None:
            self.engine.initialize(local_host, metadata_server, protocol,
                                   device_name)
        else:
            supported_backend = ["etcd", "redis"]
            metadata_backend = metadata_backend.lower()
            if metadata_backend not in supported_backend:
                raise ValueError(
                    "Mooncake Configuration error. `metadata_backend`"
                    f"should be one of {supported_backend}.")

            self.engine.initializeExt(local_host, metadata_server,
                                      protocol, device_name, metadata_backend)

    def allocate_managed_buffer(self, length: int) -> int:
        """Allocate a managed buffer of the specified length."""
        ret = self.engine.allocateManagedBuffer(length)
        if ret <= 0:
            logger.error("Allocation Return Error")
            raise Exception("Allocation Return Error")
        return ret

    def free_managed_buffer(self, buffer: int, length: int) -> int:
        """Free a previously allocated managed buffer."""
        return self.engine.freeManagedBuffer(buffer, length)

    def transfer_sync(self, remote_url: str, buffer: int, peer_buffer_address: int,
                      length: int, op: int) -> int:
        """Synchronously transfer data to the specified address."""
        ret = self.engine.transferSync(remote_url, buffer,
                                       peer_buffer_address, length, op)
        if ret < 0:
            logger.error("Transfer Return Error")
            raise Exception("Transfer Return Error")
        return ret

    def write_bytes_to_buffer(self, buffer: int, user_data: bytes,
                              length: int) -> int:
        """Write bytes to the allocated buffer."""
        return self.engine.writeBytesToBuffer(buffer, user_data, length)

    def read_bytes_from_buffer(self, buffer: int, length: int) -> bytes:
        """Read bytes from the allocated buffer."""
        return self.engine.readBytesFromBuffer(buffer, length)


KV_TRANSFER_AGENT_PORT = 18000


class KVTransferAgent:
    def __init__(self,
                 server_args: ServerArgs,
                 req_to_token_pool: ReqToTokenPool = None,
                 token_to_kv_pool_allocator: TokenToKVPoolAllocator = None,
                 layer_num: int = 0,
                 tp_rank: int = 0,
                 attn_tp_cpu_group: torch.distributed.ProcessGroup = None,
                 device: str = "cpu"):
        self.kv_buffer = {}
        self.layer_num = layer_num
        self.tp_rank = tp_rank
        self.attn_tp_cpu_group = attn_tp_cpu_group
        self.role = server_args.kv_transfer_config.role
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

        self.attn_tp_rank, self.attn_tp_size, _ = (
            compute_dp_attention_world_info(
                server_args.enable_dp_attention,
                self.tp_rank,
                server_args.tp_size,
                server_args.dp_size,
            )
        )

        tp_size_per_node = server_args.tp_size // server_args.nnodes
        self.addr = f"{socket.gethostname()}:{KV_TRANSFER_AGENT_PORT+self.tp_rank % tp_size_per_node}"

        if server_args.nnodes == 1 and server_args.dist_init_addr is None:
            dist_init_host = "127.0.0.1"
        else:
            dist_init_host, _ = server_args.dist_init_addr.split(":")

        context = zmq.Context(2)

        if self.role == "decode":
            self.recv_from_pd_disagg_controller = get_zmq_socket(
                context, zmq.PULL, f"tcp://{dist_init_host}:{PD_DISAGGREGATION_PORT+self.tp_rank+1}", False)
            self.send_to_pd_disagg_controller = get_zmq_socket(
                context, zmq.PUSH, f"tcp://{server_args.kv_transfer_config.prefill_dist_init_host}:{PD_DISAGGREGATION_PORT}", False)
        else:
            self.recv_from_pd_disagg_controller = get_zmq_socket(
                context, zmq.PULL, f"tcp://{dist_init_host}:{PD_DISAGGREGATION_PORT+self.tp_rank+1}", False)
            self.send_to_pd_disagg_controller = get_zmq_socket(
                context, zmq.PUSH, f"tcp://{server_args.kv_transfer_config.decode_dist_init_host}:{PD_DISAGGREGATION_PORT}", False)

        self.device = device
        self.engine = TransferEngine(
            self.addr, server_args.kv_transfer_config.transfer_engine_metadata_server, server_args.kv_transfer_config.transfer_engine_rdma_device)

    def set_kv_buffer(self, req: Req) -> int:
        if self.attn_tp_rank != 0:
            return 0
        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        flatten = torch.stack(
            [self.token_to_kv_pool_allocator.get_kvcache().get_key_buffer(i)[kv_indices]
             for i in range(self.layer_num)]
        )
        kv_cache = safetensors_save({"tensor": flatten.to(self.device)})
        self.kv_buffer[req.rid] = kv_cache
        return len(kv_cache)

    def get_kv_buffer(self, req: Req) -> torch.Tensor:
        if self.attn_tp_rank == 0:
            dst_ptr = self._allocate_transfer_kv_buffer(req.kv_cache_length)
            # fetch kv buffer util transfer done
            self.send_to_pd_disagg_controller.send_pyobj(KVTransferFetch(
                rid=req.rid,
                src_addr=req.kv_transfer_src_addr,
                src_rank=req.kv_transfer_src_rank,
                dst_addr=self.addr,
                dst_rank=self.tp_rank,
                dst_ptr=dst_ptr
            ))
            recv_obj = self.recv_from_pd_disagg_controller.recv_pyobj()
            if not isinstance(recv_obj, KVTransferAck):
                raise ValueError(f"Unknown message type: {type(recv_obj)}")
            elif recv_obj.code != 0:
                raise Exception(f"KV transfer failed: {recv_obj.code}")
            # read from buffer
            kv_cache = self._read_bytes_from_buffer(
                dst_ptr, req.kv_cache_length)
            # free buffer
            self._free_transfer_kv_buffer(dst_ptr, req.kv_cache_length)
            # load to device
            loaded_tensor = safetensors_load(kv_cache)["tensor"]
        else:
            loaded_tensor = None
        if self.attn_tp_size > 1:
            loaded_tensor = broadcast_pyobj(
                loaded_tensor, self.attn_tp_rank, self.attn_tp_cpu_group)
        return loaded_tensor

    def dispatch_prefilled_req(self, req: Req):
        if self.role == "decode":
            return
        if self.attn_tp_rank != 0:
            return
        self.send_to_pd_disagg_controller.send_pyobj(PrefilledReqInput(
            rid=req.rid,
            mm_inputs=None,
            input_text=req.origin_input_text,
            input_ids=req.origin_input_ids,
            sampling_params=req.sampling_params,
            return_logprob=req.return_logprob,
            logprob_start_len=req.logprob_start_len,
            top_logprobs_num=req.top_logprobs_num,
            token_ids_logprob=req.token_ids_logprob,
            stream=req.stream,
            output_ids=req.output_ids,
            kv_transfer_src_addr=self.addr,
            kv_transfer_src_rank=self.tp_rank,
            kv_cache_length=req.kv_cache_length,
        ))
    # when prefill node receive kv transfer request

    def _handle_kv_transfer_fetch(self, req: KVTransferFetch):
        kv_cache = self.kv_buffer[req.rid]
        kv_cache_length = len(kv_cache)
        src_ptr = self._allocate_transfer_kv_buffer(kv_cache_length)
        self._write_bytes_to_buffer(src_ptr, kv_cache, kv_cache_length)
        # send data to remote_addr
        op_write = 1
        self.engine.transfer_sync(
            req.dst_addr, src_ptr, req.dst_ptr, kv_cache_length, op_write)
        # send ack to remote addr
        self.send_to_pd_disagg_controller.send_pyobj(
            KVTransferAck(req.rid, req.dst_addr, req.dst_rank, 0))
        # free buffer
        self._free_transfer_kv_buffer(src_ptr, kv_cache_length)
        del self.kv_buffer[req.rid]

    def _allocate_transfer_kv_buffer(self, length: int) -> int:
        return self.engine.allocate_managed_buffer(length)

    def _free_transfer_kv_buffer(self, buffer: int, length: int):
        self.engine.free_managed_buffer(buffer, length)

    def _read_bytes_from_buffer(self, buffer: int, length: int) -> bytes:
        return self.engine.read_bytes_from_buffer(buffer, length)

    def _write_bytes_to_buffer(self, buffer: int, data: bytes, length: int):
        self.engine.write_bytes_to_buffer(buffer, data, length)

    def event_loop(self):
        if self.role == "decode":
            return
        while True:
            recv_obj = self.recv_from_pd_disagg_controller.recv_pyobj()
            if isinstance(recv_obj, KVTransferFetch):
                self._handle_kv_transfer_fetch(recv_obj)
            else:
                raise ValueError(f"Unknown message type: {type(recv_obj)}")
