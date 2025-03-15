import torch
from typing import Optional, Union, Enum
import asyncio
import zmq
import logging
from concurrent.futures import ThreadPoolExecutor

from sglang.srt.layers.attention.torch_native_backend import RadixAttention
from sglang.srt.managers.io_struct import KVTransferFetch, KVTransferAck, PrefilledReqInput
from sglang.srt.server_args import KVTransferConfig
from sglang.srt.mem_cache.radix_cache import ReqToTokenPool, TokenToKVPoolAllocator
from sglang.srt.managers.pd_disaggregation_controller import PD_DISAGGREGATION_PORT
from sglang.srt.managers.schedule_batch import Req
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


        self.initialize(local_hostname=local_host, 
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

class KVTransferAgent:
    def __init__(self, 
        config: KVTransferConfig,
        req_to_token_pool: ReqToTokenPool = None,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator = None,
        layer_num: int = 0,
        rank: int = 0, 
        device: str = "cpu"):
        self.kv_buffer = []
        self.ack_events = {}
        self.layer_num = layer_num
        self.rank = rank
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.addr = f"{config.role}-{rank}"

        context = zmq.Context(2)

        if config.role == "decode":
            self.recv_from_pd_disagg_controller = get_zmq_socket(context, zmq.PULL, f"tcp://{config.decode_dist_init_host}:{PD_DISAGGREGATION_PORT+rank}", False)
            self.send_to_pd_disagg_controller = get_zmq_socket(context, zmq.PUSH, f"tcp://{config.prefill_dist_init_host}:{PD_DISAGGREGATION_PORT+rank}", False)
        else:
            self.recv_from_pd_disagg_controller = get_zmq_socket(context, zmq.PULL, f"tcp://{config.prefill_dist_init_host}:{PD_DISAGGREGATION_PORT+rank}", False)
            self.send_to_pd_disagg_controller = get_zmq_socket(context, zmq.PUSH, f"tcp://{config.decode_dist_init_host}:{PD_DISAGGREGATION_PORT+rank}", False)

        self.device = device
        self.engine = TransferEngine(config.transfor_engine_local_host, config.transfor_engine_metadata_server, device)

    def set_kv_buffer(self, req: Req) -> int:
        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        flatten = torch.stack([self.token_to_kv_pool_allocator.get_kv_cache().get_key_buffer(i)[kv_indices] for i in range(self.layer_num)])
        kv_cache = safetensors_save({"tensor": flatten.to(self.device)}) 
        self.kv_buffer[req.rid] = kv_cache
        return len(kv_cache)

    async def get_kv_buffer(self, req: Req) -> torch.Tensor:
        dst_ptr = self._allocate_transfer_kv_buffer(req.rid)
        self.send_to_pd_disagg_controller.send_pyobj(KVTransferFetch(
            rid=req.rid,
            src_addr=req.kv_transfer_agent_addr,
            dst_addr=self.addr,
            dst_ptr=dst_ptr
        ))
        kv_cache = self._read_bytes_from_buffer(dst_ptr, req.kv_cache_length)
        # send control-plan message to get kv cache from prefill node
        # TODO: 缺少 kv cache 源地址, 目的地址。能否将message的源地址和目的地址作为header，rid，dst_ptr，length 作为 payload
        self.send_to_pd_disagg_controller.send_pyobj(KVTransferFetch(req.rid, req.kv_transfer_src, req.kv_transfer_dst, dst_ptr))
        # wait fro transfor done
        await self._wait_for_transfer_done(req.rid)
        # read from buffer
        kv_cache = self._read_bytes_from_buffer(dst_ptr, req.kv_cache_length)
        # free buffer
        self._free_transfer_kv_buffer(dst_ptr, req.kv_cache_length)
        # TODO: why del self.kv_buffer[req.rid]
        # del self.kv_buffer[req.rid]
        
        # load to device
        loaded_tensor = safetensors_load(kv_cache)["tensor"].to(self.device)
        return loaded_tensor
    
    def dispatch_prefilled_req(self, req: Req):
        self.send_to_pd_disagg_controller.send_pyobj(PrefilledReqInput(req.rid, self.addr, req.kv_cache_length))

    async def _wait_for_transfer_done(self, rid: str):
        if rid not in self.ack_events:
            self.ack_events[rid] = asyncio.Event()
        await self.ack_events[rid]
        self.ack_events[rid] = None

    # when prefill node receive kv transfer request
    def _handle_kv_transfer_fetch(self, req: KVTransferFetch):
        kv_cache = self.kv_buffer[req.rid]
        kv_cache_length = len(kv_cache)
        src_ptr = self._allocate_transfer_kv_buffer(kv_cache_length)
        # send data to remote_addr
        op_write = 1
        self.engine.transfer_sync(req.dst_addr, src_ptr, req.dst_ptr, kv_cache_length, op_write)
        # send ack to remote addr
        self.send_to_pd_disagg_controller.send_pyobj(KVTransferAck(req.rid, 0))
        # free buffer
        self._free_transfer_kv_buffer(src_ptr, kv_cache_length)

    # when decode node receive kv transfer ACK request
    def _handle_kv_transfer_ack(self, req: KVTransferAck):
        if req.rid in self.ack_events and req.code == 0:
            self.ack_events[req.rid].set()

    def _allocate_transfer_kv_buffer(self, length: int) -> int:
        return self.engine.allocate_managed_buffer(length)

    def _free_transfer_kv_buffer(self, buffer: int, length: int):
        self.engine.free_managed_buffer(buffer, length)

    def _read_bytes_from_buffer(self, buffer: int, length: int) -> bytes:
        return self.engine.read_bytes_from_buffer(buffer, length)
        
    def _write_bytes_to_buffer(self, buffer: int, data: bytes, length: int):
        self.engine.write_bytes_to_buffer(buffer, data, length)

    def event_loop(self):
        while True:
            recv_obj = self.recv_from_pd_disagg_controller.recv_pyobj()
            if isinstance(recv_obj, KVTransferFetch):
                self._handle_kv_transfer_fetch(recv_obj)
            elif isinstance(recv_obj, KVTransferAck):
                self._handle_kv_transfer_ack(recv_obj)
            else:
                raise ValueError(f"Unknown message type: {type(recv_obj)}")
