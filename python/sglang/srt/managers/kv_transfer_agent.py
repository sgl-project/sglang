import torch
from typing import Tuple, Optional
import asyncio
import zmq
from sglang.srt.layers.attention.torch_native_backend import RadixAttention
from sglang.srt.managers.io_struct import KVTransferReqInput, KVTransferReqOutput, PrefilledReqInput
from sglang.srt.server_args import KVTransferConfig
from sglang.srt.mem_cache.radix_cache import ReqToTokenPool, TokenToKVPoolAllocator
from sglang.srt.managers.pd_disaggregation_controller import PD_DISAGGREGATION_PORT
from sglang.srt.managers.schedule_batch import Req
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save
from sglang.srt.utils import get_zmq_socket

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
        self.send_to_pd_disagg_controller.send_pyobj(KVTransferReqInput(
            rid=req.rid,
            src_addr=req.kv_transfer_agent_addr,
            dst_addr=self.addr,
            dst_ptr=dst_ptr
        ))
        kv_cache = self._read_bytes_from_buffer(dst_ptr, req.kv_cache_length)
        await self._wait_for_transfer_done(req.rid)
        self._free_transfer_kv_buffer(dst_ptr, req.kv_cache_length)
        del self.kv_buffer[req.rid]
        loaded_tensor = safetensors_load(kv_cache)["tensor"].to(self.device)
        return loaded_tensor
    
    def dispatch_prefilled_req(self, req: Req):
        self.send_to_pd_disagg_controller.send_pyobj(PrefilledReqInput(req.rid, self.addr, req.kv_cache_length))

    async def _wait_for_transfer_done(self, rid: str):
        if rid not in self.ack_events:
            self.ack_events[rid] = asyncio.Event()
        await self.ack_events[rid]
        self.ack_events[rid] = None

    def _handle_kv_transfer_req(self, req: KVTransferReqInput):
        kv_cache = self.kv_buffer[req.rid]
        kv_cache_length = len(kv_cache)
        src_ptr = self._allocate_transfer_kv_buffer(kv_cache_length)
        # TODO: send data to remote_addr
        self.send_to_pd_disagg_controller.send_pyobj(KVTransferReqOutput(
            dst_addr=req.dst_addr,
            code=0
        ))
        self._free_transfer_kv_buffer(src_ptr, kv_cache_length)

    def _handle_kv_transfer_resp(self, req_id: str):
        if req_id in self.ack_events:
            self.ack_events[req_id].set()

    def _allocate_transfer_kv_buffer(self, length: int) -> int:
        pass

    def _free_transfer_kv_buffer(self, buffer: int, length: int):
        pass

    def _read_bytes_from_buffer(self, buffer: int, length: int) -> bytes:
        pass

    def event_loop(self):
        while True:
            recv_obj = self.recv_from_pd_disagg_controller.recv_pyobj()
            if isinstance(recv_obj, KVTransferReqInput):
                self._handle_kv_transfer_req(recv_obj)
            elif isinstance(recv_obj, KVTransferReqOutput):
                self._handle_kv_transfer_resp(recv_obj)
            else:
                raise ValueError(f"Unknown message type: {type(recv_obj)}")
