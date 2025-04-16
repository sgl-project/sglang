import hashlib
import torch
import os
from typing import Union
import zmq
import logging
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

from sglang.srt.managers.io_struct import PrefilledReqInput, KVTransferFetch, KVTransferAck
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.pd_disaggregation_controller import PD_DISAGGREGATION_PORT
from sglang.srt.server_args import ServerArgs
from sglang.srt.mem_cache.radix_cache import ReqToTokenPool, TokenToKVPoolAllocator
from sglang.srt.managers.schedule_batch import Req
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
            logger.error(f"Transfer Return Error: {ret}")
            raise Exception(f"Transfer Return Error: {ret}")
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
                 kv_cache_size_factor: int = 0,
                 device: str = "cpu"):
        self.kv_buffer = {}
        self.layer_num = layer_num
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size
        self.role = server_args.kv_transfer_config.role
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.handle_kv_cache_fetch_ct = {}
        self.handle_kv_cache_buffer_ptr = {}
        self.handle_kv_cache_buffer_length = {}
        self.kv_cache_size_factor = kv_cache_size_factor
        self.waiting_kv_transfers = {}
        self.waiting_kv_transfer_lock = threading.Lock()

        self.attn_tp_rank, self.attn_tp_size, _ = (
            compute_dp_attention_world_info(
                server_args.enable_dp_attention,
                self.tp_rank,
                server_args.tp_size,
                server_args.dp_size,
            )
        )

        tp_size_per_node = server_args.tp_size // server_args.nnodes
        hostname = os.environ.get(
            "KV_TRANSFER_AGENT_HOSTNAME", socket.gethostname())
        self.addr = f"{hostname}:{KV_TRANSFER_AGENT_PORT+self.tp_rank % tp_size_per_node}"
        logger.debug(f"KVTransferAgent addr: {self.addr}")

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
        self.kv_buffer[req.rid] = flatten.to(self.device)
        return self.kv_cache_size_factor + flatten.numel() * flatten.element_size()
    
    def get_kv_buffer(self, req_list: List[Req]) -> dict[str, torch.Tensor]:
        same_src_req_list = {}
        res = {}
        for req in req_list:
            if req.kv_transfer_src_addr not in same_src_req_list:
                same_src_req_list[req.kv_transfer_src_addr] = []
            same_src_req_list[req.kv_transfer_src_addr].append(req)
        for _, same_src_reqs in same_src_req_list.items():
            rid_tensor_map = self._get_kv_buffer_from_same_src(same_src_reqs)
            for rid, tensor in rid_tensor_map.items():
                res[rid] = tensor
        return res

    def _get_kv_buffer_from_same_src(self, req_list: List[Req]) -> dict[str, torch.Tensor]:
        if len(req_list) == 0:
            return {}
        batch_kv_cache_length = 0
        for req in req_list:
            batch_kv_cache_length += req.kv_cache_length
        res = {}
        batch_kv_cache = None
        try:
            # allocate buffer
            dst_ptr = self._allocate_transfer_kv_buffer(batch_kv_cache_length)
            # send fetch request
            logger.debug(f"[KVTransferAgent] Sending kv transfer fetch request")
            kv_transfer_fetch = KVTransferFetch(
                rids=[req.rid for req in req_list],
                reqs_hash=hashlib.md5(
                    ''.join([req.rid for req in req_list]).encode()).hexdigest(),
                src_addr=req_list[0].kv_transfer_src_addr,
                src_rank=req_list[0].kv_transfer_src_rank,
                dst_addr=self.addr,
                dst_rank=self.tp_rank,
                dst_ptr=dst_ptr,
                fetch_ct=self.attn_tp_size,
            )
            # send kv transfer request and wait for it to be done
            received_kv_cache_length = self._send_kv_transfer_req(
                kv_transfer_fetch)
            # read from buffer
            batch_kv_cache = self._read_bytes_from_buffer(
                dst_ptr, received_kv_cache_length)
            loaded_tensor = safetensors_load(batch_kv_cache)["tensor"]
            pt = 0
            for req in req_list:
                res[req.rid] = loaded_tensor[:, pt:pt +
                                             len(req.origin_input_ids), :, :]
                pt += len(req.origin_input_ids)
        except Exception as e:
            logger.error(f"[KVTransferAgent] Get batch kv buffer failed: {e}")
            if batch_kv_cache is not None:
                logger.error(f"[KVTransferAgent] Batch kv cache: {str(batch_kv_cache)}")
            for req in req_list:
                req.finished_reason = FINISH_ABORT(
                    message=f"Get batch kv buffer failed: {e}")
            return {}
        finally:
            # free buffer
            if dst_ptr > 0:
                self._free_transfer_kv_buffer(dst_ptr, batch_kv_cache_length)
        return res

    def _send_kv_transfer_req(self, kv_transfer_fetch: KVTransferFetch, timeout: int = 60) -> int:
        """Send kv transfer request and wait for it to be done.
        If timeout, raise an exception. Default timeout is 60 seconds.
        """
        with self.waiting_kv_transfer_lock:
            self.waiting_kv_transfers[kv_transfer_fetch.reqs_hash] = None

        self.send_to_pd_disagg_controller.send_pyobj(kv_transfer_fetch)

        logger.debug(
            f"[KVTransferAgent] Waiting for kv transfer {kv_transfer_fetch.reqs_hash} to be done")

        start_time = time.time()
        while True:
            with self.waiting_kv_transfer_lock:
                if kv_transfer_fetch.reqs_hash not in self.waiting_kv_transfers:
                    raise Exception(
                        f"KV transfer {kv_transfer_fetch.reqs_hash} not in waiting list")

                result = self.waiting_kv_transfers[kv_transfer_fetch.reqs_hash]
                if result is not None:
                    del self.waiting_kv_transfers[kv_transfer_fetch.reqs_hash]
                    if result.error_message is not None:
                        raise Exception(
                            f"KV transfer failed: {result.error_message}")
                    return result.kv_cache_length

                if time.time() - start_time > timeout:
                    raise Exception("KV transfer timeout")

            time.sleep(0.01)

    def dispatch_prefilled_req(self, req: Req):
        if self.role == "decode":
            return
        if self.attn_tp_rank != 0:
            return
        logger.debug(
            f"[KVTransferAgent] Dispatch prefilled request {req.rid}, output_ids: {req.output_ids}")
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
        logger.debug(
            f"[KVTransferAgent] Dispatched prefilled request {req.rid}, output_ids: {req.output_ids}")

    def _handle_kv_transfer_fetch(self, req: KVTransferFetch):
        batch_kv_cache_length = 0
        try:
            if req.reqs_hash not in self.handle_kv_cache_fetch_ct:
                kv_caches = torch.cat([self.kv_buffer[rid]
                                      for rid in req.rids], dim=1)
                serialized = safetensors_save({"tensor": kv_caches})
                batch_kv_cache_length = len(serialized)
                src_ptr = self._allocate_transfer_kv_buffer(
                    batch_kv_cache_length)
                self._write_bytes_to_buffer(
                    src_ptr, serialized, batch_kv_cache_length)
                self.handle_kv_cache_buffer_ptr[req.reqs_hash] = src_ptr
                self.handle_kv_cache_fetch_ct[req.reqs_hash] = 0
                self.handle_kv_cache_buffer_length[req.reqs_hash] = batch_kv_cache_length
            # send data to remote_addr
            op_write = 1
            self.engine.transfer_sync(
                req.dst_addr, self.handle_kv_cache_buffer_ptr[req.reqs_hash], req.dst_ptr, self.handle_kv_cache_buffer_length[req.reqs_hash], op_write)
            ack_error_message = None
        except Exception as e:
            logger.error(f"[KVTransferAgent] KV transfer failed: {e}")
            ack_error_message = str(e)

        # send ack to remote addr
        self.send_to_pd_disagg_controller.send_pyobj(
            KVTransferAck(None, req.reqs_hash, req.dst_addr, req.dst_rank, ack_error_message, batch_kv_cache_length))

        if ack_error_message is not None:
            return

        # free buffer
        self.handle_kv_cache_fetch_ct[req.reqs_hash] += 1
        if self.handle_kv_cache_fetch_ct[req.reqs_hash] == req.fetch_ct:
            self._free_transfer_kv_buffer(
                self.handle_kv_cache_buffer_ptr[req.reqs_hash], self.handle_kv_cache_buffer_length[req.reqs_hash])
            for rid in req.rids:
                del self.kv_buffer[rid]
            del self.handle_kv_cache_buffer_ptr[req.reqs_hash]
            del self.handle_kv_cache_fetch_ct[req.reqs_hash]
            del self.handle_kv_cache_buffer_length[req.reqs_hash]

    def _handle_kv_transfer_ack(self, kv_transfer_ack: KVTransferAck):
        with self.waiting_kv_transfer_lock:
            if kv_transfer_ack.reqs_hash not in self.waiting_kv_transfers:
                logger.debug(
                    f"[KVTransferAgent] Received kv transfer ack: {kv_transfer_ack.reqs_hash} but not in waiting list")
                return
            self.waiting_kv_transfers[kv_transfer_ack.reqs_hash] = kv_transfer_ack
            logger.debug(
                f"[KVTransferAgent] Received kv transfer ack: {kv_transfer_ack.reqs_hash} with error message: {kv_transfer_ack.error_message}")

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
                raise ValueError(f"[KVTransferAgent] Unknown message type: {type(recv_obj)}")
