import zmq
from typing import Union
import asyncio

from sglang.srt.server_args import ServerArgs, PortArgs
from sglang.srt.utils import get_zmq_socket
from sglang.utils import TypeBasedDispatcher
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOut,
    BatchStrOut,
)
from sglang.srt.managers.io_struct import PrefilledReqInput, KVTransferFetch, KVTransferAck, AbortReq

PD_DISAGGREGATION_PORT = 17000

class PDDisaggregationController:
    def __init__(self, 
                 server_args: ServerArgs, 
                 port_args: PortArgs,
                 send_to_scheduler: zmq.Socket):
        self.recv_from_transfer_agent = None
        self.send_to_transfer_agent = {}
        self.send_to_tokenizer = None

        assert server_args.kv_transfer_config is not None, "KVTransferConfig is required"
        
        context = zmq.Context(server_args.tp_size+3)
        self.recv_from_transfer_agent = get_zmq_socket(context, zmq.PULL, f"tcp://*:{PD_DISAGGREGATION_PORT}", True)
        self.send_to_scheduler = send_to_scheduler
        self.send_to_tokenizer = get_zmq_socket(context, zmq.PUSH, port_args.tokenizer_ipc_name, False)
        for rank in range(server_args.tp_size):
            self.send_to_transfer_agent[rank] = get_zmq_socket(context, zmq.PUSH, f"tcp://*:{PD_DISAGGREGATION_PORT + rank + 1}", True)

        self._request_dispatcher = TypeBasedDispatcher(
            [
                (BatchEmbeddingOut, self._handle_batch_out),
                (BatchStrOut, self._handle_batch_out),
                (AbortReq, self._handle_abort_req),
                (PrefilledReqInput, self._handle_prefilled_req),
                (KVTransferFetch, self._handle_kv_transfer_req),
                (KVTransferAck, self._handle_kv_transfer_resp),
            ]
        )

    def event_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._handle_request_loop())
        loop.close()

    async def _handle_request_loop(self):
        while True:
            recv_obj = self.recv_from_transfer_agent.recv_pyobj()
            self._request_dispatcher(recv_obj)

    def _handle_kv_transfer_req(self, req: KVTransferFetch):
        self.send_to_transfer_agent[req.src_rank].send_pyobj(req)

    def _handle_kv_transfer_resp(self, req: KVTransferAck):
        self.send_to_transfer_agent[req.dst_rank].send_pyobj(req)

    def _handle_batch_out(self, recv_obj: Union[BatchEmbeddingOut, BatchStrOut]):
        self.send_to_tokenizer.send_pyobj(recv_obj) 

    def _handle_abort_req(self, req: AbortReq):
        self.send_to_tokenizer.send_pyobj(req)
       
    def _handle_prefilled_req(self, req: PrefilledReqInput):
        self.send_to_scheduler.send_pyobj(req)
