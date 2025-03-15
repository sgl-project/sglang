import zmq
from typing import Union

from sglang.srt.server_args import ServerArgs, PortArgs
from sglang.srt.utils import get_zmq_socket
from sglang.srt.managers.io_struct import KVTransferReqInput, KVTransferReqOutput, PrefilledReqInput
from sglang.utils import TypeBasedDispatcher
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOut,
    BatchStrOut,
)

PD_DISAGGREGATION_PORT = 16000

class PDAggregationController:
    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
        self.recv_from_transfer_agent = None
        self.send_to_transfer_agent = {}
        self.send_to_tokenizer = None

        assert server_args.kv_transfer_config is not None, "KVTransferConfig is required"

        context = zmq.Context(server_args.tp_size+2)
        if server_args.kv_transfer_config.role == "decode":
            self.recv_from_transfer_agent = get_zmq_socket(context, zmq.PULL, f"tcp://{server_args.kv_transfer_config.decode_dist_init_host}:{PD_DISAGGREGATION_PORT}", True)
        else:
            self.recv_from_transfer_agent = get_zmq_socket(context, zmq.PULL, f"tcp://{server_args.kv_transfer_config.prefill_dist_init_host}:{PD_DISAGGREGATION_PORT}", True)
        
        self.send_to_tokenizer = get_zmq_socket(context, zmq.PUSH, port_args.tokenizer_ipc_name, True)
        for rank in range(server_args.tp_size):
            agent_addr = f"{server_args.kv_transfer_config.role}-{rank}"
            self.send_to_transfer_agent[agent_addr] = get_zmq_socket(context, zmq.PUSH, f"tcp://{server_args.dist_init_addr}:{PD_DISAGGREGATION_PORT + rank}", True)

        self._request_dispatcher = TypeBasedDispatcher(
            [
                (BatchEmbeddingOut, self.handle_batch_out),
                (BatchStrOut, self.handle_batch_out),
                (PrefilledReqInput, self.handle_prefilled_req),
                (KVTransferReqInput, self.handle_kv_transfer_req),
                (KVTransferReqOutput, self.handle_kv_transfer_resp),
            ]
        )

    def event_loop(self):
        while True:
            recv_obj = self.recv_from_transfer_agent.recv_pyobj()
            self._request_dispatcher(recv_obj)

    def _handle_kv_transfer_req(self, req: KVTransferReqInput):
        self.send_to_transfer_agent[req.src_addr].send_pyobj(req)

    def _handle_kv_transfer_resp(self, req: KVTransferReqOutput):
        self.send_to_transfer_agent[req.dst_addr].send_pyobj(req)

    def _handle_prefilled_req(self, req: PrefilledReqInput):
        self.send_to_tokenizer.send_pyobj(req)

    def _handle_batch_out(self, recv_obj: Union[BatchEmbeddingOut, BatchStrOut]):
        self.send_to_tokenizer.send_pyobj(recv_obj)
