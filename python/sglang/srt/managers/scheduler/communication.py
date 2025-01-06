from types import SimpleNamespace
from typing import List

import torch
import zmq
from sglang.srt.managers.io_struct import (
    AbortReq,
    CloseSessionReqInput,
    FlushCacheReq,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    ProfileReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)
from sglang.srt.managers.schedule_batch import (
    Req,
)
from sglang.srt.managers.scheduler.core import SchedulerCore, SchedulerCoreCallback
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    broadcast_pyobj,
    get_zmq_socket,
)


class SchedulerCommunication(SchedulerCoreCallback):
    def __init__(
        self,
        core: SchedulerCore,
        server_args: ServerArgs,
        port_args: PortArgs,
        tp_rank: int,
        tp_size: int,
        tp_cpu_group: torch.distributed.ProcessGroup,
    ):
        self.core = core
        self.server_args = server_args
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_cpu_group = tp_cpu_group

        context = zmq.Context(2)

        if self.tp_rank == 0 or self.server_args.enable_dp_attention:
            self._recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, port_args.scheduler_input_ipc_name
            )
            self._send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name
            )

            if self.server_args.skip_tokenizer_init:
                # Directly send to the TokenizerManager
                self._send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name
                )
            else:
                # Send to the DetokenizerManager
                self._send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.detokenizer_ipc_name
                )
        else:
            self._recv_from_tokenizer = None
            self._send_to_tokenizer = SimpleNamespace(send_pyobj=lambda x: None)
            self._send_to_detokenizer = SimpleNamespace(send_pyobj=lambda x: None)

    def recv_and_process_requests(self):
        recv_reqs = self._recv_requests()
        self._process_input_requests(recv_reqs)

    def _recv_requests(self) -> List[Req]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""
        if self.tp_rank == 0 or self.server_args.enable_dp_attention:
            recv_reqs = []

            while True:
                try:
                    recv_req = self._recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                recv_reqs.append(recv_req)
        else:
            recv_reqs = None

        if self.tp_size != 1 and not self.server_args.enable_dp_attention:
            recv_reqs = broadcast_pyobj(recv_reqs, self.tp_rank, self.tp_cpu_group)
        return recv_reqs

    def _process_input_requests(self, recv_reqs: List):
        for recv_req in recv_reqs:
            if isinstance(recv_req, TokenizedGenerateReqInput):
                self.core.handle_generate_request(recv_req)
            elif isinstance(recv_req, TokenizedEmbeddingReqInput):
                self.core.handle_embedding_request(recv_req)
            elif isinstance(recv_req, FlushCacheReq):
                self.core.flush_cache()
            elif isinstance(recv_req, AbortReq):
                self.core.abort_request(recv_req)
            elif isinstance(recv_req, UpdateWeightFromDiskReqInput):
                success, message = self.core.update_weights_from_disk(recv_req)
                self._send_to_tokenizer.send_pyobj(
                    UpdateWeightFromDiskReqOutput(success, message)
                )
            elif isinstance(recv_req, InitWeightsUpdateGroupReqInput):
                success, message = self.core.init_weights_update_group(recv_req)
                self._send_to_tokenizer.send_pyobj(
                    InitWeightsUpdateGroupReqOutput(success, message)
                )
            elif isinstance(recv_req, UpdateWeightsFromDistributedReqInput):
                success, message = self.core.update_weights_from_distributed(recv_req)
                self._send_to_tokenizer.send_pyobj(
                    UpdateWeightsFromDistributedReqOutput(success, message)
                )
            elif isinstance(recv_req, UpdateWeightsFromTensorReqInput):
                success, message = self.core.update_weights_from_tensor(recv_req)
                self._send_to_tokenizer.send_pyobj(
                    UpdateWeightsFromTensorReqOutput(success, message)
                )
            elif isinstance(recv_req, GetWeightsByNameReqInput):
                parameter = self.core.get_weights_by_name(recv_req)
                self._send_to_tokenizer.send_pyobj(GetWeightsByNameReqOutput(parameter))
            elif isinstance(recv_req, ProfileReq):
                if recv_req == ProfileReq.START_PROFILE:
                    self.core.start_profile()
                else:
                    self.core.stop_profile()
            elif isinstance(recv_req, OpenSessionReqInput):
                session_id, success = self.core.open_session(recv_req)
                self._send_to_tokenizer.send_pyobj(
                    OpenSessionReqOutput(session_id=session_id, success=success)
                )
            elif isinstance(recv_req, CloseSessionReqInput):
                self.core.close_session(recv_req)
            else:
                raise ValueError(f"Invalid request: {recv_req}")

    def send_to_detokenizer(self, obj):
        self._send_to_detokenizer.send_pyobj(obj)
