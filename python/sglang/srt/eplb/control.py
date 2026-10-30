"""Correlated terminal acknowledgements for ordered PP recorder controls.

Requests keep the existing PP relay. Only replies use the tokenizer control
socket, so no PP event loop waits for a downstream stage to receive its request.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import OrderedDict

import torch.distributed as dist
import zmq
from msgspec.structs import replace

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.stage_domain import EPLBStageDomain
from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.srt.managers.io_struct import (
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
)
from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils.network import get_zmq_socket


def recorder_control_fan_out():
    parallel = get_parallel()
    if get_exec().moe.elastic_ep_backend is not None:
        return parallel.num_dp_ranks
    # Every EP group in every stage and DP replica contributes one terminal.
    return (
        parallel.dp_size * parallel.tp_size // parallel.moe_ep_size * parallel.pp_size
    )


class ExpertDistributionCommunicator(FanOutCommunicator):
    def __init__(self, send, fan_out, *, timeout=120.0):
        super().__init__(send, fan_out, mode="queueing")
        self._timeout = timeout
        self._operation_id = None
        self._owners = set()
        self._client_id = uuid.uuid4().hex
        self._sequence = 0

    async def queueing_call(self, req):
        async with self._queueing_lock:
            if not req.operation_id:
                self._sequence += 1
                req.operation_id = f"{self._client_id}:{self._sequence}"
            self._operation_id = req.operation_id
            self._owners = set()
            self._result_event = asyncio.Event()
            self._result_values = []
            self._result_fan_out = self._fan_out
            try:
                self._send(req)
                await asyncio.wait_for(self._result_event.wait(), self._timeout)
                results = self._result_values
                failed = [r.message for r in results if not r.success]
                if failed:
                    raise RuntimeError("; ".join(failed))
                return results
            except asyncio.TimeoutError as exc:
                raise RuntimeError(
                    f"EPLB control {req.operation_id} timed out: "
                    f"received {len(self._owners)}/{self._result_fan_out} owners; "
                    f"completed={sorted(self._owners)}"
                ) from exc
            finally:
                self._operation_id = None
                self._result_event = self._result_values = None
                self._result_fan_out = None
                self._owners = set()

    def handle_recv(self, result):
        owner = (result.replica_rank, result.pp_rank, result.root_global_rank)
        if result.operation_id != self._operation_id or owner in self._owners:
            return
        self._owners.add(owner)
        super().handle_recv(result)


class ExpertDistributionControl:
    def __init__(self, *, tokenizer_ipc_name, get_model=None):
        self._tokenizer_ipc_name = tokenizer_ipc_name
        self._get_model = get_model
        self._sender = None
        self._completed = OrderedDict()
        # The terminal cache is bounded. A per-client high-water mark rejects
        # an evicted replay instead of executing its action a second time.
        self._last_sequence = {}

    def handle(self, req, *, send_result=None):
        if not req.operation_id:
            # Non-tokenizer frontends predate the correlated protocol. Keep
            # their PP1 dispatch/egress contract; PP needs stage terminals.
            if get_parallel().pp_size > 1:
                return ExpertDistributionReqOutput(
                    success=False,
                    message="PP recorder control requires an operation_id",
                )
            recorder = get_global_expert_distribution_recorder()
            actions = {
                ExpertDistributionReqType.START_RECORD: recorder.start_record,
                ExpertDistributionReqType.STOP_RECORD: recorder.stop_record,
                ExpertDistributionReqType.DUMP_RECORD: recorder.dump_record,
            }
            actions[req.action]()
            return ExpertDistributionReqOutput()
        domain = (
            EPLBStageDomain.from_world()
            if get_exec().moe.elastic_ep_backend is not None
            else EPLBStageDomain.from_runtime()
        )
        result = self.execute(req, domain=domain)
        if domain.is_root:
            reply = replace(
                result, http_worker_ipc=getattr(req, "http_worker_ipc", None)
            )
            if send_result is not None:
                send_result(reply, req)
                return
            if self._sender is None:
                self._sender = SenderWrapper(
                    get_zmq_socket(
                        zmq.Context.instance(),
                        zmq.PUSH,
                        self._tokenizer_ipc_name,
                        False,
                    )
                )
            # A replay may arrive from a different HTTP worker. Do not cache
            # the first worker's return route on the terminal result.
            self._sender.send_output(
                reply,
                req,
            )

    def execute(self, req, *, domain):
        cached = self._completed.get(req.operation_id)
        if req.operation_id and cached is not None:
            action, result = cached
            if action == req.action:
                return result
            return self._result(
                req, domain, "operation_id reused for a different action"
            )

        error = None
        try:
            client_id, sequence = req.operation_id.rsplit(":", 1)
            sequence = int(sequence)
            if not client_id or sequence <= 0:
                raise ValueError
        except (ValueError, AttributeError):
            return self._result(
                req, domain, "operation_id must be client:positive-sequence"
            )
        if sequence <= self._last_sequence.get(client_id, 0):
            return self._result(
                req, domain, "terminal replay expired; action not repeated"
            )
        if get_exec().moe.enable_eplb:
            error = "manual recorder control conflicts with the automatic EPLB owner"
        elif get_exec().moe.expert_distribution_recorder_mode is None:
            error = "expert distribution recording is not configured"
        elif req.action not in (
            ExpertDistributionReqType.START_RECORD,
            ExpertDistributionReqType.STOP_RECORD,
            ExpertDistributionReqType.DUMP_RECORD,
        ):
            error = f"unsupported action {req.action}"

        owned_layer_ids = None
        if error is None and req.action == ExpertDistributionReqType.DUMP_RECORD:
            try:
                if self._get_model is not None:
                    model = self._get_model()
                    if hasattr(type(model), "routed_experts_weights_of_layer"):
                        from sglang.srt.eplb.eplb_manager import (
                            get_routed_experts_weights_of_layer,
                        )

                        owned_layer_ids = sorted(
                            get_routed_experts_weights_of_layer(model)
                        )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

        errors = self._stage_errors(error, domain)
        if not errors:
            try:
                recorder = get_global_expert_distribution_recorder()
                if req.action == ExpertDistributionReqType.START_RECORD:
                    # A new START clears the old window, as the HTTP API
                    # documents. A replay of the same operation is cached.
                    recorder.start_record()
                elif req.action == ExpertDistributionReqType.STOP_RECORD:
                    recorder.stop_record()
                else:
                    recorder.dump_record(
                        metadata={
                            "operation_id": req.operation_id,
                            "owned_layer_ids": owned_layer_ids,
                        }
                    )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            errors = self._stage_errors(error, domain)

        result = self._result(req, domain, "; ".join(errors))
        if req.operation_id:
            self._last_sequence[client_id] = sequence
            self._completed[req.operation_id] = (req.action, result)
            if len(self._completed) > 64:
                self._completed.popitem(last=False)
        return result

    @staticmethod
    def _stage_errors(error, domain):
        if domain.ep_size == 1:
            return [error] if error else []
        errors = [None] * domain.ep_size
        dist.all_gather_object(errors, error, group=domain.cpu_group)
        return [f"EP{rank}: {error}" for rank, error in enumerate(errors) if error]

    @staticmethod
    def _result(req, domain, error):
        parallel = get_parallel()
        return ExpertDistributionReqOutput(
            operation_id=req.operation_id,
            pp_rank=domain.pp_rank,
            root_global_rank=domain.root_global_rank,
            replica_rank=parallel.dp_rank if parallel.dp_size > 1 else 0,
            success=not error,
            message=(
                f"PP{domain.pp_rank} root{domain.root_global_rank} "
                f"{req.action}: {error}"
                if error
                else ""
            ),
        )
