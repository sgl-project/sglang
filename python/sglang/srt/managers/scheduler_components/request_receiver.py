from __future__ import annotations

import logging
from dataclasses import dataclass
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Union,
)

import torch
import zmq
from torch.distributed import ReduceOp, all_reduce, barrier

from sglang.srt.disaggregation.utils import prepare_abort
from sglang.srt.distributed.communication_op import attn_cp_tp_broadcast_pyobj
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    MMInputsProcessError,
    MMInputsProcessMode,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    sock_recv,
)
from sglang.srt.managers.mm_utils import (
    discard_shm_features,
    has_shm_features,
    unwrap_shm_features,
)
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.observability.scheduler_stage_metrics import (
    SCHEDULER_STAGE_RECV_REQUESTS,
    SchedulerStageMetricsRecorder,
    scheduler_stage_method,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_mm,
    get_parallel,
    is_ep_scale_joiner,
)
from sglang.srt.utils import (
    broadcast_pyobj,
    point_to_point_pyobj,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.rust_server.server import RustServer
    from sglang.srt.server_args import ServerArgs
    from sglang.test.scripted_runtime.scheduler_hook import ScriptedSchedulerHook
    from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
        ScriptedTokenizerRecvProxy,
    )

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerRequestReceiver:
    recv_from_tokenizer: Union[zmq.Socket, ScriptedTokenizerRecvProxy, RustServer]
    recv_from_rpc: Optional[zmq.Socket]
    recv_skipper: Any
    input_blocker: Any
    mm_receiver: Any
    ps: ParallelState
    tp_group: Any
    tp_cpu_group: Any
    attn_tp_group: Any
    attn_tp_cpu_group: Any
    attn_cp_group: Any
    attn_cp_cpu_group: Any
    world_group: Any
    server_args: ServerArgs
    model_config: ModelConfig
    max_recv_per_poll: int
    stream_output: Callable[..., None]
    get_last_batch: Callable[[], Any]
    scripted_scheduler_hook: Optional[ScriptedSchedulerHook] = None
    scheduler_stage_metrics: Optional[SchedulerStageMetricsRecorder] = None

    def recv_limit_reached(self, num_recv_reqs: int) -> bool:
        if self.max_recv_per_poll < 0:
            return False
        return num_recv_reqs >= self.max_recv_per_poll

    @scheduler_stage_method(SCHEDULER_STAGE_RECV_REQUESTS)
    def recv_requests(
        self,
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput, Any]]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""

        if self.scripted_scheduler_hook is not None:
            self.scripted_scheduler_hook.step()

        if self.recv_skipper is not None:
            if not self.recv_skipper.handle(self.get_last_batch()):
                return []

        recv_reqs = self._pull_raw_reqs()

        if self.input_blocker is not None:
            recv_reqs = self.input_blocker.handle(recv_reqs)

        self._set_mm_process_mode(recv_reqs)
        recv_reqs = self._broadcast_reqs_across_ranks(recv_reqs)

        if self.ps.pp_rank == 0:
            self.unwrap_pickle_wrapper(recv_reqs)

        recv_reqs = self._apply_mm_receiver(recv_reqs)

        # EPD's waiting path can refill ``mm_inputs`` after the initial
        # request fanout. Only requests that were synchronized as NONE may
        # transition here; never overwrite an entry-rank BROADCAST decision
        # from rank-local post-fanout state.
        # This re-stamp is safe only while every MM receiver backend returns
        # rank-symmetric refill status and shapes. New backends must preserve
        # that invariant or scheduler ranks can diverge again.
        self._set_mm_process_mode(recv_reqs, only_if_none=True)

        self._finalize_shm_features(recv_reqs)
        self._materialize_broadcast_mm_inputs(recv_reqs)

        return recv_reqs

    def _set_mm_process_mode(
        self, recv_reqs: Optional[List], *, only_if_none: bool = False
    ) -> None:
        """Stamp the source-selected MM processing mode onto each work request.

        The request payload is subsequently broadcast to TP peers.  Keeping
        this mode in the payload prevents any scheduler rank from deriving
        collective participation from its local MM object state.
        """
        if not recv_reqs:
            return

        for req in recv_reqs:
            if isinstance(req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)):
                if (
                    only_if_none
                    and req.mm_inputs_process_mode is not MMInputsProcessMode.NONE
                ):
                    continue
                if isinstance(req.mm_inputs, (MultimodalInputs, MMInputsProcessError)):
                    # A PP predecessor has already prepared a shared result.
                    mode = MMInputsProcessMode.LOCAL
                elif req.mm_inputs is None:
                    mode = MMInputsProcessMode.NONE
                elif get_mm().mm_feature_transport == "cuda_vmm":
                    # CUDA-VMM has its own local materialization protocol; do
                    # not serialize GPU-backed inputs through Gloo.
                    mode = MMInputsProcessMode.LOCAL
                elif get_mm().enable_broadcast_mm_inputs_process:
                    mode = MMInputsProcessMode.BROADCAST
                else:
                    mode = MMInputsProcessMode.LOCAL
                req.mm_inputs_process_mode = mode
            elif isinstance(
                req, (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput)
            ):
                self._set_mm_process_mode(req.batch, only_if_none=only_if_none)

    def _pull_raw_reqs(self) -> Optional[List]:
        if self.ps.pp_rank == 0:
            if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
                recv_reqs = []

                # Rust ringbuffer backend: drain the in-process ring fed by the
                # embedded Rust TokenizerManager instead of a zmq socket. Same
                # non-blocking, msgpack-decoded contract as the zmq path below.
                if envs.SGLANG_RUST_SERVER.get():
                    recv_reqs.extend(
                        self.recv_from_tokenizer.drain(self.max_recv_per_poll)
                    )
                    return recv_reqs

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_req = sock_recv(self.recv_from_tokenizer, zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_req)

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_rpc = sock_recv(self.recv_from_rpc, zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_rpc)
            else:
                recv_reqs = None
        else:
            if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
                dp_offset = (
                    self.ps.attn_dp_rank * self.ps.attn_cp_size * self.ps.attn_tp_size
                )
                recv_reqs = point_to_point_pyobj(
                    [],
                    self.ps.pp_rank * self.ps.tp_size + dp_offset,
                    self.world_group.cpu_group,
                    (self.ps.pp_rank - 1) * self.ps.tp_size + dp_offset,
                    self.ps.pp_rank * self.ps.tp_size + dp_offset,
                )
            else:
                recv_reqs = None
        return recv_reqs

    def _broadcast_reqs_across_ranks(self, recv_reqs: Optional[List]) -> List:
        if get_parallel().enable_dp_attention:
            if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
                work_reqs, control_reqs = self._split_work_and_control_reqs(recv_reqs)
            else:
                work_reqs = []
                control_reqs = []

            work_reqs = attn_cp_tp_broadcast_pyobj(work_reqs)

            # When dp_attention_local_control_broadcast is enabled, each DP
            # group leader already receives control messages from the DP
            # controller, so we broadcast within attn_tp_group + attn_cp_group
            # instead of the full tp_group.  This avoids an expensive
            # all-ranks gloo sync.
            _local_ctrl = (
                get_parallel().enable_dp_attention_local_control_broadcast
                or is_ep_scale_joiner()
            )
            if _local_ctrl:
                control_reqs = attn_cp_tp_broadcast_pyobj(control_reqs)
            elif self.ps.tp_size != 1:
                control_reqs = broadcast_pyobj(
                    control_reqs,
                    self.tp_group.rank,
                    self.tp_cpu_group,
                    src=self.tp_group.ranks[0],
                )
            recv_reqs = work_reqs + control_reqs
        elif self.ps.tp_size != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )
        return recv_reqs

    def unwrap_pickle_wrapper(self, recv_reqs: Optional[List]) -> None:
        if not recv_reqs:
            return

        for req in recv_reqs:
            if isinstance(req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)):
                req.unwrap_pickle_fields()
            elif isinstance(
                req, (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput)
            ):
                for sub_req in req:
                    sub_req.unwrap_pickle_fields()

    def _apply_mm_receiver(self, recv_reqs: List) -> List:
        # Process MM requests under EPD-disaggregation mode
        if (
            self.ps.pp_rank == 0
            and get_disagg().language_only
            and get_disagg().encoder_transfer_backend
            in ["zmq_to_scheduler", "mooncake"]
        ):
            recv_reqs, abort_reqs = self.mm_receiver.process_waiting_requests(recv_reqs)
            for req, error_msg, error_code in abort_reqs:
                if error_code is None:
                    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
                elif isinstance(error_code, HTTPStatus):
                    status_code = error_code
                else:
                    status_code = HTTPStatus(int(error_code))
                prepare_abort(req, error_msg, status_code=status_code)
                self.stream_output([req], req.return_logprob)
        return recv_reqs

    def _finalize_shm_features(self, recv_reqs: Optional[List]) -> None:
        """Materialize SHM features or mark the request failed on every rank."""
        if not recv_reqs or not self.model_config.is_multimodal:
            return

        tokenized_reqs = []
        for req in recv_reqs:
            if isinstance(req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)):
                tokenized_reqs.append(req)
            elif isinstance(
                req,
                (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput),
            ):
                tokenized_reqs.extend(req.batch)
        if not tokenized_reqs or not has_shm_features(tokenized_reqs):
            return

        # 1. wait until every rank has opened the shared feature segments
        parallel = get_parallel()
        if parallel.enable_dp_attention:
            if self.ps.attn_tp_size > 1:
                barrier(group=self.attn_tp_cpu_group)
            if self.ps.attn_cp_size > 1:
                barrier(group=self.attn_cp_cpu_group)
        elif self.ps.tp_size > 1:
            barrier(group=self.tp_cpu_group)

        # 2. materialize independently so one bad VLM request does not stop the loop
        failed = torch.zeros(len(tokenized_reqs), dtype=torch.int32)
        for index, req in enumerate(tokenized_reqs):
            if not has_shm_features([req]):
                continue
            try:
                unwrap_shm_features(req)
            except Exception:
                logger.exception(
                    "Failed to materialize shared-memory multimodal features for rid=%s",
                    req.rid,
                )
                discard_shm_features(req)
                failed[index] = 1

        # 3. all ranks reject the same requests before entering model collectives
        if parallel.enable_dp_attention:
            if self.ps.attn_tp_size > 1:
                all_reduce(failed, op=ReduceOp.MAX, group=self.attn_tp_cpu_group)
            if self.ps.attn_cp_size > 1:
                all_reduce(failed, op=ReduceOp.MAX, group=self.attn_cp_cpu_group)
        elif self.ps.tp_size > 1:
            all_reduce(failed, op=ReduceOp.MAX, group=self.tp_cpu_group)

        error = MMInputsProcessError(
            "Failed to materialize shared-memory multimodal features on a scheduler rank."
        )
        for index, req in enumerate(tokenized_reqs):
            if failed[index].item():
                discard_shm_features(req)
                req.mm_inputs = error

    def _materialize_broadcast_mm_inputs(self, recv_reqs: Optional[List]) -> None:
        """Run the ordered root-only MM protocol after request fanout.

        Every rank receives the same request list, so iterating its work items
        in order gives the follow-up collectives an identical schedule.  The
        result is written back before handlers run, leaving handlers entirely
        collective-free.
        """
        if not recv_reqs:
            return

        for req in self._iter_tokenized_reqs(recv_reqs):
            if req.mm_inputs_process_mode is MMInputsProcessMode.BROADCAST:
                result, error_message = self._broadcast_mm_inputs_result(req)
                req.mm_inputs = (
                    MMInputsProcessError(error_message)
                    if error_message is not None
                    else result
                )
                # A following PP stage must not repeat the large result fanout.
                req.mm_inputs_process_mode = MMInputsProcessMode.LOCAL

    @staticmethod
    def _iter_tokenized_reqs(recv_reqs: List):
        for req in recv_reqs:
            if isinstance(req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)):
                yield req
            elif isinstance(
                req, (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput)
            ):
                yield from req

    @staticmethod
    def _materialize_mm_envelope(req):
        try:
            if has_shm_features([req]):
                unwrap_shm_features(req)
            mm_inputs = req.mm_inputs
            result = (
                mm_inputs
                if isinstance(mm_inputs, (MultimodalInputs, MMInputsProcessError))
                else MultimodalInputs.from_processor_output(mm_inputs)
            )
            return result, None
        except Exception as exc:
            # This runs only on the collective source. Preserve the concise
            # envelope shared with peers, but retain the full traceback locally
            # for server-side diagnosis.
            logger.exception(
                "Failed to materialize broadcast multimodal inputs on entry rank"
            )
            return (
                None,
                "Failed to process multimodal inputs on entry rank: "
                f"{type(exc).__name__}: {exc}",
            )

    def _broadcast_mm_inputs_result(self, req):
        """Fan out one materialized result using the request fanout topology."""
        if get_parallel().enable_dp_attention:
            is_source = self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0
            envelopes = [self._materialize_mm_envelope(req)] if is_source else []
            return attn_cp_tp_broadcast_pyobj(envelopes)[0]

        is_source = self.tp_group.rank == self.tp_group.ranks[0]
        envelope = self._materialize_mm_envelope(req) if is_source else None
        if self.ps.tp_size != 1:
            envelope = broadcast_pyobj(
                envelope,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )
        return envelope

    def _split_work_and_control_reqs(self, recv_reqs: List):
        work_reqs = [
            req
            for req in recv_reqs
            if isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        control_reqs = [
            req
            for req in recv_reqs
            if not isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        return work_reqs, control_reqs
