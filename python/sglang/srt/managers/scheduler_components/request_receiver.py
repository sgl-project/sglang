from __future__ import annotations

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

import zmq
from torch.distributed import barrier

from sglang.srt.disaggregation.utils import prepare_abort
from sglang.srt.managers.io_struct import (
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    sock_recv,
)
from sglang.srt.managers.mm_utils import (
    has_shm_features,
    unwrap_shm_features,
)
from sglang.srt.utils import (
    broadcast_pyobj,
    broadcast_pyobj_frames,
    point_to_point_pyobj,
)
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method

import os
import pickle

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.server_args import ServerArgs
    from sglang.test.scripted_runtime.scheduler_hook import ScriptedSchedulerHook
    from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
        ScriptedTokenizerRecvProxy,
    )


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerRequestReceiver:
    recv_from_tokenizer: Union[zmq.Socket, ScriptedTokenizerRecvProxy]
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
    get_last_forward_mode: Callable[[], Any]
    scripted_scheduler_hook: Optional[ScriptedSchedulerHook] = None

    def recv_limit_reached(self, num_recv_reqs: int) -> bool:
        if self.max_recv_per_poll < 0:
            return False
        return num_recv_reqs >= self.max_recv_per_poll

    @scheduler_nvtx_method("scheduler.recv_requests")
    def recv_requests(
        self,
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput, Any]]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""

        if self.scripted_scheduler_hook is not None:
            self.scripted_scheduler_hook.step()

        if self.recv_skipper is not None:
            if not self.recv_skipper.handle(self.get_last_forward_mode()):
                return []

        if self._shm_ring_enabled():
            recv_reqs = self._recv_requests_shm_ring()
            recv_reqs = self._apply_mm_receiver(recv_reqs)
            self._finalize_shm_features(recv_reqs)
            return recv_reqs

        if self._raw_frame_fast_path_enabled():
            recv_reqs = self._recv_requests_raw_frames()
            recv_reqs = self._apply_mm_receiver(recv_reqs)
            self._finalize_shm_features(recv_reqs)
            return recv_reqs

        recv_reqs = self._pull_raw_reqs()

        if self.input_blocker is not None:
            recv_reqs = self.input_blocker.handle(recv_reqs)

        recv_reqs = self._broadcast_reqs_across_ranks(recv_reqs)

        if self.ps.pp_rank == 0:
            self.unwrap_pickle_wrapper(recv_reqs)

        recv_reqs = self._apply_mm_receiver(recv_reqs)

        self._finalize_shm_features(recv_reqs)

        return recv_reqs

    def _raw_frame_fast_path_enabled(self) -> bool:
        # Opt-in, and only for the plain intra-node TP broadcast: no DP-attention,
        # tp_size>1, no input_blocker (which needs deserialized objects on rank0),
        # and not the EPD/mm-receiver rewrite path.  Under these conditions the
        # only work done between recv and broadcast is None, so we can defer
        # deserialization to after the broadcast and reuse the tokenizer's frames.
        return (
            envs.SGLANG_TP_RAW_FRAME_BROADCAST.get()
            and not self.server_args.enable_dp_attention
            and self.ps.tp_size != 1
            and self.input_blocker is None
            and not (
                self.ps.pp_rank == 0
                and self.server_args.language_only
                and self.server_args.encoder_transfer_backend
                in ["zmq_to_scheduler", "mooncake"]
            )
        )

    def _recv_requests_raw_frames(self) -> List:
        """Fast path for the plain intra-node TP broadcast.

        Rank 0 pulls the raw pickle frames off the tokenizer (and rpc) sockets
        without unpickling them, broadcasts the bytes to peer ranks with a
        single serialization, and every rank ``pickle.loads`` each frame exactly
        once.  This replaces the recv_pyobj (unpickle) + broadcast_pyobj
        (re-pickle) round-trip on the source rank.

        Safe because the fast path is only enabled when nothing between recv and
        broadcast needs the deserialized objects on rank 0 (no input_blocker, no
        DP-attention, not the EPD/mm-receiver rewrite path -- see
        ``_raw_frame_fast_path_enabled``).  Control / rpc objects flow through
        identically: they are picklable and are processed the same way after a
        single ``pickle.loads`` on every rank.  Empty iterations cost exactly one
        collective, matching the original ``broadcast_pyobj([])``.
        """
        src = self.tp_group.ranks[0]

        if self.tp_group.rank == src:
            frames = self._pull_raw_frames()
            broadcast_pyobj_frames(
                frames, self.tp_group.rank, self.tp_cpu_group, src=src
            )
        else:
            frames = broadcast_pyobj_frames(
                [], self.tp_group.rank, self.tp_cpu_group, src=src
            )
        return [pickle.loads(f) for f in frames]

    def _pull_raw_frames(self) -> List[bytes]:
        """Pull raw (still-pickled) ZMQ frames from the tokenizer + rpc sockets."""
        frames: List[bytes] = []
        while True:
            try:
                if self.recv_limit_reached(len(frames)):
                    break
                frames.append(self.recv_from_tokenizer.recv(zmq.NOBLOCK))
            except zmq.ZMQError:
                break
        if self.recv_from_rpc is not None:
            while True:
                try:
                    if self.recv_limit_reached(len(frames)):
                        break
                    frames.append(self.recv_from_rpc.recv(zmq.NOBLOCK))
                except zmq.ZMQError:
                    break
        return frames

    # ------------------------------------------------------------------
    # Experiment C (PROTOTYPE): /dev/shm ring instead of gloo broadcast.
    # ------------------------------------------------------------------
    def _shm_ring_enabled(self) -> bool:
        # PROTOTYPE / UNSAFE for live serving.  Unlike the gloo broadcast, the
        # shm ring does NOT keep request admission lock-step across TP ranks:
        # rank 0 writes and continues while peers poll independently, so a peer
        # may observe a request one iteration later than rank 0.  That breaks the
        # invariant that every TP rank builds the same batch each step, which can
        # desync the NCCL forward collectives and hang.  Enabling it requires an
        # explicit acknowledgement so it can be used only for offline
        # single-batch micro-timing, never real traffic.
        if envs.SGLANG_TP_REQ_SHM_RING.get() and not envs.SGLANG_TP_REQ_SHM_RING_ACK_UNSAFE.get():
            import logging

            logging.getLogger(__name__).warning(
                "SGLANG_TP_REQ_SHM_RING is set but the ring is unsafe for live "
                "TP serving (breaks per-iteration request lock-step). Ignoring; "
                "set SGLANG_TP_REQ_SHM_RING_ACK_UNSAFE=1 to force."
            )
            return False
        return (
            envs.SGLANG_TP_REQ_SHM_RING.get()
            and envs.SGLANG_TP_REQ_SHM_RING_ACK_UNSAFE.get()
            and not self.server_args.enable_dp_attention
            and self.ps.tp_size != 1
            and self.input_blocker is None
            and not (
                self.ps.pp_rank == 0
                and self.server_args.language_only
                and self.server_args.encoder_transfer_backend
                in ["zmq_to_scheduler", "mooncake"]
            )
        )

    def _shm_ring(self):
        """Lazily create the per-rank ring endpoint, cached module-side (the
        receiver is a frozen dataclass so it cannot hold mutable state)."""
        from sglang.srt.managers import tp_req_shm_ring as ring_mod

        src = self.tp_group.ranks[0]
        # nccl_port is shared by all TP ranks of this server -> stable key.
        key = f"tpreq-{getattr(self.server_args, 'port', 0)}-{self.ps.tp_size}"
        cache = getattr(ring_mod, "_ENDPOINT_CACHE", None)
        if cache is None:
            cache = ring_mod._ENDPOINT_CACHE = {}
        if key in cache:
            return cache[key]

        path = ring_mod.shm_path_for(key)
        slots = envs.SGLANG_TP_REQ_SHM_RING_SLOTS.get()
        slot_size = envs.SGLANG_TP_REQ_SHM_RING_SLOT_KB.get() * 1024
        if self.tp_group.rank == src:
            ep = ring_mod.TpReqShmRingWriter(path, slots, slot_size)
        else:
            # Wait for the writer to create the file.
            import time

            for _ in range(2000):
                if os.path.exists(path):
                    break
                time.sleep(0.005)
            ep = ring_mod.TpReqShmRingReader(path)
        cache[key] = ep
        return ep

    def _recv_requests_shm_ring(self) -> List:
        ep = self._shm_ring()
        src = self.tp_group.ranks[0]
        if self.tp_group.rank == src:
            frames = self._pull_raw_frames()
            if frames:
                if not ep.write_batch(frames):
                    # Batch too large for a slot: fall back to gloo for this one.
                    broadcast_pyobj_frames(
                        frames, self.tp_group.rank, self.tp_cpu_group, src=src
                    )
                    return [pickle.loads(f) for f in frames]
            return [pickle.loads(f) for f in frames]
        else:
            out = ep.poll()
            if out is None:
                # Ring overrun: this peer stalled longer than the ring capacity.
                # Re-snapshot to the current head and log; data for the skipped
                # window is unrecoverable via shm, so surface it loudly.
                import logging

                logging.getLogger(__name__).error(
                    "TP req shm ring overrun on rank %s; re-syncing (some "
                    "requests may have been missed -- increase "
                    "SGLANG_TP_REQ_SHM_RING_SLOTS)",
                    self.tp_group.rank,
                )
                ep.last_seq = ep.current_seq()
                return []
            return [pickle.loads(f) for f in out]

    def _pull_raw_reqs(self) -> Optional[List]:
        if self.ps.pp_rank == 0:
            if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
                recv_reqs = []

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
        if self.server_args.enable_dp_attention:
            if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
                work_reqs, control_reqs = self._split_work_and_control_reqs(recv_reqs)
            else:
                work_reqs = None
                control_reqs = None

            if self.ps.attn_tp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )

            if self.ps.attn_cp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_cp_group.rank,
                    self.attn_cp_cpu_group,
                    src=self.attn_cp_group.ranks[0],
                )

            # When dp_attention_local_control_broadcast is enabled, each DP
            # group leader already receives control messages from the DP
            # controller, so we broadcast within attn_tp_group + attn_cp_group
            # instead of the full tp_group.  This avoids an expensive
            # all-ranks gloo sync.
            _local_ctrl = self.server_args.enable_dp_attention_local_control_broadcast
            if _local_ctrl:
                if self.ps.attn_tp_size != 1:
                    control_reqs = broadcast_pyobj(
                        control_reqs,
                        self.attn_tp_group.rank,
                        self.attn_tp_cpu_group,
                        src=self.attn_tp_group.ranks[0],
                    )
                if self.ps.attn_cp_size != 1:
                    control_reqs = broadcast_pyobj(
                        control_reqs,
                        self.attn_cp_group.rank,
                        self.attn_cp_cpu_group,
                        src=self.attn_cp_group.ranks[0],
                    )
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
            and self.server_args.language_only
            and self.server_args.encoder_transfer_backend
            in ["zmq_to_scheduler", "mooncake"]
        ):
            recv_reqs, abort_reqs = self.mm_receiver.process_waiting_requests(recv_reqs)
            for req, error_msg, error_code in abort_reqs:
                status_code = (
                    HTTPStatus.BAD_REQUEST
                    if error_code == 400
                    else HTTPStatus.INTERNAL_SERVER_ERROR
                )
                prepare_abort(req, error_msg, status_code=status_code)
                self.stream_output([req], req.return_logprob)
        return recv_reqs

    def _finalize_shm_features(self, recv_reqs: Optional[List]) -> None:
        # Unwrap shared memory features AFTER all broadcasts complete,
        # so that ShmPointerMMData metadata (not full tensor data) is what
        # gets serialized during broadcast_pyobj.
        if recv_reqs:
            if self.model_config.is_multimodal and has_shm_features(recv_reqs):
                # The broadcast source returns with its original objects while
                # peer ranks may still be unpickling ShmPointerMMData
                # (-> shm_open).  Synchronize the same CPU groups that carried
                # SHM-backed work requests before materialize() unlinks them.
                if self.server_args.enable_dp_attention:
                    if self.ps.attn_tp_size > 1:
                        barrier(group=self.attn_tp_cpu_group)
                    if self.ps.attn_cp_size > 1:
                        barrier(group=self.attn_cp_cpu_group)
                elif self.ps.tp_size > 1:
                    barrier(group=self.tp_cpu_group)
            for req in recv_reqs:
                unwrap_shm_features(req)

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
