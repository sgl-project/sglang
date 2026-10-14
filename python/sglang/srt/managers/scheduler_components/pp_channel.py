from __future__ import annotations

import logging
from collections import defaultdict, deque
from contextlib import nullcontext
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.distributed.communication_op import attn_cp_tp_broadcast_pyobj
from sglang.srt.distributed.parallel_state import P2PWork
from sglang.srt.environ import envs
from sglang.srt.managers.utils import GenerationBatchResult, allocate_distinct_stream
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import point_to_point_pyobj

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler_components.bundles import Streams

logger = logging.getLogger(__name__)


class PPChannel:
    """The PP messages of one rank: Python objects and tensor dicts sent to the
    next rank and received from the previous one, the stream they run on, and
    the events that fence the next forward behind them."""

    def __init__(
        self,
        *,
        pp_group,
        world_group,
        attn_tp_group,
        device_module,
        streams: Streams,
    ):
        device = streams.device
        schedule_stream = streams.schedule_stream
        forward_stream = streams.forward_stream
        copy_stream = streams.copy_stream
        self.pp_group = pp_group
        self.world_group = world_group
        self.attn_tp_group = attn_tp_group
        self.device_module = device_module
        self.schedule_stream = schedule_stream
        self.forward_stream = forward_stream
        self.send_proxy_work = []
        self.send_proxy_requires_forward_fence = False
        self.launch_event = None
        self.pp_proxy_recv_event = None
        self.pp_send_done_event = None
        # With SGLANG_PP_COMM_OVERLAP, PP tensor sends/recvs run on a dedicated
        # stream so async sends do not fence the next forward through the
        # scheduler stream. Otherwise they stay on the current stream.
        self.pp_comm_stream = None
        self.pp_comm_stream_ctx = nullcontext()
        if (
            envs.SGLANG_PP_COMM_OVERLAP.get()
            and torch.cuda.is_available()
            and str(device).startswith("cuda")
        ):
            self.pp_comm_stream = allocate_distinct_stream(
                self.device_module,
                (self.schedule_stream, self.forward_stream, copy_stream),
            )
            self.pp_comm_stream_ctx = self.device_module.stream(self.pp_comm_stream)
            logger.info(
                "PP tensor communication overlap enabled on a dedicated CUDA stream"
            )

        self._pp_tensor_dict_inbox: Dict[
            str, deque[Tuple[Dict[str, torch.Tensor], Optional[torch.Event]]]
        ] = defaultdict(deque)

    @property
    def is_first_rank(self) -> bool:
        return self.pp_group.is_first_rank

    @property
    def is_last_rank(self) -> bool:
        return self.pp_group.is_last_rank

    def send_next(self, data) -> List[P2PWork]:
        return self.send_pyobj(data, async_send=True)

    def _record_comm_event(self) -> Optional[torch.Event]:
        """Mark the tail of the PP comm stream; None when not overlapping."""
        if self.pp_comm_stream is None:
            return None
        event = self.device_module.Event()
        event.record(self.pp_comm_stream)
        return event

    def commit(
        self,
        work: List[P2PWork],
        fence_next_forward: bool = False,
    ) -> None:
        with self.pp_comm_stream_ctx:
            for p2p_work in work:
                p2p_work.work.wait()
        if fence_next_forward and work:
            # CUDA Graph outputs are views of replay-owned static buffers. Do not
            # replay the next forward until NCCL has stopped reading the previous
            # proxy tensors. Eager prefill remains overlapped.
            self.pp_send_done_event = self._record_comm_event()
        work.clear()

    def commit_proxy_send(self) -> None:
        self.commit(
            self.send_proxy_work,
            fence_next_forward=self.send_proxy_requires_forward_fence,
        )
        self.send_proxy_requires_forward_fence = False

    def wait_forward_dependencies(self) -> None:
        self.forward_stream.wait_stream(self.schedule_stream)
        if self.pp_send_done_event is not None:
            self.forward_stream.wait_event(self.pp_send_done_event)
            self.pp_send_done_event = None
        if self.pp_proxy_recv_event is not None:
            self.forward_stream.wait_event(self.pp_proxy_recv_event)
            self.pp_proxy_recv_event = None

    def send_pyobj(self, data, async_send: bool = False):
        p2p_work = []
        if get_parallel().attn_tp_rank == 0 and get_parallel().attn_cp_rank == 0:
            dp_offset = (
                get_parallel().attn_dp_rank
                * get_parallel().attn_cp_size
                * get_parallel().attn_tp_size
            )
            p2p_work = point_to_point_pyobj(
                data,
                get_parallel().pp_rank * get_parallel().tp_size + dp_offset,
                self.world_group.cpu_group,
                get_parallel().pp_rank * get_parallel().tp_size + dp_offset,
                ((get_parallel().pp_rank + 1) % get_parallel().pp_size)
                * get_parallel().tp_size
                + dp_offset,
                async_send=async_send,
            )
        return p2p_work

    def recv_prev(self):
        if get_parallel().attn_tp_rank == 0 and get_parallel().attn_cp_rank == 0:
            dp_offset = (
                get_parallel().attn_dp_rank
                * get_parallel().attn_cp_size
                * get_parallel().attn_tp_size
            )
            data = point_to_point_pyobj(
                [],
                get_parallel().pp_rank * get_parallel().tp_size + dp_offset,
                self.world_group.cpu_group,
                ((get_parallel().pp_rank - 1) % get_parallel().pp_size)
                * get_parallel().tp_size
                + dp_offset,
                get_parallel().pp_rank * get_parallel().tp_size + dp_offset,
            )
        else:
            data = None

        data = attn_cp_tp_broadcast_pyobj(data)
        return data

    def send_dict(
        self,
        tensor_dict: Dict[str, torch.Tensor],
        async_send: bool = True,
        msg_type: str = "default",
        ready_event: Optional[torch.Event] = None,
    ):
        # Warn once if using default untyped messages
        if msg_type == "default":
            logger.warning_once(
                "PP send: using default untyped message. "
                "Consider adding msg_type='proxy' or 'output' to avoid recv conflicts."
            )
        tensor_dict["__msg_type__"] = msg_type
        p2p_work = []
        with self.pp_comm_stream_ctx:
            if ready_event is not None:
                self.device_module.current_stream().wait_event(ready_event)
            p2p_work.extend(
                self.pp_group.send_tensor_dict(
                    tensor_dict=tensor_dict,
                    all_gather_group=(self.attn_tp_group),
                    async_send=async_send,
                )
            )
        return p2p_work

    def send_proxy(self, result: GenerationBatchResult) -> None:
        with torch.profiler.record_function("send_proxy_dict_to_next_stage"):
            self.send_proxy_work = self.send_dict(
                result.pp_hidden_states_proxy_tensors.tensors,
                async_send=True,
                msg_type="proxy",
                ready_event=self.launch_event,
            )
        self.send_proxy_requires_forward_fence = result.can_run_cuda_graph

    def _recv_typed_dict(
        self,
        expected_kind: str = "default",
        all_gather_group: Optional = None,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Event]]:
        """Receive a typed tensor dict, demultiplexing by msg_type.

        If a message of the wrong kind is received, it's stashed in the queue
        and we continue receiving until we get the expected kind. The returned
        event marks receive completion on the PP comm stream (None otherwise).
        """
        if expected_kind in self._pp_tensor_dict_inbox:
            inbox_queue = self._pp_tensor_dict_inbox[expected_kind]
            if inbox_queue:
                return inbox_queue.popleft()

        while True:
            with self.pp_comm_stream_ctx:
                tensor_dict = self.pp_group.recv_tensor_dict(
                    all_gather_group=all_gather_group
                )
                recv_event = self._record_comm_event()
            received_kind = tensor_dict.get("__msg_type__", "default")
            if received_kind == expected_kind:
                if received_kind == "default":
                    logger.warning_once(
                        f"PP recv: got default untyped message. Content keys: {tensor_dict.keys()}"
                        "Consider adding msg_type='proxy' or 'output' to avoid recv conflicts."
                    )
                return tensor_dict, recv_event
            else:
                logger.debug(
                    f"PP recv: expected {expected_kind}, got {received_kind}, stashing"
                )
                self._pp_tensor_dict_inbox[received_kind].append(
                    (tensor_dict, recv_event)
                )

    def recv_proxy(self) -> Optional[PPProxyTensors]:
        pp_proxy_tensors = None
        if not self.pp_group.is_first_rank:
            tensor_dict, self.pp_proxy_recv_event = self._recv_typed_dict(
                expected_kind="proxy",
                all_gather_group=(self.attn_tp_group),
            )
            pp_proxy_tensors = PPProxyTensors(tensor_dict)
        return pp_proxy_tensors

    def recv_output(
        self,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Event]]:
        return self._recv_typed_dict(
            expected_kind="output",
            all_gather_group=(self.attn_tp_group),
        )
