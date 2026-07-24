import logging
from typing import List, Optional

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.base.conn import (
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVArgs,
    KVPoll,
    KVTransferMetric,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


# For warmup reqs, we don't kv transfer, we use the fake manager, sender and receiver
class FakeKVManager(BaseKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.kv_args = args
        self.req_to_decode_prefix_len = {}

    def register_to_bootstrap(self):
        pass


class FakeKVSender(BaseKVSender):
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
        req_has_disagg_prefill_dp_rank: bool = False,
    ):
        self.kv_mgr = mgr
        self.has_sent = False
        self.conclude_state: Optional[KVPoll] = None
        self.poll_count = 0  # Track number of times polled before send()
        # Threshold for auto-aborting leaked requests (health checks, connection validation)
        # while avoiding premature abortion during multi-GPU bootstrap sync.
        self._auto_abort_threshold = (
            envs.SGLANG_DISAGGREGATION_FAKE_AUTO_ABORT_THRESHOLD.get()
        )

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state

        if not self.has_sent:
            self.poll_count += 1

            # During normal operation, a request transitions from bootstrap to prefill
            # within a few polls. If poll_count exceeds threshold, it's a leaked
            # request (health check, aborted) that will never call send().
            # Auto-abort to Failed to prevent queue accumulation.
            if self.poll_count >= self._auto_abort_threshold:
                logger.warning(
                    f"FakeKVSender auto-aborting after {self.poll_count} polls without send() - "
                    "likely a leaked request (health check or aborted)"
                )
                self.conclude_state = KVPoll.Failed
                return KVPoll.Failed

            # First poll and subsequent polls (< threshold): return WaitingForInput
            # to allow multi-GPU bootstrap synchronization without premature abortion
            return KVPoll.WaitingForInput

        # Assume transfer completed instantly after send()
        logger.debug("FakeKVSender poll success after send()")
        self.conclude_state = KVPoll.Success
        return KVPoll.Success

    def get_transfer_metric(self) -> KVTransferMetric:
        return KVTransferMetric()

    def init(
        self,
        kv_indices: list[int],
        aux_index: Optional[int] = None,
    ):
        logger.debug(
            f"FakeKVSender init with kv_indices: {kv_indices}, aux_index: {aux_index}"
        )
        pass

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
    ):
        # Note: send() after auto-abort (poll_count >= threshold) is a no-op
        # since conclude_state=Failed takes precedence in poll()
        self.has_sent = True
        self.poll_count = 0  # Reset counter after send
        logger.debug(
            f"FakeKVSender send with kv_indices: {kv_indices}, state_indices: {state_indices}"
        )

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")

    def abort(self):
        self.conclude_state = KVPoll.Failed


class FakeKVReceiver(BaseKVReceiver):
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.bootstrap_done = False
        self.has_sent_metadata = False
        self.require_staging: bool = False
        self.conclude_state: Optional[KVPoll] = None

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        if not self.bootstrap_done:
            return KVPoll.Bootstrapping
        if not self.has_sent_metadata:
            return KVPoll.WaitingForInput
        logger.debug("FakeKVReceiver poll success")
        self.conclude_state = KVPoll.Success
        return KVPoll.Success

    def init(
        self,
        prefill_dp_rank: int,
    ):
        self.bootstrap_done = True

    def send_metadata(
        self,
        kv_indices: list[int],
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
    ):
        self.has_sent_metadata = True
        logger.debug(
            f"FakeKVReceiver send_metadata with kv_indices: {kv_indices}, aux_index: {aux_index}, state_indices: {state_indices}"
        )

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")

    def abort(self):
        self.conclude_state = KVPoll.Failed
