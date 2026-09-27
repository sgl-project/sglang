import logging
import time
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
        self.bootstrap_room = bootstrap_room
        # Set by any chunk, not only the last one: nothing is transferred, so a
        # chunk that never comes cannot change the outcome.
        self.has_sent = False
        self.conclude_state: Optional[KVPoll] = None
        # Read here rather than off kv_mgr: a FAKE_BOOTSTRAP_HOST req on a real
        # backend pairs this sender with that backend's KVManager, which carries
        # no waiting_timeout in prefill mode.
        self.waiting_timeout = envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get()
        self.inited = False
        self.waiting_since: Optional[float] = None

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state

        if not self.has_sent:
            timeout_result = self._check_waiting_timeout()
            if timeout_result is not None:
                return timeout_result
            # Assume handshake completed instantly
            return KVPoll.WaitingForInput

        # Assume transfer completed instantly
        logger.debug("FakeKVSender poll success")
        self.conclude_state = KVPoll.Success
        return KVPoll.Success

    def _check_waiting_timeout(self) -> Optional[KVPoll]:
        # A send() that never comes must not pin the prefill inflight queue forever.
        # No deadline before init(): the request is still in the bootstrap queue.
        if not self.inited:
            return None
        if self.waiting_since is None:
            # Clock starts at the first poll after init(), not at init() itself:
            # the scheduler stops polling while a request queues and computes
            # prefill. Monotonic, so an NTP step cannot fail a healthy request.
            self.waiting_since = time.monotonic()
            return None
        elapsed = time.monotonic() - self.waiting_since
        if elapsed < self.waiting_timeout:
            return None
        logger.warning_once(
            "Some FakeKVSender requests fail to receive a KV chunk after bootstrapping. "
            "If a greater mean TTFT is acceptable, you can 'export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
        )
        logger.debug(
            f"FakeKVSender for room {self.bootstrap_room} timed out after {elapsed:.1f}s "
            "in KVPoll.WaitingForInput; no KV chunk was ever sent."
        )
        self.conclude_state = KVPoll.Failed
        return KVPoll.Failed

    def get_transfer_metric(self) -> KVTransferMetric:
        return KVTransferMetric()

    def init(
        self,
        num_kv_indices: int,
        aux_index: Optional[int] = None,
    ):
        self.inited = True
        logger.debug(
            f"FakeKVSender init with num_kv_indices: {num_kv_indices}, aux_index: {aux_index}"
        )

    def should_send_kv_chunk(self, num_pages: int, last_chunk: bool) -> bool:
        # A zero-page last chunk must still send: poll() only concludes after send().
        return num_pages > 0 or last_chunk

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
        num_kv_tokens: Optional[int] = None,
    ):
        self.has_sent = True
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
        self.kv_mgr = mgr
        self.abort_notified: bool = False
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
            # No deadline needed here, unlike FakeKVSender: send_metadata() is
            # unconditional once the decode side preallocates, and waiting for
            # KV space is not a stalled transfer.
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
