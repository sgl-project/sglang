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
    ):
        self.kv_mgr = mgr
        self.inited = False
        self.num_pages = 0
        self.curr_idx = 0
        self.conclude_state: Optional[KVPoll] = None

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        # Stay WaitingForInput until init() runs (pop_bootstrapped polls before
        # init and rejects any other state). Once inited, conclude only after the
        # tail chunk lands (curr_idx >= num_pages); chunked prefill calls send()
        # several times, so concluding on the first send pops the req before its
        # remaining KV is released -> pool memory leak. A 0-page request (full
        # cache hit, no send) concludes immediately.
        if not self.inited or self.curr_idx < self.num_pages:
            return KVPoll.WaitingForInput

        logger.debug("FakeKVSender poll success")
        self.conclude_state = KVPoll.Success
        return KVPoll.Success

    def get_transfer_metric(self) -> KVTransferMetric:
        return KVTransferMetric()

    def init(
        self,
        num_pages: int,
        aux_index: Optional[int] = None,
    ):
        self.inited = True
        self.num_pages = num_pages
        logger.debug(
            f"FakeKVSender init with num_pages: {num_pages}, aux_index: {aux_index}"
        )

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
    ):
        self.curr_idx += len(kv_indices)
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
