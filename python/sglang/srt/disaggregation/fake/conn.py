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
        self.has_sent = False

    def poll(self) -> KVPoll:
        if self.has_sent is False:
            # Assume handshake completed instantly
            return KVPoll.WaitingForInput
        else:
            # Assume transfer completed instantly
            logger.debug("FakeKVSender poll success")
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
        self.has_sent = True
        logger.debug(
            f"FakeKVSender send with kv_indices: {kv_indices}, state_indices: {state_indices}"
        )

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


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

    def poll(self) -> KVPoll:
        if not self.bootstrap_done:
            return KVPoll.Bootstrapping
        if not self.has_sent_metadata:
            return KVPoll.WaitingForInput
        logger.debug("FakeKVReceiver poll success")
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
