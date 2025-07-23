import logging
from typing import List, Optional

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.base.conn import (
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVPoll,
)

logger = logging.getLogger(__name__)


# For warmup reqs, we don't kv transfer, we use the fake sender and receiver
class FakeKVSender(BaseKVSender):
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        self.has_sent = False

    def poll(self) -> KVPoll:
        if self.has_sent is False:
            # Assume handshake completed instantly
            return KVPoll.WaitingForInput
        else:
            # Assume transfer completed instantly
            logger.debug("FakeKVSender poll success")
            return KVPoll.Success

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
    ):
        self.has_sent = True
        logger.debug(f"FakeKVSender send with kv_indices: {kv_indices}")

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class FakeKVReceiver(BaseKVReceiver):
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        data_parallel_rank: Optional[int] = None,
    ):
        self.has_init = False

    def poll(self) -> KVPoll:
        if self.has_init is False:
            # Assume handshake completed instantly
            return KVPoll.WaitingForInput
        else:
            # Assume transfer completed instantly
            logger.debug("FakeKVReceiver poll success")
            return KVPoll.Success

    def init(self, kv_indices: list[int], aux_index: Optional[int] = None):
        self.has_init = True
        logger.debug(
            f"FakeKVReceiver init with kv_indices: {kv_indices}, aux_index: {aux_index}"
        )

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")
