import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.base.conn import (
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVArgs,
    KVPoll,
)

logger = logging.getLogger(__name__)


# For warmup reqs, we don't kv transfer, we use the fake sender and receiver
class FakeKVSender(BaseKVSender):
    def __init__(self, mgr: BaseKVManager, bootstrap_addr: str, bootstrap_room: int):
        self.has_sent = False

    def poll(self) -> KVPoll:
        if self.has_sent is False:
            # Assume handshake completed instantly
            return KVPoll.WaitingForInput
        else:
            # Assume transfer completed instantly
            logger.info("FakeKVSender poll success")
            return KVPoll.Success

    def init(
        self,
        kv_indices: list[int],
        aux_index: Optional[int] = None,
        dest_ranks: Optional[list[int]] = None,
    ):
        logger.info(
            f"FakeKVSender init with kv_indices: {kv_indices}, aux_index: {aux_index}, dest_ranks: {dest_ranks}"
        )
        pass

    def send(
        self,
        kv_indices: npt.NDArray[np.int64],
        index_slice: slice,
        is_last: bool,
    ):
        logger.info(
            f"FakeKVSender send with kv_indices: {kv_indices}, index_slice: {index_slice}, is_last: {is_last}"
        )
        if is_last:
            self.has_sent = True
            logger.info(f"FakeKVSender send success")
        else:
            self.has_sent = False
            logger.info(f"FakeKVSender send fake transferring")

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class FakeKVReceiver(BaseKVReceiver):
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.has_init = False

    def poll(self) -> KVPoll:
        if self.has_init is False:
            # Assume handshake completed instantly
            return KVPoll.WaitingForInput
        else:
            # Assume transfer completed instantly
            logger.info("FakeKVReceiver poll success")
            return KVPoll.Success

    def init(self, kv_indices: list[int], aux_index: Optional[int] = None):
        self.has_init = True
        logger.info(
            f"FakeKVReceiver init with kv_indices: {kv_indices}, aux_index: {aux_index}"
        )

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")
