import logging
from typing import List, Optional, Tuple

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
        kv_indices: Optional[list[int]] = None,
        aux_index: Optional[int] = None,
        embedding_indices: Optional[int] = None,
    ):
        logger.debug(
            f"FakeKVSender init with kv_indices: {kv_indices}, aux_index: {aux_index}, embedding_indices: {embedding_indices}"
        )
        pass

    def send(
        self,
        kv_indices: Optional[npt.NDArray[np.int32]] = None,
        embedding_index: Optional[int] = None,
    ):
        self.has_sent = True
        logger.debug(f"FakeKVSender send with kv_indices: {kv_indices}")

    def send_embedding(
        self,
        embedding_indices: List[int] = None,
        total_tokens: int = None,
        block_size: int = None,
        last_chunk: bool = False,
    ):
        self.has_sent = True
        logger.debug(
            f"FakeKVSender send_embedding with embedding_indices: {embedding_indices}, total_tokens: {total_tokens}, block_size: {block_size}, last_chunk: {last_chunk}"
        )

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class FakeKVReceiver(BaseKVReceiver):
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        prefill_dp_rank: Optional[int] = None,
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

    def init(
        self,
        kv_indices: Optional[list[int]] = None,
        aux_index: Optional[int] = None,
        embedding_indices: Optional[list[int]] = None,
        allocated_tokens: Optional[int] = None,
    ):
        self.has_init = True
        logger.debug(
            f"FakeKVReceiver init with kv_indices: {kv_indices}, aux_index: {aux_index}, embedding_indices: {embedding_indices}, allocated_tokens: {allocated_tokens}"
        )

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")
