from __future__ import annotations  # noqa: F401

from http import HTTPStatus  # noqa: F401
from typing import Any, Callable, List, Optional, Union  # noqa: F401

import zmq  # noqa: F401
from torch.distributed import barrier  # noqa: F401

from sglang.srt.disaggregation.utils import prepare_abort  # noqa: F401
from sglang.srt.managers.io_struct import (  # noqa: F401
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.mm_utils import (  # noqa: F401
    has_shm_features,
    unwrap_shm_features,
)
from sglang.srt.utils import broadcast_pyobj, point_to_point_pyobj  # noqa: F401


class SchedulerRequestReceiver:
    """Wire-level request receiver: pulls ``recv_req`` lists from zmq /
    pipeline upstream, applies recv_skipper / input_blocker guards, broadcasts
    across TP/DP/CP groups, runs MM-receiver pre-processing, and unwraps shm
    features. Owns no mutable state."""

    def __init__(
        self,
        *,
        recv_from_tokenizer,
        recv_from_rpc,
        recv_skipper,
        input_blocker,
        mm_receiver,
        ps,
        tp_group,
        tp_cpu_group,
        attn_tp_group,
        attn_tp_cpu_group,
        attn_cp_group,
        attn_cp_cpu_group,
        world_group,
        server_args,
        model_config,
        max_recv_per_poll: int,
        stream_output: Callable[..., None],
    ) -> None:
        self.recv_from_tokenizer = recv_from_tokenizer
        self.recv_from_rpc = recv_from_rpc
        self.recv_skipper = recv_skipper
        self.input_blocker = input_blocker
        self.mm_receiver = mm_receiver
        self.ps = ps
        self.tp_group = tp_group
        self.tp_cpu_group = tp_cpu_group
        self.attn_tp_group = attn_tp_group
        self.attn_tp_cpu_group = attn_tp_cpu_group
        self.attn_cp_group = attn_cp_group
        self.attn_cp_cpu_group = attn_cp_cpu_group
        self.world_group = world_group
        self.server_args = server_args
        self.model_config = model_config
        self.max_recv_per_poll = max_recv_per_poll
        self.stream_output = stream_output
