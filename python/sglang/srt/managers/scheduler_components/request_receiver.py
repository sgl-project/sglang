from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

import zmq

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.server_args import ServerArgs


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerRequestReceiver:
    recv_from_tokenizer: zmq.Socket
    recv_from_rpc: Optional[zmq.Socket]
    recv_skipper: Any
    input_blocker: Any
    mm_receiver: Any
    ps: "ParallelState"
    tp_group: Any
    tp_cpu_group: Any
    attn_tp_group: Any
    attn_tp_cpu_group: Any
    attn_cp_group: Any
    attn_cp_cpu_group: Any
    world_group: Any
    server_args: "ServerArgs"
    model_config: "ModelConfig"
    max_recv_per_poll: int
    stream_output: Callable[..., None]
    get_last_forward_mode: Callable[[], Any]
