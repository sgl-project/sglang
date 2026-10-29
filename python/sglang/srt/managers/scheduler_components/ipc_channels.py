import ipaddress
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union
from urllib.parse import urlsplit

import zmq

from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
from sglang.srt.server_args import PortArgs
from sglang.srt.utils.network import get_zmq_socket

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
        ScriptedTokenizerRecvProxy,
    )


logger = logging.getLogger(__name__)


def _has_local_clock_domain(endpoint: str) -> bool:
    """Conservatively identify endpoints that cannot route to another host."""
    if endpoint.startswith(("ipc://", "inproc://")):
        return True
    # In libzmq's source;destination syntax, URL parsing sees the source.
    if ";" in endpoint:
        return False
    try:
        parsed = urlsplit(endpoint)
        return (
            parsed.scheme == "tcp" and ipaddress.ip_address(parsed.hostname).is_loopback
        )
    except ValueError:
        # Names and nonstandard endpoints do not establish a shared clock.
        return False


@dataclass(frozen=True, slots=True, kw_only=True)
class SchedulerIpcChannels:
    recv_from_tokenizer: Union[zmq.Socket, "ScriptedTokenizerRecvProxy"]
    recv_from_rpc: Optional[zmq.Socket]
    send_to_tokenizer: SenderWrapper
    send_to_detokenizer: SenderWrapper
    send_metrics_from_scheduler: Optional[zmq.Socket]

    @classmethod
    def create(
        cls,
        *,
        port_args: PortArgs,
        is_rank_zero: bool,
        skip_tokenizer_init: bool,
        metrics_enabled: bool,
        enable_scripted_runtime: bool,
        detokenizer_router_metrics_enabled: bool = False,
    ) -> "SchedulerIpcChannels":
        context = zmq.Context(2)

        if is_rank_zero:
            recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )
            if enable_scripted_runtime:
                from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
                    ScriptedTokenizerRecvProxy,
                )

                recv_from_tokenizer = ScriptedTokenizerRecvProxy(
                    underlying=recv_from_tokenizer
                )
            recv_from_rpc = get_zmq_socket(
                context, zmq.DEALER, port_args.rpc_ipc_name, False
            )

            send_to_tokenizer_raw = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )
            if skip_tokenizer_init:
                # No decode work: send outputs straight to the tokenizer side
                # (MultiTokenizerRouter fans out when tokenizer_worker_num > 1).
                send_to_detokenizer_raw = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name, False
                )
            else:
                # Send to the DetokenizerManager
                send_to_detokenizer_raw = get_zmq_socket(
                    context, zmq.PUSH, port_args.detokenizer_ipc_name, False
                )

            send_to_tokenizer = SenderWrapper(send_to_tokenizer_raw)
            timestamped = detokenizer_router_metrics_enabled and not skip_tokenizer_init
            if timestamped and not _has_local_clock_domain(
                port_args.detokenizer_ipc_name
            ):
                logger.warning(
                    "Scheduler-to-detokenizer-router delivery latency is not "
                    "recorded for endpoint %s: monotonic clock domains may differ.",
                    port_args.detokenizer_ipc_name,
                )
                timestamped = False
            send_to_detokenizer = SenderWrapper(
                send_to_detokenizer_raw, timestamped=timestamped
            )
        else:
            recv_from_tokenizer = None
            recv_from_rpc = None
            send_to_tokenizer = SenderWrapper(None)
            send_to_detokenizer = SenderWrapper(None)

        if metrics_enabled:
            send_metrics_from_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.metrics_ipc_name, False
            )
        else:
            send_metrics_from_scheduler = None

        return cls(
            recv_from_tokenizer=recv_from_tokenizer,
            recv_from_rpc=recv_from_rpc,
            send_to_tokenizer=send_to_tokenizer,
            send_to_detokenizer=send_to_detokenizer,
            send_metrics_from_scheduler=send_metrics_from_scheduler,
        )
