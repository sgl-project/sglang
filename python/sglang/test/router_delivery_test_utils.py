from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import zmq
from prometheus_client import CollectorRegistry, Histogram

from sglang.srt.managers.multi_tokenizer_mixin import MultiDetokenizerRouter
from sglang.srt.managers.scheduler_components import ipc_channels


def create_test_channels(
    port_args, *, domain=bytes(range(24)), enabled=True, is_leader=True
):
    with (
        patch.object(ipc_channels.zmq, "Context"),
        patch.object(ipc_channels, "get_zmq_socket", return_value=MagicMock()),
        patch.object(ipc_channels, "read_clock_domain", return_value=domain),
    ):
        return ipc_channels.SchedulerIpcChannels.create(
            port_args=port_args,
            is_rank_zero=is_leader,
            skip_tokenizer_init=False,
            metrics_enabled=enabled,
            enable_scripted_runtime=False,
            detokenizer_router_metrics_enabled=enabled,
        )


def test_port_args(detokenizer_endpoint):
    return SimpleNamespace(
        scheduler_input_ipc_name="ipc:///scheduler",
        rpc_ipc_name="ipc:///rpc",
        tokenizer_ipc_name="ipc:///tokenizer",
        detokenizer_ipc_name=detokenizer_endpoint,
        metrics_ipc_name="ipc:///metrics",
    )


@contextmanager
def socket_pair():
    with zmq.Context() as context:
        with context.socket(zmq.PAIR) as sender, context.socket(zmq.PAIR) as receiver:
            for sock in (sender, receiver):
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.RCVTIMEO, 2000)
                sock.setsockopt(zmq.SNDTIMEO, 2000)
            sender.bind("inproc://clock-domain")
            receiver.connect("inproc://clock-domain")
            yield sender, receiver


class RouterFinished(Exception):
    pass


@contextmanager
def router_for_socket(receiver, domain):
    registry = CollectorRegistry()

    def registered_histogram(*args, **kwargs):
        return Histogram(*args, registry=registry, **kwargs)

    with (
        patch("prometheus_client.Histogram", side_effect=registered_histogram),
        patch("sglang.srt.managers.multi_tokenizer_mixin.zmq.Context"),
        patch(
            "sglang.srt.managers.multi_tokenizer_mixin.get_zmq_socket",
            return_value=receiver,
        ),
        patch(
            "sglang.srt.managers.multi_tokenizer_mixin.read_clock_domain",
            return_value=domain,
        ),
    ):
        router = MultiDetokenizerRouter(
            ["worker-a", "worker-b"],
            SimpleNamespace(detokenizer_ipc_name="unused"),
            enable_metrics=True,
        )
        yield router, registry


def collect_routed_messages(router, count):
    routed = []

    def record(_endpoint, message):
        routed.append(message)
        if len(routed) == count:
            raise RouterFinished

    router._send = record
    return routed
