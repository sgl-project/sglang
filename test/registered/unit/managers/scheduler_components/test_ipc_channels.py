"""Which scheduler ranks connect to the engine's RPC socket."""

import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.server_args import PortArgs  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_RPC_IPC_NAME = "ipc://rpc"


def _port_args() -> PortArgs:
    return PortArgs(
        tokenizer_ipc_name="ipc://tokenizer",
        scheduler_input_ipc_name="ipc://scheduler-input",
        detokenizer_ipc_name="ipc://detokenizer",
        nccl_port=0,
        rpc_ipc_name=_RPC_IPC_NAME,
        metrics_ipc_name="ipc://metrics",
        tokenizer_worker_ipc_name=None,
        decoupled_spec_ipc_config=None,
    )


def _rpc_peer_ranks(*, tp_size: int, attn_dp_size: int) -> list[int]:
    """TP ranks whose scheduler connects to the engine's RPC socket."""
    attn_tp_size = tp_size // attn_dp_size
    peers = []
    for tp_rank in range(tp_size):
        endpoints = []

        def fake_zmq_socket(_context, _socket_type, endpoint, _bind):
            endpoints.append(endpoint)
            return MagicMock()

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.skip_tokenizer_init = False
        with (
            published_topology(
                role="scheduler",
                ranks={"world_rank": tp_rank, "dp_rank": tp_rank // attn_tp_size},
                tp_size=tp_size,
                attn_dp_size=attn_dp_size,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.ipc_channels.get_zmq_socket",
                side_effect=fake_zmq_socket,
            ),
            patch("sglang.srt.managers.scheduler.create_load_snapshot_writer"),
        ):
            scheduler.init_ipc_channels(_port_args())
        if _RPC_IPC_NAME in endpoints:
            peers.append(tp_rank)
    return peers


class TestEngineRpcPeers(CustomTestCase):
    def test_only_the_control_broadcast_source_connects_under_dp_attention(self):
        """An engine RPC must reach every rank and be answered once: with a
        second DP group leader on the socket, RPCs routed to it are dropped
        and the caller blocks forever."""
        self.assertEqual(_rpc_peer_ranks(tp_size=4, attn_dp_size=2), [0])


if __name__ == "__main__":
    unittest.main()
