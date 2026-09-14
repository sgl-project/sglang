import unittest
from dataclasses import dataclass
from unittest.mock import patch

import zmq

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.ipc_channels import (
    SchedulerIpcChannels,
)
from sglang.srt.server_args import PortArgs

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


@dataclass(frozen=True)
class _FakeSocket:
    index: int


@dataclass(frozen=True)
class _SocketCreation:
    context: object
    socket_type: int
    endpoint: str
    bind: bool
    socket: _FakeSocket


class _SocketFactory:
    def __init__(self):
        self.creations: list[_SocketCreation] = []

    def __call__(self, context, socket_type, endpoint, bind):
        socket = _FakeSocket(index=len(self.creations))
        self.creations.append(
            _SocketCreation(
                context=context,
                socket_type=socket_type,
                endpoint=endpoint,
                bind=bind,
                socket=socket,
            )
        )
        return socket


class TestSchedulerIpcChannels(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.port_args = PortArgs(
            tokenizer_ipc_name="ipc:///tokenizer",
            scheduler_input_ipc_name="ipc:///scheduler-input",
            detokenizer_ipc_name="ipc:///detokenizer",
            nccl_port=12345,
            rpc_ipc_name="ipc:///rpc",
            metrics_ipc_name="ipc:///metrics",
            tokenizer_worker_ipc_name=None,
            decoupled_spec_ipc_config=None,
        )

    def _create_channels(
        self,
        *,
        is_rank_zero=True,
        skip_tokenizer_init=False,
        metrics_enabled=False,
        enable_scripted_runtime=False,
    ):
        context = object()
        socket_factory = _SocketFactory()

        with (
            patch(
                "sglang.srt.managers.scheduler_components.ipc_channels.zmq.Context",
                return_value=context,
            ) as mock_context,
            patch(
                "sglang.srt.managers.scheduler_components.ipc_channels.get_zmq_socket",
                side_effect=socket_factory,
            ),
        ):
            channels = SchedulerIpcChannels.create(
                port_args=self.port_args,
                is_rank_zero=is_rank_zero,
                skip_tokenizer_init=skip_tokenizer_init,
                metrics_enabled=metrics_enabled,
                enable_scripted_runtime=enable_scripted_runtime,
            )

        mock_context.assert_called_once_with(2)
        self.assertTrue(
            all(creation.context is context for creation in socket_factory.creations)
        )
        return channels, socket_factory.creations

    def assertSocketTopology(self, creations, expected):
        self.assertEqual(
            [
                (creation.socket_type, creation.endpoint, creation.bind)
                for creation in creations
            ],
            expected,
        )

    def test_rank_zero_connects_standard_scheduler_channels(self):
        channels, creations = self._create_channels()

        self.assertSocketTopology(
            creations,
            [
                (zmq.PULL, self.port_args.scheduler_input_ipc_name, False),
                (zmq.DEALER, self.port_args.rpc_ipc_name, False),
                (zmq.PUSH, self.port_args.tokenizer_ipc_name, False),
                (zmq.PUSH, self.port_args.detokenizer_ipc_name, False),
            ],
        )
        self.assertIs(channels.recv_from_tokenizer, creations[0].socket)
        self.assertIs(channels.recv_from_rpc, creations[1].socket)
        self.assertIs(channels.send_to_tokenizer.socket, creations[2].socket)
        self.assertIs(channels.send_to_detokenizer.socket, creations[3].socket)
        self.assertIsNone(channels.send_metrics_from_scheduler)

    def test_skip_tokenizer_init_routes_output_to_tokenizer(self):
        channels, creations = self._create_channels(skip_tokenizer_init=True)

        self.assertSocketTopology(
            creations,
            [
                (zmq.PULL, self.port_args.scheduler_input_ipc_name, False),
                (zmq.DEALER, self.port_args.rpc_ipc_name, False),
                (zmq.PUSH, self.port_args.tokenizer_ipc_name, False),
                (zmq.PUSH, self.port_args.tokenizer_ipc_name, False),
            ],
        )
        self.assertIs(channels.send_to_tokenizer.socket, creations[2].socket)
        self.assertIs(channels.send_to_detokenizer.socket, creations[3].socket)
        self.assertIsNot(
            channels.send_to_tokenizer.socket, channels.send_to_detokenizer.socket
        )

    @patch(
        "sglang.test.scripted_runtime.tokenizer_recv_proxy."
        "ScriptedTokenizerRecvProxy"
    )
    def test_scripted_runtime_wraps_tokenizer_receiver(self, mock_proxy_type):
        proxy = object()
        mock_proxy_type.return_value = proxy

        channels, creations = self._create_channels(enable_scripted_runtime=True)

        mock_proxy_type.assert_called_once_with(underlying=creations[0].socket)
        self.assertIs(channels.recv_from_tokenizer, proxy)
        self.assertIs(channels.recv_from_rpc, creations[1].socket)

    def test_non_rank_zero_disables_regular_channels(self):
        channels, creations = self._create_channels(
            is_rank_zero=False,
            enable_scripted_runtime=True,
        )

        self.assertEqual(creations, [])
        self.assertIsNone(channels.recv_from_tokenizer)
        self.assertIsNone(channels.recv_from_rpc)
        self.assertIsNone(channels.send_to_tokenizer.socket)
        self.assertIsNone(channels.send_to_detokenizer.socket)
        self.assertIsNone(channels.send_metrics_from_scheduler)

    def test_metrics_channel_is_independent_of_rank_zero(self):
        channels, creations = self._create_channels(
            is_rank_zero=False,
            metrics_enabled=True,
        )

        self.assertSocketTopology(
            creations,
            [(zmq.PUSH, self.port_args.metrics_ipc_name, False)],
        )
        self.assertIs(channels.send_metrics_from_scheduler, creations[0].socket)
        self.assertIsNone(channels.recv_from_tokenizer)
        self.assertIsNone(channels.recv_from_rpc)
        self.assertIsNone(channels.send_to_tokenizer.socket)
        self.assertIsNone(channels.send_to_detokenizer.socket)


if __name__ == "__main__":
    unittest.main()
