import asyncio
import importlib
import unittest
from types import SimpleNamespace
from typing import Any, Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=2, suite="stage-b-test-1-gpu-small")

CURVE_PUBLIC_KEY_METADATA = "x-sglang-curve-public-key"


def _load_torch() -> Optional[Any]:
    try:
        return cast(Any, importlib.import_module("torch"))
    except ModuleNotFoundError:
        return None


_torch = _load_torch()
GPU_AVAILABLE = bool(_torch is not None and _torch.cuda.is_available())


class DummyGrpcContext:
    def __init__(self, metadata=None):
        self._metadata = metadata or []
        self.code = None
        self.details = None

    def invocation_metadata(self):
        return self._metadata

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


@unittest.skipUnless(GPU_AVAILABLE, "GPU required for encode curve plumbing tests.")
class TestGrpcCurveMetadataClientHelpers(CustomTestCase):
    @patch("grpc.insecure_channel")
    @patch("smg_grpc_proto.sglang_encoder_pb2_grpc.SglangEncoderStub")
    def test_grpc_encode_request_forwards_curve_metadata(
        self, mock_stub_cls, mock_insecure_channel
    ):
        from sglang.srt.disaggregation.encode_receiver import _grpc_encode_request

        stub = MagicMock()
        stub_cls = mock_stub_cls.return_value = stub
        stub_cls.Encode.return_value = object()
        mock_insecure_channel.return_value = MagicMock()

        _grpc_encode_request(
            "127.0.0.1:50051",
            {
                "mm_items": ["image.png"],
                "req_id": "req-1",
                "num_parts": 1,
                "part_idx": 0,
                "prefill_host": "127.0.0.1",
                "embedding_port": 31000,
                "curve_public_key": "A" * 40,
            },
        )

        self.assertEqual(
            stub.Encode.call_args.kwargs["metadata"],
            ((CURVE_PUBLIC_KEY_METADATA, "A" * 40),),
        )

    @patch("grpc.insecure_channel")
    @patch("smg_grpc_proto.sglang_encoder_pb2_grpc.SglangEncoderStub")
    def test_grpc_send_request_forwards_curve_metadata(
        self, mock_stub_cls, mock_insecure_channel
    ):
        from sglang.srt.disaggregation.encode_receiver import _grpc_send_request

        stub = MagicMock()
        stub_cls = mock_stub_cls.return_value = stub
        mock_insecure_channel.return_value = MagicMock()

        _grpc_send_request(
            "127.0.0.1:50051",
            {
                "req_id": "req-2",
                "prefill_host": "127.0.0.1",
                "embedding_port": 32000,
                "session_id": "session-1",
                "buffer_address": 12345,
                "curve_public_key": "B" * 40,
            },
        )

        self.assertEqual(
            stub.Send.call_args.kwargs["metadata"],
            ((CURVE_PUBLIC_KEY_METADATA, "B" * 40),),
        )

    @patch("grpc.insecure_channel")
    @patch("smg_grpc_proto.sglang_encoder_pb2_grpc.SglangEncoderStub")
    def test_grpc_scheduler_receive_url_forwards_curve_metadata(
        self, mock_stub_cls, mock_insecure_channel
    ):
        from sglang.srt.disaggregation.encode_receiver import (
            _grpc_scheduler_receive_url,
        )

        stub = MagicMock()
        stub_cls = mock_stub_cls.return_value = stub
        mock_insecure_channel.return_value = MagicMock()

        _grpc_scheduler_receive_url(
            "127.0.0.1:50051",
            "req-3",
            "127.0.0.1:33000",
            2,
            curve_public_key="C" * 40,
        )

        self.assertEqual(
            stub.SchedulerReceiveUrl.call_args.kwargs["metadata"],
            ((CURVE_PUBLIC_KEY_METADATA, "C" * 40),),
        )


@unittest.skipUnless(GPU_AVAILABLE, "GPU required for encode curve plumbing tests.")
class TestGrpcCurveMetadataServerHandlers(CustomTestCase):
    def _make_server(self, *, backend: str):
        from sglang.srt.disaggregation.encode_grpc_server import SGLangEncoderServer

        send_mock = AsyncMock()
        encoder = cast(
            Any,
            SimpleNamespace(
                encode=AsyncMock(return_value=(0, 0, 0, None, None)),
                send=send_mock,
                embedding_to_send={},
            ),
        )
        server_args = cast(Any, SimpleNamespace(encoder_transfer_backend=backend))
        return (
            SGLangEncoderServer(
                encoder=encoder,
                send_sockets=[],
                server_args=server_args,
            ),
            send_mock,
        )

    def test_encode_handler_forwards_curve_metadata_to_direct_send(self):
        server, send_mock = self._make_server(backend="zmq_to_tokenizer")
        request = SimpleNamespace(
            mm_items=["image.png"],
            req_id="req-4",
            num_parts=1,
            part_idx=0,
            prefill_host="127.0.0.1",
            embedding_port=[34000],
        )
        context = DummyGrpcContext(metadata=[(CURVE_PUBLIC_KEY_METADATA, "D" * 40)])

        asyncio.run(server.Encode(request, context))

        send_mock.assert_awaited_once_with(
            req_id="req-4",
            prefill_host="127.0.0.1",
            embedding_port=34000,
            curve_public_key="D" * 40,
        )

    def test_send_handler_forwards_curve_metadata(self):
        server, send_mock = self._make_server(backend="mooncake")
        request = SimpleNamespace(
            req_id="req-5",
            prefill_host="127.0.0.1",
            embedding_port=35000,
            session_id="session-2",
            buffer_address=777,
        )
        context = DummyGrpcContext(metadata=[(CURVE_PUBLIC_KEY_METADATA, "E" * 40)])

        asyncio.run(server.Send(request, context))

        send_mock.assert_awaited_once_with(
            req_id="req-5",
            prefill_host="127.0.0.1",
            embedding_port=35000,
            session_id="session-2",
            buffer_address=777,
            curve_public_key="E" * 40,
        )

    def test_scheduler_receive_url_handler_forwards_curve_metadata(self):
        from sglang.srt.disaggregation.encode_grpc_server import SGLangEncoderServer

        encoder = cast(Any, SimpleNamespace())
        server_args = cast(
            Any, SimpleNamespace(encoder_transfer_backend="zmq_to_scheduler")
        )
        server = SGLangEncoderServer(
            encoder=encoder,
            send_sockets=[],
            server_args=server_args,
        )
        request = SimpleNamespace(
            req_id="req-6",
            receive_count=3,
            receive_url="127.0.0.1:36000",
        )
        context = DummyGrpcContext(metadata=[(CURVE_PUBLIC_KEY_METADATA, "F" * 40)])

        with patch(
            "sglang.srt.disaggregation.encode_grpc_server.handle_scheduler_receive_url_request",
            new=AsyncMock(),
        ) as mock_handler:
            asyncio.run(server.SchedulerReceiveUrl(request, context))

        mock_handler.assert_awaited_once_with(
            {
                "req_id": "req-6",
                "receive_count": 3,
                "receive_url": "127.0.0.1:36000",
                "curve_public_key": "F" * 40,
            }
        )


@unittest.skipUnless(GPU_AVAILABLE, "GPU required for encode curve plumbing tests.")
class TestHttpEncodeErrorCurveForwarding(CustomTestCase):
    def test_handle_encode_request_forwards_curve_metadata_on_error(self):
        import sglang.srt.disaggregation.encode_server as encode_server

        send_mock = AsyncMock()
        fake_encoder = SimpleNamespace(
            server_args=SimpleNamespace(encoder_transfer_backend="zmq_to_scheduler"),
            mm_global_cache=None,
            encode=AsyncMock(return_value=(0, 0, 0, "boom", 500)),
            send=send_mock,
            embedding_to_send={},
            background_tasks=set(),
        )
        request = {
            "mm_items": ["image.png"],
            "req_id": "req-7",
            "num_parts": 1,
            "part_idx": 0,
            "modality": "IMAGE",
            "prefill_host": "127.0.0.1",
            "embedding_port": [37000],
            "curve_public_key": "G" * 40,
        }

        with patch.object(encode_server, "encoder", fake_encoder), patch.object(
            encode_server, "send_sockets", []
        ):
            asyncio.run(encode_server.handle_encode_request(request))

        send_mock.assert_awaited_once_with(
            req_id="req-7",
            prefill_host="127.0.0.1",
            embedding_port=37000,
            curve_public_key="G" * 40,
        )


if __name__ == "__main__":
    unittest.main()
