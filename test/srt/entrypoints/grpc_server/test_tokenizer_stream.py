import hashlib
import io
import zipfile
from pathlib import Path

import pytest

try:
    import sglang.srt.entrypoints.grpc_server as grpc_server
except Exception as exc:  # pragma: no cover - optional dependency guard
    pytest.skip(
        f"gRPC server dependencies not available ({exc})",
        allow_module_level=True,
    )

SGLangSchedulerServicer = grpc_server.SGLangSchedulerServicer
sglang_scheduler_pb2 = grpc_server.sglang_scheduler_pb2
from sglang.srt.server_args import ServerArgs


class DummyRequestManager:
    def auto_create_handle_loop(self) -> None:
        pass

    def get_server_info(self) -> dict:
        return {
            "active_requests": 0,
            "paused": False,
            "last_receive_time": 0.0,
        }


class DummyContext:
    def __init__(self) -> None:
        self.code = None
        self.details = None

    def set_code(self, code) -> None:  # pragma: no cover - simple setter
        self.code = code

    def set_details(self, details: str) -> None:  # pragma: no cover - simple setter
        self.details = details


@pytest.mark.asyncio
async def test_get_tokenizer_stream(tmp_path: Path) -> None:
    (tmp_path / "tokenizer.json").write_text(
        '{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],'
        '"model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1},"unk_token":"[UNK]"},'
        '"pre_tokenizer":{"type":"Whitespace"}}',
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text(
        '{"chat_template": "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"}',
        encoding="utf-8",
    )

    request_manager = DummyRequestManager()
    server_args = ServerArgs(model_path=str(tmp_path), tokenizer_path=str(tmp_path))
    servicer = SGLangSchedulerServicer(
        request_manager=request_manager,
        server_args=server_args,
        model_info={},
        scheduler_info={},
    )

    metadata, _, bundle_bytes = servicer._prepare_tokenizer_bundle()
    expected_fingerprint = hashlib.sha256(bundle_bytes).hexdigest()

    received = []
    async for chunk in servicer.GetTokenizer(
        sglang_scheduler_pb2.GetTokenizerRequest(),
        DummyContext(),
    ):
        received.append(chunk)

    assert len(received) >= 2
    assert received[0].WhichOneof("chunk") == "metadata"
    assert received[0].metadata.fingerprint == expected_fingerprint

    archive_bytes = b"".join(
        chunk.file_chunk.data
        for chunk in received
        if chunk.WhichOneof("chunk") == "file_chunk"
    )
    assert hashlib.sha256(archive_bytes).hexdigest() == expected_fingerprint

    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
        names = set(archive.namelist())
    assert "tokenizer.json" in names
    assert "tokenizer_config.json" in names
