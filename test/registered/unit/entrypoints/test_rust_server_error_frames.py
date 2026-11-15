"""Python's native `/generate` error bodies, recorded for the Rust server.

`build_rust_error_frames_fixture` runs the Python handler (`serve_generate_request`)
on requests whose generation raises, unary and streaming, and records what a
client reads. `rust/sglang-server/testdata/error_frames.json` holds the result;
the Rust server's tests (`native_api.rs`) answer the same failures and compare.
"""

import asyncio
import json
import os
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

from sglang.srt.entrypoints import http_server
from sglang.srt.managers.io_struct import GenerateReqInput

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

RUST_ERROR_FRAMES_FIXTURE = os.path.join(
    os.path.dirname(__file__),
    *[os.pardir] * 4,
    "rust/sglang-server/testdata/error_frames.json",
)


class _StatusValueError(ValueError):
    """A `ValueError` raised with its own HTTP status, as a server raises one
    for a request it cannot serve right now."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _Request:
    headers: dict = {}

    async def is_disconnected(self) -> bool:
        return False


def _tokenizer_manager(error: Exception) -> SimpleNamespace:
    async def generate_request(obj, request):
        raise error
        yield  # an async generator, as the tokenizer manager's is

    return SimpleNamespace(
        generate_request=generate_request, create_abort_task=lambda obj: None
    )


async def _answer(error: Exception, stream: bool) -> dict:
    http_server.set_global_state(
        SimpleNamespace(tokenizer_manager=_tokenizer_manager(error))
    )
    response = await http_server.serve_generate_request(
        GenerateReqInput(text="hi", stream=stream), _Request()
    )
    if not stream:
        return {"status": response.status_code, "body": json.loads(response.body)}
    data = [
        chunk.decode() if isinstance(chunk, bytes) else chunk
        async for chunk in response.body_iterator
    ]
    frames = "".join(data).split("\n\n")
    assert frames[-2:] == ["data: [DONE]", ""], frames
    (frame,) = frames[:-2]
    return {"status": response.status_code, "frame": json.loads(frame[len("data: ") :])}


def build_rust_error_frames_fixture() -> dict:
    """Regenerate the fixture with ``json.dump(build_rust_error_frames_fixture(),
    f, indent=2)`` plus a trailing newline."""
    cases = {
        "invalid_argument": ValueError("bad sampling"),
        "status_503": _StatusValueError("the encoder returned no embeddings", 503),
    }
    saved = http_server._global_state
    try:
        return {
            name: {
                "message": str(error),
                "unary": asyncio.run(_answer(error, stream=False)),
                "stream": asyncio.run(_answer(error, stream=True)),
            }
            for name, error in cases.items()
        }
    finally:
        http_server._global_state = saved


class TestRustServerErrorFrames(CustomTestCase):
    def test_fixture_matches_python(self):
        with open(RUST_ERROR_FRAMES_FIXTURE) as f:
            self.assertEqual(json.load(f), build_rust_error_frames_fixture())


if __name__ == "__main__":
    unittest.main()
