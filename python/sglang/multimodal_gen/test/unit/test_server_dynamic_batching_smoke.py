from __future__ import annotations

import base64
import threading
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from sglang.multimodal_gen.test.server.test_server_common import DiffusionServerBase
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
)


def _noop() -> None:
    """Provide a typed no-op callback for fake clients."""


def _image_b64(color: tuple[int, int, int], image_format: str = "PNG") -> str:
    """Encode one 64x64 RGB image for a fake Images API response."""
    buffer = BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _dynamic_case() -> DiffusionTestCase:
    """Build a small two-request dynamic batching test case."""
    common_extras = {
        "num_inference_steps": 2,
        "guidance_scale": 4.0,
        "generator_device": "cpu",
    }
    return DiffusionTestCase(
        "bagel_t2i",
        DiffusionServerArgs(model_path="test-bagel", modality="image"),
        DiffusionSamplingParams(prompt="main", output_size="64x64"),
        dynamic_batching_requests=(
            DiffusionSamplingParams(
                prompt="prompt-a",
                output_size="64x64",
                output_format="png",
                extras={**common_extras, "seed": 11},
            ),
            DiffusionSamplingParams(
                prompt="a longer prompt-b",
                output_size="64x64",
                output_format="png",
                extras={**common_extras, "seed": 22},
            ),
        ),
    )


def _context(log_path: Path) -> SimpleNamespace:
    """Build the server context surface used by the batching smoke."""
    return SimpleNamespace(
        stdout_file=log_path,
        process=SimpleNamespace(poll=lambda: None),
        log_tail=lambda: log_path.read_text(encoding="utf-8"),
    )


def _client(
    response_id: str,
    image_b64: str,
    on_generate: Callable[[], None] = _noop,
    on_close: Callable[[], None] = _noop,
) -> tuple[SimpleNamespace, Mock, Mock]:
    """Build an independent fake Images API client with observable cleanup."""
    parsed = SimpleNamespace(
        id=response_id,
        data=[SimpleNamespace(b64_json=image_b64)],
    )

    def generate(**_kwargs: object) -> SimpleNamespace:
        on_generate()
        return SimpleNamespace(parse=Mock(return_value=parsed))

    generate_mock = Mock(side_effect=generate)
    close_mock = Mock(side_effect=on_close)
    client = SimpleNamespace(
        images=SimpleNamespace(
            with_raw_response=SimpleNamespace(generate=generate_mock)
        ),
        close=close_mock,
    )
    return client, generate_mock, close_mock


def _append_merged_markers(log_path: Path) -> None:
    """Append the scheduler markers required by the server smoke."""
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(
            "Dynamic batch dispatch: size=2/2, user_max=2\n"
            "Processed dynamic batch of 2/2 request(s)\n"
        )


def _four_clients(
    reference_images: tuple[str, str],
    batched_images: tuple[str, str],
    on_batch_generate: Callable[[], None],
) -> tuple[list[SimpleNamespace], list[Mock], list[Mock]]:
    """Build two singleton-reference clients followed by two batch clients."""
    clients: list[SimpleNamespace] = []
    generate_mocks: list[Mock] = []
    close_mocks: list[Mock] = []
    for index, image_b64 in enumerate((*reference_images, *batched_images)):
        callback = _noop if index < 2 else on_batch_generate
        client, generate_mock, close_mock = _client(
            f"request-{index}", image_b64, callback
        )
        clients.append(client)
        generate_mocks.append(generate_mock)
        close_mocks.append(close_mock)
    return clients, generate_mocks, close_mocks


def test_dynamic_batching_smoke_requires_merged_execution_markers(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text("old server output\n", encoding="utf-8")
    lock = threading.Lock()
    batch_call_count = 0

    def record_batch_generate() -> None:
        nonlocal batch_call_count
        with lock:
            batch_call_count += 1
            if batch_call_count == 2:
                _append_merged_markers(log_path)

    red = _image_b64((255, 0, 0))
    blue = _image_b64((0, 0, 255))
    clients, generate_mocks, close_mocks = _four_clients(
        (red, blue), (red, blue), record_batch_generate
    )
    server = DiffusionServerBase()
    server._client = Mock(side_effect=clients)

    server._test_dynamic_batching_smoke(_context(log_path), _dynamic_case())

    assert server._client.call_count == 4
    assert all(call.kwargs["timeout"] == 120 for call in server._client.call_args_list)
    for generate_mock, close_mock in zip(generate_mocks, close_mocks):
        generate_mock.assert_called_once()
        close_mock.assert_called_once()
    assert [
        generate_mock.call_args.kwargs["extra_body"]["seed"]
        for generate_mock in generate_mocks
    ] == [11, 22, 11, 22]


def test_dynamic_batching_smoke_rejects_swapped_request_outputs(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text("", encoding="utf-8")
    lock = threading.Lock()
    batch_call_count = 0

    def record_batch_generate() -> None:
        nonlocal batch_call_count
        with lock:
            batch_call_count += 1
            if batch_call_count == 2:
                _append_merged_markers(log_path)

    red = _image_b64((255, 0, 0))
    blue = _image_b64((0, 0, 255))
    clients, _, close_mocks = _four_clients(
        (red, blue), (blue, red), record_batch_generate
    )
    server = DiffusionServerBase()
    server._client = Mock(side_effect=clients)

    with pytest.raises(AssertionError, match="maps to the wrong request"):
        server._test_dynamic_batching_smoke(_context(log_path), _dynamic_case())

    for close_mock in close_mocks:
        close_mock.assert_called_once()


def test_dynamic_batching_smoke_ignores_old_or_dispatch_only_markers(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "server.log"
    _append_merged_markers(log_path)
    lock = threading.Lock()
    batch_call_count = 0

    def record_dispatch_only() -> None:
        nonlocal batch_call_count
        with lock:
            batch_call_count += 1
            if batch_call_count == 2:
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write("Dynamic batch dispatch: size=2/2, user_max=2\n")

    red = _image_b64((255, 0, 0))
    blue = _image_b64((0, 0, 255))
    clients, _, _ = _four_clients((red, blue), (red, blue), record_dispatch_only)
    server = DiffusionServerBase()
    server._client = Mock(side_effect=clients)

    with (
        patch(
            "sglang.multimodal_gen.test.server.test_server_common."
            "_DYNAMIC_BATCH_LOG_TIMEOUT_SECS",
            0.01,
        ),
        pytest.raises(
            pytest.fail.Exception, match="missing dynamic batching server evidence"
        ),
    ):
        server._test_dynamic_batching_smoke(_context(log_path), _dynamic_case())


def test_dynamic_batching_smoke_reads_markers_at_timeout_boundary(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text("", encoding="utf-8")
    lock = threading.Lock()
    batch_call_count = 0
    marker_timer: threading.Timer | None = None

    def schedule_markers() -> None:
        nonlocal batch_call_count, marker_timer
        with lock:
            batch_call_count += 1
            if batch_call_count == 2:
                marker_timer = threading.Timer(
                    0.02, _append_merged_markers, args=(log_path,)
                )
                marker_timer.start()

    red = _image_b64((255, 0, 0))
    blue = _image_b64((0, 0, 255))
    clients, _, _ = _four_clients((red, blue), (red, blue), schedule_markers)
    server = DiffusionServerBase()
    server._client = Mock(side_effect=clients)

    try:
        with patch(
            "sglang.multimodal_gen.test.server.test_server_common."
            "_DYNAMIC_BATCH_LOG_TIMEOUT_SECS",
            0.05,
        ):
            server._test_dynamic_batching_smoke(_context(log_path), _dynamic_case())
    finally:
        if marker_timer is not None:
            marker_timer.join(timeout=1)


def test_dynamic_batching_smoke_rejects_wrong_image_format(tmp_path: Path) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text("", encoding="utf-8")
    jpeg = _image_b64((255, 0, 0), image_format="JPEG")
    png = _image_b64((0, 0, 255))
    clients, _, close_mocks = _four_clients((jpeg, png), (jpeg, png), _noop)
    server = DiffusionServerBase()
    server._client = Mock(side_effect=clients)

    with pytest.raises(AssertionError, match="ignored requested output format 'png'"):
        server._test_dynamic_batching_smoke(_context(log_path), _dynamic_case())

    for close_mock in close_mocks:
        close_mock.assert_called_once()


def test_dynamic_batching_smoke_closes_clients_and_joins_threads_on_failure(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text("", encoding="utf-8")
    red = _image_b64((255, 0, 0))
    blue = _image_b64((0, 0, 255))
    reference_a, _, close_reference_a = _client("reference-a", red)
    reference_b, _, close_reference_b = _client("reference-b", blue)

    blocked_started = threading.Event()
    unblock_request = threading.Event()

    def block_until_closed(**_kwargs: object) -> SimpleNamespace:
        blocked_started.set()
        if not unblock_request.wait(timeout=2):
            raise AssertionError("blocked request was not released by client.close()")
        raise RuntimeError("blocked request cancelled")

    blocked_generate = Mock(side_effect=block_until_closed)
    close_blocked = Mock(side_effect=unblock_request.set)
    blocked_client = SimpleNamespace(
        images=SimpleNamespace(
            with_raw_response=SimpleNamespace(generate=blocked_generate)
        ),
        close=close_blocked,
    )

    def fail_after_peer_starts(**_kwargs: object) -> SimpleNamespace:
        assert blocked_started.wait(timeout=1)
        raise RuntimeError("request failed")

    bad_generate = Mock(side_effect=fail_after_peer_starts)
    close_bad = Mock()
    bad_client = SimpleNamespace(
        images=SimpleNamespace(
            with_raw_response=SimpleNamespace(generate=bad_generate)
        ),
        close=close_bad,
    )
    server = DiffusionServerBase()
    server._client = Mock(
        side_effect=[reference_a, reference_b, blocked_client, bad_client]
    )

    with pytest.raises(RuntimeError, match="request failed"):
        server._test_dynamic_batching_smoke(_context(log_path), _dynamic_case())

    assert not [
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith("dynamic-batching-bagel_t2i-")
    ]
    for close_mock in (
        close_reference_a,
        close_reference_b,
        close_blocked,
        close_bad,
    ):
        close_mock.assert_called_once()
