"""CPU coverage of reporter ownership across the server entry points."""

import asyncio
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from sglang.srt import runtime_context as rc
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def isolated_context():
    rc.reset_context()
    yield
    rc.reset_context()


@pytest.fixture
def http_runtime(monkeypatch):
    from sglang.srt import load_reporter
    from sglang.srt.entrypoints import http_server

    # Exercise the real lifespan with model loading and serving handlers isolated.
    manager = SimpleNamespace(serving_chat_class=Mock(), worker_id=1)
    monkeypatch.setattr(
        http_server,
        "_global_state",
        SimpleNamespace(
            tokenizer_manager=manager, template_manager=object(), scheduler_info={}
        ),
    )
    for name in (
        "OpenAIServingCompletion",
        "OpenAIServingEmbedding",
        "OpenAIServingClassify",
        "OpenAIServingScore",
        "OpenAIServingRerank",
        "OpenAIServingTokenize",
        "OpenAIServingDetokenize",
        "OpenAIServingTranscription",
        "OllamaServing",
        "AnthropicServing",
    ):
        monkeypatch.setattr(http_server, name, Mock())
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.entrypoints.openai.serving_responses",
        SimpleNamespace(OpenAIServingResponses=Mock()),
    )
    monkeypatch.setattr(http_server, "_wait_and_warmup", Mock())
    monkeypatch.setattr(http_server.envs.EXA_API_KEY, "get", lambda: None)

    handle = SimpleNamespace(close=AsyncMock())
    start = AsyncMock(return_value=handle)
    monkeypatch.setattr(load_reporter, "start_load_reporter", start)
    native_start = Mock(return_value=object())
    native_stop = Mock()
    monkeypatch.setattr(
        http_server, "_start_native_grpc_server_for_runtime", native_start
    )
    monkeypatch.setattr(http_server, "_shutdown_native_grpc_server", native_stop)

    def make_app(*, single=True, reporter_port=None, grpc_port=None):
        args = ServerArgs(
            model_path="dummy",
            load_reporter_port=reporter_port,
            grpc_port=grpc_port,
            tokenizer_worker_num=1 if single else 2,
        )
        rc.publish(args, role="tokenizer")
        # A live bag can differ from the raw record. Entry points must read it.
        rc.get_context().override("test-dp-size", dp_size=3)
        monkeypatch.setattr(
            http_server, "init_multi_tokenizer", AsyncMock(return_value=args)
        )
        return SimpleNamespace(
            state=SimpleNamespace(),
            is_single_tokenizer_mode=single,
            server_args=args,
            warmup_thread_kwargs={},
        )

    return SimpleNamespace(
        make_app=make_app,
        lifespan=http_server.lifespan,
        manager=manager,
        start=start,
        handle=handle,
        native_start=native_start,
        native_stop=native_stop,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "single,reporter_port,grpc_port",
    [
        (True, None, None),
        (True, None, 30101),
        (True, 30100, 30101),
        (False, None, None),
        (False, 30100, None),
    ],
)
async def test_http_ownership_and_native_grpc(
    http_runtime, single, reporter_port, grpc_port
):
    runtime = http_runtime
    app = runtime.make_app(
        single=single, reporter_port=reporter_port, grpc_port=grpc_port
    )
    async with runtime.lifespan(app):
        if single and reporter_port is not None:
            runtime.start.assert_awaited_once()
            args, source = runtime.start.call_args.args
            assert args is app.server_args
            assert source.expected_dp_ranks() == frozenset({0, 1, 2})
            runtime.handle.close.assert_not_awaited()
        else:
            runtime.start.assert_not_awaited()
        if grpc_port is not None:
            runtime.native_start.assert_called_once()
        else:
            runtime.native_start.assert_not_called()

    if single and reporter_port is not None:
        runtime.handle.close.assert_awaited_once()
    else:
        runtime.handle.close.assert_not_awaited()
    runtime.native_stop.assert_called_once_with(
        runtime.native_start.return_value if grpc_port is not None else None
    )


@pytest.mark.asyncio
async def test_http_closes_reporter_when_native_grpc_start_fails(http_runtime):
    runtime = http_runtime
    app = runtime.make_app(reporter_port=30100, grpc_port=30101)
    runtime.native_start.side_effect = RuntimeError("native gRPC startup failed")
    with pytest.raises(RuntimeError, match="native gRPC startup failed"):
        async with runtime.lifespan(app):
            pytest.fail("lifespan must propagate native gRPC startup failure")
    runtime.handle.close.assert_awaited_once()


@pytest.mark.parametrize("enabled", [False, True])
def test_multi_tokenizer_reporter_reads_published_parallel_config(monkeypatch, enabled):
    from sglang.srt import load_reporter
    from sglang.srt.managers.multi_tokenizer_mixin import MultiTokenizerRouter

    args = ServerArgs(model_path="dummy", load_reporter_port=30100 if enabled else None)
    rc.publish(args, role="tokenizer")
    rc.get_context().override("test-dp-size", dp_size=3)
    router = MultiTokenizerRouter.__new__(MultiTokenizerRouter)
    router.server_args = args
    reader = SimpleNamespace(read_all=Mock(return_value=[]), close=Mock())
    router.load_snapshot_reader = reader if enabled else None
    router._loop = asyncio.new_event_loop()
    thread = threading.Thread(target=router._loop.run_forever, daemon=True)
    thread.start()
    handle = SimpleNamespace(close=AsyncMock())
    start = AsyncMock(return_value=handle)
    monkeypatch.setattr(load_reporter, "start_load_reporter", start)
    try:
        router._load_reporter_handle = router._start_load_reporter_owner()
        if enabled:
            assert router._load_reporter_handle is handle
            start.assert_awaited_once()
            received_args, source = start.call_args.args
            assert received_args is args
            assert source.expected_dp_ranks() == frozenset({0, 1, 2})
        else:
            assert router._load_reporter_handle is None
            start.assert_not_awaited()
        router.close()
        if enabled:
            handle.close.assert_awaited_once()
            reader.close.assert_called_once()
        else:
            handle.close.assert_not_awaited()
    finally:
        router._loop.call_soon_threadsafe(router._loop.stop)
        thread.join(timeout=5)
        router._loop.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
