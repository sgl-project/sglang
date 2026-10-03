"""Unit tests for TraceLab serving replay semantics."""

import asyncio
import gzip
import json
from argparse import Namespace
from dataclasses import replace

import pytest

from sglang.benchmark.tracelab import (
    CorpusPrompts,
    LengthBounds,
    StreamMetrics,
    SyntheticPrompts,
    iter_rounds,
    iter_sse_data,
    parse_round,
    replay_sessions,
    run_trace,
    send_generate,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_canonical_replay_preserves_generated_context_and_corpus(tmp_path):
    source = tmp_path / "trace.csv"
    source.write_text(
        "id,round_idx,prefix_len,input_len,output_len,arrival_time,tool_wait_after_ms\n"
        "a,0,0,4,2,10,20\n"
        "a,1,6,2,1,10,0\n"
    )
    first, second = list(iter_rounds(str(source)))
    assert (first.arrival_time_ms, first.tool_wait_after_ms) == (10, 20)
    prompts = CorpusPrompts(list(range(10, 110)))
    initial, reused = prompts.build(first)
    assert reused == 0 and len(initial) == 4
    assert initial[1:] == [token + 1 for token in initial[:-1]]
    prompts.commit_output(first, initial, [7, 8])
    following, reused = prompts.build(second)
    assert following[:6] == initial + [7, 8]
    assert reused == 6 and len(following) == 8


def record(index, tokens=128, output=16):
    return {
        "session_id": "session-a",
        "round_id": f"round-{index}",
        "round_index": index,
        "input_tokens_total": tokens,
        "prefix_tokens": tokens // 2,
        "newly_append_tokens": tokens - tokens // 2,
        "output_tokens": output,
    }


@pytest.mark.parametrize("compressed", [False, True])
def test_trace_filter_preserves_observed_lengths_and_identity(tmp_path, compressed):
    source = tmp_path / ("trace.jsonl.gz" if compressed else "trace.jsonl")
    opener = gzip.open if compressed else open
    records = [record(0, 64), record(1, 128), record(2, 256), record(3, 512)]
    with opener(source, "wt") as stream:
        for item in records:
            stream.write(json.dumps(item) + "\n")
    rows = list(iter_rounds(str(source), LengthBounds(128, 256), limit=2))
    assert [(r.round_index, r.input_tokens) for r in rows] == [(1, 128), (2, 256)]
    assert [r.round_id for r in rows] == ["round-1", "round-2"]
    assert all(r.session_id == "session-a" for r in rows)
    with opener(source, "wt") as stream:
        for session in ("session-a", "session-b"):
            for item in records:
                stream.write(json.dumps(dict(item, session_id=session)) + "\n")
    rows = list(iter_rounds(str(source), LengthBounds(128, 512), 4, 2))
    assert [(r.session_id, r.round_index) for r in rows] == [
        ("session-a", 1),
        ("session-a", 2),
        ("session-b", 1),
        ("session-b", 2),
    ]


def test_cache_accounting_is_not_added_to_total_again():
    item = record(0)
    item["claude_cache_read_input_tokens"] = 64
    item["claude_cache_creation_input_tokens"] = 60
    assert parse_round(item).input_tokens == 128
    item["input_tokens_total"] += 1
    with pytest.raises(ValueError):
        parse_round(item)


def test_synthetic_prefix_is_bounded_and_session_local():
    generator = SyntheticPrompts(range(3, 100), seed=7)
    first = parse_round(record(0, 128))
    prompt, reused = generator.build(first)
    assert len(prompt) == 128 and reused == 0
    generator.commit_output(first, prompt, [7] * first.output_tokens)
    second = parse_round(record(1, 256))
    following, reused = generator.build(second)
    assert len(following) == 256 and reused == 128
    assert following[:128] == prompt
    other, reused = generator.build(replace(second, session_id="other"))
    assert reused == 0 and other != following
    assert SyntheticPrompts(range(3, 100), seed=7).build(first)[0] == prompt


@pytest.mark.parametrize("key,value", [("output_tokens", -1), ("round_index", True)])
def test_invalid_counts_are_rejected(key, value):
    item = record(0)
    item[key] = value
    with pytest.raises(ValueError):
        parse_round(item)


def test_bounds_reject_invalid_range():
    with pytest.raises(ValueError):
        LengthBounds(256, 128)


def test_replay_orders_sessions_and_bounds_concurrency():
    async def exercise():
        rows = [parse_round(record(1)), parse_round(record(0))]
        rows += [replace(row, session_id="session-b") for row in rows]
        active = set()
        order = []
        peak = 0

        async def send(row):
            nonlocal peak
            assert row.session_id not in active
            active.add(row.session_id)
            peak = max(peak, len(active))
            await asyncio.sleep(0)
            order.append((row.session_id, row.round_index))
            active.remove(row.session_id)
            return row.output_tokens

        results = await replay_sessions(rows, send, concurrency=2)
        assert peak == 2
        assert len(results) == len(rows)
        for session in ("session-a", "session-b"):
            assert [index for sid, index in order if sid == session] == [0, 1]

    asyncio.run(exercise())


def test_stream_timing_handles_batched_tokens_and_metadata_only_events():
    metrics = StreamMetrics(10.0)
    metrics.observe(0, 10.1)
    metrics.observe(3, 10.5)
    metrics.observe(3, 10.6)
    metrics.observe(5, 10.9)
    metrics.observe(6, 11.0)
    result = metrics.result()
    assert result["ttft_s"] == pytest.approx(0.5)
    assert result["mean_tbt_s"] == pytest.approx(0.5 / 3)
    assert result["chunk_token_counts"] == [2, 1]
    assert result["output_tokens"] == 6
    with pytest.raises(ValueError):
        metrics.observe(5, 11.1)


def test_sse_preserves_multiline_events_and_ignores_comments():
    async def exercise():
        async def content():
            for line in (
                b": ping\r\n",
                b"data: first\r\n",
                b"data: second\r\n",
                b"\r\n",
                b"data: [DONE]\n",
                b"\n",
            ):
                yield line

        assert [data async for data in iter_sse_data(content())] == [
            "first\nsecond",
            "[DONE]",
        ]

    asyncio.run(exercise())


def test_replay_rate_applies_across_sessions():
    async def exercise():
        rows = [replace(parse_round(record(0)), session_id=str(i)) for i in range(3)]
        starts = []

        async def send(row):
            starts.append(asyncio.get_running_loop().time())

        await replay_sessions(rows, send, concurrency=3, request_rate=20)
        assert all(b - a >= 0.045 for a, b in zip(starts, starts[1:]))

    asyncio.run(exercise())


@pytest.mark.parametrize("terminal", [True, False])
def test_generate_transport_reads_fragmented_stream_and_requires_completion(terminal):
    import aiohttp
    from aiohttp import web

    async def exercise():
        row = parse_round(record(0, tokens=4, output=2))

        async def handler(request):
            payload = await request.json()
            assert payload["input_ids"] == [3, 4, 5, 6]
            assert payload["sampling_params"]["max_new_tokens"] == 2
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            data = b'data: {"meta_info":{"completion_tokens":2,"cached_tokens":0}}\n\n'
            for fragment in (data[:7], data[7:23], data[23:]):
                await response.write(fragment)
                await asyncio.sleep(0)
            if terminal:
                await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_post("/generate", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = runner.addresses[0][1]
            async with aiohttp.ClientSession() as client:
                if terminal:
                    result = await send_generate(
                        client, f"http://127.0.0.1:{port}", row, [3, 4, 5, 6]
                    )
                    assert result["output_tokens"] == 2
                    assert result["output_length_matched"]
                    assert result["ttft_s"] >= 0
                    assert result["mean_tbt_s"] is None
                else:
                    with pytest.raises(RuntimeError):
                        await send_generate(
                            client, f"http://127.0.0.1:{port}", row, [3, 4, 5, 6]
                        )
        finally:
            await runner.cleanup()

    asyncio.run(exercise())


def test_driver_writes_measured_results_without_overwriting(tmp_path, monkeypatch):
    from aiohttp import web
    from transformers import AutoTokenizer

    class Tokenizer:
        all_special_ids = [0, 1, 2]

        def get_vocab(self):
            return {str(i): i for i in range(32)}

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda _: Tokenizer())
    source = tmp_path / "trace.jsonl"
    source.write_text("\n".join(json.dumps(record(i, 8, 3)) for i in range(2)))
    destination = tmp_path / "result.jsonl"

    async def exercise():
        requests = []

        async def handler(request):
            body = await request.json()
            requests.append(body)
            assert len(body["input_ids"]) == 8
            assert not set(body["input_ids"]) & {0, 1, 2}
            return web.Response(
                text='data: {"output_ids":[7,8,9],"meta_info":{"completion_tokens":3}}\n\ndata: [DONE]\n\n',
                content_type="text/event-stream",
            )

        app = web.Application()
        app.router.add_post("/generate", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            await web.TCPSite(runner, "127.0.0.1", 0).start()
            args = Namespace(
                dataset_path=str(source),
                tokenizer="fake",
                text_file=None,
                token_pool_limit=1000,
                min_input_len=1,
                max_input_len=10,
                min_output_len=1,
                max_output_len=4,
                num_requests=2,
                max_rounds_per_session=None,
                seed=7,
                request_timeout=10,
                api_key=None,
                output_file=str(destination),
                concurrency=2,
                request_rate=float("inf"),
                base_url=f"http://127.0.0.1:{runner.addresses[0][1]}",
            )
            assert await run_trace(args) == 0
            records = [
                json.loads(line) for line in destination.read_text().splitlines()
            ]
            assert records[0]["selected_rounds"] == 2
            requests_recorded = [r for r in records if r["type"] == "request"]
            assert all(r["success"] for r in requests_recorded)
            assert [r["round_index"] for r in requests_recorded] == [0, 1]
            assert records[-1]["successful"] == 2
            assert records[-1]["output_tokens"] == 6
            assert records[-1]["ttft_s"]["p50"] >= 0
            with pytest.raises(FileExistsError):
                await run_trace(args)
            assert len(requests) == 2
        finally:
            await runner.cleanup()

    asyncio.run(exercise())
