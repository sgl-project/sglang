"""Request attribution survives failed streaming, cancellation and later work."""

import asyncio
import json

import pytest
from sglang.benchmark.tracelab import TraceRound, send_generate
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Response:
    def __init__(self, events, status=200, error=None):
        self.events, self.status, self.error = events, status, error
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def raise_for_status(self):
        if self.error:
            raise self.error

    def __aiter__(self):
        async def lines():
            for event in self.events:
                if isinstance(event, BaseException):
                    raise event
                data = event if isinstance(event, str) else json.dumps(event)
                yield ("data: " + data + "\n").encode()
                yield b"\n"

        return lines()


class Client:
    def __init__(self, response):
        self.response, self.payloads = response, []

    def post(self, url, *, json):
        self.payloads.append(json)
        return self.response


ROW = TraceRound("session", "round", 0, 2, 2, 0)
TOKEN = {"output_ids": [7], "meta_info": {"completion_tokens": 1, "cached_tokens": 1}}


@pytest.mark.parametrize(
    "events,error_type",
    [
        ([TOKEN], RuntimeError),
        (
            [
                TOKEN,
                {"error": "backend failed", "meta_info": {"finish_reason": "error"}},
            ],
            RuntimeError,
        ),
        ([TOKEN, "not-json"], ValueError),
        ([TOKEN, []], ValueError),
        ([TOKEN, {"meta_info": None}], ValueError),
        ([TOKEN, {"meta_info": {"completion_tokens": True}}], ValueError),
        ([TOKEN, asyncio.TimeoutError("deadline")], asyncio.TimeoutError),
        ([TOKEN, asyncio.CancelledError()], asyncio.CancelledError),
        (["[DONE]"], ValueError),
    ],
)
def test_failed_stream_retains_partial_state(events, error_type):
    async def exercise():
        records = []
        client = Client(Response(events))
        with pytest.raises(error_type):
            await send_generate(
                client,
                "http://test",
                ROW,
                [3, 4],
                request_id="actual-rid",
                on_result=records.append,
            )
        assert len(records) == 1
        result = records[0]
        assert client.payloads[0]["rid"] == result["request_id"] == "actual-rid"
        assert result["request_started_ns"] <= result["request_finished_ns"]
        assert result["http_status"] == 200 and not result["success"]
        assert result["status"] == (
            "cancelled" if error_type is asyncio.CancelledError else "failed"
        )
        assert result["error"]["type"]
        assert result["output_ids"] == ([] if events == ["[DONE]"] else [7])
        if events != ["[DONE]"]:
            assert result["output_tokens"] == 1
            assert result["server_metadata"]["cached_tokens"] == 1

    asyncio.run(exercise())


def test_http_failure_and_invalid_prompt_have_terminal_records():
    async def exercise():
        for ids, status, failure, exception in (
            ([3, 4], 503, RuntimeError("unavailable"), RuntimeError),
            ([], 200, None, ValueError),
        ):
            records = []
            client = Client(Response([], status=status, error=failure))
            with pytest.raises(exception):
                await send_generate(
                    client, "http://test", ROW, ids, on_result=records.append
                )
            assert len(records) == 1 and records[0]["status"] == "failed"
            assert records[0]["output_tokens"] == 0
            assert records[0]["http_status"] == (status if ids else None)
            assert len(client.payloads) == bool(ids)

    asyncio.run(exercise())


def test_success_metadata_only_events_unique_ids_and_export_failure(tmp_path):
    async def exercise():
        path = tmp_path / "requests.jsonl"

        def persist(record):
            with path.open("a") as output:
                output.write(json.dumps(record) + "\n")
                output.flush()

        client = Client(
            Response(
                [
                    TOKEN,
                    {"output_ids": [8], "meta_info": {"completion_tokens": 2}},
                    {"meta_info": {"finish_reason": {"type": "length"}}},
                    "[DONE]",
                ]
            )
        )
        results = await asyncio.gather(
            *(
                send_generate(client, "http://test", ROW, [3, 4], on_result=persist)
                for _ in range(4)
            )
        )
        assert len({row["request_id"] for row in results}) == 4
        assert {row["request_id"] for row in results} == {
            p["rid"] for p in client.payloads
        }
        with pytest.raises(OSError):
            assert len(path.read_text().splitlines()) == 4
            raise OSError("profile export failed after requests")
        for row in map(json.loads, path.read_text().splitlines()):
            assert (
                row["success"] and row["status"] == "completed" and row["error"] is None
            )
            assert row["stream_complete"] and row["output_ids"] == [7, 8]
            assert row["server_metadata"]["finish_reason"] == {"type": "length"}

    asyncio.run(exercise())


def test_short_output_is_incomplete_not_success():
    async def exercise():
        records = []
        row = await send_generate(
            Client(Response([TOKEN, "[DONE]"])),
            "http://test",
            ROW,
            [3, 4],
            on_result=records.append,
        )
        assert records == [row]
        assert row["stream_complete"] and not row["output_length_matched"]
        assert row["status"] == "incomplete" and not row["success"] and row["error"]

    asyncio.run(exercise())


def test_result_sink_failure_is_not_hidden():
    async def exercise():
        def fail(record):
            raise OSError("cannot persist")

        with pytest.raises(OSError):
            await send_generate(
                Client(Response([TOKEN, "[DONE]"])),
                "http://test",
                ROW,
                [3, 4],
                on_result=fail,
            )

    asyncio.run(exercise())


def test_server_abort_after_full_token_count_is_not_success():
    async def exercise():
        reason = {"type": "abort", "message": "worker failed", "status_code": 500}
        records = []
        client = Client(
            Response(
                [
                    {"output_ids": [7, 8], "meta_info": {"completion_tokens": 2}},
                    {"meta_info": {"finish_reason": reason}},
                    "[DONE]",
                ]
            )
        )
        with pytest.raises(RuntimeError):
            await send_generate(
                client, "http://test", ROW, [3, 4], on_result=records.append
            )
        assert len(records) == 1
        assert records[0]["output_tokens"] == 2 and records[0]["status"] == "failed"
        assert not records[0]["success"]
        assert records[0]["server_metadata"]["finish_reason"] == reason

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel", [False, True])
def test_cli_flushes_failed_or_cancelled_request(tmp_path, monkeypatch, cancel):
    from argparse import Namespace

    from aiohttp import web
    from sglang.benchmark.tracelab import run_trace
    from transformers import AutoTokenizer

    class Tokenizer:
        all_special_ids = []

        def get_vocab(self):
            return {str(i): i for i in range(32)}

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda _: Tokenizer())
    source = tmp_path / "trace.jsonl"
    source.write_text(
        json.dumps(
            dict(
                session_id="s",
                round_id="r",
                round_index=0,
                input_tokens_total=2,
                prefix_tokens=0,
                newly_append_tokens=2,
                output_tokens=2,
            )
        )
    )
    destination = tmp_path / "result.jsonl"

    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()
        sent_ids = []

        async def handler(request):
            sent_ids.append((await request.json())["rid"])
            entered.set()
            if cancel:
                await release.wait()
            return web.Response(status=503, text="unavailable")

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
                token_pool_limit=100,
                min_input_len=1,
                max_input_len=10,
                min_output_len=1,
                max_output_len=10,
                num_requests=None,
                max_rounds_per_session=None,
                seed=0,
                request_timeout=60,
                api_key=None,
                output_file=str(destination),
                concurrency=1,
                request_rate=float("inf"),
                base_url=f"http://127.0.0.1:{runner.addresses[0][1]}",
            )
            task = asyncio.create_task(run_trace(args))
            await asyncio.wait_for(entered.wait(), 30)
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                assert await task == 1
            rows = [json.loads(line) for line in destination.read_text().splitlines()]
            terminal = [row for row in rows if row["type"] == "request"]
            assert len(terminal) == 1
            row = terminal[0]
            assert row["request_id"] == sent_ids[0]
            assert row["request_started_ns"] <= row["request_finished_ns"]
            assert row["status"] == ("cancelled" if cancel else "failed")
            assert not row["success"] and row["error"]
            if not cancel:
                assert row["http_status"] == 503 and rows[-1]["failed"] == 1
        finally:
            release.set()
            await runner.cleanup()

    asyncio.run(exercise())
