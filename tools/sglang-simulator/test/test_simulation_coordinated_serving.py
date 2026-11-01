"""Exercise the virtual clock through real HTTP and coordinator sockets."""

import asyncio
import json
import socket

import aiohttp
import pytest
import requests
from sglang_simulator.simulation.client import AsyncCoordinatorClient
from sglang_simulator.simulation.coordinator import Cancel, Ready, Submit
from test_simulation_sglang_serving import ASSETS, SIM_CONFIGS, SGLangServingRunner

MILLISECOND = 1_000_000


def _start_server(tmp_path, monkeypatch, extra_args=None):
    tmp_path.mkdir()
    table = tmp_path / "replay.json"
    table.write_text(json.dumps({"[[1, 8]]": 0.01}), encoding="utf-8")
    config = json.loads(SIM_CONFIGS["replay"].read_text(encoding="utf-8"))
    config["predictor"]["database_path"] = str(table)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        endpoint = f"tcp://127.0.0.1:{sock.getsockname()[1]}"
    monkeypatch.setenv("SGLANG_SIMULATOR_COORDINATOR", endpoint)
    runner = SGLangServingRunner(config_path, tmp_path, extra_args=extra_args)
    # Bootstrap readiness must not wait for the first Ready or generate work.
    try:
        response = requests.get(f"{runner.base_url}/model_info", timeout=5)
        response.raise_for_status()
    except BaseException:
        runner.shutdown()
        raise
    return runner, endpoint


async def _generate(
    session, base_url, rid, token, *, delay=0, input_length=8, output_length=1
):
    await asyncio.sleep(delay)
    async with session.post(
        f"{base_url}/generate",
        json={
            "rid": rid,
            "input_ids": [token] * input_length,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output_length,
                "ignore_eos": True,
            },
            "stream": False,
        },
    ) as response:
        response.raise_for_status()
        body = await response.json()
        assert body["meta_info"]["id"] == rid
        assert body["meta_info"]["completion_tokens"] == output_length
        assert body["meta_info"]["finish_reason"]["type"] == "length"
        return body


@pytest.mark.parametrize("interval", [1, 192])
def test_native_prefill_cadence_preserves_http_fork_join(
    tmp_path, monkeypatch, interval
):
    runner, endpoint = _start_server(
        tmp_path / f"cadence-{interval}",
        monkeypatch,
        extra_args=["--prefill-decode-interval", str(interval)],
    )
    try:
        # The same deadlines and fork/join results must hold while native
        # cadence passes repeatedly defer pending prefills without a forward.
        asyncio.run(_fork_join(runner.base_url, endpoint, 0))
    finally:
        runner.shutdown()


async def _fork_join(base_url, endpoint, delay):
    tasks = {}
    grants = []
    # This pool is transport capacity, not simulated workload concurrency.
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with (
        aiohttp.ClientSession(
            connector=connector, timeout=aiohttp.ClientTimeout(total=30)
        ) as session,
        AsyncCoordinatorClient(endpoint) as client,
    ):
        try:
            tasks["root0000"] = asyncio.create_task(
                _generate(session, base_url, "root0000", 11, delay=delay)
            )
            grants.append(
                await client.request(
                    Ready("fork-join", 0, [Submit("root0000")], 5 * MILLISECOND)
                )
            )
            assert grants[-1].now_ns == 5 * MILLISECOND
            assert not grants[-1].terminals

            tasks["timer000"] = asyncio.create_task(
                _generate(session, base_url, "timer000", 21, delay=delay)
            )
            grants.append(
                await client.request(Ready("fork-join", 1, [Submit("timer000")]))
            )
            assert [(event.rid, event.at_ns) for event in grants[-1].terminals] == [
                ("root0000", 10 * MILLISECOND)
            ]
            # A terminal alone cannot release children before the HTTP output.
            await tasks["root0000"]
            await asyncio.sleep(delay)

            for rid, token, http_delay in (
                ("left0000", 31, delay),
                ("right000", 41, 0),
            ):
                tasks[rid] = asyncio.create_task(
                    _generate(session, base_url, rid, token, delay=http_delay)
                )
            grants.append(
                await client.request(
                    Ready("fork-join", 2, [Submit("left0000"), Submit("right000")])
                )
            )
            assert [(event.rid, event.at_ns) for event in grants[-1].terminals] == [
                ("timer000", 20 * MILLISECOND),
                ("left0000", 20 * MILLISECOND),
                ("right000", 20 * MILLISECOND),
            ]
            await asyncio.gather(
                tasks["timer000"], tasks["left0000"], tasks["right000"]
            )
            await asyncio.sleep(delay)

            tasks["join0000"] = asyncio.create_task(
                _generate(session, base_url, "join0000", 51, delay=delay)
            )
            grants.append(
                await client.request(Ready("fork-join", 3, [Submit("join0000")]))
            )
            assert [(event.rid, event.at_ns) for event in grants[-1].terminals] == [
                ("join0000", 30 * MILLISECOND)
            ]
            await tasks["join0000"]
            grants.append(
                await client.request(Ready("fork-join", 4, [], finished=True))
            )
            assert grants[-1].status == "FINISHED"
            assert grants[-1].now_ns == 30 * MILLISECOND
            return [
                (
                    grant.now_ns,
                    [(event.rid, event.kind, event.at_ns) for event in grant.terminals],
                    grant.status,
                )
                for grant in grants
            ]
        finally:
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)


def test_closed_loop_http_is_independent_of_wall_time(tmp_path, monkeypatch):
    results = []
    for index, delay in enumerate((0, 0.08)):
        runner, endpoint = _start_server(tmp_path / f"run-{index}", monkeypatch)
        try:
            results.append(asyncio.run(_fork_join(runner.base_url, endpoint, delay)))
        finally:
            runner.shutdown()
    assert results[0] == results[1]


async def _cancel_during_batch(base_url, endpoint, native_waiting):
    tasks = []
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with (
        aiohttp.ClientSession(
            connector=connector, timeout=aiohttp.ClientTimeout(total=30)
        ) as session,
        AsyncCoordinatorClient(endpoint) as client,
    ):
        try:
            # This server has one running slot: the second request must wait.
            rids = ["running0", "waiting0"] if native_waiting else ["running0"]
            for index, rid in enumerate(rids):
                tasks.append(
                    asyncio.create_task(
                        _generate(session, base_url, rid, 61 + index, output_length=4)
                    )
                )
            timer = await client.request(
                Ready("cancel", 0, [Submit(rid) for rid in rids], 5 * MILLISECOND)
            )
            assert timer.now_ns == 5 * MILLISECOND
            assert not timer.terminals
            cancel_round = 1
            if not native_waiting:
                rids.append("pending0")
                tasks.append(
                    asyncio.create_task(_generate(session, base_url, "pending0", 71))
                )
                timer = await client.request(
                    Ready("cancel", 1, [Submit("pending0")], 6 * MILLISECOND)
                )
                assert timer.now_ns == 6 * MILLISECOND
                assert not timer.terminals
                cancel_round = 2

            for task in tasks:
                task.cancel()
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            assert all(
                isinstance(outcome, asyncio.CancelledError) for outcome in outcomes
            )
            cancelled = await client.request(
                Ready(
                    "cancel",
                    cancel_round,
                    [Cancel(rid) for rid in rids],
                    finished=True,
                )
            )
            assert [
                (event.rid, event.kind, event.at_ns) for event in cancelled.terminals
            ] == [(rid, "CANCELLED", timer.now_ns) for rid in rids]
            assert cancelled.status == "CONTINUE"
            finished = await client.request(
                Ready("cancel", cancel_round + 1, [], finished=True)
            )
            assert finished.status == "FINISHED"
            assert finished.now_ns == 10 * MILLISECOND
            assert not finished.terminals
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize(
    "native_waiting", [False, True], ids=["bridge-pending", "scheduler-waiting"]
)
def test_cancelled_requests_drain_the_running_batch(
    tmp_path, monkeypatch, native_waiting
):
    runner, endpoint = _start_server(
        tmp_path / "cancel", monkeypatch, ["--max-running-requests", "1"]
    )
    try:
        asyncio.run(_cancel_during_batch(runner.base_url, endpoint, native_waiting))
    finally:
        runner.shutdown()


def test_native_decode_cancel_keeps_only_pre_cancel_output_tokens(
    tmp_path, monkeypatch
):
    ledger = tmp_path / "token_events.jsonl"
    monkeypatch.setenv("SGLANG_SIMULATOR_TOKEN_EVENTS", str(ledger))
    runner, endpoint = _start_server(tmp_path / "cancel-ledger", monkeypatch)

    async def run():
        async with (
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session,
            AsyncCoordinatorClient(endpoint) as client,
        ):
            task = asyncio.create_task(
                _generate(session, runner.base_url, "running0", 61, output_length=4)
            )
            try:
                timer = await client.request(
                    Ready("cancel-ledger", 0, [Submit("running0")], 12 * MILLISECOND)
                )
                assert timer.now_ns == 12 * MILLISECOND
                assert not timer.terminals
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                cancelled = await client.request(
                    Ready("cancel-ledger", 1, [Cancel("running0")], finished=True)
                )
                assert [(t.kind, t.at_ns) for t in cancelled.terminals] == [
                    ("CANCELLED", 12 * MILLISECOND)
                ]
                final = await client.request(
                    Ready("cancel-ledger", 2, [], finished=True)
                )
                assert final.status == "FINISHED"
                assert final.now_ns == 20 * MILLISECOND
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    try:
        asyncio.run(run())
        events = [json.loads(line) for line in ledger.read_text().splitlines()]
        commits = [event for event in events if event["event"] == "output_commit"]
        assert commits == [
            {
                "event": "output_commit",
                "at_ns": 10 * MILLISECOND,
                "commits": [{"rid": "running0", "output_index": 0, "token_count": 1}],
            }
        ]
        assert events[-1]["status"] == "FINISHED"
        assert events[-1]["total_output_tokens"] == 1
    finally:
        runner.shutdown()


async def _cancel_middle_chunk(base_url, endpoint):
    tasks = []
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with (
        aiohttp.ClientSession(
            connector=connector, timeout=aiohttp.ClientTimeout(total=30)
        ) as session,
        AsyncCoordinatorClient(endpoint) as client,
    ):
        try:
            chunked = asyncio.create_task(
                _generate(session, base_url, "chunk000", 81, input_length=192)
            )
            tasks.append(chunked)
            timer = await client.request(
                Ready("chunk-abort", 0, [Submit("chunk000")], 15 * MILLISECOND)
            )
            assert timer.now_ns == 15 * MILLISECOND
            assert not timer.terminals
            chunked.cancel()
            await asyncio.gather(chunked, return_exceptions=True)
            cancelled = await client.request(
                Ready("chunk-abort", 1, [Cancel("chunk000")])
            )
            assert [(event.kind, event.at_ns) for event in cancelled.terminals] == [
                ("CANCELLED", 15 * MILLISECOND)
            ]

            # With one request slot and 256 KV tokens, this 192-token request
            # depends on releasing the aborted request's slot and partial KV.
            followup = asyncio.create_task(
                _generate(session, base_url, "after000", 91, input_length=192)
            )
            tasks.append(followup)
            completed = await client.request(
                Ready("chunk-abort", 2, [Submit("after000")])
            )
            assert [
                (event.rid, event.kind, event.at_ns) for event in completed.terminals
            ] == [("after000", "COMPLETED", 50 * MILLISECOND)]
            await followup
            finished = await client.request(Ready("chunk-abort", 3, [], finished=True))
            assert finished.status == "FINISHED"
            assert finished.now_ns == 50 * MILLISECOND
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def test_middle_chunk_cancellation_releases_request_and_kv(tmp_path, monkeypatch):
    runner, endpoint = _start_server(
        tmp_path / "chunk-cancel",
        monkeypatch,
        [
            "--max-running-requests",
            "1",
            "--max-total-tokens",
            "256",
            "--chunked-prefill-size",
            "64",
        ],
    )
    try:
        asyncio.run(_cancel_middle_chunk(runner.base_url, endpoint))
        # The server remains responsive after native idle memory checks.
        response = requests.get(f"{runner.base_url}/server_info", timeout=5)
        response.raise_for_status()
        assert runner.server_proc.poll() is None
    finally:
        runner.shutdown()


async def _completion_stream(session, base_url, rid, output_length):
    finished = []
    usage = None
    done = False
    async with session.post(
        f"{base_url}/v1/completions",
        json={
            "model": str(ASSETS / "qwen3-8b"),
            "rid": rid,
            "prompt": [101 if output_length == 1 else 111] * 8,
            "max_tokens": output_length,
            "ignore_eos": True,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        response.raise_for_status()
        async for line in response.content:
            if not line.startswith(b"data: "):
                continue
            data = line[len(b"data: ") :].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            assert "error" not in event
            assert event["id"] == rid
            finished.extend(
                choice["finish_reason"]
                for choice in event["choices"]
                if choice.get("finish_reason")
            )
            if event.get("usage"):
                usage = event["usage"]
    assert done
    assert finished == ["length"]
    assert usage["completion_tokens"] == output_length


async def _cancel_after_peer_completion(base_url, endpoint):
    tasks = []
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with (
        aiohttp.ClientSession(
            connector=connector, timeout=aiohttp.ClientTimeout(total=30)
        ) as session,
        AsyncCoordinatorClient(endpoint) as client,
    ):
        try:
            short = asyncio.create_task(
                _completion_stream(session, base_url, "short000", 1)
            )
            long = asyncio.create_task(
                _completion_stream(session, base_url, "long0000", 4)
            )
            tasks.extend([short, long])
            completed = await client.request(
                Ready("peer-cancel", 0, [Submit("short000"), Submit("long0000")])
            )
            assert [
                (event.rid, event.kind, event.at_ns) for event in completed.terminals
            ] == [("short000", "COMPLETED", 10 * MILLISECOND)]
            # The cancellation follows A's full SSE stream and its Grant, not
            # either request's first-token delivery.
            await short
            long.cancel()
            await asyncio.gather(long, return_exceptions=True)
            cancelled = await client.request(
                Ready("peer-cancel", 1, [Cancel("long0000")], finished=True)
            )
            assert [
                (event.rid, event.kind, event.at_ns) for event in cancelled.terminals
            ] == [("long0000", "CANCELLED", 10 * MILLISECOND)]
            assert cancelled.status == "CONTINUE"
            finished = await client.request(Ready("peer-cancel", 2, [], finished=True))
            # Native running_batch still owns B after the prior batch result.
            assert finished.status == "FINISHED"
            assert finished.now_ns == 20 * MILLISECOND
            assert not finished.terminals
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def test_openai_stream_cancellation_drains_native_running_batch(tmp_path, monkeypatch):
    runner, endpoint = _start_server(tmp_path / "peer-cancel", monkeypatch)
    try:
        asyncio.run(_cancel_after_peer_completion(runner.base_url, endpoint))
        response = requests.get(f"{runner.base_url}/server_info", timeout=5)
        response.raise_for_status()
        assert runner.server_proc.poll() is None
    finally:
        runner.shutdown()


async def _chat_timing_stream(session, base_url, rid, read_delay):
    snapshots = []
    usage = None
    done = False
    event_type = None
    async with session.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": str(ASSETS / "qwen3-8b"),
            "rid": rid,
            "messages": [{"role": "user", "content": "hello"}],
            "input_ids": [101] * 8,
            "max_tokens": 4,
            "ignore_eos": True,
            "temperature": 0,
            "n": 1,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        response.raise_for_status()
        await asyncio.sleep(read_delay)
        async for line in response.content:
            if not line.strip():
                event_type = None
                continue
            if line.startswith(b"event: "):
                event_type = line[len(b"event: ") :].strip()
                continue
            if not line.startswith(b"data: "):
                continue
            data = line[len(b"data: ") :].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            assert "error" not in event
            if event_type == b"sglang_simulator":
                snapshots.append(event)
            else:
                assert event["id"] == rid
                if any(
                    choice.get("delta", {}).get("content")
                    for choice in event.get("choices", [])
                ):
                    assert snapshots, "content preceded its timing snapshot"
                if event.get("usage"):
                    usage = event["usage"]
    assert done and snapshots and usage
    final = snapshots[-1]
    assert final["terminal_ns"] == final["commit_ns"]
    assert final["prompt_tokens"] == usage["prompt_tokens"] == 8
    assert final["completion_tokens"] == usage["completion_tokens"] == 4
    assert final["cached_tokens"] == usage["prompt_tokens_details"]["cached_tokens"]
    assert all(snapshot["rid"] == rid for snapshot in snapshots)
    assert all(
        snapshot["first_token_ns"] == final["first_token_ns"] for snapshot in snapshots
    )
    assert all(
        snapshot["cached_tokens"] == final["cached_tokens"] for snapshot in snapshots
    )
    return final


async def _chat_timing_run(base_url, endpoint):
    results = []
    async with (
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session,
        AsyncCoordinatorClient(endpoint) as client,
    ):
        for index, delay in enumerate((0, 0.08)):
            rid = f"request{index}"
            task = asyncio.create_task(
                _chat_timing_stream(session, base_url, rid, delay)
            )
            try:
                grant = await client.request(Ready("timing", index, [Submit(rid)]))
                result = await task
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            arrival = index * 40 * MILLISECOND
            assert result["arrival_ns"] == arrival
            assert result["first_token_ns"] == arrival + 10 * MILLISECOND
            assert result["last_token_ns"] == arrival + 40 * MILLISECOND
            assert result["terminal_ns"] == grant.now_ns
            assert [(event.rid, event.at_ns) for event in grant.terminals] == [
                (rid, result["terminal_ns"])
            ]
            results.append(result)
        final = await client.request(Ready("timing", 2, [], finished=True))
        assert final.status == "FINISHED"
    assert results[0]["cached_tokens"] == 0
    assert results[1]["cached_tokens"] > 0


@pytest.mark.parametrize("incremental", [False, True])
def test_chat_timing_metadata_matches_native_usage(tmp_path, monkeypatch, incremental):
    args = ["--stream-interval", "3"]
    if incremental:
        args.append("--incremental-streaming-output")
    runner, endpoint = _start_server(tmp_path / "chat-timing", monkeypatch, args)
    try:
        asyncio.run(_chat_timing_run(runner.base_url, endpoint))
    finally:
        runner.shutdown()
