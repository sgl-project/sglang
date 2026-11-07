"""Closed-loop clients use ordinary HTTP and wall time in BLOCKING mode."""

import asyncio
import json
import time

import aiohttp
from test_simulation_sglang_serving import ASSETS, SIM_CONFIGS, SGLangServingRunner

BATCH_SECONDS = 0.05
OUTPUT_TOKENS = 4
TOOL_SECONDS = 0.1


async def _chat(session, base_url, messages):
    started = time.perf_counter()
    content_times = []
    content = []
    usage = None
    finishes = []
    rid = None
    done = False
    async with session.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": str(ASSETS / "qwen3-8b"),
            "messages": messages,
            "max_completion_tokens": OUTPUT_TOKENS,
            "ignore_eos": True,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        response.raise_for_status()
        async for line in response.content:
            # An unmodified client must not receive coordinator timing events.
            assert not line.startswith(b"event: sglang_simulator")
            if not line.startswith(b"data: "):
                continue
            data = line[len(b"data: ") :].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            assert "error" not in event
            rid = event["id"] if rid is None else rid
            assert event["id"] == rid
            for choice in event["choices"]:
                text = choice.get("delta", {}).get("content")
                if text:
                    content.append(text)
                    content_times.append(time.perf_counter())
                if choice.get("finish_reason"):
                    finishes.append(choice["finish_reason"])
            if event.get("usage"):
                usage = event["usage"]
    finished = time.perf_counter()
    assert done and rid and content_times and usage
    assert finishes == ["length"]
    assert usage["completion_tokens"] == OUTPUT_TOKENS
    assert usage["prompt_tokens"] > 0
    # Lower bounds catch accidentally unpaced output without relying on host speed.
    assert content_times[0] - started >= BATCH_SECONDS * 0.9
    assert content_times[-1] - started >= BATCH_SECONDS * OUTPUT_TOKENS * 0.9
    return {
        "rid": rid,
        "text": "".join(content),
        "started": started,
        "finished": finished,
    }


async def _fork_join(base_url, connection_limit):
    connector = aiohttp.TCPConnector(limit=connection_limit)
    async with aiohttp.ClientSession(
        connector=connector, timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        messages = [{"role": "user", "content": "Plan two independent tasks."}]
        root = await _chat(session, base_url, messages)
        history = messages + [{"role": "assistant", "content": root["text"]}]
        await asyncio.sleep(TOOL_SECONDS)
        branches = await asyncio.gather(
            *(
                _chat(
                    session,
                    base_url,
                    history + [{"role": "user", "content": f"Execute task {name}."}],
                )
                for name in ("left", "right")
            )
        )
        joined = await _chat(
            session,
            base_url,
            history
            + [
                {
                    "role": "user",
                    "content": "Combine results: "
                    + " ".join(b["text"] for b in branches),
                }
            ],
        )
    assert min(b["started"] for b in branches) >= root["finished"] + TOOL_SECONDS * 0.9
    assert joined["started"] >= max(b["finished"] for b in branches)
    return [root, *branches, joined]


async def _run_workload(base_url):
    async with aiohttp.ClientSession() as session:
        # Serving must work before profiling, and warmup must not enter its results.
        warmup = await _chat(
            session, base_url, [{"role": "user", "content": "Warmup."}]
        )
        async with session.post(f"{base_url}/start_profile", json={}) as response:
            response.raise_for_status()
        results = []
        for limit in (1, 2):
            results.extend(await _fork_join(base_url, limit))
        async with session.post(f"{base_url}/stop_profile", json={}) as response:
            response.raise_for_status()
    return warmup, results


def test_plain_chat_fork_join_uses_wall_time(tmp_path, monkeypatch):
    monkeypatch.delenv("SGLANG_SIMULATOR_COORDINATOR", raising=False)
    table = tmp_path / "replay.json"
    table.write_text(json.dumps({"[[1, 8]]": BATCH_SECONDS}), encoding="utf-8")
    config = json.loads(SIM_CONFIGS["replay"].read_text(encoding="utf-8"))
    config["predictor"]["database_path"] = str(table)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    runner = SGLangServingRunner(
        config_path,
        tmp_path,
        mode="blocking",
        extra_args=["--chat-template", "chatml", "--skip-server-warmup"],
    )
    try:
        warmup, results = asyncio.run(_run_workload(runner.base_url))
    finally:
        runner.shutdown()
    requests = [
        json.loads(line)
        for line in (runner.output_dir / "request.jsonl").read_text().splitlines()
    ]
    assert len(requests) == len(results) == 8
    assert {item["rid"] for item in requests} == {item["rid"] for item in results}
    assert warmup["rid"] not in {item["rid"] for item in requests}
    assert all(item["output_length"] == OUTPUT_TOKENS for item in requests)
    iterations = [
        json.loads(line)
        for line in (runner.output_dir / "iteration.jsonl").read_text().splitlines()
    ]
    assert any(len(item["requests"]) > 1 for item in iterations)
