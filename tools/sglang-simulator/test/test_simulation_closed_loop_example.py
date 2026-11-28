"""The serial reference driver bounds failures and reaps both transports."""

import asyncio
import importlib.util
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from sglang_simulator.simulation.coordinator import Grant, Terminal


def _load_example():
    path = Path(__file__).resolve().parents[1] / "examples" / "closed_loop.py"
    spec = importlib.util.spec_from_file_location("closed_loop", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    return example


@pytest.mark.parametrize("failure", [None, "http", "control", "missing_response"])
def test_serial_driver_completes_or_reaps_transports(monkeypatch, failure):
    example = _load_example()

    async def exercise():
        original_tasks = asyncio.all_tasks()
        http_started = asyncio.Event()
        control_started = asyncio.Event()
        release_response = asyncio.Event()
        control_tasks = []
        rounds = []
        control_error = RuntimeError("control exchange failed")

        async def generate(request):
            await request.json()
            http_started.set()
            await control_started.wait()
            if failure == "http":
                raise web.HTTPBadRequest(text="rejected before scheduler admission")
            if failure in ("control", "missing_response"):
                await release_response.wait()
            return web.json_response({"output_ids": [1, 1, 1, 1]})

        class Control:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def request(self, ready):
                control_tasks.append(asyncio.current_task())
                rounds.append(ready)
                control_started.set()
                await http_started.wait()
                if failure == "http":
                    await asyncio.Event().wait()
                if failure == "control":
                    raise control_error
                now_ns = (10, 5_000_000_010, 5_000_000_020, 5_000_000_020)[ready.round]
                terminals = [
                    Terminal(operation.rid, "COMPLETED", now_ns)
                    for operation in ready.operations
                ]
                return Grant(
                    ready.run_id,
                    ready.round,
                    now_ns,
                    terminals,
                    status="FINISHED" if ready.finished else "CONTINUE",
                )

        monkeypatch.setattr(example, "AsyncCoordinatorClient", lambda _: Control())
        session_factory = aiohttp.ClientSession

        def short_timeout_session(**kwargs):
            # Exercise the example's real timeout without making the test wait 30s.
            assert kwargs["timeout"].total == 30
            if failure == "missing_response":
                kwargs["timeout"] = aiohttp.ClientTimeout(total=0.1)
            return session_factory(**kwargs)

        monkeypatch.setattr(example.aiohttp, "ClientSession", short_timeout_session)
        app = web.Application()
        app.router.add_post("/generate", generate)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        base_url = f"http://127.0.0.1:{runner.addresses[0][1]}"
        try:
            task = asyncio.create_task(example.run(base_url, "unused", 5))
            done, _ = await asyncio.wait({task}, timeout=2)
            if not done:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                pytest.fail("driver did not terminate after a transport failure")
            if failure is None:
                assert task.result() == [
                    {"turn": 0, "completed_ns": 10},
                    {"turn": 1, "completed_ns": 5_000_000_020},
                ]
                assert len(rounds) == 4 and rounds[-1].finished
            elif failure == "http":
                with pytest.raises(aiohttp.ClientResponseError, match="400"):
                    task.result()
            elif failure == "control":
                assert task.exception() is control_error
            else:
                with pytest.raises(asyncio.TimeoutError):
                    task.result()
                assert len(rounds) == 1
            assert all(task.done() for task in control_tasks)
        finally:
            release_response.set()
            await runner.cleanup()
        assert asyncio.all_tasks() == original_tasks

    asyncio.run(exercise())
