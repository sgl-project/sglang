"""The reference driver preserves causality despite transport delays."""

import asyncio
import importlib.util

import pytest
from sglang_simulator.simulation.coordinator import Grant, Terminal
from test_simulation_coordinated_serving import MILLISECOND, _start_server
from test_simulation_sglang_serving import EXAMPLES


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "fork_join", EXAMPLES / "fork_join.py"
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    return example


def test_fork_join_example_is_independent_of_http_time(tmp_path, monkeypatch):
    example = _load_example()
    timelines = []
    for index, delay in enumerate((0, 0.08)):
        runner, endpoint = _start_server(tmp_path / f"run-{index}", monkeypatch)

        async def send(http, base_url, payload):
            await asyncio.sleep(delay)
            async with http.post(f"{base_url}/generate", json=payload) as response:
                response.raise_for_status()
                await asyncio.sleep(delay)
                return await response.json()

        try:
            timelines.append(
                asyncio.run(example.run(runner.base_url, endpoint, 5, send=send))
            )
        finally:
            runner.shutdown()

    expected = [
        ("root", 0, 20, 2),
        ("left", 5020, 5040, 2),
        ("right", 5020, 5080, 6),
        ("join", 5080, 5090, 1),
    ]
    assert timelines[0] == timelines[1]
    assert timelines[0] == [
        {
            "node": node,
            "arrival_ns": arrival * MILLISECOND,
            "completed_ns": completed * MILLISECOND,
            "output_tokens": output_tokens,
        }
        for node, arrival, completed, output_tokens in expected
    ]


@pytest.mark.parametrize("grant_terminal", [False, True])
def test_http_failure_aborts_and_reaps_pending_tasks(monkeypatch, grant_terminal):
    example = _load_example()

    async def exercise():
        original_tasks = asyncio.all_tasks()
        http_tasks, control_tasks = [], []
        waiting = asyncio.Event()
        error = RuntimeError("HTTP request rejected before scheduler admission")

        class Control:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def request(self, ready):
                control_tasks.append(asyncio.current_task())
                if ready.round == 0:
                    return Grant(
                        ready.run_id,
                        ready.round,
                        20 * MILLISECOND,
                        [
                            Terminal(
                                ready.operations[0].rid, "COMPLETED", 20 * MILLISECOND
                            )
                        ],
                    )
                if ready.round == 1:
                    return Grant(ready.run_id, ready.round, ready.next_wakeup_ns, [])
                assert ready.round == 2
                waiting.set()
                if grant_terminal:
                    return Grant(
                        ready.run_id,
                        ready.round,
                        5040 * MILLISECOND,
                        [
                            Terminal(
                                ready.operations[0].rid, "COMPLETED", 5040 * MILLISECOND
                            )
                        ],
                    )
                await asyncio.Event().wait()

        async def send(http, base_url, payload):
            http_tasks.append(asyncio.current_task())
            sequence = int(payload["rid"].rsplit(".", 1)[1], 16)
            if sequence == 0:
                return {"output_ids": [1, 1], "meta_info": {"completion_tokens": 2}}
            if sequence == 1:
                await asyncio.Event().wait()
            await waiting.wait()
            if grant_terminal:
                # Leave time to begin consuming the peer's terminal HTTP result.
                await asyncio.sleep(0.01)
            raise error

        monkeypatch.setattr(
            example, "AsyncCoordinatorClient", lambda endpoint: Control()
        )
        with pytest.raises(RuntimeError) as caught:
            await asyncio.wait_for(
                example.run("unused", "unused", send=send), timeout=1
            )
        assert caught.value is error
        assert len(http_tasks) == len(control_tasks) == 3
        assert all(task.done() for task in [*http_tasks, *control_tasks])
        assert asyncio.all_tasks() == original_tasks

    asyncio.run(exercise())
