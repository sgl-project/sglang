"""Real AISimulate, SGLang HTTP and Ready/Grant integration on a CPU host.

The simulator CPU workflow installs the pinned AISimulate extra and SGLang.
Missing SDKs fail this integration instead of turning the whole group into skips.
These tests use the repository's existing SOL/SILICON estimator examples.
"""

import asyncio
import importlib

import aiohttp
import pytest
from sglang_simulator.simulation.client import AsyncCoordinatorClient
from sglang_simulator.simulation.coordinator import Ready, Submit
from test_simulation_coordinated_serving import _generate, _start_server


async def _dependent_chain(base_url, endpoint, delay):
    tasks = []
    records = []
    async with (
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session,
        AsyncCoordinatorClient(endpoint) as client,
    ):
        try:
            for round_id, rid in enumerate(("sdk-root", "sdk-leaf")):
                task = asyncio.create_task(
                    _generate(
                        session,
                        base_url,
                        rid,
                        11 + round_id,
                        delay=delay,
                        input_length=128,
                        output_length=2,
                    )
                )
                tasks.append(task)
                grant = await client.request(
                    Ready("sdk-chain", round_id, [Submit(rid)])
                )
                assert [(event.rid, event.kind) for event in grant.terminals] == [
                    (rid, "COMPLETED")
                ]
                body = await task
                timing = body["meta_info"]["sglang_simulator"]
                assert timing["rid"] == rid
                assert timing["completion_tokens"] == 2
                assert timing["terminal_ns"] == grant.now_ns
                assert timing["arrival_ns"] < timing["first_token_ns"]
                assert timing["first_token_ns"] <= timing["last_token_ns"]
                assert timing["last_token_ns"] == timing["terminal_ns"]
                records.append(
                    (
                        grant.now_ns,
                        [
                            (event.rid, event.kind, event.at_ns)
                            for event in grant.terminals
                        ],
                        timing,
                        body["meta_info"]["prompt_tokens"],
                        body["meta_info"]["completion_tokens"],
                    )
                )
                # The child is released only after its parent's HTTP response.
                await asyncio.sleep(delay)
            finished = await client.request(Ready("sdk-chain", 2, [], finished=True))
            assert finished.status == "FINISHED"
            assert finished.now_ns == records[-1][0]
            assert not finished.terminals
            return records
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("database_mode", ["sol", "silicon"])
def test_real_aisimulate_closed_loop_is_independent_of_http_delay(
    tmp_path, monkeypatch, database_mode
):
    importlib.import_module("aisimulate.sdk")
    observed = []
    for index, delay in enumerate((0, 0.025)):
        runner, endpoint = _start_server(
            tmp_path / f"{database_mode}-{index}",
            monkeypatch,
            config_name=f"aisimulate_{database_mode}",
        )
        try:
            observed.append(
                asyncio.run(_dependent_chain(runner.base_url, endpoint, delay))
            )
        finally:
            runner.shutdown()
    assert observed[0] == observed[1]
