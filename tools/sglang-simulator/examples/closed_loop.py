"""Two dependent HTTP requests separated by a virtual tool delay."""

import argparse
import asyncio
import json
import uuid

import aiohttp
from sglang_simulator.simulation.client import AsyncCoordinatorClient
from sglang_simulator.simulation.coordinator import Ready, Submit


async def run(base_url: str, coordinator: str, tool_delay_s: float) -> list[dict]:
    run_id = uuid.uuid4().hex
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with (
        aiohttp.ClientSession(
            connector=connector, timeout=aiohttp.ClientTimeout(total=30)
        ) as http,
        AsyncCoordinatorClient(coordinator) as control,
    ):
        round_id = 0
        events = []
        for turn in range(2):
            rid = f"{run_id}.{turn:016x}"

            async def send(rid=rid, turn=turn):
                async with http.post(
                    f"{base_url}/generate",
                    json={
                        "rid": rid,
                        "input_ids": [17] * (32 + turn),
                        "sampling_params": {
                            "max_new_tokens": 4,
                            "ignore_eos": True,
                        },
                    },
                ) as response:
                    response.raise_for_status()
                    return await response.json()

            tasks = (
                asyncio.create_task(send()),
                asyncio.create_task(
                    control.request(Ready(run_id, round_id, [Submit(rid)]))
                ),
            )
            try:
                _, grant = await asyncio.gather(*tasks)
            finally:
                # A failed exchange aborts the run; reap its HTTP/control peer.
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            if len(grant.terminals) != 1 or grant.terminals[0].kind != "COMPLETED":
                raise RuntimeError(f"unexpected request terminal: {grant}")
            events.append({"turn": turn, "completed_ns": grant.now_ns})
            round_id += 1
            if turn == 0:
                await control.request(
                    Ready(
                        run_id,
                        round_id,
                        [],
                        next_wakeup_ns=grant.now_ns + round(tool_delay_s * 1e9),
                    )
                )
                round_id += 1
        await control.request(Ready(run_id, round_id, [], finished=True))
        return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--coordinator", default="tcp://127.0.0.1:30001")
    parser.add_argument("--tool-delay", type=float, default=5.0)
    args = parser.parse_args()
    if not 0 <= args.tool_delay < float("inf"):
        parser.error("--tool-delay must be nonnegative and finite")
    print(
        json.dumps(
            asyncio.run(run(args.base_url, args.coordinator, args.tool_delay)), indent=2
        )
    )
