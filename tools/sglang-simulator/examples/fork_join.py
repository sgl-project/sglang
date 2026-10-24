"""A reference driver: root -> tool wait -> two parallel requests -> join."""

import argparse
import asyncio
import json
import math
import uuid

import aiohttp
from sglang_simulator.simulation.client import AsyncCoordinatorClient
from sglang_simulator.simulation.coordinator import Ready, Submit


async def generate(http, base_url, payload):
    async with http.post(f"{base_url}/generate", json=payload) as response:
        response.raise_for_status()
        return await response.json()


async def run(base_url, coordinator, tool_delay_s=5.0, *, send=generate):
    if not math.isfinite(tool_delay_s) or tool_delay_s < 0:
        raise ValueError("tool_delay_s must be nonnegative and finite")
    run_id = uuid.uuid4().hex
    tasks, names, arrivals, results = {}, {}, {}, {}
    operations, events = [], []
    now_ns = round_id = 0
    wakeup_ns = None
    prompt = [17] * 8
    # HTTP can fail before the scheduler sees its payload and grants any event.
    failure = asyncio.get_running_loop().create_future()

    def check_http(task):
        if not task.cancelled() and (error := task.exception()) is not None:
            if not failure.done():
                failure.set_result(error)

    async def checked(awaitable):
        task = asyncio.ensure_future(awaitable)
        try:
            await asyncio.wait((task, failure), return_when=asyncio.FIRST_COMPLETED)
            if failure.done():
                raise failure.result()
            return await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    # Transport capacity must not prevent a round's payloads from arriving.
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with (
        aiohttp.ClientSession(
            connector=connector, timeout=aiohttp.ClientTimeout(total=30)
        ) as http,
        AsyncCoordinatorClient(coordinator) as control,
    ):

        def submit(name, input_ids, output_length):
            rid = f"{run_id}.{len(names):016x}"
            names[rid] = name
            arrivals[name] = now_ns
            tasks[rid] = asyncio.create_task(
                send(
                    http,
                    base_url,
                    {
                        "rid": rid,
                        "input_ids": input_ids,
                        "sampling_params": {
                            "max_new_tokens": output_length,
                            "ignore_eos": True,
                        },
                    },
                )
            )
            tasks[rid].add_done_callback(check_http)
            operations.append(Submit(rid))

        try:
            submit("root", prompt, 2)
            while True:
                grant = await checked(
                    control.request(
                        Ready(
                            run_id,
                            round_id,
                            operations,
                            next_wakeup_ns=wakeup_ns,
                            finished=len(results) == 4,
                        )
                    )
                )
                operations = []
                round_id += 1
                now_ns = grant.now_ns
                for terminal in grant.terminals:
                    if terminal.kind != "COMPLETED":
                        raise RuntimeError(f"request failed: {terminal}")
                    # Do not await other branches: they may need another Ready.
                    body = await checked(tasks[terminal.rid])
                    del tasks[terminal.rid]
                    name = names[terminal.rid]
                    results[name] = body["output_ids"]
                    events.append(
                        {
                            "node": name,
                            "arrival_ns": arrivals[name],
                            "completed_ns": terminal.at_ns,
                            "output_tokens": body["meta_info"]["completion_tokens"],
                        }
                    )
                    if name == "root":
                        wakeup_ns = now_ns + round(tool_delay_s * 1e9)

                # Apply all workload transitions before yielding the clock.
                if wakeup_ns is not None and now_ns >= wakeup_ns:
                    wakeup_ns = None
                    prefix = prompt + results["root"]
                    submit("left", prefix + [19] * 8, 2)
                    submit("right", prefix + [23] * 8, 6)
                if "left" in results and "right" in results and len(names) == 3:
                    submit("join", prompt + results["left"] + results["right"], 1)
                if grant.status == "FINISHED":
                    return events
        finally:
            # Failure aborts this run; closing HTTP alone is not a simulated Cancel.
            for task in tasks.values():
                task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--coordinator", default="tcp://127.0.0.1:30001")
    parser.add_argument("--tool-delay", type=float, default=5.0)
    args = parser.parse_args()
    if not math.isfinite(args.tool_delay) or args.tool_delay < 0:
        parser.error("--tool-delay must be nonnegative and finite")
    print(
        json.dumps(
            asyncio.run(run(args.base_url, args.coordinator, args.tool_delay)), indent=2
        )
    )
