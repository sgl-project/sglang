"""Validate a running Ascend DSV4 + DSPARK mixed-chunk server.

Enable SGLANG_ENABLE_MIXED_PREFILL_SPEC_COORDINATION=1, --enable-mixed-chunk,
and --cuda-graph-backend-prefill disabled on the server. Use enough SWA cache
for the concurrent prompts and a chunk budget larger than one physical page.
Run with --url http://127.0.0.1:PORT --output /tmp/mixed-dspark-results.
For the flag-off control, pass --expect-mixed 0.
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import httpx


def mixed_steps(text):
    return sum(
        float(v)
        for v in re.findall(
            r"^sglang:mixed_verify_steps_total(?:\{[^\n]*\})? ([0-9.e+\-]+)$",
            text,
            re.MULTILINE,
        )
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:32480")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expect-mixed", type=int, choices=[0, 1], default=1)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(exist_ok=True, parents=True)
    timeout = httpx.Timeout(300, connect=10)
    async with httpx.AsyncClient(base_url=args.url, timeout=timeout) as client:
        for _ in range(180):
            try:
                response = await client.get("/health", timeout=3)
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)
        else:
            raise RuntimeError("server did not become healthy")

        before = (await client.get("/metrics")).text
        (out / "metrics-before.txt").write_text(before)
        info = await client.get("/server_info")
        (out / "server-info-before.json").write_text(info.text)
        results = []

        async def request(name, length, count, first=None, sampling=None):
            payload = {
                "input_ids": [100 + (i % 37) for i in range(length)],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": count,
                    "ignore_eos": True,
                    **(sampling or {}),
                },
                "stream": first is not None,
            }
            start = time.monotonic()
            if first is None:
                response = await client.post("/generate", json=payload)
                response.raise_for_status()
                value = response.json()
            else:
                value = None
                async with client.stream("POST", "/generate", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        value = json.loads(line[6:])
                        first.set()
            assert value is not None
            (out / f"{name}.json").write_text(
                json.dumps(value, ensure_ascii=False, indent=2)
            )
            meta = value["meta_info"]
            assert meta["prompt_tokens"] == length, meta
            assert meta["completion_tokens"] == count, meta
            assert not meta.get("finish_reason", {}).get("type") == "abort", meta
            results.append(
                {
                    "name": name,
                    "input_tokens": length,
                    "output_tokens": count,
                    "elapsed_seconds": time.monotonic() - start,
                    "spec_verify_ct": meta.get("spec_verify_ct"),
                    "spec_accept_length": meta.get("spec_accept_length"),
                }
            )

        for n in [127, 128, 129, 255, 256, 257]:
            await request(f"boundary-{n}", n, 16)
        for round_id in range(args.rounds):
            await client.get("/flush_cache")
            first = asyncio.Event()
            anchor = asyncio.create_task(request(f"anchor-{round_id}", 127, 128, first))
            await asyncio.wait_for(first.wait(), 120)
            await asyncio.gather(
                anchor,
                request(f"long-{round_id}-0", 1025, 32),
                request(f"long-{round_id}-1", 513, 48),
                request(f"long-{round_id}-2", 255, 32),
            )
        # Sampled decoding and penalties share the same heterogeneous row split.
        first = asyncio.Event()
        anchor = asyncio.create_task(
            request(
                "sampling-anchor",
                129,
                64,
                first,
                sampling={
                    "temperature": 0.7,
                    "top_k": 10,
                    "top_p": 0.9,
                    "repetition_penalty": 1.05,
                },
            )
        )
        await asyncio.wait_for(first.wait(), 120)
        await asyncio.gather(anchor, request("sampling-prefill", 1023, 32))

        after = (await client.get("/metrics")).text
        (out / "metrics-after.txt").write_text(after)
        info = await client.get("/server_info")
        (out / "server-info-after.json").write_text(info.text)
        delta = mixed_steps(after) - mixed_steps(before)
        summary = {
            "requests": len(results),
            "output_tokens": sum(r["output_tokens"] for r in results),
            "mixed_verify_steps": delta,
            "results": results,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        assert (delta > 0) == bool(args.expect_mixed), delta
        assert (await client.get("/health")).status_code == 200


if __name__ == "__main__":
    asyncio.run(main())
