"""Real P->D transfers and native /generate requests, including the >128 boundary."""

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import httpx

ROOT = Path("pd-top-logprobs-results")
RESULTS = []
BOOTSTRAP_PORT = 17730
TIMEOUT = 150
P = "http://127.0.0.1:17710"
D = "http://127.0.0.1:17720"
G = "http://127.0.0.1:17700"


def save(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def payload(k, room=None, prompt=None):
    result = {
        "text": prompt or "The capital of France is",
        "sampling_params": {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
        "return_logprob": True,
        "top_logprobs_num": k,
        "stream": False,
    }
    if room is not None:
        result.update(
            bootstrap_host="127.0.0.1",
            bootstrap_port=BOOTSTRAP_PORT,
            bootstrap_room=room,
        )
    return result


def check_decode(value, k):
    meta = value["meta_info"]
    assert meta.get("finish_reason", {}).get("type") == "length", meta.get(
        "finish_reason"
    )
    assert meta["completion_tokens"] == 8, meta
    rows = meta["output_top_logprobs"]
    tokens = meta["output_token_logprobs"]
    assert len(rows) == 8 and len(tokens) == 8, (len(rows), len(tokens))
    for row in rows:
        assert len(row) == k, (k, len(row))
        assert len({int(item[1]) for item in row}) == k
        assert all(math.isfinite(float(item[0])) for item in row)
        assert all(
            float(row[i][0]) >= float(row[i + 1][0]) - 1e-6 for i in range(k - 1)
        )
    return meta


def record(name, **kwargs):
    row = {"name": name, **kwargs}
    RESULTS.append(row)
    save("results/progress.json", RESULTS)
    print(json.dumps(row, ensure_ascii=False), flush=True)


async def paired(client, k, index):
    room = (time.time_ns() // 1000 + index) % ((1 << 60) - 1)
    body = payload(k, room)
    save(f"results/requests/paired_{k}.json", body)
    begin = time.monotonic()
    d, p = await asyncio.gather(
        client.post(D + "/generate", json=body), client.post(P + "/generate", json=body)
    )
    save(
        f"results/responses/paired_{k}_prefill.json",
        {"status": p.status_code, "body": p.json()},
    )
    save(
        f"results/responses/paired_{k}_decode.json",
        {"status": d.status_code, "body": d.json()},
    )
    assert p.status_code == d.status_code == 200, (
        p.status_code,
        p.text[:2000],
        d.status_code,
        d.text[:2000],
    )
    dp = d.json()
    pp = p.json()
    dm = check_decode(dp, k)
    pm = pp["meta_info"]
    assert pm["completion_tokens"] == 1, pm
    assert len(pm["output_top_logprobs"][0]) == k
    first_p = pm["output_top_logprobs"][0]
    first_d = dm["output_top_logprobs"][0]
    assert [int(x[1]) for x in first_p] == [int(x[1]) for x in first_d]
    delta = max(abs(float(a[0]) - float(b[0])) for a, b in zip(first_p, first_d))
    assert delta == 0, delta
    assert int(pm["output_token_logprobs"][0][1]) == int(
        dm["output_token_logprobs"][0][1]
    )
    record(
        f"paired_{k}",
        status="passed",
        top_k=k,
        prefill_tokens=1,
        decode_tokens=8,
        first_token_top_logprobs_exact_match=True,
        first_token_max_abs_diff=delta,
        elapsed_s=time.monotonic() - begin,
        bootstrap_room=room,
    )


async def gateway(client, name, k, stream=False, prompt=None):
    body = payload(k, prompt=prompt)
    body["stream"] = stream
    save(f"results/requests/{name}.json", body)
    begin = time.monotonic()
    if stream:
        value = None
        events = []
        async with client.stream("POST", G + "/generate", json=body) as response:
            assert response.status_code == 200, (
                response.status_code,
                await response.aread(),
            )
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    value = json.loads(line[6:])
                    events.append(value)
        assert value is not None
        save(f"results/responses/{name}_events.json", events)
    else:
        response = await client.post(G + "/generate", json=body)
        assert response.status_code == 200, (response.status_code, response.text[:3000])
        value = response.json()
    save(f"results/responses/{name}.json", value)
    meta = check_decode(value, k)
    record(
        name,
        status="passed",
        top_k=k,
        stream=stream,
        completion_tokens=8,
        output_rows=len(meta["output_top_logprobs"]),
        row_widths=[len(x) for x in meta["output_top_logprobs"]],
        elapsed_s=time.monotonic() - begin,
    )


async def main(args):
    global ROOT, P, D, G, BOOTSTRAP_PORT, TIMEOUT
    ROOT = Path(args.output_dir).expanduser().resolve()
    P = args.prefill_url.rstrip("/")
    D = args.decode_url.rstrip("/")
    G = args.router_url.rstrip("/")
    BOOTSTRAP_PORT = args.bootstrap_port
    TIMEOUT = args.timeout
    for folder in ("results", "evidence"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(
        timeout=TIMEOUT, limits=httpx.Limits(max_connections=32)
    ) as client:
        # Router health can become ready before the PD workers are registered.
        start = time.monotonic()
        while True:
            try:
                response = await client.get(G + "/readiness")
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            assert time.monotonic() - start < TIMEOUT, "Router did not become ready"
            await asyncio.sleep(2)
        for role, url in [("prefill", P), ("decode", D)]:
            info = (await client.get(url + "/server_info")).json()
            assert info["device"] == "npu" and info["disaggregation_mode"] == role
            assert info["disaggregation_transfer_backend"] == "ascend"
            assert info["disaggregation_max_top_logprobs_num"] == 256
            save("evidence/server_info_" + role + ".json", info)
        for k in (1, 128, 129, 256):
            await paired(client, k, k)
        for k in (128, 129, 256):
            await gateway(client, f"gateway_{k}", k)
        await gateway(client, "gateway_stream_256", 256, True)
        await asyncio.gather(
            *(
                gateway(
                    client,
                    f"concurrent_{i}",
                    256,
                    prompt=f"Question {i}: explain why water freezes. Answer:",
                )
                for i in range(4)
            )
        )
        # Reject before bootstrap or queue admission on both PD roles.
        for role, url in [("prefill", P), ("decode", D)]:
            body = payload(257, time.time_ns() // 1000)
            response = await client.post(url + "/generate", json=body)
            save(
                f"results/responses/overflow_257_{role}.json",
                {"status": response.status_code, "body": response.text},
            )
            assert response.status_code == 400, (
                role,
                response.status_code,
                response.text[:2000],
            )
            assert (
                "257" in response.text
                and "256" in response.text
                and "capacity" in response.text
            )
            record("overflow_257_" + role, status="passed", http_status=400)
        await gateway(client, "gateway_after_rejection_256", 256)
        for role, url in [("prefill", P), ("decode", D)]:
            assert (await client.get(url + "/health")).status_code == 200
            response = await client.get(url + "/metrics")
            (ROOT / "evidence" / f"metrics_{role}_after.txt").write_text(response.text)
    save(
        "results/summary.json",
        {
            "status": "passed",
            "tests": RESULTS,
            "device": "Ascend NPU",
            "transfer_backend": "ascend",
            "transfer_protocol": "sdma",
            "max_top_logprobs_num": 256,
            "source_revision": args.source_revision,
            "test_scope": "Same-host two-NPU Prefill/Decode with real P->D KV and first-token metadata transfer; direct paired HTTP and standard router requests",
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-url", default="http://127.0.0.1:17710")
    parser.add_argument("--decode-url", default="http://127.0.0.1:17720")
    parser.add_argument("--router-url", default="http://127.0.0.1:17700")
    parser.add_argument("--bootstrap-port", type=int, default=17730)
    parser.add_argument("--output-dir", default="pd-top-logprobs-results")
    parser.add_argument("--timeout", type=float, default=150)
    parser.add_argument(
        "--source-revision", help="Optional source revision for the result summary"
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
