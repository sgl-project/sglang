"""Fixed-order AgentX hash-block replay for paired GDN snapshot measurements."""

import argparse
import hashlib
import json
import random
import statistics
import time
from functools import lru_cache
from pathlib import Path

import requests

TRACE_REVISION = "23f152f6f0f9399a85901b89a6458def0ef16729"
DIRECTED = ((185, 3), (244, 4), (232, 13), (302, 11))


@lru_cache(maxsize=8192)
def block(session, value):
    rng = random.Random(f"{session}:{value}")
    return tuple(rng.randrange(1000, 150000) for _ in range(64))


def select_trace(path, selection):
    eligible = []
    directed = {}
    with path.open() as stream:
        for row, line in enumerate(stream):
            session = json.loads(line)
            if session["block_size"] != 64 or session["hash_id_scope"] != "local":
                continue
            requests_ = [r for r in session["requests"] if r["type"] in {"s", "n"}]
            if selection == "directed":
                for target, start in DIRECTED:
                    if row == target:
                        directed[row] = (session["id"], requests_[start : start + 3])
            else:
                turns = requests_[:8]
                if len(turns) != 8 or any(
                    not 0 < r["in"] <= 65000 or len(r["hash_ids"]) * 64 != r["in"]
                    for r in turns
                ):
                    continue
                score = hashlib.sha256(
                    f"branch-tail-ab-v1:{session['id']}".encode()
                ).hexdigest()
                eligible.append((score, session["id"], turns))
    if selection == "directed":
        assert len(directed) == len(DIRECTED)
        return [directed[row] for row, _ in DIRECTED]
    assert len(eligible) >= 12
    return [(session, turns) for _, session, turns in sorted(eligible)[:12]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--selection", choices=("directed", "unselected"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sessions = select_trace(args.trace, args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def flush():
        response = requests.post(
            args.base_url + "/flush_cache", params={"timeout": 30}, timeout=60
        )
        response.raise_for_status()

    def generate(session, req, rid, salt):
        ids = [token for value in req["hash_ids"] for token in block(session, value)]
        ids = ids[: req["in"]]
        assert len(ids) == req["in"]
        payload = {
            "rid": rid,
            "input_ids": ids,
            "cache_salt": salt,
            "stream": True,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1,
                "ignore_eos": True,
            },
        }
        start = time.perf_counter()
        ttft = None
        final = None
        with requests.post(
            args.base_url + "/generate", json=payload, stream=True, timeout=240
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                if not line.startswith(b"data: "):
                    continue
                value = line[6:].strip()
                if value == b"[DONE]":
                    break
                event = json.loads(value)
                assert "error" not in event, event
                if event.get("output_ids"):
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    final = event
        assert ttft is not None and final is not None
        meta = final["meta_info"]
        assert meta["prompt_tokens"] == len(ids) and meta["completion_tokens"] == 1
        return {
            "rid": rid,
            "prompt": len(ids),
            "cached": meta["cached_tokens"],
            "input_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "ttft_s": ttft,
            "wall_s": time.perf_counter() - start,
            "output_ids": final["output_ids"],
        }

    # Warm up the longest shape, then isolate every measured arm from warmup.
    session, req = max(
        ((s, r) for s, reqs in sessions for r in reqs), key=lambda x: x[1]["in"]
    )
    generate(session, req, "warmup", "warmup")
    flush()
    with args.output.with_suffix(".jsonl").open("w") as stream:

        def emit(row):
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(row["rid"], row["cached"], row["ttft_s"], flush=True)

        if args.selection == "directed":
            for i, (session, reqs) in enumerate(sessions):
                flush()
                for turn, req in enumerate(reqs):
                    row = generate(
                        session, req, f"directed-{i}-{turn}", f"directed-{session}"
                    )
                    row.update(session=i, turn=turn)
                    emit(row)
            # Same-length cold controls exercise no extra snapshot candidate.
            flush()
            cold = dict(reqs[1])
            cold["hash_ids"] = cold["hash_ids"][:94] + [999999]
            cold["in"] = 6049
            for i in range(4):
                row = generate(
                    "no-candidate", cold, f"no-candidate-{i}", f"no-candidate-{i}"
                )
                row.update(session=-1, turn=i)
                emit(row)
        else:
            for turn in range(8):
                for i, (session, reqs) in enumerate(sessions):
                    row = generate(
                        session,
                        reqs[turn],
                        f"unselected-{i}-{turn}",
                        f"unselected-{session}",
                    )
                    row.update(session=i, turn=turn)
                    emit(row)
    manifest = [
        (session, [(req["in"], req["hash_ids"]) for req in reqs])
        for session, reqs in sessions
    ]
    output = {
        "trace_revision": TRACE_REVISION,
        "manifest_sha256": hashlib.sha256(json.dumps(manifest).encode()).hexdigest(),
        "selection": args.selection,
        "requests": len(rows),
        "prompt_tokens": sum(row["prompt"] for row in rows),
        "cached_tokens": sum(row["cached"] for row in rows),
        "ttft_p50_s": statistics.median(row["ttft_s"] for row in rows),
        "replay": "sequential compressed hash-block reconstruction; one generated token",
    }
    args.output.with_suffix(".json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
