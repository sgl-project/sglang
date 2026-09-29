# Usage: record --url URL --prompts ids.json --output arm.json; compare --help.
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def request(url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        body = response.read()
        return (
            json.loads(body)
            if "json" in response.headers.get("Content-Type", "")
            else body.decode()
        )


def record(url, prompts, output):
    token_ids = json.loads(prompts.read_text())
    result = {
        "prompt_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
        "prompt_count": len(token_ids),
        "sampling_params": {"temperature": 0, "max_new_tokens": 64, "ignore_eos": True},
        "batches": {},
    }
    for concurrency in (1, 8):
        rows = []
        result["batches"][str(concurrency)] = rows
        for start in range(0, len(token_ids), concurrency):
            request(url=url, path="/flush_cache")
            responses = request(
                url=url,
                path="/generate",
                payload={
                    "input_ids": token_ids[start : start + concurrency],
                    "sampling_params": result["sampling_params"],
                    "return_logprob": True,
                    "logprob_start_len": -1,
                    "top_logprobs_num": 5,
                },
            )
            assert len(responses) == len(token_ids[start : start + concurrency])
            for response in responses:
                meta = response["meta_info"]
                tokens = meta["output_token_logprobs"]
                assert len(tokens) == 64, meta
                rows.append(
                    {
                        "index": len(rows),
                        "token_ids": [item[1] for item in tokens],
                        "logprobs": [item[0] for item in tokens],
                        "top_logprobs": meta["output_top_logprobs"],
                    }
                )
            output.write_text(json.dumps(result, indent=2) + "\n")


def compare(eager, graph, output):
    left, right = json.loads(eager.read_text()), json.loads(graph.read_text())
    assert left["prompt_sha256"] == right["prompt_sha256"]
    assert left["sampling_params"] == right["sampling_params"]
    assert left["prompt_count"] == right["prompt_count"]
    result = {"prompt_sha256": left["prompt_sha256"], "comparisons": {}}
    for concurrency in ("1", "8"):
        a, b = left["batches"][concurrency], right["batches"][concurrency]
        assert len(a) == len(b) == left["prompt_count"] and a
        rows = []
        for eager_row, graph_row in zip(a, b):
            same_prefix = True
            positions = []
            for pos, (token_a, token_b, lp_a, lp_b) in enumerate(
                zip(
                    eager_row["token_ids"],
                    graph_row["token_ids"],
                    eager_row["logprobs"],
                    graph_row["logprobs"],
                )
            ):
                positions.append(
                    {
                        "position": pos,
                        "eager_argmax": token_a,
                        "graph_argmax": token_b,
                        "argmax_equal": token_a == token_b,
                        "same_prefix": same_prefix,
                        "eager_logprob": lp_a,
                        "graph_logprob": lp_b,
                        "logprob_abs_delta": abs(lp_a - lp_b),
                    }
                )
                same_prefix = same_prefix and token_a == token_b
            rows.append(
                {
                    "index": eager_row["index"],
                    "equal": same_prefix,
                    "positions": positions,
                }
            )
        result["comparisons"][concurrency] = {
            "matched_prompts": sum(row["equal"] for row in rows),
            "total_prompts": len(rows),
            "matched_tokens": sum(
                p["argmax_equal"] for row in rows for p in row["positions"]
            ),
            "total_tokens": sum(len(row["positions"]) for row in rows),
            "max_shared_prefix_logprob_delta": max(
                p["logprob_abs_delta"]
                for row in rows
                for p in row["positions"]
                if p["same_prefix"]
            ),
            "rows": rows,
        }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return all(
        item["matched_prompts"] == item["total_prompts"]
        for item in result["comparisons"].values()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    rec = actions.add_parser("record")
    rec.add_argument("--url", required=True)
    rec.add_argument("--prompts", type=Path, required=True)
    rec.add_argument("--output", type=Path, required=True)
    cmp = actions.add_parser("compare")
    cmp.add_argument("--eager", type=Path, required=True)
    cmp.add_argument("--graph", type=Path, required=True)
    cmp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "record":
        record(url=args.url, prompts=args.prompts, output=args.output)
    else:
        raise SystemExit(
            not compare(eager=args.eager, graph=args.graph, output=args.output)
        )
