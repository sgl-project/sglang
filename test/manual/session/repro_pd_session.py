#!/usr/bin/env python3
"""Reproduce decode session KV ownership through two SGLang workers.

Python standard library only. Copy this file anywhere; no router, local
tokenizer or SGLang import is required. Inspect worker logs
for row/KV leaks as well as this client's exit status.
"""

import argparse
import concurrent.futures
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode", required=True, help="decode HTTP URL")
    parser.add_argument("--prefill", required=True, help="prefill HTTP URL")
    parser.add_argument("--bootstrap-host", help="prefill host reachable from decode")
    parser.add_argument("--bootstrap-port", type=int, default=8998)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    if min(args.cycles, args.turns, args.tokens, args.timeout, args.bootstrap_port) < 1:
        parser.error("counts, timeout and port must be positive")
    args.bootstrap_host = (
        args.bootstrap_host or urllib.parse.urlparse(args.prefill).hostname
    )

    def post(base, path, payload):
        headers = {"Content-Type": "application/json"}
        if os.environ.get("SGLANG_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["SGLANG_API_KEY"]
        req = urllib.request.Request(
            base.rstrip("/") + path, json.dumps(payload).encode(), headers
        )
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                body = response.read().decode()
        except urllib.error.HTTPError as error:
            with error:
                detail = error.read().decode()[:1000]
            raise RuntimeError(f"{path}: HTTP {error.code}: {detail}") from error
        data = json.loads(body) if body else None
        if isinstance(data, dict):
            reason = data.get("meta_info", {}).get("finish_reason") or {}
            if (
                data.get("error")
                or data.get("success") is False
                or reason.get("type") == "abort"
            ):
                raise RuntimeError(f"{path}: {data}")
        return data

    # Tokenize on the server; retain exact generated IDs, without text roundtrips.
    chunks = [
        post(
            args.decode,
            "/tokenize",
            {
                "prompt": f" Synthetic turn {turn}: continue counting.",
                "add_special_tokens": turn == 0,
            },
        )["tokens"]
        for turn in range(args.turns)
    ]
    for cycle in range(args.cycles):
        sid = "pd-repro-" + uuid.uuid4().hex
        history = []
        error = None
        try:
            opened = post(
                args.decode,
                "/open_session",
                {"session_id": sid, "streaming": True, "capacity_of_str_len": 1000000},
            )
            actual = opened.get("session_id") if isinstance(opened, dict) else opened
            if actual != sid:
                raise RuntimeError(f"Unexpected opened session: {opened}")
            for turn, suffix in enumerate(chunks):
                full = history + suffix
                common = {
                    "bootstrap_host": args.bootstrap_host,
                    "bootstrap_port": args.bootstrap_port,
                    "bootstrap_room": uuid.uuid4().int & ((1 << 63) - 1),
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": args.tokens,
                        "ignore_eos": True,
                        "no_stop_trim": True,
                    },
                }
                # Prefill is stateless: its single sampled token is not decode's
                # multi-token history. Only decode owns and appends the session.
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    prefill = pool.submit(
                        post, args.prefill, "/generate", {**common, "input_ids": full}
                    )
                    decode = pool.submit(
                        post,
                        args.decode,
                        "/generate",
                        {**common, "input_ids": suffix, "session_params": {"id": sid}},
                    )
                    result = decode.result()
                    prefill.result()
                meta = result["meta_info"]
                generated = result.get("output_ids")
                if meta["prompt_tokens"] != len(full):
                    raise RuntimeError(
                        f"Lost history: expected {len(full)} prompt tokens, got {meta}"
                    )
                if (
                    not isinstance(generated, list)
                    or len(generated) != meta["completion_tokens"]
                ):
                    raise RuntimeError(
                        "Native /generate output_ids are required for exact history replay"
                    )
                history = full + generated
                print(
                    json.dumps(
                        {
                            "cycle": cycle,
                            "turn": turn,
                            "prompt_tokens": len(full),
                            "completion_tokens": len(generated),
                            "cached_tokens": meta.get("cached_tokens"),
                            "output_ids": generated,
                        }
                    ),
                    flush=True,
                )
        except Exception as caught:
            error = caught
        finally:
            try:
                post(args.decode, "/close_session", {"session_id": sid})
            except Exception as cleanup:
                print(
                    json.dumps({"cleanup_error": str(cleanup), "session_id": sid}),
                    flush=True,
                )
                error = error or cleanup
        if error:
            raise error
    print(
        json.dumps(
            {
                "http_checks_passed": True,
                "cycles": args.cycles,
                "note": "Also check worker invariants for request-row and KV leaks.",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"failed": True, "message": str(error)}), flush=True)
        raise SystemExit(1)
