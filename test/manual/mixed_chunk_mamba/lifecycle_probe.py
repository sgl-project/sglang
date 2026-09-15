#!/usr/bin/env python3
"""Deterministic mixed-chunk checkpoint lifecycle probe for sglang #39342.

Two phases, run against separately launched servers of the same build:
  --phase mixed  start K long decoders, wait until they stream, then send
                 P (shared ~5k-token system prefix + question), R (same prefix,
                 other question) and P2 (identical to P) while they decode.
                 Every chunk of P is scheduled into a MIXED batch.
  --phase ref    the same P/R/P2 with nothing else running (pure EXTEND).
                 Launch this server with chunked_prefill_size reduced by K so
                 the chunk boundaries and checkpoint depths coincide with the
                 mixed phase (mixed prefill budget = chunked_prefill_size - K).
The scheduler log (probe_instrumentation.patch) yields per request:
claim (pool, slot, depth) -> forward (tracked rows, one-token tails) ->
donate (depth, slot, fingerprint, result) -> restore (matched, slot,
fingerprint). compare_probe.py joins the two phases by depth.
"""

import argparse
import ast
import json
import random
import re
import threading
import time

import requests

CLAIM = re.compile(
    r"\[39342\] claim rid=(\S+) pool=(\S+) slot=(\S+) depth=(\S+) track_seqlen=(\S+) "
    r"last_idx=(\S+) next_idx=(\S+) extend=\[(\d+),(\d+)\)"
)
MIXED = re.compile(
    r"\[39342\] mixed bs=(\d+) extend_rows=(\d+) decode_rows=(\d+) pools=(\[.*?\]) "
    r"rids=(\[.*?\]) mask=(\S.*?) seqlens=(\S.*)$"
)
FORWARD = re.compile(
    r"\[39342\] forward mode=(\S+) bs=(\d+) one_token_rows=(\S+) pools=(\[.*?\]) tracked=(.*)$"
)
DONATE = re.compile(
    r"\[39342\] donate rid=(\S+) kind=(\S+) chunked=(\S+) depth=(\S+) slot=(\S+) "
    r"mamba_exist=(\S+) fp=(\(.*\))"
)
DONATE_RESULT = re.compile(
    r"\[39342\] donate-result rid=(\S+) is_finished=(\S+) mamba_exist=(\S+)"
)
DONATE_SKIP = re.compile(r"\[39342\] donate-skip rid=(\S+) kind=(\S+)")
RESTORE = re.compile(
    r"\[39342\] restore rid=(\S+) matched=(\d+) slot=(\S+) fp=(\(.*\))"
)


def build_prefix(seed, n_facts):
    rng = random.Random(seed)
    facts = [rng.randint(0, 999) for _ in range(n_facts)]
    prefix = " ".join(f"fact{i}: the value is {facts[i]}." for i in range(n_facts))
    return prefix, facts


def question(i):
    return f"What is fact{i}? Answer with the number only."


class Client:
    def __init__(self, url, model):
        self.url, self.model = url, model

    def flush(self):
        requests.post(f"{self.url}/flush_cache", timeout=300).raise_for_status()
        time.sleep(1.0)

    def _body(self, rid, prefix, user, max_tokens, ignore_eos=False, stream=False):
        return {
            "model": self.model,
            "rid": rid,
            "messages": [
                {"role": "system", "content": prefix},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "ignore_eos": ignore_eos,
            "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def chat(self, rid, prefix, user, max_tokens):
        t0 = time.perf_counter()
        r = requests.post(
            f"{self.url}/v1/chat/completions",
            json=self._body(rid, prefix, user, max_tokens),
            timeout=1200,
        )
        r.raise_for_status()
        b = r.json()
        usage = b.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        return {
            "rid": rid,
            "text": b["choices"][0]["message"]["content"],
            "latency_s": time.perf_counter() - t0,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

    def stream_decoder(self, rid, user, max_tokens, started, sink):
        r = requests.post(
            f"{self.url}/v1/chat/completions",
            json=self._body(
                rid,
                "You are a storyteller.",
                user,
                max_tokens,
                ignore_eos=True,
                stream=True,
            ),
            stream=True,
            timeout=1800,
        )
        n = 0
        for line in r.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            if line.strip() == b"data: [DONE]":
                break
            n += 1
            if n >= 3:
                started.set()
        started.set()
        sink[rid] = n


def parse_events(text):
    ev = {
        "claim": [],
        "mixed": [],
        "forward": [],
        "donate": [],
        "donate_result": [],
        "donate_skip": [],
        "restore": [],
    }
    for line in text.splitlines():
        if "[39342]" not in line:
            continue
        m = CLAIM.search(line)
        if m:
            ev["claim"].append(
                dict(
                    rid=m.group(1),
                    pool=int(m.group(2)),
                    slot=int(m.group(3)),
                    depth=int(m.group(4)),
                    track_seqlen=int(m.group(5)),
                    last_idx=int(m.group(6)),
                    next_idx=int(m.group(7)),
                    extend=(int(m.group(8)), int(m.group(9))),
                )
            )
            continue
        m = MIXED.search(line)
        if m:
            ev["mixed"].append(
                dict(
                    bs=int(m.group(1)),
                    extend_rows=int(m.group(2)),
                    decode_rows=int(m.group(3)),
                    pools=ast.literal_eval(m.group(4)),
                    rids=ast.literal_eval(m.group(5)),
                    mask=ast.literal_eval(m.group(6)),
                    seqlens=ast.literal_eval(m.group(7)),
                )
            )
            continue
        m = FORWARD.search(line)
        if m:
            tracked = m.group(5)
            ev["forward"].append(
                dict(
                    mode=m.group(1),
                    bs=int(m.group(2)),
                    one_token_rows=None if m.group(3) == "None" else int(m.group(3)),
                    pools=ast.literal_eval(m.group(4)),
                    tracked=None if tracked == "None" else ast.literal_eval(tracked),
                )
            )
            continue
        m = DONATE.search(line)
        if m:
            ex = m.group(6)
            ev["donate"].append(
                dict(
                    rid=m.group(1),
                    kind=m.group(2),
                    chunked=m.group(3),
                    depth=None if m.group(4) == "None" else int(m.group(4)),
                    slot=int(m.group(5)),
                    mamba_exist=None if ex == "?" else ex == "True",
                    fp=ast.literal_eval(m.group(7)),
                )
            )
            continue
        m = DONATE_RESULT.search(line)
        if m:
            ev["donate_result"].append(
                dict(
                    rid=m.group(1),
                    is_finished=m.group(2) == "True",
                    mamba_exist=None if m.group(3) == "None" else m.group(3) == "True",
                )
            )
            # attach to the latest donate of that rid whose result is unknown
            for d in reversed(ev["donate"]):
                if d["rid"] == m.group(1) and d["mamba_exist"] is None:
                    d["mamba_exist"] = ev["donate_result"][-1]["mamba_exist"]
                    break
            continue
        m = DONATE_SKIP.search(line)
        if m:
            ev["donate_skip"].append(dict(rid=m.group(1), kind=m.group(2)))
            continue
        m = RESTORE.search(line)
        if m:
            ev["restore"].append(
                dict(
                    rid=m.group(1),
                    matched=int(m.group(2)),
                    slot=int(m.group(3)),
                    fp=ast.literal_eval(m.group(4)),
                )
            )
    return ev


def chain_for(ev, rid):
    claims = [c for c in ev["claim"] if c["rid"] == rid]
    out = []
    for c in claims:
        key = (c["pool"], c["slot"], c["track_seqlen"])
        hits = [
            dict(bs=f["bs"], one_token_rows=f["one_token_rows"], mode=f["mode"])
            for f in ev["forward"]
            if f["tracked"] and any(tuple(t) == key for t in f["tracked"])
        ]
        out.append(dict(claim=c, tracked_in_forwards=hits))
    mixed = [m for m in ev["mixed"] if rid in m["rids"]]
    return dict(
        rid=rid,
        claims=out,
        mixed_batches=len(mixed),
        mixed_decode_rows=[m["decode_rows"] for m in mixed],
        mixed_mask_for_rid=[
            m["mask"][m["rids"].index(rid)] if isinstance(m["mask"], list) else None
            for m in mixed
        ],
        donations=[d for d in ev["donate"] if d["rid"] == rid],
        donate_skips=[d for d in ev["donate_skip"] if d["rid"] == rid],
        restores=[r for r in ev["restore"] if r["rid"] == rid],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["mixed", "ref"], required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--decoders", type=int, default=8)
    ap.add_argument("--decode-tokens", type=int, default=600)
    ap.add_argument("--prefix-facts", type=int, default=380)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()

    cl = Client(args.url, args.model)
    prefix, facts = build_prefix(args.seed, args.prefix_facts)
    rng = random.Random(args.seed + 1)
    qi, qj = rng.sample(range(args.prefix_facts), 2)
    expected = {"P": str(facts[qi]), "R": str(facts[qj]), "P2": str(facts[qi])}
    name = args.phase
    with open(args.log, "rb") as f:
        f.seek(0, 2)
        offset = f.tell()

    cl.flush()
    threads, started, sink = [], [], {}
    if args.phase == "mixed":
        for k in range(args.decoders):
            evt = threading.Event()
            t = threading.Thread(
                target=cl.stream_decoder,
                args=(
                    f"D{k}-{name}",
                    f"Tell a very long story about the number {k}.",
                    args.decode_tokens,
                    evt,
                    sink,
                ),
                daemon=True,
            )
            t.start()
            threads.append(t)
            started.append(evt)
        for evt in started:
            evt.wait(timeout=180)
        time.sleep(0.5)
    out = {}
    out["P"] = cl.chat(f"P-{name}", prefix, question(qi), args.max_tokens)
    out["R"] = cl.chat(f"R-{name}", prefix, question(qj), args.max_tokens)
    out["P2"] = cl.chat(f"P2-{name}", prefix, question(qi), args.max_tokens)
    for t in threads:
        t.join(timeout=1800)
    time.sleep(1.0)
    with open(args.log, "r", errors="replace") as f:
        f.seek(offset)
        ev = parse_events(f.read())

    rids = [f"{k}-{name}" for k in ("P", "R", "P2")]
    report = {
        "tag": args.tag,
        "phase": args.phase,
        "expected": expected,
        "decoder_tokens_streamed": sink,
        "answers": {
            k: dict(
                v,
                expected=expected[k],
                correct=v["text"].strip().startswith(expected[k]),
            )
            for k, v in out.items()
        },
        "counts": {k: len(v) for k, v in ev.items()},
        "chains": {rid: chain_for(ev, rid) for rid in rids},
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    brief = {
        "phase": args.phase,
        "answers": {
            k: (v["text"][:20], v["correct"], v["prompt_tokens"], v["cached_tokens"])
            for k, v in report["answers"].items()
        },
        "P_claims": [
            (
                c["claim"]["depth"],
                c["claim"]["slot"],
                len(c["tracked_in_forwards"]),
                any(f["one_token_rows"] for f in c["tracked_in_forwards"]),
            )
            for c in report["chains"][f"P-{name}"]["claims"]
        ],
        "P_mixed_batches": report["chains"][f"P-{name}"]["mixed_batches"],
        "P_donations": [
            (d["depth"], d["slot"], d["kind"], d["mamba_exist"])
            for d in report["chains"][f"P-{name}"]["donations"]
        ],
        "R_restore": [
            (r["matched"], r["slot"]) for r in report["chains"][f"R-{name}"]["restores"]
        ],
        "P2_restore": [
            (r["matched"], r["slot"])
            for r in report["chains"][f"P2-{name}"]["restores"]
        ],
    }
    print(json.dumps(brief, default=str))


if __name__ == "__main__":
    main()
