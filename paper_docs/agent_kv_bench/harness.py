"""Agent KV-cache pressure harness (innovation #2 evaluation).

Drives N concurrent multi-turn agent sessions against an OpenAI-compatible
SGLang endpoint and measures the cost of the *continuation* request (the turn
sent after a simulated tool execution), which is exactly the request whose
prefix either hits the radix cache (cheap incremental prefill) or was evicted
under pressure (full recompute).

Session structure per round r:
  1. action turn    : full conversation -> short assistant reply (cached)
  2. tool gap       : sleep(latency sampled from the tool's calibrated table)
  3. continuation   : conversation + tool result -> model continues.
                      ^ TTFT / prompt_tokens of THIS request is the metric.
                      hit  -> prompt_tokens ~= incremental (tool result only)
                      miss -> prompt_tokens ~= full prefix (regret: evicted)

Assistant turns reuse the model's actual output text, so the rendered token
prefix is continuous and radix-cache hits are exact. Tools are represented at
the message level (no API tool binding): the caching behavior only depends on
the token sequence, and the tool-call parser plays no role in it.

Usage (host):
  python harness.py --concurrency 32 --rounds 6 --out run_lru_n32.jsonl
"""
import argparse
import asyncio
import json
import math
import random
import statistics
import time

import openai

# --- calibrated tool latency table: name -> (mu, sigma) of lognormal seconds ---
TOOL_TABLE = {
    "quick_fs": (math.log(0.3), 0.5),    # ls / cat / grep-like: 0.1-1s
    "web_search": (math.log(2.0), 0.6),  # 1-5s
    "code_edit": (math.log(4.0), 0.6),   # 2-8s
    "run_tests": (math.log(10.0), 0.8),  # 4-30s
}
TOOL_MIX = {"quick_fs": 0.5, "web_search": 0.2, "run_tests": 0.2, "code_edit": 0.1}

SYSTEM = (
    "You are a coding agent working inside a repository. At each step, name the "
    "single tool you would call next (quick_fs, web_search, code_edit or "
    "run_tests) with its arguments, then stop. Keep the reply under 30 words."
)

TASKS = [
    "Investigate why the build fails and fix it.",
    "Add input validation to the public API endpoints.",
    "Reduce the latency of the search endpoint.",
    "Migrate the config loader to the new schema.",
    "Find and fix the flaky unit test in the CI suite.",
    "Optimize the CSV import path for large files.",
]

DOC_TMPL = (
    "def handler_{i}(payload, ctx=None):\n"
    "    # section {i}: validate and dispatch (rev {sess})\n"
    "    data = parse_payload(payload)\n"
    "    if not data or not data.get('id'):\n"
    "        return error_response(400, 'missing id')\n"
    "    result = transform(data, ctx)\n"
    "    return json_ok_{sess}(result)\n\n"
)


def build_doc(sess: int, start: int, target_chars: int) -> str:
    """Deterministic filler reaching ~target_chars, unique per (session, block):
    mimics repo files the agent keeps in context (what makes prefixes unique)."""
    parts, total, i = [], 0, start
    while total < target_chars:
        chunk = DOC_TMPL.format(i=i, sess=sess % 89)
        parts.append(chunk)
        total += len(chunk)
        i += 1
    return "".join(parts)


def sample_tool(rng: random.Random) -> str:
    x, acc = rng.random(), 0.0
    for name, w in TOOL_MIX.items():
        acc += w
        if x < acc:
            return name
    return "quick_fs"


def sample_latency(tool: str, rng: random.Random) -> float:
    mu, sigma = TOOL_TABLE[tool]
    return rng.lognormvariate(mu, sigma)


def tool_result_text(tool: str, rng: random.Random) -> str:
    if tool == "quick_fs":
        files = [f"src/module_{rng.randint(0, 99):02d}.py" for _ in range(6)]
        return "repository listing:\n" + "\n".join(files)
    if tool == "web_search":
        return (f"search result: recommended fix documented at "
                f"docs.example.com/guide/{rng.randint(100, 999)} with a code sample.")
    if tool == "code_edit":
        return (f"patch applied to src/module_{rng.randint(0, 99):02d}.py, "
                f"{rng.randint(3, 40)} lines changed.")
    failed = rng.randint(0, 3)
    return (f"test suite: {rng.randint(80, 300)} passed, {failed} failed "
            f"in {rng.uniform(4, 28):.1f}s"
            + (f"\nFAILED: test_case_{rng.randint(0, 999)}" if failed else ""))


class Session:
    def __init__(self, sid: int, rounds: int, base_chars: int, growth_chars: int,
                 rng: random.Random):
        self.sid = sid
        self.rounds = rounds
        self.rng = rng
        self.growth_chars = growth_chars
        self.doc_i = 0
        self.messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": (
                f"Repository snapshot (files in context):\n\n```python\n"
                f"{self.next_doc(base_chars)}\n```\n\n"
                f"Task: {TASKS[sid % len(TASKS)]}. Which tool do you call first?")},
        ]
        self.tools_planned = [sample_tool(rng) for _ in range(rounds)]
        self.gaps = [sample_latency(t, rng) for t in self.tools_planned]

    def next_doc(self, target_chars: int) -> str:
        doc = build_doc(self.sid, self.doc_i, target_chars)
        self.doc_i += max(1, target_chars // 300)
        return doc

    def grow(self, tool: str):
        """Unique context growth per round (retrieved document chunk)."""
        self.messages.append({"role": "user", "content": (
            f"Additional context retrieved via {tool}:\n\n```python\n"
            f"{self.next_doc(self.growth_chars)}\n```")})


async def run_turn(client, model, messages, max_tokens):
    """Returns (text, ttft_ms, prompt_tokens). TTFT from first streamed content."""
    t0 = time.perf_counter()
    ttft = None
    parts = []
    usage = None
    stream = await client.chat.completions.create(
        model=model, messages=messages, temperature=0.001, max_tokens=max_tokens,
        stream=True, stream_options={"include_usage": True},
    )
    async for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            parts.append(chunk.choices[0].delta.content)
    if ttft is None:
        ttft = (time.perf_counter() - t0) * 1000
    return "".join(parts).strip() or "(no text)", ttft, usage


async def run_session(sid, args, client, model, out_f, lock):
    rng = random.Random(args.seed * 100003 + sid)
    s = Session(sid, args.rounds, args.base_chars, args.growth_chars, rng)
    for r in range(s.rounds):
        tool, gap = s.tools_planned[r], s.gaps[r]
        # --- action turn (prefix gets cached server-side here) ---
        text1, ttft1, usage1 = await run_turn(client, model, s.messages, args.action_tokens)
        s.messages.append({"role": "assistant", "content": text1})
        # --- tool gap ---
        await asyncio.sleep(gap)
        # --- continuation turn (METRIC) ---
        result = tool_result_text(tool, rng)
        s.messages.append({"role": "user", "content": (
            f"Tool `{tool}` finished in {gap:.2f}s:\n{result}\n\n"
            f"Given this, which tool do you call next?")})
        text2, ttft2, usage2 = await run_turn(client, model, s.messages, args.action_tokens)
        s.messages.append({"role": "assistant", "content": text2})
        s.grow(tool)
        rec = {
            "sid": sid, "round": r, "tool": tool, "gap_s": round(gap, 3),
            "ttft_action_ms": round(ttft1, 1), "ttft_cont_ms": round(ttft2, 1),
            "prompt_action": usage1.prompt_tokens if usage1 else None,
            "prompt_cont": usage2.prompt_tokens if usage2 else None,
            "ts": time.time(),
        }
        async with lock:
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()


def summarize(recs, dur, args):
    cont_t = [r["ttft_cont_ms"] for r in recs]
    act_t = [r["ttft_action_ms"] for r in recs]
    pc = [r["prompt_cont"] for r in recs if r["prompt_cont"]]
    # Hit/miss classifier: sglang does not expose cached_tokens, but the TTFT
    # cost per prompt token separates cleanly (cache hit: prefill is only the
    # ~100-token increment -> ~0.01 ms/tok; miss: full recompute ~0.2-0.5 ms/tok).
    # miss  <=>  ttft_ms > 0.1 * prompt_tokens + 200
    def is_miss(r):
        return r["prompt_cont"] and r["ttft_cont_ms"] > 0.1 * r["prompt_cont"] + 200
    miss = [r for r in recs if is_miss(r)]
    hit = [r for r in recs if not is_miss(r)]
    lines = [
        f"===== KV-bench summary =====",
        f"config: N={args.concurrency} rounds={args.rounds} seed={args.seed} wall={dur:.0f}s requests={len(recs)}",
        f"action TTFT  ms: p50={statistics.median(act_t):.0f} mean={statistics.mean(act_t):.0f}",
        f"cont TTFT    ms: p50={statistics.median(cont_t):.0f} "
        f"p90={sorted(cont_t)[int(len(cont_t) * 0.9)]:.0f} mean={statistics.mean(cont_t):.0f}",
        f"evicted(miss): {len(miss)}/{len(recs)} = {len(miss) / len(recs) * 100:.1f}%",
    ]
    if hit:
        lines.append(f"  hit  TTFT p50={statistics.median([r['ttft_cont_ms'] for r in hit]):.0f}ms "
                     f"prompt p50={statistics.median([r['prompt_cont'] for r in hit]):.0f}")
    if miss:
        lines.append(f"  miss TTFT p50={statistics.median([r['ttft_cont_ms'] for r in miss]):.0f}ms "
                     f"prompt p50={statistics.median([r['prompt_cont'] for r in miss]):.0f}")
    lines.append(f"total continuation prompt tokens: {sum(pc)}")
    return "\n".join(lines)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:30000/v1")
    ap.add_argument("--model", default="/data/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--base-chars", type=int, default=9000, help="~2.2k tok initial doc")
    ap.add_argument("--growth-chars", type=int, default=5000, help="~1.2k tok growth/round")
    ap.add_argument("--action-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="kvbench_run.jsonl")
    args = ap.parse_args()

    client = openai.AsyncOpenAI(base_url=args.endpoint, api_key="EMPTY")
    lock = asyncio.Lock()
    out_f = open(args.out, "w")
    t_start = time.time()
    await asyncio.gather(*[run_session(i, args, client, args.model, out_f, lock)
                           for i in range(args.concurrency)])
    out_f.close()
    recs = [json.loads(l) for l in open(args.out)]
    summary = summarize(recs, time.time() - t_start, args)
    print(summary)
    with open(args.out.replace(".jsonl", "_summary.txt"), "w") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    asyncio.run(main())
