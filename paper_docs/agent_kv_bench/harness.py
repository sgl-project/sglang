"""Agent KV-cache pressure harness (innovation #2 evaluation).

Drives N concurrent multi-turn agent sessions against an OpenAI-compatible
SGLang endpoint using the standard tool-calling loop, and measures the cost of
the *continuation* request (the turn sent after a simulated tool execution),
whose prefix either hits the radix cache (cheap incremental prefill) or was
evicted under pressure (full recompute).

Session structure per round r:
  1. action turn    : conversation + tools -> model emits <tool_call> (the
                      finished turn's KV gets inserted into the radix cache)
  2. tool gap       : sleep(latency sampled from the called tool's table)
  3. continuation   : tool result appended -> model continues (METRIC TURN:
                      TTFT + prompt_tokens; hit ~ incremental prefill, miss ~
                      full recompute)

TTFT is measured on streaming continuation turns. sglang does not expose
cached_tokens, so hit/miss is classified from TTFT-per-prompt-token (the two
regimes differ by >20x at long contexts).

Usage (host):
  python harness.py --concurrency 32 --rounds 6 --out run.jsonl
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

TOOLS = [
    {"type": "function", "function": {"name": "quick_fs", "description": "Fast filesystem operation: list, read or search files in the repo.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "pattern": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the web for documentation or solutions.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "code_edit", "description": "Apply an edit to a source file.",
     "parameters": {"type": "object", "properties": {"file": {"type": "string"}, "change": {"type": "string"}}, "required": ["file", "change"]}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run the project test suite and return results.",
     "parameters": {"type": "object", "properties": {"scope": {"type": "string"}}, "required": ["scope"]}}},
]

SYSTEM = (
    "You are a coding agent working inside a repository. Use the provided tools "
    "to make progress on the task, one tool call per turn."
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
        parts.append(DOC_TMPL.format(i=i, sess=sess % 89))
        total += len(parts[-1])
        i += 1
    return "".join(parts)


# Workload-defined tool mix: each round the harness samples a tool from this
# distribution and instructs the model to call it (a benchmark property, not
# left to the model's preference, which skews heavily toward quick tools).
TOOL_MIX = {"quick_fs": 0.5, "web_search": 0.2, "run_tests": 0.2, "code_edit": 0.1}

TOOL_EXAMPLE = {
    "quick_fs": '{"path": "src/"}',
    "web_search": '{"query": "proper fix for this failure"}',
    "code_edit": '{"file": "src/main.py", "change": "guard the empty case"}',
    "run_tests": '{"scope": "all"}',
}


def sample_latency(tool: str, rng: random.Random, scale: float = 1.0) -> float:
    mu, sigma = TOOL_TABLE[tool]
    return rng.lognormvariate(mu, sigma) * scale


def sample_tool(rng: random.Random) -> str:
    """Workload-defined tool mix (the harness instructs the model per round)."""
    x, acc = rng.random(), 0.0
    for name, w in TOOL_MIX.items():
        acc += w
        if x < acc:
            return name
    return "quick_fs"


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
    def __init__(self, sid: int, base_chars: int, growth_chars: int):
        self.sid = sid
        self.growth_chars = growth_chars
        self.doc_i = 0
        self.messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": (
                f"Repository snapshot (files in context):\n\n```python\n"
                f"{self.next_doc(base_chars)}\n```\n\n"
                f"Task: {TASKS[sid % len(TASKS)]}. Use a tool to start.")},
        ]

    def next_doc(self, target_chars: int) -> str:
        doc = build_doc(self.sid, self.doc_i, target_chars)
        self.doc_i += max(1, target_chars // 300)
        return doc

    def grow(self):
        """Unique context growth per round (retrieved document chunk)."""
        self.messages.append({"role": "user", "content": (
            f"Additional retrieved context:\n\n```python\n"
            f"{self.next_doc(self.growth_chars)}\n```")})


async def run_action_turn(client, model, messages, max_tokens):
    """Non-streaming tool-calling turn. Returns (assistant_msg, prompt_tokens)."""
    resp = await client.chat.completions.create(
        model=model, messages=messages, tools=TOOLS, tool_choice="auto",
        temperature=0.001, max_tokens=max_tokens,
    )
    msg = resp.choices[0].message
    prompt_tokens = resp.usage.prompt_tokens if resp.usage else None
    # normalize to an OpenAI history message
    hist = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        hist["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return hist, prompt_tokens


async def run_continuation_turn(client, model, messages, max_tokens):
    """Streaming turn; TTFT = first streamed delta (content or tool call)."""
    t0 = time.perf_counter()
    ttft = None
    usage = None
    stream = await client.chat.completions.create(
        model=model, messages=messages, tools=TOOLS, tool_choice="auto",
        temperature=0.001, max_tokens=max_tokens, stream=True,
        stream_options={"include_usage": True},
    )
    async for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        if chunk.choices:
            d = chunk.choices[0].delta
            if (d and d.content) or (d and d.tool_calls):
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
    if ttft is None:
        ttft = (time.perf_counter() - t0) * 1000
    return ttft, usage.prompt_tokens if usage else None


async def run_session(sid, args, client, model, out_f, lock):
    rng = random.Random(args.seed * 100003 + sid)
    s = Session(sid, args.base_chars, args.growth_chars)
    for r in range(args.rounds):
        # --- instruct the tool for this round (workload-defined mix) ---
        intended = sample_tool(rng)
        s.messages.append({"role": "user", "content": (
            f"Next step: use the `{intended}` tool with arguments like "
            f"`{TOOL_EXAMPLE[intended]}`.")})
        # --- action turn: model emits a tool call; prefix cached server-side ---
        hist, prompt_a = await run_action_turn(client, model, s.messages, args.action_tokens)
        s.messages.append(hist)
        calls = hist.get("tool_calls") or []
        if not calls:  # model replied in prose; use the intended tool anyway
            calls = [{"id": f"fallback_{sid}_{r}", "type": "function",
                      "function": {"name": intended, "arguments": TOOL_EXAMPLE[intended]}}]
        # --- tool gap: parallel calls overlap; wait for the slowest ---
        tools_called = [tc["function"]["name"] for tc in calls]
        gaps = [sample_latency(t, rng, args.latency_scale) for t in tools_called]
        gap = max(gaps)
        await asyncio.sleep(gap)
        # --- continuation turn (METRIC) ---
        for tc in calls:
            s.messages.append({"role": "tool", "tool_call_id": tc["id"], "content":
                               tool_result_text(tc["function"]["name"], rng)})
        ttft_c, prompt_c = await run_continuation_turn(client, model, s.messages, args.action_tokens)
        # close the assistant turn for history continuity
        s.messages.append({"role": "assistant", "content": "Continuing."})
        s.grow()
        rec = {
            "sid": sid, "round": r, "tools": tools_called, "intended": intended,
            "gap_s": round(gap, 3),
            "ttft_action_ms": None, "ttft_cont_ms": round(ttft_c, 1),
            "prompt_action": prompt_a, "prompt_cont": prompt_c,
            "ts": time.time(),
        }
        async with lock:
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()


def summarize(recs, dur, args):
    cont_t = [r["ttft_cont_ms"] for r in recs]
    pc = [r["prompt_cont"] for r in recs if r["prompt_cont"]]
    # miss <=> ttft_ms > 0.1 * prompt_tokens + 200 (hit ~0.01 ms/tok, miss ~0.2-1)
    def is_miss(r):
        return r["prompt_cont"] and r["ttft_cont_ms"] > 0.1 * r["prompt_cont"] + 200
    miss = [r for r in recs if is_miss(r)]
    hit = [r for r in recs if not is_miss(r)]
    gapw = [(r["gap_s"], r["ttft_cont_ms"]) for r in recs]
    # recompute-token cost actually paid: prompt_tokens of missed turns
    wasted = sum(r["prompt_cont"] for r in miss if r["prompt_cont"])
    lines = [
        "===== KV-bench summary =====",
        f"config: N={args.concurrency} rounds={args.rounds} seed={args.seed} wall={dur:.0f}s requests={len(recs)}",
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
    # gap-conditional miss rate: fast tools (<=1s) vs slow (>1s)
    fast = [r for r in recs if r["gap_s"] <= 1.0]
    slow = [r for r in recs if r["gap_s"] > 1.0]
    if fast:
        lines.append(f"  fast-tool(<=1s) n={len(fast)} miss={sum(map(is_miss, fast)) / len(fast) * 100:.0f}%")
    if slow:
        lines.append(f"  slow-tool(>1s)  n={len(slow)} miss={sum(map(is_miss, slow)) / len(slow) * 100:.0f}%")
    lines.append(f"wasted recompute tokens (miss turns): {wasted}")
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
    ap.add_argument("--action-tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--latency-scale", type=float, default=1.0,
                    help="multiply all sampled tool latencies (sensitivity analysis)")
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
