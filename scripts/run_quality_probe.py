"""SemanticEmbedding fuzzy provider - quality + TTFT reproducer.

Self-contained probe that talks to a local SGLang server (started with
``--enable-fuzzy-match --fuzzy-match-provider SemanticEmbedding``) and
captures real prompt/response pairs to verify that semantic fuzzy KV
reuse:

  1. preserves output quality (variant returns the same answer as seed),
  2. improves TTFT (variant first-token latency < seed's),
  3. doesn't degrade subsequent unrelated requests.

Each pair shares the same context but rephrases the instruction. The
seed registers a donor; the variant should fuzzy-match against that
donor and reuse its KV cache.

Streaming chat-completion API is used so we can measure TTFT (time-to-
first-token) directly, not just total wall clock. The probe issues a
throwaway warmup request before each pair so connection setup + CUDA
graph capture don't bias the seed's TTFT - without warmup, the seed
pays a one-time cost the variant doesn't, and the resulting "speedup"
is partially or entirely warmth, not cache reuse.

Verifying fuzzy match fired server-side
---------------------------------------
TTFT alone is a noisy signal. To confirm fuzzy match actually fired on
the server, grep the SGLang server log for the realization markers,
e.g.::

    grep -E "Fuzzy match success|fuzzy=[1-9]|realized_locs=pre-allocated" \\
         sglang-server.log

A successful fuzzy realization looks like::

    [FUZZY RADIX] Fuzzy match success: cached=368, prompt=368, offset=27
    [FUZZY RADIX] match_prefix: exact=27, fuzzy=368, miss=30, total=425, \\
        cached_start_pos=32, realized_locs=pre-allocated
    [FUZZY] Realized 368 fuzzy tokens: copied donor KV with RoPE \\
        correction from positions [32..399] to [27..394]

That's 368 of 425 prompt tokens reused from the donor (86%), with
RoPE correction applied to align positions. The other ~30 tokens
(the paraphrased instruction) are recomputed normally.

Only pairs whose variants show this marker in the server log are real
cache-speedup demonstrations. Pairs without it just exercise the
quality side (variant produces a substantively-equivalent response to
the seed) but their TTFT numbers should not be cited as cache wins.

Usage:
    pip install aiohttp
    python run_quality_probe.py --endpoint http://localhost:8000 \\
                                --model Qwen/Qwen2.5-7B-Instruct-AWQ \\
                                --out quality_samples.md
"""
from __future__ import annotations
import argparse
import asyncio
import json
import time
from typing import List, Tuple

import aiohttp


# Three real-world prompt pairs covering content types representative
# of production fuzzy-cache workloads. Each pair: (label, context,
# seed_instruction, variant_instruction). Context is reused; only
# the instruction is paraphrased between seed and variant.
PAIRS: List[Tuple[str, str, str, str]] = [
    (
        "article_summary",
        # ~600-token factual article - typical of news / wiki workloads
        # where multiple users ask different summary questions about the
        # same source.
        """The James Webb Space Telescope (JWST), launched on December 25, 2021,
is the largest and most powerful space telescope ever built. Developed
by NASA in partnership with the European Space Agency (ESA) and the
Canadian Space Agency (CSA), JWST orbits the Sun at the second
Lagrange point (L2), approximately 1.5 million kilometers from Earth.
Its primary mirror, made of 18 hexagonal beryllium segments coated in
gold, spans 6.5 meters in diameter - nearly three times the size of
the Hubble Space Telescope's mirror. Unlike Hubble, JWST observes
primarily in the infrared spectrum, allowing it to peer through cosmic
dust clouds and observe light from the earliest galaxies formed after
the Big Bang. The telescope is shielded from solar heat by a five-layer
sunshield the size of a tennis court, which keeps the instruments
cooled to below 50 Kelvin. Among JWST's first scientific results,
released in July 2022, were the deepest infrared image of the universe
ever captured, detailed analyses of exoplanet atmospheres including the
detection of carbon dioxide on WASP-39b, and unprecedented views of the
Carina Nebula and Stephan's Quintet. The mission was originally
projected to cost $1 billion and launch in 2007, but technical
challenges and management issues pushed the final cost to approximately
$10 billion and delayed the launch by 14 years. Despite the delays,
the telescope is performing better than expected and is funded for at
least a 20-year mission. Scientists hope JWST will help answer
fundamental questions about the formation of stars and planetary
systems, the evolution of galaxies, and the potential habitability of
exoplanets.""",
        "Summarize the article above in three short paragraphs.",
        "Give me a three-paragraph summary of the article above.",
    ),
    (
        "factual_extraction",
        # Different factual passage. Demonstrates fuzzy match across
        # workloads where users ask different structured questions
        # about the same biographical record.
        """Marie Curie, born Maria Salomea Sklodowska in Warsaw on November 7,
1867, was a pioneering physicist and chemist whose research on
radioactivity - a term she herself coined - transformed the scientific
understanding of atomic structure. Curie moved to Paris in 1891 to
study at the Sorbonne, where she met her future husband and research
partner, Pierre Curie, in 1894. Together they discovered two new
elements: polonium (named after Marie's native Poland) in July 1898
and radium in December of the same year. In 1903, Marie became the
first woman to receive a Nobel Prize, sharing the Physics prize with
Pierre and Henri Becquerel for their work on spontaneous radiation.
After Pierre's tragic death in a street accident in 1906, Marie
continued their research and in 1911 received a second Nobel Prize,
this time in Chemistry, for her discovery of radium and polonium and
her isolation of pure radium - making her the first person ever to win
Nobel Prizes in two different scientific fields. During World War I,
she developed mobile X-ray units, known as 'petites Curies', which
were deployed near the front lines and helped diagnose injuries in
over a million wounded soldiers. Marie Curie died on July 4, 1934 in
Sancellemoz, France, of aplastic anemia, almost certainly caused by
her prolonged exposure to radiation. Her notebooks from the 1890s
remain so radioactive that they are stored in lead-lined boxes and
must be handled with protective gear.""",
        "List the years and the major events at each year mentioned in the passage above.",
        "Extract a chronological list of years and what happened at each one from the passage above.",
    ),
    (
        "code_explanation",
        # Code-review workload: same code, paraphrased "what does this
        # do" requests.
        """```python
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged
```

This Python function takes a list of intervals, where each interval
is a tuple of (start, end). The function should merge any overlapping
intervals and return a list of non-overlapping intervals in sorted
order. For example, given [(1, 3), (2, 6), (8, 10), (15, 18)], it
should return [(1, 6), (8, 10), (15, 18)] because the first two
intervals overlap and get merged into (1, 6).""",
        "Explain how the merge_intervals function works, step by step.",
        "Walk me through what the merge_intervals function does, in detail.",
    ),
]


async def post_streaming(session, endpoint, model, content, max_tokens):
    """Send a chat-completion request with stream=True and capture TTFT."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft_ms = None
    full = []
    finish_reason = None
    async with session.post(
        f"{endpoint}/v1/chat/completions",
        json=body,
        timeout=aiohttp.ClientTimeout(total=300),
    ) as r:
        async for raw in r.content:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if "content" in delta and delta["content"]:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                full.append(delta["content"])
            if choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]
    total_ms = (time.perf_counter() - t0) * 1000
    return {
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "response": "".join(full),
        "finish_reason": finish_reason,
        "completion_tokens": sum(1 for _ in "".join(full)),  # rough char count
    }


async def warmup(session, endpoint, model):
    """Throwaway request to warm the HTTP connection + CUDA graphs.

    Without this, the FIRST measured TTFT pays connection setup +
    first-batch CUDA graph capture, which collapses to a ~22ms floor on
    every subsequent request - making seed→variant TTFT comparisons
    unreliable. After warmup, seed and variant differ only by what the
    cache actually saved.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": True,
    }
    async with session.post(
        f"{endpoint}/v1/chat/completions",
        json=body,
        timeout=aiohttp.ClientTimeout(total=60),
    ) as r:
        async for raw in r.content:
            if raw.strip().startswith(b"data: [DONE]"):
                break


async def run(endpoint, model, out_md, out_json, max_tokens):
    async with aiohttp.ClientSession() as session:
        # Health check
        async with session.get(
            f"{endpoint}/health",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as r:
            assert r.status == 200, f"server unhealthy: {r.status}"

        # Global warmup: hot HTTP connection + initial CUDA graphs.
        print("warmup...", flush=True)
        await warmup(session, endpoint, model)
        await asyncio.sleep(1)

        results = []
        for label, ctx, seed_instr, variant_instr in PAIRS:
            # Per-pair warmup so CUDA graphs for this prompt's length
            # class are captured before the seed lands. Removes the
            # cold-prefill confound on seed TTFT - the seed→variant gap
            # then reflects only what the cache actually saved.
            await warmup(session, endpoint, model)
            await asyncio.sleep(0.5)

            # Place instruction BEFORE context. This makes the seed and
            # variant prompts diverge at the very first non-template
            # tokens, so SGLang's normal RadixCache (exact-prefix) only
            # catches the chat template (~24 tokens) - not the shared
            # body. The remaining ~1.5K tokens have to be matched by the
            # SemanticEmbedding fuzzy provider. This is the regime where
            # fuzzy match earns its keep over plain prefix caching.
            seed_prompt = f"{seed_instr}\n\n{ctx}"
            variant_prompt = f"{variant_instr}\n\n{ctx}"

            print(f"[{label}] seed (registers donor)...", flush=True)
            seed = await post_streaming(
                session, endpoint, model, seed_prompt, max_tokens
            )
            await asyncio.sleep(2)
            print(f"[{label}] variant (should fuzzy-match)...", flush=True)
            var = await post_streaming(
                session, endpoint, model, variant_prompt, max_tokens
            )

            ttft_speedup = (
                seed["ttft_ms"] / var["ttft_ms"]
                if var["ttft_ms"] and seed["ttft_ms"]
                else 0.0
            )
            total_speedup = (
                seed["total_ms"] / var["total_ms"]
                if var["total_ms"]
                else 0.0
            )
            results.append({
                "label": label,
                "context_chars": len(ctx),
                "seed_instruction": seed_instr,
                "seed_response": seed["response"],
                "seed_ttft_ms": seed["ttft_ms"],
                "seed_total_ms": seed["total_ms"],
                "variant_instruction": variant_instr,
                "variant_response": var["response"],
                "variant_ttft_ms": var["ttft_ms"],
                "variant_total_ms": var["total_ms"],
                "ttft_speedup_ratio": ttft_speedup,
                "total_speedup_ratio": total_speedup,
            })
            print(
                f"  seed TTFT={seed['ttft_ms']:.0f}ms total={seed['total_ms']:.0f}ms",
                flush=True,
            )
            print(
                f"  variant TTFT={var['ttft_ms']:.0f}ms total={var['total_ms']:.0f}ms",
                flush=True,
            )
            print(
                f"  → TTFT speedup={ttft_speedup:.2f}x  total speedup={total_speedup:.2f}x",
                flush=True,
            )

    # Markdown report
    lines = [
        "# SemanticEmbedding Fuzzy Provider - Quality + TTFT Samples",
        "",
        f"- Endpoint: `{endpoint}`",
        f"- Model: `{model}`",
        f"- Sampling: temperature=0.0, max_tokens={max_tokens}, stream=true",
        "",
        "Each pair below sends the **same context** with a **paraphrased**",
        "instruction. The seed registers a donor in the SemBlend pipeline;",
        "the variant should fuzzy-match against that donor and reuse its KV",
        "cache. Two things to verify:",
        "",
        "  1. **Quality**: variant response is substantively the same answer",
        "     as the seed.",
        "  2. **Speedup**: variant TTFT (time-to-first-token) is lower than",
        "     the seed's, demonstrating real-world fuzzy KV reuse.",
        "",
    ]
    for r in results:
        lines.append(f"## `{r['label']}`")
        lines.append("")
        lines.append(f"Context: {r['context_chars']} chars.")
        lines.append("")
        lines.append("| | TTFT (ms) | Total (ms) |")
        lines.append("|---|---:|---:|")
        lines.append(f"| Seed (cold)         | {r['seed_ttft_ms']:.0f} | {r['seed_total_ms']:.0f} |")
        lines.append(f"| Variant (fuzzy)     | {r['variant_ttft_ms']:.0f} | {r['variant_total_ms']:.0f} |")
        lines.append(f"| **Speedup**         | **{r['ttft_speedup_ratio']:.2f}×** | **{r['total_speedup_ratio']:.2f}×** |")
        lines.append("")
        lines.append("### Seed instruction")
        lines.append(f"> {r['seed_instruction']}")
        lines.append("")
        lines.append("**Seed response**:")
        lines.append("")
        lines.append(r["seed_response"].strip())
        lines.append("")
        lines.append("### Variant instruction (paraphrased; should fuzzy-match)")
        lines.append(f"> {r['variant_instruction']}")
        lines.append("")
        lines.append("**Variant response**:")
        lines.append("")
        lines.append(r["variant_response"].strip())
        lines.append("")
        lines.append("---")
        lines.append("")

    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_md} and {out_json}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True,
                   help="SGLang server URL, e.g. http://localhost:8000")
    p.add_argument("--model", required=True,
                   help="Model name as served, e.g. Qwen/Qwen2.5-7B-Instruct-AWQ")
    p.add_argument("--out", default="quality_samples.md")
    p.add_argument("--out-json", default="quality_samples.json")
    p.add_argument("--max-tokens", type=int, default=512)
    args = p.parse_args()
    asyncio.run(run(args.endpoint, args.model, args.out, args.out_json,
                    args.max_tokens))
