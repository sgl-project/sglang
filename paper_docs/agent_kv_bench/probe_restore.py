"""Micro-probe: isolated L2-restore cost vs full recompute (quiet system).

Steps:
  1. Send request A (unique ~25k-token prefix) -> cached in L1 (+L2 backup).
  2. Send B, C, D (other unique 25k prefixes) -> force-evict A from L1.
  3. Re-send A's continuation -> TTFT = L2 restore path (no queue).
  4. Send E (fresh 25k prefix)  -> TTFT = full recompute (no queue).
Compare TTFT(A2) vs TTFT(E). Both max_tokens=1 (prefill-only).
"""
import sys
import time

import openai

sys.path.insert(0, "/home/xubowen/mxfp4/sglang/paper_docs/agent_kv_bench")
from harness import build_doc  # deterministic unique docs

client = openai.OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
MODEL = "/data/models/Qwen3-4B-Instruct-2507"
DOC_CHARS = 100000  # ~25k tokens


def msg(tag):
    return [{"role": "user", "content": (
        f"Repository snapshot {tag} (files in context):\n\n```python\n"
        f"{build_doc(abs(hash(tag)) % 89, 0, DOC_CHARS)}\n```\n\nReply with exactly: OK.")}]


def send(tag, max_tokens=1):
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model=MODEL, messages=msg(tag), temperature=0.001,
        max_tokens=max_tokens)
    dt = (time.perf_counter() - t0) * 1000
    return dt, r.usage.prompt_tokens


def main():
    dtA1, ptA = send("probeA")
    print(f"A cold    : {dtA1:8.0f} ms  prompt={ptA}")
    for t in ("probeB", "probeC", "probeD"):
        dt, pt = send(t)
        print(f"{t} fill  : {dt:8.0f} ms  prompt={pt}")
    time.sleep(2)  # let backup writes settle
    dtA2, ptA2 = send("probeA")
    print(f"A restore : {dtA2:8.0f} ms  prompt={ptA2}")
    dtE, ptE = send("probeE")
    print(f"E recomp  : {dtE:8.0f} ms  prompt={ptE}")
    # second round with different tags for stability
    send("probeA2")
    for t in ("probeF", "probeG", "probeH"):
        send(t)
    time.sleep(2)
    dtA4, _ = send("probeA2")
    print(f"A2 restore: {dtA4:8.0f} ms")
    dtE2, _ = send("probeI")
    print(f"I recomp  : {dtE2:8.0f} ms")


main()
