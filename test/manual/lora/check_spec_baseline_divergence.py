"""Does a spec-on vs spec-off divergence need LoRA at all?

Run this when the spec+LoRA harness
(``test/manual/lora/run_spec_lora_matrix.py``) reports a mismatch that
survives its filters. It runs the *same* speculative config with **no
adapters loaded**, comparing spec-off against spec-on greedy outputs and
sampling each side twice so run-to-run instability is visible.

If the base model diverges on the same prompt, the cause is the model and
the speculative path (EAGLE3 topk>1 tree verify in particular), not the LoRA
integration -- which is exactly what it showed for
Qwen3-30B-A3B-Instruct-2507 on prompt #2: DIFFERS, and UNSTABLE-self, with
zero adapters involved.

The harness's own filters cannot answer this: they can tell whether a pair
reproduces within one server config, but not whether the divergence depends
on LoRA being present. That needs this second config.

Usage (from a checkout, with a GPU):
    python test/manual/lora/check_spec_baseline_divergence.py
"""

import os
import subprocess
import sys
import time

import requests

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "List three primary colors.",
    "Write a one-sentence story about a brave detective on Mars.",
    "Explain what a hash table is in two sentences.",
]
BASE = "http://127.0.0.1:31000"
COMMON = [
    "--tp",
    "4",
    "--moe-runner-backend",
    "triton",
    "--attention-backend",
    "flashinfer",
    "--prefill-attention-backend",
    "fa4",
    "--decode-attention-backend",
    "fa4",
    "--mem-fraction-static",
    "0.8",
]
SPEC = [
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    "lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "4",
    "--speculative-num-draft-tokens",
    "8",
]


def launch(extra):
    env = dict(os.environ, SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="1")
    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "--port",
            "31000",
        ]
        + COMMON
        + extra,
        stdout=open("/scratch/loraspec/logs/attrib_server.log", "a"),
        stderr=subprocess.STDOUT,
        env=env,
    )
    for _ in range(120):
        time.sleep(10)
        try:
            if requests.get(BASE + "/health", timeout=3).ok:
                return p
        except Exception:
            pass
    raise SystemExit("server did not come up")


def gen():
    r = requests.post(
        BASE + "/generate",
        json={
            "text": PROMPTS,
            "sampling_params": {"temperature": 0, "max_new_tokens": 32},
        },
        timeout=900,
    )
    r.raise_for_status()
    return [x["text"] for x in r.json()]


results = {}
for label, extra in [("nospec", []), ("spec", SPEC)]:
    p = launch(extra)
    try:
        results[label] = [gen(), gen()]
    finally:
        p.terminate()
        p.wait(timeout=120)
    time.sleep(10)

print("=" * 70)
for i, prompt in enumerate(PROMPTS):
    a1, a2 = results["nospec"][0][i], results["nospec"][1][i]
    b1, b2 = results["spec"][0][i], results["spec"][1][i]
    stable = "stable" if (a1 == a2 and b1 == b2) else "UNSTABLE-self"
    verdict = "same" if a1 == b1 else "DIFFERS"
    print(f"prompt#{i}: nospec-vs-spec={verdict} ({stable})")
    if a1 != b1:
        print(f"   nospec: {a1!r}")
        print(f"   spec  : {b1!r}")
