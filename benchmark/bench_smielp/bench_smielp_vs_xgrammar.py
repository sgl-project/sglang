"""
Phase 4 — Part B: Benchmark SMIELP vs XGrammar JSON extraction.

Measures end-to-end latency (time-to-first-structured-result) for:
  A) SMIELP  — single forward pass, logit slicing, result in meta_info["smielp"]
  B) XGrammar — JSON schema constrained decoding, N tokens for N output fields

Both methods classify intent + fill entity slots on the same flight-booking prompts.

Usage (Linux + GPU, sglang installed):
    # Terminal 1 — start server
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-0.5B-Instruct \
        --enable-custom-logit-processor \
        --port 30000

    # Terminal 2 — run benchmark
    python benchmark/bench_smielp/bench_smielp_vs_xgrammar.py \
        --host http://localhost:30000 \
        --warmup 5 \
        --n-requests 100 \
        --batch-sizes 1 4 8 16

Output: latency table + throughput comparison printed to stdout.
"""

import argparse
import json
import statistics
import sys
import time
from typing import Optional

# ── Imports (graceful for offline/dev environments) ────────────────────────────
try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed.  pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    from transformers import AutoTokenizer
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

# Add project python path so logit_slicing is importable.
import importlib.util, pathlib
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_SRC = _PROJECT_ROOT / "python" / "sglang" / "srt" / "logit_slicing"

def _load_module(dotted_name, filename):
    spec = importlib.util.spec_from_file_location(dotted_name, _SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = dotted_name.rsplit(".", 1)[0]
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod

_schema_mod = _load_module("sglang.srt.logit_slicing.schema",       "schema.py")
_anchor_mod = _load_module("sglang.srt.logit_slicing.vocab_anchor", "vocab_anchor.py")
_proc_mod    = _load_module("sglang.srt.logit_slicing.processor",         "processor.py")
_proc_b_mod  = _load_module("sglang.srt.logit_slicing.processor_phase_b", "processor_phase_b.py")

SimultaneousMultiIntentEntityLogitProcessor = _proc_mod.SimultaneousMultiIntentEntityLogitProcessor
SMIELPWithHiddenStates = _proc_b_mod.SMIELPWithHiddenStates
IntentSchema = _schema_mod.IntentSchema
NERSchema    = _schema_mod.NERSchema
SlotSchema   = _schema_mod.SlotSchema
build_anchor_config  = _anchor_mod.build_anchor_config
build_phase_b_config = _anchor_mod.build_phase_b_config
STRATEGY_FIRST_TOKEN = _anchor_mod.STRATEGY_FIRST_TOKEN

# ── NER Schema ────────────────────────────────────────────────────────────────

INTENTS = ["book", "cancel", "status"]   # single-token labels in Qwen2.5

NER_SCHEMA = NERSchema(
    intents=IntentSchema(labels=INTENTS),
    slots=[
        SlotSchema(name="city",  labels=["O", "B", "I"]),
        SlotSchema(name="date",  labels=["O", "B", "I"]),
        SlotSchema(name="hotel", labels=["none", "present"]),
    ],
)

# JSON schema for XGrammar baseline — equivalent output structure.
XGRAMMAR_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "intent":   {"type": "string", "enum": INTENTS},
        "city_tag": {"type": "string", "enum": ["O", "B", "I"]},
        "date_tag": {"type": "string", "enum": ["O", "B", "I"]},
        "hotel":    {"type": "string", "enum": ["none", "present"]},
    },
    "required": ["intent", "city_tag", "date_tag", "hotel"],
    "additionalProperties": False,
}

# ── Test prompts ───────────────────────────────────────────────────────────────

PROMPTS = [
    "I want to book a flight to Paris next Tuesday and stay at the Marriott.",
    "Cancel my reservation for the flight to London.",
    "What is the status of my booking to New York?",
    "Book me a flight to Tokyo next week, no hotel needed.",
    "I need to check on my upcoming flight to Berlin.",
    "Please cancel my hotel in Rome along with the flight.",
    "Is my flight to Sydney still confirmed?",
    "Book a one-way ticket to Dubai arriving on Friday.",
]

SYSTEM_PROMPT = (
    "You are a flight booking assistant. "
    "Classify the user intent and extract city, date, and hotel slot information."
)


def _chat_messages(user_text: str) -> list:
    return [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_text},
    ]


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _post(host: str, endpoint: str, payload: dict, timeout: int = 60) -> dict:
    url = f"{host}{endpoint}"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _check_server_ready(host: str) -> bool:
    try:
        resp = requests.get(f"{host}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# ── SMIELP benchmark ───────────────────────────────────────────────────────────

def _build_smielp_params(tokenizer) -> dict:
    """Build the custom_params dict for the SMIELP processor."""
    return build_anchor_config(NER_SCHEMA, tokenizer, strategy=STRATEGY_FIRST_TOKEN)


def _run_smielp_request(host: str, prompt: str, custom_params: dict, proc_str: str) -> tuple[float, dict]:
    """Fire one SMIELP request and return (latency_s, result_dict)."""
    payload = {
        "model": "default",
        "messages": _chat_messages(prompt),
        "max_tokens": 1,
        "temperature": 0.0,
        "extra_body": {
            "custom_logit_processor": proc_str,
            "custom_params": custom_params,
        },
    }
    t0 = time.perf_counter()
    data = _post(host, "/v1/chat/completions", payload)
    latency = time.perf_counter() - t0

    smielp_result = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("meta_info", {})
        .get("smielp", [{}])[0]
    )
    return latency, smielp_result


def _run_smielp_phase_b_request(
    host: str, prompt: str, custom_params: dict, proc_str: str
) -> tuple[float, dict]:
    """Fire one Phase B request (with hidden_states) and return (latency_s, result_dict)."""
    payload = {
        "model": "default",
        "messages": _chat_messages(prompt),
        "max_tokens": 1,
        "temperature": 0.0,
        "extra_body": {
            "custom_logit_processor": proc_str,
            "custom_params": custom_params,
            "return_hidden_states": True,
        },
    }
    t0 = time.perf_counter()
    data = _post(host, "/v1/chat/completions", payload)
    latency = time.perf_counter() - t0
    smielp_result = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("meta_info", {})
        .get("smielp", [{}])[0]
    )
    return latency, smielp_result


def _run_xgrammar_request(host: str, prompt: str) -> tuple[float, dict]:
    """Fire one XGrammar JSON-constrained request and return (latency_s, result_dict)."""
    payload = {
        "model": "default",
        "messages": _chat_messages(prompt),
        "max_tokens": 64,
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ner_result",
                "schema": XGRAMMAR_JSON_SCHEMA,
            },
        },
    }
    t0 = time.perf_counter()
    data = _post(host, "/v1/chat/completions", payload)
    latency = time.perf_counter() - t0

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {"raw": content}
    return latency, result


# ── Metrics ────────────────────────────────────────────────────────────────────

def _stats(latencies: list[float]) -> dict:
    return {
        "n":      len(latencies),
        "mean":   statistics.mean(latencies) * 1000,
        "median": statistics.median(latencies) * 1000,
        "p95":    sorted(latencies)[int(0.95 * len(latencies))] * 1000,
        "min":    min(latencies) * 1000,
        "max":    max(latencies) * 1000,
    }


def _print_table(
    smielp_a_stats: dict,
    smielp_b_stats: Optional[dict],
    xgrammar_stats: dict,
    n_requests: int,
    model: str,
):
    header = f"\n{'='*80}"
    print(header)
    print(f"  SMIELP (A+B) vs XGrammar — model: {model}  n={n_requests}")
    print(header)
    col = f"  {'Metric':<12}  {'Phase A (ms)':>14}  "
    if smielp_b_stats:
        col += f"{'Phase B (ms)':>14}  "
    col += f"{'XGrammar (ms)':>14}  {'SpeedupA':>10}"
    if smielp_b_stats:
        col += f"  {'SpeedupB':>10}"
    print(col)
    print(f"  {'-'*72}")
    for key in ("mean", "median", "p95", "min", "max"):
        a = smielp_a_stats[key]
        x = xgrammar_stats[key]
        ratio_a = x / a if a > 0 else float("inf")
        row = f"  {key:<12}  {a:>14.1f}  "
        if smielp_b_stats:
            b = smielp_b_stats[key]
            ratio_b = x / b if b > 0 else float("inf")
            row += f"{b:>14.1f}  "
        row += f"{x:>14.1f}  {ratio_a:>9.2f}x"
        if smielp_b_stats:
            row += f"  {ratio_b:>9.2f}x"
        print(row)
    print(header)
    speedup_a = xgrammar_stats["median"] / smielp_a_stats["median"]
    print(f"  Phase A median speedup: {speedup_a:.2f}x  (1 decode token vs ~20-40)")
    if smielp_b_stats:
        speedup_b = xgrammar_stats["median"] / smielp_b_stats["median"]
        overhead  = smielp_b_stats["median"] - smielp_a_stats["median"]
        print(f"  Phase B median speedup: {speedup_b:.2f}x  (hidden_state overhead: +{overhead:.1f} ms)")
    print(header)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark SMIELP vs XGrammar")
    parser.add_argument("--host",        default="http://localhost:30000", help="sglang server URL")
    parser.add_argument("--model",       default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--warmup",      type=int,  default=5,     help="warmup requests (not counted)")
    parser.add_argument("--n-requests",  type=int,  default=50,    help="timed requests per method")
    parser.add_argument("--timeout",     type=int,  default=120,   help="per-request HTTP timeout (s)")
    parser.add_argument("--phase-b",     action="store_true",      help="also benchmark Phase B (requires --enable-return-hidden-states on server)")
    args = parser.parse_args()

    print(f"[bench] Checking server at {args.host} …")
    if not _check_server_ready(args.host):
        print(
            f"ERROR: server not reachable at {args.host}.\n"
            "Start it with:\n"
            f"  python -m sglang.launch_server --model-path {args.model} "
            "--enable-custom-logit-processor --port 30000",
            file=sys.stderr,
        )
        sys.exit(1)
    print("[bench] Server ready.")

    if not _HAS_TRANSFORMERS:
        print("ERROR: 'transformers' not installed.  pip install transformers", file=sys.stderr)
        sys.exit(1)

    print(f"[bench] Loading tokenizer: {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    custom_params_a = _build_smielp_params(tokenizer)
    proc_a_str = SimultaneousMultiIntentEntityLogitProcessor.to_str()
    print(f"[bench] Phase A schema anchored. Processor serialized ({len(proc_a_str)} bytes).")

    # Phase B: also needs the embedding matrix
    custom_params_b = None
    proc_b_str = None
    if args.phase_b:
        try:
            import torch as _torch
            from transformers import AutoModelForCausalLM
            print(f"[bench] Loading model for Phase B embedding matrix …")
            m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=_torch.float16, device_map="cpu")
            emb = m.model.embed_tokens.weight.detach().float()
            del m
            custom_params_b = build_phase_b_config(NER_SCHEMA, tokenizer, emb, strategy=STRATEGY_FIRST_TOKEN)
            proc_b_str = SMIELPWithHiddenStates.to_str()
            print(f"[bench] Phase B config built. Processor serialized ({len(proc_b_str)} bytes).")
        except Exception as e:
            print(f"[bench] Phase B unavailable: {e}. Skipping Phase B.")

    prompts_cycle = (PROMPTS * ((args.n_requests + args.warmup) // len(PROMPTS) + 1))

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f"[bench] Warming up ({args.warmup} requests each method) …")
    for i in range(args.warmup):
        p = prompts_cycle[i]
        _run_smielp_request(args.host, p, custom_params_a, proc_a_str)
        if custom_params_b:
            _run_smielp_phase_b_request(args.host, p, custom_params_b, proc_b_str)
        _run_xgrammar_request(args.host, p)
    print("[bench] Warmup done.")

    # ── Phase A timed run ─────────────────────────────────────────────────────
    print(f"[bench] Timing Phase A SMIELP ({args.n_requests} requests) …")
    smielp_a_latencies = []
    for i in range(args.n_requests):
        p = prompts_cycle[args.warmup + i]
        lat, result = _run_smielp_request(args.host, p, custom_params_a, proc_a_str)
        smielp_a_latencies.append(lat)
        if i < 3:
            print(f"  [{i}] {lat*1000:.0f} ms  intent={result.get('intent', {}).get('label', '?')}")

    # ── Phase B timed run (if available) ──────────────────────────────────────
    smielp_b_latencies = []
    if custom_params_b:
        print(f"[bench] Timing Phase B SMIELP ({args.n_requests} requests) …")
        for i in range(args.n_requests):
            p = prompts_cycle[args.warmup + i]
            lat, result = _run_smielp_phase_b_request(args.host, p, custom_params_b, proc_b_str)
            smielp_b_latencies.append(lat)
            if i < 3:
                mode = result.get("mode", "?")
                print(f"  [{i}] {lat*1000:.0f} ms  intent={result.get('intent', {}).get('label', '?')} mode={mode}")

    # ── XGrammar timed run ────────────────────────────────────────────────────
    print(f"[bench] Timing XGrammar ({args.n_requests} requests) …")
    xgrammar_latencies = []
    for i in range(args.n_requests):
        p = prompts_cycle[args.warmup + i]
        lat, result = _run_xgrammar_request(args.host, p)
        xgrammar_latencies.append(lat)
        if i < 3:
            print(f"  [{i}] {lat*1000:.0f} ms  intent={result.get('intent', '?')}")

    # ── Report ────────────────────────────────────────────────────────────────
    _print_table(
        _stats(smielp_a_latencies),
        _stats(smielp_b_latencies) if smielp_b_latencies else None,
        _stats(xgrammar_latencies),
        n_requests=args.n_requests,
        model=args.model,
    )


if __name__ == "__main__":
    main()
