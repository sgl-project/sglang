"""Loop-7 M0 oracle sweep — the budget-vs-scorer decider (fail-closed/binding).

For each NIAH trial: compute the needle's LOGICAL token span (robust: raw-prompt
offset mapping, no chat template, so offline tokens match the server's logical
domain exactly), register it via the cross-process trial file, then issue a raw
/generate with max_new_tokens=1. The server-side recall oracle (eager mode)
records, per (trial, layer), the needle's score rank and score-only recall@K on
the live all-reduced score tensor — which tells us whether a budget > 2048 would
recover the needle (budget-limited) or not (scorer-limited).

**Binding contract (fail-closed).** The sink is truncated before the measured
run, then after the sweep this driver READS the sink and asserts that EVERY
issued trial produced success records and that there are zero hard failure
markers (``span_out_of_range`` / ``exception``). A missing length (the old
silently-absent 64K case) or a fail-closed marker makes the sweep exit non-zero
instead of silently producing a partial artifact.

Requires the DS server booted EAGER (--disable-cuda-graph) with the recall
oracle activated CONFIG-BORNE (so it reaches the TP workers — env does not):
  --double-sparsity-config '{... , "recall_oracle": true}'
The server then writes the fixed default sink/trial-file paths; this driver uses
the same paths (env override, else the module defaults).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import requests  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "test", "manual"))
import test_double_sparsity_v32 as h  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from sglang.srt.layers.attention.double_sparsity import oracle_artifact_sink as sink  # noqa: E402

TOKENIZER_FILE = os.environ.get(
    "DS_TOKENIZER_FILE",
    "/cluster-storage/models/deepseek-ai/DeepSeek-V3.2/tokenizer.json",
)

# Hard failure markers — these mean the oracle measured wrongly and the run must
# fail. ``no_active_trial`` is soft (a stray non-trial decode can legitimately
# have no trial); it is reported but does not by itself fail the run.
_HARD_FAILURES = ("span_out_of_range", "exception")


def _generate_decode(base, prompt, decode_steps):
    """Issue a raw /generate that FORCES ``decode_steps`` decode forwards.

    DS selection (and thus the oracle hook) runs ONLY in decode (forward_mode
    DECODE), never prefill — and these instruction-style NIAH prompts emit EOS
    immediately on raw /generate, giving a prefill-only forward with NO decode
    step. ``ignore_eos`` forces continuation so the decode-time selector runs and
    the oracle records. The first output token comes from prefill, so request
    ``decode_steps + 1`` tokens to get ``decode_steps`` decode forwards. Returns
    the server-measured prompt_tokens.
    """
    r = requests.post(
        f"{base}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": int(decode_steps) + 1,
                "temperature": 0,
                "ignore_eos": True,
            },
        },
        timeout=600,
    )
    return r.json()["meta_info"].get("prompt_tokens")


def needle_logical_span(tok, prompt, needle):
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    cstart = prompt.find(needle)
    if cstart < 0:
        return [], len(ids)
    cend = cstart + len(needle)
    span = [i for i, (a, b) in enumerate(offs) if a < cend and b > cstart]
    return span, len(ids)


def _read_sink(path):
    """Read the JSONL sink; tolerate a partially-written trailing line."""
    recs = []
    if not path or not os.path.exists(path):
        return recs
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass  # trailing partial write; the next flush completes it
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384])
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--out", default="development/loop7/oracle_trials_index.jsonl")
    ap.add_argument(
        "--decode-steps",
        type=int,
        default=4,
        help="forced decode forwards per trial (DS selection is decode-only).",
    )
    ap.add_argument(
        "--no-require-records",
        action="store_true",
        help="skip the fail-closed expected-record assertions (diagnostic only).",
    )
    args = ap.parse_args()

    base = os.environ.get("DS_BASE_URL", "http://127.0.0.1:30000")

    # Resolve the cross-process paths: env override, else the module defaults the
    # config-borne server also uses. Export the trial-file path so set_active_trial
    # writes where the server reads (the driver never latches the config enable).
    trial_file = os.environ.get("SGLANG_DS_RECALL_ORACLE_TRIAL_FILE") or sink.default_trial_file()
    sink_path = os.environ.get("SGLANG_DS_RECALL_ORACLE_PATH") or sink.default_sink_path()
    os.environ["SGLANG_DS_RECALL_ORACLE_TRIAL_FILE"] = trial_file
    print(f"[oracle-sweep] trial_file={trial_file}")
    print(f"[oracle-sweep] sink={sink_path}")

    # Truncate the sink so the measured run starts clean (drops server-warmup
    # no_active_trial markers written before this driver started).
    try:
        open(sink_path, "w").close()
    except OSError as e:
        print(f"[oracle-sweep] WARNING: could not truncate sink {sink_path}: {e}")

    tok = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE)

    issued = defaultdict(list)  # length -> [request_id, ...]
    with open(args.out, "w") as index_fh:
        for L in args.lengths:
            t0 = time.time()
            done = 0
            for idx in range(args.num):
                needle = h._niah_needle(L, idx)
                prompt = h._make_niah_prompt(L, seed=1000 + idx, needle=needle)
                span, ntok = needle_logical_span(tok, prompt, needle)
                if not span:
                    continue
                req_id = f"L{L}-i{idx}"
                sink.set_active_trial(req_id, idx, span)
                try:
                    ptoks = _generate_decode(base, prompt, args.decode_steps)
                finally:
                    sink.clear_active_trial()
                issued[L].append(req_id)
                index_fh.write(
                    f'{{"request_id":"{req_id}","length_words":{L},"needle_span_start":{span[0]},'
                    f'"needle_span_end":{span[-1]},"offline_tokens":{ntok},"server_tokens":{ptoks},'
                    f'"token_match":{str(ptoks == ntok).lower()}}}\n'
                )
                index_fh.flush()
                done += 1
            print(
                f"[{L:>6}w] issued {done}/{args.num} oracle trials ({time.time()-t0:.1f}s)",
                flush=True,
            )
    print(f"\ntrial index -> {args.out}")
    print(f"oracle records -> {sink_path}")

    # ---- Fail-closed verification: every issued trial must have records ----
    recs = _read_sink(sink_path)
    success_ids = set()
    failures = defaultdict(int)
    for r in recs:
        if "failure" in r:
            failures[r["failure"].split(":")[0]] += 1
        elif r.get("request_id") is not None:
            success_ids.add(r["request_id"])

    print("\n[oracle-sweep] record verification:")
    missing = []
    for L in args.lengths:
        ids = issued[L]
        have = [rid for rid in ids if rid in success_ids]
        miss = [rid for rid in ids if rid not in success_ids]
        print(f"  [{L:>6}w] {len(have)}/{len(ids)} trials produced oracle records")
        missing += [(L, rid) for rid in miss]
    if failures:
        print(f"  failure markers: {dict(failures)}")

    if args.no_require_records:
        print("[oracle-sweep] --no-require-records set; skipping hard assertions.")
        return

    hard_fail = sum(failures[k] for k in _HARD_FAILURES)
    problems = []
    if missing:
        problems.append(
            f"{len(missing)} issued trial(s) produced NO oracle records "
            f"(missing-length regression): {missing[:8]}{' ...' if len(missing) > 8 else ''}"
        )
    if hard_fail:
        problems.append(
            f"{hard_fail} hard fail-closed marker(s): "
            f"{{{', '.join(f'{k}={failures[k]}' for k in _HARD_FAILURES if failures[k])}}}"
        )
    if problems:
        print("\n[oracle-sweep] FAIL-CLOSED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("[oracle-sweep] OK: all issued trials recorded; no hard failures.")


if __name__ == "__main__":
    main()
