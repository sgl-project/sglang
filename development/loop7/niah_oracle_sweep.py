"""Loop-7 M0 oracle sweep — the budget-vs-scorer decider.

For each NIAH trial: compute the needle's LOGICAL token span (robust: raw-prompt
offset mapping, no chat template, so offline tokens match the server's logical
domain exactly), register it via the cross-process trial file, then issue a raw
/generate with max_new_tokens=1. The server-side recall oracle (eager mode)
records, per (trial, layer), the needle's score rank and score-only recall@K on
the live all-reduced score tensor — which tells us whether a budget > 2048 would
recover the needle (budget-limited) or not (scorer-limited).

Requires the DS server booted EAGER with:
  SGLANG_DS_RECALL_ORACLE=1
  SGLANG_DS_RECALL_ORACLE_PATH=<sink.jsonl>
  SGLANG_DS_RECALL_ORACLE_TRIAL_FILE=<trial.json>
and this driver run with the SAME SGLANG_DS_RECALL_ORACLE_TRIAL_FILE set.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "test", "manual"))
import test_double_sparsity_v32 as h  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from sglang.srt.layers.attention.double_sparsity import oracle_artifact_sink as sink  # noqa: E402

TOKENIZER_FILE = os.environ.get(
    "DS_TOKENIZER_FILE",
    "/cluster-storage/models/deepseek-ai/DeepSeek-V3.2/tokenizer.json",
)


def needle_logical_span(tok, prompt, needle):
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    cstart = prompt.find(needle)
    if cstart < 0:
        return [], len(ids)
    cend = cstart + len(needle)
    span = [i for i, (a, b) in enumerate(offs) if a < cend and b > cstart]
    return span, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384])
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--out", default="development/loop7/oracle_trials_index.jsonl")
    args = ap.parse_args()

    base = os.environ.get("DS_BASE_URL", "http://127.0.0.1:30000")
    if not os.environ.get("SGLANG_DS_RECALL_ORACLE_TRIAL_FILE"):
        print("ERROR: SGLANG_DS_RECALL_ORACLE_TRIAL_FILE must be set (same path as the server).")
        sys.exit(2)
    tok = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE)

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
                    text, ptoks = h._generate(base, prompt, max_new_tokens=1, use_chat=False)
                finally:
                    sink.clear_active_trial()
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
    print(f"oracle records -> {os.environ.get('SGLANG_DS_RECALL_ORACLE_PATH','(server sink path)')}")


if __name__ == "__main__":
    main()
