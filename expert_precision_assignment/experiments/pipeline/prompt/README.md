# prompt/ — openai-format prompt JSONLs for the sweep pipeline

Custom-task prompts live here. Each task writes at least two parallel files:

| file | role |
|---|---|
| `<task>.jsonl` | openai-chat-format prompts; `pipeline/run_sweep.sh <task>` feeds this to `bench_serving --dataset-name openai` |
| `<task>.meta.jsonl` | per-row ground truth + metadata; scored against trace by index |
| `livecodebench_v6.private_tests.pkl` | **LCB only** — pickle dict `{question_id: b64z}` carrying the bulky base64+zlib-encoded test cases. Kept out of `.meta.jsonl` because one problem alone can be 92 MB compressed; inlining would bloat meta to ~4 GB. Scorer loads this once at startup and looks up by `question_id`. |

**Prompts and meta are parallel by index.** Row `i` of `.jsonl` is the prompt that produces `generated_texts[i]` in the trace, which is scored against row `i` of `.meta.jsonl`. `bench_serving` preserves input order, so offline scoring works by `zip(trace.generated_texts, open(meta.jsonl))`. For LCB, the scorer additionally joins on `meta[i].question_id → private_tests[question_id]`.

## Generate the JSONL pair

All commands run from this `pipeline/prompt/` directory:

```bash
# SuperGPQA (26,529 MCQs; seeded shuffle so first N covers all disciplines)
python prepare_prompts_supergpqa.py

# Smoke test with a small subset:
python prepare_prompts_supergpqa.py --limit 100 --seed 1

# IFBench (AllenAI, 300 prompts; no shuffle)
python prepare_prompts_ifbench.py

# LiveCodeBench v6 (1,055 problems; optional date window for contamination)
python prepare_prompts_lcb_v6.py
python prepare_prompts_lcb_v6.py --start_date 2024-08-01 --end_date 2025-04-15
```

Each prep script writes two files to this directory.

## Run the sweep against prepared prompts

Once `pipeline/prompt/<task>.jsonl` exists, `pipeline/kv_calib/run_calib.sh`
and `pipeline/run_sweep.sh` auto-detect it and switch to `MODE=openai`.
The commands below run from `experiments/`:

```bash
# Calibrate KV against the BF16 baseline (first 128 prompts by default).
NUM_PROMPTS=128 bash pipeline/kv_calib/run_calib.sh supergpqa

# Regenerate task-specific configs with amortized KV sizing.
python pipeline/gen_config/gen_all.py --task supergpqa \
    --calib_json data/kv_calib/supergpqa.json

# Full sweep (all 26,529 SuperGPQA prompts × 66 configs).
bash pipeline/run_sweep.sh supergpqa

# Score traces offline.
for t in data/results/supergpqa/mc*_*.jsonl; do
    python pipeline/scoring/score_traces_supergpqa.py \
        --trace "$t" --meta pipeline/prompt/supergpqa.meta.jsonl
done

# Collect summary CSV (auto-merges .scores.json sidecars).
python pipeline/collect_result/collect_results.py \
    --results_dir data/results/supergpqa \
    --out_csv data/results/supergpqa/summary.csv
```

The same flow applies for `ifbench` and `livecodebench_v6` — substitute the task name.

## OpenAI-chat prompt schema (what each prep script writes to `<task>.jsonl`)

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
 "max_tokens": 512, "temperature": 0.0, "top_p": 1.0}
```

`bench_serving --dataset-name openai` loads this via `sglang.benchmark.datasets.openai_dataset`. Per-request sampling params (`temperature`, `top_p`, anything that isn't `messages` / `max_tokens` / `model`) are passed through as `extra_request_body` to the chat-completions endpoint.
