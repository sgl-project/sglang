# Double Sparsity benchmark

Driver for the DS perf + quality gates. Measures decode tok/s, TBT (p50/p95),
and NIAH retrieval accuracy across:

1. `branch_ds_off` — DS branch with DS disabled (regression baseline).
2. `branch_ds_on`  — DS enabled (the speedup measurement).

See [`DESIGN.md`](./DESIGN.md) for the full design rationale, the gate
definitions, the production recipe, and the current status. This README
is a quick reference for invoking the bench.

## Prerequisites

- 8× H200 (or 4× H200 with reduced KV) for the 70B headline runs.
- SGLang installed with FA3 backend + matching `sgl-kernel`.
- A calibration JSON. The conc=16/tb=2048 headline requires
  retrieval-shaped calibration:

```bash
python3 scripts/double_sparsity/make_retrieval_calib_prompts.py \
    --output /workspace/ds_retrieval_calib_prompts.txt \
    --n-prompts 128 --target-chars 20000 --seed 0

python3 scripts/double_sparsity/calibrate.py \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --output /workspace/calib_llama_3_1_70b_retrieval_s32.json \
    --heavy-channels 32 --n-samples 64 --seq-len 4096 \
    --prompts-file /workspace/ds_retrieval_calib_prompts.txt \
    --device-map auto
```

Wikitext calibration is sufficient for the conc=32/tb=8192 headline; see
[`DESIGN.md` §2.6](./DESIGN.md) for when each calibration shape applies.

## Running

### Long-context concurrency sweep (recommended)

`run_70b_sweep.sh` launches one DS-off and one DS-on server (not 10
separate launches) and sweeps concurrency in `{1,4,8,16}` per leg:

```bash
CTX=131072 N_REQUESTS=8 OUTPUT_LEN=512 CONCURRENCIES=1,4,8,16 \
  bash benchmark/double_sparsity/run_70b_sweep.sh \
    /workspace/calib_llama_3_1_70b_retrieval_s32.json
```

The driver writes `bench_70b_sweep_131072/branch_ds_{off,on}.json` and
runs `compare.py` to render the per-concurrency table + pick the best
speedup point.

### Single-point comparison

Each config is a separate process. Example DS-on at the headline operating
point (both gates pass):

```bash
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
PYTHONPATH=python python3 benchmark/double_sparsity/bench_decode.py \
  --config branch_ds_on \
  --calibration /workspace/calib_llama_3_1_70b_retrieval_s32.json \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --context-len 131072 --output-len 512 \
  --n-requests 32 --concurrency 16 \
  --tp-size 8 --mem-fraction-static 0.85 --max-running-requests 32 \
  --token-budget 2048 --recent-tokens 64 --sink-tokens 4 \
  --min-seq-len 4096 --max-selected-per-request 8192 \
  --block-t 1024 --k-block 64 \
  --selector-backend torch \
  --niah --niah-n-samples 10 \
  --output-json /tmp/branch_ds_on.json
```

Compare:

```bash
python3 benchmark/double_sparsity/compare.py \
  --branch-off /path/to/off.json --branch-on /tmp/branch_ds_on.json
```

## Reproducing the published JSONs

- `repro_session/conc16_move_left/conc16_tb2048_retrieval_torch.json`
  — headline conc=16 result.
- `repro_session/conc16_move_left/conc32_tb8192_wikitext_torch_recheck.json`
  — conc=32 recheck on the updated code.
- `repro_session/sweep_70b_128k_tbt_win/` — earlier sweep data referenced
  in DESIGN.md.
- `repro_session/sweep_70b_128k_tbt_win/nsys/` — nsys kernel diff at the
  conc=32 winning point. Heavy artifacts (`.nsys-rep`/`.sqlite`) live
  outside the repo; see `nsys/MANIFEST.md`.

## Gates

`compare.py` reports two gates:

- **Perf**: `tbt_p50(DS_on) ≤ 0.90 × tbt_p50(DS_off)` at some shared
  concurrency point.
- **Quality**: `niah_accuracy(DS_on) ≥ niah_accuracy(DS_off) − 0.02`
  (both legs must run with `--niah`; otherwise the guard reports `UNKNOWN`).

Both gates must pass simultaneously at the same operating point to ship.
