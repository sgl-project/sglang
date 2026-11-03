<!-- Companion of scripts/tune_dflash_moe_draft.py and the DFlash/DSpark MoE draft support in this PR (sgl-project/sglang#47181). The benchmark scripts referenced below live in yushengsu-thu/SpecForge#10, serving-bench/qwen3-8b-dspark-moe/. -->

# Evaluating a SpecForge MoE DSpark draft end to end: serving, kernel tuning, results

This guide takes any MoE DSpark draft trained with SpecForge (for example the Qwen3-8B 2 x 12288 top-1 draft, or the 8 x 1536 one)
from an HF export to the agreed result tables and chart. It uses two pull requests:

| PR | what it provides |
|---|---|
| [sgl-project/sglang#47181](https://github.com/sgl-project/sglang/pull/47181) (`yushengsu-thu:dflash-moe-ffn`) | the serving side: SGLang builds the draft's MoE FFN from the HF config fields (`DFlashMoE` in `python/sglang/srt/models/dflash.py`, the `Qwen3MoeDSparkModel` alias in `dspark.py`), runs it through SGLang's `FusedMoE` (Triton fused-MoE kernel), and loads a tuned kernel config for the draft's expert shape when one is present. |
| [yushengsu-thu/SpecForge#10](https://github.com/yushengsu-thu/SpecForge/pull/10) (`dspark-moe-serving-bench`, directory `serving-bench/qwen3-8b-dspark-moe/`) | the benchmark side: checkpoint selection by accept length (`ckpt-sweep/`), the per-task and mixed-workload throughput protocol (`mixed-and-tasks/`), **kernel tuning** (`tune/`), the **report generator** (`make_report.py`), and the results of the 8 x 1536 and 2 x 12288 drafts (`RESULTS.md`, `RESULTS-2x12288.md`). |

The pipeline is: build SGLang with #47181 -> point the server at your draft -> tune the fused-MoE kernel config for the draft's expert
shape -> run the throughput protocol (dense baseline once, MoE with the tuned config) -> `make_report.py`.

Hardware assumed below: a box with 8 x H200, nothing else running on the GPUs you use (every script checks that its GPU has no compute
process and under 1 GB in use before starting a server, and refuses otherwise). Times are for H200 / Triton 3.8.0.

## 1. Set up the two PRs

### 1a. SGLang with #47181 (serving)

Either check out the PR branch:

```bash
git clone -b dflash-moe-ffn https://github.com/yushengsu-thu/sglang.git sglang
cd sglang && python -m venv venv && . venv/bin/activate
pip install torch==2.14.1            # the Rust TreeCore extension needs torch at build time
pip install -e "python[all]" --no-build-isolation
```

or apply the PR to SGLang main `c10e0e17` with the patch shipped in SpecForge#10 (`serving-bench/qwen3-8b-dspark-moe/pr47181-sglang-c10e0e17.patch`,
`git am -3`). The build used for all numbers here is main `c10e0e17` + #47181, torch 2.14.1+cu130, flashinfer 0.7.2, triton 3.8.0.
Check: `python -c "from sglang.srt.models.dflash import DFlashMoE; print('ok')"`.

### 1b. SpecForge#10 (benchmark client and scripts)

```bash
git clone -b dspark-moe-serving-bench https://github.com/yushengsu-thu/SpecForge.git SpecForge-serve
pip install --no-deps -e SpecForge-serve          # the `specforge benchmark` client; --no-deps keeps the serving venv's torch/transformers
pip install click datasets tqdm pydantic pyyaml psutil requests safetensors huggingface-hub accelerate   # whatever of these is missing
export HF_HOME=/scratch/hf_home                   # writable cache: the client downloads gsm8k / math500 / humaneval / mbpp / mt-bench on first use
```

The scripts live in `SpecForge-serve/serving-bench/qwen3-8b-dspark-moe/`. They were written for a fixed box layout; copy them to a work dir
(`BENCH_DIR`, default `/scratch/ckpt-sweep/bench`) and set the few paths they take from the environment:

```bash
B=/scratch/ckpt-sweep/bench; mkdir -p $B/logs
cp SpecForge-serve/serving-bench/qwen3-8b-dspark-moe/mixed-and-tasks/* SpecForge-serve/serving-bench/qwen3-8b-dspark-moe/tune/* $B/
ln -s /path/to/sglang/venv $B/venv                 # task_sweep*.sh / mixed_sweep.sh activate $B/venv for the server
# client: CLIENT_PY (a python that can import specforge) and SPECFORGE_SRC (the SpecForge-serve checkout) are read by the sweep scripts
```

Target model: a local directory of `Qwen/Qwen3-8B` (`TARGET`, default `/scratch/moe2ep/models/Qwen3-8B`). Dense reference draft:
`jiapingW/Qwen3-8b-dspark-SpecForge` (its root files; default `/scratch/moe8b16/eval/models/jw-dense`).

## 2. Load your MoE draft

Use the HF export of the checkpoint (SpecForge's per-step `step-N/` / `epoch-k/` folders are already in this layout: `config.json` +
`model.safetensors` (+ tokenizer files)). The server needs these `config.json` fields:

| field | value |
|---|---|
| `architectures` | `["Qwen3MoeDSparkModel"]` (SpecForge exports may say `Qwen3DSparkModel`; change it) |
| `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `norm_topk_prob` | the draft's MoE shape (e.g. 2 / 1 / 12288 / false) |
| `dflash_config.moe_router_selection_bias` (older exports) or the newer expert-bias naming | router bias on/off; both tensor names (`gate.selection_bias`, `gate.expert_bias`) are accepted |
| `hidden_size`, `num_hidden_layers`, attention fields, `dflash_config.target_layer_ids`, `markov_*` | as exported |

Start a server and check the log:

```bash
python -m sglang.launch_server --model-path $TARGET --tp-size 1 --dtype bfloat16 --attention-backend flashinfer \
  --mem-fraction-static 0.8 --trust-remote-code --host 127.0.0.1 --port 31600 --max-running-requests 256 --cuda-graph-max-bs-decode 256 \
  --context-length 16384 --disable-radix-cache --speculative-algorithm DSPARK --speculative-draft-model-path /path/to/draft \
  --speculative-dspark-block-size 7
```

The log must contain `DFLASH MoE FFN: <E> routed experts, width <N>, routing softmax top-1, expert_bias=True, shared=none, runner=triton` and
`Initialized DSpark draft runner ... gamma=7, verify_num_draft_tokens=8`. Before a tuned config exists it also says
`Using default MoE kernel config. Performance might be sub-optimal!` — that is what step 3 fixes.

`--speculative-dspark-block-size 7` is the V8 setting (7 drafted tokens, verify width 8) used for every number in this repository; block size 15
is V16. If you need to choose among several checkpoints first, `ckpt-sweep/` runs the accept-length eval over a list of exports
(`sweep_all.sh`, `summarize_sweep.py`) — rank by pooled request-macro accept length.

## 3. Tune the fused-MoE kernel config for the draft (`tune/`, see `tune/README.md`)

Why: SGLang picks Triton tile parameters for the MoE GEMMs from a per-shape JSON; without one it uses a heuristic that, for a draft batch of 7
tokens, lands on the large-batch config. For the 2 x 12288 draft that alone cost 9-12% of e2e throughput; the tuned, TMA-aware configs
brought the MoE draft to within 1-5% of the dense draft (accept length is unaffected: the config only changes tiling).

1. Write a stand-in config for the tuner (it reads the MoE shape through the `Qwen3MoeForCausalLM` path), e.g. `shape-cfg/config.json`:
   ```json
   {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", "hidden_size": 4096, "intermediate_size": 12288,
    "moe_intermediate_size": 12288, "num_experts": 2, "num_experts_per_tok": 1, "norm_topk_prob": false, "num_hidden_layers": 5,
    "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936, "dtype": "bfloat16"}
   ```
   Change `num_experts`, `moe_intermediate_size`, `num_experts_per_tok`, `num_hidden_layers` to your draft's values.
2. Choose the token-count buckets: the kernel sees M = (draft tokens per request) x (requests in the batch) = 7 x c at block size 7.
   Default buckets `7 56 112 224 448 896 1792 3584` cover c = 1, 8, 16, 32, 64, 128, 256 and prefill batches; the loader picks the nearest bucket.
3. Run (one bucket per GPU; ~15 min per bucket with the pruned 648-config search, ~1 h with `FULL_SPACE=1`):
   ```bash
   SHAPE_CFG=/path/shape-cfg DRAFT=/path/to/draft TARGET=$TARGET OUT=/scratch/moe-config-mydraft \
     bash $B/tune_moe_draft_e2e.sh
   ```
   It writes balanced top-1 routing dumps, runs `tune_tma_one.py` per bucket (down GEMM with/without TMA, then up GEMM with/without TMA, best
   joint BLOCK_M), assembles `$OUT/configs/triton_<ver>/E=<E>,N=<N>,device_name=<GPU>.json` (+ `_down.json`), then starts one server with
   `SGLANG_MOE_CONFIG_DIR=$OUT` and checks the log says `Using MoE kernel config from ...`, `Up MoE TMA is enabled`, `Down MoE TMA is enabled`,
   measures accept length (must equal the untuned value) and draft / verify / step time at c = 1, 8, 32, 128.
4. Serve with the config: `SGLANG_MOE_CONFIG_DIR=$OUT python -m sglang.launch_server ...` (or copy the two JSON files into sglang's
   `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_<ver>/`). The config is specific to the expert shape, GPU model and
   Triton version; another checkpoint of the same architecture reuses it, a different shape or GPU needs a new tune.

`--moe-runner-backend` / `--speculative-moe-runner-backend` do not reach a DSpark draft in `c10e0e17` + #47181 (the draft is built before the
MoE flags are initialized), so the Triton kernel above is what runs; `tune/sglang-dflash-init-moe-config.patch` fixes that if you want to try
`flashinfer_cutlass` or `triton_kernel` (both measured slower or equal for the 2 x 12288 shape).

## 4. Run the throughput protocol

Rules: one server + one client per empty GPU, TP=1, block size 7, 5 repeat groups per draft, accept length measured once at c=1, per task
`specforge benchmark` with the full prompt sets (gsm8k 1319, math500 500, humaneval 164, mbpp 257, mt-bench 160; temperature 0,
max_new_tokens 2048) at c = 1/8/16/32/64 (+128/256 for gsm8k and math500), and the mixed stream (`mixed_bench.py`: the five sets shuffled
into one 2,400-request stream, seed 42) at c = 1 ... 256.

- Dense baseline (once per box; reuse afterwards): `mixed-and-tasks/task_sweep.sh` and `mixed_sweep.sh` per group, e.g. groups
  `dense-g1 ... dense-g5` (`GPUS=<gpu> PORT=<port> BENCH_DIR=$B bash task_sweep.sh dense-g1 $DENSE_DRAFT`, then `mixed_sweep.sh`).
- MoE with the tuned config: `SGLANG_MOE_CONFIG_DIR=$OUT PREFIX=moe-mydraft bash $B/run_e2e_tma2.sh` runs the 5 groups split into units
  (big tasks / small tasks / mixed c=1 / mixed c>=8) over 8 GPUs in about 70 min, or `E2E=1` on the tune script above does it right after
  tuning. Units of one group are named `<prefix>-gX-{big,small,m1,mrest}`.
- Results land in `$B/logs/tasks/<name>/<task>_c<c>.json` and `$B/logs/mixed/<name>/mixed_c<c>.json`; every server start logs its GPU check,
  and the MoE server logs must show the tuned config lines from step 3.

## 5. Generate the results and the chart (`make_report.py`)

```bash
python $B/make_report.py \
  --dense $B/logs:dense-g1,dense-g2,dense-g3,dense-g4,dense-g5 \
  --moe   $B/logs:moe-mydraft-g1-big+moe-mydraft-g1-small,moe-mydraft-g2-big+moe-mydraft-g2-small,moe-mydraft-g3-big+moe-mydraft-g3-small,moe-mydraft-g4-big+moe-mydraft-g4-small,moe-mydraft-g5-big+moe-mydraft-g5-small \
  --out report/
```

(`dirA+dirB` joins the unit dirs of one group; mixed units are found automatically from the matching `-m1` / `-mrest` names.) `report/report.md`
contains, in this order and format:

1. Accept length: `| | Avg (all requests) | Avg (per task) | gsm8k | math500 | humaneval | mbpp | mt-bench |`, rows dense / MoE / diff / diff %
   (Avg (all requests) = pooled request-macro over the 2,400 requests, Avg (per task) = mean of the five task values).
2. One table per task: `| c | dense tok/s | MoE tok/s | diff % | dense step ms | MoE step ms |`, cells mean ± sample std over the groups,
   diff = difference of the means; step ms = latency x min(c, requests) / verify steps.
3. The mixed stream in the same format (step ms = mean per-request e2e latency / verify steps).
4. `report/mixed_throughput.png`: x = 1, 8, 16, 32, 64, 128, 256 equally spaced, y = tok/s from 0, dense blue circles, MoE orange squares,
   title "Qwen3-8B DSpark, block size 7: e2e throughput, dense vs MoE draft (mixed workload, 5 datasets)".

A repeat more than 25% off its cell median is dropped and listed at the end (a one-off stall in a 6-second measurement, not a kernel effect).
`RESULTS-2x12288.md` is this output for the 2 x 12288 draft.

## Timings and pitfalls

- Tune 15 min (pruned) to 1 h (full) per bucket per GPU; MoE 5-group protocol ~70 min on 8 GPUs; dense baseline the same, once.
- The first server start with `flashinfer_cutlass` JIT-compiles its SM90 MoE module (~15 min); Triton configs are compiled at CUDA-graph
  capture (~1 min).
- Measure dense and MoE on the same box if you can: two H200 boxes differed by -0.6 ... +1.4% on the dense gsm8k sweep.
- Keep the accept length identical between untuned and tuned runs: a changed value means the config changed numerics or the weights were
  loaded in the wrong order (gate/up swap).
