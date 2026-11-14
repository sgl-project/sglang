# Ascend NPU practice: DeepSeek-V4-Flash on SGLang

This directory records an end-to-end practice run of **DeepSeek-V4-Flash** on Huawei Ascend NPUs with
SGLang: container setup, weight preparation, server launch, a reproducible benchmark matrix, and the
measured results with the reasoning behind them.

It was produced for the **昇腾开源软件极客营 · SGLang Meetup (Shanghai, 2026-11-14)** practice session and is
written so that someone else can repeat the whole thing from this directory alone.

## What was the target

| Item | Value |
| --- | --- |
| Model | `DeepSeek-V4-Flash` — 284B total / 13B active MoE, FP4+FP8 native checkpoint, 1M context, hybrid CSA+HCA attention, bundled MTP head |
| Serving engine | SGLang (branch `ifmn/npu/sglang-practice-for-meetup`), Ascend NPU backend |
| Hardware | Ascend 950PR/DT series (primary) or A2 series (see feasibility note below) |
| Deployment mode | Single node, PD-mixed (no PD disaggregation: the allocated slice is 4 NPUs, while the validated PD split needs 16 dies) |

## Accuracy (MMLU 5-shot, measured)

**89.04%** over 1,015 answered questions (1,140 asked across 57 subjects, response rate 89.0%; conservative
bound **79.3%** if unanswered count as wrong). Scored by next-token logprob over the four letter options, not
string parsing, using the MMLU copy staged on the meetup box and
[`deepseek-v4-flash/bench/mmlu_eval.py`](deepseek-v4-flash/bench/mmlu_eval.py).

## Acceptance package

- [`acceptance/acceptance-report.zh-CN.md`](acceptance/acceptance-report.zh-CN.md) — full 验收报告 (environment,
  seven-defect fix chain, all measurements, reproduction, limitations)
- [`acceptance/evidence/`](acceptance/evidence/) — screenshots + [`INDEX.md`](acceptance/evidence/INDEX.md)
  mapping each claim to its artifact

## Read in this order

1. [`deepseek-v4-flash/README.md`](deepseek-v4-flash/README.md) — the runbook: feasibility, container, weights, launch, benchmark, and what each step proves.
2. [`deepseek-v4-flash/docs/experiment-log.md`](deepseek-v4-flash/docs/experiment-log.md) — timestamped log of what was tried, what broke, and why.
3. [`deepseek-v4-flash/docs/measurements.md`](deepseek-v4-flash/docs/measurements.md) — the measured numbers and how to read them.
4. [`deepseek-v4-flash/docs/troubleshooting.md`](deepseek-v4-flash/docs/troubleshooting.md) — symptom → cause → action.

## Feasibility note: 4 NPUs is the whole story

DeepSeek-V4-Flash is a large checkpoint and the Ascend support matrix is specific:

| Allocation | HBM available | W8A8 checkpoint (300 GB) fits? | Native FP4 checkpoint (159.6 GB) runs? | Verdict |
| --- | --- | --- | --- | --- |
| 950PR/DT × 4 | chip-dependent; needs ≥ ~100 GB/die bare-fit, ≥ 128 GB/die to be useful | yes, if total HBM ≥ ~400 GB bare-fit / ~512 GB practical | yes (MXFP8/MXFP4 is 950-only) | **intended path** |
| A2 × 4 | 4 × 64 GB = 256 GB | **no** | **no** (A2 has neither MXFP8 nor MXFP4) | not feasible for this model |
| A2 × 8 | 8 × 64 GB = 512 GB | yes (`--tp-size 8`) | no | feasible via W8A8 |
| A3 × 4 cards (= 8 dies) | 8 × 64 GB = 512 GB | yes (`--tp-size 8`) | no | feasible via W8A8 |

The repository's own support matrix (`docs/docs/hardware-platforms/ascend-npus/reference/support_models.mdx`)
lists `Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp` as ✅ on both **A2 Series** and **A3 Series** — so A2 is a
supported target, it just needs >= 8 dies (512 GB). There is **no W4A8 variant of V4-Flash** (the only
W4A8 DeepSeek entries are for V4-**Pro**, which is larger overall), so "use a smaller quantization to fit
4 dies" is not an available option.

So `env/00_common.sh` refuses to launch on 4×A2 and prints the exact capacity request to send to the
organizer. That is a result, not a failure: it is the difference between "the model is slow" and "the model
cannot fit".

## FAQ: what is a "4 NPU" allocation, exactly?

The hierarchy matters, because the words are easy to mix up:

```text
node / server  (one computer: CPU sockets, host RAM, disks, NICs)
   └── card  (a physical accelerator card in a PCIe/slot — "卡")
         └── die  (one accelerator chip on the card — "芯片"/NPU device; /dev/davinciN)
               └── AI cores (Cube = matrix, Vector = elementwise, Scalar = control)
```

- An **NPU is not a computer.** It is one accelerator device inside a computer, the same role a GPU plays
  on an NVIDIA box. `NPU : GPU :: CANN : CUDA :: HCCL : NCCL`.
- **"4 NPU" = one server with 4 accelerator devices**, not four servers. They are used *together* for one
  model instance through model parallelism (TP/EP/DP) over the on-node link (HCCS); each device holds a
  shard of the weights and a share of the KV cache.
- On **950PR/DT** one card has one die, so 4 cards = 4 devices = `--tp-size 4`.
  On **A3** one card has two dies, so 4 cards = 8 devices = `--tp-size 8`. Always check
  `npu-smi info -l` and `ls /dev/davinci*` instead of assuming from the card count.
- **Multiple computers** only enter the picture in multi-node deployment (e.g. PD disaggregation = a
  prefill server + a decode server, KV moved over RDMA). The lab's `4NPU 256G 64核 500G` means: one
  machine, 4 NPU devices, 256 GB host RAM, 64 CPU cores, 500 GB disk.

## FAQ: does 4 NPUs mean 4 concurrent requests?

No. "4 NPUs" is a statement about **how many accelerators cooperate on one model instance** (with
`--tp-size 4`, every token is computed across all four). Concurrency is a separate quantity — the number of
requests the scheduler can keep in flight — and it is bounded by three things, in this order:

```text
1. memory : concurrent_requests ≈ KV_budget / (kw_per_token × context_tokens + swa_state)
             KV_budget ≈ total_HBM × --mem-fraction-static − weights
2. scheduler: --max-running-requests, --max-prefill-tokens, --chunked-prefill-size
3. SLO     : the largest batch that still meets your TTFT/TPOT targets (this is the real limit)
```

For DeepSeek-V4-Flash: KV ≈ 5,940 B/token (BF16 KV) + ≈3.2 MB/request sliding-window state,
weights 300.1 GB. On a 512 GB slice (≈0.8 mem fraction → ≈110 GB of KV) that is roughly:

| context / request | memory-bound ceiling | scheduler cap used in this package |
| --- | --- | --- |
| 4k tokens | ~4,000 requests | 80 (raise `MAX_RUNNING` when SLO allows) |
| 32k tokens | ~550 requests | 80 |
| 128k tokens | ~140 requests | 80 |

So one 4-NPU instance can hold hundreds-to-thousands of *resident* requests and serves a much smaller
number *well*; the benchmark matrix in `bench/` exists precisely to find where "well" stops (the knee).
A useful inversion: **1 NPU that fits the model can also serve many concurrent requests** — concurrency
comes from batching, not from counting devices.

## Layout

```text
npu/sglang-practice-for-meetup/
└── deepseek-v4-flash/
    ├── README.md                  runbook (start here)
    ├── env/00_common.sh           hardware detection, image/model path, feasibility gate
    ├── deploy/01_container.sh     docker run for an N-NPU slice
    ├── deploy/02_weights.sh       fetch + verify the W8A8 MTP checkpoint
    ├── deploy/03_launch.sh        launch_server (PD-mixed, verified env/flags)
    ├── deploy/04_smoke.sh         health + chat completion + accuracy/plausibility check
    ├── bench/run_matrix.sh        5-cell benchmark matrix, writes results/*.jsonl + *.meta.json
    ├── bench/summarize.py         results -> docs/measurements.md tables
    ├── bench/collect_telemetry.sh npu-smi + server metrics sampler
    ├── results/                   outputs of a run (jsonl, meta, logs)
    └── docs/                      experiment log, measurements, troubleshooting
```

## Reproduce

```bash
cd npu/sglang-practice-for-meetup/deepseek-v4-flash
cp env/00_common.sh /tmp/env.sh && source /tmp/env.sh     # detects hardware, sets paths
bash deploy/01_container.sh                               # prints/pulls the image and starts the container
bash deploy/02_weights.sh                                 # idempotent; resumes
bash deploy/03_launch.sh                                  # writes results/server.log, waits for readiness
bash deploy/04_smoke.sh
bash bench/run_matrix.sh                                  # ~20 min
python3 bench/summarize.py                                # regenerates docs/measurements.md
```

Every script is idempotent, prints the command it runs, and refuses to guess: if a required variable or
device is missing it stops with the exact question to ask.

## Status of this practice

- Tooling, deployment sequence, benchmark matrix and analysis are complete and verified against the
  upstream SGLang Ascend documentation **in this repository** (see `docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v4_flash.mdx`).
- Measurement run is **pending a working NPU allocation**; the log records the blocker verbatim and the
  capacity request. Numbers will be filled in `docs/measurements.md` by `bench/summarize.py` when the run
  completes. No performance number in this directory is estimated, extrapolated, or copied from a vendor
  slide.
