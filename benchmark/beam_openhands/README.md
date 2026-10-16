# Beam NVFP4: OpenHands replay on four GB300 GPUs

The reproduction lives on the public
[`bench-openhands-gb300-sweep-beam` branch](https://github.com/sgl-project/sglang/tree/bench-openhands-gb300-sweep-beam/benchmark/beam_openhands),
in **`benchmark/beam_openhands`**. It reproduces the Beam TP4 curve with concurrency
**1, 4, 8, 16, 32**; TEP4 is available as a second layout.

This is a performance replay with **simulated acceptance length 4** and real
DFlash2 proposals. It does not measure model accuracy or natural acceptance.
It uses four **GB300** GPUs.

## 1. Serving runtime and checkpoints

Use an otherwise idle Linux node with four GB300 GPUs, CUDA 13, and a working
Beam serving environment. Clone this branch:

```bash
git clone --branch bench-openhands-gb300-sweep-beam \
  https://github.com/sgl-project/sglang.git
cd sglang
```

Install the serving environment following this repository's installation
instructions (`python3 -m pip install -e ./python` in a dedicated serving
environment). The measured runtime versions are:

| Package | Version |
|---|---|
| PyTorch | 2.14.1, CUDA 13 build |
| FlashInfer Python | 0.7.1 |
| sglang-kernel | 0.4.9 |
| transformers | 5.19.0 |
| tokenizers | 0.23.2 |

`setup_server.sh` checks these versions and the imported serving source. It
never installs into or changes the serving environment. The launcher imports
this checkout's `python/` tree and requires it to match the tested engine commit
[`f7473673ddf6175f301782af57e684d33d3e3f96`](https://github.com/sgl-project/sglang/commit/f7473673ddf6175f301782af57e684d33d3e3f96)
on the public `beam` branch. The measured runs used a prepared
runtime; this package does not claim validation of a floating Docker tag.

Download the target and released draft to **local directories outside the
benchmark work directory**, after granting the account access where required:

```bash
hf download reflection/Beam-1.0-501B-A23B-NVFP4 \
  --revision b23f9e28019fc26dc0c5b5d4dc7382cee1ceb24f \
  --local-dir /models/Beam-1.0-501B-A23B-NVFP4
hf download reflection/Beam-1.0-501B-A23B-DFlash2 \
  --revision 8d26c952ec6a82a9a18825d6126c9f6948149ad9 \
  --local-dir /models/Beam-1.0-501B-A23B-DFlash2
```

Keep the `.cache/huggingface/download` revision metadata created by `--local-dir`.
The launcher checks the target's shard metadata and the released draft's SHA256.
It never downloads model weights during a run.

## 2. Prepare the client and dataset

The serving Python and the isolated benchmark client are separate. The client
needs CPython 3.12. Replace the interpreter paths below with your installation:

```bash
cd benchmark/beam_openhands
python3 reproduce.py init \
  --server-python /opt/sglang/bin/python3 \
  --client-python /usr/bin/python3.12 \
  --model-path /models/Beam-1.0-501B-A23B-NVFP4 \
  --draft-path /models/Beam-1.0-501B-A23B-DFlash2
python3 reproduce.py prepare runtime/settings.json
```

`init` generates `runtime/settings.json`, `runtime/site.env`, and a bootstrap
server profile. It is idempotent for identical settings and refuses conflicting
overwrites. Use a new `--work-dir` inside this checkout for a different experiment.
Authentication can be supplied through the normal Hugging Face environment;
credentials are never written into settings or job files.

`prepare` runs the following under the independent memory/disk guard:

1. Check the serving runtime and pinned source.
2. Install the closed `client-requirements.txt` set into `runtime/venv`.
3. Build and validate 1,024 conversations with the released Beam tokenizer.
4. Install `matplotlib==3.10.8` in a separate `runtime/plot-env`.

Setup logs and resource evidence are under `runtime/prepare-*/`. Setup failures
are retained. No verifier source, tests, defaults, or scoring are modified.

## 3. Run and plot

First inspect the plan and exact serving command:

```bash
python3 reproduce.py plan runtime/settings.json --results runtime/tp4-01
python3 launch_server.py runtime/settings.json tp4 1 0 --print
```

Run inside a persistent terminal such as `tmux`:

```bash
python3 reproduce.py run runtime/settings.json --results runtime/tp4-01
runtime/plot-env/bin/python plot_curve.py runtime/tp4-01 \
  --layouts tp4 --percentile p50
```

The driver starts a **fresh server for each point**, runs the original pinned
EvalScope timing loop through the Beam API plugin, audits every request, and
stops the owned server before the next point. Output includes `curve.json`,
`curve.png`, `curve.svg`, and `curve.pdf`, plus each case's commands, native
SQLite records, summaries, validation, and resource logs. Raw request evidence
is ignored by Git.

To reproduce both layouts, configure a separate work directory with
`--layouts tep4 tp4` in `init`. TEP4 uses TP4/EP4; TP4 uses TP4/EP1. Add
`--concurrencies 1 4 16 32` if you want that selected subset. cc=2 was not measured
and is not a supported recipe point. Use a new results directory when settings
change; failed measurements are preserved and not retried automatically.

Plotting selections do not change raw results:

```bash
runtime/plot-env/bin/python plot_curve.py runtime/tp4-01 --percentile both
runtime/plot-env/bin/python plot_curve.py runtime/tp4-01 \
  --layouts tp4 --percentile p50 --concurrencies 1 4 16 32 \
  --output runtime/tp4-selected
# Plot the checked-in, payload-free reference measurements without GPU work:
runtime/plot-env/bin/python plot_curve.py reference-results.json \
  --layouts tp4 --percentile p50 --output runtime/reference-plot
```

## Fixed serving recipe

| Setting | Value |
|---|---|
| Parallelism | TP4, EP1, attention-DP1; optional TEP4 changes EP to 4 |
| MoE | No explicit runner or all-to-all flag; resolved TRTLLM / none is audited |
| Draft | DFLASH, eight draft tokens, TRTLLM MHA; no draft-window override |
| Simulation | Acceptance 4, `match-expected`, `real-draft-token` |
| Attention / KV | TRTLLM MHA, page 64, native KV dtype, unified hybrid KV pool |
| Context / prefill | 1,048,576 / 16,384; prefill-decode interval 0 |
| GPU memory fraction | 0.80 |
| HiCache | Off at cc=1/4/8; **88 GB per rank** at cc=16/32 |
| Server and graph request caps | Equal to the measured client concurrency |
| Frontend | One tokenizer worker, HTTP/1.1, Beam reasoning/tool parsers |

`SGLANG_RESPONSE_TEMPLATE_TURN_GRAMMAR=0` is intentional for forced draft proposals.
The launch environment, effective simulation values, and server-resolved flags
are saved with each point. Single-node NCCL bootstrap uses IPv4 loopback; no
cross-node network configuration is required by this recipe.

## Workload, metrics, and reference values

- EvalScope commit: `acd09b44384d53174768bb1063f675420f76fae9`.
- Dataset revisions are pinned in `prepare_dataset.sh` and [SOURCE.json](SOURCE.json).
  Seed 42; thirteen turns; 74,160 bare first-turn tokens; 753 added tokens on later
  turns; exactly 220 output tokens per request. Rendered input grows with history.
- Each point warms up two conversations at cc=1 from the dataset tail, validates
  them, waits for idle, flushes the cache, then measures `2*cc` conversations from
  the head. The five-point TP4 recipe has **1,586 measured requests**, plus 130
  warm-up requests.
- Streaming requests use `temperature=0`, `ignore_eos=true`, and
  `chat_template_kwargs={"allow_missing_thinking":true}` throughout. These match
  the recorded Beam run.
- Beam's adapter normalizes declared special tokens and seven Harmony controls
  in replayed assistant text and retains server usage. EvalScope's timed streaming
  loop is unchanged. All thirteen turns and 220-token output lengths are audited.

| Axis | Definition |
|---|---|
| Y: total tok/s/GPU | Native total input + output throughput divided by 4, **including cached input** |
| X: p50 interactivity | `1000 / p50 TPOT(ms)` in tokens/s/user |
| Optional p90 interactivity | `1000 / p90 TPOT(ms)`; slower-tail interactivity |

Native EvalScope TTFT starts at the first SSE choices packet, which may be
role-only. These values differ from AgentX first-content timing and do not
isolate GPU decode speed.

Recorded TP4 measurements from this recipe:

| CC | Total tok/s/GPU | p50 tok/s/user |
|---:|---:|---:|
| 1 | 33,697.9 | 625.00 |
| 4 | 74,124.2 | 411.52 |
| 8 | 97,679.2 | 336.70 |
| 16 | 111,934.3 | 266.67 |
| 32 | 127,588.4 | 185.87 |

[reference-results.json](reference-results.json) also records TEP4, p90, counts,
and durations. These are single measurements with no confidence intervals. All
reported points passed the request audit with zero errors. The package is a
reproduction recipe, not a guarantee of identical performance on a different runtime.

## Resource protection and recovery

The measured node had an approximately 860 GiB cgroup limit. The recommended entry
point budgets **600 GiB additional host working memory** for runs containing
HiCache points, or 64 GiB for low-concurrency-only runs, and retains a 200 GiB host
reserve. It refuses insufficient headroom before launching.

Independent supervisors check every 0.5 seconds and stop only their owned processes for:

- Working memory (`memory.current - active_file - inactive_file`) above 78% of the
  cgroup limit, or host available memory below 200 GiB.
- Free experiment-volume disk below 1 TiB, or experiment footprint above 80 GiB.
- A new container OOM event, host OOM-kill increment, or monitor failure.

Weights are outside the experiment footprint. Core dumps are disabled. GPU
occupancy, ports, and a cooperative ownership lock are checked before launch.
Runs in the same checkout share `runtime/.gpu-owner.lock` even when they use
different work directories. Share the checkout mount across containers using
the same GPUs; the lock is not a cluster-wide scheduler.
These guards do not control other jobs or a separate filesystem quota.

A stopped controller can leave its independent supervisors running. Inspect each
`node0/status.json`, `client-guard/status.json`, and `DONE` file before recovery;
never start a second driver over live work. `guarded_node.py stop CASE/node0`
requests cleanup of that owned server. Completed points resume only with the same
plan and committed harness. Failed points require a new results directory.

## Validation

From the repository root:

```bash
python3 -m unittest discover -s benchmark/beam_openhands/tests -v
# Run the dependency-dependent cases with the prepared client as well:
benchmark/beam_openhands/runtime/venv/bin/python -m unittest \
  discover -s benchmark/beam_openhands/tests -v
```

These tests cover this benchmark package, not Beam verifier tests. The underlying
serving/replay recipe produced the recorded GPU measurements; packaging and setup
are checked separately. [SOURCE.json](SOURCE.json) records the upstream adaptation
and exact commits. No model weights, credentials, or raw request payloads are
published with the code.
