# Muse Spark Open NVFP4: OpenHands sweep on 8 x B300

Run an OpenHands multi-turn performance sweep with the public
[`sgl-project/sglang` `muse-spark` runtime](https://github.com/sgl-project/sglang/tree/muse-spark).
The harness uses one TP8/EP8 replica, DFlash v2 with an eight-token draft block,
FP8 KV cache, four tokenizer workers, and HTTP/2.

This is a performance replay with **simulated acceptance length 4.5** (`match-expected`,
`real-draft-token`), not an accuracy evaluation. Block verification is off. Accepted draft
proposals are not verified samples, so generated text is not a quality signal.

## Setup

Build the public Muse runtime on an x86_64 CUDA host:

```bash
git clone --branch bench-openhands-b300-sweep https://github.com/sgl-project/sglang.git
cd sglang
docker build -f docker/muse_spark/muse-spark.Dockerfile -t muse-spark-openhands .
```

The Dockerfile installs the public `muse-spark` branch and its pinned CUDA 13 dependencies.
`setup_server.sh` checks those dependencies and installs this checkout's Python source
editable without dependency resolution. It does not use an older FlashInfer patch recipe.

Download these checkpoints to local directories before running (the target requires
Hugging Face access approval):

- [`meta-models/Muse-Spark-Open-816B-A42B-NVFP4`](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4)
- [`RadixArk/Muse-Spark-Open-DFlash2`](https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2)

Then run in the image, with an otherwise idle eight-GPU node:

```bash
docker run --rm --gpus all --ipc=host \
  -v "$PWD":/workspace/sglang -v /models:/models \
  -w /workspace/sglang/benchmark/muse_spark_openhands \
  -e GPU_LOCK_FILE=/models/.muse-openhands-gpu.lock \
  -e MODEL_PATH=/models/Muse-Spark-Open-816B-A42B-NVFP4 \
  -e DRAFT_PATH=/models/Muse-Spark-Open-DFlash2 \
  muse-spark-openhands bash run_sweep.sh
```

Use a model directory containing `generation_config.json`. The harness reads the runtime's
checkpoint-default sampling parameters, validates them, records them in `results/sampling.json`,
and sends them explicitly through EvalScope. Both servers and client use these defaults;
no historical greedy preset is applied. The benchmark's 220-token limit and `ignore_eos`
are workload controls. Unsupported checkpoint defaults fail before GPU startup.

## Workload and protocol

- Client: EvalScope commit `acd09b44384d53174768bb1063f675420f76fae9`, with dependencies
  pinned in `client-requirements.txt` and installed into a separate venv.
- Dataset: 1,024 conversations from `nebius/SWE-rebench-openhands-trajectories`, padded
  with `nvidia/OpenScienceReasoning-2` text using the target tokenizer. Seed 42;
  13 turns; first-turn bare-token target 74,160 and subsequent-turn target 753.
  Rendered requests include growing conversation history and template tokens.
  Dataset Hub revisions are not pinned; retain the generated JSON and its recorded
  SHA256 for comparisons. Existing JSON is reused rather than rebuilt.
- Execution order: **128, 64, 48, 32, 16, 8, 4, 1** concurrent conversations.
- Each point starts a fresh server, validates its configuration and simulation environment,
  runs two warmup conversations at concurrency 1 from the dataset tail, checks observed
  acceptance length, and flushes the cache before measurement.
- Measurement uses `2 * concurrency` fresh conversations from the dataset head,
  13 sequential turns each, and 220 output tokens per request. The full sweep has
  **7,826 measured requests**; warmups and rejected balance attempts are excluded.
- HiCache defaults to **200 GB per rank** at concurrency >=16 and is off below 16.
  Use `HICACHE_GB=160` to reproduce that cache setting; report it with the results.
- On HiCache points, the largest tokenizer-worker connection share must be no more
  than `ceil(1.25 * concurrency / 4)`. Imbalanced attempts are preserved and retried
  at most six times. A final imbalanced attempt completes and is marked `OK_UNBALANCED`.
  Request failures still fail the run; completed points are not rerun on resume.
- Client timeouts: connect 600 seconds, read 3,600 seconds, total 21,600 seconds.

## Resources and results

The script refuses occupied GPUs or ports and holds a sweep lock through
cleanup. Containers sharing GPUs must share the lock file, as in the example above. It checks the smaller of host and cgroup memory headroom, reserves 300 GiB,
requires 1,927 GiB free disk, and limits allocated benchmark/model/cache storage to
1.2 TB. HiCache preflight includes `8 * HICACHE_GB + 240 + reserve` GiB. Guard failures
stop only this sweep's server and client. Size the container and mounted volumes before
running; the guard does not control unrelated jobs or a separate filesystem quota.

`results/results.tsv` contains one row per rung. Each point retains server
configuration, logs, simulation environment, warmup output, native EvalScope summaries,
percentiles, and SQLite request data. Failed and rejected attempts are preserved.

| Metric | Definition |
|---|---|
| `interactivity_p50` | `1000 / p50 TPOT (ms)`, from client-observed streaming |
| `total_tok_s_per_gpu` | EvalScope total input + output tokens/s divided by 8, including cached prompt tokens |
| `out_tok_s` | Output tokens/s across all eight GPUs |
| `tpot_p90_ms` | p90 client TPOT; `1000 / p90 TPOT` gives slow-tail interactivity |
| `accept_len` | Server-reported simulated acceptance after the warmup flush |

These client metrics do not isolate GPU decode speed. Preserve `OK_UNBALANCED` labels
when plotting or comparing points.

## Optional settings

| Variable | Default |
|---|---|
| `SERVER_PYTHON` | `python3` in the prepared Muse image |
| `CONCURRENCIES` | `128 64 48 32 16 8 4 1` |
| `HICACHE_GB`, `HICACHE_MIN_CC` | `200`, `16` |
| `NUM_CONVERSATIONS` | `1024` (at least `2 * max_concurrency + 2`) |
| `GATE_FACTOR`, `GATE_MAX_ATTEMPTS` | `1.25`, `6`; factor 0 disables the split gate |
| `SIM_ACC_TOKEN_MODE` | `real-draft-token` |
| `RESULTS_DIR` | `results/` |
| `CACHE_DIR` | `cache/`; JIT cache directory |
| `HF_HOME` | `$CACHE_DIR/huggingface`; dataset download cache |
| `TMPDIR` | `/tmp`; keep it short for ZeroMQ Unix socket paths |
| `PORT` | `30100` |
| `GPU_LOCK_FILE` | `/tmp/muse-openhands-gpu.lock` |
| `EXTRA_SERVER_ARGS` | unset; extra arguments are still subject to configuration checks |

Keep results directories separate when changing model, sampling, cache, or workload settings.
