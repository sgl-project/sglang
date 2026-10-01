# H100 Capture Experiments

Use the image digest and dependency overlay in `h100-runtime-lock.json`.
`requirements-h100.txt` is an overlay on that image, not a standalone lock for
an arbitrary host. The initial image's Torch 2.11/kernel 0.4.4 pair cannot run
this source checkout; changing only the kernel produces a Torch ABI failure.

The resident lab uses these GPU-visible paths:

- Lab: `/gpfs/users/fuxuanwei-1/dspark-maas-lab`
- Source: `sglang/python`, selected through `PYTHONPATH`
- Experiment Python: `venvs/capture/bin/python`
- Offline wheel cache: `wheels`
- Model: `/gpfs/models/huggingface.co/Qwen/Qwen3-0___6B`
- Logs/results: `state/logs` and `state/results`, indexed by submitted job ID

The experiment venv inherits the base image packages and installs the locked
overlay with `--no-index --find-links <wheel-directory> --no-deps`. The source
is installed editable with `SGLANG_BUILD_RUST_EXTS=none` for these text HTTP
tests; this does not validate the native Rust entrypoints. Optional inherited
FlashInfer 0.6.12 cubin/JIT-cache distributions were removed because they are
incompatible with FlashInfer 0.6.17. Supported source JIT is used with version
checks enabled. The idle worker continues to use the base Python/Torch pair.

Every GPU experiment goes through the resident worker so the idle load is
stopped and reaped before test execution, then resumed afterward:

```bash
LAB=/gpfs/users/fuxuanwei-1/dspark-maas-lab
python3 "$LAB/bin/gpu_worker.py" --state-dir "$LAB/state" submit \
  --cwd "$LAB/sglang" --timeout-seconds 900 \
  --env "PYTHONPATH=$LAB/sglang/python" \
  --env "PATH=$LAB/venvs/capture/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  -- "$LAB/venvs/capture/bin/python" \
  test/registered/storage/test_training_capture_runtime.py \
  --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B -v
```

Stage changed source before submission with `rsync -aR --exclude=__pycache__`,
using paths relative to the implementation checkout. The submit command returns
a job ID; inspect its result JSON for a terminal outcome and exit code, and its
log for assertions. A queue or heartbeat file alone is not test completion.

Other experiment commands, under the same queue/environment:

```bash
python -m unittest discover -s test/registered/unit/training_capture -v
python -m pytest test/registered/storage/test_training_snapshot_mooncake.py -v -s
python mooncake-study/experiments/diagnose_qwen3_kv.py --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B
```

The real capture test starts isolated Mooncake master/data-node/client processes
and an HTTP Catalog test double. The segment owner outlives the producer to
test snapshot lifetime. It starts one observed ordinary SGLang server and then
one normal CUDA-graph SGLang server. Synthetic request tokens and temporary
model-observer dumps stay inside the test's temporary directory.

Two AR overlap servers then run with ordinary execution and decode CUDA graphs.
They check one-iteration result lag, three-request batches padded to a four-row
graph, prefix dedup/remapping, exact Host capacity, EOS, delayed grammar sampling
and a streamed request aborted while results are pending. Each mode waits for the
abort's fenced Catalog failure and reads its completed snapshots after producer
exit. The correctness driver waits for enough spare capture reservations before
each request/batch; it is not a saturation or throughput benchmark.

The overlap observer follows actual forward input tokens and preserves the first
KV observation for each position, because later Radix prefix dedup can change the
physical mapping. It reads raw vocabulary scores before serving-side processing.
Its synchronous reads are test-only. Ordinary AR and static DSpark capture both
support default overlap scheduling. Speculative observation also follows actual
verify inputs and model-accepted output prefixes, since CPU output IDs can lag.

`diagnose_qwen3_kv.py` compares HF eager/SDPA and BF16/FP32 without any Mooncake
or capture code. Reloading each dtype is intentional: casting an entire model
to BF16 would also narrow FP32 RoPE frequency buffers and invalidate the
comparison. This numerical diagnostic is separate from the zero-error online
capture check. No real training, cross-node RDMA or performance SLO is certified
by these experiments.

The runtime test also exports synthetic `DSparkTargetKVDraftModel` checkpoints
from a retrieved sample's teacher/KV contract. Four additional servers check
ordinary and CUDA-graph speculative generation with and without overlap,
including batches, both verify accept/reject branches and continued serving
after management API rejections. A test-only observer compares projected draft
KV against per-layer reference math and checks actual graph replay with no
target hidden capture.
The overlap pair also checks three-request graph padding, exact Host capacity
and streamed abort while verify results are pending. The abort helper waits for
fenced Catalog failure and reservation recycling, and rejects quarantined slots.
The fixture's forced proposal weights are not a trained draft. This extends the
same command above; allow 900 seconds and retain the job log for evidence.

The command also starts an ordinary overlap server with adaptive admission
enabled. Its test-only entrypoint holds the first background write using a file
beside the publication journal. While a spare Host slot remains available, a
second generation request completes and is excluded from capture by the stalled
writer's cooldown. Removing the pause lets the original snapshot publish; after
ratio recovery, a later request produces another validated Store snapshot. This
exercise adds two completed snapshots and does not measure a latency SLO.

That adaptive server also enables `--enable-metrics`. The test scrapes the
actual HTTP `/metrics` endpoint while the writer is held and after release,
checking multiprocess export of the zero/recovered sampling ratio, READY and
adaptive-exclusion counters, writer age, reset reservation gauges and zero
quarantine. The producer's monitoring thread is independent of its writer and
Catalog threads. Unit tests additionally cover bounded labels, repeated metric
updates, exporter failure/retry and a blocked Catalog. The provisioned dashboard
is `examples/monitoring/grafana/dashboards/json/training-capture-dashboard.json`;
its serving-latency panels do not establish capture-overhead thresholds.

### BF16 Rounding Isolation

The actual fixed-input parity command remains the serving gate. The following
separate diagnostic runs three comparisons on one retained checkpoint/fixture:
production, reference auxiliary arithmetic with real Triton attention, and
reference auxiliary/attention arithmetic while retaining real paged KV writes
and reads. It never writes `validation/parity.json` or certifies serving, and
blocks execution of the target decoder.

```bash
PYTHONPATH=python:/path/to/SpecForge python mooncake-study/experiments/diagnose_target_kv_parity_bf16.py \
  --checkpoint /path/to/retained-draft \
  --target-path /models/Qwen3-0.6B \
  --reference-attention sdpa
```

This distinguishes layout/checkpoint errors from numerical differences in the
same-input operators. It deliberately recomputes diagnostic attention after
the actual paged backend runs, so its timing is not production performance and
its equality result is not evidence for the unmodified attention backend.

Focused regressions:

```bash
python -m pytest -q \
  test/registered/unit/spec/test_dspark_target_kv.py \
  test/registered/unit/training_capture \
  test/registered/spec/dspark/test_dspark_stacked_ctx_kv_parity.py \
  test/registered/unit/managers/test_io_struct.py \
  test/registered/unit/model_executor/runner/test_decode_cuda_graph_runner.py \
  test/registered/spec/dspark/test_dspark_draft_path_default.py
```

## Fixed-Input Draft Parity

The additional reference uses SpecForge commit
`e10ea2fa3c248a4f60d636791dd71efb67338c9c1`. Stage its `specforge` package under
`$LAB/specforge-reference/` and include both `$LAB/sglang/python` and
`$LAB/specforge-reference` in each queued experiment's `PYTHONPATH`.
The SpecForge repository itself does not need modifications.

Run the small-model checks through the same resident queue:

```bash
python -m pytest -q test/registered/spec/dspark/test_dspark_target_kv_parity.py
```

These compare actual training/serving layers for three Markov head variants,
backpropagate cached CE+TV128, take an optimizer step, export the weights and
compare again after a fresh serving load. They also check future-token/KV
isolation and failed validation reports. They are test-only adapters, not the
production SpecForge training pipeline.

The real captured fixture is retained outside Git at
`$LAB/fixtures/qwen3-target-kv-parity-20261001`. It contains snapshot tensors read
back from Mooncake and the synthetic two-layer draft used by the runtime test.
It includes the target's shared-weight references, not the target decoder.
Run the explicit gate under the queue with:

```bash
python -m sglang.test.dspark_target_kv_parity \
  --checkpoint "$LAB/fixtures/qwen3-target-kv-parity-20261001" \
  --target-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B
```

`--reference-attention` accepts `eager`, `sdpa` and `flex_attention`; serving uses
Triton. `validation/parity.json` records all numerical stages and runtime/artifact
identities, and a failed comparison exits nonzero. Failed reruns cannot leave an
older successful report. Run one validation at a time per checkpoint directory.
The real BF16 gate currently fails; see `../IMPLEMENTATION_STATUS.md`. The normal
capture runtime test remains an independent, exact Store readback check.

For diagnosis only, FP32 weights/activations and native auxiliary operators can
be compared while preserving the actual paged-attention path:

```bash
python mooncake-study/experiments/diagnose_target_kv_parity_fp32.py \
  --checkpoint "$LAB/fixtures/qwen3-target-kv-parity-20261001" \
  --target-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B
```

The production RMSNorm kernel cannot execute FP32 activations; this diagnostic
substitutes native norm/activation/RoPE and disables fused in-place QK norm.
It reads only saved KV/teacher tensors and shared weights, forbids target decoder
execution, and prints its result without writing a serving parity certificate.
Passing it does not satisfy the BF16 gate or establish training quality.
