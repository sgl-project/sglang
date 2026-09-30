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
  --cwd "$LAB/sglang" --timeout-seconds 600 \
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

`diagnose_qwen3_kv.py` compares HF eager/SDPA and BF16/FP32 without any Mooncake
or capture code. Reloading each dtype is intentional: casting an entire model
to BF16 would also narrow FP32 RoPE frequency buffers and invalidate the
comparison. This numerical diagnostic is separate from the zero-error online
capture check. No real training, cross-node RDMA or performance SLO is certified
by these experiments.
