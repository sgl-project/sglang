# Kernel benchmark regression CI

A lightweight, **model-agnostic** performance-regression gate for SGLang CUDA/Triton
kernels.

It reuses the existing `sgl-kernel/benchmark/bench_*.py` benchmarks (no kernel
launch logic is duplicated), captures their throughput / latency into a structured
JSON, and compares a fresh run against a **nightly-generated ground truth** with a
relative tolerance (default **±5%**).

The suite is seeded with the kernels that dominate current decode/prefill paths and
is designed to **grow across model families** — adding a kernel is one line in
`registry.py`.

## Why

Many `bench_*.py` files are not wired into any CI suite, so a code change can
silently regress a kernel's performance. This harness closes that gap with the same
ground-truth pattern the diffusion CI already uses.

CI GPUs are **not isolated**, so absolute numbers drift run to run. The design
accounts for that:

- **Ground truth comes from the nightly job**, not a value committed in-tree.
- The PR run pulls that ground truth and compares with a **relative tolerance**.
- A regression **fails the job but is safe to re-run** — transient noise passes on
  retry; a real regression keeps failing.
- Each config is measured **best-of-N** (`--repeat`) to suppress jitter.

## Flow

```
   nightly-kernel-bench-gt.yml                     pr-test-kernel-bench.yml
   (schedule, sm90 + sm100 runners)                (pull_request, sm90 runner)
            |                                                  |
   generate <sku>.json  ──publish──>  sgl-project/ci-data  ──pull (raw URL)──> compare
                                      kernel-bench/                             ±tolerance
                                          sm90.json                                |
                                          sm100.json                          pass / fail (re-runnable)
```

One ground-truth file per GPU SKU (`sm90` = Hopper, `sm100` = Blackwell). Cases
whose `min_compute_capability` exceeds the current GPU (e.g. CUTLASS MLA needs
Blackwell) are skipped, so the sm90 PR job only compares sm90-capable kernels.

## CLI

```bash
# From sgl-kernel/benchmark/

# List the registered cases (no GPU required).
python3 -m kernel_bench_regression list

# Generate a ground-truth JSON (run on the target GPU).
python3 -m kernel_bench_regression generate --out gt.json --repeat 5

# Compare a fresh run against a local or remote ground truth.
python3 -m kernel_bench_regression compare --gt-file gt.json
python3 -m kernel_bench_regression compare \
    --gt-url https://raw.githubusercontent.com/sgl-project/ci-data/main/kernel-bench/sm90.json \
    --tolerance 0.05
```

`compare` exits non-zero when any shared config is more than `--tolerance` worse
than the ground truth (lower latency / higher throughput is never a regression).
Configs present on only one side are reported but never fail the run.

## Adding a kernel

Append one `BenchCase` to `KERNEL_BENCH_CASES` in [`registry.py`](registry.py):

```python
BenchCase(
    case_id="my_kernel",            # stable key in the ground truth — never rename
    bench_file="bench_my_kernel.py",
    mark_attr="benchmark",          # the triton.testing.perf_report object
    provider="sgl-kernel",          # the line_arg VALUE of the kernel under test
    metric="TFLOPs",                # "us" | "TFLOPs" | "GB/s"
    higher_is_better=True,
    tags=("gemm",),                 # kernel categories for grouping (model-agnostic)
    component="what it computes",
    # min_compute_capability=(10, 0),  # for Blackwell-only kernels
    # extra_args={...},                # constant kwargs for the bench function
    # configs_override=[(1,), (16,)],  # pin a small, deterministic config set
)
```

The next nightly run regenerates the ground truth to include it.

## Bootstrapping

The very first time, run the nightly workflow manually
(`workflow_dispatch`) so `kernel-bench/sm90.json` and `sm100.json` exist in
`sgl-project/ci-data` before the PR gate can pull them.

## Files

| File | Purpose |
| --- | --- |
| `registry.py` | Declarative list of tracked kernel cases. |
| `runner.py` | Imports each bench, drives the kernel provider, captures metrics. |
| `compare.py` | Pure, GPU-free tolerance comparison + report formatting. |
| `__main__.py` | `list` / `generate` / `compare` CLI. |
| `test_compare.py` | CPU-only unit tests for the comparison math. |
| `../../scripts/ci/utils/kernel/publish_kernel_bench_gt.py` | Publishes GT JSON to ci-data. |
