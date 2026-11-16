---
name: rerun-test
description: Run the CI tests a change touches with the `/rerun-test` PR comment instead of relying on the per-PR Base run. Use when a PR changes code whose tests run in the extra, nightly, or weekly tier, when asked to run the related tests on a PR, or before asking for a merge.
---

# rerun-test

Base CI covers only the core default paths, and its workload may not exercise a given change. Tests for other features run in the `extra`, `nightly`, and `weekly` tiers, so a green Base run does not cover them. `/rerun-test` runs chosen test files against the PR head, each on the runner it is registered for, whatever its tier.

## Pick the tests

1. Tests the PR adds or modifies under `test/registered/` (`/rerun-test --changed` selects them).
2. Tests that exercise the changed code: grep `test/registered/` for the changed modules, functions, and server flags.
3. One end-to-end test for each subsystem the change touches (PD, HiCache, speculative decoding, LoRA, a model family), whatever its tier.

Prefer the narrowest spec that covers the change: `file.py::TestClass` or `file.py::TestClass.test_method`.

## Post

Put every spec in one comment, for example `/rerun-test registered/hicache/test_hicache_storage_file_backend.py registered/lora/test_lora_update.py`. The handler replies with the run links and marks the reply when the runs finish.

## Limits

- A file estimated above 60 minutes is rejected; it runs only in its scheduled tier.
- CPU tests run only for the ubuntu CPU suite (`base-a-test-cpu`); Xeon and arm64 suites are not supported.
- NPU tests run as whole files (no `::` selector) through `_npu-pr-test-stage.yml`, with the PR installed as in `pr-test-npu.yml`. Only the a3 suites that a job in `pr-test-npu.yml`, `nightly-test-npu.yml`, or `full-test-npu.yml` runs through that stage are supported; the perf/accuracy single-node stage, `base-a-test-npu`, a2/a5, and multi-node NPU tests are rejected. A file registered in a nightly and a non-nightly suite on the same runner runs only in the non-nightly one.
- AMD tests run in the AMD CI container, set up as the suite's job in `pr-test-amd.yml` or `pr-test-amd-extra.yml` sets it up, on that job's runner. AMD nightly suites and the disaggregation suite are rejected.
- Without `--hw`, a file runs on the first of CUDA/CPU, NPU, and AMD that it registers. `--hw` takes `cuda`, `npu`, `amd`, a comma-separated list (`--hw cuda,amd`), or `all` (every backend the file registers). With a list, a backend a dispatched file does not register is noted as skipped next to its command. `/rerun-group <group>` takes `--hw` the same way.
- No PR-local `sglang-kernel` wheel is built. A change under `python/sglang/kernels/aot/` needs the normal PR CI.
- Each CUDA, CPU, or AMD file runs directly (`python3 <file> -f`), not through `test/run_suite.py`, and `/run-cpu-ci` runs main's stage workflow. A change to the CI runner itself (`run_suite.py`, `ci_utils.py`, `fork_test_worker.py`, the stage workflows) is validated by the PR's own PR Test run.

Command syntax and permissions: [contribution guide](https://docs.sglang.io/developer_guide/contribution_guide.html#how-to-trigger-ci-tests).
