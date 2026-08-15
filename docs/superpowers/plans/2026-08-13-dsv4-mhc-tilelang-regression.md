# DSV4 MHC TileLang Regression Investigation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether DeepSeek-V4-Flash-0731 on latest SGLang `main` executes `mhc_pre_gemm_sqrsum_tilelang` and `mhc_pre_gemm_sqrsum_splitk_kernel`, quantify any performance loss caused by removing `wg_wait=0`, and implement and validate a minimal fix if a regression exists.

**Architecture:** Use `/sgl-workspace/sglang` on the `dsv4-2` 8×B300 devbox as the source-under-test, with the exact user launch configuration on four GPUs. Trace runtime dispatch before benchmarking, then compare matched kernel variants with warm compilation caches and repeated CUDA-event measurements; only after a causal A/B result will the investigation change production code.

**Tech Stack:** Python 3.12, PyTorch 2.13/CUDA 13, TileLang 0.1.11 or the version pinned by latest SGLang `main`, NVIDIA B300, SGLang server and kernel microbenchmarks.

**Spec:** User request in this Codex task and [sgl-project/sglang#29554](https://github.com/sgl-project/sglang/pull/29554).

## Execution Outcome

- Tested upstream `main` at `704e51283605d76d5961d6e014330964fdb1523b` on `dsv4-2` with SGLang `0.5.18.dev519`, TileLang `0.1.11`, PyTorch `2.13.0+cu130`, CUDA 13.0, and B300 SM 10.3.
- The exact requested launch has no relevant environment overrides. `SGLANG_OPT_DEEPGEMM_HC_PRENORM` therefore remains `True`, so DSV4 dispatches the MHC GEMM+sqrsum work to `tf32_hc_prenorm_gemm` before either TileLang fallback can be selected.
- Startup prewarmed 22 MHC prenorm split buckets on every TP rank. The cache contained 22 `sm100_tf32_hc_prenorm_gemm` variants and no textual cache hit for either target TileLang kernel. Both a short decode request and a 12,032-token prefill request completed successfully.
- TileLang v0.1.11 and current upstream `main` define ordinary `T.gemm` as synchronous and forward an internal wait value of `0`. The v0.1.8 public `wg_wait=0` argument was removed when asynchronous Hopper and Blackwell scheduling moved to the explicit `wgmma_gemm`/`wait_wgmma` and `tcgen05_gemm`/`mbarrier_wait_parity` interfaces.
- Consequently, the target kernels cannot cause a current default-path DSV4 regression, and the removed argument is not a missing synchronization setting. Controlled fallback benchmarking and a production patch were intentionally skipped because both plan gates—runtime use and changed synchronization semantics—failed.
- Runtime log: `/data/profile/logs/mhc_latest_main_704e512836.log`. The test server was stopped after verification; the devbox was not released.

## Global Constraints

- Run the user-provided DeepSeek-V4-Flash-0731 configuration with tensor parallelism 4 on `dsv4-2`.
- Update `/sgl-workspace/sglang` to the latest upstream `main` and reinstall its dependencies before the first SGLang run.
- Do not release the devbox.
- Do not modify a dirty remote checkout; preserve any pre-existing artifacts and experiments.
- Compare one synchronization semantic at a time with identical shapes, inputs, warmup, repeat count, clocks, and software revision.
- Do not propose or implement a fix until runtime use and a causal performance regression are both demonstrated.

---

### Task 1: Establish the exact source and software baseline

**Files:**
- Inspect: `/sgl-workspace/sglang/.git`
- Inspect: `/sgl-workspace/sglang/python/sglang/kernels/ops/layernorm/mhc.py`
- Inspect: SGLang dependency metadata that pins TileLang

**Interfaces:**
- Consumes: `dsv4-2` devbox access through `rx devbox run dsv4-2`.
- Produces: immutable SGLang commit SHA, TileLang version/commit, CUDA/PyTorch/GPU versions, and a clean installed checkout.

- [ ] **Step 1: Record repository state and active processes**

  Run `git -C /sgl-workspace/sglang status --short --branch`, `git -C /sgl-workspace/sglang remote -v`, `nvidia-smi`, and process inspection before changing anything.

- [ ] **Step 2: Update to upstream latest main**

  Fetch `origin`, fast-forward the clean `main` branch to `origin/main`, and record `git rev-parse HEAD` plus `git show -s --format=fuller HEAD`.

- [ ] **Step 3: Reinstall the updated source and dependencies**

  Follow the repository's current developer installation command, then import `sglang`, `tilelang`, and `torch` and record their installed paths and versions.

- [ ] **Step 4: Verify the CLI resolves to the updated checkout**

  Run `sglang serve --help` and a Python import-path check; both must resolve to `/sgl-workspace/sglang` before benchmarking.

### Task 2: Prove whether DSV4 executes both kernels

**Files:**
- Inspect: `/sgl-workspace/sglang/python/sglang/kernels/ops/layernorm/mhc.py`
- Inspect: `/sgl-workspace/sglang/python/sglang/srt/models/`
- Create remotely: `/data/profile/logs/mhc_usage_latest_main.log`

**Interfaces:**
- Consumes: latest-main runtime from Task 1 and the user launch command.
- Produces: call-chain evidence, exact tensor shapes, branch predicates, and per-kernel invocation counts during startup/prefill/decode.

- [ ] **Step 1: Trace static dispatch from DSV4 model code to MHC kernels**

  Follow the model layer into the MHC operator and enumerate every predicate selecting `mhc_pre_gemm_sqrsum_tilelang`, `mhc_pre_gemm_sqrsum_splitk_kernel`, or an alternative.

- [ ] **Step 2: Add temporary runtime-only instrumentation**

  Instrument the two wrappers in a disposable remote worktree or use Python-level wrapping so rank 0 logs kernel name, shape, dtype, split-K parameters, and invocation count without changing the baseline source tree.

- [ ] **Step 3: Launch the exact server configuration**

  Run `sglang serve --trust-remote-code --model-path deepseek-ai/DeepSeek-V4-Flash-0731 --tp 4 --moe-runner-backend flashinfer_mxfp4 --speculative-algorithm DSPARK --swa-full-tokens-ratio 0.1 --host 0.0.0.0 --port 30000`, wait for health, then issue representative prefill and decode requests.

- [ ] **Step 4: Capture and classify actual calls**

  Save counts and shapes separately for compilation/startup, prefill, extend, and decode. A kernel is considered used only if runtime instrumentation records an invocation after model initialization.

### Task 3: Reproduce and isolate the alleged `wg_wait` regression

**Files:**
- Inspect: PR #29554 diff and its parent commits
- Inspect: installed TileLang source and generated TIR/CUDA for the two kernels
- Create remotely: `/data/profile/mhc_wg_wait_ab/`

**Interfaces:**
- Consumes: the exact DSV4 shapes observed in Task 2.
- Produces: repeatable A/B latency distributions and generated-code differences for explicit old semantics versus latest-main semantics.

- [ ] **Step 1: Reconstruct the pre-upgrade and post-upgrade variants**

  Identify the exact TileLang APIs and versions on both sides of PR #29554, including where `wg_wait=0` was accepted, what removing it changes, and whether a replacement decorator/config/context now carries that setting.

- [ ] **Step 2: Build a focused correctness-and-latency harness**

  Use the observed DSV4 shapes and dtypes, compare both variants against the same PyTorch reference, warm each compiled kernel, and time repeated launches with CUDA events and synchronization outside the timed region.

- [ ] **Step 3: Run controlled A/B measurements**

  Alternate A/B order over multiple rounds, preserve clocks and process conditions, and report median, p10, p90, and relative delta for each observed shape. Clear or segregate TileLang caches only when comparing compilation artifacts, not steady-state execution.

- [ ] **Step 4: Inspect generated synchronization and pipeline code**

  Diff generated TIR/PTX/CUDA or compiler diagnostics for wait-group/barrier placement and correlate the code change with the measured delta. Reject the `wg_wait` hypothesis if generated semantics and timings do not differ consistently.

### Task 4: Implement the minimal confirmed fix

**Files:**
- Modify conditionally: `/sgl-workspace/sglang/python/sglang/kernels/ops/layernorm/mhc.py`
- Test conditionally: the nearest existing MHC kernel correctness/benchmark test identified by repository search

**Interfaces:**
- Consumes: one confirmed root cause and working latest-TileLang reference pattern from Task 3.
- Produces: one source change restoring intended scheduling semantics without changing numerical results or unrelated kernels.

- [ ] **Step 1: Write a failing regression check before production changes**

  Add the smallest test or benchmark assertion that distinguishes the confirmed bad generated schedule/API configuration from the intended one; run it on unmodified latest `main` and save the failure.

- [ ] **Step 2: Apply one TileLang scheduling change**

  Use the supported latest-TileLang mechanism found in Task 3. Do not combine tuning changes, refactors, or unrelated cleanup.

- [ ] **Step 3: Run correctness tests**

  Compare both kernels across all DSV4 shapes observed in Task 2 against the reference with the repository's dtype-appropriate tolerances.

- [ ] **Step 4: Repeat the identical A/B benchmark**

  Re-run Task 3's harness against unmodified latest `main` and the single patched variant; require a repeatable improvement rather than a one-run win.

### Task 5: Validate end-to-end impact and report

**Files:**
- Create remotely: `/data/profile/logs/dsv4_mhc_fixed.log`
- Preserve remotely: Task 2–4 benchmark logs and generated compiler artifacts

**Interfaces:**
- Consumes: confirmed patch from Task 4, or the no-regression conclusion from Task 3.
- Produces: final answers to kernel usage, regression causality, and supported fix, with SHAs and quantitative evidence.

- [ ] **Step 1: Relaunch DSV4 with the same command**

  Use the same four GPUs, flags, cache state policy, request workload, and measurement window as the baseline.

- [ ] **Step 2: Verify runtime calls and service correctness**

  Confirm the intended kernels still run, health checks pass, and representative responses remain valid.

- [ ] **Step 3: Compare end-to-end metrics**

  Report startup compile time separately from steady-state prefill/decode latency and throughput so JIT compilation does not contaminate kernel execution results.

- [ ] **Step 4: Deliver a reproducible conclusion**

  Include SGLang and TileLang versions, DSV4 call-path evidence, per-shape benchmark table, root cause, exact fix or no-fix rationale, and paths to preserved remote artifacts.
