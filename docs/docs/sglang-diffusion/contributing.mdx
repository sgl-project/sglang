---
title: "Contributing to SGLang Diffusion"
metatags:
    description: "This guide outlines the requirements for contributing to the SGLang Diffusion module (sglang.multimodalgen)."
---

This guide outlines the requirements for contributing to the SGLang Diffusion module (`sglang.multimodal_gen`).

## Contributor Guides

- [Support New Models](/docs/sglang-diffusion/support_new_models): implementation guide for adding new diffusion pipelines
- [Helper ownership](/docs/sglang-diffusion/support_new_models#place-helpers-with-their-owners): where to put shared and model-specific utilities
- [CI Performance](./ci_perf): update and regenerate perf baselines
- [SGLang Plugin System](/docs/hardware-platforms/plugin): package a hardware platform outside the SGLang repository

## On AI-Assisted ("Vibe Coding") PRs

Vibe-coded PRs are welcome — we judge code quality, not how it was produced. The bar is the same for all PRs:

- **No over-commenting.** If the name says it all, skip the docstring.
- **No over-catching.** Don't guard against errors that virtually never happen in practice.
- **Test before submitting.** AI-generated code can be subtly wrong — verify correctness end-to-end.

<a id="commit-message-convention" />

## Pull request titles

Follow the repository-wide [PR title convention](/docs/developer_guide/contribution_guide#pull-request-titles). Use `diffusion` as the scope, or `diffusion/<submodule-or-model>` when a more specific scope helps.

```text
[Perf] diffusion/scheduler: Reduce scheduling overhead
[Fix] diffusion/cli: Fix argument parsing
[Feat] diffusion/wan: Add model support
[Doc] diffusion: Update the contribution guide
[Perf] [NVIDIA] diffusion/attention: Optimize attention kernels
```

Put the change type in the first bracketed tag. `diffusion` belongs in the scope, not in a hardware tag or a separate `[Diffusion]` prefix. Omit the hardware tag unless the change is hardware-specific.

## Performance Reporting

For PRs that impact **latency**, **throughput**, or **memory usage**, you **should** provide a performance comparison report.

### How to Generate a Report

1.  **Baseline**: run the benchmark (for a single generation task)
    ```bash
    $ sglang generate --model-path <model> --prompt "A benchmark prompt" --perf-dump-path baseline.json
    ```

2.  **New**: run the same benchmark, without modifying any server_args or sampling_params
    ```bash
    $ sglang generate --model-path <model> --prompt "A benchmark prompt" --perf-dump-path new.json
    ```

3.  **Compare**: run the compare script, which will print a Markdown table to the console
    ```bash
    $ python python/sglang/multimodal_gen/benchmarks/compare_perf.py baseline.json new.json [new2.json ...]
    ### Performance Comparison Report
    ...
    ```
4. **Paste**: paste the table into the PR description

## CI-Based Change Protection

Consider adding tests to the `pr-test` or `nightly-test` suites to safeguard your changes, especially for PRs that:

- support a new model
    - add a testcase for this new model to `testcase_configs.py`
- support or fix important features
- significantly improve performance

Please run the according testcase, then update/add the baseline to `perf_baselines.json` by following the instruction in console if applicable.

See [test](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/test) for examples

### Host-sync regression tests

A denoising step or a DiT forward should not block the host on the device. One `item()`, `float(t)`, `if t:`, `.cpu()`, `nonzero` or pageable `torch.tensor(..., device="cuda")` per step stops the host from queueing the next step while the GPU runs this one. `test/unit/host_sync_guard.py` has three guards for unit tests:

- `no_cuda_sync(allow=...)` and `assert_no_cuda_sync(fn, ...)` run the code under `torch.cuda.set_sync_debug_mode` and fail with the call site of every sync whose issuing function is not listed in `allow` (`"<file suffix>:<function>"`). They skip without CUDA. Call the code once outside the guard first, so first-use setup is not counted.
- `SyncOnRead` is a CPU stand-in for device tensors (timesteps, sigmas, masks): reading its values on the host raises. Run the step once with plain tensors and once with stand-ins, and require bitwise-equal outputs.
- `forbid_host_syncs()` is a dispatch mode for code that builds its own tensors: it fails on the ops that sync on CUDA whatever tensor they read (scalar reads, `nonzero`, boolean-mask indexing, data-dependent output shapes).

`test_host_sync.py` shows both on the shared denoising loop. Unit tests run in the `multimodal-gen-unit-test` CI job on an H100, so the CUDA guard runs there.

## Out-of-tree platform changes

Use the `sglang.multimodal_gen.platforms` entry-point group when a hardware integration can live in a separately installed package. The [plugin guide](/docs/hardware-platforms/plugin#add-diffusion-support) describes the activation callback, diffusion platform contract, and selection behavior.

When you change the platform interface or discovery implementation in SGLang:

- Add focused CPU-only tests with a fake OOT platform.
- Run the relevant SRT plugin tests when changing shared hook-registry behavior.
- Document workload-specific kernels or optimized paths that still require vendor integration.
