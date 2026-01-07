# Contributing to SGLang Diffusion

This guide outlines the requirements for contributing to the SGLang Diffusion module (`sglang.multimodal_gen`).

## 1. Commit Message Convention

We follow a structured commit message format to maintain a clean history.

**Format:**
```text
[diffusion] <scope>: <subject>
```

**Examples:**
- `[diffusion] cli: add --perf-dump-path argument`
- `[diffusion] scheduler: fix deadlock in batch processing`
- `[diffusion] model: support Stable Diffusion 3.5`

**Rules:**
- **Prefix**: Always start with `[diffusion]`.
- **Scope** (Optional): `cli`, `scheduler`, `model`, `pipeline`, `docs`, etc.
- **Subject**: Imperative mood, short and clear (e.g., "add feature" not "added feature").

## 2. Performance Reporting

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

## 3. CI-Based Change Protection

Consider adding tests to the `pr-test` or `nightly-test` suites to safeguard your changes, especially for PRs that:

1. support a new model
2. support or fix important features
3. significantly improve performance

See [test](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/test) for examples
