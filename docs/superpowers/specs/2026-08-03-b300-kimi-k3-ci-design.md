# B300 Kimi-K3 Base-C CI Design

## Goal

Add always-on 8-GPU B300 base-C coverage for Kimi-K3. The new CI test must
launch the locally prestaged Kimi-K3 checkpoint with the Low Latency and
Balanced recipes and require at least 0.95 accuracy on a 200-example GSM8K
evaluation for each recipe.

## Scope

This change will:

- Add the `8-gpu-b300` CUDA runner configuration.
- Add the `base-c-test-8-gpu-b300` per-commit suite.
- Wire the suite into the primary `pr-test.yml` workflow and its finish gate.
- Add one Kimi-K3 B300 model E2E test file containing one test class per launch
  recipe.

This change will not:

- Add a B300 suite or job to `pr-test-extra.yml`.
- Change production serving code.
- Download or copy the Kimi-K3 target checkpoint during CI.
- Add performance, speculative-acceptance, or basic-sanity thresholds beyond
  the requested GSM8K accuracy checks.

## CI Architecture

The suite will follow the repository's current `stage` plus `runner_config`
registration model.

1. `scripts/ci/runner_configs.yml` will define `8-gpu-b300` using the standard
   CUDA dependency installer, v6 artifact downloads, and the literal
   `8-gpu-b300` GitHub Actions runner label.
2. `test/run_suite.py` will list `base-c-test-8-gpu-b300` as a CUDA
   per-commit suite.
3. `.github/workflows/pr-test.yml` will add a reusable-workflow job named
   `base-c-test-8-gpu-b300`. It will depend on the same base-C prerequisites as
   the other large-model jobs, use `runner_config: 8-gpu-b300`, allow 60
   minutes for the suite, and allow 3600 seconds for the test file.
4. `pr-test-finish.needs` will include the new job so a B300 failure or
   cancellation fails the aggregate PR test result.

No extra-CI job will be added. Partition fanout remains controlled by the
existing dynamic partition computation; because this suite initially contains
one file, both recipes run sequentially on one 8-GPU B300 runner.

## Test File Design

Create `test/registered/models_e2e/test_kimi_k3_b300.py` and register it with:

```python
register_cuda_ci(est_time=1800, stage="base-c", runner_config="8-gpu-b300")
```

The file will define two defensive `CustomTestCase` classes that also inherit
`GSM8KMixin`:

- `TestKimiK3B300LowLatency`
- `TestKimiK3B300Balanced`

Each class will:

- Set `gsm8k_score_threshold = 0.95`.
- Set `gsm8k_num_examples = 200`.
- Launch its own server once in `setUpClass`.
- Terminate the server in `tearDownClass` only when the process attribute was
  successfully created.

The classes remain separate instead of looping over configurations in one test
method. This preserves normal unittest reporting, isolates server lifecycle
failures, and mirrors the referenced B200 model E2E test pattern. Keeping both
classes in one file prevents the dynamic suite partitioner from consuming two
8-GPU B300 runners concurrently.

## Model Paths

The target checkpoint is pinned to the verified prestaged snapshot:

```text
/data/radixark/model-cache/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569
```

The Low Latency draft checkpoint remains the Hugging Face model identifier:

```text
RadixArk/Kimi-K3-DSpark
```

The target uses an absolute snapshot path rather than `moonshotai/Kimi-K3` or
`try_cached_model` so CI cannot drift to a different revision and does not
depend on a target-model network download.

## Launch Recipes

`popen_launch_server` receives the target model path and derives host and port
from `DEFAULT_URL_FOR_TEST`; therefore the test does not duplicate
`--model-path`, `--host`, or `--port` in `other_args`.

### Low Latency

```text
--trust-remote-code
--tp-size 8
--mem-fraction-static 0.85
--reasoning-parser kimi_k3
--tool-call-parser kimi_k3
--mamba-full-memory-ratio 0.86
--speculative-algorithm DSPARK
--speculative-draft-model-path RadixArk/Kimi-K3-DSpark
--speculative-dspark-block-size 7
--enable-linear-replayssm-spec
```

### Balanced

```text
--trust-remote-code
--tp-size 8
--dcp-size 8
--disable-custom-all-reduce
--mem-fraction-static 0.85
--reasoning-parser kimi_k3
--tool-call-parser kimi_k3
--mamba-full-memory-ratio 7.21
--enable-hierarchical-cache
```

The numeric values, including the Balanced `7.21` mamba full-memory ratio, are
copied exactly from the requested recipes.

## Accuracy and Failure Behavior

`GSM8KMixin` will flush the server cache and execute the existing completion-
API, 5-shot GSM8K evaluation with 128 worker threads. Both recipes must score
at least 0.95 across 200 examples. A launch failure, server crash, timeout, or
score below 0.95 fails the test file and therefore the B300 base-C job.

Cleanup is defensive so a partially completed `setUpClass` does not mask the
original launch error with a missing-process exception.

## Validation

Before publishing the PR, run repository-level checks that do not require an
8-GPU B300 host:

- Parse and resolve the new runner configuration with
  `scripts/ci/runner_configs.py`.
- Collect the registered suite through `test/run_suite.py`/the CI registry and
  confirm the new test resolves to `base-c-test-8-gpu-b300`.
- Compile the new Python test file.
- Parse the modified workflow YAML.
- Run formatting and whitespace checks on the changed files.
- Review the final diff against this design.

The actual model launches and GSM8K accuracy gates require the self-hosted
8-GPU B300 runner and will execute in the new base-C CI job. Because the change
contains CI configuration and a new E2E test rather than production behavior,
validation uses registry/configuration checks instead of an artificial failing
production unit test.

## Delivery

The work will be committed on `codex/ci-b300-kimi-k3`, pushed to `origin`, and
opened as a draft pull request. The PR description will enumerate the CI wiring,
the two launch recipes, the pinned checkpoint, the 0.95 GSM8K gate, and the
local validation performed.
