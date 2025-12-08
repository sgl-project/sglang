# Contribution Guide

Welcome to **SGLang**! We appreciate your interest in contributing. This guide provides a concise overview of how to set up your environment, run tests, build documentation, and open a Pull Request (PR). Whether you’re fixing a small bug or developing a major feature, we encourage following these steps for a smooth contribution process.

## Install SGLang from Source

### Fork and clone the repository

**Note**: New contributors do **not** have the write permission to push to the official SGLang repo. Please fork the repository under your GitHub account, then clone your fork locally.

```bash
git clone https://github.com/<your_user_name>/sglang.git
```

### Build from source

Refer to [Install SGLang from Source](../get_started/install.md#method-2-from-source).

## Format code with pre-commit

We use [pre-commit](https://pre-commit.com/) to maintain consistent code style checks. Before pushing your changes, please run:

```bash
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```

- **`pre-commit run --all-files`** manually runs all configured checks, applying fixes if possible. If it fails the first time, re-run it to ensure lint errors are fully resolved. Make sure your code passes all checks **before** creating a Pull Request.
- **Do not commit** directly to the `main` branch. Always create a new branch (e.g., `feature/my-new-feature`), push your changes, and open a PR from that branch.

## Run and add unit tests

If you add a new feature or fix a bug, please add corresponding unit tests to ensure coverage and prevent regression.
SGLang uses Python's built-in [unittest](https://docs.python.org/3/library/unittest.html) framework.
For detailed instructions on running tests and integrating them into CI, refer to [test/README.md](https://github.com/sgl-project/sglang/tree/main/test/README.md).

## Write documentations

We recommend new contributors start from writing documentation, which helps you quickly understand SGLang codebase.
For more details, please refer to [docs/README.md](https://github.com/sgl-project/sglang/tree/main/docs/README.md).

## Test the accuracy
If your code changes the model output, please run the accuracy tests. A quick sanity check is the few-shot GSM8K.

```
# Launch a server
python3 -m sglang.launch_server --model Qwen/Qwen2-7B-Instruct

# Evaluate
python3 -m sglang.test.few_shot_gsm8k --num-questions 200
```

Please note that the above script is primarily a sanity check, not a rigorous accuracy or speed test.
This test can have significant variance (1%–5%) in accuracy due to batching and the non-deterministic nature of the inference engine.
Also, do not rely on the "Latency/Output throughput" from this script, as it is not a proper speed test.

GSM8K is too easy for state-of-the-art models nowadays. Please try your own more challenging accuracy tests.
You can find additional accuracy eval examples in:
- [test_eval_accuracy_large.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_eval_accuracy_large.py)
- [test_gpt_oss_1gpu.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_gpt_oss_1gpu.py)

## Benchmark the speed
Refer to [Benchmark and Profiling](../developer_guide/benchmark_and_profiling.md).

## Requesting a review for merge
You can follow the pull request merge process described in [MAINTAINER.md](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md).
You will need to work with the Merge Oncall, Codeowner, and other reviewers to get their approvals.
Then your PR can be merged.

## How to Trigger CI Tests

We have a lot of open PRs but limited CI machines, so only top and trusted contributors have permission to trigger CI tests.
Users with permission are listed in the [CI_PERMISSIONS.json](https://github.com/sgl-project/sglang/blob/main/.github/CI_PERMISSIONS.json)

For CI to run on a pull request, it must have the "run-ci" label. Authorized users can add the label or rerun failed tests by commenting on the PR with one of these commands:

- `/tag-run-ci-label`: Adds the "run-ci" label. Every future commit will trigger CI.
- `/rerun-failed-ci`: Reruns the failed or flaky tests from the most recent commit.
- `/tag-and-rerun-ci`: A single command that performs both `/tag-run-ci-label` and `/rerun-failed-ci`.
- `/rerun-stage <stage-name>`: Reruns a specific test stage without waiting for its dependencies. This is useful when you want to quickly validate a fix for a specific test failure instead of waiting ~30 minutes for preceding stages to complete.

If you have permission, the [Slash Command Handler](https://github.com/sgl-project/sglang/actions/workflows/slash-command-handler.yml) will run your command and react with a 👍 to your comment. It may take up to a few minutes for the reaction to appear. Here’s a usage [example](https://github.com/sgl-project/sglang/pull/14253#issuecomment-3599509302).

To avoid spamming a PR with too many `/rerun-failed-ci` comments, you can also trigger the command by editing an existing comment and adding any suffix (e.g., `/rerun-failed-ci try again`).

Example of rerunning a single test stage: `/rerun-stage unit-test-backend-4-gpu`.

If you don’t have permission, please ask maintainers to trigger CI for you.

### CI rate limits

We apply CI rate limits to prevent abuse and ensure fair usage of our CI resources.

Each CI workflow has a default limit defined in its workflow configuration file. For example, in [pr-gate.yml](https://github.com/sgl-project/sglang/blob/main/.github/workflows/pr-gate.yml), the default cooldown period is 120 minutes, and each workflow can override it via the `cool-down-minutes` input parameter:

```yaml
cool-down-minutes:
  description: "Default cooldown period in minutes; 0 disables rate limiting"
  type: number
  default: 120
```

Users listed in [CI_PERMISSIONS.json](https://github.com/sgl-project/sglang/blob/main/.github/CI_PERMISSIONS.json) may have a per-user cooldown interval. In practice, we use the minimum of the workflow’s default window and the user-specific interval.


## Code style guidance
- Avoid code duplication. If the same code snippet (more than five lines) appears multiple times, extract it into a shared function.
- Minimize device synchronization. Reduce expensive CPU-GPU synchronization operations, such as `tensor.item()` or `tensor.cpu()`, whenever possible. Use vectorized code.
- Prioritize extreme efficiency. SGLang is a runtime, and most of your code runs on the critical path for every request. Optimize all minor overheads as much as possible, especially in the model forward code.
  - A common pattern is some runtime checks in the model forward pass (e.g., [this](https://github.com/sgl-project/sglang/blob/f1b0eda55c2c4838e8ab90a0fac7fb1e3d7064ab/python/sglang/srt/models/deepseek_v2.py#L486-L491)). These are very likely the same for every layer. Please cache the result as a single boolean value whenever possible.
- Make functions as pure as possible. Avoid in-place modification of arguments.
- Keep files concise. If a file exceeds 2,000 lines of code, split it into multiple smaller files. (e.g., `scheduler.py`, `scheduler_output_processor_mixin.py`)
- Keep tests run fast.
  - If a single test file run longer than 500 seconds, split it into multiple smaller files (e.g., `test_eagle_infer_a.py`, `test_eagle_infer_b.py`).
  - If a single job in a github workflow runs longer than 30 mins, split it into smaller jobs/steps.
  - Reuse server launches in your unit tests to make tests run faster.
- When supporting new hardware or features, follow these guidelines:
  - Do not drastically change existing code.
  - Always prefer new files to introduce specific components for your new hardware (e.g., `allocator_ascend.py`).
  - If you write multiple if/else blocks for new features, ensure the common path (e.g., NVIDIA hardware or the existing code path) is the first branch.

## How to update sgl-kernel
Since sglang and sgl-kernel are separate Python packages, our current GitHub CI infrastructure does not support updating a kernel and using it immediately within the same pull request (PR).
To add a new kernel or modify an existing one in the sgl-kernel package, you must use multiple PRs.

Follow these steps:

1. Submit a PR to update the sgl-kernel source code without using it in sglang python package (e.g., [#8884](https://github.com/sgl-project/sglang/pull/8884/files)).
2. Bump the version of sgl-kernel (e.g., [#9220](https://github.com/sgl-project/sglang/pull/9220/files)).
   - Once merged, this will trigger an automatic release of the sgl-kernel wheel to PyPI.
   - If not urgent, you can wait for other people to release the wheel. A new version will typically be released within one week.
3. Apply the changes:
   - Update the sgl-kernel version in `sglang/python/pyproject.toml` to use the modified kernels.
   - Update the related caller code in the sglang to use the new kernel.

## Tips for newcomers

If you want to contribute but don’t have a specific idea in mind, pick issues labeled [“good first issue” or “help wanted”](https://github.com/sgl-project/sglang/issues?q=is%3Aissue+label%3A%22good+first+issue%22%2C%22help+wanted%22). These tasks typically have lower complexity and provide an excellent introduction to the codebase. Also check out this [code walk-through](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/sglang/code-walk-through) for a deeper look into SGLang’s workflow.

If you have any questions or want to start a discussion, please feel free to ask in our [Slack channel](https://slack.sglang.io).

Thank you for your interest in SGLang. Happy coding!
