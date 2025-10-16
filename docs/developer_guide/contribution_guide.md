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

## Request a review
You can identify potential reviewers for your code by checking the [code owners](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and [reviewers](https://github.com/sgl-project/sglang/blob/main/.github/REVIEWERS.md) files.
Another effective strategy is to review the file modification history and contact individuals who have frequently edited the files.
If you modify files protected by code owners, their approval is required to merge the code.

## How to trigger CI
To trigger CI, the pull request must have the "run-ci" label.

- If you have write access to sgl-project/sglang, your pull request will be automatically tagged by @sglang-bot.
- If you have triage access to sgl-project/sglang, you can manually add the label by clicking "Labels" on the right side of your pull request page.
- If you do not have the above access, please request a review and ask other maintainers to add the label for you.

## General code style
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

If you have any questions or want to start a discussion, please feel free to ask in our [Slack channel](https://slack.sglang.ai).

Thank you for your interest in SGLang. Happy coding!
