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

## Request a Review
You can identify potential reviewers for your code by checking the [code owners](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and [reviewers](https://github.com/sgl-project/sglang/blob/main/.github/REVIEWERS.md) files.
Another effective strategy is to review the file modification history and contact individuals who have frequently edited the files.
If you modify files protected by code owners, their approval is required to merge the code.

## General Code Style
- Avoid code duplication. If the same code snippet (more than five lines) appears multiple times, extract it into a shared function.
- Minimize device synchronization. Reduce expensive CPU-GPU synchronization operations, such as `tensor.item()` or `tensor.cpu()`, whenever possible. Use vectorized code.
- Keep files concise. If a file exceeds 2,000 lines of code, split it into multiple smaller files.
- Prioritize extreme efficiency. SGLang is a runtime, and most of your code runs on the critical path for every request. Optimize every minor overhead as much as possible.

## Tips for newcomers

If you want to contribute but don’t have a specific idea in mind, pick issues labeled [“good first issue” or “help wanted”](https://github.com/sgl-project/sglang/issues?q=is%3Aissue+label%3A%22good+first+issue%22%2C%22help+wanted%22). These tasks typically have lower complexity and provide an excellent introduction to the codebase. Also check out this [code walk-through](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/sglang/code-walk-through) for a deeper look into SGLang’s workflow.

If you have any questions or want to start a discussion, please feel free to ask in our [Slack channel](https://slack.sglang.ai).

Thank you for your interest in SGLang. Happy coding!
