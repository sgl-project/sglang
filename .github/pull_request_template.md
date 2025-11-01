<!-- Thank you for your contribution! Please follow these guidelines to enhance your pull request. If anything is unclear, submit your PR and reach out to maintainers for assistance. Join our Slack community at https://slack.sglang.ai to discuss further. -->

## Motivation

Enable config-driven, parameterized GLU activations for MoE across Triton/CUTLASS/native paths (bf16/fp8/fp4), while preserving existing fast paths and backward compatibility. Provide a single, unified activation entry that supports `silu`, `gelu`, `swish/swiglu`, `geglu`, `reglu` with optional `alpha`, `limit` (clamp), and `up_shift`. Defaults match the current behavior (e.g., standard SiLU/GEGLU fast paths) and no official configs need changes.

## Modifications

<!-- Detail the changes made in this pull request. -->

## Accuracy Tests

<!-- If this pull request affects model outputs (e.g., changes to the kernel or model forward code), provide accuracy test results. -->

## Benchmarking and Profiling

<!-- If this pull request impacts inference speed, provide benchmarking and profiling results. -->

## Checklist

- [ ] Format your code according to the [Format code with pre-commit](https://docs.sglang.ai/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [ ] Add unit tests according to the [Run and add unit tests](https://docs.sglang.ai/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [ ] Update documentation according to [Write documentations](https://docs.sglang.ai/developer_guide/contribution_guide.html#write-documentations).
- [ ] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.ai/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.ai/developer_guide/contribution_guide.html#benchmark-the-speed).
