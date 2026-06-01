<!-- Thank you for your contribution! Please follow these guidelines to enhance your pull request. If anything is unclear, submit your PR and reach out to maintainers for assistance. Join our Slack community at https://slack.sglang.io to discuss further. -->

## Motivation

`trtllm_mha` + FROZEN_KV MTP (NEXTN) on hybrid-SWA models (e.g. Gemma-4) crashes with a CUDA illegal memory access once the SWA pool fills past the SWA-vs-full layer ratio (issue [#26957](https://github.com/sgl-project/sglang/issues/26957), prior attempt [#26531](https://github.com/sgl-project/sglang/pull/26531)). `TRTLLMHAAttnBackend` cached its SWA state at `__init__` from `model_runner.token_to_kv_pool`. On the draft worker that is the draft's own non-SWA pool, so `use_sliding_window_kv_pool` was `False` and the SWA→full page-index translation became dead code — SWA layers then got raw full-pool indices and read the smaller SWA k-cache out of bounds.

## Modifications

- Resolve the SWA pool from `model_runner.token_to_kv_pool_allocator` (`get_kvcache()`) instead of `token_to_kv_pool`. In FrozenKVMTP the draft's `token_to_kv_pool` is swapped to the target's `SWAKVPool` per forward call, but `token_to_kv_pool_allocator` is set once at init and is stable — the correct source of truth.
- Mirrors FlashInfer, which already gates SWA eligibility on the allocator (`isinstance(token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)`, `flashinfer_backend.py:1027`).
- Not registering the Gemma-4 draft as hybrid-SWA: the draft owns no KV (frozen), so its own `SWAKVPool` would be keyed by draft layer ids while attention runs on remapped target layer ids (KeyError + wasted alloc); the allocator already points at the correct target pool.

## Accuracy Tests

Reproduced end-to-end on B200 (SM100) with `repro_26957_e2e.py`: crashes before the fix (CUDA illegal memory access), passes after.

## Speed Tests and Profiling

<!-- If this pull request impacts inference speed, provide benchmarking and profiling results. -->

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [x] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).

## Review and Merge Process

1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process).
2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers.
3. Trigger CI tests with [comments](https://docs.sglang.io/developer_guide/contribution_guide.html#how-to-trigger-ci-tests) or contact authorized users to do so.
   - Common commands include `/tag-and-rerun-ci`, `/tag-run-ci-label`, `/rerun-failed-ci`
4. After green CI and required approvals, ask Merge Oncalls or people with Write permission to merge the PR.
