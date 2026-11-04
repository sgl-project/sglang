# P0 baseline integration review record

2026-11-04. Scope: P0 only. The fixed upstream snapshot is merged into `rfc-kvc`, conflicts are resolved, and the working tree remains an uncommitted merge for parent review, independent tests, and commit. No commit or push was performed. P1–P9 remain pending.

The user's implementation authorization supersedes source-plan C3. The narrower instruction for this task reserves committing to the parent. Root `PLAN.md` contains the implementation specification with that authorization and adopted D1–D7 defaults. The source outside this repository was read only:
`/home/youwei/bzh/project/TokenCake/RFC/docs/sglang/session-kv-orchestrator-implementation.zh-CN.md`.

| Adopted default | P0 treatment and later scope |
| --- | --- |
| D1 | Existing optional retention/quota policies and their defaults are preserved. A future Router orchestration mode will disable local automatic policy ownership by default; no Router mode is added here. |
| D2 | Future initial deployment: one Router, independent TP=1 Engines, FULL attention, persistent-host HiCache, overlap disabled. Existing hybrid behavior remains covered by CPU regressions. |
| D3 | Future `sgl-kv-orchestrator` crate embedded in Router. No crate is added in P0. |
| D4 | Future L2 eviction protection and L3 soft-pin. Existing soft priority remains soft priority. |
| D5 | Future single control writer per Mooncake protection domain. No ownership service is added. |
| D6 | Existing action semantics remain compatible; future public actions get independent action types. |
| D7 | Future bounded in-memory session graph with application re-report and unknown observations after restart. No persistence or HA claim is made. |

Actual baseline and environment:

```text
HEAD: aedd69c0b0d1dcecf08d69f38ba6fc7de8be7633
MERGE_HEAD: 4517a5ff1bbd1f944d6bde1819300ed14af7b698
Merge base: f02458fff127d918cb573914e5e6d5dd8b6fb8d1
Source plan SHA256: 522831b9f909a0ff96d2dd80ed3632364bb3ceac53d6b3cf2b13f34411599727
3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 13:27:36) [GCC 11.2.0]
torch 2.14.1
pytest 9.1.1
pytest-subtests not installed separately
msgspec 0.22.0
transformers 5.17.0
ruff 0.17.1
rustc 1.99.1 (58cf8f9c3 2026-10-13), parent-provisioned task-local stable toolchain
System Rust: 1.75.0 (not used for the successful build)
CPU tests: real CPU KV pools, fake CUDA driver registration and controlled DMA, temporary file storage
No GPU server, model weights, or live Mooncake service used by P0 tests
```

Merge commands, from the initially clean working tree:

```bash
git fetch /home/youwei/bzh/project/TokenCake/RFC/sglang-origin 4517a5ff1bbd1f944d6bde1819300ed14af7b698
git merge --no-commit --no-ff FETCH_HEAD
```

The merge base is `f02458fff127d918cb573914e5e6d5dd8b6fb8d1`. There is no resulting merge commit yet: `HEAD` and `MERGE_HEAD` above identify its two intended parents. The broad upstream delta is part of this fixed baseline integration, including upstream docs and Router/indexer additions. No independent docs-website edits were made.

All textual conflicts (three files), with resolutions:

| Path | Resolution |
| --- | --- |
| `python/sglang/srt/managers/schedule_policy.py` | Keep both `running_request_token_offset` and upstream `mamba_prefill_chunk_grid`. Preserve shared decode-growth accounting and upstream DCP/Mamba chunk alignment. |
| `python/sglang/srt/managers/scheduler.py` | Import both helpers. Keep upstream bootstrap-queue priority input together with KV guidance, admission preparation, request binding, owner progress, flush/abort and weight-lifecycle hooks. |
| `python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py` | Use upstream `BASE_COMPONENT_TYPE` while retaining the `host` distinction and priority-claim override of device session ordering. Keep claim split/forget/reset hooks with upstream split actions and eviction action propagation. |

No file was replaced wholesale with an ours/theirs resolution.

Additional manual P0 edits (complete list beyond the three conflict files):

- `test/registered/unit/mem_cache/test_kv_control_cache.py`: construct upstream tree slots through `tree_slots_of` and bind allocator state to the actual request pool state, matching the production KV configurator.
- `test/registered/unit/mem_cache/test_kv_control_transfers.py`: construct tree slots, declare TP1 DCP rank zero, and replace the registration-only no-op with a stateful fake CUDA driver covering initialization through destruction. Production register/unregister bookkeeping runs unchanged. Duplicate registration and unowned unregistration fail; teardown must leave zero registered ranges. File CPU staging temporarily clears `pin_memory`; teardown restores the original ownership flag before destroying the host pool.
- `PLAN.md`: full source specification, updated C3 authorization, adopted D1–D7, explicit P0-only implementation scope and pending phases.
- `PLAN-REVIEW-LOG.md`: this evidence and full file manifest.

Existing operation, reference, capacity and byte-content assertions were retained. The shared fixture adds a resource-leak assertion; no assertions were relaxed, no production cleanup errors were swallowed, and no high-level cleanup function was mocked out.

Integration review covered Scheduler progress/admission/abort/flush, cache-controller completion and verified backup prefix accounting, unified-cache tree/component APIs, storage attach/detach drain guards, request/reply structures, namespace isolation and weight fencing. KV publishing remains wired through the upstream publisher and scheduler idle/metrics/pause call sites. This is source review plus the CPU evidence below; it does not certify live GPU scheduling or event transport.

```text
AST parse: 44 integration-path Python files passed.
Byte-preserved against original HEAD: python/sglang/srt/managers/kv_control
Byte-preserved against original HEAD: python/sglang/srt/mem_cache/unified_cache/control_preparation.py
Byte-preserved against original HEAD: python/sglang/srt/mem_cache/unified_cache/control_storage.py
Byte-preserved against original HEAD: python/sglang/srt/mem_cache/unified_cache/control_storage_reads.py
Byte-preserved against original HEAD: python/sglang/srt/mem_cache/unified_cache/control_transfers.py
Byte-preserved against original HEAD: python/sglang/srt/mem_cache/unified_cache/kv_control_adapter.py
Byte-preserved against original HEAD: python/sglang/srt/mem_cache/unified_cache/priority_claims.py
Byte-preserved against original HEAD: python/sglang/srt/managers/scheduler_components/weight_updater.py
Byte-preserved against upstream: python/sglang/srt/managers/scheduler_components/kv_events_publisher.py

```

Required proof command:

```bash
PYTHONPATH=python .venv/bin/python -m pytest -q --tb=short   test/registered/unit/managers/test_kv_control.py   test/registered/unit/managers/test_kv_policy.py   test/registered/unit/mem_cache/test_kv_control_cache.py   test/registered/unit/mem_cache/test_kv_control_service.py   test/registered/unit/mem_cache/test_kv_control_transfers.py   test/registered/unit/mem_cache/test_kv_policy_runtime.py
```

All pytest output is retained under `/tmp/sglang-rfc-build/`. Failures were inspected through the final summary, including subtest failures:

| Run / log | Result and diagnosis |
| --- | --- |
| Parent-provided pre-merge baseline | 49 passed, 17 failed/subfailed, 21 subtests passed. Reported failure: CUDA unregister on a fake/unregistered allocation, followed by `cudaGetErrorString(int=713)` TypeError. Not rerun on the old tree here. |
| `p0-tests-initial.log` | 46 failed/subfailed, 30 passed, 11 subtests passed, 24.86 s. Post-merge construction rejected the old `tree_components` keyword. |
| `p0-tests-fixture-api.log` | 9 failed/subfailed, 57 passed, 21 subtests passed, 24.99 s. Exposed missing allocator Mamba state and undeclared `attn_dcp_rank`. |
| `p0-tests-registration-leak.log` | 5 failed/subfailed, 58 passed, 24 subtests passed, 19.24 s. New live-registration assertion caught storage fixtures leaving `pin_memory=False` through destruction. |
| `p0-tests.log` | **62 passed, 25 subtests passed in 18.89 s**, exit 0. Required six-file command. |
| `p0-cache-isolated.log` | **10 passed, 7 subtests passed in 16.33 s**, exit 0. `test_kv_control_cache.py` in its own process. |
| `p0-transfers-isolated.log` | **11 passed in 18.06 s**, exit 0. `test_kv_control_transfers.py` in its own process. |
| `p0-scheduling.log` | **66 passed, 64 subtests passed in 19.79 s**, exit 0. Existing `test_schedule_policy.py` and `test_prefill_adder.py` verify the merged scheduling paths. |

Each pytest process exited after its summary. No post-summary hang or lingering non-daemon worker was observed in these runs. No .venv packages or global toolchain configuration were changed.

Router/indexer proof, using the already provisioned toolchain without a global override:

```bash
PATH=/home/youwei/.cache/sglang-rfc-tools/cargo/bin:$PATH CARGO_HOME=/home/youwei/.cache/sglang-rfc-tools/cargo RUSTUP_HOME=/home/youwei/.cache/sglang-rfc-tools/rustup cargo +stable check --manifest-path sgl-router/Cargo.toml -p sgl-router -p sgl-kv-indexer
```

`p0-router-check.log`: exit 0, `Finished dev profile [unoptimized + debuginfo] target(s) in 1m 04s`. Router/indexer build is not blocked. The repository toolchain file requests Rust 1.92; this check explicitly used available stable 1.99.1. No Rust source or Cargo.lock changes were needed. Router runtime tests were not run.

Additional checks:

- AST parsing passed for 44 Python files under the integration paths (`p0-integration-audit.log`).
- `ruff check --select F` passed on all five manually changed code/test files (`p0-lint.log`).
- `git diff --check HEAD -- <the five manually changed code/test files>` passed.
- Unrestricted default Ruff reports 230 style diagnostics across those files. Per-file counts match their inherited baseline: 49/116/52 for the three production files at upstream, 10/3 for the two tests at original HEAD. Details: `p0-ruff-comparison.json` and `p0-ruff-*.json`. P0 does not claim unrestricted lint is clean.
- Full-merge whitespace checks flag upstream CRLF/trailing whitespace in four Hunyuan files. All four were verified byte-identical to the fixed upstream snapshot: `python/sglang/multimodal_gen/runtime/models/dits/hunyuan_image3.py`, `python/sglang/multimodal_gen/runtime/models/dits/hunyuan_image3_utils.py`, `python/sglang/multimodal_gen/runtime/pipelines/hunyuan_image3.py`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/hunyuan_image3/ar_stage.py`. Evidence: `p0-diff-check.log`. No unrelated formatting rewrite was made.

Remaining limits and handoff:

- P0's requested CPU regressions and Router/indexer build pass. The parent still owns independent review, testing and the merge commit. Staged changes are reviewable; the merge remains open.
- Actual CUDA driver error formatting and real pinned-memory/DMA behavior are unvalidated by the fake-driver fixture. Production host cleanup remains unchanged. CPU file-store success is not evidence of Mooncake or GPU execution.
- P1 protocol/schema/capability fixtures, P2 backend/PREFETCH, P3 DEMOTE, P4 PIN, P5 Router association/observation, P6 orchestration/routing, P7 feedback/recovery, P8 Mooncake/GPU joint acceptance/benchmarks and P9 expanded configurations are all pending. No new public action, PIN capability, orchestrator crate, Mooncake protection owner, or runtime deployment was introduced.

The full changed-file list below is generated by `git diff --name-status HEAD`; rename records include both source and destination. It includes inherited upstream changes and the seven manual P0 files enumerated above. A duplicate machine-readable manifest is available at `/tmp/sglang-rfc-build/p0-changed-files.txt`; original conflicts are recorded in `/tmp/sglang-rfc-build/p0-conflicts.txt`.

Total manifest records: 4284.

<details>
<summary>Complete P0 merge changed-file manifest</summary>

```text
M	.agents/skills/add-jit-kernel/SKILL.md
M	.agents/skills/ci-workflow-guide/SKILL.md
M	.agents/skills/clean-startup-log/references/noise-sources.md
M	.agents/skills/cookbook-review-pr/SKILL.md
M	.agents/skills/kernel-organization/SKILL.md
M	.agents/skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md
M	.agents/skills/llm-torch-profiler-analysis/references/source-map.md
M	.agents/skills/llm-torch-profiler-analysis/scripts/triage_kernel_helpers.py
M	.agents/skills/processor-model-parity/SKILL.md
M	.agents/skills/sglang-runtime-context/SKILL.md
M	.claude/rules/no-getattr-defensive.md
M	.claude/rules/unit-test-admission.md
M	.github/CODEOWNERS
M	.github/FOLDER_README.md
M	.github/actions/check-pr-test-health/action.test.cjs
M	.github/actions/check-pr-test-health/action.yml
M	.github/actions/download-rust-ext/action.yml
M	.github/labeler.yml
D	.github/workflows/_npu-analyze-failure.yml
M	.github/workflows/_npu-pr-test-stage.yml
M	.github/workflows/_npu-single-node-test-stage.yml
M	.github/workflows/_pr-test-check-changes.yml
M	.github/workflows/_pr-test-sgl-kernel-build.yml
M	.github/workflows/_pr-test-simulator-cpu.yml
A	.github/workflows/_pr-test-simulator-gpu.yml
M	.github/workflows/_pr-test-stage-cpu.yml
M	.github/workflows/_pr-test-stage.yml
M	.github/workflows/bot-bump-docs-version.yml
M	.github/workflows/bot-bump-flashinfer-version.yml
M	.github/workflows/bot-bump-kernel-version-to-sglang.yml
M	.github/workflows/bot-bump-kernel-version.yml
M	.github/workflows/bot-bump-sglang-version.yml
M	.github/workflows/ci-coverage-overview.yml
M	.github/workflows/coverage-collection-npu.yml
M	.github/workflows/diffusion-ci-gt-gen-npu.yml
M	.github/workflows/diffusion-ci-gt-gen.yml
R098	.github/workflows/nightly-experimental-sgl-router-docker.yml	.github/workflows/nightly-sgl-router-docker.yml
M	.github/workflows/nightly-test-amd-miles-rocm10-mi30x.yml
M	.github/workflows/nightly-test-amd-miles-rocm10.yml
M	.github/workflows/nightly-test-amd.yml
M	.github/workflows/pr-gate.yml
M	.github/workflows/pr-test-amd-extra.yml
M	.github/workflows/pr-test-amd.yml
M	.github/workflows/pr-test-extra.yml
M	.github/workflows/pr-test-jit-kernel.yml
M	.github/workflows/pr-test-mlu.yml
M	.github/workflows/pr-test-multimodal-gen.yml
M	.github/workflows/pr-test-npu.yml
M	.github/workflows/pr-test-ppu.yml
M	.github/workflows/pr-test-sgl-kernel.yml
M	.github/workflows/pr-test-sgl-router.yml
M	.github/workflows/pr-test-xeon.yml
M	.github/workflows/pr-test.yml
M	.github/workflows/release-docker-amd-gfx1151-nightly.yml
M	.github/workflows/release-docker-amd-rocm10.yml
M	.github/workflows/release-docker-amd-rocm720-nightly.yml
M	.github/workflows/release-docker-intel-xpu-nightly.yml
M	.github/workflows/release-docker-npu-nightly.yml
M	.github/workflows/release-docker-npu.yml
M	.github/workflows/release-pypi-nightly.yml
M	.github/workflows/release-pypi-pr.yml
M	.github/workflows/release-pypi.yml
A	.github/workflows/release-whl-deepep-v25.yml
M	.github/workflows/release-whl-deepep.yml
M	.github/workflows/release-whl-deepgemm.yml
A	.github/workflows/release-whl-flashmla.yml
M	.github/workflows/release-whl-kernel.yml
M	.github/workflows/rerun-test.yml
A	.github/workflows/run-cpu-ci.yml
M	.github/workflows/runner-utilization.yml
M	.github/workflows/slash-command-handler.yml
M	.github/workflows/weekly-update-est-time.yml
M	.gitignore
M	.pre-commit-config.yaml
M	3rdparty/amd/wheel/sgl-kernel/build_rocm.sh
A	PLAN-REVIEW-LOG.md
A	PLAN.md
M	README.md
M	benchmark/bench_linear_attention/bench_cutedsl_kda_decode.py
M	benchmark/bench_linear_attention/bench_gdn_decode.py
M	benchmark/bench_linear_attention/bench_kda_decode.py
M	benchmark/kernels/attention/bench_gdn_replayssm_decode.py
M	benchmark/kernels/bench_kda_verify_sweep.py
M	benchmark/kernels/deepseek/benchmark_q8kv8_kv_gather.py
M	benchmark/kernels/deepseek/benchmark_q8kv8_q_prep.py
M	benchmark/kernels/fused_moe_triton/common_utils.py
M	benchmark/kernels/quantization/tuning_mxfp8_native_gfx95.py
A	benchmark/kernels/sampling/bench_deterministic_fast_sample.py
M	docker/Dockerfile
M	docker/Dockerfile.cu134
A	docker/beam/beam.Dockerfile
M	docker/npu.Dockerfile
M	docker/rocm.Dockerfile
A	docker/sgl-deep-ep-v25.Dockerfile
A	docker/sgl-flash-mla.Dockerfile
M	docker/sgl-router.Dockerfile
M	docker/xpu.Dockerfile
A	docs/cards/logos/recursive.png
M	docs/cookbook/autoregressive/DeepReinforce/Ornith-1.0.mdx
A	docs/cookbook/autoregressive/DeepReinforce/Ornith-1.5.mdx
M	docs/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx
M	docs/cookbook/autoregressive/DeepSeek/DeepSeek-R1.mdx
M	docs/cookbook/autoregressive/DeepSeek/DeepSeek-V3.mdx
M	docs/cookbook/autoregressive/DeepSeek/DeepSeek-V3_1.mdx
M	docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
M	docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx
M	docs/cookbook/autoregressive/Ernie/Ernie4.5.mdx
M	docs/cookbook/autoregressive/GLM/GLM-5.2.mdx
M	docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx
M	docs/cookbook/autoregressive/GLM/GLM-5.3.mdx
M	docs/cookbook/autoregressive/Google/Gemma4.mdx
M	docs/cookbook/autoregressive/InclusionAI/Ling-3.1-flash.mdx
A	docs/cookbook/autoregressive/Kev/kev.mdx
M	docs/cookbook/autoregressive/LiquidAI/LFM2.5.mdx
A	docs/cookbook/autoregressive/LiquidAI/LFM3.mdx
M	docs/cookbook/autoregressive/Meta/MuseSparkOpen.mdx
M	docs/cookbook/autoregressive/MiniMax/MiniMax-M2.5.mdx
A	docs/cookbook/autoregressive/MiniMax/MiniMax-M3.1.mdx
M	docs/cookbook/autoregressive/MiniMax/MiniMax-M3.mdx
M	docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx
M	docs/cookbook/autoregressive/OpenAI/GPT-OSS.mdx
M	docs/cookbook/autoregressive/Qwen/Qwen3-Next.mdx
M	docs/cookbook/autoregressive/Qwen/Qwen3.5.mdx
M	docs/cookbook/autoregressive/Qwen/Qwen3.6.mdx
M	docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx
A	docs/cookbook/autoregressive/Recursive/Flint.mdx
M	docs/cookbook/autoregressive/Reflection/Beam-1_0.mdx
M	docs/cookbook/autoregressive/intro.mdx
M	docs/cookbook/diffusion/Cosmos/Cosmos3.mdx
M	docs/cookbook/diffusion/Ernie-Image/Ernie-Image.mdx
M	docs/cookbook/diffusion/FLUX/FLUX.3.mdx
M	docs/cookbook/diffusion/FLUX/FLUX.mdx
A	docs/cookbook/diffusion/HY-Nomas/HY-Nomas.mdx
M	docs/cookbook/diffusion/Ideogram/Ideogram4.mdx
M	docs/cookbook/diffusion/Krea/Krea-2.mdx
M	docs/cookbook/diffusion/LTX/LTX2 & LTX2.3.mdx
M	docs/cookbook/diffusion/LTX/LTX2.5.mdx
M	docs/cookbook/diffusion/LingBot-World/LingBot-World-2.0.mdx
M	docs/cookbook/diffusion/LingBot-World/LingBot-World.mdx
M	docs/cookbook/diffusion/LongLive/LongLive-2.0.mdx
M	docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx
M	docs/cookbook/diffusion/OpenMOSS/MOSS-Image.mdx
M	docs/cookbook/diffusion/Qwen-Image/Qwen-Image.mdx
M	docs/cookbook/diffusion/SANA-WM/SANA-WM.mdx
M	docs/cookbook/diffusion/Wan/Wan-Animate-2.mdx
M	docs/cookbook/diffusion/Wan/Wan2.1.mdx
M	docs/docs.json
M	docs/docs/advanced_features/attention_backend.mdx
M	docs/docs/advanced_features/breakable_cuda_graph.mdx
M	docs/docs/advanced_features/dcp.mdx
A	docs/docs/advanced_features/engram.mdx
M	docs/docs/advanced_features/expert_parallelism.mdx
M	docs/docs/advanced_features/hicache_best_practices.mdx
M	docs/docs/advanced_features/hicache_design.mdx
A	docs/docs/advanced_features/kv_cache_sharing.mdx
A	docs/docs/advanced_features/paged_experts.mdx
M	docs/docs/advanced_features/pd_disaggregation.mdx
M	docs/docs/advanced_features/server_arguments.mdx
M	docs/docs/advanced_features/server_arguments_undocumented.txt
M	docs/docs/advanced_features/sgl_model_gateway.mdx
M	docs/docs/advanced_features/sglang_for_rl.mdx
M	docs/docs/advanced_features/speculative_decoding.mdx
M	docs/docs/basic_usage/aws_sagemaker.mdx
M	docs/docs/basic_usage/native_api.mdx
M	docs/docs/developer_guide/contribution_guide.mdx
M	docs/docs/developer_guide/layer_boundary.mdx
M	docs/docs/hardware-platforms/ascend-npus/evaluation/accuracy_evaluation.mdx
M	docs/docs/hardware-platforms/ascend-npus/faq.mdx
M	docs/docs/hardware-platforms/ascend-npus/getting-started/installation.mdx
M	docs/docs/hardware-platforms/ascend-npus/getting-started/quick_start.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/deepseek_r1.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/deepseek_v3_2.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/kimi_k2_6.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/mimo_v2_5_pro.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/mimo_v2_flash.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/minimax_m2_5.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/qwen3_235b_a22b.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/qwen3_30b_a3b.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/qwen3_32b.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/qwen3_8b.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/qwen3_next_80b_a3b_instruct.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_r1.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v3_2.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v4_flash.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/glm_5_1.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/glm_5_2.mdx
A	docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/glm_5_3.mdx
A	docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/glm_5_3_flash.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/hy3.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/kimi_k2_6.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/mimo_v2_flash.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/minimax_m2_5.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_235b_a22b.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_30b_a3b.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_32b.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_5_397b.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_6_27b.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_6_35b_a3b.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_8_max.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_8b.mdx
M	docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_next_80b_a3b_instruct.mdx
M	docs/docs/hardware-platforms/ascend-npus/optimization/quantization.mdx
M	docs/docs/hardware-platforms/ascend-npus/reference/glossary.mdx
M	docs/docs/hardware-platforms/cpu_server.mdx
M	docs/docs/hardware-platforms/plugin.mdx
A	docs/docs/hardware-platforms/sail.mdx
M	docs/docs/hardware-platforms/tpu.mdx
M	docs/docs/hardware-platforms/xpu.mdx
M	docs/docs/references/environment_variables.mdx
M	docs/docs/sglang-diffusion/api/cli.mdx
M	docs/docs/sglang-diffusion/api/openai_api.mdx
M	docs/docs/sglang-diffusion/comfyui.mdx
M	docs/docs/sglang-diffusion/contributing.mdx
M	docs/docs/sglang-diffusion/deployment_cookbook.mdx
M	docs/docs/sglang-diffusion/disaggregation.mdx
M	docs/docs/sglang-diffusion/environment_variables.mdx
M	docs/docs/sglang-diffusion/fused_kernels.mdx
M	docs/docs/sglang-diffusion/models_with_pe.mdx
M	docs/docs/sglang-diffusion/performance-optimization.mdx
M	docs/docs/sglang-diffusion/quantization.mdx
M	docs/docs/sglang-diffusion/realtime_models.mdx
M	docs/docs/sglang-diffusion/support_new_models.mdx
M	docs/docs/supported-models/decision_models.mdx
M	docs/scripts/tests/diffusion-builders.test.mjs
A	docs/scripts/tests/minimax-m25-deployment.test.mjs
M	docs/scripts/tests/wan-animate-2-config.test.mjs
M	docs/src/snippets/_agentx.jsx
M	docs/src/snippets/_deployment.jsx
M	docs/src/snippets/_playground.jsx
A	docs/src/snippets/agentx/MiniMaxAI/minimax-m3.1.jsx
M	docs/src/snippets/agentx/Qwen/qwen3.5.jsx
M	docs/src/snippets/agentx/Qwen/qwen3.8-flash-next.jsx
M	docs/src/snippets/agentx/deepseek-ai/deepseek-v4.jsx
M	docs/src/snippets/agentx/deepseek-ai/deepseek-v4_1.jsx
M	docs/src/snippets/agentx/meta-models/muse-spark-open.jsx
M	docs/src/snippets/agentx/zai-org/glm-5.2.jsx
A	docs/src/snippets/agentx/zai-org/glm-5.3.jsx
M	docs/src/snippets/autoregressive/deepseek-ocr-deployment.jsx
M	docs/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx
M	docs/src/snippets/autoregressive/deepseek-v3-deployment.jsx
M	docs/src/snippets/autoregressive/deepseek-v31-deployment.jsx
M	docs/src/snippets/autoregressive/ernie-45-deployment.jsx
M	docs/src/snippets/autoregressive/gemma4-deployment.jsx
M	docs/src/snippets/autoregressive/gpt-oss-deployment.jsx
M	docs/src/snippets/autoregressive/intern-s1-deployment.jsx
M	docs/src/snippets/autoregressive/kimi-k25-deployment.jsx
M	docs/src/snippets/autoregressive/kimi-k26-deployment.jsx
M	docs/src/snippets/autoregressive/kimi-k27-code-deployment.jsx
M	docs/src/snippets/autoregressive/minimax-m2-deployment.jsx
M	docs/src/snippets/autoregressive/minimax-m25-deployment.jsx
M	docs/src/snippets/autoregressive/minimax-m27-deployment.jsx
M	docs/src/snippets/autoregressive/qwen3-next-deployment.jsx
M	docs/src/snippets/autoregressive/qwen35-deployment.jsx
M	docs/src/snippets/autoregressive/qwen36-deployment.jsx
M	docs/src/snippets/configs/BlackForestLabs/flux.jsx
A	docs/src/snippets/configs/HY-Nomas/hy-nomas.jsx
M	docs/src/snippets/configs/Lightricks/ltx2.jsx
M	docs/src/snippets/configs/Lightricks/ltx25.jsx
A	docs/src/snippets/configs/LiquidAI/lfm3-benchmarks.jsx
A	docs/src/snippets/configs/LiquidAI/lfm3.jsx
M	docs/src/snippets/configs/MiniMaxAI/minimax-h3.jsx
A	docs/src/snippets/configs/MiniMaxAI/minimax-m3.1.jsx
M	docs/src/snippets/configs/OpenMOSS/mova.jsx
M	docs/src/snippets/configs/Qwen/qwen-image-edit.jsx
M	docs/src/snippets/configs/Qwen/qwen-image.jsx
M	docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx
M	docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx
A	docs/src/snippets/configs/RecursiveSI/flint.jsx
M	docs/src/snippets/configs/TongyiMAI/z-image-turbo.jsx
M	docs/src/snippets/configs/Wan/wan-animate-2.jsx
M	docs/src/snippets/configs/Wan/wan21.jsx
M	docs/src/snippets/configs/Wan/wan22.jsx
M	docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
M	docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx
M	docs/src/snippets/configs/inclusionAI/ling-3.1-flash-benchmarks.jsx
M	docs/src/snippets/configs/inclusionAI/ling-3.1-flash.jsx
M	docs/src/snippets/configs/meta-models/muse-spark.jsx
A	docs/src/snippets/configs/ornith-ai/ornith-1.5.jsx
M	docs/src/snippets/configs/popular-models.jsx
M	docs/src/snippets/configs/reflection/beam-1.0.jsx
M	docs/src/snippets/configs/zai-org/glm-5.2.jsx
M	docs/src/snippets/configs/zai-org/glm-5.3-flash-benchmarks.jsx
M	docs/src/snippets/configs/zai-org/glm-5.3-flash.jsx
M	docs/src/snippets/configs/zai-org/glm-5.3.jsx
M	docs/src/snippets/diffusion/model-catalog.jsx
M	examples/checkpoint_engine/update.py
D	experimental/sgl-router/src/state/kv_events/prefix_provider.rs
D	experimental/sgl-router/tests/component/tokenizer/deepseek.rs
D	experimental/sgl-router/tests/fixtures/chat_render_parity/deepseek-v4-flash.json
D	experimental/sgl-router/tests/fixtures/deepseek/v4.json
D	experimental/sgl-router/tests/fixtures/deepseek/v41.json
D	experimental/sgl-router/tests/scripts/generate_deepseek_parity.py
M	proto/sglang/api/v1/generate.proto
A	proto/sglang/api/v1/hicache.proto
M	proto/sglang/api/v1/service.proto
M	python/pyproject.toml
M	python/pyproject_cpu.toml
M	python/pyproject_npu.toml
M	python/pyproject_other.toml
M	python/pyproject_xpu.toml
M	python/setup.py
M	python/sglang/__init__.py
M	python/sglang/api/v1/api_types.py
M	python/sglang/benchmark/datasets/common.py
M	python/sglang/benchmark/datasets/generated_shared_prefix.py
M	python/sglang/benchmark/datasets/speed_bench.py
M	python/sglang/benchmark/dspark_sps_profiler.py
M	python/sglang/benchmark/one_batch_server.py
M	python/sglang/benchmark/serving.py
M	python/sglang/kernels/README.md
M	python/sglang/kernels/aot/Dockerfile
M	python/sglang/kernels/aot/README.md
M	python/sglang/kernels/aot/build.sh
M	python/sglang/kernels/aot/csrc/common_extension.cc
M	python/sglang/kernels/aot/csrc/common_extension_musa.cc
A	python/sglang/kernels/aot/csrc/common_extension_sail.cc
M	python/sglang/kernels/aot/csrc/cpu/norm.cpp
M	python/sglang/kernels/aot/csrc/cpu/topk.cpp
M	python/sglang/kernels/aot/csrc/cpu/torch_extension_cpu.cpp
M	python/sglang/kernels/aot/csrc/metal/README.md
M	python/sglang/kernels/aot/csrc/metal/rope_pool_fused.cpp
A	python/sglang/kernels/aot/csrc/moe/gguf_moe_weights.cuh
M	python/sglang/kernels/aot/csrc/moe/moe_sum.cu
M	python/sglang/kernels/aot/csrc/quantization/gguf/dequantize.cuh
M	python/sglang/kernels/aot/csrc/quantization/gguf/gguf_kernel.cu
M	python/sglang/kernels/aot/csrc/quantization/gguf/moe.cuh
A	python/sglang/kernels/aot/include/sgl_kernel/quantization/gguf/device_compat.cuh
R099	python/sglang/kernels/aot/csrc/quantization/gguf/ggml-common.h	python/sglang/kernels/aot/include/sgl_kernel/quantization/gguf/ggml-common.h
R098	python/sglang/kernels/aot/csrc/quantization/gguf/moe_vec.cuh	python/sglang/kernels/aot/include/sgl_kernel/quantization/gguf/moe_vec.cuh
A	python/sglang/kernels/aot/include/sgl_kernel/quantization/gguf/quantize_q8_1.cuh
R099	python/sglang/kernels/aot/csrc/quantization/gguf/vecdotq.cuh	python/sglang/kernels/aot/include/sgl_kernel/quantization/gguf/vecdotq.cuh
M	python/sglang/kernels/aot/include/sgl_kernel_ops.h
M	python/sglang/kernels/aot/pyproject.toml
A	python/sglang/kernels/aot/pyproject_sail.toml
M	python/sglang/kernels/aot/python/sgl_kernel/__init__.py
M	python/sglang/kernels/aot/python/sgl_kernel/metal.py
M	python/sglang/kernels/aot/python/sgl_kernel/moe.py
M	python/sglang/kernels/aot/setup_metal.py
A	python/sglang/kernels/aot/setup_sail.py
M	python/sglang/kernels/fused_op.py
M	python/sglang/kernels/jit/csrc/deepseek_v4/c1.cuh
M	python/sglang/kernels/jit/csrc/deepseek_v4/c2.cuh
M	python/sglang/kernels/jit/csrc/deepseek_v4/flashmla_sched_meta.cuh
M	python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh
M	python/sglang/kernels/jit/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh
M	python/sglang/kernels/jit/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh
M	python/sglang/kernels/jit/csrc/deepseek_v4/store.cuh
M	python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh
M	python/sglang/kernels/jit/csrc/deepseek_v4/wo_a_fused.cuh
A	python/sglang/kernels/jit/csrc/diffusion/minimax_h3_vae_output.cuh
M	python/sglang/kernels/jit/csrc/distributed/all_reduce_fusion.cuh
A	python/sglang/kernels/jit/csrc/distributed/beam_all_reduce.cuh
M	python/sglang/kernels/jit/csrc/elementwise/qknorm.cuh
A	python/sglang/kernels/jit/csrc/elementwise/small_field_copy.cuh
M	python/sglang/kernels/jit/csrc/gemm/hopper_bf16_gemv.cuh
M	python/sglang/kernels/jit/csrc/gemm/marlin/gptq_marlin.cuh
M	python/sglang/kernels/jit/csrc/gemm/marlin/kernel.h
M	python/sglang/kernels/jit/csrc/gemm/marlin/marlin_template.h
M	python/sglang/kernels/jit/csrc/gemm/marlin_moe/kernel.h
M	python/sglang/kernels/jit/csrc/gemm/marlin_moe/marlin_template.h
M	python/sglang/kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh
M	python/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh
M	python/sglang/kernels/jit/csrc/gemm/sm120_fp8_gemv.cuh
A	python/sglang/kernels/jit/csrc/gemm/sm120_nvfp4_moe_mtp_mma.cuh
A	python/sglang/kernels/jit/csrc/gemm/sm120_nvfp4_weight_transpose.cuh
M	python/sglang/kernels/jit/csrc/gemm/tiny_gemm.cuh
M	python/sglang/kernels/jit/csrc/kimi_k3/mla_output_gate.cuh
A	python/sglang/kernels/jit/csrc/kvcacheio/page_row.cuh
M	python/sglang/kernels/jit/csrc/lora/trtllm_lora_temp/moe_lora_merged_align_kernel.cu
M	python/sglang/kernels/jit/csrc/minimax/fused_gemma_qknorm_rope.cuh
M	python/sglang/kernels/jit/csrc/minimax/fused_store_kv_index.cuh
A	python/sglang/kernels/jit/csrc/minimax/msa_v1/attn/dequant.cuh
A	python/sglang/kernels/jit/csrc/minimax/nvfp4_store_kv_index.cuh
M	python/sglang/kernels/jit/csrc/moe/align_single_token.cuh
A	python/sglang/kernels/jit/csrc/moe/cutlass_prepare_moe_input.cuh
M	python/sglang/kernels/jit/csrc/moe/expert_pack_mxfp4.cuh
A	python/sglang/kernels/jit/csrc/moe/gguf_moe_vec.cuh
M	python/sglang/kernels/jit/csrc/moe/moe_align_kernel.cu
M	python/sglang/kernels/jit/csrc/moe/moe_permute_prepare.cu
A	python/sglang/kernels/jit/csrc/moe/paged_experts.cuh
A	python/sglang/kernels/jit/csrc/moe/shuffle_weighted_sum.cuh
M	python/sglang/kernels/jit/csrc/muse_spark/norm_fp4_quant.cuh
M	python/sglang/kernels/jit/csrc/muse_spark/routing.cuh
M	python/sglang/kernels/jit/csrc/muse_spark/row_sum.cuh
M	python/sglang/kernels/jit/csrc/qprep_bf16_fp8_sm90/kernel.cuh
M	python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/fp4_utils.cuh
M	python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kv_layout.cuh
M	python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh
M	python/sglang/kernels/jit/include/sgl_kernel/gemm/dot_product.cuh
A	python/sglang/kernels/jit/include/sgl_kernel/quantization/gguf/device_compat.cuh
A	python/sglang/kernels/jit/include/sgl_kernel/quantization/gguf/ggml-common.h
A	python/sglang/kernels/jit/include/sgl_kernel/quantization/gguf/moe_vec.cuh
A	python/sglang/kernels/jit/include/sgl_kernel/quantization/gguf/quantize_q8_1.cuh
A	python/sglang/kernels/jit/include/sgl_kernel/quantization/gguf/vecdotq.cuh
M	python/sglang/kernels/jit/utils/arch.py
M	python/sglang/kernels/jit/utils/common.py
M	python/sglang/kernels/jit/utils/compile/__init__.py
M	python/sglang/kernels/jit/utils/compile/cache.py
M	python/sglang/kernels/jit/utils/compile/loader.py
A	python/sglang/kernels/jit/utils/compile/sailify.py
M	python/sglang/kernels/jit/utils/compile/spec.py
M	python/sglang/kernels/jit/utils/compile/toolchain.py
R088	python/sglang/kernels/ops/muse_spark/helion_runtime.py	python/sglang/kernels/jit/utils/helion_runtime.py
A	python/sglang/kernels/jit/utils/hipcc.py
M	python/sglang/kernels/kda_kernels/README.md
A	python/sglang/kernels/kda_kernels/csrc/attention/kda_decode_gfx950.cu
A	python/sglang/kernels/kda_kernels/csrc/attention/kda_prefill_gfx950.cu
M	python/sglang/kernels/kda_kernels/csrc/diffusion/ltx2_qknorm_split_rope.cuh
A	python/sglang/kernels/kda_kernels/kda_decode_gfx950.py
A	python/sglang/kernels/kda_kernels/kda_prefill_gfx950.py
M	python/sglang/kernels/kda_kernels/qwen38_qsa_sm121/README.md
M	python/sglang/kernels/kda_kernels/qwen38_qsa_sm121/__init__.py
M	python/sglang/kernels/kda_kernels/qwen38_qsa_sm121/kernel.py
M	python/sglang/kernels/ops/__init__.py
M	python/sglang/kernels/ops/activation/__init__.py
R100	python/sglang/kernels/ops/elementwise/bias_gelu.py	python/sglang/kernels/ops/activation/bias_gelu.py
A	python/sglang/kernels/ops/activation/gelu_and_mul_triton.py
R100	python/sglang/kernels/ops/activation/_jit_situ_and_mul.py	python/sglang/kernels/ops/activation/kimi_k3_situ_and_mul.py
R093	python/sglang/kernels/ops/moe/minimax_m3_swiglu.py	python/sglang/kernels/ops/activation/minimax_m3_swiglu.py
A	python/sglang/kernels/ops/activation/silu_and_mul_clamp.py
M	python/sglang/kernels/ops/activation/softcap.py
A	python/sglang/kernels/ops/activation/swiglu_mxfp8_swizzled.py
M	python/sglang/kernels/ops/attention/__init__.py
R100	python/sglang/kernels/ops/kvcache/aiter_unified_attention.py	python/sglang/kernels/ops/attention/aiter_unified_page_table.py
M	python/sglang/kernels/ops/attention/attn_res.py
M	python/sglang/kernels/ops/attention/attn_res_hip.py
M	python/sglang/kernels/ops/attention/dcp_kernels.py
M	python/sglang/kernels/ops/attention/decode_attention.py
M	python/sglang/kernels/ops/attention/dsa/cp_split.py
R100	python/sglang/kernels/ops/attention/cutedsl_fp8_paged_mqa_logits.py	python/sglang/kernels/ops/attention/dsa/cutedsl_fp8_paged_mqa_logits.py
M	python/sglang/kernels/ops/attention/dsa/cutedsl_paged_mqa_logits.py
M	python/sglang/kernels/ops/attention/dsa/dequant_k_cache.py
R100	python/sglang/kernels/ops/attention/fused_metadata_copy.py	python/sglang/kernels/ops/attention/dsa/fused_metadata_copy.py
M	python/sglang/kernels/ops/attention/dsa/hip_gfx950/__init__.py
A	python/sglang/kernels/ops/attention/dsa/indexer_q.py
R100	python/sglang/kernels/ops/attention/dsa_kpool_metadata/__init__.py	python/sglang/kernels/ops/attention/dsa/kpool_metadata/__init__.py
R097	python/sglang/kernels/ops/attention/dsa_kpool_metadata/decode.py	python/sglang/kernels/ops/attention/dsa/kpool_metadata/decode.py
R097	python/sglang/kernels/ops/attention/dsa_kpool_metadata/draft_extend.py	python/sglang/kernels/ops/attention/dsa/kpool_metadata/draft_extend.py
R100	python/sglang/kernels/ops/attention/dsa_kpool_metadata/scan.py	python/sglang/kernels/ops/attention/dsa/kpool_metadata/scan.py
R096	python/sglang/kernels/ops/attention/dsa_kpool_metadata/verify.py	python/sglang/kernels/ops/attention/dsa/kpool_metadata/verify.py
R100	python/sglang/kernels/ops/attention/dsa_metadata.py	python/sglang/kernels/ops/attention/dsa/metadata.py
M	python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py
M	python/sglang/kernels/ops/attention/dsa/transform_index.py
M	python/sglang/kernels/ops/attention/dsa/triton_kernel.py
M	python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py
M	python/sglang/kernels/ops/attention/dsa/triton_sparse_mla_decode.py
M	python/sglang/kernels/ops/attention/dsa/triton_sparse_mla_prefill.py
M	python/sglang/kernels/ops/attention/dsv4/__init__.py
M	python/sglang/kernels/ops/attention/dsv4/attn.py
R098	python/sglang/kernels/ops/attention/dsv4_attn_metadata_kernels.py	python/sglang/kernels/ops/attention/dsv4/attn_metadata_kernels.py
M	python/sglang/kernels/ops/attention/dsv4/decode_attention_sm100.py
M	python/sglang/kernels/ops/attention/dsv4/decode_attention_sm100_gluon.py
M	python/sglang/kernels/ops/attention/dsv4/dequant_k_cache.py
M	python/sglang/kernels/ops/attention/dsv4/fp4_indexer.py
M	python/sglang/kernels/ops/attention/dsv4/fp4_indexer_hip.py
M	python/sglang/kernels/ops/attention/dsv4/fp4_rope_fake_quant.py
M	python/sglang/kernels/ops/attention/dsv4/index_logits.py
M	python/sglang/kernels/ops/attention/dsv4/kv_layout.py
R088	python/sglang/kernels/ops/attention/dsv4/elementwise.py	python/sglang/kernels/ops/attention/dsv4/norm_rope.py
M	python/sglang/kernels/ops/attention/dsv4/online_c128_mtp.py
R099	python/sglang/kernels/ops/attention/deepseek_v4_rope.py	python/sglang/kernels/ops/attention/dsv4/rope.py
M	python/sglang/kernels/ops/attention/dsv4/sparse_prefill_kernels.py
A	python/sglang/kernels/ops/attention/dsv4/tilelang_fp8_attention.py
M	python/sglang/kernels/ops/attention/dsv4/topk.py
R096	python/sglang/kernels/ops/attention/nsa_triton_decode/__init__.py	python/sglang/kernels/ops/attention/dsv4/triton_sparse_decode/__init__.py
R100	python/sglang/kernels/ops/attention/nsa_triton_decode/triton_mla_kernels_decode_fused.py	python/sglang/kernels/ops/attention/dsv4/triton_sparse_decode/triton_mla_kernels_decode_fused.py
R100	python/sglang/kernels/ops/attention/nsa_triton_decode/triton_mla_kernels_decode_optimized.py	python/sglang/kernels/ops/attention/dsv4/triton_sparse_decode/triton_mla_kernels_decode_optimized.py
M	python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_decode.py
M	python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/runtime.py
M	python/sglang/kernels/ops/attention/dsv4/wo_a.py
M	python/sglang/kernels/ops/attention/extend_attention.py
M	python/sglang/kernels/ops/attention/fla/chunk.py
M	python/sglang/kernels/ops/attention/fla/chunk_o.py
M	python/sglang/kernels/ops/attention/fla/fused_norm_gate.py
M	python/sglang/kernels/ops/attention/fla/utils.py
M	python/sglang/kernels/ops/attention/flash_attention_v4_sm120.py
M	python/sglang/kernels/ops/attention/flash_attn/cute/flash_fwd_mla_sm100.py
A	python/sglang/kernels/ops/attention/flash_mla_interface.py
R100	python/sglang/kernels/ops/kvcache/rope_cache.py	python/sglang/kernels/ops/attention/fused_qk_rope_reshape_and_cache.py
M	python/sglang/kernels/ops/attention/helion/kda_replayssm.py
A	python/sglang/kernels/ops/attention/kda_fused_decode_gfx950.py
R083	python/sglang/kernels/ops/kvcache/kv_indices.py	python/sglang/kernels/ops/attention/kv_indices.py
R100	python/sglang/kernels/ops/attention/fla/fused_gdn_gating.py	python/sglang/kernels/ops/attention/linear/fused_gdn_gating.py
R100	python/sglang/kernels/ops/attention/fla/fused_kda_conv_recurrent_verify.py	python/sglang/kernels/ops/attention/linear/fused_kda_conv_recurrent_verify.py
R100	python/sglang/kernels/ops/attention/fla/fused_recurrent_linear_replayssm.py	python/sglang/kernels/ops/attention/linear/fused_recurrent_linear_replayssm.py
R098	python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py	python/sglang/kernels/ops/attention/linear/fused_sigmoid_gating_recurrent.py
R100	python/sglang/kernels/ops/attention/fla/gdn_replayssm_spec_decode.py	python/sglang/kernels/ops/attention/linear/gdn_replayssm_spec_decode.py
R100	python/sglang/kernels/ops/attention/fla/gdn_replayssm_spec_fold.py	python/sglang/kernels/ops/attention/linear/gdn_replayssm_spec_fold.py
A	python/sglang/kernels/ops/attention/linear/kda_gfx950_prefill.py
R100	python/sglang/kernels/ops/attention/fla/kda_replayssm_spec_decode.py	python/sglang/kernels/ops/attention/linear/kda_replayssm_spec_decode.py
M	python/sglang/kernels/ops/attention/linear/lightning_attn.py
D	python/sglang/kernels/ops/attention/log_scaling_tau.py
M	python/sglang/kernels/ops/attention/metadata.py
M	python/sglang/kernels/ops/attention/minimax_qknorm_rope.py
A	python/sglang/kernels/ops/attention/minimax_sparse/common/tile_ptr.py
M	python/sglang/kernels/ops/attention/minimax_sparse/decode/flash_with_topk_idx.py
M	python/sglang/kernels/ops/attention/minimax_sparse/decode/topk_sparse.py
M	python/sglang/kernels/ops/attention/minimax_sparse/prefill/flash_with_topk_idx.py
R100	python/sglang/srt/layers/attention/minimax_sparse_ops/gluon_prefill.py	python/sglang/kernels/ops/attention/minimax_sparse/prefill/gluon_prefill.py
M	python/sglang/kernels/ops/attention/minimax_sparse/prefill/topk_sparse.py
A	python/sglang/kernels/ops/attention/mla_qk_prep.py
A	python/sglang/kernels/ops/attention/msa_v1/__init__.py
A	python/sglang/kernels/ops/attention/msa_v1/compact_dequant.py
A	python/sglang/kernels/ops/attention/msa_v1/decode_plan.py
A	python/sglang/kernels/ops/attention/msa_v1/flashinfer_decode.py
R098	python/sglang/kernels/ops/muse_spark/fused_qk_norm.py	python/sglang/kernels/ops/attention/muse_spark_qk_norm_rope_store.py
R098	python/sglang/kernels/ops/muse_spark/helion_qk_norm.py	python/sglang/kernels/ops/attention/muse_spark_qk_norm_rope_store_helion.py
M	python/sglang/kernels/ops/attention/qprep_bf16_fp8_sm90.py
M	python/sglang/kernels/ops/attention/sparse_mla_q8kv8_prefill_sm90.py
A	python/sglang/kernels/ops/attention/table_qk_norm_rope.py
A	python/sglang/kernels/ops/attention/triton_block_ptr_compat.py
R099	python/sglang/kernels/ops/kvcache/trtllm_mha_graph_metadata.py	python/sglang/kernels/ops/attention/trtllm_mha_graph_metadata.py
R100	python/sglang/kernels/ops/kvcache/trtllm_mha_page_table.py	python/sglang/kernels/ops/attention/trtllm_mha_page_table.py
M	python/sglang/kernels/ops/attention/utils.py
M	python/sglang/kernels/ops/attention/vattn_asm_gfx950/__init__.py
M	python/sglang/kernels/ops/communication/__init__.py
M	python/sglang/kernels/ops/communication/all_reduce.py
M	python/sglang/kernels/ops/communication/all_reduce_fusion.py
A	python/sglang/kernels/ops/communication/attn_res_collective.py
A	python/sglang/kernels/ops/communication/beam_all_reduce.py
R100	python/sglang/kernels/ops/gemm/cutlass_all_gather_gemm.py	python/sglang/kernels/ops/communication/cutlass_all_gather_gemm.py
R100	python/sglang/kernels/ops/gemm/cutlass_gemm_reduce_scatter.py	python/sglang/kernels/ops/communication/cutlass_gemm_reduce_scatter.py
A	python/sglang/kernels/ops/communication/muse_spark_all_gather.py
R098	python/sglang/kernels/ops/muse_spark/all_gather_norm_residual.py	python/sglang/kernels/ops/communication/muse_spark_all_gather_norm_residual.py
R095	python/sglang/kernels/ops/muse_spark/fused_post_expert_norm_weighted_sum_trtllm.py	python/sglang/kernels/ops/communication/muse_spark_moe_finalize_all_reduce.py
R088	python/sglang/kernels/ops/muse_spark/fi_trtllm_moe_finalize_with_rmsnorm_cutedsl.py	python/sglang/kernels/ops/communication/muse_spark_moe_finalize_all_reduce_cutedsl.py
R100	python/sglang/kernels/ops/muse_spark/helion_post_expert.py	python/sglang/kernels/ops/communication/muse_spark_moe_finalize_all_reduce_helion.py
R076	python/sglang/kernels/ops/gemm/sp_collective_gemm.py	python/sglang/kernels/ops/communication/sp_collective_gemm.py
M	python/sglang/kernels/ops/diffusion/README.md
M	python/sglang/kernels/ops/diffusion/__init__.py
M	python/sglang/kernels/ops/diffusion/activation/silu_mul_bitexact.py
M	python/sglang/kernels/ops/diffusion/common/platform.py
A	python/sglang/kernels/ops/diffusion/layout/head_group_copy_triton.py
A	python/sglang/kernels/ops/diffusion/layout/minimax_h3_vae_output_jit.py
A	python/sglang/kernels/ops/diffusion/modulate/head_gate_triton.py
A	python/sglang/kernels/ops/diffusion/modulate/indexed_scale_shift_mxfp8_triton.py
M	python/sglang/kernels/ops/diffusion/modulate/scale_shift_triton.py
M	python/sglang/kernels/ops/diffusion/norm/channel_rmsnorm_preserve_reduction.py
M	python/sglang/kernels/ops/diffusion/norm/cutedsl_common/norm_fusion.py
M	python/sglang/kernels/ops/diffusion/norm/cutedsl_common/reduce.py
A	python/sglang/kernels/ops/diffusion/norm/layernorm_modulate_xpu_triton.py
M	python/sglang/kernels/ops/diffusion/norm/rmsnorm_cast_before_weight_bitexact.py
M	python/sglang/kernels/ops/diffusion/norm/scale_residual_norm_cutedsl.py
A	python/sglang/kernels/ops/diffusion/norm/scale_residual_norm_scale_shift_sycl.py
A	python/sglang/kernels/ops/diffusion/quantization/swiglu_fp8_rowwise_triton.py
A	python/sglang/kernels/ops/diffusion/rope/flux3_video_qknorm_rope_triton.py
A	python/sglang/kernels/ops/diffusion/rope/rmsnorm_interleaved_rope_cutedsl.py
M	python/sglang/kernels/ops/diffusion/sites/fused_linear_gelu_site.py
M	python/sglang/kernels/ops/elementwise/__init__.py
A	python/sglang/kernels/ops/elementwise/beam_fused.py
M	python/sglang/kernels/ops/elementwise/elementwise.py
R090	python/sglang/kernels/ops/attention/mla_output_gate.py	python/sglang/kernels/ops/elementwise/mla_output_gate.py
M	python/sglang/kernels/ops/elementwise/row_scale.py
R085	python/sglang/kernels/ops/muse_spark/row_sum.py	python/sglang/kernels/ops/elementwise/row_sum.py
M	python/sglang/kernels/ops/embeddings/__init__.py
M	python/sglang/kernels/ops/embeddings/engram_gate.py
M	python/sglang/kernels/ops/embeddings/engram_gather.py
M	python/sglang/kernels/ops/embeddings/engram_hash.py
R100	python/sglang/kernels/ops/speculative/ngram_embedding.py	python/sglang/kernels/ops/embeddings/ngram_embedding.py
M	python/sglang/kernels/ops/gemm/__init__.py
M	python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py
A	python/sglang/kernels/ops/gemm/dense_bf16_gemm_warp_splitk.py
A	python/sglang/kernels/ops/gemm/dsv4_indexer_head_weights_hip.py
M	python/sglang/kernels/ops/gemm/dsv4_wo_a.py
A	python/sglang/kernels/ops/gemm/flashinfer_bmm_fp8.py
A	python/sglang/kernels/ops/gemm/fp8_gfx1250.py
R097	python/sglang/kernels/ops/elementwise/hc_mix.py	python/sglang/kernels/ops/gemm/hc_mix_splitk.py
M	python/sglang/kernels/ops/gemm/hopper_bf16_gemv.py
R100	python/sglang/kernels/ops/muse_spark/router_gemm.py	python/sglang/kernels/ops/gemm/muse_spark_router_gemm.py
A	python/sglang/kernels/ops/gemm/muse_spark_router_gemv_cutedsl.py
R057	python/sglang/kernels/ops/quantization/mxfp8_aiter_gfx95.py	python/sglang/kernels/ops/gemm/mxfp8_aiter_gfx95.py
A	python/sglang/kernels/ops/gemm/mxfp8_amd_gfx95.py
R100	python/sglang/kernels/ops/quantization/mxfp8_flydsl_gfx95.py	python/sglang/kernels/ops/gemm/mxfp8_flydsl_gfx95.py
R100	python/sglang/kernels/ops/quantization/mxfp8_gemv_gfx95_configs.json	python/sglang/kernels/ops/gemm/mxfp8_gemv_gfx95_configs.json
R100	python/sglang/kernels/ops/quantization/mxfp8_native_amd_gfx95.py	python/sglang/kernels/ops/gemm/mxfp8_native_amd_gfx95.py
R100	python/sglang/kernels/ops/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py	python/sglang/kernels/ops/gemm/nvfp4_gemm_swiglu_nvfp4_quant.py
A	python/sglang/kernels/ops/gemm/router_gate_matvec.py
M	python/sglang/kernels/ops/gemm/router_gemv.py
M	python/sglang/kernels/ops/gemm/sm120_fp8_gemv.py
A	python/sglang/kernels/ops/gemm/sm120_nvfp4_moe.py
A	python/sglang/kernels/ops/gemm/smallm_fp8_blockscale_gfx950/__init__.py
A	python/sglang/kernels/ops/gemm/smallm_fp8_blockscale_gfx950/smallm_fp8_blockscale_gemm.hip
M	python/sglang/kernels/ops/gemm/smallm_fp8_gfx950/__init__.py
M	python/sglang/kernels/ops/gemm/tiny_gemm.py
M	python/sglang/kernels/ops/kv_canary/__init__.py
M	python/sglang/kernels/ops/kv_canary/scatter_req_token_ids.py
M	python/sglang/kernels/ops/kvcache/__init__.py
M	python/sglang/kernels/ops/kvcache/cache_ops.py
R100	python/sglang/kernels/ops/attention/fused_store_index_cache.py	python/sglang/kernels/ops/kvcache/dsa_fused_store_index_cache.py
R100	python/sglang/kernels/ops/attention/dsa/index_buf_accessor.py	python/sglang/kernels/ops/kvcache/dsa_index_k_cache.py
A	python/sglang/kernels/ops/kvcache/dsv41_window_layout.py
R100	python/sglang/kernels/ops/attention/dsv4/c128_cleanup.py	python/sglang/kernels/ops/kvcache/dsv4_c128_cleanup.py
A	python/sglang/kernels/ops/kvcache/dsv4_fp4_index_k_cache.py
A	python/sglang/kernels/ops/kvcache/dsv4_fp4_index_k_cache_hip.py
R100	python/sglang/kernels/ops/attention/dsv4/index_buf_accessor.py	python/sglang/kernels/ops/kvcache/dsv4_main_kv_store.py
R073	python/sglang/kernels/ops/attention/dsv4/request_window_copy.py	python/sglang/kernels/ops/kvcache/dsv4_request_window_copy.py
A	python/sglang/kernels/ops/kvcache/dsv4_unified_kv_store.py
M	python/sglang/kernels/ops/kvcache/hicache.py
M	python/sglang/kernels/ops/kvcache/minimax_store_kv_index.py
M	python/sglang/kernels/ops/kvcache/mla_buffer.py
R100	python/sglang/kernels/ops/quantization/mxfp8_interleave_sf.py	python/sglang/kernels/ops/kvcache/mxfp8_interleave_sf.py
A	python/sglang/kernels/ops/kvcache/mxfp8_quant_store_kv.py
A	python/sglang/kernels/ops/kvcache/nvfp4_hicache.py
A	python/sglang/kernels/ops/kvcache/nvfp4_store_kv_index.py
A	python/sglang/kernels/ops/kvcache/shuffle_5d_cache.py
M	python/sglang/kernels/ops/layernorm/__init__.py
R081	python/sglang/kernels/ops/muse_spark/fused_op_tuning.py	python/sglang/kernels/ops/layernorm/_muse_spark_tuning.py
A	python/sglang/kernels/ops/layernorm/beam_fused.py
A	python/sglang/kernels/ops/layernorm/fused_rmsnorm.py
M	python/sglang/kernels/ops/layernorm/hc_combine_norm.py
M	python/sglang/kernels/ops/layernorm/mhc.py
M	python/sglang/kernels/ops/layernorm/mhc_boundary_hip.py
M	python/sglang/kernels/ops/layernorm/mhc_mega.py
R096	python/sglang/kernels/ops/muse_spark/fused_norm_gate.py	python/sglang/kernels/ops/layernorm/muse_spark_norm_gate.py
R098	python/sglang/kernels/ops/muse_spark/helion_norm_gate.py	python/sglang/kernels/ops/layernorm/muse_spark_norm_gate_helion.py
R100	python/sglang/kernels/ops/muse_spark/norm_gate.py	python/sglang/kernels/ops/layernorm/muse_spark_norm_gate_jit.py
R089	python/sglang/kernels/ops/muse_spark/fused_norm_residual.py	python/sglang/kernels/ops/layernorm/muse_spark_norm_residual.py
R098	python/sglang/kernels/ops/muse_spark/norm_residual.py	python/sglang/kernels/ops/layernorm/muse_spark_norm_residual_jit.py
R056	python/sglang/kernels/ops/muse_spark/norm_fp4_quant.py	python/sglang/kernels/ops/layernorm/muse_spark_rmsnorm_nvfp4_quant.py
R099	python/sglang/kernels/ops/muse_spark/helion_norm_fp4_quant.py	python/sglang/kernels/ops/layernorm/muse_spark_rmsnorm_nvfp4_quant_helion.py
M	python/sglang/kernels/ops/layernorm/norm.py
R100	python/sglang/kernels/ops/quantization/rmsnorm_fake_quant_amd_gfx95.py	python/sglang/kernels/ops/layernorm/rmsnorm_fake_quant_amd_gfx95.py
A	python/sglang/kernels/ops/layernorm/rmsnorm_mxfp8.py
M	python/sglang/kernels/ops/lora/__init__.py
A	python/sglang/kernels/ops/lora/common/__init__.py
A	python/sglang/kernels/ops/lora/common/lora_a.py
A	python/sglang/kernels/ops/lora/common/lora_b.py
A	python/sglang/kernels/ops/lora/common/route_view.py
A	python/sglang/kernels/ops/lora/common/routing.py
M	python/sglang/kernels/ops/lora/dense/embedding_lora_a.py
M	python/sglang/kernels/ops/lora/moe/__init__.py
A	python/sglang/kernels/ops/lora/moe/activation.py
A	python/sglang/kernels/ops/lora/moe/activation_delta.py
A	python/sglang/kernels/ops/lora/moe/align_rows.py
A	python/sglang/kernels/ops/lora/moe/cutedsl/__init__.py
A	python/sglang/kernels/ops/lora/moe/cutedsl/api.py
A	python/sglang/kernels/ops/lora/moe/cutedsl/kernel_sm100_bf16.py
A	python/sglang/kernels/ops/lora/moe/cutedsl/kernel_sm100_fp8.py
A	python/sglang/kernels/ops/lora/moe/cutedsl/kernel_sm90_bf16.py
A	python/sglang/kernels/ops/lora/moe/cutedsl/kernel_sm90_fp8.py
A	python/sglang/kernels/ops/lora/moe/cutedsl/schedule_builder.py
A	python/sglang/kernels/ops/lora/moe/cutedsl/scheduler.py
A	python/sglang/kernels/ops/lora/moe/dispatch_checks.py
A	python/sglang/kernels/ops/lora/moe/dispatch_contiguous.py
A	python/sglang/kernels/ops/lora/moe/dispatch_masked.py
A	python/sglang/kernels/ops/lora/moe/dispatch_masked_small.py
A	python/sglang/kernels/ops/lora/moe/finalize.py
A	python/sglang/kernels/ops/lora/moe/fp8_quant.py
A	python/sglang/kernels/ops/lora/moe/fused_act.py
A	python/sglang/kernels/ops/lora/moe/lora_b.py
M	python/sglang/kernels/ops/lora/moe/trtllm_lora_temp/moe_lora_merged_align.py
M	python/sglang/kernels/ops/lora/moe/trtllm_lora_temp/topk_softmax_pack.py
M	python/sglang/kernels/ops/lora/moe/trtllm_lora_temp/virtual_experts.py
M	python/sglang/kernels/ops/lora/moe/virtual_experts.py
M	python/sglang/kernels/ops/lplb/__init__.py
M	python/sglang/kernels/ops/mamba/__init__.py
A	python/sglang/kernels/ops/mamba/gdn_prefill_track.py
R098	python/sglang/kernels/ops/attention/fla/kda_prefill_graph_track.py	python/sglang/kernels/ops/mamba/kda_prefill_track.py
M	python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py
M	python/sglang/kernels/ops/memory/__init__.py
M	python/sglang/kernels/ops/memory/gpu_tensor_hash.py
A	python/sglang/kernels/ops/memory/small_field_copy.py
M	python/sglang/kernels/ops/mm/__init__.py
M	python/sglang/kernels/ops/mm/process/__init__.py
A	python/sglang/kernels/ops/mm/process/lanczos.py
M	python/sglang/kernels/ops/moe/__init__.py
A	python/sglang/kernels/ops/moe/cutlass_prepare_moe_input.py
M	python/sglang/kernels/ops/moe/dsv4.py
M	python/sglang/kernels/ops/moe/ep_moe_kernels.py
A	python/sglang/kernels/ops/moe/eplb_map_record.py
M	python/sglang/kernels/ops/moe/expert_pack_mxfp4.py
M	python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py
A	python/sglang/kernels/ops/moe/gguf_moe_vec.py
A	python/sglang/kernels/ops/moe/gguf_moe_weights.py
A	python/sglang/kernels/ops/moe/gguf_moe_weights_aot.py
R100	python/sglang/kernels/ops/moe/_jit_situ_and_mul_quant.py	python/sglang/kernels/ops/moe/kimi_k3_situ_and_mul_quant.py
A	python/sglang/kernels/ops/moe/mega_gate.py
R100	python/sglang/kernels/ops/quantization/minimax_quant_ue8m0.py	python/sglang/kernels/ops/moe/minimax_quant_ue8m0.py
M	python/sglang/kernels/ops/moe/moe_align.py
M	python/sglang/kernels/ops/moe/moe_align_single_token.py
M	python/sglang/kernels/ops/moe/moe_align_small_numel.py
M	python/sglang/kernels/ops/moe/moe_fused_gate.py
M	python/sglang/kernels/ops/moe/moe_fused_mul_sum.py
M	python/sglang/kernels/ops/moe/moe_wna16_marlin.py
R081	python/sglang/kernels/ops/muse_spark/fused_post_expert_norm_weighted_sum_triton.py	python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py
R098	python/sglang/kernels/ops/muse_spark/fused_muse_spark_routing.py	python/sglang/kernels/ops/moe/muse_spark_routing.py
R096	python/sglang/kernels/ops/muse_spark/routing.py	python/sglang/kernels/ops/moe/muse_spark_routing_jit.py
M	python/sglang/kernels/ops/moe/mxfp8_moe_amd_gfx95.py
A	python/sglang/kernels/ops/moe/paged_experts.py
M	python/sglang/kernels/ops/moe/router.py
A	python/sglang/kernels/ops/moe/shuffle_weighted_sum.py
M	python/sglang/kernels/ops/moe/smallm_moe_gfx950/__init__.py
M	python/sglang/kernels/ops/moe/smallm_router_gfx950/__init__.py
D	python/sglang/kernels/ops/moe/triton_sigmoid_gate_mul.py
D	python/sglang/kernels/ops/muse_spark/__init__.py
M	python/sglang/kernels/ops/quantization/__init__.py
A	python/sglang/kernels/ops/quantization/act_quant_tilelang.py
A	python/sglang/kernels/ops/quantization/act_quant_triton.py
A	python/sglang/kernels/ops/quantization/bf16x3_split.py
R083	python/sglang/kernels/ops/attention/dsa/quant_k_cache.py	python/sglang/kernels/ops/quantization/dsa_quant_k_cache.py
R100	python/sglang/kernels/ops/attention/dsv4/torch_quant.py	python/sglang/kernels/ops/quantization/fp4_fake_quant.py
A	python/sglang/kernels/ops/quantization/fp4_indexer_quant.py
R098	python/sglang/kernels/ops/attention/dsv4/fp8_wo_a.py	python/sglang/kernels/ops/quantization/fp8_group_major_quant_ue8m0.py
M	python/sglang/kernels/ops/quantization/mxfp8_amd_gfx95.py
R052	python/sglang/kernels/ops/layernorm/mxfp8_epilogue.py	python/sglang/kernels/ops/quantization/mxfp8_epilogue.py
M	python/sglang/kernels/ops/quantization/mxfp8_quant.py
M	python/sglang/kernels/ops/quantization/mxfp8_swizzled_triton.py
M	python/sglang/kernels/ops/quantization/per_token_group_quant.py
R100	python/sglang/kernels/ops/moe/rocm_moe_utils.py	python/sglang/kernels/ops/quantization/rocm_upscale.py
M	python/sglang/kernels/ops/sampling/__init__.py
A	python/sglang/kernels/ops/sampling/renorm.py
A	python/sglang/kernels/ops/sampling/softcap_row_argmax.py
M	python/sglang/kernels/ops/speculative/__init__.py
M	python/sglang/kernels/ops/speculative/cache_locs.py
M	python/sglang/kernels/ops/speculative/dflash.py
A	python/sglang/kernels/ops/speculative/dspark/blockwise_rejection.py
M	python/sglang/kernels/ops/speculative/dspark/dspark_accept.py
M	python/sglang/kernels/ops/speculative/dspark/dspark_draft_model.py
M	python/sglang/kernels/ops/speculative/dspark/dspark_verify_window.py
M	python/sglang/kernels/ops/speculative/dspark/fused_kv_write.py
M	python/sglang/kernels/ops/speculative/dspark/simulated_bonus.py
A	python/sglang/kernels/ops/speculative/dspark/stage_commit.py
A	python/sglang/kernels/ops/speculative/dspark/static_accept_tail.py
M	python/sglang/kernels/ops/speculative/multi_layer_eagle.py
A	python/sglang/kernels/ops/speculative/pack_verify_mask.py
M	python/sglang/kernels/ops/speculative/reconstruct_tree.py
A	python/sglang/kernels/ops/speculative/simulated_accept.py
M	python/sglang/kernels/ops/speculative/spec_tree.py
R100	python/sglang/srt/speculative/uno_tree.py	python/sglang/kernels/ops/speculative/uno_tree.py
A	python/sglang/kernels/ops/speculative/verify_sampling.py
A	python/sglang/kernels/plugin_kernels/README.md
A	python/sglang/kernels/plugin_kernels/__init__.py
A	python/sglang/kernels/plugin_kernels/amd/__init__.py
R083	python/sglang/srt/monokernels/spec.py	python/sglang/kernels/plugin_kernels/amd/spec.py
A	python/sglang/kernels/plugin_kernels/nvidia/__init__.py
A	python/sglang/kernels/plugin_kernels/nvidia/spec.py
M	python/sglang/kernels/spec.py
M	python/sglang/multimodal_gen/.agents/skills/sglang-diffusion-benchmark-profile/existing-fast-paths.md
M	python/sglang/multimodal_gen/.agents/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
M	python/sglang/multimodal_gen/.agents/skills/sglang-diffusion-performance/SKILL.md
M	python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/README.md
M	python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/__init__.py
M	python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/core/generator.py
M	python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/core/preflight.py
M	python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/core/server_api.py
M	python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/executors/adapter.py
M	python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/executors/base.py
M	python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/executors/flux.py
M	python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/executors/minimax_h3.py
M	python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/executors/qwen_image.py
M	python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/executors/zimage.py
M	python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/nodes.py
M	python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/options.py
M	python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/test/test_h3_request.py
A	python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/test/test_sgld_server_mode.py
M	python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/utils.py
A	python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/web/worker_status.js
A	python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/workflows/flux_controlnet_sgld.json
A	python/sglang/multimodal_gen/apps/comfyui_worker_patches/__init__.py
A	python/sglang/multimodal_gen/apps/comfyui_worker_patches/h3.py
A	python/sglang/multimodal_gen/apps/comfyui_worker_patches/tests/test_worker_profiles.py
M	python/sglang/multimodal_gen/configs/models/dits/__init__.py
A	python/sglang/multimodal_gen/configs/models/dits/nomas.py
M	python/sglang/multimodal_gen/configs/models/dits/wan_animate_2.py
M	python/sglang/multimodal_gen/configs/models/vaes/base.py
M	python/sglang/multimodal_gen/configs/models/vaes/minimax_h3_video.py
A	python/sglang/multimodal_gen/configs/models/vaes/nomas.py
M	python/sglang/multimodal_gen/configs/pipeline_configs/base.py
M	python/sglang/multimodal_gen/configs/pipeline_configs/flux.py
M	python/sglang/multimodal_gen/configs/pipeline_configs/glm_image.py
M	python/sglang/multimodal_gen/configs/pipeline_configs/krea2.py
M	python/sglang/multimodal_gen/configs/pipeline_configs/lingbot_world.py
M	python/sglang/multimodal_gen/configs/pipeline_configs/ltx_2.py
M	python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py
M	python/sglang/multimodal_gen/configs/pipeline_configs/model_deployment_config.py
M	python/sglang/multimodal_gen/configs/pipeline_configs/moss_image.py
A	python/sglang/multimodal_gen/configs/pipeline_configs/nomas.py
M	python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py
M	python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image21.py
M	python/sglang/multimodal_gen/configs/pipeline_configs/wan.py
M	python/sglang/multimodal_gen/configs/sample/minimax_h3.py
A	python/sglang/multimodal_gen/configs/sample/nomas.py
M	python/sglang/multimodal_gen/configs/sample/sampling_params.py
M	python/sglang/multimodal_gen/configs/sample/wan.py
A	python/sglang/multimodal_gen/csrc/attn/vmoba_attn/tests/test_vmoba_selection.py
A	python/sglang/multimodal_gen/csrc/attn/vmoba_attn/tests/test_vmoba_selection_gpu.py
M	python/sglang/multimodal_gen/csrc/attn/vmoba_attn/vmoba/vmoba.py
M	python/sglang/multimodal_gen/registry.py
M	python/sglang/multimodal_gen/runtime/breakable_cuda_graph/runner.py
M	python/sglang/multimodal_gen/runtime/disaggregation/orchestrator.py
M	python/sglang/multimodal_gen/runtime/disaggregation/request_state.py
M	python/sglang/multimodal_gen/runtime/disaggregation/roles.py
M	python/sglang/multimodal_gen/runtime/disaggregation/scheduler_mixin.py
M	python/sglang/multimodal_gen/runtime/distributed/device_communicators/ipc_a2a.py
M	python/sglang/multimodal_gen/runtime/distributed/device_communicators/ipc_a2a_multi.py
M	python/sglang/multimodal_gen/runtime/distributed/group_coordinator.py
M	python/sglang/multimodal_gen/runtime/distributed/parallel_state.py
M	python/sglang/multimodal_gen/runtime/entrypoints/control_requests.py
M	python/sglang/multimodal_gen/runtime/entrypoints/http_server.py
M	python/sglang/multimodal_gen/runtime/entrypoints/openai/image_api.py
M	python/sglang/multimodal_gen/runtime/entrypoints/openai/mesh_api.py
M	python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py
A	python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/adapters/nomas_realtime_adapter.py
M	python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/generate_session.py
M	python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/realtime_video_api.py
M	python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/registry.py
M	python/sglang/multimodal_gen/runtime/entrypoints/openai/storage.py
M	python/sglang/multimodal_gen/runtime/entrypoints/openai/stores.py
M	python/sglang/multimodal_gen/runtime/entrypoints/openai/utils.py
M	python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py
A	python/sglang/multimodal_gen/runtime/entrypoints/openai/video_progress.py
M	python/sglang/multimodal_gen/runtime/entrypoints/utils.py
M	python/sglang/multimodal_gen/runtime/ipc_array.py
M	python/sglang/multimodal_gen/runtime/launch_server.py
M	python/sglang/multimodal_gen/runtime/layers/attention/backends/aiter.py
M	python/sglang/multimodal_gen/runtime/layers/attention/backends/attention_backend.py
M	python/sglang/multimodal_gen/runtime/layers/attention/backends/flash_attn.py
M	python/sglang/multimodal_gen/runtime/layers/attention/backends/sdpa.py
M	python/sglang/multimodal_gen/runtime/layers/attention/backends/video_sparse_attn.py
M	python/sglang/multimodal_gen/runtime/layers/attention/backends/video_sparse_attn_h3.py
M	python/sglang/multimodal_gen/runtime/layers/attention/backends/xpu_backend.py
M	python/sglang/multimodal_gen/runtime/layers/attention/layer.py
M	python/sglang/multimodal_gen/runtime/layers/layernorm.py
M	python/sglang/multimodal_gen/runtime/layers/linear.py
M	python/sglang/multimodal_gen/runtime/layers/lora/linear.py
M	python/sglang/multimodal_gen/runtime/layers/mlp.py
M	python/sglang/multimodal_gen/runtime/layers/parallel_conv.py
M	python/sglang/multimodal_gen/runtime/layers/quantization/__init__.py
M	python/sglang/multimodal_gen/runtime/layers/quantization/convrot_int8_comfy_kitchen.py
M	python/sglang/multimodal_gen/runtime/layers/quantization/fp8.py
D	python/sglang/multimodal_gen/runtime/layers/quantization/method_names.py
M	python/sglang/multimodal_gen/runtime/layers/quantization/modelopt_quant.py
M	python/sglang/multimodal_gen/runtime/layers/rotary_embedding/base.py
M	python/sglang/multimodal_gen/runtime/layers/rotary_embedding/mrope.py
M	python/sglang/multimodal_gen/runtime/layers/rotary_embedding/utils.py
M	python/sglang/multimodal_gen/runtime/layers/visual_embedding.py
M	python/sglang/multimodal_gen/runtime/loader/comfyui_checkpoints/flux.py
M	python/sglang/multimodal_gen/runtime/loader/comfyui_checkpoints/spec.py
M	python/sglang/multimodal_gen/runtime/loader/comfyui_checkpoints/zimage.py
M	python/sglang/multimodal_gen/runtime/loader/component_loaders/text_encoder_loader.py
M	python/sglang/multimodal_gen/runtime/loader/component_loaders/vae_loader.py
M	python/sglang/multimodal_gen/runtime/loader/fsdp_load.py
M	python/sglang/multimodal_gen/runtime/loader/rank_local_checkpoint.py
M	python/sglang/multimodal_gen/runtime/loader/utils.py
M	python/sglang/multimodal_gen/runtime/loader/weight_utils.py
M	python/sglang/multimodal_gen/runtime/managers/forward_context.py
M	python/sglang/multimodal_gen/runtime/managers/gpu_worker.py
A	python/sglang/multimodal_gen/runtime/managers/progress_reporter.py
A	python/sglang/multimodal_gen/runtime/managers/request_gc.py
M	python/sglang/multimodal_gen/runtime/managers/scheduler.py
M	python/sglang/multimodal_gen/runtime/managers/worker_bootstrap.py
M	python/sglang/multimodal_gen/runtime/models/adapter/ltx_2_connector.py
M	python/sglang/multimodal_gen/runtime/models/bridges/mova_dual_tower.py
M	python/sglang/multimodal_gen/runtime/models/dits/causal_wanvideo.py
M	python/sglang/multimodal_gen/runtime/models/dits/cosmos3_sim_auto.py
M	python/sglang/multimodal_gen/runtime/models/dits/cosmos_dreams.py
M	python/sglang/multimodal_gen/runtime/models/dits/flux.py
M	python/sglang/multimodal_gen/runtime/models/dits/flux3.py
M	python/sglang/multimodal_gen/runtime/models/dits/flux_2.py
M	python/sglang/multimodal_gen/runtime/models/dits/glm_image.py
M	python/sglang/multimodal_gen/runtime/models/dits/helios.py
M	python/sglang/multimodal_gen/runtime/models/dits/hunyuan3d.py
M	python/sglang/multimodal_gen/runtime/models/dits/hunyuan_image3.py
M	python/sglang/multimodal_gen/runtime/models/dits/hunyuan_image3_utils.py
M	python/sglang/multimodal_gen/runtime/models/dits/hunyuanvideo.py
M	python/sglang/multimodal_gen/runtime/models/dits/ideogram.py
M	python/sglang/multimodal_gen/runtime/models/dits/longcat_image.py
M	python/sglang/multimodal_gen/runtime/models/dits/ltx_2.py
M	python/sglang/multimodal_gen/runtime/models/dits/ming_image.py
M	python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py
M	python/sglang/multimodal_gen/runtime/models/dits/minimax_h3_adaln_cache.py
M	python/sglang/multimodal_gen/runtime/models/dits/minimax_h3_vdn_attention.py
M	python/sglang/multimodal_gen/runtime/models/dits/moss_image.py
A	python/sglang/multimodal_gen/runtime/models/dits/nomas.py
A	python/sglang/multimodal_gen/runtime/models/dits/nomas_kv_cache.py
A	python/sglang/multimodal_gen/runtime/models/dits/nomas_layers.py
M	python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py
M	python/sglang/multimodal_gen/runtime/models/dits/qwen_image21.py
M	python/sglang/multimodal_gen/runtime/models/dits/sana.py
M	python/sglang/multimodal_gen/runtime/models/dits/sana_video2.py
M	python/sglang/multimodal_gen/runtime/models/dits/sana_wm.py
M	python/sglang/multimodal_gen/runtime/models/dits/sana_wm_components.py
M	python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py
M	python/sglang/multimodal_gen/runtime/models/dits/zimage.py
M	python/sglang/multimodal_gen/runtime/models/registry.py
M	python/sglang/multimodal_gen/runtime/models/schedulers/kandinsky6_piflow.py
M	python/sglang/multimodal_gen/runtime/models/schedulers/scheduling_flow_match_euler_discrete.py
M	python/sglang/multimodal_gen/runtime/models/schedulers/scheduling_helios.py
M	python/sglang/multimodal_gen/runtime/models/sensenova_u1/neo_unify/modeling_fm_modules.py
M	python/sglang/multimodal_gen/runtime/models/sensenova_u1/neo_unify/modeling_neo_chat.py
M	python/sglang/multimodal_gen/runtime/models/vaes/common.py
M	python/sglang/multimodal_gen/runtime/models/vaes/flux3_video_vae.py
M	python/sglang/multimodal_gen/runtime/models/vaes/ltx_2_vae.py
M	python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3.py
M	python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_vae_cuda_opt.py
A	python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/batching.py
M	python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/klvae.py
M	python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/processor.py
A	python/sglang/multimodal_gen/runtime/models/vaes/nomas.py
M	python/sglang/multimodal_gen/runtime/models/vaes/wan_vae_cuda_opt.py
M	python/sglang/multimodal_gen/runtime/models/vaes/wanvae.py
M	python/sglang/multimodal_gen/runtime/models/vocoder/ltx_2_vocoder.py
M	python/sglang/multimodal_gen/runtime/pipelines/diffusers_pipeline.py
M	python/sglang/multimodal_gen/runtime/pipelines/ernie_image.py
M	python/sglang/multimodal_gen/runtime/pipelines/flux3_image.py
M	python/sglang/multimodal_gen/runtime/pipelines/flux3_image_weights.py
M	python/sglang/multimodal_gen/runtime/pipelines/hunyuan3d_pipeline.py
M	python/sglang/multimodal_gen/runtime/pipelines/hunyuan_image3.py
M	python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py
M	python/sglang/multimodal_gen/runtime/pipelines/minimax_h3_pipeline.py
A	python/sglang/multimodal_gen/runtime/pipelines/nomas_pipeline.py
M	python/sglang/multimodal_gen/runtime/pipelines/sana_video.py
M	python/sglang/multimodal_gen/runtime/pipelines/wan_animate_2_pipeline.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/comfyui_mode.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/composed_pipeline_base.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/executors/parallel_executor.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/executors/pipeline_executor.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/lora/format_adapter.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/lora/peft_adapter.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/lora/pipeline.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/request_utils.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/schedule_batch.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/base.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_dmd.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/image_encoding.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/input_validation.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/cosmos3.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/cosmos3_simulation/dreams.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/cosmos3_simulation/sim_auto.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/cosmos3_simulation/transfer.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/flux3_action.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/flux3_image.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/helios_denoising.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/hunyuan_image3/ar_stage.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/ideogram.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/joy_echo/denoising.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/llada_image/conditioning.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/longlive2.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/ltx_2/denoising.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/ltx_2/denoising_av.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/ltx_2/text_connector.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/release_metadata.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/comfyui_step.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/timestep_preparation.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/moss_image.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/mova.py
A	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/nomas/__init__.py
A	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/nomas/denoise_loop.py
A	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/nomas/request_validation.py
A	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/nomas/stages.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/qwen_image21.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/qwen_image_layered.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/sana_wm/streaming.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/sana_wm/streaming_refiner.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/wan_ti2v.py
M	python/sglang/multimodal_gen/runtime/pipelines_core/stages/progressive_resolution/ideogram.py
M	python/sglang/multimodal_gen/runtime/platforms/aiter.py
M	python/sglang/multimodal_gen/runtime/platforms/cpu.py
M	python/sglang/multimodal_gen/runtime/platforms/cuda.py
M	python/sglang/multimodal_gen/runtime/platforms/interface.py
M	python/sglang/multimodal_gen/runtime/platforms/rocm.py
A	python/sglang/multimodal_gen/runtime/post_training/flow_rollout_controller.py
M	python/sglang/multimodal_gen/runtime/post_training/rl_dataclasses.py
D	python/sglang/multimodal_gen/runtime/post_training/rollout_denoising_mixin.py
A	python/sglang/multimodal_gen/runtime/post_training/rollout_hooks.py
D	python/sglang/multimodal_gen/runtime/post_training/scheduler_rl_debug_mixin.py
D	python/sglang/multimodal_gen/runtime/post_training/scheduler_rl_mixin.py
A	python/sglang/multimodal_gen/runtime/postprocess/bounded_cache.py
M	python/sglang/multimodal_gen/runtime/postprocess/realesrgan_upscaler.py
M	python/sglang/multimodal_gen/runtime/postprocess/rife_interpolator.py
A	python/sglang/multimodal_gen/runtime/realtime/nomas.py
A	python/sglang/multimodal_gen/runtime/realtime/states/nomas.py
M	python/sglang/multimodal_gen/runtime/scheduler_client.py
M	python/sglang/multimodal_gen/runtime/server_args/auto_tune.py
M	python/sglang/multimodal_gen/runtime/server_args/disagg.py
M	python/sglang/multimodal_gen/runtime/server_args/server_args.py
M	python/sglang/multimodal_gen/runtime/server_warmup.py
M	python/sglang/multimodal_gen/runtime/utils/camera_geometry.py
M	python/sglang/multimodal_gen/runtime/utils/common.py
M	python/sglang/multimodal_gen/runtime/utils/hf_diffusers_utils.py
M	python/sglang/multimodal_gen/runtime/utils/image_io.py
M	python/sglang/multimodal_gen/runtime/utils/logging_utils.py
M	python/sglang/multimodal_gen/runtime/utils/mesh3d_utils.py
M	python/sglang/multimodal_gen/runtime/utils/model_overlay.py
M	python/sglang/multimodal_gen/runtime/utils/nvtx_pytorch_hooks.py
M	python/sglang/multimodal_gen/runtime/utils/perf_logger.py
M	python/sglang/multimodal_gen/runtime/utils/precision.py
M	python/sglang/multimodal_gen/runtime/utils/vision.py
M	python/sglang/multimodal_gen/runtime/warmup_request_builder.py
M	python/sglang/multimodal_gen/runtime/weights/source.py
M	python/sglang/multimodal_gen/test/runner/diffusion_suite_runner.py
M	python/sglang/multimodal_gen/test/runner/pytest_runner.py
M	python/sglang/multimodal_gen/test/server/ascend/test_glm_image_distributed.py
M	python/sglang/multimodal_gen/test/server/gpu_cases.py
M	python/sglang/multimodal_gen/test/server/perf_baselines/b200.json
M	python/sglang/multimodal_gen/test/server/perf_baselines/h100.json
M	python/sglang/multimodal_gen/test/server/test_server_common.py
M	python/sglang/multimodal_gen/test/server/test_server_utils.py
M	python/sglang/multimodal_gen/test/server/testcase_configs.py
M	python/sglang/multimodal_gen/test/single_test_file/test_disagg_server.py
M	python/sglang/multimodal_gen/test/single_test_file/test_ipc_a2a_multi_gpu.py
A	python/sglang/multimodal_gen/test/single_test_file/test_nomas_offline.py
M	python/sglang/multimodal_gen/test/single_test_file/test_usp_pipelined_attention_multi_gpu.py
M	python/sglang/multimodal_gen/test/test_utils.py
M	python/sglang/multimodal_gen/test/unit/conftest.py
A	python/sglang/multimodal_gen/test/unit/host_sync_guard.py
A	python/sglang/multimodal_gen/test/unit/manual/bench_lora_merge_addmm.py
M	python/sglang/multimodal_gen/test/unit/progressive_resolution/test_progressive.py
M	python/sglang/multimodal_gen/test/unit/realtime/test_realtime_runtime.py
M	python/sglang/multimodal_gen/test/unit/sana_wm/test_streaming_forward_long.py
M	python/sglang/multimodal_gen/test/unit/test_aiter_attention_impl.py
M	python/sglang/multimodal_gen/test/unit/test_async_output_save.py
A	python/sglang/multimodal_gen/test/unit/test_attention_bcg_outputs.py
M	python/sglang/multimodal_gen/test/unit/test_auto_residency.py
A	python/sglang/multimodal_gen/test/unit/test_camera_geometry.py
M	python/sglang/multimodal_gen/test/unit/test_cfg_parallel_warmup.py
A	python/sglang/multimodal_gen/test/unit/test_ci_gpu_idle.py
M	python/sglang/multimodal_gen/test/unit/test_comfyui_adapters.py
M	python/sglang/multimodal_gen/test/unit/test_comfyui_checkpoint_spec.py
M	python/sglang/multimodal_gen/test/unit/test_comfyui_generator.py
M	python/sglang/multimodal_gen/test/unit/test_comfyui_h3.py
M	python/sglang/multimodal_gen/test/unit/test_comfyui_lora_node.py
M	python/sglang/multimodal_gen/test/unit/test_comfyui_options.py
M	python/sglang/multimodal_gen/test/unit/test_comfyui_session.py
M	python/sglang/multimodal_gen/test/unit/test_convrot_int8_backends.py
M	python/sglang/multimodal_gen/test/unit/test_cosmos3.py
M	python/sglang/multimodal_gen/test/unit/test_cosmos3_rollout.py
M	python/sglang/multimodal_gen/test/unit/test_cosmos3_sim_auto.py
M	python/sglang/multimodal_gen/test/unit/test_cosmos_dreams.py
M	python/sglang/multimodal_gen/test/unit/test_cuda_attention_backend.py
M	python/sglang/multimodal_gen/test/unit/test_diffusion_bcg_padding.py
M	python/sglang/multimodal_gen/test/unit/test_diffusion_bcg_tp_graph_capture.py
M	python/sglang/multimodal_gen/test/unit/test_disagg_extra_tensors.py
A	python/sglang/multimodal_gen/test/unit/test_disagg_orchestration.py
M	python/sglang/multimodal_gen/test/unit/test_disagg_roles.py
M	python/sglang/multimodal_gen/test/unit/test_dp_routing.py
A	python/sglang/multimodal_gen/test/unit/test_dynamic_cudnn_sdpa_routing.py
A	python/sglang/multimodal_gen/test/unit/test_dynamic_cudnn_sdpa_selection.py
M	python/sglang/multimodal_gen/test/unit/test_flux3_action.py
M	python/sglang/multimodal_gen/test/unit/test_flux3_image_dit.py
M	python/sglang/multimodal_gen/test/unit/test_flux3_image_dit_quant.py
M	python/sglang/multimodal_gen/test/unit/test_flux3_prims_ts_config.py
A	python/sglang/multimodal_gen/test/unit/test_flux_true_cfg_scale.py
M	python/sglang/multimodal_gen/test/unit/test_fsdp_load.py
M	python/sglang/multimodal_gen/test/unit/test_fused_lora_compose.py
A	python/sglang/multimodal_gen/test/unit/test_gfx1250_flydsl_linear.py
M	python/sglang/multimodal_gen/test/unit/test_glm_image_multi_output.py
M	python/sglang/multimodal_gen/test/unit/test_gpu_worker_cpu_threads.py
M	python/sglang/multimodal_gen/test/unit/test_health_warmup_gate.py
M	python/sglang/multimodal_gen/test/unit/test_helios_denoising_profiler.py
M	python/sglang/multimodal_gen/test/unit/test_hf_diffusers_utils.py
A	python/sglang/multimodal_gen/test/unit/test_host_sync.py
M	python/sglang/multimodal_gen/test/unit/test_hunyuan_image3_tp_broadcast.py
M	python/sglang/multimodal_gen/test/unit/test_ideogram4.py
M	python/sglang/multimodal_gen/test/unit/test_idle_cache_release.py
A	python/sglang/multimodal_gen/test/unit/test_image_response_bytes.py
M	python/sglang/multimodal_gen/test/unit/test_input_validation.py
M	python/sglang/multimodal_gen/test/unit/test_ipc_a2a_lifecycle.py
M	python/sglang/multimodal_gen/test/unit/test_ipc_array.py
M	python/sglang/multimodal_gen/test/unit/test_ipc_cuda.py
M	python/sglang/multimodal_gen/test/unit/test_kandinsky6_bf16_dtype.py
M	python/sglang/multimodal_gen/test/unit/test_kandinsky6_layerwise_offload.py
M	python/sglang/multimodal_gen/test/unit/test_krea2_fp8.py
M	python/sglang/multimodal_gen/test/unit/test_launch_server_shutdown.py
M	python/sglang/multimodal_gen/test/unit/test_load_inclusive_e2e.py
M	python/sglang/multimodal_gen/test/unit/test_logging_utils.py
M	python/sglang/multimodal_gen/test/unit/test_lora_inference_mode.py
M	python/sglang/multimodal_gen/test/unit/test_lora_peft.py
M	python/sglang/multimodal_gen/test/unit/test_lora_pipeline.py
A	python/sglang/multimodal_gen/test/unit/test_ltx23_per_sample_adaln.py
A	python/sglang/multimodal_gen/test/unit/test_ltx2_ada_values_layout.py
M	python/sglang/multimodal_gen/test/unit/test_ltx2_bcg_coords.py
A	python/sglang/multimodal_gen/test/unit/test_ltx2_premerged_stage2.py
A	python/sglang/multimodal_gen/test/unit/test_ltx2_prompt_kv_cache.py
A	python/sglang/multimodal_gen/test/unit/test_ltx2_prompt_mask.py
A	python/sglang/multimodal_gen/test/unit/test_ltx2_prompt_mask_registers.py
A	python/sglang/multimodal_gen/test/unit/test_ltx_vae_parallel_decode.py
M	python/sglang/multimodal_gen/test/unit/test_ming_image.py
M	python/sglang/multimodal_gen/test/unit/test_minimax_h3_adaln_cache.py
M	python/sglang/multimodal_gen/test/unit/test_minimax_h3_admission.py
M	python/sglang/multimodal_gen/test/unit/test_minimax_h3_pdd_offline.py
A	python/sglang/multimodal_gen/test/unit/test_minimax_h3_vae_fast_path.py
M	python/sglang/multimodal_gen/test/unit/test_minimax_h3_vdn.py
M	python/sglang/multimodal_gen/test/unit/test_model_overlay.py
M	python/sglang/multimodal_gen/test/unit/test_moss_image.py
M	python/sglang/multimodal_gen/test/unit/test_moss_image_parallel.py
A	python/sglang/multimodal_gen/test/unit/test_mova_denoising_loop.py
M	python/sglang/multimodal_gen/test/unit/test_multiple_task_types.py
A	python/sglang/multimodal_gen/test/unit/test_nomas.py
A	python/sglang/multimodal_gen/test/unit/test_nomas_parallel.py
A	python/sglang/multimodal_gen/test/unit/test_nomas_realtime.py
M	python/sglang/multimodal_gen/test/unit/test_nvtx_pytorch_hooks.py
M	python/sglang/multimodal_gen/test/unit/test_openai_image_api.py
M	python/sglang/multimodal_gen/test/unit/test_openai_security.py
M	python/sglang/multimodal_gen/test/unit/test_openai_utils.py
M	python/sglang/multimodal_gen/test/unit/test_output_saving.py
M	python/sglang/multimodal_gen/test/unit/test_parallel_linear_weight_loading.py
A	python/sglang/multimodal_gen/test/unit/test_patch_embed_tokens.py
M	python/sglang/multimodal_gen/test/unit/test_perf_dump_write.py
M	python/sglang/multimodal_gen/test/unit/test_performance_metrics.py
A	python/sglang/multimodal_gen/test/unit/test_pipeline_regressions.py
M	python/sglang/multimodal_gen/test/unit/test_plain_state_dict_loaders.py
M	python/sglang/multimodal_gen/test/unit/test_precision_consistency.py
M	python/sglang/multimodal_gen/test/unit/test_prompt_enhancement.py
A	python/sglang/multimodal_gen/test/unit/test_quality_fusion_lora.py
M	python/sglang/multimodal_gen/test/unit/test_qwen3_5_encoder.py
M	python/sglang/multimodal_gen/test/unit/test_qwen_image21.py
M	python/sglang/multimodal_gen/test/unit/test_qwen_image21_cuda.py
M	python/sglang/multimodal_gen/test/unit/test_qwen_image_layered.py
M	python/sglang/multimodal_gen/test/unit/test_qwen_image_rope.py
A	python/sglang/multimodal_gen/test/unit/test_registry.py
M	python/sglang/multimodal_gen/test/unit/test_residual_gate_add2.py
A	python/sglang/multimodal_gen/test/unit/test_rocm_conv3d_patch.py
M	python/sglang/multimodal_gen/test/unit/test_rowwise_fp8.py
A	python/sglang/multimodal_gen/test/unit/test_runtime_utils_regressions.py
M	python/sglang/multimodal_gen/test/unit/test_sampling_params.py
M	python/sglang/multimodal_gen/test/unit/test_sana_video.py
M	python/sglang/multimodal_gen/test/unit/test_sana_video2.py
M	python/sglang/multimodal_gen/test/unit/test_scheduler_client.py
M	python/sglang/multimodal_gen/test/unit/test_scheduler_rollout_unit.py
M	python/sglang/multimodal_gen/test/unit/test_sensenova_u1.py
A	python/sglang/multimodal_gen/test/unit/test_sequence_parallel_over_cfg_ranks.py
M	python/sglang/multimodal_gen/test/unit/test_sequential_server_requests.py
M	python/sglang/multimodal_gen/test/unit/test_server_args.py
M	python/sglang/multimodal_gen/test/unit/test_storage.py
M	python/sglang/multimodal_gen/test/unit/test_suite_partitioning.py
M	python/sglang/multimodal_gen/test/unit/test_text_encoder_loader.py
M	python/sglang/multimodal_gen/test/unit/test_usp_ring_replicated.py
M	python/sglang/multimodal_gen/test/unit/test_vae_loader.py
M	python/sglang/multimodal_gen/test/unit/test_vae_spatial_parallel_decode.py
M	python/sglang/multimodal_gen/test/unit/test_vae_tiled_decode_groups.py
M	python/sglang/multimodal_gen/test/unit/test_video_job_lifecycle.py
A	python/sglang/multimodal_gen/test/unit/test_video_models_host_sync.py
A	python/sglang/multimodal_gen/test/unit/test_vsa_h3_ref2va_regions.py
M	python/sglang/multimodal_gen/test/unit/test_wan_animate_2_config.py
M	python/sglang/multimodal_gen/test/unit/test_wan_temb_table_slices.py
M	python/sglang/multimodal_gen/test/unit/test_wan_ti2v_helpers.py
A	python/sglang/multimodal_gen/test/unit/test_wan_vae_attention_rows.py
A	python/sglang/multimodal_gen/test/unit/test_wan_vae_dupup.py
M	python/sglang/multimodal_gen/test/unit/test_weight_utils.py
A	python/sglang/multimodal_gen/test/unit/test_zimage_rope_cache.py
M	python/sglang/srt/arg_groups/argparse_actions.py
M	python/sglang/srt/arg_groups/attention_hook.py
M	python/sglang/srt/arg_groups/choices.py
M	python/sglang/srt/arg_groups/cp_tp_group_sharing_hook.py
M	python/sglang/srt/arg_groups/deepseek_v4_hook.py
A	python/sglang/srt/arg_groups/engram_config.py
M	python/sglang/srt/arg_groups/field_order.py
M	python/sglang/srt/arg_groups/fields/disagg.py
M	python/sglang/srt/arg_groups/fields/exec_.py
M	python/sglang/srt/arg_groups/fields/lora.py
M	python/sglang/srt/arg_groups/fields/memory.py
M	python/sglang/srt/arg_groups/fields/model.py
M	python/sglang/srt/arg_groups/fields/schedule.py
M	python/sglang/srt/arg_groups/fields/serving.py
M	python/sglang/srt/arg_groups/fields/spec.py
M	python/sglang/srt/arg_groups/hicache_hook.py
M	python/sglang/srt/arg_groups/hisparse_hook.py
M	python/sglang/srt/arg_groups/kv_cache_hook.py
M	python/sglang/srt/arg_groups/kv_shard_hook.py
M	python/sglang/srt/arg_groups/lora_hook.py
M	python/sglang/srt/arg_groups/mamba_hook.py
M	python/sglang/srt/arg_groups/mega_moe_hook.py
M	python/sglang/srt/arg_groups/memory_hook.py
M	python/sglang/srt/arg_groups/model_hook.py
M	python/sglang/srt/arg_groups/model_override_base.py
M	python/sglang/srt/arg_groups/model_overrides/__init__.py
A	python/sglang/srt/arg_groups/model_overrides/beam.py
M	python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py
A	python/sglang/srt/arg_groups/model_overrides/flint.py
M	python/sglang/srt/arg_groups/model_overrides/kimi_k3.py
M	python/sglang/srt/arg_groups/model_overrides/minimax_m3.py
M	python/sglang/srt/arg_groups/moe_hook.py
M	python/sglang/srt/arg_groups/overrides.py
M	python/sglang/srt/arg_groups/parallel_hook.py
M	python/sglang/srt/arg_groups/pd_disaggregation_hook.py
M	python/sglang/srt/arg_groups/pipeline.py
M	python/sglang/srt/arg_groups/resolution_hooks.py
M	python/sglang/srt/arg_groups/serving_hook.py
M	python/sglang/srt/arg_groups/speculative_hook.py
M	python/sglang/srt/arg_groups/validation_hook.py
M	python/sglang/srt/batch_invariant_ops/__init__.py
M	python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
M	python/sglang/srt/batch_overlap/operations.py
M	python/sglang/srt/batch_overlap/two_batch_overlap.py
M	python/sglang/srt/beam_search/coordinator.py
M	python/sglang/srt/beam_search/fork.py
M	python/sglang/srt/checkpoint_engine/update.py
M	python/sglang/srt/configs/__init__.py
A	python/sglang/srt/configs/beam.py
A	python/sglang/srt/configs/draft_parallel.py
A	python/sglang/srt/configs/flint.py
M	python/sglang/srt/configs/inkling.py
M	python/sglang/srt/configs/linear_attn_model_registry.py
M	python/sglang/srt/configs/minimax_vl.py
M	python/sglang/srt/configs/model_config.py
M	python/sglang/srt/configs/muse_glimmer.py
M	python/sglang/srt/configs/nano_nemotron_vl.py
A	python/sglang/srt/configs/shared_experts.py
M	python/sglang/srt/configs/update_config.py
M	python/sglang/srt/constrained/grammar_manager.py
M	python/sglang/srt/constrained/llguidance_backend.py
M	python/sglang/srt/constrained/reasoner_grammar_backend.py
A	python/sglang/srt/debug_utils/rocprofiler_register.py
M	python/sglang/srt/disaggregation/base/conn.py
M	python/sglang/srt/disaggregation/checksum.py
M	python/sglang/srt/disaggregation/common/conn.py
M	python/sglang/srt/disaggregation/common/dcp_pack.py
M	python/sglang/srt/disaggregation/common/staging_buffer.py
M	python/sglang/srt/disaggregation/common/staging_handler.py
M	python/sglang/srt/disaggregation/common/utils.py
M	python/sglang/srt/disaggregation/decode.py
M	python/sglang/srt/disaggregation/decode_hicache_mixin.py
M	python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py
M	python/sglang/srt/disaggregation/encoder/receiver.py
A	python/sglang/srt/disaggregation/minimax.py
M	python/sglang/srt/disaggregation/mooncake/conn.py
M	python/sglang/srt/disaggregation/mori/conn.py
M	python/sglang/srt/disaggregation/nixl/conn.py
M	python/sglang/srt/disaggregation/prefill.py
M	python/sglang/srt/disaggregation/role_switch.py
M	python/sglang/srt/disaggregation/utils.py
M	python/sglang/srt/distributed/bootstrap.py
R100	python/sglang/kernels/ops/communication/mp.py	python/sglang/srt/distributed/comm_cleanup.py
M	python/sglang/srt/distributed/device_communicators/configs/custom_all_reduce_v2.py
M	python/sglang/srt/distributed/device_communicators/custom_all_reduce.py
M	python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py
D	python/sglang/srt/distributed/device_communicators/npu_communicator.py
M	python/sglang/srt/distributed/device_communicators/pynccl.py
M	python/sglang/srt/distributed/device_communicators/pynccl_allocator.py
M	python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py
M	python/sglang/srt/distributed/parallel_state.py
M	python/sglang/srt/dllm/algorithm/base.py
M	python/sglang/srt/dllm/algorithm/joint_threshold.py
M	python/sglang/srt/dllm/algorithm/joint_threshold_indel.py
M	python/sglang/srt/dllm/algorithm/low_confidence.py
M	python/sglang/srt/dllm/attention.py
M	python/sglang/srt/dllm/mixin/req.py
M	python/sglang/srt/entrypoints/anthropic/protocol.py
M	python/sglang/srt/entrypoints/anthropic/serving.py
M	python/sglang/srt/entrypoints/engine.py
M	python/sglang/srt/entrypoints/grpc_bridge.py
M	python/sglang/srt/entrypoints/http_server.py
M	python/sglang/srt/entrypoints/ollama/serving.py
M	python/sglang/srt/entrypoints/openai/chat_encoding.py
A	python/sglang/srt/entrypoints/openai/chat_encoding_adapters.py
D	python/sglang/srt/entrypoints/openai/encoding_dsv4.py
D	python/sglang/srt/entrypoints/openai/encoding_dsv41.py
M	python/sglang/srt/entrypoints/openai/parallel_prompt_encode.py
M	python/sglang/srt/entrypoints/openai/protocol.py
M	python/sglang/srt/entrypoints/openai/responses_adapters.py
A	python/sglang/srt/entrypoints/openai/responses_tools/__init__.py
A	python/sglang/srt/entrypoints/openai/responses_tools/beam.py
M	python/sglang/srt/entrypoints/openai/serving_base.py
M	python/sglang/srt/entrypoints/openai/serving_chat.py
M	python/sglang/srt/entrypoints/openai/serving_decisions.py
A	python/sglang/srt/entrypoints/openai/serving_deepseek_recipe.py
A	python/sglang/srt/entrypoints/openai/serving_inkling.py
M	python/sglang/srt/entrypoints/openai/serving_responses.py
M	python/sglang/srt/entrypoints/openai/serving_tokenize.py
M	python/sglang/srt/entrypoints/openai/usage_processor.py
M	python/sglang/srt/entrypoints/sidecar.py
A	python/sglang/srt/entrypoints/systemone/kev.py
M	python/sglang/srt/entrypoints/systemone/serving.py
M	python/sglang/srt/environ.py
D	python/sglang/srt/eplb/eplb_map_record_fused.py
M	python/sglang/srt/eplb/expert_distribution.py
M	python/sglang/srt/eplb/expert_location.py
M	python/sglang/srt/eplb/expert_location_dispatch.py
M	python/sglang/srt/function_call/base_format_detector.py
A	python/sglang/srt/function_call/beam_detector.py
M	python/sglang/srt/function_call/deepseekv41_detector.py
M	python/sglang/srt/function_call/function_call_parser.py
M	python/sglang/srt/function_call/glm47_moe_detector.py
M	python/sglang/srt/function_call/glm4_moe_detector.py
M	python/sglang/srt/function_call/json_array_parser.py
M	python/sglang/srt/function_call/minimax_m3.py
M	python/sglang/srt/function_call/parser_names.py
M	python/sglang/srt/function_call/qwen3_coder_detector.py
A	python/sglang/srt/function_call/response_template_structural_tag.py
M	python/sglang/srt/hardware_backend/mlx/kv_cache/auxiliary_state.py
M	python/sglang/srt/hardware_backend/mlx/model_runner.py
M	python/sglang/srt/hardware_backend/mlx/model_runner_stub.py
M	python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py
M	python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py
M	python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py
M	python/sglang/srt/hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py
M	python/sglang/srt/hardware_backend/npu/attention/ascend_kda_backend.py
M	python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py
M	python/sglang/srt/hardware_backend/npu/dsv4/c128_sidecar_component.py
M	python/sglang/srt/hardware_backend/npu/dsv4/dsv4_allocator.py
M	python/sglang/srt/hardware_backend/npu/dsv4/dsv4_common_hooks.py
M	python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py
M	python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py
M	python/sglang/srt/hardware_backend/npu/memory_pool_npu.py
M	python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py
D	python/sglang/srt/hardware_backend/npu/moe/finalize_routing.py
D	python/sglang/srt/hardware_backend/npu/moe/init_routing.py
A	python/sglang/srt/hardware_backend/npu/moe/routing.py
A	python/sglang/srt/hardware_backend/npu/moe/standard_dispatch.py
M	python/sglang/srt/hardware_backend/npu/moe/topk.py
M	python/sglang/srt/hardware_backend/npu/quantization/fp4_moe_methods.py
M	python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py
M	python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py
M	python/sglang/srt/hardware_backend/npu/quantization/w8a8_mxfp8.py
M	python/sglang/srt/hardware_backend/npu/utils.py
M	python/sglang/srt/hardware_backend/xpu/attention/xpu_gdn_backend.py
M	python/sglang/srt/hardware_backend/xpu/kernels/fla/fused_sigmoid_gating_recurrent.py
A	python/sglang/srt/kernel_plugins/__init__.py
R100	python/sglang/srt/monokernels/__init__.py	python/sglang/srt/kernel_plugins/amd/__init__.py
R058	python/sglang/srt/monokernels/runtime.py	python/sglang/srt/kernel_plugins/amd/binding.py
A	python/sglang/srt/kernel_plugins/nvidia/__init__.py
A	python/sglang/srt/kernel_plugins/nvidia/binding.py
A	python/sglang/srt/kernel_plugins/nvidia/eagle_binding.py
M	python/sglang/srt/kv_canary/api.py
M	python/sglang/srt/layers/activation.py
M	python/sglang/srt/layers/attention/aiter_backend.py
M	python/sglang/srt/layers/attention/aiter_utils.py
M	python/sglang/srt/layers/attention/attention_registry.py
M	python/sglang/srt/layers/attention/base_attn_backend.py
M	python/sglang/srt/layers/attention/cutedsl_mla_backend.py
M	python/sglang/srt/layers/attention/deepseek_v4_backend.py
M	python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
M	python/sglang/srt/layers/attention/deepseek_v4_mega_backend.py
M	python/sglang/srt/layers/attention/dots_hybrid_backend.py
M	python/sglang/srt/layers/attention/dsa/dsa_backend_mtp_precompute.py
M	python/sglang/srt/layers/attention/dsa/dsa_dcp.py
M	python/sglang/srt/layers/attention/dsa/dsa_indexer.py
M	python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py
M	python/sglang/srt/layers/attention/dsa/dsa_indexer_metadata.py
M	python/sglang/srt/layers/attention/dsa/dsa_metadata_manager.py
M	python/sglang/srt/layers/attention/dsa/dsa_npu_indexer.py
M	python/sglang/srt/layers/attention/dsa/dsa_prefill_cuda_graph.py
R100	python/sglang/kernels/ops/attention/dsa/hip_gfx950/fused_decode.py	python/sglang/srt/layers/attention/dsa/gfx950_fused_indexer.py
M	python/sglang/srt/layers/attention/dsa/kpool_prefill_cuda_graph.py
M	python/sglang/srt/layers/attention/dsa/utils.py
M	python/sglang/srt/layers/attention/dsa_backend.py
M	python/sglang/srt/layers/attention/dsv4/compressor_trtllm.py
M	python/sglang/srt/layers/attention/dsv4/compressor_v2.py
M	python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py
M	python/sglang/srt/layers/attention/dsv4/low_ratio_backend_hip.py
M	python/sglang/srt/layers/attention/dsv4/metadata.py
A	python/sglang/srt/layers/attention/dsv4/online_c128_mtp.py
M	python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py
R100	python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/env_gate.py	python/sglang/srt/layers/attention/dsv4/unified_kv_gate.py
M	python/sglang/srt/layers/attention/dsv4/v41_indexer/__init__.py
M	python/sglang/srt/layers/attention/dsv4/v41_indexer/dense_blocks.py
M	python/sglang/srt/layers/attention/dsv4/v41_indexer/full_topk.py
M	python/sglang/srt/layers/attention/dsv4/v41_indexer/scoring.py
M	python/sglang/srt/layers/attention/dsv4/v41_indexer/sparse_table.py
M	python/sglang/srt/layers/attention/dsv4/v41_indexer/types.py
M	python/sglang/srt/layers/attention/flashattention_backend.py
M	python/sglang/srt/layers/attention/flashinfer_backend.py
M	python/sglang/srt/layers/attention/flashinfer_mla_backend.py
M	python/sglang/srt/layers/attention/flashmla_backend.py
M	python/sglang/srt/layers/attention/graph_utils.py
M	python/sglang/srt/layers/attention/graph_variants.py
M	python/sglang/srt/layers/attention/hip_flash_mla.py
M	python/sglang/srt/layers/attention/hpc_ops_backend.py
M	python/sglang/srt/layers/attention/hybrid_attn_backend.py
M	python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
M	python/sglang/srt/layers/attention/intel_amx_backend.py
M	python/sglang/srt/layers/attention/linear/gdn_backend.py
A	python/sglang/srt/layers/attention/linear/gdn_prefill_graph.py
M	python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py
M	python/sglang/srt/layers/attention/linear/kda_backend.py
A	python/sglang/srt/layers/attention/linear/kda_flashinfer_prefill_graph.py
M	python/sglang/srt/layers/attention/linear/kda_prefill_graph.py
M	python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl.py
M	python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py
M	python/sglang/srt/layers/attention/linear/kernels/kda_cutedsl.py
M	python/sglang/srt/layers/attention/linear/kernels/kda_flashinfer.py
M	python/sglang/srt/layers/attention/linear/kernels/kda_flashinfer_prefill.py
M	python/sglang/srt/layers/attention/linear/kernels/kda_triton.py
M	python/sglang/srt/layers/attention/linear/lightning_backend.py
M	python/sglang/srt/layers/attention/linear/short_conv_backend.py
M	python/sglang/srt/layers/attention/mamba/mamba2_metadata.py
M	python/sglang/srt/layers/attention/minicpm/attention_adapter.py
M	python/sglang/srt/layers/attention/minicpm/backend.py
M	python/sglang/srt/layers/attention/minicpm/sparse_utils.py
M	python/sglang/srt/layers/attention/minimax_sparse_backend.py
A	python/sglang/srt/layers/attention/minimax_sparse_nvfp4_backend.py
M	python/sglang/srt/layers/attention/minimax_sparse_ops/indexer_cp.py
M	python/sglang/srt/layers/attention/minimax_sparse_ops/minimax_sparse.py
A	python/sglang/srt/layers/attention/minimax_sparse_ops/msa_plan.py
D	python/sglang/srt/layers/attention/minimax_sparse_ops/naive/flash_with_topk_idx.py
D	python/sglang/srt/layers/attention/minimax_sparse_ops/naive/topk_sparse.py
D	python/sglang/srt/layers/attention/nsa/index_buf_accessor.py
D	python/sglang/srt/layers/attention/nsa/quant_k_cache.py
D	python/sglang/srt/layers/attention/nsa/tilelang_kernel.py
D	python/sglang/srt/layers/attention/nsa/triton_kernel.py
M	python/sglang/srt/layers/attention/qsa/kernel.py
M	python/sglang/srt/layers/attention/qsa/metadata.py
M	python/sglang/srt/layers/attention/qsa/mqa.py
M	python/sglang/srt/layers/attention/qsa/qsa_indexer.py
M	python/sglang/srt/layers/attention/qsa/sparse_attn.py
M	python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py
M	python/sglang/srt/layers/attention/tbo_backend.py
D	python/sglang/srt/layers/attention/tokenspeed_mla_backend.py
M	python/sglang/srt/layers/attention/torch_flex_backend.py
M	python/sglang/srt/layers/attention/torch_native_backend.py
M	python/sglang/srt/layers/attention/triton_backend.py
M	python/sglang/srt/layers/attention/trtllm_mha_backend.py
M	python/sglang/srt/layers/attention/trtllm_mla_backend.py
M	python/sglang/srt/layers/attention/wave_backend.py
M	python/sglang/srt/layers/attention/xpu_backend.py
M	python/sglang/srt/layers/attn_residual.py
M	python/sglang/srt/layers/aux_hidden_states.py
M	python/sglang/srt/layers/communication/k3_sp_collective.py
A	python/sglang/srt/layers/communication/muse_spark_finalize_workspace.py
R054	python/sglang/kernels/ops/muse_spark/gemm_ag.py	python/sglang/srt/layers/communication/muse_spark_gemm_ag.py
A	python/sglang/srt/layers/communication/sp_collective_gemm_workspace.py
M	python/sglang/srt/layers/cp/base.py
A	python/sglang/srt/layers/cp/bcg/__init__.py
R079	python/sglang/srt/layers/cp/bcg.py	python/sglang/srt/layers/cp/bcg/bcg.py
A	python/sglang/srt/layers/cp/bcg/interleave_bcg.py
A	python/sglang/srt/layers/cp/bcg/zigzag_bcg.py
M	python/sglang/srt/layers/cp/interleave.py
M	python/sglang/srt/layers/cp/padding.py
M	python/sglang/srt/layers/cp/zigzag.py
M	python/sglang/srt/layers/dcp/comm.py
M	python/sglang/srt/layers/dcp/planner.py
M	python/sglang/srt/layers/dp_attention.py
M	python/sglang/srt/layers/engram.py
A	python/sglang/srt/layers/engram_staging.py
M	python/sglang/srt/layers/hyperconnection.py
A	python/sglang/srt/layers/layer_boundary/fusions/custom_all_reduce_fusion.py
M	python/sglang/srt/layers/layer_boundary/fusions/cutedsl.py
A	python/sglang/srt/layers/layer_boundary/fusions/moe_finalize.py
A	python/sglang/srt/layers/layer_boundary/fusions/norm.py
M	python/sglang/srt/layers/layer_boundary/residual/add_norm.py
M	python/sglang/srt/layers/layer_boundary/residual/batch.py
M	python/sglang/srt/layers/layernorm.py
M	python/sglang/srt/layers/layernorm_sp.py
M	python/sglang/srt/layers/linear.py
M	python/sglang/srt/layers/logits_processor.py
M	python/sglang/srt/layers/moe/__init__.py
M	python/sglang/srt/layers/moe/cutlass_moe.py
M	python/sglang/srt/layers/moe/cutlass_w4a8_moe.py
M	python/sglang/srt/layers/moe/dwdp/transport.py
M	python/sglang/srt/layers/moe/ep_moe/layer.py
M	python/sglang/srt/layers/moe/flashinfer_trtllm_moe.py
M	python/sglang/srt/layers/moe/fused_moe_triton/aiter_mxfp4_w4a8_moe.py
M	python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py
M	python/sglang/srt/layers/moe/fused_moe_triton/layer.py
M	python/sglang/srt/layers/moe/hash_topk.py
M	python/sglang/srt/layers/moe/kt_ep_wrapper.py
A	python/sglang/srt/layers/moe/mega_gate.py
M	python/sglang/srt/layers/moe/mega_moe.py
M	python/sglang/srt/layers/moe/mega_moe_flydsl.py
M	python/sglang/srt/layers/moe/moe_runner/aiter.py
M	python/sglang/srt/layers/moe/moe_runner/ascend.py
M	python/sglang/srt/layers/moe/moe_runner/base.py
M	python/sglang/srt/layers/moe/moe_runner/deep_gemm.py
M	python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py
M	python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py
M	python/sglang/srt/layers/moe/moe_runner/marlin.py
M	python/sglang/srt/layers/moe/moe_runner/triton.py
A	python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=256,N=640,device_name=NVIDIA_GB300.json
A	python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=256,N=640,device_name=NVIDIA_GB300_down.json
A	python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_8_0/E=72,N=2048,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json
A	python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_8_0/E=72,N=2048,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json
M	python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py
M	python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py
A	python/sglang/srt/layers/moe/paged_experts/__init__.py
A	python/sglang/srt/layers/moe/paged_experts/executor.py
A	python/sglang/srt/layers/moe/paged_experts/formats.py
A	python/sglang/srt/layers/moe/paged_experts/method.py
A	python/sglang/srt/layers/moe/paged_experts/residency.py
A	python/sglang/srt/layers/moe/paged_experts/runners.py
A	python/sglang/srt/layers/moe/paged_experts/store.py
M	python/sglang/srt/layers/moe/qwen4_decode.py
M	python/sglang/srt/layers/moe/token_dispatcher/__init__.py
A	python/sglang/srt/layers/moe/token_dispatcher/ascend_ops.py
D	python/sglang/srt/layers/moe/token_dispatcher/ascend_tp.py
M	python/sglang/srt/layers/moe/token_dispatcher/base.py
M	python/sglang/srt/layers/moe/token_dispatcher/deepep.py
M	python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py
M	python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py
M	python/sglang/srt/layers/moe/token_dispatcher/pplx.py
M	python/sglang/srt/layers/moe/token_dispatcher/standard.py
M	python/sglang/srt/layers/moe/topk.py
M	python/sglang/srt/layers/moe/utils.py
M	python/sglang/srt/layers/moe/waterfill.py
M	python/sglang/srt/layers/muse_spark/muse_spark_bf16_moe.py
M	python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py
M	python/sglang/srt/layers/muse_spark/muse_spark_megamoe.py
M	python/sglang/srt/layers/n_gram_embedding.py
A	python/sglang/srt/layers/pointer_head.py
M	python/sglang/srt/layers/quantization/__init__.py
M	python/sglang/srt/layers/quantization/awq/awq.py
M	python/sglang/srt/layers/quantization/awq/schemes/awq_cpu.py
M	python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py
M	python/sglang/srt/layers/quantization/base_config.py
M	python/sglang/srt/layers/quantization/base_scheme.py
M	python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py
M	python/sglang/srt/layers/quantization/compressed_tensors/schemes/__init__.py
A	python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_mxfp4.py
A	python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_mxfp4_moe.py
M	python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxint4_moe.py
M	python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py
M	python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py
M	python/sglang/srt/layers/quantization/compressed_tensors/utils.py
M	python/sglang/srt/layers/quantization/expert_pack.py
M	python/sglang/srt/layers/quantization/fp8.py
A	python/sglang/srt/layers/quantization/fp8_activation.py
M	python/sglang/srt/layers/quantization/fp8_hip.py
M	python/sglang/srt/layers/quantization/fp8_utils.py
M	python/sglang/srt/layers/quantization/gguf.py
M	python/sglang/srt/layers/quantization/gptq/gptq.py
M	python/sglang/srt/layers/quantization/gptq/schemes/gptq_cpu.py
M	python/sglang/srt/layers/quantization/gptq/schemes/gptq_moe.py
M	python/sglang/srt/layers/quantization/marlin_utils_fp4.py
A	python/sglang/srt/layers/quantization/minimax_m3_nvfp4.py
M	python/sglang/srt/layers/quantization/modelopt_quant.py
M	python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_mxfp8.py
M	python/sglang/srt/layers/quantization/moe_wna16.py
M	python/sglang/srt/layers/quantization/mxfp4.py
M	python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py
M	python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py
M	python/sglang/srt/layers/quantization/mxfp4_humming_moe.py
M	python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py
M	python/sglang/srt/layers/quantization/nvfp4_online.py
M	python/sglang/srt/layers/quantization/quark/quark.py
M	python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py
M	python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
M	python/sglang/srt/layers/quantization/quark/schemes/quark_w4a8_mxfp4_moe.py
M	python/sglang/srt/layers/quantization/unquant.py
M	python/sglang/srt/layers/radix_attention.py
M	python/sglang/srt/layers/radix_linear_attention.py
M	python/sglang/srt/layers/rotary_embedding/base.py
M	python/sglang/srt/layers/rotary_embedding/mrope.py
M	python/sglang/srt/layers/sampler.py
M	python/sglang/srt/lora/backend/ascend_backend.py
M	python/sglang/srt/lora/backend/base_backend.py
M	python/sglang/srt/lora/backend/chunked_backend.py
M	python/sglang/srt/lora/backend/lora_registry.py
M	python/sglang/srt/lora/backend/torch_backend.py
M	python/sglang/srt/lora/backend/triton_backend.py
A	python/sglang/srt/lora/backend/triton_v2_backend.py
R100	experimental/sgl-router/tests/e2e/infra/__init__.py	python/sglang/srt/lora/dense/__init__.py
A	python/sglang/srt/lora/dense/configs/README.md
A	python/sglang/srt/lora/dense/configs/default.plans.json
A	python/sglang/srt/lora/dense/configs/sm100.plans.json
A	python/sglang/srt/lora/dense/configs/sm90.plans.json
A	python/sglang/srt/lora/dense/mla_correction.py
A	python/sglang/srt/lora/dense/plan.py
A	python/sglang/srt/lora/dense/runner.py
M	python/sglang/srt/lora/layers.py
M	python/sglang/srt/lora/lora_manager.py
M	python/sglang/srt/lora/lora_moe_runner_marlin.py
M	python/sglang/srt/lora/lora_moe_runners.py
M	python/sglang/srt/lora/marlin_lora_temp/activation.py
M	python/sglang/srt/lora/marlin_lora_temp/direct_decode.py
M	python/sglang/srt/lora/marlin_lora_temp/moe_runner.py
M	python/sglang/srt/lora/marlin_lora_temp/shared_outer.py
M	python/sglang/srt/lora/mem_pool.py
A	python/sglang/srt/lora/moe/__init__.py
A	python/sglang/srt/lora/moe/base_gemm_provider/__init__.py
A	python/sglang/srt/lora/moe/base_gemm_provider/base.py
A	python/sglang/srt/lora/moe/base_gemm_provider/contiguous_row_domain.py
A	python/sglang/srt/lora/moe/base_gemm_provider/cutedsl_bf16.py
A	python/sglang/srt/lora/moe/base_gemm_provider/cutedsl_common.py
A	python/sglang/srt/lora/moe/base_gemm_provider/cutedsl_fp8.py
A	python/sglang/srt/lora/moe/base_gemm_provider/gemm_config_store.py
A	python/sglang/srt/lora/moe/base_gemm_provider/input_alignment.py
A	python/sglang/srt/lora/moe/base_gemm_provider/marlin_nvfp4.py
A	python/sglang/srt/lora/moe/base_gemm_provider/masked_row_domain.py
A	python/sglang/srt/lora/moe/base_gemm_provider/triton_bf16.py
A	python/sglang/srt/lora/moe/base_gemm_provider/triton_fp8.py
A	python/sglang/srt/lora/moe/configs/README.md
A	python/sglang/srt/lora/moe/configs/base_gemm/README.md
A	python/sglang/srt/lora/moe/configs/base_gemm/provider=cutedsl_fp8_contiguous,E=256,N1=1024,N2=2048,K=2048,device_name=NVIDIA_H200.json
A	python/sglang/srt/lora/moe/configs/base_gemm/provider=cutedsl_fp8_contiguous,E=256,N1=256,N2=2048,K=2048,device_name=NVIDIA_B200.json
A	python/sglang/srt/lora/moe/configs/base_gemm/provider=cutedsl_fp8_contiguous,E=256,N1=256,N2=2048,K=2048,device_name=NVIDIA_H200.json
A	python/sglang/srt/lora/moe/configs/base_gemm/provider=cutedsl_fp8_contiguous,E=512,N1=256,N2=4096,K=4096,device_name=NVIDIA_B200.json
A	python/sglang/srt/lora/moe/configs/base_gemm/provider=cutedsl_fp8_contiguous,E=512,N1=256,N2=4096,K=4096,device_name=NVIDIA_H200.json
A	python/sglang/srt/lora/moe/configs/base_gemm/provider=cutedsl_fp8_contiguous,E=512,N1=512,N2=4096,K=4096,device_name=NVIDIA_B200.json
A	python/sglang/srt/lora/moe/configs/base_gemm/provider=cutedsl_fp8_contiguous,E=512,N1=512,N2=4096,K=4096,device_name=NVIDIA_GB300.json
A	python/sglang/srt/lora/moe/configs/base_gemm/provider=cutedsl_fp8_masked,E=256,N1=256,N2=2048,K=2048,device_name=NVIDIA_B200.json
A	python/sglang/srt/lora/moe/configs/base_gemm/provider=cutedsl_fp8_masked,E=256,N1=256,N2=2048,K=2048,device_name=NVIDIA_GB300.json
A	python/sglang/srt/lora/moe/configs/base_gemm/provider=cutedsl_fp8_masked,E=256,N1=256,N2=2048,K=2048,device_name=NVIDIA_H200.json
A	python/sglang/srt/lora/moe/configs/base_gemm/provider=cutedsl_fp8_masked,E=512,N1=256,N2=4096,K=4096,device_name=NVIDIA_B200.json
A	python/sglang/srt/lora/moe/configs/base_gemm/provider=cutedsl_fp8_masked,E=512,N1=256,N2=4096,K=4096,device_name=NVIDIA_H200.json
A	python/sglang/srt/lora/moe/configs/base_gemm/provider=cutedsl_fp8_masked,E=512,N1=512,N2=4096,K=4096,device_name=NVIDIA_GB300.json
A	python/sglang/srt/lora/moe/configs/default.plans.json
A	python/sglang/srt/lora/moe/configs/sm100.plans.json
A	python/sglang/srt/lora/moe/configs/sm90.plans.json
A	python/sglang/srt/lora/moe/plan.py
A	python/sglang/srt/lora/moe/quant_info.py
A	python/sglang/srt/lora/moe/routing.py
A	python/sglang/srt/lora/moe/runner.py
M	python/sglang/srt/lora/trtllm_lora_temp/specialized_expand.py
M	python/sglang/srt/lora/utils.py
A	python/sglang/srt/lora/workspace.py
M	python/sglang/srt/managers/cache_controller.py
M	python/sglang/srt/managers/data_parallel_controller.py
A	python/sglang/srt/managers/deferred_output.py
M	python/sglang/srt/managers/detokenizer_manager.py
M	python/sglang/srt/managers/disagg_service.py
M	python/sglang/srt/managers/hisparse_coordinator.py
M	python/sglang/srt/managers/io_struct.py
M	python/sglang/srt/managers/load_snapshot.py
M	python/sglang/srt/managers/mm_schedule.py
M	python/sglang/srt/managers/mm_utils.py
M	python/sglang/srt/managers/multi_tokenizer_mixin.py
M	python/sglang/srt/managers/overlap_utils.py
M	python/sglang/srt/managers/schedule_batch.py
M	python/sglang/srt/managers/schedule_policy.py
M	python/sglang/srt/managers/scheduler.py
M	python/sglang/srt/managers/scheduler_components/batch_result_processor.py
M	python/sglang/srt/managers/scheduler_components/invariant_checker.py
M	python/sglang/srt/managers/scheduler_components/load_inquirer.py
M	python/sglang/srt/managers/scheduler_components/metrics_reporter.py
M	python/sglang/srt/managers/scheduler_components/output_streamer.py
M	python/sglang/srt/managers/scheduler_components/pool_stats_observer.py
M	python/sglang/srt/managers/scheduler_components/request_receiver.py
A	python/sglang/srt/managers/scheduler_components/shm_tag_exchange.py
M	python/sglang/srt/managers/scheduler_pp_mixin.py
M	python/sglang/srt/managers/tokenizer_control_mixin.py
M	python/sglang/srt/managers/tokenizer_manager.py
M	python/sglang/srt/managers/tp_worker.py
M	python/sglang/srt/managers/utils.py
M	python/sglang/srt/mem_cache/allocation.py
M	python/sglang/srt/mem_cache/allocator/base.py
M	python/sglang/srt/mem_cache/allocator/hisparse.py
M	python/sglang/srt/mem_cache/allocator/page_interleave.py
M	python/sglang/srt/mem_cache/allocator/swa.py
D	python/sglang/srt/mem_cache/allocator/unified_hybrid_swa.py
D	python/sglang/srt/mem_cache/allocator/unified_mamba.py
M	python/sglang/srt/mem_cache/base_prefix_cache.py
M	python/sglang/srt/mem_cache/base_swa_memory_pool.py
M	python/sglang/srt/mem_cache/buffer_mode/pipeline.py
M	python/sglang/srt/mem_cache/cache_init_params.py
M	python/sglang/srt/mem_cache/common.py
M	python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
M	python/sglang/srt/mem_cache/device_pool_info.py
A	python/sglang/srt/mem_cache/draft_kv_budget.py
M	python/sglang/srt/mem_cache/dsa_cache_layer_split.py
M	python/sglang/srt/mem_cache/dsv41_request_window.py
M	python/sglang/srt/mem_cache/hicache_auto_size.py
M	python/sglang/srt/mem_cache/hicache_key_scheme.py
M	python/sglang/srt/mem_cache/hicache_storage.py
M	python/sglang/srt/mem_cache/hisparse_memory_pool.py
M	python/sglang/srt/mem_cache/hybrid_cache/host_pool_config.py
M	python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py
M	python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py
M	python/sglang/srt/mem_cache/hybrid_cache/linker_pool_assembler.py
M	python/sglang/srt/mem_cache/index_key_cache.py
A	python/sglang/srt/mem_cache/kv/__init__.py
A	python/sglang/srt/mem_cache/kv/alloc/__init__.py
R082	python/sglang/srt/mem_cache/allocator/unified_sub_pool.py	python/sglang/srt/mem_cache/kv/alloc/band.py
A	python/sglang/srt/mem_cache/kv/alloc/chain.py
A	python/sglang/srt/mem_cache/kv/alloc/id_space.py
A	python/sglang/srt/mem_cache/kv/alloc/interim.py
R086	python/sglang/srt/mem_cache/kv_loc_plan.py	python/sglang/srt/mem_cache/kv/alloc/iteration_ids.py
A	python/sglang/srt/mem_cache/kv/alloc/pricing.py
R079	python/sglang/srt/mem_cache/kv_index_translator.py	python/sglang/srt/mem_cache/kv/alloc/runner_ids.py
A	python/sglang/srt/mem_cache/kv/build.py
A	python/sglang/srt/mem_cache/kv/decl.py
A	python/sglang/srt/mem_cache/kv/layout/__init__.py
A	python/sglang/srt/mem_cache/kv/layout/base.py
A	python/sglang/srt/mem_cache/kv/layout/mha.py
A	python/sglang/srt/mem_cache/kv/layout/mla.py
A	python/sglang/srt/mem_cache/kv/layout/state.py
A	python/sglang/srt/mem_cache/kv/memory/__init__.py
A	python/sglang/srt/mem_cache/kv/memory/arena.py
A	python/sglang/srt/mem_cache/kv/memory/host.py
A	python/sglang/srt/mem_cache/kv/memory/views.py
A	python/sglang/srt/mem_cache/kv/plan.py
A	python/sglang/srt/mem_cache/kv/requests.py
A	python/sglang/srt/mem_cache/kv/routed_cache.py
A	python/sglang/srt/mem_cache/kv/router.py
A	python/sglang/srt/mem_cache/kv/sizing.py
A	python/sglang/srt/mem_cache/kv/state_storage.py
A	python/sglang/srt/mem_cache/kv/storage.py
M	python/sglang/srt/mem_cache/kv_cache_builder.py
M	python/sglang/srt/mem_cache/kv_cache_configurator.py
M	python/sglang/srt/mem_cache/l2_transfer.py
M	python/sglang/srt/mem_cache/layout/fused_draft.py
M	python/sglang/srt/mem_cache/memory_pool.py
M	python/sglang/srt/mem_cache/memory_pool_host.py
A	python/sglang/srt/mem_cache/minimax_m3_nvfp4_pool.py
M	python/sglang/srt/mem_cache/mla_host_dedup.py
M	python/sglang/srt/mem_cache/page_interleave.py
M	python/sglang/srt/mem_cache/page_interleave_pool.py
M	python/sglang/srt/mem_cache/ple_state_pool.py
A	python/sglang/srt/mem_cache/pool_host/_kvcacheio.py
M	python/sglang/srt/mem_cache/pool_host/base.py
M	python/sglang/srt/mem_cache/pool_host/common.py
M	python/sglang/srt/mem_cache/pool_host/dsa.py
M	python/sglang/srt/mem_cache/pool_host/group.py
M	python/sglang/srt/mem_cache/pool_host/mamba.py
M	python/sglang/srt/mem_cache/pool_host/mha.py
M	python/sglang/srt/mem_cache/pool_host/mha_mxfp8.py
A	python/sglang/srt/mem_cache/pool_host/minimax_m3_nvfp4.py
M	python/sglang/srt/mem_cache/pool_host/mla.py
A	python/sglang/srt/mem_cache/pool_host/page_interleave.py
M	python/sglang/srt/mem_cache/pool_host/page_unified.py
M	python/sglang/srt/mem_cache/pool_host/unified.py
M	python/sglang/srt/mem_cache/prefill_budget.py
M	python/sglang/srt/mem_cache/registry.py
M	python/sglang/srt/mem_cache/rust_tree_core/adapter.py
A	python/sglang/srt/mem_cache/static_kv_storage.py
M	python/sglang/srt/mem_cache/storage/backend_factory.py
M	python/sglang/srt/mem_cache/storage/lmcache/lmcache_unified_radix_cache.py
M	python/sglang/srt/mem_cache/storage/mmap/mmap_allocator.py
M	python/sglang/srt/mem_cache/storage/mooncake_store/README.md
M	python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_direct_linker.py
M	python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py
M	python/sglang/srt/mem_cache/storage/tensorcast_store/test_tensorcast_store.py
M	python/sglang/srt/mem_cache/storage/umbp/umbp_direct_linker.py
M	python/sglang/srt/mem_cache/swa_memory_pool.py
M	python/sglang/srt/mem_cache/unified_cache/cache_action.py
M	python/sglang/srt/mem_cache/unified_cache/components/README.md
M	python/sglang/srt/mem_cache/unified_cache/components/base.py
M	python/sglang/srt/mem_cache/unified_cache/components/full.py
M	python/sglang/srt/mem_cache/unified_cache/components/mamba.py
M	python/sglang/srt/mem_cache/unified_cache/components/swa.py
A	python/sglang/srt/mem_cache/unified_cache/linker_config.py
A	python/sglang/srt/mem_cache/unified_cache/linker_factory.py
M	python/sglang/srt/mem_cache/unified_cache/storage_attachment.py
M	python/sglang/srt/mem_cache/unified_cache/tree_core_registry.py
M	python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py
M	python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py
M	python/sglang/srt/mem_cache/unified_cache/unified_tree_core_interface.py
M	python/sglang/srt/mem_cache/unified_draft_pool.py
M	python/sglang/srt/mem_cache/unified_memory_pool.py
M	python/sglang/srt/mem_cache/unified_radix_cache.py
M	python/sglang/srt/model_executor/cpu_graph_runner.py
M	python/sglang/srt/model_executor/cuda_graph_buffer_registry.py
M	python/sglang/srt/model_executor/encoder_swa_replay.py
M	python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py
M	python/sglang/srt/model_executor/forward_batch_info.py
M	python/sglang/srt/model_executor/forward_context.py
A	python/sglang/srt/model_executor/forward_observer.py
M	python/sglang/srt/model_executor/model_runner.py
M	python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py
M	python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py
M	python/sglang/srt/model_executor/model_runner_components/layer_setup.py
M	python/sglang/srt/model_executor/model_runner_components/moe_ep_setup.py
M	python/sglang/srt/model_executor/model_runner_components/ngram_embedding_manager.py
M	python/sglang/srt/model_executor/model_runner_components/spec_aux_hidden_state.py
M	python/sglang/srt/model_executor/model_runner_components/weight_exporter.py
M	python/sglang/srt/model_executor/model_runner_components/weight_updater.py
M	python/sglang/srt/model_executor/pool_configurator.py
M	python/sglang/srt/model_executor/runner/__init__.py
M	python/sglang/srt/model_executor/runner/base_runner.py
M	python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
M	python/sglang/srt/model_executor/runner/eager_runner.py
A	python/sglang/srt/model_executor/runner/early_verify_metadata.py
M	python/sglang/srt/model_executor/runner/flashinfer_autotune.py
M	python/sglang/srt/model_executor/runner/metadata_glue_graph.py
M	python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py
A	python/sglang/srt/model_executor/runner/row_bucket_graph_runner.py
M	python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py
M	python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py
M	python/sglang/srt/model_executor/runner_utils/capture_mode.py
A	python/sglang/srt/model_executor/runner_utils/forked_metadata_prep.py
M	python/sglang/srt/model_executor/runner_utils/shared_read_event.py
M	python/sglang/srt/model_loader/loader.py
M	python/sglang/srt/model_loader/utils.py
M	python/sglang/srt/model_loader/weight_utils.py
M	python/sglang/srt/models/afmoe.py
M	python/sglang/srt/models/bailing_mm_v3.py
M	python/sglang/srt/models/bailing_moe.py
M	python/sglang/srt/models/bailing_moe_linear.py
M	python/sglang/srt/models/bailing_moe_nextn.py
M	python/sglang/srt/models/bailing_moe_v3.py
A	python/sglang/srt/models/beam.py
M	python/sglang/srt/models/cohere2_moe.py
M	python/sglang/srt/models/dbrx.py
M	python/sglang/srt/models/deepseek.py
M	python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py
M	python/sglang/srt/models/deepseek_common/amd/deepseek_v4_gfx95_dense.py
M	python/sglang/srt/models/deepseek_common/attention_backend_handler.py
M	python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py
M	python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha_rocm.py
M	python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
M	python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py
M	python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py
M	python/sglang/srt/models/deepseek_common/decoder_swa_bounded_replay.py
M	python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py
M	python/sglang/srt/models/deepseek_common/moe_transport.py
M	python/sglang/srt/models/deepseek_common/utils.py
M	python/sglang/srt/models/deepseek_nextn.py
M	python/sglang/srt/models/deepseek_ocr.py
M	python/sglang/srt/models/deepseek_v2.py
M	python/sglang/srt/models/deepseek_v4.py
M	python/sglang/srt/models/deepseek_v4_boundary.py
M	python/sglang/srt/models/deepseek_v4_dspark.py
M	python/sglang/srt/models/deepseek_v4_mhc.py
M	python/sglang/srt/models/deepseek_v4_nextn.py
M	python/sglang/srt/models/deepseek_v4_platform_stage.py
A	python/sglang/srt/models/deepseek_v4_replay_tail.py
M	python/sglang/srt/models/deepseek_vl2.py
M	python/sglang/srt/models/dflash.py
M	python/sglang/srt/models/dots3_common/modeling.py
M	python/sglang/srt/models/dots3_common/nextn.py
M	python/sglang/srt/models/dots_vlm.py
M	python/sglang/srt/models/dspark.py
A	python/sglang/srt/models/dspark_minimax.py
M	python/sglang/srt/models/ernie4.py
M	python/sglang/srt/models/ernie45_moe_vl.py
M	python/sglang/srt/models/exaone_moe.py
A	python/sglang/srt/models/flint.py
M	python/sglang/srt/models/gemma3_mm.py
M	python/sglang/srt/models/gemma4_causal.py
M	python/sglang/srt/models/glm4_moe.py
M	python/sglang/srt/models/glm4_moe_lite.py
M	python/sglang/srt/models/glm4_moe_lite_nextn.py
M	python/sglang/srt/models/glm4_moe_nextn.py
M	python/sglang/srt/models/glm4v_moe.py
M	python/sglang/srt/models/glm5_next.py
M	python/sglang/srt/models/gpt_oss.py
M	python/sglang/srt/models/granitemoe.py
M	python/sglang/srt/models/grok.py
M	python/sglang/srt/models/hunyuan.py
M	python/sglang/srt/models/hunyuan_v3.py
M	python/sglang/srt/models/hunyuan_v4.py
M	python/sglang/srt/models/hunyuan_v4_nextn.py
M	python/sglang/srt/models/inkling.py
M	python/sglang/srt/models/inkling_common/attn.py
M	python/sglang/srt/models/inkling_common/dense_mlp.py
M	python/sglang/srt/models/inkling_common/lora.py
M	python/sglang/srt/models/inkling_common/moe.py
M	python/sglang/srt/models/interns1pro.py
M	python/sglang/srt/models/interns2_mobius.py
M	python/sglang/srt/models/iquest_q1.py
M	python/sglang/srt/models/jet_nemotron.py
M	python/sglang/srt/models/kimi_k25.py
M	python/sglang/srt/models/kimi_k25_eagle3.py
M	python/sglang/srt/models/kimi_k3.py
M	python/sglang/srt/models/kimi_k3_vl.py
M	python/sglang/srt/models/kimi_linear.py
M	python/sglang/srt/models/kimi_vl.py
M	python/sglang/srt/models/laguna.py
M	python/sglang/srt/models/lfm2_moe.py
M	python/sglang/srt/models/llada2.py
M	python/sglang/srt/models/llama.py
M	python/sglang/srt/models/llama4.py
M	python/sglang/srt/models/llama_eagle3.py
M	python/sglang/srt/models/llava.py
M	python/sglang/srt/models/longcat_flash.py
M	python/sglang/srt/models/mimo_v2.py
M	python/sglang/srt/models/mindspore.py
M	python/sglang/srt/models/minimax_m2.py
M	python/sglang/srt/models/minimax_m3.py
M	python/sglang/srt/models/minimax_m3_vl.py
M	python/sglang/srt/models/minimax_vl_common.py
M	python/sglang/srt/models/mistral.py
M	python/sglang/srt/models/mixtral.py
M	python/sglang/srt/models/muse_glimmer.py
M	python/sglang/srt/models/muse_spark.py
M	python/sglang/srt/models/nano_nemotron_vl.py
M	python/sglang/srt/models/nemotron_h.py
A	python/sglang/srt/models/nemotron_h_nvfp4_layout.py
M	python/sglang/srt/models/olmoe.py
M	python/sglang/srt/models/phi3_small.py
M	python/sglang/srt/models/phimoe.py
M	python/sglang/srt/models/pixtral.py
M	python/sglang/srt/models/qwen2.py
M	python/sglang/srt/models/qwen2_moe.py
M	python/sglang/srt/models/qwen3.py
M	python/sglang/srt/models/qwen3_5.py
M	python/sglang/srt/models/qwen3_5_text.py
M	python/sglang/srt/models/qwen3_moe.py
M	python/sglang/srt/models/qwen3_next.py
M	python/sglang/srt/models/qwen4_exp.py
M	python/sglang/srt/models/roberta.py
M	python/sglang/srt/models/sarvam_moe.py
M	python/sglang/srt/models/sdar_moe.py
M	python/sglang/srt/models/step3_vl.py
M	python/sglang/srt/models/step3p5.py
M	python/sglang/srt/models/utils.py
M	python/sglang/srt/models/xllm.py
M	python/sglang/srt/models/xverse_moe.py
D	python/sglang/srt/monokernels/guards.py
M	python/sglang/srt/multimodal/dsv41/vl_routing.py
M	python/sglang/srt/multimodal/encoder_preprocessing.py
M	python/sglang/srt/multimodal/inkling/feature_extraction.py
M	python/sglang/srt/multimodal/inkling/image_processing.py
M	python/sglang/srt/multimodal/inkling/image_processing_rust.py
A	python/sglang/srt/multimodal/modality.py
M	python/sglang/srt/multimodal/processors/base_processor.py
M	python/sglang/srt/multimodal/processors/inkling.py
A	python/sglang/srt/multimodal/processors/minimax_m31_media.py
M	python/sglang/srt/multimodal/processors/minimax_m3_vl.py
M	python/sglang/srt/multimodal/processors/nano_nemotron_vl.py
M	python/sglang/srt/multimodal/processors/voxtral.py
M	python/sglang/srt/observability/forward_pass_metrics.py
A	python/sglang/srt/observability/http_request_metrics.py
M	python/sglang/srt/observability/metrics_collector.py
M	python/sglang/srt/observability/trace.py
A	python/sglang/srt/parser/beam.py
M	python/sglang/srt/parser/chat_parsing/content_parsers.py
M	python/sglang/srt/parser/chat_parsing/response_parser.py
A	python/sglang/srt/parser/deepseek_recipe_output.py
A	python/sglang/srt/parser/deepseek_recipe_renderer.py
A	python/sglang/srt/parser/glm53_chat_template.py
A	python/sglang/srt/parser/inkling_output.py
M	python/sglang/srt/parser/inkling_renderer.py
M	python/sglang/srt/parser/inkling_tokenizer.py
M	python/sglang/srt/parser/jinja_template_utils.py
A	python/sglang/srt/parser/python_chat_template.py
M	python/sglang/srt/parser/reasoning_parser.py
M	python/sglang/srt/parser/reasoning_parser_names.py
A	python/sglang/srt/parser/response_template.py
M	python/sglang/srt/parser/template_detection.py
M	python/sglang/srt/parser/template_manager.py
M	python/sglang/srt/platforms/__init__.py
M	python/sglang/srt/platforms/cuda.py
M	python/sglang/srt/platforms/device_mixin.py
M	python/sglang/srt/platforms/interface.py
M	python/sglang/srt/platforms/musa.py
M	python/sglang/srt/platforms/rocm.py
A	python/sglang/srt/platforms/sail.py
M	python/sglang/srt/ray/data_parallel_controller.py
M	python/sglang/srt/ray/engine.py
M	python/sglang/srt/runtime_context.py
M	python/sglang/srt/rust_server/config.py
M	python/sglang/srt/rust_server/server.py
M	python/sglang/srt/sampling/sampling_params.py
M	python/sglang/srt/sampling/watermarking/README.md
M	python/sglang/srt/sampling/watermarking/core.py
M	python/sglang/srt/server_args.py
M	python/sglang/srt/session/streaming_session.py
M	python/sglang/srt/speculative/adaptive_spec_params.py
M	python/sglang/srt/speculative/base_spec_worker.py
M	python/sglang/srt/speculative/decoupled_spec_io.py
A	python/sglang/srt/speculative/dflash_dp.py
M	python/sglang/srt/speculative/dflash_info.py
M	python/sglang/srt/speculative/dflash_utils.py
M	python/sglang/srt/speculative/dflash_worker_v2.py
A	python/sglang/srt/speculative/draft_parallel.py
M	python/sglang/srt/speculative/draft_utils.py
M	python/sglang/srt/speculative/dspark_components/dspark_config.py
A	python/sglang/srt/speculative/dspark_components/dspark_deferred_commit.py
M	python/sglang/srt/speculative/dspark_components/dspark_draft.py
M	python/sglang/srt/speculative/dspark_components/dspark_draft_sampler.py
M	python/sglang/srt/speculative/dspark_components/dspark_kv_inject.py
M	python/sglang/srt/speculative/dspark_components/dspark_observability.py
M	python/sglang/srt/speculative/dspark_components/dspark_planner.py
A	python/sglang/srt/speculative/dspark_components/dspark_sampling_accept_graph.py
M	python/sglang/srt/speculative/dspark_components/dspark_verify.py
M	python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py
M	python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py
M	python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py
M	python/sglang/srt/speculative/eagle_info.py
A	python/sglang/srt/speculative/eagle_megakernel.py
A	python/sglang/srt/speculative/eagle_post_verify_graphs.py
M	python/sglang/srt/speculative/eagle_utils.py
M	python/sglang/srt/speculative/eagle_worker_common.py
M	python/sglang/srt/speculative/eagle_worker_v2.py
M	python/sglang/srt/speculative/frozen_kv_mtp_cuda_graph_runner.py
M	python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
M	python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py
M	python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py
M	python/sglang/srt/speculative/ngram_info.py
M	python/sglang/srt/speculative/ngram_worker.py
M	python/sglang/srt/speculative/ragged_verify.py
M	python/sglang/srt/speculative/spec_info.py
M	python/sglang/srt/speculative/spec_utils.py
M	python/sglang/srt/speculative/standalone_worker_v2.py
M	python/sglang/srt/speculative/uno_worker_v2.py
M	python/sglang/srt/state_capturer/routed_experts.py
M	python/sglang/srt/utils/async_probe.py
M	python/sglang/srt/utils/common.py
M	python/sglang/srt/utils/cuda_vmm_utils.py
M	python/sglang/srt/utils/device_timer.py
M	python/sglang/srt/utils/hf_transformers/common.py
M	python/sglang/srt/utils/hf_transformers/config.py
M	python/sglang/srt/utils/hf_transformers/processor.py
M	python/sglang/srt/utils/network.py
M	python/sglang/srt/utils/offloader.py
M	python/sglang/srt/utils/patch_tokenizer.py
M	python/sglang/srt/utils/request_logger.py
M	python/sglang/srt/utils/server_args_config_parser.py
M	python/sglang/srt/utils/weight_checker.py
M	python/sglang/srt/weight_cache/ipc_loader.py
M	python/sglang/test/ascend/e2e/test_npu_accuracy_utils.py
M	python/sglang/test/ascend/npu_eval_accuracy_kit.py
M	python/sglang/test/dsv41_flash_common.py
A	python/sglang/test/hicache_xpu_common.py
M	python/sglang/test/kernels/deepseek_v4/common.py
M	python/sglang/test/kernels/deepseek_v4/dsv41_kv_quant_reference.py
A	python/sglang/test/kernels/deepseek_v4/fp4_index_k_reference.py
R100	python/sglang/kernels/ops/quantization/fp8_cvt.py	python/sglang/test/kernels/deepseek_v4/fp8_cvt.py
M	python/sglang/test/kernels/deepseek_v4/k_cache_reference.py
A	python/sglang/test/kernels/dsa_k_cache_reference.py
A	python/sglang/test/kernels/kv_canary/scatter_req_token_ids_reference.py
A	python/sglang/test/kernels/minimax_h3_vae.py
R096	python/sglang/kernels/ops/muse_spark/triton_sum.py	python/sglang/test/kernels/triton_sum.py
M	python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py
M	python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py
M	python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py
M	python/sglang/test/kits/attention_unittest/attention_methods/gdn_attention.py
M	python/sglang/test/kits/attention_unittest/attention_methods/kda_attention.py
M	python/sglang/test/kits/attention_unittest/attention_methods/lightning_attention.py
M	python/sglang/test/kits/attention_unittest/attention_methods/mamba2_attention.py
M	python/sglang/test/kits/attention_unittest/attention_methods/mla_attention.py
M	python/sglang/test/kits/attention_unittest/runner_modes/cuda_graph_decode_runner.py
M	python/sglang/test/kits/attention_unittest/runner_modes/speculative_draft_extend_runner.py
M	python/sglang/test/kits/attention_unittest/runner_modes/speculative_draft_runner.py
M	python/sglang/test/kits/attention_unittest/runner_modes/speculative_target_verify_runner.py
A	python/sglang/test/kits/unified_radix_cache/__init__.py
A	python/sglang/test/kits/unified_radix_cache/config.py
A	python/sglang/test/kits/unified_radix_cache/core.py
A	python/sglang/test/kits/unified_radix_cache/fixtures.py
A	python/sglang/test/kits/unified_radix_cache/hicache.py
A	python/sglang/test/kits/unified_radix_cache/insert_walk.py
A	python/sglang/test/kits/unified_radix_cache/kv_events.py
A	python/sglang/test/kits/unified_radix_cache/load_back.py
A	python/sglang/test/kits/unified_radix_cache/mixins.py
R094	test/registered/unit/mem_cache/rust_unified_tree_core_inspector.py	python/sglang/test/kits/unified_radix_cache/rust_unified_tree_core_inspector.py
A	python/sglang/test/kits/unified_radix_cache/storage.py
R086	test/registered/unit/mem_cache/unified_tree_core_inspection_interface.py	python/sglang/test/kits/unified_radix_cache/unified_tree_core_inspection_interface.py
R081	test/registered/unit/mem_cache/unified_tree_core_inspector.py	python/sglang/test/kits/unified_radix_cache/unified_tree_core_inspector.py
M	python/sglang/test/kv_canary/fixtures.py
M	python/sglang/test/mem_cache_utils.py
M	python/sglang/test/mooncake_utils.py
A	python/sglang/test/muse_spark_utils.py
M	python/sglang/test/scripted_runtime/tree_core_inspection.py
M	python/sglang/test/server_fixtures/disaggregation_fixture.py
M	python/sglang/test/test_marlin_utils.py
M	python/sglang/test/test_utils.py
A	python/sglang/test/tree_slots.py
M	python/sglang/test/unified_allocator_fixtures.py
A	python/sglang/test/unified_allocator_trace.py
M	rust/Cargo.lock
M	rust/sglang-api-types/src/generated/sglang.api.v1.rs
M	rust/sglang-api-types/src/generated/sglang.api.v1.serde.rs
M	rust/sglang-api-types/src/generated/sglang.api.v1.tonic.rs
M	rust/sglang-api-types/testdata/json_parity.json
M	rust/sglang-api-types/tests/json_contract.rs
M	rust/sglang-api-types/tests/json_parity.rs
M	rust/sglang-bench/src/request.rs
M	rust/sglang-bench/tests/grpc.rs
M	rust/sglang-mm/README.md
M	rust/sglang-mm/src/driver.rs
M	rust/sglang-processor/Cargo.toml
M	rust/sglang-processor/README.md
M	rust/sglang-processor/src/lib.rs
M	rust/sglang-processor/src/openai/chat.rs
M	rust/sglang-processor/src/openai/completions.rs
M	rust/sglang-processor/src/openai/mod.rs
M	rust/sglang-processor/src/openai/request.rs
M	rust/sglang-processor/src/openai/wire.rs
M	rust/sglang-processor/src/py.rs
M	rust/sglang-processor/src/render/loader.rs
M	rust/sglang-processor/src/render/mod.rs
M	rust/sglang-processor/src/render/models/deepseek_v4.rs
D	rust/sglang-processor/src/render/models/deepseek_v41.rs
M	rust/sglang-processor/src/render/models/glm53.rs
M	rust/sglang-processor/src/render/models/kimi_k3.rs
M	rust/sglang-processor/src/render/models/mod.rs
A	rust/sglang-processor/src/render/models/pydantic_dump.rs
M	rust/sglang-processor/src/render/models/qwen38.rs
M	rust/sglang-processor/src/render/models/schema_types.rs
M	rust/sglang-processor/src/render/selection.rs
A	rust/sglang-processor/src/render/template.rs
M	rust/sglang-processor/src/render/thinking.rs
M	rust/sglang-processor/src/think/mod.rs
M	rust/sglang-processor/src/think/models/deepseek_v4.rs
M	rust/sglang-processor/src/think/models/glm45.rs
M	rust/sglang-processor/src/think/models/qwen3.rs
M	rust/sglang-processor/src/tokenizer/mod.rs
M	rust/sglang-processor/tests/fixtures/openai_parity/deepseek-v4-flash-0731.json
M	rust/sglang-processor/tests/fixtures/openai_parity/deepseek-v4.1-flash.json
M	rust/sglang-processor/tests/fixtures/openai_parity/glm-5.3.json
M	rust/sglang-processor/tests/fixtures/openai_parity/kimi-k3.json
M	rust/sglang-processor/tests/fixtures/openai_parity/qwen3.8-27b.json
M	rust/sglang-processor/tests/fixtures/parity/deepseek-v4-flash-0731.json
M	rust/sglang-processor/tests/fixtures/parity/deepseek-v4.1-flash.json
M	rust/sglang-processor/tests/fixtures/parity/glm-5.3-flash.json
M	rust/sglang-processor/tests/fixtures/parity/glm-5.3.json
M	rust/sglang-processor/tests/fixtures/parity/qwen3.8-2.4t-a95b.json
M	rust/sglang-processor/tests/fixtures/parity/qwen3.8-27b.json
M	rust/sglang-processor/tests/fixtures/parity/qwen3.8-flash-next.json
M	rust/sglang-processor/tests/openai_parity.rs
M	rust/sglang-processor/tests/parity.rs
M	rust/sglang-processor/tests/scripts/generate_openai_parity.py
M	rust/sglang-processor/tests/scripts/generate_parity.py
A	rust/sglang-radix-tree/src/cache_action.rs
M	rust/sglang-radix-tree/src/components/full.rs
M	rust/sglang-radix-tree/src/components/mamba.rs
M	rust/sglang-radix-tree/src/components/mod.rs
M	rust/sglang-radix-tree/src/components/swa.rs
M	rust/sglang-radix-tree/src/lib.rs
M	rust/sglang-radix-tree/src/node.rs
M	rust/sglang-radix-tree/src/python_bindings.rs
A	rust/sglang-radix-tree/src/python_bindings/cache_action.rs
M	rust/sglang-radix-tree/src/tests/components/base.rs
M	rust/sglang-radix-tree/src/tests/components/full.rs
M	rust/sglang-radix-tree/src/tests/components/mamba.rs
M	rust/sglang-radix-tree/src/tests/components/swa.rs
M	rust/sglang-radix-tree/src/tests/node.rs
M	rust/sglang-radix-tree/src/tests/test_utils.rs
M	rust/sglang-radix-tree/src/tests/unified_tree_core.rs
M	rust/sglang-radix-tree/src/unified_tree_core.rs
M	rust/sglang-server/Cargo.toml
M	rust/sglang-server/src/api_server.rs
M	rust/sglang-server/src/api_server/core.rs
M	rust/sglang-server/src/api_server/core/error.rs
A	rust/sglang-server/src/api_server/core/fanout.rs
M	rust/sglang-server/src/api_server/core/frame.rs
D	rust/sglang-server/src/api_server/core/handle.rs
A	rust/sglang-server/src/api_server/core/metrics.rs
A	rust/sglang-server/src/api_server/core/peer.rs
M	rust/sglang-server/src/api_server/core/prefetch.rs
M	rust/sglang-server/src/api_server/disaggregation/bootstrap.rs
M	rust/sglang-server/src/api_server/grpc/app.rs
M	rust/sglang-server/src/api_server/grpc/native_api.rs
M	rust/sglang-server/src/api_server/grpc/service.rs
A	rust/sglang-server/src/api_server/handlers.rs
A	rust/sglang-server/src/api_server/handlers/control.rs
A	rust/sglang-server/src/api_server/handlers/generate.rs
A	rust/sglang-server/src/api_server/handlers/handle.rs
A	rust/sglang-server/src/api_server/handlers/health.rs
A	rust/sglang-server/src/api_server/handlers/hicache.rs
A	rust/sglang-server/src/api_server/handlers/info.rs
A	rust/sglang-server/src/api_server/handlers/test_utils.rs
A	rust/sglang-server/src/api_server/handlers/tokenize.rs
M	rust/sglang-server/src/api_server/http.rs
M	rust/sglang-server/src/api_server/http/app.rs
A	rust/sglang-server/src/api_server/http/generate_hook.rs
A	rust/sglang-server/src/api_server/http/hicache.rs
M	rust/sglang-server/src/api_server/http/info.rs
M	rust/sglang-server/src/api_server/http/loads.rs
M	rust/sglang-server/src/api_server/http/metrics.rs
M	rust/sglang-server/src/api_server/http/native_api.rs
M	rust/sglang-server/src/api_server/http/openai.rs
M	rust/sglang-server/src/api_server/http/openai/chat.rs
M	rust/sglang-server/src/api_server/http/openai/completions.rs
A	rust/sglang-server/src/api_server/http/openai/lowered.rs
M	rust/sglang-server/src/api_server/http/openai/template.rs
D	rust/sglang-server/src/api_server/http/openai/template_builtins.rs
D	rust/sglang-server/src/api_server/http/openai/template_legacy.rs
D	rust/sglang-server/src/api_server/http/openai/template_loader.rs
M	rust/sglang-server/src/api_server/http/openai/test_utils.rs
M	rust/sglang-server/src/api_server/http/openai/tokenize.rs
M	rust/sglang-server/src/api_server/http/openai/tools.rs
M	rust/sglang-server/src/lib.rs
M	rust/sglang-server/src/message/config.rs
M	rust/sglang-server/src/message/detok.rs
M	rust/sglang-server/src/message/finish_reason.rs
M	rust/sglang-server/src/message/io_struct.rs
M	rust/sglang-server/src/message/request.rs
M	rust/sglang-server/src/message/response.rs
M	rust/sglang-server/src/message/sampling.rs
M	rust/sglang-server/src/message/types.rs
M	rust/sglang-server/src/message/wire.rs
M	rust/sglang-server/src/multi_modality/payload.rs
M	rust/sglang-server/src/multi_modality/worker.rs
M	rust/sglang-server/src/tokenizer_manager/channel.rs
M	rust/sglang-server/src/tokenizer_manager/detokenizer.rs
M	rust/sglang-server/src/tokenizer_manager/from_scheduler.rs
M	rust/sglang-server/src/tokenizer_manager/to_scheduler.rs
M	rust/sglang-server/src/tokenizer_manager/to_scheduler_tests.rs
M	rust/sglang-server/src/tokenizer_manager/to_scheduler_validation.rs
M	rust/sglang-server/src/tokenizer_manager/tokenizer.rs
M	rust/sglang-server/src/utils/prometheus.rs
M	rust/sglang-server/src/utils/runtime.rs
A	rust/sglang-server/testdata/frame_metadata.json
A	rust/sglang-server/testdata/special_token_tokenizer.json
A	rust/sglang-server/testdata/tokenizer_metrics.json
M	scripts/build_sgl_deep_gemm.sh
M	scripts/build_sgl_deepep.sh
A	scripts/build_sgl_deepep_v25.sh
A	scripts/build_sgl_flash_mla.sh
M	scripts/ci/amd/amd_ci_install_dependency.sh
M	scripts/ci/amd/amd_ci_start_container.sh
M	scripts/ci/amd/amd_ci_start_container_disagg.sh
M	scripts/ci/cuda/ci_install_dependency.sh
M	scripts/ci/npu/npu_ci_install_dependency.sh
D	scripts/ci/npu/precise-test/analyze_failure_report.py
A	scripts/ci/npu/precise-test/bucket_recommended_tests.py
M	scripts/ci/npu/precise-test/coveragerc
A	scripts/ci/npu/precise-test/list_multimodal_tests.py
A	scripts/ci/npu/precise-test/precise_test_coverage_plugin.py
M	scripts/ci/npu/precise-test/run_tests_with_coverage.sh
M	scripts/ci/npu/precise-test/test_selector.py
A	scripts/ci/opt_in_ci_tiers.json
A	scripts/ci/ppu/build_and_test.sh
A	scripts/ci/ppu/check_dependencies.py
A	scripts/ci/ppu/prepare_runtime.py
A	scripts/ci/ppu/run_selftests.sh
A	scripts/ci/ppu/run_test.sh
A	scripts/ci/ppu/test_ppu_ci_dependencies.py
A	scripts/ci/ppu/test_ppu_ci_docker.py
A	scripts/ci/ppu/test_ppu_ci_suite.py
M	scripts/ci/rerun_test_groups.json
M	scripts/ci/runner_configs.yml
M	scripts/ci/utils/diffusion/compute_diffusion_partitions.py
A	scripts/ci/utils/diffusion/test_compute_diffusion_partitions.py
M	scripts/ci/utils/runner_utilization_report.py
M	scripts/ci/utils/slash_command_handler.py
M	scripts/ci/utils/test_runner_utilization_report.py
M	scripts/ci/xpu/xpu_ci_start_container.sh
A	scripts/lint/check_dockerfile_wheel_pins.py
A	scripts/lint/check_opt_in_ci_tiers.py
M	scripts/release/bump_kernel_version_to_sglang.py
M	scripts/release/check_kernel_version_to_sglang.py
M	scripts/release/get_version_tag.py
M	scripts/release/utils.py
A	scripts/rename_sgl_flash_mla_whl.sh
M	scripts/update_deepep_whl_index.py
A	scripts/update_flashmla_whl_index.py
M	scripts/update_kernel_whl_index.py
M	sgl-model-gateway/README.md
M	sgl-model-gateway/bindings/python/src/lib.rs
M	sgl-model-gateway/bindings/python/src/sglang_router/router.py
M	sgl-model-gateway/bindings/python/src/sglang_router/router_args.py
M	sgl-model-gateway/src/config/builder.rs
M	sgl-model-gateway/src/config/types.rs
M	sgl-model-gateway/src/config/validation.rs
M	sgl-model-gateway/src/main.rs
A	sgl-model-gateway/src/routers/http/decode_gate.rs
M	sgl-model-gateway/src/routers/http/mod.rs
M	sgl-model-gateway/src/routers/http/pd_router.rs
M	sgl-model-gateway/tests/common/mock_worker.rs
M	sgl-model-gateway/tests/routing/pd_routing_test.rs
R100	experimental/sgl-router/.gitignore	sgl-router/.gitignore
R100	experimental/sgl-router/ARCHITECTURE.md	sgl-router/ARCHITECTURE.md
R097	experimental/sgl-router/BENCHMARKS.md	sgl-router/BENCHMARKS.md
R090	experimental/sgl-router/Cargo.lock	sgl-router/Cargo.lock
R097	experimental/sgl-router/Cargo.toml	sgl-router/Cargo.toml
A	sgl-router/EXAMPLES.md
R097	experimental/sgl-router/README.md	sgl-router/README.md
R091	experimental/sgl-router/REFERENCE.md	sgl-router/REFERENCE.md
R100	experimental/sgl-router/assets/architecture-deployment.png	sgl-router/assets/architecture-deployment.png
R100	experimental/sgl-router/benches/policy_select.rs	sgl-router/benches/policy_select.rs
R100	experimental/sgl-router/benches/tree_lookup.rs	sgl-router/benches/tree_lookup.rs
R062	experimental/sgl-router/deny.toml	sgl-router/deny.toml
R095	experimental/sgl-router/monitoring/README.md	sgl-router/monitoring/README.md
R099	experimental/sgl-router/monitoring/grafana-dashboard.json	sgl-router/monitoring/grafana-dashboard.json
R100	experimental/sgl-router/rust-toolchain.toml	sgl-router/rust-toolchain.toml
R100	experimental/sgl-router/sgl-kv-indexer/.gitignore	sgl-router/sgl-kv-indexer/.gitignore
R100	experimental/sgl-router/sgl-kv-indexer/Cargo.toml	sgl-router/sgl-kv-indexer/Cargo.toml
R099	experimental/sgl-router/sgl-kv-indexer/README.md	sgl-router/sgl-kv-indexer/README.md
R100	experimental/sgl-router/sgl-kv-indexer/build.rs	sgl-router/sgl-kv-indexer/build.rs
R100	experimental/sgl-router/sgl-kv-indexer/proto/kv_indexer.proto	sgl-router/sgl-kv-indexer/proto/kv_indexer.proto
R100	experimental/sgl-router/sgl-kv-indexer/src/admission.rs	sgl-router/sgl-kv-indexer/src/admission.rs
R100	experimental/sgl-router/sgl-kv-indexer/src/bin/kv-indexer-bridge.rs	sgl-router/sgl-kv-indexer/src/bin/kv-indexer-bridge.rs
R100	experimental/sgl-router/sgl-kv-indexer/src/bin/kv-indexer-server.rs	sgl-router/sgl-kv-indexer/src/bin/kv-indexer-server.rs
R100	experimental/sgl-router/sgl-kv-indexer/src/bridge.rs	sgl-router/sgl-kv-indexer/src/bridge.rs
R096	experimental/sgl-router/sgl-kv-indexer/src/client.rs	sgl-router/sgl-kv-indexer/src/client.rs
R100	experimental/sgl-router/sgl-kv-indexer/src/lib.rs	sgl-router/sgl-kv-indexer/src/lib.rs
R100	experimental/sgl-router/sgl-kv-indexer/src/lora.rs	sgl-router/sgl-kv-indexer/src/lora.rs
R100	experimental/sgl-router/sgl-kv-indexer/src/memory_backend.rs	sgl-router/sgl-kv-indexer/src/memory_backend.rs
R098	experimental/sgl-router/sgl-kv-indexer/src/service.rs	sgl-router/sgl-kv-indexer/src/service.rs
R100	experimental/sgl-router/sgl-kv-indexer/src/shutdown.rs	sgl-router/sgl-kv-indexer/src/shutdown.rs
R100	experimental/sgl-router/sgl-kv-indexer/tests/common/id.rs	sgl-router/sgl-kv-indexer/tests/common/id.rs
R100	experimental/sgl-router/sgl-kv-indexer/tests/common/kv.rs	sgl-router/sgl-kv-indexer/tests/common/kv.rs
R100	experimental/sgl-router/sgl-kv-indexer/tests/common/net.rs	sgl-router/sgl-kv-indexer/tests/common/net.rs
R100	experimental/sgl-router/sgl-kv-indexer/tests/grpc_contract.rs	sgl-router/sgl-kv-indexer/tests/grpc_contract.rs
R100	experimental/sgl-router/sgl-kv-indexer/tests/memory_integration.rs	sgl-router/sgl-kv-indexer/tests/memory_integration.rs
R100	experimental/sgl-router/src/buckets_reorg.rs	sgl-router/src/buckets_reorg.rs
R095	experimental/sgl-router/src/config/cli.rs	sgl-router/src/config/cli.rs
R100	experimental/sgl-router/src/config/mod.rs	sgl-router/src/config/mod.rs
R100	experimental/sgl-router/src/config/sampling.rs	sgl-router/src/config/sampling.rs
R097	experimental/sgl-router/src/config/types.rs	sgl-router/src/config/types.rs
R100	experimental/sgl-router/src/discovery/k8s.rs	sgl-router/src/discovery/k8s.rs
R100	experimental/sgl-router/src/discovery/k8s/peers.rs	sgl-router/src/discovery/k8s/peers.rs
R100	experimental/sgl-router/src/discovery/mod.rs	sgl-router/src/discovery/mod.rs
R100	experimental/sgl-router/src/discovery/static_urls.rs	sgl-router/src/discovery/static_urls.rs
R100	experimental/sgl-router/src/discovery/types.rs	sgl-router/src/discovery/types.rs
R100	experimental/sgl-router/src/health/circuit_breaker.rs	sgl-router/src/health/circuit_breaker.rs
R100	experimental/sgl-router/src/health/mod.rs	sgl-router/src/health/mod.rs
R100	experimental/sgl-router/src/lib.rs	sgl-router/src/lib.rs
R097	experimental/sgl-router/src/main.rs	sgl-router/src/main.rs
R100	experimental/sgl-router/src/policies/admission.rs	sgl-router/src/policies/admission.rs
R088	experimental/sgl-router/src/policies/buckets.rs	sgl-router/src/policies/buckets.rs
R099	experimental/sgl-router/src/policies/cache_aware.rs	sgl-router/src/policies/cache_aware.rs
R100	experimental/sgl-router/src/policies/decode.rs	sgl-router/src/policies/decode.rs
R100	experimental/sgl-router/src/policies/dp_rank.rs	sgl-router/src/policies/dp_rank.rs
R100	experimental/sgl-router/src/policies/factory.rs	sgl-router/src/policies/factory.rs
R100	experimental/sgl-router/src/policies/load_based.rs	sgl-router/src/policies/load_based.rs
R099	experimental/sgl-router/src/policies/mod.rs	sgl-router/src/policies/mod.rs
R100	experimental/sgl-router/src/policies/power_of_two.rs	sgl-router/src/policies/power_of_two.rs
R100	experimental/sgl-router/src/policies/random.rs	sgl-router/src/policies/random.rs
R100	experimental/sgl-router/src/policies/registry.rs	sgl-router/src/policies/registry.rs
R100	experimental/sgl-router/src/policies/round_robin.rs	sgl-router/src/policies/round_robin.rs
R100	experimental/sgl-router/src/policies/scoring/admission.rs	sgl-router/src/policies/scoring/admission.rs
R100	experimental/sgl-router/src/policies/scoring/argmax.rs	sgl-router/src/policies/scoring/argmax.rs
R100	experimental/sgl-router/src/policies/scoring/mod.rs	sgl-router/src/policies/scoring/mod.rs
R100	experimental/sgl-router/src/policies/scoring/prefix_cache.rs	sgl-router/src/policies/scoring/prefix_cache.rs
R080	experimental/sgl-router/src/policies/selection.rs	sgl-router/src/policies/selection.rs
R100	experimental/sgl-router/src/policies/session_aware.rs	sgl-router/src/policies/session_aware.rs
R100	experimental/sgl-router/src/policies/sticky.rs	sgl-router/src/policies/sticky.rs
R100	experimental/sgl-router/src/policies_reorg/admission.rs	sgl-router/src/policies_reorg/admission.rs
R100	experimental/sgl-router/src/policies_reorg/affinity.rs	sgl-router/src/policies_reorg/affinity.rs
R099	experimental/sgl-router/src/policies_reorg/cache_aware.rs	sgl-router/src/policies_reorg/cache_aware.rs
R100	experimental/sgl-router/src/policies_reorg/factory.rs	sgl-router/src/policies_reorg/factory.rs
R100	experimental/sgl-router/src/policies_reorg/mod.rs	sgl-router/src/policies_reorg/mod.rs
R100	experimental/sgl-router/src/policies_reorg/power_of_two.rs	sgl-router/src/policies_reorg/power_of_two.rs
R100	experimental/sgl-router/src/policies_reorg/session_aware.rs	sgl-router/src/policies_reorg/session_aware.rs
R100	experimental/sgl-router/src/protocol/anthropic/mod.rs	sgl-router/src/protocol/anthropic/mod.rs
R100	experimental/sgl-router/src/protocol/anthropic/request.rs	sgl-router/src/protocol/anthropic/request.rs
R100	experimental/sgl-router/src/protocol/anthropic/response.rs	sgl-router/src/protocol/anthropic/response.rs
R100	experimental/sgl-router/src/protocol/anthropic/stream.rs	sgl-router/src/protocol/anthropic/stream.rs
R100	experimental/sgl-router/src/protocol/mod.rs	sgl-router/src/protocol/mod.rs
R100	experimental/sgl-router/src/protocol/sse.rs	sgl-router/src/protocol/sse.rs
R093	experimental/sgl-router/src/proxy/abort.rs	sgl-router/src/proxy/abort.rs
R100	experimental/sgl-router/src/proxy/grpc.rs	sgl-router/src/proxy/grpc.rs
R098	experimental/sgl-router/src/proxy/mod.rs	sgl-router/src/proxy/mod.rs
R100	experimental/sgl-router/src/proxy/sse.rs	sgl-router/src/proxy/sse.rs
R100	experimental/sgl-router/src/server/app.rs	sgl-router/src/server/app.rs
R098	experimental/sgl-router/src/server/app_context.rs	sgl-router/src/server/app_context.rs
R100	experimental/sgl-router/src/server/error.rs	sgl-router/src/server/error.rs
R100	experimental/sgl-router/src/server/grpc.rs	sgl-router/src/server/grpc.rs
R100	experimental/sgl-router/src/server/header_utils.rs	sgl-router/src/server/header_utils.rs
R100	experimental/sgl-router/src/server/inflight.rs	sgl-router/src/server/inflight.rs
R099	experimental/sgl-router/src/server/metrics.rs	sgl-router/src/server/metrics.rs
R100	experimental/sgl-router/src/server/mod.rs	sgl-router/src/server/mod.rs
R100	experimental/sgl-router/src/server/routes/cache.rs	sgl-router/src/server/routes/cache.rs
R099	experimental/sgl-router/src/server/routes/chat.rs	sgl-router/src/server/routes/chat.rs
R085	experimental/sgl-router/src/server/routes/chat/forward.rs	sgl-router/src/server/routes/chat/forward.rs
R058	experimental/sgl-router/src/server/routes/chat/openai.rs	sgl-router/src/server/routes/chat/openai.rs
R099	experimental/sgl-router/src/server/routes/chat/preparation.rs	sgl-router/src/server/routes/chat/preparation.rs
R100	experimental/sgl-router/src/server/routes/chat/reorg.rs	sgl-router/src/server/routes/chat/reorg.rs
R100	experimental/sgl-router/src/server/routes/health.rs	sgl-router/src/server/routes/health.rs
R100	experimental/sgl-router/src/server/routes/messages.rs	sgl-router/src/server/routes/messages.rs
R097	experimental/sgl-router/src/server/routes/metrics.rs	sgl-router/src/server/routes/metrics.rs
R100	experimental/sgl-router/src/server/routes/mod.rs	sgl-router/src/server/routes/mod.rs
R100	experimental/sgl-router/src/server/routes/models.rs	sgl-router/src/server/routes/models.rs
R100	experimental/sgl-router/src/server/routes/tokenize.rs	sgl-router/src/server/routes/tokenize.rs
R100	experimental/sgl-router/src/server/shutdown.rs	sgl-router/src/server/shutdown.rs
R100	experimental/sgl-router/src/state/affinity_store.rs	sgl-router/src/state/affinity_store.rs
R100	experimental/sgl-router/src/state/kv_events/block_size_oracle.rs	sgl-router/src/state/kv_events/block_size_oracle.rs
R100	experimental/sgl-router/src/state/kv_events/bootstrap.rs	sgl-router/src/state/kv_events/bootstrap.rs
R100	experimental/sgl-router/src/state/kv_events/bootstrap/fetch.rs	sgl-router/src/state/kv_events/bootstrap/fetch.rs
R100	experimental/sgl-router/src/state/kv_events/bootstrap/peers.rs	sgl-router/src/state/kv_events/bootstrap/peers.rs
R100	experimental/sgl-router/src/state/kv_events/bootstrap/tracker.rs	sgl-router/src/state/kv_events/bootstrap/tracker.rs
R100	experimental/sgl-router/src/state/kv_events/bootstrap/vet.rs	sgl-router/src/state/kv_events/bootstrap/vet.rs
R100	experimental/sgl-router/src/state/kv_events/discovery.rs	sgl-router/src/state/kv_events/discovery.rs
R100	experimental/sgl-router/src/state/kv_events/hash.rs	sgl-router/src/state/kv_events/hash.rs
R100	experimental/sgl-router/src/state/kv_events/index.rs	sgl-router/src/state/kv_events/index.rs
R100	experimental/sgl-router/src/state/kv_events/index/coordinator.rs	sgl-router/src/state/kv_events/index/coordinator.rs
R100	experimental/sgl-router/src/state/kv_events/index/fallback.rs	sgl-router/src/state/kv_events/index/fallback.rs
R100	experimental/sgl-router/src/state/kv_events/index/graft.rs	sgl-router/src/state/kv_events/index/graft.rs
R100	experimental/sgl-router/src/state/kv_events/index/probe.rs	sgl-router/src/state/kv_events/index/probe.rs
R100	experimental/sgl-router/src/state/kv_events/index/producer.rs	sgl-router/src/state/kv_events/index/producer.rs
R100	experimental/sgl-router/src/state/kv_events/index/sweep.rs	sgl-router/src/state/kv_events/index/sweep.rs
R100	experimental/sgl-router/src/state/kv_events/index/test_support.rs	sgl-router/src/state/kv_events/index/test_support.rs
R100	experimental/sgl-router/src/state/kv_events/index/tests.rs	sgl-router/src/state/kv_events/index/tests.rs
R100	experimental/sgl-router/src/state/kv_events/mod.rs	sgl-router/src/state/kv_events/mod.rs
R053	experimental/sgl-router/src/state/kv_events/pending.rs	sgl-router/src/state/kv_events/pending.rs
A	sgl-router/src/state/kv_events/prefix_provider.rs
R100	experimental/sgl-router/src/state/kv_events/subscriber.rs	sgl-router/src/state/kv_events/subscriber.rs
R100	experimental/sgl-router/src/state/kv_events/tally.rs	sgl-router/src/state/kv_events/tally.rs
R100	experimental/sgl-router/src/state/kv_events/tree.rs	sgl-router/src/state/kv_events/tree.rs
R100	experimental/sgl-router/src/state/kv_events/tree/snapshot.rs	sgl-router/src/state/kv_events/tree/snapshot.rs
R100	experimental/sgl-router/src/state/kv_events/wire.rs	sgl-router/src/state/kv_events/wire.rs
R100	experimental/sgl-router/src/state/load_monitor/engine_ranking.rs	sgl-router/src/state/load_monitor/engine_ranking.rs
R095	experimental/sgl-router/src/state/load_monitor/engine_reported_load.rs	sgl-router/src/state/load_monitor/engine_reported_load.rs
R100	experimental/sgl-router/src/state/load_monitor/mod.rs	sgl-router/src/state/load_monitor/mod.rs
R100	experimental/sgl-router/src/state/load_monitor/router_inflight_load.rs	sgl-router/src/state/load_monitor/router_inflight_load.rs
R100	experimental/sgl-router/src/state/mod.rs	sgl-router/src/state/mod.rs
R095	experimental/sgl-router/src/tokenizer/adapter.rs	sgl-router/src/tokenizer/adapter.rs
R095	experimental/sgl-router/src/tokenizer/chat_formatter.rs	sgl-router/src/tokenizer/chat_formatter.rs
R100	experimental/sgl-router/src/tokenizer/kimi.rs	sgl-router/src/tokenizer/kimi.rs
R099	experimental/sgl-router/src/tokenizer/mod.rs	sgl-router/src/tokenizer/mod.rs
R100	experimental/sgl-router/src/tokenizer/stats.rs	sgl-router/src/tokenizer/stats.rs
A	sgl-router/src/workers/decode_gate.rs
R094	experimental/sgl-router/src/workers/introspect.rs	sgl-router/src/workers/introspect.rs
R095	experimental/sgl-router/src/workers/manager.rs	sgl-router/src/workers/manager.rs
R086	experimental/sgl-router/src/workers/mod.rs	sgl-router/src/workers/mod.rs
R100	experimental/sgl-router/src/workers/registry.rs	sgl-router/src/workers/registry.rs
R100	experimental/sgl-router/src/workers/worker.rs	sgl-router/src/workers/worker.rs
R100	experimental/sgl-router/tests/component/discovery/mod.rs	sgl-router/tests/component/discovery/mod.rs
R100	experimental/sgl-router/tests/component/discovery/static_urls.rs	sgl-router/tests/component/discovery/static_urls.rs
R100	experimental/sgl-router/tests/component/health/circuit_breaker.rs	sgl-router/tests/component/health/circuit_breaker.rs
R100	experimental/sgl-router/tests/component/health/mod.rs	sgl-router/tests/component/health/mod.rs
R100	experimental/sgl-router/tests/component/main.rs	sgl-router/tests/component/main.rs
R100	experimental/sgl-router/tests/component/policies/bucket_domains.rs	sgl-router/tests/component/policies/bucket_domains.rs
R100	experimental/sgl-router/tests/component/policies/cache_prefix_provider.rs	sgl-router/tests/component/policies/cache_prefix_provider.rs
R100	experimental/sgl-router/tests/component/policies/decode.rs	sgl-router/tests/component/policies/decode.rs
R100	experimental/sgl-router/tests/component/policies/fused_score.rs	sgl-router/tests/component/policies/fused_score.rs
R100	experimental/sgl-router/tests/component/policies/kv_events_hash_parity.rs	sgl-router/tests/component/policies/kv_events_hash_parity.rs
R100	experimental/sgl-router/tests/component/policies/kv_events_peer_bootstrap.rs	sgl-router/tests/component/policies/kv_events_peer_bootstrap.rs
R100	experimental/sgl-router/tests/component/policies/kv_events_tree_concurrent.rs	sgl-router/tests/component/policies/kv_events_tree_concurrent.rs
R100	experimental/sgl-router/tests/component/policies/kv_events_two_subscribers.rs	sgl-router/tests/component/policies/kv_events_two_subscribers.rs
R100	experimental/sgl-router/tests/component/policies/mod.rs	sgl-router/tests/component/policies/mod.rs
R100	experimental/sgl-router/tests/component/policies/power_of_two.rs	sgl-router/tests/component/policies/power_of_two.rs
R100	experimental/sgl-router/tests/component/policies/round_robin.rs	sgl-router/tests/component/policies/round_robin.rs
R100	experimental/sgl-router/tests/component/policies/zmq_helpers.rs	sgl-router/tests/component/policies/zmq_helpers.rs
R100	experimental/sgl-router/tests/component/policies_reorg.rs	sgl-router/tests/component/policies_reorg.rs
R100	experimental/sgl-router/tests/component/policies_reorg_admission.rs	sgl-router/tests/component/policies_reorg_admission.rs
R100	experimental/sgl-router/tests/component/policies_reorg_cache_aware.rs	sgl-router/tests/component/policies_reorg_cache_aware.rs
R100	experimental/sgl-router/tests/component/policies_reorg_load.rs	sgl-router/tests/component/policies_reorg_load.rs
R100	experimental/sgl-router/tests/component/policies_reorg_power_of_two.rs	sgl-router/tests/component/policies_reorg_power_of_two.rs
R100	experimental/sgl-router/tests/component/policies_reorg_session_aware.rs	sgl-router/tests/component/policies_reorg_session_aware.rs
R100	experimental/sgl-router/tests/component/policies_reorg_slo.rs	sgl-router/tests/component/policies_reorg_slo.rs
A	sgl-router/tests/component/tokenizer/deepseek.rs
R100	experimental/sgl-router/tests/component/tokenizer/kimi.rs	sgl-router/tests/component/tokenizer/kimi.rs
R100	experimental/sgl-router/tests/component/tokenizer/mod.rs	sgl-router/tests/component/tokenizer/mod.rs
R100	experimental/sgl-router/tests/component/tokenizer/parity.rs	sgl-router/tests/component/tokenizer/parity.rs
R100	experimental/sgl-router/tests/component/tokenizer/render_parity.rs	sgl-router/tests/component/tokenizer/render_parity.rs
R100	experimental/sgl-router/tests/component/workers/concurrent_state.rs	sgl-router/tests/component/workers/concurrent_state.rs
R100	experimental/sgl-router/tests/component/workers/manager.rs	sgl-router/tests/component/workers/manager.rs
R100	experimental/sgl-router/tests/component/workers/mod.rs	sgl-router/tests/component/workers/mod.rs
R100	experimental/sgl-router/tests/e2e/chat_completions/test_hicache_storage_tiers.py	sgl-router/tests/e2e/chat_completions/test_hicache_storage_tiers.py
R100	experimental/sgl-router/tests/e2e/chat_completions/test_load_based_policy.py	sgl-router/tests/e2e/chat_completions/test_load_based_policy.py
R100	experimental/sgl-router/tests/e2e/chat_completions/test_validation.py	sgl-router/tests/e2e/chat_completions/test_validation.py
R091	experimental/sgl-router/tests/e2e/conftest.py	sgl-router/tests/e2e/conftest.py
A	sgl-router/tests/e2e/infra/__init__.py
R091	experimental/sgl-router/tests/e2e/infra/gateway.py	sgl-router/tests/e2e/infra/gateway.py
R100	experimental/sgl-router/tests/e2e/infra/model_pool.py	sgl-router/tests/e2e/infra/model_pool.py
R100	experimental/sgl-router/tests/e2e/infra/model_specs.py	sgl-router/tests/e2e/infra/model_specs.py
R100	experimental/sgl-router/tests/e2e/infra/test_model_pool.py	sgl-router/tests/e2e/infra/test_model_pool.py
R100	experimental/sgl-router/tests/e2e/k8s_integration/Dockerfile.fake_kv_worker	sgl-router/tests/e2e/k8s_integration/Dockerfile.fake_kv_worker
R100	experimental/sgl-router/tests/e2e/k8s_integration/Dockerfile.fake_worker	sgl-router/tests/e2e/k8s_integration/Dockerfile.fake_worker
R081	experimental/sgl-router/tests/e2e/k8s_integration/Dockerfile.router	sgl-router/tests/e2e/k8s_integration/Dockerfile.router
R100	experimental/sgl-router/tests/e2e/k8s_integration/conftest.py	sgl-router/tests/e2e/k8s_integration/conftest.py
R100	experimental/sgl-router/tests/e2e/k8s_integration/fake_kv_worker.py	sgl-router/tests/e2e/k8s_integration/fake_kv_worker.py
R100	experimental/sgl-router/tests/e2e/k8s_integration/fake_worker.py	sgl-router/tests/e2e/k8s_integration/fake_worker.py
R100	experimental/sgl-router/tests/e2e/k8s_integration/manifests/kv-bootstrap.yaml	sgl-router/tests/e2e/k8s_integration/manifests/kv-bootstrap.yaml
R100	experimental/sgl-router/tests/e2e/k8s_integration/manifests/namespace.yaml	sgl-router/tests/e2e/k8s_integration/manifests/namespace.yaml
R100	experimental/sgl-router/tests/e2e/k8s_integration/manifests/rbac-cluster-scoped.yaml	sgl-router/tests/e2e/k8s_integration/manifests/rbac-cluster-scoped.yaml
R100	experimental/sgl-router/tests/e2e/k8s_integration/manifests/rbac.yaml	sgl-router/tests/e2e/k8s_integration/manifests/rbac.yaml
R100	experimental/sgl-router/tests/e2e/k8s_integration/manifests/router-cluster-scoped.yaml	sgl-router/tests/e2e/k8s_integration/manifests/router-cluster-scoped.yaml
R100	experimental/sgl-router/tests/e2e/k8s_integration/manifests/router.yaml	sgl-router/tests/e2e/k8s_integration/manifests/router.yaml
R100	experimental/sgl-router/tests/e2e/k8s_integration/requirements.txt	sgl-router/tests/e2e/k8s_integration/requirements.txt
R097	experimental/sgl-router/tests/e2e/k8s_integration/setup.sh	sgl-router/tests/e2e/k8s_integration/setup.sh
R100	experimental/sgl-router/tests/e2e/k8s_integration/test_conftest_helpers.py	sgl-router/tests/e2e/k8s_integration/test_conftest_helpers.py
R100	experimental/sgl-router/tests/e2e/k8s_integration/test_cross_namespace.py	sgl-router/tests/e2e/k8s_integration/test_cross_namespace.py
R100	experimental/sgl-router/tests/e2e/k8s_integration/test_discovery.py	sgl-router/tests/e2e/k8s_integration/test_discovery.py
R100	experimental/sgl-router/tests/e2e/k8s_integration/test_h2c_forwarding.py	sgl-router/tests/e2e/k8s_integration/test_h2c_forwarding.py
R100	experimental/sgl-router/tests/e2e/k8s_integration/test_kv_peer_bootstrap.py	sgl-router/tests/e2e/k8s_integration/test_kv_peer_bootstrap.py
R100	experimental/sgl-router/tests/e2e/k8s_integration/test_lifecycle.py	sgl-router/tests/e2e/k8s_integration/test_lifecycle.py
R100	experimental/sgl-router/tests/e2e/k8s_integration/test_reconciliation.py	sgl-router/tests/e2e/k8s_integration/test_reconciliation.py
R100	experimental/sgl-router/tests/e2e/k8s_integration/test_shutdown_drain.py	sgl-router/tests/e2e/k8s_integration/test_shutdown_drain.py
R091	experimental/sgl-router/tests/e2e/pyproject.toml	sgl-router/tests/e2e/pyproject.toml
R100	experimental/sgl-router/tests/e2e/requirements.txt	sgl-router/tests/e2e/requirements.txt
R100	experimental/sgl-router/tests/e2e/test_chat_smoke.py	sgl-router/tests/e2e/test_chat_smoke.py
R100	experimental/sgl-router/tests/e2e/test_gpu_allotment.py	sgl-router/tests/e2e/test_gpu_allotment.py
R100	experimental/sgl-router/tests/e2e/test_tokenize_smoke.py	sgl-router/tests/e2e/test_tokenize_smoke.py
R100	experimental/sgl-router/tests/fixtures/array_content_rendering.json	sgl-router/tests/fixtures/array_content_rendering.json
R100	experimental/sgl-router/tests/fixtures/chat_render_parity/glm-5.2.json	sgl-router/tests/fixtures/chat_render_parity/glm-5.2.json
R100	experimental/sgl-router/tests/fixtures/chat_render_parity/gpt-oss-20b.json	sgl-router/tests/fixtures/chat_render_parity/gpt-oss-20b.json
R100	experimental/sgl-router/tests/fixtures/chat_render_parity/minimax-m3.json	sgl-router/tests/fixtures/chat_render_parity/minimax-m3.json
R100	experimental/sgl-router/tests/fixtures/chat_render_parity/qwen3-8b.json	sgl-router/tests/fixtures/chat_render_parity/qwen3-8b.json
R100	experimental/sgl-router/tests/fixtures/chat_render_parity/qwen3.5-27b.json	sgl-router/tests/fixtures/chat_render_parity/qwen3.5-27b.json
R100	experimental/sgl-router/tests/fixtures/chat_render_parity/qwen3.8-27b.json	sgl-router/tests/fixtures/chat_render_parity/qwen3.8-27b.json
R100	experimental/sgl-router/tests/fixtures/kimi_k3.rs	sgl-router/tests/fixtures/kimi_k3.rs
R100	experimental/sgl-router/tests/fixtures/kimi_k3/README.md	sgl-router/tests/fixtures/kimi_k3/README.md
R100	experimental/sgl-router/tests/fixtures/kimi_k3/merges.txt	sgl-router/tests/fixtures/kimi_k3/merges.txt
R100	experimental/sgl-router/tests/fixtures/kimi_k3/prompts.json	sgl-router/tests/fixtures/kimi_k3/prompts.json
R100	experimental/sgl-router/tests/fixtures/kimi_k3/tokenizer_config.json	sgl-router/tests/fixtures/kimi_k3/tokenizer_config.json
R100	experimental/sgl-router/tests/fixtures/kv_events_hash_parity.json	sgl-router/tests/fixtures/kv_events_hash_parity.json
R100	experimental/sgl-router/tests/fixtures/router_v2_e2e_prefill_buckets.json	sgl-router/tests/fixtures/router_v2_e2e_prefill_buckets.json
R100	experimental/sgl-router/tests/fixtures/tiny_tokenizer.json	sgl-router/tests/fixtures/tiny_tokenizer.json
R100	experimental/sgl-router/tests/fixtures/tokenizer_config.json	sgl-router/tests/fixtures/tokenizer_config.json
R100	experimental/sgl-router/tests/fixtures/tokenizer_parity/deepseek-v3p2/long.json	sgl-router/tests/fixtures/tokenizer_parity/deepseek-v3p2/long.json
R100	experimental/sgl-router/tests/fixtures/tokenizer_parity/deepseek-v3p2/multi_turn_with_tools.json	sgl-router/tests/fixtures/tokenizer_parity/deepseek-v3p2/multi_turn_with_tools.json
R100	experimental/sgl-router/tests/fixtures/tokenizer_parity/deepseek-v3p2/short.json	sgl-router/tests/fixtures/tokenizer_parity/deepseek-v3p2/short.json
R100	experimental/sgl-router/tests/fixtures/tokenizer_parity/deepseek-v3p2/special_token_heavy.json	sgl-router/tests/fixtures/tokenizer_parity/deepseek-v3p2/special_token_heavy.json
R100	experimental/sgl-router/tests/fixtures/tokenizer_parity/gpt-oss-20b/long.json	sgl-router/tests/fixtures/tokenizer_parity/gpt-oss-20b/long.json
R100	experimental/sgl-router/tests/fixtures/tokenizer_parity/gpt-oss-20b/multi_turn_with_tools.json	sgl-router/tests/fixtures/tokenizer_parity/gpt-oss-20b/multi_turn_with_tools.json
R100	experimental/sgl-router/tests/fixtures/tokenizer_parity/gpt-oss-20b/short.json	sgl-router/tests/fixtures/tokenizer_parity/gpt-oss-20b/short.json
R100	experimental/sgl-router/tests/fixtures/tokenizer_parity/gpt-oss-20b/special_token_heavy.json	sgl-router/tests/fixtures/tokenizer_parity/gpt-oss-20b/special_token_heavy.json
R100	experimental/sgl-router/tests/fixtures/tokenizer_parity/qwen3-30b/long.json	sgl-router/tests/fixtures/tokenizer_parity/qwen3-30b/long.json
R100	experimental/sgl-router/tests/fixtures/tokenizer_parity/qwen3-30b/multi_turn_with_tools.json	sgl-router/tests/fixtures/tokenizer_parity/qwen3-30b/multi_turn_with_tools.json
R100	experimental/sgl-router/tests/fixtures/tokenizer_parity/qwen3-30b/short.json	sgl-router/tests/fixtures/tokenizer_parity/qwen3-30b/short.json
R100	experimental/sgl-router/tests/fixtures/tokenizer_parity/qwen3-30b/special_token_heavy.json	sgl-router/tests/fixtures/tokenizer_parity/qwen3-30b/special_token_heavy.json
R099	experimental/sgl-router/tests/proxy/bucket_routing.rs	sgl-router/tests/proxy/bucket_routing.rs
R100	experimental/sgl-router/tests/proxy/cache_aware_input_ids.rs	sgl-router/tests/proxy/cache_aware_input_ids.rs
R100	experimental/sgl-router/tests/proxy/chat_routing.rs	sgl-router/tests/proxy/chat_routing.rs
R100	experimental/sgl-router/tests/proxy/chat_routing/cancellation.rs	sgl-router/tests/proxy/chat_routing/cancellation.rs
R098	experimental/sgl-router/tests/proxy/chat_routing/reorg.rs	sgl-router/tests/proxy/chat_routing/reorg.rs
R100	experimental/sgl-router/tests/proxy/chat_routing/reorg/session_aware.rs	sgl-router/tests/proxy/chat_routing/reorg/session_aware.rs
R100	experimental/sgl-router/tests/proxy/chat_routing/reorg/slo.rs	sgl-router/tests/proxy/chat_routing/reorg/slo.rs
R100	experimental/sgl-router/tests/proxy/common/cache_aware_fixture.rs	sgl-router/tests/proxy/common/cache_aware_fixture.rs
R100	experimental/sgl-router/tests/proxy/common/mock_worker.rs	sgl-router/tests/proxy/common/mock_worker.rs
R100	experimental/sgl-router/tests/proxy/common/mod.rs	sgl-router/tests/proxy/common/mod.rs
R100	experimental/sgl-router/tests/proxy/common/streaming.rs	sgl-router/tests/proxy/common/streaming.rs
R073	experimental/sgl-router/tests/proxy/dp_rank_routing.rs	sgl-router/tests/proxy/dp_rank_routing.rs
R100	experimental/sgl-router/tests/proxy/embeddings_routing.rs	sgl-router/tests/proxy/embeddings_routing.rs
R100	experimental/sgl-router/tests/proxy/external_indexer_routing.rs	sgl-router/tests/proxy/external_indexer_routing.rs
R100	experimental/sgl-router/tests/proxy/failover.rs	sgl-router/tests/proxy/failover.rs
R100	experimental/sgl-router/tests/proxy/generate_routing.rs	sgl-router/tests/proxy/generate_routing.rs
R100	experimental/sgl-router/tests/proxy/graceful_shutdown.rs	sgl-router/tests/proxy/graceful_shutdown.rs
R094	experimental/sgl-router/tests/proxy/grpc_routing.rs	sgl-router/tests/proxy/grpc_routing.rs
R100	experimental/sgl-router/tests/proxy/h2c_forward.rs	sgl-router/tests/proxy/h2c_forward.rs
R100	experimental/sgl-router/tests/proxy/header_forwarding.rs	sgl-router/tests/proxy/header_forwarding.rs
R100	experimental/sgl-router/tests/proxy/inbound_h2c.rs	sgl-router/tests/proxy/inbound_h2c.rs
R100	experimental/sgl-router/tests/proxy/main.rs	sgl-router/tests/proxy/main.rs
R100	experimental/sgl-router/tests/proxy/messages_api.rs	sgl-router/tests/proxy/messages_api.rs
R100	experimental/sgl-router/tests/proxy/openai_generate.rs	sgl-router/tests/proxy/openai_generate.rs
R100	experimental/sgl-router/tests/proxy/pd_bootstrap_injection.rs	sgl-router/tests/proxy/pd_bootstrap_injection.rs
R100	experimental/sgl-router/tests/proxy/pd_bootstrap_injection/reliability.rs	sgl-router/tests/proxy/pd_bootstrap_injection/reliability.rs
R100	experimental/sgl-router/tests/proxy/pd_pool_isolation.rs	sgl-router/tests/proxy/pd_pool_isolation.rs
R100	experimental/sgl-router/tests/proxy/pd_protocol_binding.rs	sgl-router/tests/proxy/pd_protocol_binding.rs
R100	experimental/sgl-router/tests/proxy/radix_tree_routing.rs	sgl-router/tests/proxy/radix_tree_routing.rs
R100	experimental/sgl-router/tests/proxy/reorg_launch.rs	sgl-router/tests/proxy/reorg_launch.rs
R100	experimental/sgl-router/tests/proxy/rerank_routing.rs	sgl-router/tests/proxy/rerank_routing.rs
R100	experimental/sgl-router/tests/proxy/retry.rs	sgl-router/tests/proxy/retry.rs
R100	experimental/sgl-router/tests/proxy/roundrobin_input_ids.rs	sgl-router/tests/proxy/roundrobin_input_ids.rs
R100	experimental/sgl-router/tests/proxy/sampling_overrides.rs	sgl-router/tests/proxy/sampling_overrides.rs
R100	experimental/sgl-router/tests/proxy/shared_prefill_admission.rs	sgl-router/tests/proxy/shared_prefill_admission.rs
R100	experimental/sgl-router/tests/proxy/sticky_input_ids.rs	sgl-router/tests/proxy/sticky_input_ids.rs
R100	experimental/sgl-router/tests/proxy/sticky_routing.rs	sgl-router/tests/proxy/sticky_routing.rs
R068	experimental/sgl-router/tests/proxy/timeout.rs	sgl-router/tests/proxy/timeout.rs
R095	experimental/sgl-router/tests/scripts/generate_array_content_fixture.py	sgl-router/tests/scripts/generate_array_content_fixture.py
R082	experimental/sgl-router/tests/scripts/generate_chat_render_parity.py	sgl-router/tests/scripts/generate_chat_render_parity.py
R097	experimental/sgl-router/tests/scripts/generate_kimi_parity.py	sgl-router/tests/scripts/generate_kimi_parity.py
R098	experimental/sgl-router/tests/scripts/generate_kv_events_hash_parity.py	sgl-router/tests/scripts/generate_kv_events_hash_parity.py
R097	experimental/sgl-router/tests/scripts/generate_parity_fixtures.py	sgl-router/tests/scripts/generate_parity_fixtures.py
M	test/README.md
A	test/manual/ascend/llm_models/test_npu_glm_5_1_w4a8_draft_prefetch.py
A	test/manual/ascend/test_llguidance_npu_mask.py
A	test/manual/ascend/test_npu_prefill_cp_backend.py
M	test/manual/attention/test_trtllm_mla_backend.py
M	test/manual/kernels/attention/check_dsa_dequant_k_cache.py
M	test/manual/kernels/attention/check_dsa_transform_index.py
M	test/manual/kernels/attention/check_dsv4_dequant_k_cache.py
A	test/manual/kernels/attention/test_flash_fwd_mla_sm100.py
R097	python/sglang/srt/layers/attention/minimax_sparse_ops/tests/test_msa_fp8_parity.py	test/manual/kernels/attention/test_minimax_msa_fp8_parity.py
A	test/manual/kernels/bench_dense_swiglu_quant.py
M	test/manual/kernels/benchmark/attention/bench_gdn_replayssm_fold.py
A	test/manual/kernels/diffusion/bench_swiglu_fp8_rowwise.py
R098	test/manual/layers/attention/dsa/test_get_k_scale_triton_kernel.py	test/manual/kernels/kvcache/test_dsa_get_k_scale_triton_kernel.py
R099	test/manual/layers/attention/dsa/test_index_buf_accessor.py	test/manual/kernels/kvcache/test_dsa_index_k_cache.py
R094	test/manual/kernels/attention/check_dsa_quant_k_cache.py	test/manual/kernels/quantization/check_dsa_quant_k_cache.py
R097	test/manual/layers/attention/dsa/test_act_quant_triton.py	test/manual/kernels/quantization/test_act_quant_triton.py
A	test/manual/kv_transfer/test_disaggregation_staging_nixl_xpu.py
A	test/manual/layers/test_engram_peer_shards.py
A	test/manual/mem_cache/gen_unified_layout_golden.py
A	test/manual/mem_cache/unified_trace_compare.py
A	test/manual/mem_cache/unified_trace_scenarios/float_park_and_recentre.json
A	test/manual/mem_cache/unified_trace_scenarios/float_relocation_closed_gate.json
A	test/manual/mem_cache/unified_trace_scenarios/hicache_reservation.json
A	test/manual/mem_cache/unified_trace_scenarios/pending_reuse_behind_gate.json
A	test/manual/mem_cache/unified_trace_scenarios/segment_frees.json
A	test/manual/mem_cache/unified_trace_scenarios/segment_frees_state_chain.json
A	test/manual/mem_cache/unified_trace_scenarios/state_slots_exhausted.json
A	test/manual/mem_cache/unified_trace_scenarios/tree_joint_reclaim.json
A	test/manual/mem_cache/unified_trace_scenarios/window_retirement_p64.json
A	test/manual/mem_cache/unified_transfer_compare.py
M	test/manual/minimax_m3/test_npu_topk.py
A	test/manual/parser/gen_inkling_tmlv0_golden.py
M	test/manual/test_dsv41_ragged_prefill.py
M	test/manual/test_kda_target_verify.py
M	test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py
M	test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py
M	test/registered/amd/accuracy/mi35x/test_qwen38_mxfp4_eval_mi35x.py
M	test/registered/amd/disaggregation/test_mori_backend_basic.py
M	test/registered/amd/perf/mi35x/test_qwen35_fp8_ar_fusion_mi35x.py
A	test/registered/amd/test_aiter_vectorized_5d_decode.py
M	test/registered/amd/test_deepseek_v32_basic.py
M	test/registered/amd/test_deepseek_v3_basic.py
M	test/registered/amd/test_deepseek_v3_mtp.py
M	test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py
M	test/registered/amd/test_deepseek_v4_pro_fp4_dspark.py
M	test/registered/amd/test_dsv4_hip_bcg_metadata.py
A	test/registered/amd/test_fp8_gemm_gfx1250.py
A	test/registered/amd/test_fp8_moe_triton_runner_with_aiter.py
M	test/registered/amd/test_kimi_k2_instruct.py
M	test/registered/amd/test_moriep_small.py
A	test/registered/amd/test_moriep_small_extra.py
A	test/registered/amd/test_qwen35_gdn_online_mxfp4.py
M	test/registered/amd/test_qwen3_coder_next_8gpu.py
A	test/registered/amd/test_smallm_fp8_blockscale_gfx950.py
M	test/registered/amd/test_vattn_segplan_mi35x.py
R100	test/srt/test_aiter_cross_layer_kv_guard.py	test/registered/attention/test_aiter_cross_layer_kv_guard.py
A	test/registered/attention/test_aiter_fp8_kv_scales_paths.py
M	test/registered/attention/test_aiter_fp8_q_unified_attention.py
M	test/registered/attention/test_aiter_gluon_h12_fp8.py
R070	test/srt/test_aiter_swa_kv_pool_resolver.py	test/registered/attention/test_aiter_swa_kv_pool_resolver.py
M	test/registered/attention/test_chunk_gated_delta_rule.py
M	test/registered/attention/test_create_kvindices.py
M	test/registered/attention/test_deterministic.py
A	test/registered/attention/test_dflash_mla_backends.py
M	test/registered/attention/test_flash_attention_4.py
M	test/registered/attention/test_fused_fp8_kv_write.py
M	test/registered/attention/test_gdn_fused_split_head_ratios.py
M	test/registered/attention/test_gdn_noncontiguous_stride.py
M	test/registered/attention/test_gdn_prefill_cutedsl.py
M	test/registered/attention/test_gdn_prefill_layout.py
M	test/registered/attention/test_gemma4_swa_triton_oob_regression.py
M	test/registered/attention/test_gemma4_unified_swa_virtual_ids.py
M	test/registered/attention/test_hybrid_attn_backend.py
M	test/registered/attention/test_kda_kernels.py
M	test/registered/attention/test_kda_prefill_cutedsl.py
M	test/registered/attention/test_kda_prefill_flashinfer.py
M	test/registered/attention/test_normal_decode_set_metadata.py
M	test/registered/attention/test_qwen35_deterministic.py
M	test/registered/attention/test_torch_native_attention_backend.py
M	test/registered/attention/test_triton_attention_kernels.py
M	test/registered/attention/test_triton_sliding_window.py
M	test/registered/attention/test_trtllm_mha_encoder_only.py
M	test/registered/attention/test_trtllm_mha_graph_metadata.py
M	test/registered/attention/test_trtllm_mha_page_table.py
M	test/registered/attention/test_trtllm_mha_stride_parity.py
M	test/registered/attention/test_trtllm_mla_stride_parity.py
M	test/registered/attention/test_unified_memory_deterministic.py
M	test/registered/attention/test_verify_splitkv.py
M	test/registered/attention/unittests/KNOWN_FAILURES.md
M	test/registered/attention/unittests/dense/test_extend_init_contract.py
M	test/registered/attention/unittests/dense/test_fa4.py
M	test/registered/attention/unittests/dense/test_flashinfer.py
M	test/registered/attention/unittests/dense/test_flex_attention.py
M	test/registered/attention/unittests/dense/test_hybrid_attn.py
M	test/registered/attention/unittests/dense/test_tbo.py
M	test/registered/attention/unittests/dense/test_torch_native.py
M	test/registered/attention/unittests/dense/test_triton.py
M	test/registered/attention/unittests/dense/test_trtllm_mha.py
M	test/registered/attention/unittests/dsa/README.md
M	test/registered/attention/unittests/dsa/test_dsa.py
M	test/registered/attention/unittests/dsv4/test_deepseek_v4.py
M	test/registered/attention/unittests/dsv4/test_dsv41_bcg_hip.py
M	test/registered/attention/unittests/dsv4/test_dsv41_fused_compress.py
M	test/registered/attention/unittests/dsv4/test_hip_flash_mla_aiter_sparse.py
M	test/registered/attention/unittests/gdn/test_flashinfer.py
M	test/registered/attention/unittests/gdn/test_gdn_cutedsl_ring_verify.py
M	test/registered/attention/unittests/gdn/test_gdn_replayssm_spec_fold.py
M	test/registered/attention/unittests/gdn/test_linear_replayssm_decode.py
M	test/registered/attention/unittests/gdn/test_torch_native.py
M	test/registered/attention/unittests/gdn/test_triton.py
M	test/registered/attention/unittests/hybrid_linear/test_flashinfer_mla_chunk_metadata.py
M	test/registered/attention/unittests/hybrid_linear/test_kda_fused_accept_indices.py
M	test/registered/attention/unittests/hybrid_linear/test_verify_buffer_fixup_hook.py
M	test/registered/attention/unittests/kda/test_triton.py
M	test/registered/attention/unittests/lightning/test_triton.py
M	test/registered/attention/unittests/mamba/test_mamba2.py
M	test/registered/attention/unittests/mamba/test_replay_state_indices_validator.py
M	test/registered/attention/unittests/mla/README.md
M	test/registered/attention/unittests/mla/test_flashinfer.py
M	test/registered/attention/unittests/mla/test_flashmla.py
D	test/registered/attention/unittests/mla/test_tokenspeed_mla.py
M	test/registered/attention/unittests/mla/test_triton.py
M	test/registered/attention/unittests/swa/test_flashinfer.py
M	test/registered/attention/unittests/swa/test_torch_native.py
M	test/registered/attention/unittests/swa/test_triton.py
M	test/registered/backends/test_deepseek_v3_fp4_cutedsl_moe.py
M	test/registered/backends/test_flashinfer_fusion_preflight.py
M	test/registered/backends/test_torch_compile.py
M	test/registered/basic_perf/test_eagle3_latency.py
M	test/registered/basic_perf/test_embeddings_api.py
M	test/registered/basic_perf/test_lora_latency.py
M	test/registered/basic_perf/test_moe_throughput.py
M	test/registered/basic_perf/test_pp_throughput.py
M	test/registered/basic_perf/test_score_api.py
M	test/registered/basic_perf/test_serving_latency.py
M	test/registered/basic_perf/test_serving_throughput.py
M	test/registered/basic_perf/test_torch_compile_throughput.py
M	test/registered/basic_perf/test_vlm_serving_aiter.py
M	test/registered/basic_perf/test_vlm_serving_fa3.py
M	test/registered/basic_perf/test_vlm_serving_flashinfer.py
M	test/registered/bench_fn/test_benchmark_datasets_api.py
M	test/registered/bench_fn/test_steady_state_benchmark.py
M	test/registered/chunked_prefill/test_mm_chunked_embedding_unit.py
M	test/registered/chunked_prefill/test_scripted_core_1gpu.py
M	test/registered/chunked_prefill/test_scripted_core_4gpu.py
M	test/registered/chunked_prefill/test_scripted_swa_1gpu.py
M	test/registered/constrained_decoding/test_constrained_decoding.py
M	test/registered/core/test_basic_sanity.py
M	test/registered/core/test_basic_sanity_dflash.py
M	test/registered/core/test_basic_sanity_dspark.py
M	test/registered/core/test_basic_sanity_eagle3.py
M	test/registered/core/test_engine_child_pids.py
M	test/registered/core/test_hidden_states.py
M	test/registered/core/test_no_extra_forked_cuda_context.py
M	test/registered/core/test_request_queue_validation.py
M	test/registered/core/test_srt_empty_deps.py
M	test/registered/core/test_srt_endpoint.py
M	test/registered/core/test_srt_engine.py
M	test/registered/cp/test_cp_strategy_unit.py
M	test/registered/cp/test_deepseek_v3_cp_single_node.py
M	test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py
M	test/registered/cp/test_dsa_prefill_cp.py
D	test/registered/cp/test_glm53_flash_cp.py
M	test/registered/cp/test_gpt_oss_4gpu_mxfp4_cp.py
M	test/registered/cp/test_gqa_prefill_cp.py
M	test/registered/cpu/test_norm.py
M	test/registered/cpu/test_topk.py
M	test/registered/cuda_graph/breakable/test_bcg_with_lora.py
M	test/registered/cuda_graph/breakable/test_bcg_with_speculative_decoding.py
M	test/registered/cuda_graph/breakable/test_breakable_cuda_graph.py
M	test/registered/cuda_graph/breakable/test_deepseek_dual_stream_outputs.py
M	test/registered/cuda_graph/full_prefill/test_full_cuda_graph_prefill.py
M	test/registered/dcp/test_dcp_layout_unit.py
M	test/registered/dcp/test_dsa_dcp_attention.py
M	test/registered/dcp/test_dsv31_dcp8_gsm8k.py
M	test/registered/dcp/test_glm53_dcp2.py
M	test/registered/dcp/test_kimi_linear_dcp4.py
M	test/registered/dcp/test_kimi_linear_dcp_dspark4.py
M	test/registered/dcp/test_trtllm_mla_family_dcp_metadata.py
M	test/registered/debug_utils/test_crash_dump.py
M	test/registered/debug_utils/test_tensor_dump_forward_hook.py
M	test/registered/disaggregation/test_disaggregation_aarch64.py
M	test/registered/disaggregation/test_disaggregation_basic.py
M	test/registered/disaggregation/test_disaggregation_chunked_prefill_abort.py
M	test/registered/disaggregation/test_disaggregation_decode_radix_cache.py
M	test/registered/disaggregation/test_disaggregation_decode_radix_cache_swa.py
M	test/registered/disaggregation/test_disaggregation_different_tp.py
A	test/registered/disaggregation/test_disaggregation_different_tp_deepseek_v3.py
M	test/registered/disaggregation/test_disaggregation_dp_attention.py
M	test/registered/disaggregation/test_disaggregation_dsv4.py
M	test/registered/disaggregation/test_disaggregation_dwdp_gpt_oss.py
M	test/registered/disaggregation/test_disaggregation_hisparse.py
M	test/registered/disaggregation/test_disaggregation_hybrid_attention.py
M	test/registered/disaggregation/test_disaggregation_inkling_mxfp8.py
M	test/registered/disaggregation/test_disaggregation_kimi_linear.py
A	test/registered/disaggregation/test_disaggregation_kv_checksum.py
M	test/registered/disaggregation/test_disaggregation_nixl.py
M	test/registered/disaggregation/test_disaggregation_optimistic_prefill.py
M	test/registered/disaggregation/test_disaggregation_pp.py
M	test/registered/disaggregation/test_disaggregation_rust_server.py
M	test/registered/disaggregation/test_disaggregation_unified_memory.py
M	test/registered/disaggregation/test_epd_disaggregation.py
M	test/registered/dllm/test_dllm_batching_fdfo.py
M	test/registered/dllm/test_dllm_fdfo_joint_threshold.py
M	test/registered/dllm/test_llada2_mini_amd.py
M	test/registered/dp_attn/test_dp_attention.py
M	test/registered/dp_attn/test_dp_attention_bcg_kl.py
A	test/registered/dp_attn/test_dp_attention_deepseek_v3.py
A	test/registered/e2e/diffusion/test_async_video_serving.py
A	test/registered/e2e/disaggregation/test_disaggregation_kv_cache_sharding.py
M	test/registered/e2e/disaggregation/test_disaggregation_lora.py
M	test/registered/e2e/disaggregation/test_disaggregation_swa_different_tp.py
M	test/registered/e2e/disaggregation/test_disaggregation_unified_memory_swa.py
M	test/registered/e2e/disaggregation/test_disaggregation_unified_memory_tri.py
A	test/registered/e2e/disaggregation/test_glm53_cp_bcg_dflash.py
M	test/registered/e2e/disaggregation/test_kimi_linear_pd_dcp4.py
M	test/registered/e2e/disaggregation/test_prefill_complete_allocation.py
M	test/registered/e2e/dp_attn/test_dp_attention_local_control_broadcast.py
M	test/registered/e2e/dsv4/test_dsv4_unified_fp8_compress_store.py
M	test/registered/e2e/dsv4/test_dsv4_unified_fp8_scatter.py
M	test/registered/e2e/entrypoints/test_engine_multi_tokenizer.py
M	test/registered/e2e/gdn/test_qwen35_gdn_multi_item_scoring.py
M	test/registered/e2e/hicache/test_hicache_storage_lora.py
M	test/registered/e2e/hicache/test_hicache_unified_memory.py
A	test/registered/e2e/hicache/test_hicache_xpu_mla_kernels.py
A	test/registered/e2e/hicache/test_hicache_xpu_no_storage.py
M	test/registered/e2e/layers/test_logprob_memory_distributed.py
M	test/registered/e2e/lora/test_lora_moe_prefill_cuda_graph.py
A	test/registered/e2e/model_executor/test_decoder_tail_moe_bound.py
A	test/registered/e2e/models/test_beam_10_b200_pd.py
M	test/registered/e2e/models/test_compressed_tensors_models.py
M	test/registered/e2e/models/test_deepseek_v3_fp4.py
M	test/registered/e2e/models/test_deepseek_v3_mtp.py
R075	test/registered/e2e/models/test_deepseek_v41_flash_gb300_dp4ep4.py	test/registered/e2e/models/test_deepseek_v41_flash_gb300_dp2ep2.py
M	test/registered/e2e/models/test_deepseek_v41_flash_gb300_encoder_bcg.py
R057	test/registered/e2e/models/test_deepseek_v41_flash_gb300_tp4ep4.py	test/registered/e2e/models/test_deepseek_v41_flash_gb300_tp2ep2.py
M	test/registered/e2e/models/test_deepseek_v41_flash_gb300_trtllm_bounded_bcg.py
M	test/registered/e2e/models/test_deepseek_v4_flash_fp4_b200.py
M	test/registered/e2e/models/test_deepseek_v4_flash_fp4_b200_trtllm.py
M	test/registered/e2e/models/test_deepseek_v4_flash_fp4_h200.py
M	test/registered/e2e/models/test_deepseek_v4_flash_fp4_megamoe_b200.py
A	test/registered/e2e/models/test_deepseek_v4_flash_fp4_tbo_h200.py
M	test/registered/e2e/models/test_dsa_glm52_nvfp4_dp_mtp.py
M	test/registered/e2e/models/test_dsa_glm52_nvfp4_tp_mtp.py
A	test/registered/e2e/models/test_dsa_glm52_pd_kv_shard_hicache.py
M	test/registered/e2e/models/test_dsa_glm52_pd_mtp_cp_layersplit.py
M	test/registered/e2e/models/test_gemma4_fp8_per_expert_loading.py
M	test/registered/e2e/models/test_generation_models.py
M	test/registered/e2e/models/test_glm53_flash_b200.py
M	test/registered/e2e/models/test_glm53_flash_h200_high_throughput.py
M	test/registered/e2e/models/test_glm53_flash_h200_low_latency.py
M	test/registered/e2e/models/test_gpt_oss_4gpu_mxfp4.py
M	test/registered/e2e/models/test_gpt_oss_sm120.py
M	test/registered/e2e/models/test_inkling.py
M	test/registered/e2e/models/test_inkling_small_nvfp4.py
M	test/registered/e2e/models/test_inkling_unified.py
M	test/registered/e2e/models/test_kimi_k3_b300.py
M	test/registered/e2e/models/test_kimi_k3_b300_low_latency.py
M	test/registered/e2e/models/test_kimi_linear_models.py
M	test/registered/e2e/models/test_kimi_linear_unified_memory_dcp_blackwell.py
M	test/registered/e2e/models/test_layernorm_sp.py
M	test/registered/e2e/models/test_mimo_v2.py
M	test/registered/e2e/models/test_mimo_v2_flash.py
M	test/registered/e2e/models/test_minimax_m25_basic.py
M	test/registered/e2e/models/test_muse_spark_12_b300_pd.py
A	test/registered/e2e/models/test_muse_spark_unified_memory.py
M	test/registered/e2e/models/test_nvidia_nemotron_3_nano.py
M	test/registered/e2e/models/test_nvidia_nemotron_3_super_bf16_mtp.py
M	test/registered/e2e/models/test_qwen35_fp4_mtp.py
M	test/registered/e2e/models/test_qwen3_next_models.py
M	test/registered/e2e/models/test_qwen3_next_models_extra.py
M	test/registered/e2e/models/test_qwen3_next_models_mtp.py
M	test/registered/e2e/models/test_qwen4_exp_models.py
M	test/registered/e2e/models/test_step3p5_flash_chain_mtp.py
M	test/registered/e2e/models/test_transformers_backend_eval.py
M	test/registered/e2e/models/test_transformers_models.py
M	test/registered/e2e/models/test_vlm_models.py
M	test/registered/e2e/models_large/test_deepseek_v32_indexcache.py
A	test/registered/e2e/moe/test_paged_experts.py
A	test/registered/e2e/moe/test_paged_experts_gpt_oss.py
M	test/registered/e2e/openai_server/test_watermark.py
M	test/registered/e2e/pp/test_pp_spec.py
M	test/registered/e2e/pp/test_pp_spec_embed_scan.py
M	test/registered/e2e/quantization/test_autoround_quantization.py
M	test/registered/e2e/quantization/test_awq.py
M	test/registered/e2e/quantization/test_fp8kv_triton.py
M	test/registered/e2e/quantization/test_gguf.py
M	test/registered/e2e/quantization/test_gptqmodel_dynamic.py
M	test/registered/e2e/quantization/test_modelopt_fp8.py
M	test/registered/e2e/quantization/test_nvfp4_gemm_sm120.py
M	test/registered/e2e/quantization/test_w8a8_quantization.py
M	test/registered/e2e/radix_cache/test_mamba_extra_buffer_spec_grammar.py
M	test/registered/e2e/rust_server/test_mm_transport_1node_cpu.py
M	test/registered/e2e/rust_server/test_mm_transport_1node_cuda_ipc.py
M	test/registered/e2e/rust_server/test_mm_transport_1node_tp2_shm.py
M	test/registered/e2e/rust_server/test_mm_transport_2node_cpu.py
M	test/registered/e2e/scoring/test_setwise_scoring.py
M	test/registered/e2e/speculative/test_dflash_domino.py
A	test/registered/e2e/speculative/test_dflash_dp_attention.py
A	test/registered/e2e/speculative/test_eagle_reject_sampling_deterministic.py
A	test/registered/e2e/speculative/test_multi_layer_eagle_reject_sampling_deterministic.py
M	test/registered/e2e/xpu/test_dspark_intel_xpu.py
M	test/registered/e2e/xpu/test_xpu_fp8_linear.py
M	test/registered/ep/test_deepep_large.py
M	test/registered/ep/test_deepep_small.py
M	test/registered/ep/test_deepep_small_extra.py
M	test/registered/ep/test_routed_experts_dp_readback.py
M	test/registered/ep/test_tbo_shared_experts_fusion.py
M	test/registered/eplb/test_lplb_distributed.py
M	test/registered/expert_pack/test_expert_pack_mxfp4.py
M	test/registered/expert_pack/test_expert_pack_runtime.py
M	test/registered/expert_pack/test_kimi_k3_gguf.py
M	test/registered/function_call/test_kimik3_detector.py
D	test/registered/gb300/test_kimi_k25_nvfp4_dp.py
D	test/registered/gb300/test_kimi_k25_nvfp4_tp.py
D	test/registered/gb300/test_qwen35_fp8_dp.py
D	test/registered/gb300/test_qwen35_fp8_tp.py
M	test/registered/hicache/test_dsa_host_release.py
A	test/registered/hicache/test_hicache_l3_object_layout.py
M	test/registered/hicache/test_hicache_spec_file_storage.py
M	test/registered/hicache/test_hicache_spec_mooncake_storage.py
M	test/registered/hicache/test_hicache_storage.py
M	test/registered/hicache/test_hicache_storage_3fs_backend.py
M	test/registered/hicache/test_hicache_storage_file_backend.py
M	test/registered/hicache/test_hicache_storage_mooncake_backend.py
M	test/registered/hicache/test_hicache_storage_runtime_attach_detach.py
M	test/registered/hicache/test_hicache_storage_umbp_backend.py
M	test/registered/hicache/test_hicache_variants.py
M	test/registered/hicache/test_k_only_host_release.py
A	test/registered/hicache/test_page_unified_qwen3_30b_accuracy.py
A	test/registered/hicache/test_page_unified_storage.py
M	test/registered/hicache/test_pp_with_hicache.py
M	test/registered/hicache/test_qwen35_hicache.py
M	test/registered/input_embedding/test_input_embeds_chunked.py
R093	test/registered/kernels/benchmark/elementwise/bench_bias_gelu.py	test/registered/kernels/benchmark/activation/bench_bias_gelu.py
M	test/registered/kernels/benchmark/attention/bench_dsa_triton_sparse_mla_prefill.py
M	test/registered/kernels/benchmark/attention/bench_fp4_index_logits.py
M	test/registered/kernels/benchmark/attention/bench_minimax_qknorm_rope.py
M	test/registered/kernels/benchmark/attention/bench_online_c128_mtp.py
M	test/registered/kernels/benchmark/attention/bench_sparse_mla_q8kv8_prefill_sm90.py
M	test/registered/kernels/benchmark/communication/bench_custom_all_reduce.py
M	test/registered/kernels/benchmark/communication/bench_tp_qknorm.py
R098	test/registered/kernels/benchmark/speculative/bench_ngram_compute_decode.py	test/registered/kernels/benchmark/embeddings/bench_ngram_compute_decode.py
R097	test/registered/kernels/benchmark/speculative/bench_ngram_update_token_table.py	test/registered/kernels/benchmark/embeddings/bench_ngram_update_token_table.py
M	test/registered/kernels/benchmark/gemm/bench_kda_fp8_skinny_gemm.py
A	test/registered/kernels/benchmark/memory/bench_small_field_copy.py
A	test/registered/kernels/benchmark/moe/bench_gguf_moe_weights.py
A	test/registered/kernels/benchmark/moe/bench_shuffle_weighted_sum.py
M	test/registered/kernels/benchmark/quantization/bench_per_token_group_quant.py
M	test/registered/kernels/ops/activation/test_activation.py
R095	test/registered/kernels/ops/elementwise/test_bias_gelu.py	test/registered/kernels/ops/activation/test_bias_gelu.py
M	test/registered/kernels/ops/activation/test_fused_op_gpu_parity.py
A	test/registered/kernels/ops/activation/test_swiglu_mxfp8_swizzled.py
A	test/registered/kernels/ops/attention/dsv4/test_flashmla_v41.py
M	test/registered/kernels/ops/attention/dsv4/test_mega_attention.py
M	test/registered/kernels/ops/attention/dsv4/test_swapab_attention.py
M	test/registered/kernels/ops/attention/dsv4/test_v41_kv_store.py
M	test/registered/kernels/ops/attention/qsa/test_qsa.py
A	test/registered/kernels/ops/attention/qsa/test_qsa_determinism.py
M	test/registered/kernels/ops/attention/qsa/test_qsa_fused_kv_prepare.py
M	test/registered/kernels/ops/attention/qsa/test_qsa_indexer.py
A	test/registered/kernels/ops/attention/test_attn_res_mix_large_tokens.py
M	test/registered/kernels/ops/attention/test_combine_topk_swa_indices.py
M	test/registered/kernels/ops/attention/test_cutedsl_gdn.py
M	test/registered/kernels/ops/attention/test_dcp_lse_combine.py
M	test/registered/kernels/ops/attention/test_dense_prefill_indexer.py
M	test/registered/kernels/ops/attention/test_dsa_indexer.py
M	test/registered/kernels/ops/attention/test_dsa_kpool_metadata_fusion.py
M	test/registered/kernels/ops/attention/test_dsa_metadata.py
M	test/registered/kernels/ops/attention/test_dsa_sink_pad_cache.py
M	test/registered/kernels/ops/attention/test_dsa_transform_index.py
M	test/registered/kernels/ops/attention/test_dsv32_indexer_fusion.py
M	test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py
A	test/registered/kernels/ops/attention/test_dsv4_dequant_k_cache.py
M	test/registered/kernels/ops/attention/test_dsv4_indexer_quant.py
M	test/registered/kernels/ops/attention/test_dsv4_q_rope_store.py
M	test/registered/kernels/ops/attention/test_dsv4_trtllm_q_metadata.py
M	test/registered/kernels/ops/attention/test_flash_attention_4_sm103.py
M	test/registered/kernels/ops/attention/test_flash_mla_backends.py
A	test/registered/kernels/ops/attention/test_flashinfer_host_plan.py
A	test/registered/kernels/ops/attention/test_flashinfer_window_verify.py
M	test/registered/kernels/ops/attention/test_flashmla_sched_meta.py
M	test/registered/kernels/ops/attention/test_fp4_indexer.py
M	test/registered/kernels/ops/attention/test_fp4_indexer_hip.py
M	test/registered/kernels/ops/attention/test_fused_kda_conv_recurrent_verify.py
M	test/registered/kernels/ops/attention/test_fused_metadata_copy.py
M	test/registered/kernels/ops/attention/test_fused_verify_triton_gdn.py
A	test/registered/kernels/ops/attention/test_gdn_prefill_graph.py
M	test/registered/kernels/ops/attention/test_inkling_attn_prologue_tau.py
M	test/registered/kernels/ops/attention/test_inkling_checkpoint_indices.py
A	test/registered/kernels/ops/attention/test_kda_fused_decode_gfx950.py
M	test/registered/kernels/ops/attention/test_kda_fused_verify_backend.py
A	test/registered/kernels/ops/attention/test_kda_gfx950_prefill.py
M	test/registered/kernels/ops/attention/test_kda_helion.py
M	test/registered/kernels/ops/attention/test_kda_mtp_cutedsl_replayssm_ring.py
M	test/registered/kernels/ops/attention/test_kda_prefill.py
M	test/registered/kernels/ops/attention/test_kda_prefill_graph.py
M	test/registered/kernels/ops/attention/test_kda_replayssm_fold.py
M	test/registered/kernels/ops/attention/test_kda_replayssm_fold_batched.py
M	test/registered/kernels/ops/attention/test_kda_replayssm_ring_fused.py
M	test/registered/kernels/ops/attention/test_kda_replayssm_ring_ragged.py
A	test/registered/kernels/ops/attention/test_minimax_m31_msa_plan.py
A	test/registered/kernels/ops/attention/test_minimax_m31_page_row_views.py
M	test/registered/kernels/ops/attention/test_minimax_qknorm_rope.py
R098	python/sglang/srt/layers/attention/minimax_sparse_ops/tests/test_flash_with_topk_idx.py	test/registered/kernels/ops/attention/test_minimax_sparse_flash_with_topk_idx.py
R096	python/sglang/srt/layers/attention/minimax_sparse_ops/tests/test_fp8_attn_gemm.py	test/registered/kernels/ops/attention/test_minimax_sparse_fp8_attn_gemm.py
R097	python/sglang/srt/layers/attention/minimax_sparse_ops/tests/test_sparse_gqa.py	test/registered/kernels/ops/attention/test_minimax_sparse_gqa.py
A	test/registered/kernels/ops/attention/test_msa_v1_decode_plan.py
A	test/registered/kernels/ops/attention/test_muse_spark_qk_norm_rope_store.py
M	test/registered/kernels/ops/attention/test_q8kv8_dsa_nope_bf16_prefill.py
M	test/registered/kernels/ops/attention/test_q8kv8_sparse_prefill_backend.py
M	test/registered/kernels/ops/attention/test_qprep_bf16_fp8_sm90.py
M	test/registered/kernels/ops/attention/test_sm120_paged_mqa_logits.py
M	test/registered/kernels/ops/attention/test_sparse_mla_q8kv8_prefill_sm90.py
M	test/registered/kernels/ops/attention/test_tilelang_dsa_zero_rope.py
M	test/registered/kernels/ops/attention/test_topk_v2.py
M	test/registered/kernels/ops/attention/test_triton_sparse_mla_hip.py
M	test/registered/kernels/ops/attention/test_trtllm_kv_norm_rope_store.py
M	test/registered/kernels/ops/attention/test_wo_a_fused.py
A	test/registered/kernels/ops/communication/test_all_reduce_residual_norm.py
M	test/registered/kernels/ops/communication/test_amd_deterministic_custom_allreduce.py
M	test/registered/kernels/ops/communication/test_ar_fusion.py
A	test/registered/kernels/ops/communication/test_beam_all_reduce.py
M	test/registered/kernels/ops/communication/test_collectives.py
M	test/registered/kernels/ops/communication/test_custom_all_reduce.py
A	test/registered/kernels/ops/communication/test_muse_spark_moe_finalize_all_reduce.py
M	test/registered/kernels/ops/communication/test_symm_mem_all_gather.py
M	test/registered/kernels/ops/communication/test_tp_qknorm.py
A	test/registered/kernels/ops/diffusion/test_flux2_xpu_norm_fusions.py
A	test/registered/kernels/ops/diffusion/test_flux3_video_qknorm_rope.py
M	test/registered/kernels/ops/diffusion/test_fp8_rowwise.py
A	test/registered/kernels/ops/diffusion/test_head_gate.py
A	test/registered/kernels/ops/diffusion/test_indexed_scale_shift_mxfp8.py
M	test/registered/kernels/ops/diffusion/test_layout.py
A	test/registered/kernels/ops/diffusion/test_minimax_h3_vae_output.py
M	test/registered/kernels/ops/diffusion/test_model_fast_paths.py
M	test/registered/kernels/ops/diffusion/test_modulate.py
M	test/registered/kernels/ops/diffusion/test_norm.py
A	test/registered/kernels/ops/diffusion/test_rmsnorm_interleaved_rope.py
M	test/registered/kernels/ops/diffusion/test_rope_ltx2.py
M	test/registered/kernels/ops/diffusion/test_sites.py
R095	test/registered/kernels/ops/attention/test_mla_output_gate.py	test/registered/kernels/ops/elementwise/test_mla_output_gate.py
M	test/registered/kernels/ops/elementwise/test_row_scale.py
A	test/registered/kernels/ops/elementwise/test_row_sum.py
R081	test/registered/kernels/ops/moe/test_sigmoid_gate_mul.py	test/registered/kernels/ops/elementwise/test_sigmoid_gate_mul.py
R089	test/registered/kernels/ops/moe/test_sigmoid_gate_mul_broadcast.py	test/registered/kernels/ops/elementwise/test_sigmoid_gate_mul_broadcast.py
R098	test/registered/kernels/ops/speculative/test_ngram_embedding.py	test/registered/kernels/ops/embeddings/test_ngram_embedding.py
M	test/registered/kernels/ops/embeddings/test_vocab_parallel_embedding.py
M	test/registered/kernels/ops/gemm/test_cutedsl_bf16_gemm.py
M	test/registered/kernels/ops/gemm/test_hopper_bf16_gemv.py
M	test/registered/kernels/ops/gemm/test_minimax_fused_qkv_index_gemm.py
A	test/registered/kernels/ops/gemm/test_muse_spark_router_gemv_cutedsl.py
R089	test/registered/kernels/ops/quantization/test_mxfp8_amd_gfx95.py	test/registered/kernels/ops/gemm/test_mxfp8_amd_gfx95.py
A	test/registered/kernels/ops/gemm/test_router_gemv.py
M	test/registered/kernels/ops/gemm/test_sm120_fp8_linear.py
M	test/registered/kernels/ops/gemm/test_tiny_gemm.py
M	test/registered/kernels/ops/gemm/test_triton_scaled_mm.py
M	test/registered/kernels/ops/kv_canary/test_scatter_req_token_ids.py
M	test/registered/kernels/ops/kv_canary/test_utils.py
A	test/registered/kernels/ops/kvcache/test_dsa_index_cache_move.py
R093	test/registered/kernels/ops/attention/test_dsv41_request_window_copy.py	test/registered/kernels/ops/kvcache/test_dsv41_request_window_copy.py
A	test/registered/kernels/ops/kvcache/test_dsv4_fp4_index_k_cache.py
R099	test/registered/kernels/ops/attention/test_fused_store_index_cache.py	test/registered/kernels/ops/kvcache/test_fused_store_index_cache.py
M	test/registered/kernels/ops/kvcache/test_hicache.py
M	test/registered/kernels/ops/kvcache/test_hicache_page_first_write_back.py
A	test/registered/kernels/ops/kvcache/test_minimax_m31_hicache.py
A	test/registered/kernels/ops/kvcache/test_minimax_m31_nvfp4.py
M	test/registered/kernels/ops/kvcache/test_mxfp8_host_allocation.py
A	test/registered/kernels/ops/kvcache/test_page_unified_host_pool.py
M	test/registered/kernels/ops/kvcache/test_set_mla_kv_buffer.py
M	test/registered/kernels/ops/layernorm/test_hc_boundary_hip.py
R082	test/registered/kernels/ops/layernorm/test_hc_combine.py	test/registered/kernels/ops/layernorm/test_mhc_combine.py
M	test/registered/kernels/ops/layernorm/test_mhc_kernels.py
A	test/registered/kernels/ops/layernorm/test_mhc_stream_mean.py
A	test/registered/kernels/ops/layernorm/test_muse_spark_norm_ops.py
A	test/registered/kernels/ops/layernorm/test_muse_spark_rmsnorm_nvfp4_quant.py
M	test/registered/kernels/ops/layernorm/test_mxfp8_epilogue.py
M	test/registered/kernels/ops/layernorm/test_qknorm.py
M	test/registered/kernels/ops/lora/test_chunked_sgmv_cuda_graph.py
A	test/registered/kernels/ops/lora/test_dense_lora_backend.py
A	test/registered/kernels/ops/lora/test_dense_lora_geometry.py
A	test/registered/kernels/ops/lora/test_dense_lora_inkling_sink.py
A	test/registered/kernels/ops/lora/test_dense_lora_kernels.py
A	test/registered/kernels/ops/lora/test_dense_lora_mla_correction.py
A	test/registered/kernels/ops/lora/test_dense_lora_routing_graph.py
A	test/registered/kernels/ops/lora/test_moe_lora_bind_execution.py
A	test/registered/kernels/ops/lora/test_moe_lora_contiguous_row_domain.py
A	test/registered/kernels/ops/lora/test_moe_lora_down_b_into_base.py
A	test/registered/kernels/ops/lora/test_moe_lora_fp8_provider_numerics.py
A	test/registered/kernels/ops/lora/test_moe_lora_fused_dispatch.py
A	test/registered/kernels/ops/lora/test_moe_lora_masked_capacity_graph.py
A	test/registered/kernels/ops/lora/test_moe_lora_moe_runner_numerics.py
A	test/registered/kernels/ops/lora/test_moe_lora_nvfp4_provider_numerics.py
A	test/registered/kernels/ops/lora/test_moe_lora_per_row_b.py
A	test/registered/kernels/ops/lora/test_moe_lora_prepared_input_reuse.py
A	test/registered/kernels/ops/lora/test_moe_lora_routing.py
A	test/registered/kernels/ops/lora/test_moe_lora_runner_attach.py
A	test/registered/kernels/ops/lora/test_moe_lora_schedule_builder.py
A	test/registered/kernels/ops/lora/test_moe_lora_shared_outer_numerics.py
A	test/registered/kernels/ops/lora/test_moe_lora_small_masked_prepare_paths.py
M	test/registered/kernels/ops/mamba/test_mtp_hicache.py
M	test/registered/kernels/ops/memory/test_small_copy.py
A	test/registered/kernels/ops/mm/test_lanczos.py
M	test/registered/kernels/ops/moe/test_block_int8.py
A	test/registered/kernels/ops/moe/test_cutlass_fp8_inactive_routes.py
A	test/registered/kernels/ops/moe/test_cutlass_w4a8_inactive_routes.py
A	test/registered/kernels/ops/moe/test_eplb_map_record.py
A	test/registered/kernels/ops/moe/test_fused_moe_block_fp8_scales.py
A	test/registered/kernels/ops/moe/test_gguf_moe_weights.py
M	test/registered/kernels/ops/moe/test_humming_fp8_dispatch.py
M	test/registered/kernels/ops/moe/test_int8_kernel.py
M	test/registered/kernels/ops/moe/test_jit_grouped_topk.py
M	test/registered/kernels/ops/moe/test_marlin_moe.py
M	test/registered/kernels/ops/moe/test_marlin_packed_topk_unpack.py
A	test/registered/kernels/ops/moe/test_mega_gate.py
A	test/registered/kernels/ops/moe/test_minimax_m31_nvfp4_trtllm_moe.py
M	test/registered/kernels/ops/moe/test_minimax_m3_mxfp8.py
M	test/registered/kernels/ops/moe/test_minimax_quant_scatter.py
M	test/registered/kernels/ops/moe/test_moe_align_block_size.py
M	test/registered/kernels/ops/moe/test_moe_fused_gate.py
M	test/registered/kernels/ops/moe/test_moe_route_radix4.py
M	test/registered/kernels/ops/moe/test_moe_wna16_marlin.py
A	test/registered/kernels/ops/moe/test_muse_spark_routing.py
A	test/registered/kernels/ops/moe/test_paged_experts_kernels.py
M	test/registered/kernels/ops/moe/test_rocm_router_gate.py
M	test/registered/kernels/ops/moe/test_round_robin_ep_balance.py
A	test/registered/kernels/ops/moe/test_sm120_topk_gate_small_m.py
M	test/registered/kernels/ops/quantization/test_compressed_tensors_fp4_numeric.py
M	test/registered/kernels/ops/quantization/test_dsv4_fp8_cast.py
A	test/registered/kernels/ops/quantization/test_fp4_indexer_quant.py
M	test/registered/kernels/ops/quantization/test_fp8_grid_producers_gfx95.py
M	test/registered/kernels/ops/quantization/test_fp8_kernel.py
R097	test/registered/kernels/ops/attention/test_fp8_wo_a.py	test/registered/kernels/ops/quantization/test_fp8_wo_a.py
M	test/registered/kernels/ops/quantization/test_mxfp8_swizzled.py
M	test/registered/kernels/ops/quantization/test_per_token_group_quant.py
A	test/registered/kernels/ops/sampling/test_lm_head_output_dtype.py
M	test/registered/kernels/ops/speculative/test_boundary_kv_fix_kernels.py
A	test/registered/kernels/ops/speculative/test_deterministic_fast_sample.py
M	test/registered/kernels/ops/speculative/test_draft_decode_window.py
A	test/registered/kernels/ops/speculative/test_dspark_fused_kv_write.py
M	test/registered/kernels/ops/speculative/test_lilicorr_cuda.py
M	test/registered/kernels/ops/speculative/test_next_verify_prep.py
M	test/registered/kernels/ops/speculative/test_spec_kv_indices_grid.py
M	test/registered/kv_canary/test_self_e2e_baseline.py
M	test/registered/kv_canary/test_self_e2e_bench_speed.py
M	test/registered/kv_canary/test_self_e2e_pd_baseline.py
M	test/registered/kv_canary/test_self_e2e_pd_perturb.py
M	test/registered/kv_canary/test_self_e2e_perturb_raise.py
M	test/registered/kv_canary/test_self_e2e_perturb_real_kv_unused_cache.py
M	test/registered/kv_canary/test_self_e2e_perturb_real_kv_used.py
M	test/registered/kv_canary/test_self_e2e_perturb_req_to_token.py
M	test/registered/kv_canary/test_self_e2e_pp_baseline.py
M	test/registered/kv_canary/test_self_e2e_pp_perturb.py
M	test/registered/kv_canary/test_self_e2e_pr_25015.py
M	test/registered/kv_canary/test_self_e2e_pr_26329.py
M	test/registered/kv_canary/test_self_unit_buffer_alloc.py
M	test/registered/kv_canary/test_self_unit_capacities.py
M	test/registered/kv_canary/test_self_unit_capture_refusal.py
M	test/registered/kv_canary/test_self_unit_endpoint.py
M	test/registered/kv_canary/test_self_unit_future_tensor.py
M	test/registered/kv_canary/test_self_unit_perturb.py
M	test/registered/kv_canary/test_self_unit_plan_input.py
M	test/registered/kv_canary/test_self_unit_pool_patcher.py
M	test/registered/kv_canary/test_self_unit_pool_patcher_utils.py
M	test/registered/kv_canary/test_self_unit_radix_walker.py
M	test/registered/kv_canary/test_self_unit_req_to_expected_token_ids_manager.py
M	test/registered/kv_canary/test_self_unit_runner_health.py
M	test/registered/kv_canary/test_self_unit_runner_per_forward.py
M	test/registered/kv_canary/test_self_unit_runner_swa_divergence.py
M	test/registered/kv_canary/test_self_unit_runner_sweep.py
M	test/registered/kv_canary/test_self_unit_sweep_plan_builder.py
M	test/registered/kv_canary/test_self_unit_token_oracle.py
M	test/registered/kv_canary/test_self_unit_violation.py
M	test/registered/layers/mamba/test_causal_conv1d.py
M	test/registered/layers/mamba/test_mamba2_mixer.py
M	test/registered/layers/mamba/test_mamba_slot_fused.py
M	test/registered/layers/mamba/test_mamba_ssm.py
M	test/registered/layers/mamba/test_mamba_ssm_ssd.py
M	test/registered/layers/test_layernorm_fusion.py
M	test/registered/lora/test_fused_moe_lora_kernel.py
M	test/registered/lora/test_lora_drainer.py
M	test/registered/lora/test_lora_eviction.py
M	test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py
M	test/registered/lora/test_lora_moe_tp_logprob_diff.py
M	test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py
M	test/registered/lora/test_lora_openai_api.py
M	test/registered/lora/test_lora_overlap_loading.py
M	test/registered/lora/test_lora_qwen3_5_4b_logprob_diff.py
M	test/registered/lora/test_lora_qwen3_8b_logprob_diff.py
M	test/registered/lora/test_moe_lora_info.py
M	test/registered/lora/test_multi_lora_backend.py
M	test/registered/lora/test_virtual_experts_kernels.py
M	test/registered/mem_cache/test_post_capture_kv_sizing.py
M	test/registered/mla/test_mla_fp8.py
M	test/registered/mla/test_mla_int8_deepseek_v3.py
M	test/registered/mock_model/test_e2e_cp.py
M	test/registered/mock_model/test_e2e_pd.py
M	test/registered/mock_model/test_e2e_pp.py
M	test/registered/mock_model/test_e2e_spec_eagle.py
M	test/registered/mock_model/test_e2e_tp.py
M	test/registered/mock_model/test_self_e2e_perturb_next_token_swap.py
M	test/registered/mock_model/test_self_unit_canary_mock_wiring.py
M	test/registered/mock_model/test_self_unit_install.py
M	test/registered/mock_model/test_self_unit_oracle.py
M	test/registered/mock_model/test_self_unit_oracle_torch_vs_ref.py
M	test/registered/mock_model/test_self_unit_sampler_hookpoint.py
A	test/registered/model_loading/test_cpu_offloader.py
M	test/registered/model_loading/test_external_models.py
M	test/registered/model_loading/test_load_weights_from_remote_instance.py
M	test/registered/model_loading/test_startup_weight_load.py
M	test/registered/model_loading/test_utils_update_weights.py
M	test/registered/model_loading/test_weight_cache_daemon.py
M	test/registered/model_loading/test_weight_loader_v2_e2e.py
M	test/registered/moe/test_cutedsl_moe.py
M	test/registered/moe/test_flashinfer_a2a_cutedsl_v2.py
A	test/registered/moe/test_flashinfer_fp8_routed.py
M	test/registered/moe/test_fused_append_remap_per_rank_shared_slots.py
M	test/registered/moe/test_fused_append_shared_experts.py
M	test/registered/moe/test_fused_append_shared_experts_top6.py
M	test/registered/moe/test_fused_moe.py
M	test/registered/moe/test_glm4_moe_models.py
M	test/registered/moe/test_hash_topk.py
M	test/registered/moe/test_hpc_ops_moe.py
M	test/registered/moe/test_moe_ep.py
M	test/registered/moe/test_moe_ep_extra.py
A	test/registered/moe/test_shared_expert_routing_extra.py
M	test/registered/moe/test_topk_padded_region.py
M	test/registered/moe/test_topk_renormalize_degenerate.py
M	test/registered/moe/test_torch_compile_moe.py
M	test/registered/moe/test_triton_fused_moe.py
M	test/registered/moe/test_triton_moe_channel_fp8_kernel.py
A	test/registered/moe/test_triton_moe_padding.py
M	test/registered/npu/accuracy/glm5_top64_pruned/test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py
M	test/registered/npu/accuracy/kimi_k3/test_npu_kimi_k3_w4a8_32p_gpqa.py
M	test/registered/npu/accuracy/qwen3_6_27b/test_npu_qwen3_6_27b_1p_gpqa.py
M	test/registered/npu/accuracy/qwen3_6_27b/test_npu_qwen3_6_27b_w8a8_1p_in3k5_out1k5_50ms_gpqa.py
M	test/registered/npu/accuracy/qwen3_6_35b_a3b/test_npu_qwen3_6_35b_a3b_1p_aime26.py
M	test/registered/npu/accuracy/qwen3_6_35b_a3b/test_npu_qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms_aime26.py
M	test/registered/npu/basic_function/HiCache/test_npu_hicache_mamba.py
D	test/registered/npu/basic_function/HiCache/test_npu_hicache_mla.py
A	test/registered/npu/basic_function/api_related/test_npu_return_input_ids.py
A	test/registered/npu/basic_function/api_related/test_npu_return_output_ids.py
M	test/registered/npu/basic_function/dllm/test_npu_llada2_mini_fdfo.py
M	test/registered/npu/basic_function/dp_attn/test_npu_dp_attention.py
M	test/registered/npu/basic_function/parallel_strategy/data_parallelism/test_npu_load_balance_method_pd_disaggregation.py
A	test/registered/npu/basic_function/pd_disaggregation/test_npu_pd_decode_retraction_backup.py
M	test/registered/npu/basic_function/pd_disaggregation/test_npu_pd_disaggregation.py
A	test/registered/npu/basic_function/runtime_opts/test_npu_constrained_json_whitespace.py
A	test/registered/npu/basic_function/runtime_opts/test_npu_mla_fia_hicache_bf16.py
D	test/registered/npu/basic_function/runtime_opts/test_npu_mla_fia_w8a8int8.py
D	test/registered/npu/basic_function/runtime_opts/test_npu_mla_w8a8int8.py
M	test/registered/npu/performance/qwen3_6_27b/test_npu_qwen3_6_27b_2p_in64k_out1k_prefix90_50ms.py
M	test/registered/npu/performance/qwen3_6_27b/test_npu_qwen3_6_27b_w8a8_1p_in64k_out1k_50ms.py
M	test/registered/npu/rl/indexer_replay/test_npu_return_indexer_topk_deepseek_v4.py
M	test/registered/npu/rl/indexer_replay/test_npu_return_indexer_topk_glm52.py
M	test/registered/observability/test_customized_info_streaming.py
M	test/registered/observability/test_encoder_server_metrics.py
M	test/registered/observability/test_metrics.py
M	test/registered/observability/test_tracing.py
M	test/registered/observability/test_tracing_disaggregation.py
M	test/registered/openai_server/basic/test_http2_server.py
M	test/registered/openai_server/basic/test_openai_completion_rust.py
M	test/registered/openai_server/basic/test_serving_transcription.py
M	test/registered/openai_server/features/test_openai_server_hidden_states.py
M	test/registered/openai_server/function_call/test_anthropic_tool_use.py
M	test/registered/openai_server/function_call/test_openai_function_calling.py
M	test/registered/openai_server/validation/test_large_max_new_tokens.py
M	test/registered/openai_server/validation/test_matched_stop.py
M	test/registered/openai_server/validation/test_request_length_validation.py
M	test/registered/page_major/test_page_major_gpt_oss.py
M	test/registered/page_major/test_page_major_qwen_hybrid.py
M	test/registered/pp/test_pp_gemma4.py
M	test/registered/pp/test_pp_parallel_compat.py
M	test/registered/pp/test_pp_single_node.py
M	test/registered/pp/test_pp_single_node_extra.py
M	test/registered/prefill_only/test_embedding_models.py
M	test/registered/prefill_only/test_multi_item_scoring.py
M	test/registered/prefill_only/test_openai_embedding.py
M	test/registered/prefill_only/test_pooled_hidden_states.py
M	test/registered/prefill_only/test_reward_models.py
M	test/registered/prefill_only/test_score_api.py
M	test/registered/prefill_only/test_score_engine.py
M	test/registered/prefill_only/test_serving_rerank.py
M	test/registered/profiling/test_start_profile.py
M	test/registered/radix_cache/test_int8_mamba_checkpoint_e2e.py
M	test/registered/radix_cache/test_mamba2_extra_buffer_kl.py
M	test/registered/radix_cache/test_mamba_cache_kl.py
M	test/registered/radix_cache/test_radix_attention.py
M	test/registered/radix_cache/test_radix_cache_hit.py
M	test/registered/radix_cache/test_swa_radix_cache_kl.py
A	test/registered/radix_cache/unified_radix_tree/linker/test_unified_cache_linker_kimi_linear.py
M	test/registered/radix_cache/unified_radix_tree/linker/test_unified_cache_linker_kl_dsv4.py
A	test/registered/radix_cache/unified_radix_tree/linker/test_unified_cache_linker_kl_dsv41.py
M	test/registered/radix_cache/unified_radix_tree/linker/test_unified_cache_linker_kl_glm52.py
M	test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_cp.py
M	test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_dcp.py
M	test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_dsv4.py
M	test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_dsv4_pp.py
M	test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_full.py
M	test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_glm52.py
M	test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_hybrid_bitexact.py
M	test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_mamba.py
M	test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_swa.py
M	test/registered/reasoning/test_reasoning.py
M	test/registered/rl/test_fp32_lm_head.py
M	test/registered/rl/test_lora_load_from_tensor.py
M	test/registered/rl/test_multi_instance_release_memory_occupation.py
M	test/registered/rl/test_return_indexer_topk.py
M	test/registered/rl/test_return_routed_experts.py
M	test/registered/rl/test_tensor_bucket.py
M	test/registered/rl/test_update_weights_from_disk_blackwell.py
M	test/registered/rl/test_update_weights_from_distributed.py
M	test/registered/rl/test_update_weights_from_tensor.py
M	test/registered/rotary/test_mrope_axis_map.py
M	test/registered/sampling/test_deterministic_gumbel_u1.py
M	test/registered/sampling/test_original_logprobs.py
M	test/registered/sampling/test_penalty.py
M	test/registered/sampling/test_pytorch_sampling_backend.py
M	test/registered/sampling/test_sampling_mask.py
M	test/registered/scheduler/test_abort_with_metrics.py
M	test/registered/scheduler/test_async_d2h_copy_worker_server.py
M	test/registered/scheduler/test_load_snapshot_server.py
M	test/registered/scheduler/test_min_free_slots_delayer.py
M	test/registered/scheduler/test_mixed_chunked_prefill.py
M	test/registered/scheduler/test_overlap_skip_finishing_decode.py
M	test/registered/scheduler/test_priority_scheduling.py
M	test/registered/scheduler/test_retract_decode.py
M	test/registered/scheduler/test_retract_decode_logprob.py
M	test/registered/scheduler/test_scheduler_control.py
M	test/registered/scripted_runtime/test_scripted_runtime_core.py
M	test/registered/sessions/test_session_control.py
M	test/registered/sessions/test_session_latency.py
M	test/registered/sessions/test_streaming_session.py
M	test/registered/sessions/test_streaming_session_extra.py
M	test/registered/sessions/test_streaming_session_swa.py
M	test/registered/sessions/test_streaming_session_swa_extra.py
M	test/registered/spec/dflash/test_dflash.py
M	test/registered/spec/dspark/test_dspark_block_accept_estimator.py
M	test/registered/spec/dspark/test_dspark_confidence_metrics.py
M	test/registered/spec/dspark/test_dspark_dp_spec_prefill_coordination.py
M	test/registered/spec/dspark/test_dspark_dp_tier.py
M	test/registered/spec/dspark/test_dspark_draft_graph.py
M	test/registered/spec/dspark/test_dspark_draft_path_default.py
M	test/registered/spec/dspark/test_dspark_info_dumper.py
M	test/registered/spec/dspark/test_dspark_kernel_parity.py
A	test/registered/spec/dspark/test_dspark_minimax_mxfp8_stacked_kv.py
M	test/registered/spec/dspark/test_dspark_scheduler.py
M	test/registered/spec/dspark/test_dspark_simulated_bonus.py
M	test/registered/spec/dspark/test_dspark_sps_table.py
M	test/registered/spec/dspark/test_dspark_sts.py
M	test/registered/spec/dspark/test_ragged_verify.py
M	test/registered/spec/dspark/test_ragged_verify_backend_capability.py
M	test/registered/spec/eagle/test_adaptive_speculative.py
M	test/registered/spec/eagle/test_deepseek_v3_fp4_mtp_small.py
M	test/registered/spec/eagle/test_dp_spec_prefill_coordination.py
M	test/registered/spec/eagle/test_dp_spec_prefill_coordination_mixed.py
M	test/registered/spec/eagle/test_eagle_constrained_decoding.py
M	test/registered/spec/eagle/test_eagle_reject_sampling.py
M	test/registered/spec/eagle/test_spec_eagle.py
M	test/registered/spec/eagle/test_spec_eagle_fa3.py
M	test/registered/spec/eagle/test_spec_eagle_page.py
M	test/registered/spec/eagle/test_spec_eagle_parity.py
M	test/registered/spec/eagle/test_spec_eagle_stress.py
M	test/registered/spec/eagle/test_spec_eagle_topk.py
M	test/registered/spec/eagle/test_spec_eagle_topk_page.py
M	test/registered/spec/eagle/test_spec_eagle_triton.py
M	test/registered/spec/test_constrained_decoding_spec_reasoning.py
M	test/registered/spec/test_frozen_kv_mtp.py
M	test/registered/spec/test_gemma4_dflash_31b_extra.py
M	test/registered/spec/test_gemma4_mtp_31b_extra.py
M	test/registered/spec/test_spec_mixed_chunk.py
M	test/registered/spec/test_spec_ngram.py
M	test/registered/spec/test_spec_ngram_extra.py
M	test/registered/spec/test_spec_standalone.py
M	test/registered/spec/test_spec_standalone_extra.py
M	test/registered/tokenizer/test_multi_tokenizer.py
M	test/registered/tokenizer/test_skip_tokenizer_init.py
M	test/registered/unit/api/v1/test_json_contract.py
M	test/registered/unit/api/v1/test_parity.py
A	test/registered/unit/attention/test_npu_mlaprolog.py
M	test/registered/unit/attention/test_triton_sparse_mla_prefill.py
A	test/registered/unit/batch_invariant_ops/test_fused_add_rms_norm_batch_invariant.py
A	test/registered/unit/batch_invariant_ops/test_matmul_persistent_configs.py
A	test/registered/unit/batch_invariant_ops/test_qwen_batch_invariance.py
M	test/registered/unit/batch_overlap/test_tbo_filter_batch_marker.py
M	test/registered/unit/beam_search/test_beam_search_core.py
M	test/registered/unit/beam_search/test_fork.py
M	test/registered/unit/beam_search/test_output_decode.py
M	test/registered/unit/bench/test_bench_serving_goodput.py
M	test/registered/unit/bench/test_bench_serving_prompt_len.py
M	test/registered/unit/bench/test_benchmark_endpoint.py
M	test/registered/unit/cli/test_serve_backends.py
M	test/registered/unit/cli/test_utils.py
M	test/registered/unit/compilation/test_torch_compile_decoration.py
A	test/registered/unit/configs/test_inkling_config.py
M	test/registered/unit/configs/test_laguna_config.py
M	test/registered/unit/configs/test_linear_attn_model_registry.py
M	test/registered/unit/configs/test_minicpm_config.py
M	test/registered/unit/configs/test_model_config.py
M	test/registered/unit/configs/test_model_config_parser_registry.py
M	test/registered/unit/configs/test_model_config_scaling.py
M	test/registered/unit/configs/test_model_config_shapes.py
M	test/registered/unit/configs/test_multimodal_piecewise_cuda_graph.py
M	test/registered/unit/configs/test_nano_nemotron_vl_config.py
M	test/registered/unit/configs/test_quant_config_parsing.py
M	test/registered/unit/configs/test_zaya_config.py
M	test/registered/unit/constrained/test_grammar_manager.py
M	test/registered/unit/constrained/test_llguidance_batched_mask.py
M	test/registered/unit/constrained/test_mistral_common_xgrammar.py
M	test/registered/unit/constrained/test_reasoner_grammar_backend.py
M	test/registered/unit/cp/test_qsa_cp_unit.py
M	test/registered/unit/disaggregation/test_admission_abort_not_enqueued.py
M	test/registered/unit/disaggregation/test_ascend_page_dcp.py
A	test/registered/unit/disaggregation/test_common_sender_lifecycle.py
M	test/registered/unit/disaggregation/test_dcp_pack.py
A	test/registered/unit/disaggregation/test_decode_hicache_l2.py
M	test/registered/unit/disaggregation/test_decode_hicache_tree_core.py
M	test/registered/unit/disaggregation/test_decode_prealloc_no_host_sync.py
M	test/registered/unit/disaggregation/test_decode_queue_cleanup.py
M	test/registered/unit/disaggregation/test_decode_req_to_token_pool.py
M	test/registered/unit/disaggregation/test_deferred_bootstrap.py
M	test/registered/unit/disaggregation/test_deferred_decode_kv_release.py
M	test/registered/unit/disaggregation/test_disaggregation_wire.py
A	test/registered/unit/disaggregation/test_dsa_sharded_pd_transfer.py
D	test/registered/unit/disaggregation/test_dsv4_retract_rebootstrap.py
M	test/registered/unit/disaggregation/test_dsv4_unified_fp8_pd.py
M	test/registered/unit/disaggregation/test_encode_receiver.py
M	test/registered/unit/disaggregation/test_encode_server.py
D	test/registered/unit/disaggregation/test_encoder_health.py
D	test/registered/unit/disaggregation/test_encoder_scheduler.py
M	test/registered/unit/disaggregation/test_fake_kv_sender.py
M	test/registered/unit/disaggregation/test_head_slice_params.py
M	test/registered/unit/disaggregation/test_kimi_k3_encoder_mode.py
M	test/registered/unit/disaggregation/test_kv_checksum.py
M	test/registered/unit/disaggregation/test_kv_events.py
M	test/registered/unit/disaggregation/test_kv_transfer_replica_metric.py
M	test/registered/unit/disaggregation/test_minimax_sparse_disagg_state_kv_args.py
D	test/registered/unit/disaggregation/test_mooncake_custom_mem_pool_batch.py
M	test/registered/unit/disaggregation/test_mooncake_deferred_kv_release.py
M	test/registered/unit/disaggregation/test_mooncake_efa_allocator.py
A	test/registered/unit/disaggregation/test_mooncake_failed_session_probe.py
M	test/registered/unit/disaggregation/test_mooncake_transfer_batching.py
M	test/registered/unit/disaggregation/test_mori_deferred_kv_release.py
M	test/registered/unit/disaggregation/test_nixl_backend_basic.py
M	test/registered/unit/disaggregation/test_nixl_deferred_kv_release.py
D	test/registered/unit/disaggregation/test_nixl_dsv4_kv_slots.py
D	test/registered/unit/disaggregation/test_nixl_sender_failure_cleanup.py
A	test/registered/unit/disaggregation/test_npu_hybrid_mla_kv_args.py
M	test/registered/unit/disaggregation/test_pd_role_switch.py
M	test/registered/unit/disaggregation/test_pp_consensus_store.py
M	test/registered/unit/disaggregation/test_pp_hybrid_kv_transfer.py
M	test/registered/unit/disaggregation/test_pp_mla_kv_transfer.py
M	test/registered/unit/disaggregation/test_prefill_abort_result_cleanup.py
M	test/registered/unit/disaggregation/test_prefill_complete_integration.py
M	test/registered/unit/disaggregation/test_prefill_result_polling.py
M	test/registered/unit/disaggregation/test_receiver_connection_pool.py
M	test/registered/unit/disaggregation/test_register_to_bootstrap.py
M	test/registered/unit/disaggregation/test_specv2_kvcache_offloading.py
M	test/registered/unit/disaggregation/test_staging_draft_kv_slots.py
M	test/registered/unit/disaggregation/test_unified_memory_move_gate.py
M	test/registered/unit/distributed/test_attn_cp_tp_broadcast.py
M	test/registered/unit/distributed/test_cuda_wrapper.py
M	test/registered/unit/distributed/test_custom_all_reduce_v2_capability.py
M	test/registered/unit/distributed/test_gated_launch.py
M	test/registered/unit/distributed/test_get_default_distributed_backend.py
A	test/registered/unit/distributed/test_oot_group_coordinator_hooks.py
M	test/registered/unit/distributed/test_parallel_state.py
M	test/registered/unit/distributed/test_pcie_ipc_ar.py
M	test/registered/unit/distributed/test_pp_comm_overlap.py
A	test/registered/unit/distributed/test_pynccl.py
A	test/registered/unit/distributed/test_symm_mem_registration_watchdog.py
M	test/registered/unit/dllm/algorithm/test_joint_threshold_indel.py
M	test/registered/unit/dllm/test_gemma4_cuda_graph.py
M	test/registered/unit/dllm/test_gemma4_diffusion.py
M	test/registered/unit/dllm/test_gemma4_renoise.py
M	test/registered/unit/dllm/test_gemma4_uniform_lifecycle.py
M	test/registered/unit/dsa/test_dsa_gfx950_fused_workspace.py
M	test/registered/unit/elastic_ep/test_topology.py
M	test/registered/unit/entrypoints/anthropic/test_serving.py
M	test/registered/unit/entrypoints/openai/test_exa_search.py
M	test/registered/unit/entrypoints/openai/test_matched_stop.py
M	test/registered/unit/entrypoints/openai/test_parallel_prompt_encode.py
M	test/registered/unit/entrypoints/openai/test_protocol.py
M	test/registered/unit/entrypoints/openai/test_responses_custom_tools.py
A	test/registered/unit/entrypoints/openai/test_responses_namespace_tools.py
A	test/registered/unit/entrypoints/openai/test_responses_namespaces.py
M	test/registered/unit/entrypoints/openai/test_responses_protocol.py
A	test/registered/unit/entrypoints/openai/test_responses_shell.py
M	test/registered/unit/entrypoints/openai/test_serving_chat.py
M	test/registered/unit/entrypoints/openai/test_serving_completions.py
M	test/registered/unit/entrypoints/openai/test_serving_decisions.py
A	test/registered/unit/entrypoints/openai/test_serving_deepseek_recipe.py
M	test/registered/unit/entrypoints/openai/test_serving_embedding.py
A	test/registered/unit/entrypoints/openai/test_serving_inkling.py
M	test/registered/unit/entrypoints/openai/test_serving_responses.py
M	test/registered/unit/entrypoints/openai/test_serving_responses_stream.py
A	test/registered/unit/entrypoints/openai/test_serving_tokenize.py
M	test/registered/unit/entrypoints/openai/test_usage_processor.py
M	test/registered/unit/entrypoints/openai/test_whisper_adapter.py
M	test/registered/unit/entrypoints/openai/utils.py
M	test/registered/unit/entrypoints/systemone/test_joint_schema.py
M	test/registered/unit/entrypoints/test_generate_api_contract.py
M	test/registered/unit/entrypoints/test_grpc_bridge.py
M	test/registered/unit/entrypoints/test_http_server_auth_levels.py
A	test/registered/unit/entrypoints/test_http_server_reuseport.py
M	test/registered/unit/entrypoints/test_http_server_warmup.py
M	test/registered/unit/entrypoints/test_ready.py
M	test/registered/unit/entrypoints/test_rust_server_dp_ports.py
M	test/registered/unit/entrypoints/test_server_info.py
M	test/registered/unit/entrypoints/test_server_warmup.py
M	test/registered/unit/entrypoints/test_ssl_cert_refresher.py
M	test/registered/unit/eplb/test_balanced_packing.py
M	test/registered/unit/eplb/test_dispatch_dtype_preservation.py
A	test/registered/unit/eplb/test_expert_location_weight_loading.py
M	test/registered/unit/eplb/test_megamoe_prefill_recorder.py
M	test/registered/unit/eplb/test_waterfill_eplb.py
M	test/registered/unit/function_call/test_deepseekv41_detector.py
M	test/registered/unit/function_call/test_dots_detector.py
M	test/registered/unit/function_call/test_function_call_parser.py
M	test/registered/unit/function_call/test_glm47_schema_types.py
M	test/registered/unit/function_call/test_hunyuan_detector.py
M	test/registered/unit/function_call/test_iquest_q1_detector.py
M	test/registered/unit/function_call/test_k2_v3_detector.py
M	test/registered/unit/function_call/test_kimik3_structural_tag.py
M	test/registered/unit/function_call/test_minimax_m3_detector.py
M	test/registered/unit/function_call/test_muse_glimmer_format.py
A	test/registered/unit/function_call/test_response_template_structural_tag.py
M	test/registered/unit/hardware_backend/mlx/test_attention_patching.py
A	test/registered/unit/hardware_backend/mlx/test_auxiliary_state_acquire.py
M	test/registered/unit/hardware_backend/mlx/test_max_running_requests.py
M	test/registered/unit/hardware_backend/mlx/test_mlx_quantization_override.py
M	test/registered/unit/hardware_backend/mlx/test_mlx_remote_code_gate.py
M	test/registered/unit/hardware_backend/mlx/test_mlx_runner_pool_contract.py
R096	test/registered/unit/monokernels/test_mono_contract.py	test/registered/unit/kernel_plugins/amd/test_mono_contract.py
R090	test/registered/unit/monokernels/test_mono_framework.py	test/registered/unit/kernel_plugins/amd/test_mono_framework.py
A	test/registered/unit/kernel_plugins/nvidia/_fakes.py
A	test/registered/unit/kernel_plugins/nvidia/test_eagle_binding.py
A	test/registered/unit/kernel_plugins/nvidia/test_megakernel_binding.py
A	test/registered/unit/kernel_plugins/nvidia/test_megakernel_guards.py
A	test/registered/unit/kernel_plugins/nvidia/test_megakernel_runtime.py
M	test/registered/unit/kernels/test_device_guard.py
M	test/registered/unit/kernels/test_fused_op.py
M	test/registered/unit/kernels/test_fused_op_dispatch.py
M	test/registered/unit/kernels/test_jit_cache.py
M	test/registered/unit/kernels/test_kernels_namespace.py
M	test/registered/unit/layer_boundary/test_append_stages.py
M	test/registered/unit/layer_boundary/test_attn_bank_residual_ops.py
M	test/registered/unit/layer_boundary/test_aux_capture_deferred_allreduce.py
M	test/registered/unit/layer_boundary/test_aux_hidden_states.py
M	test/registered/unit/layer_boundary/test_batch_owned_residual.py
M	test/registered/unit/layer_boundary/test_boundary_edges.py
M	test/registered/unit/layer_boundary/test_boundary_output_contracts.py
M	test/registered/unit/layer_boundary/test_boundary_output_policy.py
A	test/registered/unit/layer_boundary/test_custom_all_reduce_fusion.py
M	test/registered/unit/layer_boundary/test_cutedsl_ar_fusion.py
M	test/registered/unit/layer_boundary/test_declared_attention_cp.py
M	test/registered/unit/layer_boundary/test_declared_decoder_boundary.py
M	test/registered/unit/layer_boundary/test_deepstack_deferred_allreduce.py
D	test/registered/unit/layer_boundary/test_dense_mlp_under_prefill_cp.py
M	test/registered/unit/layer_boundary/test_dp_attention_cp_gather.py
M	test/registered/unit/layer_boundary/test_draft_layer_declarations.py
M	test/registered/unit/layer_boundary/test_gated_residual_ops.py
M	test/registered/unit/layer_boundary/test_glm5_next_dflash_capture.py
M	test/registered/unit/layer_boundary/test_ihc_residual_ops.py
M	test/registered/unit/layer_boundary/test_independent_stage_construction.py
M	test/registered/unit/layer_boundary/test_kimi_k3_stage_kernels.py
M	test/registered/unit/layer_boundary/test_layernorm_sp.py
M	test/registered/unit/layer_boundary/test_longcat_flash_shortcut.py
M	test/registered/unit/layer_boundary/test_moe_output_reduction.py
M	test/registered/unit/layer_boundary/test_nemotron_h_aux_capture.py
D	test/registered/unit/layer_boundary/test_nemotron_h_mtp_reduction.py
M	test/registered/unit/layer_boundary/test_nemotron_h_stages.py
M	test/registered/unit/layer_boundary/test_prepare_attn_steps.py
M	test/registered/unit/layer_boundary/test_producer_completion.py
M	test/registered/unit/layer_boundary/test_qwen4_exp_ple_rows.py
M	test/registered/unit/layer_boundary/test_residual_stream.py
M	test/registered/unit/layer_boundary/test_row_transport.py
M	test/registered/unit/layer_boundary/test_short_conv_boundary.py
M	test/registered/unit/layer_boundary/test_stage_boundaries_built_once.py
M	test/registered/unit/layer_boundary/test_stage_layers_built_in_a_stack.py
M	test/registered/unit/layer_boundary/test_stage_producers_leave_sums.py
M	test/registered/unit/layer_boundary/test_tbo_unreduced_input.py
M	test/registered/unit/layer_boundary/test_token_order_attention.py
A	test/registered/unit/layers/attention/linear/kernels/test_kda_flashinfer_prefill.py
M	test/registered/unit/layers/attention/linear/kernels/test_kda_nvidia.py
M	test/registered/unit/layers/attention/linear/kernels/test_kda_ptx.py
D	test/registered/unit/layers/attention/test_aiter_fp8_asm_gqa.py
M	test/registered/unit/layers/attention/test_aiter_verify_metadata.py
M	test/registered/unit/layers/attention/test_cuda_graph_window_rail_completeness.py
M	test/registered/unit/layers/attention/test_dsa_backend_dp_padding.py
M	test/registered/unit/layers/attention/test_dsa_head_gate_guard.py
M	test/registered/unit/layers/attention/test_dsa_indexer_idle.py
M	test/registered/unit/layers/attention/test_dsa_mqa_logits_chunking.py
M	test/registered/unit/layers/attention/test_dsa_q8kv8_bf16_prefill_routing.py
M	test/registered/unit/layers/attention/test_encoder_decoder_varlen_gather.py
M	test/registered/unit/layers/attention/test_flashattention_graph_metadata.py
A	test/registered/unit/layers/attention/test_flashattention_kernel_dispatch.py
M	test/registered/unit/layers/attention/test_flashattention_pa_swa_prefill_lens_size.py
M	test/registered/unit/layers/attention/test_flashattention_return_lse.py
M	test/registered/unit/layers/attention/test_fused_draft_window_metadata.py
M	test/registered/unit/layers/attention/test_gdn_prefill_backend_policy.py
M	test/registered/unit/layers/attention/test_index_topk_share.py
M	test/registered/unit/layers/attention/test_kda_helion_dispatcher.py
M	test/registered/unit/layers/attention/test_kpool_preshuffled_cache_layout.py
M	test/registered/unit/layers/attention/test_kv_shard_hooks.py
M	test/registered/unit/layers/attention/test_kv_translate_ownership.py
M	test/registered/unit/layers/attention/test_linear_attn_config.py
M	test/registered/unit/layers/attention/test_llada2_cfg_flashinfer_backend.py
M	test/registered/unit/layers/attention/test_local_attention.py
M	test/registered/unit/layers/attention/test_mamba_track_state_dtype.py
M	test/registered/unit/layers/attention/test_minimax_indexer_cp_gate.py
A	test/registered/unit/layers/attention/test_minimax_sparse_nvfp4_idle.py
M	test/registered/unit/layers/attention/test_mla_decode_geometry.py
M	test/registered/unit/layers/attention/test_triton_verify_metadata.py
M	test/registered/unit/layers/attention/test_trtllm_mha_paged_kv.py
M	test/registered/unit/layers/attention/test_vattn_asm_segment_plan.py
M	test/registered/unit/layers/attention/test_verify_mask.py
M	test/registered/unit/layers/attention/test_verify_widening_is_wired.py
M	test/registered/unit/layers/attention/test_vision_backend_selection.py
D	test/registered/unit/layers/attention/test_vision_seqlens.py
M	test/registered/unit/layers/attention/test_vision_strided_qkv.py
M	test/registered/unit/layers/deep_gemm_wrapper/test_warmup_memory.py
A	test/registered/unit/layers/moe/paged_experts/test_paged_experts.py
M	test/registered/unit/layers/moe/test_aiter_megamoe_rank_sync.py
M	test/registered/unit/layers/moe/test_aiter_runner.py
D	test/registered/unit/layers/moe/test_copy_weight_views_before_h2d.py
M	test/registered/unit/layers/moe/test_deepep_import_guard.py
M	test/registered/unit/layers/moe/test_deepep_v2_buffer_lifecycle.py
M	test/registered/unit/layers/moe/test_deepep_v2_masked_slab.py
M	test/registered/unit/layers/moe/test_deepep_v2_wire_dtype.py
M	test/registered/unit/layers/moe/test_flashinfer_cutlass_swiglu_params.py
M	test/registered/unit/layers/moe/test_flashinfer_dispatcher.py
M	test/registered/unit/layers/moe/test_flashinfer_megamoe.py
M	test/registered/unit/layers/moe/test_fused_moe_native.py
D	test/registered/unit/layers/moe/test_fused_moe_triton_config.py
M	test/registered/unit/layers/moe/test_fused_shared_expert_scaling.py
M	test/registered/unit/layers/moe/test_mega_moe_decode_capacity.py
M	test/registered/unit/layers/moe/test_mega_moe_deepgemm_api.py
M	test/registered/unit/layers/moe/test_moe_runner_config_placement.py
M	test/registered/unit/layers/moe/test_moe_runner_extensions.py
M	test/registered/unit/layers/moe/test_standard_dispatcher.py
A	test/registered/unit/layers/moe/test_standard_dispatcher_runner_backend.py
M	test/registered/unit/layers/moe/test_topk_correction_bias_cache.py
A	test/registered/unit/layers/moe/test_topk_routing_inputs.py
D	test/registered/unit/layers/moe/test_w4afp8_requant_geometry.py
A	test/registered/unit/layers/muse_spark/test_muse_spark_runner_dispatch.py
D	test/registered/unit/layers/quantization/test_aiter_a16w16_gate.py
M	test/registered/unit/layers/quantization/test_compressed_tensors_fp4_schemes.py
M	test/registered/unit/layers/quantization/test_compressed_tensors_lm_head.py
M	test/registered/unit/layers/quantization/test_compressed_tensors_mixed_precision.py
M	test/registered/unit/layers/quantization/test_compressed_tensors_wna16_moe_no_linear.py
M	test/registered/unit/layers/quantization/test_compressed_tensors_wna16_moe_reload.py
M	test/registered/unit/layers/quantization/test_deepgemm_ue8m0_requant.py
A	test/registered/unit/layers/quantization/test_dense_swiglu_fp8.py
A	test/registered/unit/layers/quantization/test_flashinfer_trtllm_bf16_moe_reload.py
M	test/registered/unit/layers/quantization/test_flashinfer_trtllm_fp8_fallback.py
M	test/registered/unit/layers/quantization/test_fp4_kv_cache_quant_method.py
M	test/registered/unit/layers/quantization/test_fp8_blockwise_linear_backends.py
M	test/registered/unit/layers/quantization/test_fp8_moe_runner_fallback.py
M	test/registered/unit/layers/quantization/test_fp8_moe_runner_ownership.py
M	test/registered/unit/layers/quantization/test_fp8_moe_weight_gating.py
M	test/registered/unit/layers/quantization/test_fp8_utils.py
D	test/registered/unit/layers/quantization/test_fp8_utils_mxfp4.py
A	test/registered/unit/layers/quantization/test_gguf_inactive_routes.py
M	test/registered/unit/layers/quantization/test_int8_linear_methods.py
D	test/registered/unit/layers/quantization/test_marlin_utils_fp8.py
M	test/registered/unit/layers/quantization/test_modelopt_nvfp4.py
M	test/registered/unit/layers/quantization/test_modelopt_nvfp4_moe_dispatch.py
M	test/registered/unit/layers/quantization/test_modelopt_nvfp4_moe_scales.py
A	test/registered/unit/layers/quantization/test_modelslim_mxfp8.py
M	test/registered/unit/layers/quantization/test_moe_wna16_ep_shard.py
M	test/registered/unit/layers/quantization/test_mxfp4_situ_weight_layout.py
M	test/registered/unit/layers/quantization/test_mxfp4_sm100_trtllm_gen.py
M	test/registered/unit/layers/quantization/test_mxfp4_sm120_cutlass.py
M	test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py
A	test/registered/unit/layers/quantization/test_mxfp4_trtllm_autotune_key.py
M	test/registered/unit/layers/quantization/test_nvfp4_embedding.py
M	test/registered/unit/layers/quantization/test_nvfp4_linear_backends.py
M	test/registered/unit/layers/quantization/test_nvfp4_moe_backends.py
A	test/registered/unit/layers/quantization/test_nvfp4_online_weight_loader.py
M	test/registered/unit/layers/quantization/test_quark_config.py
A	test/registered/unit/layers/quantization/test_quark_mxfp4_asm.py
A	test/registered/unit/layers/quantization/test_quark_shared_expert_online_mxfp4_loader.py
M	test/registered/unit/layers/quantization/test_quark_utils.py
M	test/registered/unit/layers/quantization/test_quark_w4a4_mxfp4_moe.py
M	test/registered/unit/layers/quantization/test_quark_w8a8_fp8.py
M	test/registered/unit/layers/quantization/test_quark_w8a8_fp8_ptpc.py
M	test/registered/unit/layers/quantization/test_unquant_apply_with_addend.py
M	test/registered/unit/layers/test_attn_tp_reduce_rejection.py
M	test/registered/unit/layers/test_bailing_mrope_shape.py
M	test/registered/unit/layers/test_conv_layer.py
M	test/registered/unit/layers/test_dp_attention.py
M	test/registered/unit/layers/test_dp_attention_cp_replicas.py
M	test/registered/unit/layers/test_dsv41_candidate_blocks.py
M	test/registered/unit/layers/test_dsv41_candidate_graph_variants.py
M	test/registered/unit/layers/test_dsv41_late_layer_tail_readers.py
M	test/registered/unit/layers/test_dsv4_kv_splits_heuristic.py
M	test/registered/unit/layers/test_dsv4_nonpaged_indexer.py
M	test/registered/unit/layers/test_dual_gemm.py
M	test/registered/unit/layers/test_engram_history.py
M	test/registered/unit/layers/test_flashattention_paged_mha.py
M	test/registered/unit/layers/test_flashinfer_comm_fusion.py
M	test/registered/unit/layers/test_gdn_mis_metadata.py
M	test/registered/unit/layers/test_joint_schema_head.py
M	test/registered/unit/layers/test_kimi_mlp_parallel_groups.py
M	test/registered/unit/layers/test_kpool_planner_cpu_mirror.py
M	test/registered/unit/layers/test_kpool_stream_scheduling.py
M	test/registered/unit/layers/test_layernorm_allreduce_fusion.py
D	test/registered/unit/layers/test_linear_attention_masked_idle_rank.py
M	test/registered/unit/layers/test_linear_logical_groups.py
M	test/registered/unit/layers/test_linear_parallel_groups.py
M	test/registered/unit/layers/test_logprob_fast_input.py
M	test/registered/unit/layers/test_mamba2_track_ssm_indices.py
M	test/registered/unit/layers/test_mamba_prefill_track_metadata.py
M	test/registered/unit/layers/test_mamba_state_scatter_triton.py
A	test/registered/unit/layers/test_mamba_track_states_triton.py
M	test/registered/unit/layers/test_minicpm_attention_adapter.py
M	test/registered/unit/layers/test_minicpm_sparse_cache.py
M	test/registered/unit/layers/test_minicpm_sparse_metadata.py
M	test/registered/unit/layers/test_mova.py
M	test/registered/unit/layers/test_radix_attention.py
M	test/registered/unit/layers/test_radix_attention_padded_extend.py
A	test/registered/unit/layers/test_rope_hip_fallback_cache_dtype.py
D	test/registered/unit/layers/test_xielu.py
M	test/registered/unit/layers/test_yarn_cache_extension.py
M	test/registered/unit/lora/test_experimental_sgl_marlin_policy.py
M	test/registered/unit/lora/test_experimental_sgl_marlin_runtime_unit.py
M	test/registered/unit/lora/test_experimental_sgl_marlin_shared_outer_reduce.py
M	test/registered/unit/lora/test_inkling_linearized_lora_unit.py
M	test/registered/unit/lora/test_laguna_hidden_dim_unit.py
A	test/registered/unit/lora/test_lora_forward_with_base.py
A	test/registered/unit/lora/test_lora_inactive_batch_skip.py
M	test/registered/unit/lora/test_lora_manager_init_modules.py
M	test/registered/unit/lora/test_lora_moe_inplace_unit.py
M	test/registered/unit/lora/test_lora_spec_verify_batch_info.py
M	test/registered/unit/lora/test_mem_pool_ep_unit.py
A	test/registered/unit/lora/test_moe_lora_gemm_config_store.py
A	test/registered/unit/lora/test_moe_lora_masked_capacity.py
A	test/registered/unit/lora/test_moe_lora_plan_tables.py
A	test/registered/unit/lora/test_moe_lora_prefill_graph_contract.py
A	test/registered/unit/lora/test_moe_lora_provider_contract.py
A	test/registered/unit/lora/test_moe_lora_route_factory.py
A	test/registered/unit/lora/test_moe_lora_stage_dispatch.py
M	test/registered/unit/lora/test_triton_dp_attention_unit.py
M	test/registered/unit/lora/test_uno_inactive_lora_batch.py
M	test/registered/unit/lora/test_uno_lora_targets.py
M	test/registered/unit/managers/scheduler_components/test_dp_attn.py
M	test/registered/unit/managers/scheduler_components/test_dynamic_chunk_sizer.py
M	test/registered/unit/managers/scheduler_components/test_invariant_checker.py
M	test/registered/unit/managers/scheduler_components/test_output_sender.py
A	test/registered/unit/managers/scheduler_components/test_shm_tag_exchange.py
M	test/registered/unit/managers/test_async_d2h_copy_worker.py
M	test/registered/unit/managers/test_auxiliary_output.py
M	test/registered/unit/managers/test_bailing_modality_metadata.py
M	test/registered/unit/managers/test_batch_result_processor_hidden_states.py
M	test/registered/unit/managers/test_batch_result_processor_mamba_boundary.py
M	test/registered/unit/managers/test_batch_result_processor_spec_grammar.py
M	test/registered/unit/managers/test_data_parallel_controller.py
M	test/registered/unit/managers/test_dcp_logical_capacity.py
A	test/registered/unit/managers/test_deferred_output.py
M	test/registered/unit/managers/test_detailed_annotations.py
M	test/registered/unit/managers/test_deterministic_chunk_align.py
M	test/registered/unit/managers/test_disagg_idle_step_counters.py
M	test/registered/unit/managers/test_embed_overrides.py
M	test/registered/unit/managers/test_exportable_env_vars.py
M	test/registered/unit/managers/test_finish_length_speculative.py
M	test/registered/unit/managers/test_flat_raw_top_logprobs.py
M	test/registered/unit/managers/test_grammar_stop_speculative.py
M	test/registered/unit/managers/test_hidden_state_server_mode.py
M	test/registered/unit/managers/test_hisparse_spec_coordinator.py
M	test/registered/unit/managers/test_hisparse_unit.py
M	test/registered/unit/managers/test_io_struct.py
M	test/registered/unit/managers/test_kv_page_invariants.py
M	test/registered/unit/managers/test_load_inquirer.py
M	test/registered/unit/managers/test_load_snapshot_backends.py
M	test/registered/unit/managers/test_lora_update_result_merge.py
M	test/registered/unit/managers/test_mamba_checkpoint_depth.py
A	test/registered/unit/managers/test_mix_with_running_mamba_track.py
M	test/registered/unit/managers/test_mm_hashes.py
A	test/registered/unit/managers/test_mm_pad_input_tokens.py
M	test/registered/unit/managers/test_mm_process_config.py
M	test/registered/unit/managers/test_mm_shm_error_consensus.py
M	test/registered/unit/managers/test_mm_token_counts.py
M	test/registered/unit/managers/test_mm_utils_split.py
M	test/registered/unit/managers/test_multi_tokenizer_mixin.py
M	test/registered/unit/managers/test_multimodal_abort_cleanup.py
M	test/registered/unit/managers/test_output_streamer_customized_info.py
M	test/registered/unit/managers/test_output_streamer_logprobs.py
M	test/registered/unit/managers/test_pp_cp_rank_offsets.py
M	test/registered/unit/managers/test_pp_output_exchange_order.py
M	test/registered/unit/managers/test_prefill_adder.py
M	test/registered/unit/managers/test_prefill_delayer_gather.py
M	test/registered/unit/managers/test_prefill_delayer_high_watermark.py
M	test/registered/unit/managers/test_priority_scheduling_disaggregation.py
M	test/registered/unit/managers/test_req_custom_params_ref.py
M	test/registered/unit/managers/test_request_class_override_points.py
M	test/registered/unit/managers/test_retraction_order.py
M	test/registered/unit/managers/test_sampling_mask_validation.py
M	test/registered/unit/managers/test_schedule_batch_convert_decode_to_extend.py
M	test/registered/unit/managers/test_schedule_batch_out_of_place.py
M	test/registered/unit/managers/test_schedule_batch_prepare_for_decode.py
M	test/registered/unit/managers/test_schedule_batch_req_pool_indices.py
M	test/registered/unit/managers/test_schedule_policy.py
D	test/registered/unit/managers/test_schedule_policy_dfs_weight.py
D	test/registered/unit/managers/test_scheduler_batch_is_full_latch.py
M	test/registered/unit/managers/test_scheduler_chunked_abort_race.py
M	test/registered/unit/managers/test_scheduler_chunked_req_gate.py
M	test/registered/unit/managers/test_scheduler_dp_converted_batch_full.py
M	test/registered/unit/managers/test_scheduler_flush_cache.py
M	test/registered/unit/managers/test_scheduler_flush_cache_after_retract.py
A	test/registered/unit/managers/test_scheduler_health_check_signal.py
M	test/registered/unit/managers/test_scheduler_held_row_admission.py
M	test/registered/unit/managers/test_scheduler_hicache_attach.py
M	test/registered/unit/managers/test_scheduler_hicache_events.py
M	test/registered/unit/managers/test_scheduler_init_req_max_new_tokens.py
M	test/registered/unit/managers/test_scheduler_on_idle_load.py
M	test/registered/unit/managers/test_scheduler_pause_generation.py
M	test/registered/unit/managers/test_scheduler_prefill_decode_interval.py
M	test/registered/unit/managers/test_scheduler_recv_skipper.py
D	test/registered/unit/managers/test_scheduler_running_batch_finished.py
M	test/registered/unit/managers/test_scheduler_sampling_mask_validation.py
M	test/registered/unit/managers/test_scheduler_timeouts.py
M	test/registered/unit/managers/test_scheduler_weight_version_tracking.py
M	test/registered/unit/managers/test_setwise_score_mixin.py
M	test/registered/unit/managers/test_stop_str_speculative.py
M	test/registered/unit/managers/test_tokenizer_config_updates.py
M	test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py
M	test/registered/unit/managers/test_trim_matched_stop.py
D	test/registered/unit/managers/test_uno_token_accounting.py
M	test/registered/unit/managers/test_vocab_boundary_finish.py
M	test/registered/unit/mem_cache/pool_host/test_kvcacheio_asymmetric.py
M	test/registered/unit/mem_cache/test_asymmetric_mha_pool.py
M	test/registered/unit/mem_cache/test_asymmetric_mha_pool_host_unit.py
M	test/registered/unit/mem_cache/test_buffer_mode_sidecar.py
M	test/registered/unit/mem_cache/test_buffer_swa_tail_only.py
M	test/registered/unit/mem_cache/test_component_split_contract.py
M	test/registered/unit/mem_cache/test_decode_radix_cache_validation.py
M	test/registered/unit/mem_cache/test_decode_radix_lock_ref.py
M	test/registered/unit/mem_cache/test_decode_retraction_backup.py
M	test/registered/unit/mem_cache/test_dense_fused_draft_geometry.py
M	test/registered/unit/mem_cache/test_dllm_fdfo_kv_reuse.py
A	test/registered/unit/mem_cache/test_draft_depth_state.py
M	test/registered/unit/mem_cache/test_dsa_layer_shard_utils.py
M	test/registered/unit/mem_cache/test_dsa_pool_host_unit.py
M	test/registered/unit/mem_cache/test_dsv4_c4_state_lifecycle.py
M	test/registered/unit/mem_cache/test_dsv4_compressed_pools.py
M	test/registered/unit/mem_cache/test_dsv4_hicache_l2.py
M	test/registered/unit/mem_cache/test_dsv4_unified_fp8_pool.py
M	test/registered/unit/mem_cache/test_evict_policy.py
A	test/registered/unit/mem_cache/test_external_integration_surface.py
A	test/registered/unit/mem_cache/test_external_linker_factory.py
M	test/registered/unit/mem_cache/test_fcfs_waiting_prefix_refresh.py
M	test/registered/unit/mem_cache/test_flashkda_strided_state_access.py
M	test/registered/unit/mem_cache/test_full_loc_fast_path.py
M	test/registered/unit/mem_cache/test_fused_draft_placement.py
M	test/registered/unit/mem_cache/test_hicache_auto_size.py
M	test/registered/unit/mem_cache/test_hicache_copy_rounds.py
M	test/registered/unit/mem_cache/test_hicache_dcp_host_pool.py
M	test/registered/unit/mem_cache/test_hicache_file_lru_unit.py
M	test/registered/unit/mem_cache/test_hicache_host_register.py
A	test/registered/unit/mem_cache/test_hicache_kvcacheio_gate.py
M	test/registered/unit/mem_cache/test_hicache_load_back_timing.py
M	test/registered/unit/mem_cache/test_hicache_mtp_state.py
M	test/registered/unit/mem_cache/test_hicache_nixl_cleaner.py
M	test/registered/unit/mem_cache/test_hicache_nixl_storage.py
M	test/registered/unit/mem_cache/test_hicache_pp_sync_drain.py
M	test/registered/unit/mem_cache/test_hicache_seaweedfs_storage.py
M	test/registered/unit/mem_cache/test_hicache_staged_write_back_dispatch.py
M	test/registered/unit/mem_cache/test_hicache_storage_format_tag.py
A	test/registered/unit/mem_cache/test_hicache_storage_registration_hook.py
A	test/registered/unit/mem_cache/test_hicache_unified_file_chunks.py
A	test/registered/unit/mem_cache/test_hicache_unified_key_scheme.py
M	test/registered/unit/mem_cache/test_hicache_write_back_ordering.py
M	test/registered/unit/mem_cache/test_hisparse_allocator.py
M	test/registered/unit/mem_cache/test_hisparse_max_token_pool_size.py
M	test/registered/unit/mem_cache/test_hisparse_slot_translation.py
M	test/registered/unit/mem_cache/test_host_memory.py
M	test/registered/unit/mem_cache/test_hybrid_pool_assembler.py
A	test/registered/unit/mem_cache/test_index_key_cache_move.py
M	test/registered/unit/mem_cache/test_inkling_sconv_strided_conv_state.py
R084	test/registered/unit/mem_cache/test_kv_loc_plan.py	test/registered/unit/mem_cache/test_iteration_ids.py
M	test/registered/unit/mem_cache/test_kda_fused_decode_strided_state.py
A	test/registered/unit/mem_cache/test_kv_cache_configurator_decode_table.py
M	test/registered/unit/mem_cache/test_kv_cache_configurator_draft_gate.py
A	test/registered/unit/mem_cache/test_kv_cache_configurator_npu_indexer.py
M	test/registered/unit/mem_cache/test_kv_control_cache.py
M	test/registered/unit/mem_cache/test_kv_control_transfers.py
A	test/registered/unit/mem_cache/test_kv_layer_access.py
A	test/registered/unit/mem_cache/test_kv_layer_router.py
A	test/registered/unit/mem_cache/test_kv_layer_router_parts.py
M	test/registered/unit/mem_cache/test_kv_plan_uniform_extend.py
A	test/registered/unit/mem_cache/test_kv_storage_lifecycle.py
M	test/registered/unit/mem_cache/test_kv_stride_hygiene.py
D	test/registered/unit/mem_cache/test_layout_compat.py
M	test/registered/unit/mem_cache/test_linker_pool_assembler.py
M	test/registered/unit/mem_cache/test_lmcache_component_cursors.py
M	test/registered/unit/mem_cache/test_mamba_capacity.py
M	test/registered/unit/mem_cache/test_mamba_donated_alloc_ratio.py
M	test/registered/unit/mem_cache/test_mamba_path_state_cap.py
M	test/registered/unit/mem_cache/test_mamba_state_transfer_buffers.py
M	test/registered/unit/mem_cache/test_mamba_unittest.py
M	test/registered/unit/mem_cache/test_mem_cache_utils.py
M	test/registered/unit/mem_cache/test_mem_pool_host.py
A	test/registered/unit/mem_cache/test_mha_pool_fused_kv_layout.py
A	test/registered/unit/mem_cache/test_minimax_m31_cache_transfer.py
M	test/registered/unit/mem_cache/test_minimax_sparse_pool_host_unit.py
M	test/registered/unit/mem_cache/test_minimax_sparse_pool_pd_unit.py
M	test/registered/unit/mem_cache/test_mla_host_dedup_primitives.py
M	test/registered/unit/mem_cache/test_mmap_allocator.py
M	test/registered/unit/mem_cache/test_mooncake_group_semantics.py
A	test/registered/unit/mem_cache/test_mooncake_host_allocator.py
M	test/registered/unit/mem_cache/test_mooncake_ssd_offload_path.py
M	test/registered/unit/mem_cache/test_mooncake_store_config.py
M	test/registered/unit/mem_cache/test_mooncake_tenant_config.py
M	test/registered/unit/mem_cache/test_multi_ended_allocator.py
M	test/registered/unit/mem_cache/test_mxfp8_scale_transfer_buffers.py
A	test/registered/unit/mem_cache/test_npu_index_k_elision.py
M	test/registered/unit/mem_cache/test_npu_mamba_async_layout.py
A	test/registered/unit/mem_cache/test_page_interleave_gather_multi_gpu.py
M	test/registered/unit/mem_cache/test_page_interleave_shard.py
A	test/registered/unit/mem_cache/test_page_sharded_dsa_attention.py
M	test/registered/unit/mem_cache/test_page_unified_layout.py
A	test/registered/unit/mem_cache/test_page_unified_staging.py
M	test/registered/unit/mem_cache/test_paged_allocator_lazy_release.py
M	test/registered/unit/mem_cache/test_paged_view.py
M	test/registered/unit/mem_cache/test_pd_envelope_transfer_layout.py
M	test/registered/unit/mem_cache/test_physical_write_loc.py
M	test/registered/unit/mem_cache/test_pp_prefetch_ticket.py
M	test/registered/unit/mem_cache/test_prefill_memory_budget.py
M	test/registered/unit/mem_cache/test_prefix_valid_fp8_pool.py
M	test/registered/unit/mem_cache/test_pure_swa_radix_cache.py
D	test/registered/unit/mem_cache/test_qsa_kv_pool.py
M	test/registered/unit/mem_cache/test_quantized_kv_pool.py
D	test/registered/unit/mem_cache/test_radix_cache_slru_accuracy.py
M	test/registered/unit/mem_cache/test_radix_cache_unit.py
M	test/registered/unit/mem_cache/test_radix_force_miss.py
M	test/registered/unit/mem_cache/test_registry.py
M	test/registered/unit/mem_cache/test_replayssm_ring_accounting.py
A	test/registered/unit/mem_cache/test_request_table.py
M	test/registered/unit/mem_cache/test_retraction_mamba_backup.py
A	test/registered/unit/mem_cache/test_routed_kv_cache_shape.py
R088	test/registered/unit/mem_cache/test_kv_index_translator.py	test/registered/unit/mem_cache/test_runner_ids.py
R078	test/registered/unit/mem_cache/test_kv_index_translator_draft_disposition.py	test/registered/unit/mem_cache/test_runner_ids_draft_disposition.py
M	test/registered/unit/mem_cache/test_rust_tree_core.py
M	test/registered/unit/mem_cache/test_rust_tree_core_errors.py
M	test/registered/unit/mem_cache/test_rust_tree_core_integration.py
D	test/registered/unit/mem_cache/test_rust_unified_radix_cache_unittest.py
M	test/registered/unit/mem_cache/test_session_token_share_unit.py
M	test/registered/unit/mem_cache/test_session_unified_radix_cache.py
A	test/registered/unit/mem_cache/test_sharded_dsa_hicache.py
A	test/registered/unit/mem_cache/test_state_acquire.py
M	test/registered/unit/mem_cache/test_storage_prefetch_lifecycle.py
M	test/registered/unit/mem_cache/test_streaming_session_unit.py
M	test/registered/unit/mem_cache/test_swa_alloc_extend_page_estimation.py
M	test/registered/unit/mem_cache/test_swa_cpu_copy_filter.py
M	test/registered/unit/mem_cache/test_swa_eviction_interval.py
M	test/registered/unit/mem_cache/test_swa_locked_full_recover_unified.py
M	test/registered/unit/mem_cache/test_swa_pool_v_head_dim.py
M	test/registered/unit/mem_cache/test_swa_ring_page_return.py
M	test/registered/unit/mem_cache/test_swa_unittest.py
M	test/registered/unit/mem_cache/test_tlru_eviction_policy.py
M	test/registered/unit/mem_cache/test_tree_core_registry.py
M	test/registered/unit/mem_cache/test_umbp_direct_linker_rank_keys.py
M	test/registered/unit/mem_cache/test_umbp_host_allocator.py
A	test/registered/unit/mem_cache/test_unified_allocator_trace.py
M	test/registered/unit/mem_cache/test_unified_byte_accounting.py
M	test/registered/unit/mem_cache/test_unified_byte_budget_sizing.py
M	test/registered/unit/mem_cache/test_unified_cache_linker.py
M	test/registered/unit/mem_cache/test_unified_capacity_memo.py
M	test/registered/unit/mem_cache/test_unified_conv_compaction.py
M	test/registered/unit/mem_cache/test_unified_dynamic_gate_capacity.py
M	test/registered/unit/mem_cache/test_unified_float_move_gate.py
M	test/registered/unit/mem_cache/test_unified_free_no_host_sync.py
M	test/registered/unit/mem_cache/test_unified_handout_zeroing.py
M	test/registered/unit/mem_cache/test_unified_hicache_regressions.py
M	test/registered/unit/mem_cache/test_unified_hicache_strided_state.py
M	test/registered/unit/mem_cache/test_unified_hicache_transfer_lifecycle.py
M	test/registered/unit/mem_cache/test_unified_joint_allocation_eviction.py
A	test/registered/unit/mem_cache/test_unified_kv_storage.py
A	test/registered/unit/mem_cache/test_unified_layout_golden.py
M	test/registered/unit/mem_cache/test_unified_mamba_compaction.py
M	test/registered/unit/mem_cache/test_unified_mamba_views.py
M	test/registered/unit/mem_cache/test_unified_mha_views.py
M	test/registered/unit/mem_cache/test_unified_mla_block_table.py
M	test/registered/unit/mem_cache/test_unified_mla_gpu_parity.py
M	test/registered/unit/mem_cache/test_unified_mla_views.py
M	test/registered/unit/mem_cache/test_unified_npool_sweep.py
M	test/registered/unit/mem_cache/test_unified_pending_event_batches.py
M	test/registered/unit/mem_cache/test_unified_post_capture_sizing.py
A	test/registered/unit/mem_cache/test_unified_pricing_property.py
M	test/registered/unit/mem_cache/test_unified_radix_allocation_eviction.py
M	test/registered/unit/mem_cache/test_unified_radix_cache_bench.py
D	test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py
M	test/registered/unit/mem_cache/test_unified_radix_hicache_dispatch.py
D	test/registered/unit/mem_cache/test_unified_radix_lock_ref.py
M	test/registered/unit/mem_cache/test_unified_swa_shared_virtual_ids.py
M	test/registered/unit/mem_cache/test_unified_swa_tail_allocation.py
M	test/registered/unit/mem_cache/test_unified_tri_joint_reclaim.py
M	test/registered/unit/mem_cache/test_unified_tri_pool.py
D	test/registered/unit/mem_cache/test_uno_allocation_sizing.py
A	test/registered/unit/mem_cache/test_window_read.py
A	test/registered/unit/mem_cache/unified_layout_golden.json
A	test/registered/unit/mem_cache/unified_radix_cache/test_unified_radix_cache_core.py
A	test/registered/unit/mem_cache/unified_radix_cache/test_unified_radix_cache_hicache.py
A	test/registered/unit/mem_cache/unified_radix_cache/test_unified_radix_cache_insert_walk.py
A	test/registered/unit/mem_cache/unified_radix_cache/test_unified_radix_cache_kv_events.py
A	test/registered/unit/mem_cache/unified_radix_cache/test_unified_radix_cache_load_back.py
A	test/registered/unit/mem_cache/unified_radix_cache/test_unified_radix_cache_rust.py
A	test/registered/unit/mem_cache/unified_radix_cache/test_unified_radix_cache_storage.py
M	test/registered/unit/memory/test_ptr_table.py
M	test/registered/unit/model_executor/model_runner_components/test_attention_backend_setup.py
A	test/registered/unit/model_executor/model_runner_components/test_aux_hidden_state_capture.py
M	test/registered/unit/model_executor/model_runner_components/test_cuda_graph_setup.py
A	test/registered/unit/model_executor/model_runner_components/test_dcp_target_verify_backend.py
M	test/registered/unit/model_executor/model_runner_components/test_layer_setup.py
M	test/registered/unit/model_executor/model_runner_components/test_spec_aux_hidden_state.py
M	test/registered/unit/model_executor/runner/test_decode_cuda_graph_runner.py
M	test/registered/unit/model_executor/runner/test_decode_cuda_graph_shared_read_fence.py
A	test/registered/unit/model_executor/runner/test_dsv41_draft_metadata_glue.py
A	test/registered/unit/model_executor/runner/test_dsv41_target_verify_metadata_glue.py
A	test/registered/unit/model_executor/runner/test_early_verify_metadata.py
M	test/registered/unit/model_executor/runner/test_flashinfer_autotune_sync.py
A	test/registered/unit/model_executor/runner/test_forked_draft_metadata_prep.py
A	test/registered/unit/model_executor/runner/test_graph_routing_context.py
M	test/registered/unit/model_executor/runner/test_prefill_cuda_graph_padding.py
M	test/registered/unit/model_executor/runner/test_prefill_shared_read_done.py
M	test/registered/unit/model_executor/runner_backend/test_backend_resolution.py
M	test/registered/unit/model_executor/runner_utils/test_graph_pool_borrow.py
M	test/registered/unit/model_executor/runner_utils/test_imports.py
A	test/registered/unit/model_executor/test_aux_hidden_state_row_alignment.py
M	test/registered/unit/model_executor/test_chunked_prefix_cache_gate.py
M	test/registered/unit/model_executor/test_compensated_mhc_update_guard.py
M	test/registered/unit/model_executor/test_cuda_graph_buffer_registry.py
M	test/registered/unit/model_executor/test_cuda_graph_config_backends.py
M	test/registered/unit/model_executor/test_derived_weight_cache.py
M	test/registered/unit/model_executor/test_draft_graph_recapture.py
M	test/registered/unit/model_executor/test_draft_runner_skips_lora.py
M	test/registered/unit/model_executor/test_eager_on_graph.py
M	test/registered/unit/model_executor/test_elastic_cuda_graph_setup.py
M	test/registered/unit/model_executor/test_encoder_swa_replay.py
M	test/registered/unit/model_executor/test_forward_metadata_plan_record.py
A	test/registered/unit/model_executor/test_forward_observer.py
M	test/registered/unit/model_executor/test_hisparse_pool_configurator.py
M	test/registered/unit/model_executor/test_kv_canary_headroom.py
M	test/registered/unit/model_executor/test_mlp_sync_hybrid_target_verify.py
M	test/registered/unit/model_executor/test_mlp_sync_pad_unpad.py
M	test/registered/unit/model_executor/test_num_token_non_padded_localization.py
M	test/registered/unit/model_executor/test_pool_configurator.py
M	test/registered/unit/model_executor/test_prefill_cuda_graph_runner.py
M	test/registered/unit/model_executor/test_prefill_cuda_graph_runner_helpers.py
M	test/registered/unit/model_executor/test_spec_aux_hidden_state_resolver.py
M	test/registered/unit/model_executor/test_unified_out_cache_loc_rebind.py
M	test/registered/unit/model_executor/test_weight_update_tensor_rank.py
M	test/registered/unit/model_loader/test_default_model_loader.py
M	test/registered/unit/model_loader/test_instanttensor_loader.py
M	test/registered/unit/model_loader/test_prefetch_checkpoints.py
M	test/registered/unit/model_loader/test_presharded_loader.py
M	test/registered/unit/model_loader/test_quantized_reload_scale_layout.py
M	test/registered/unit/model_loader/test_remote_instance_loader.py
M	test/registered/unit/model_loader/test_runai_model_streamer_loader.py
M	test/registered/unit/model_loader/test_weight_utils.py
M	test/registered/unit/models/test_bailing_vl_loader.py
M	test/registered/unit/models/test_cosmos3.py
M	test/registered/unit/models/test_cosmos3_edge.py
M	test/registered/unit/models/test_deepseek_moe_transport.py
M	test/registered/unit/models/test_deepseek_nextn_mm_embed.py
M	test/registered/unit/models/test_deepseek_pp_mtp_embedding.py
A	test/registered/unit/models/test_deepseek_quark_block_fp8_loading.py
M	test/registered/unit/models/test_deepseek_v41_bcg_multimodal.py
M	test/registered/unit/models/test_deepseek_v4_amd_fused_mhc.py
M	test/registered/unit/models/test_deepseek_v4_amd_tp4.py
M	test/registered/unit/models/test_deepseek_v4_boundary.py
M	test/registered/unit/models/test_deepseek_v4_cp_kv_store.py
M	test/registered/unit/models/test_deepseek_v4_pipeline.py
M	test/registered/unit/models/test_deepseek_v4_platform_dispatch.py
M	test/registered/unit/models/test_deepseek_v4_rope_policy.py
M	test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py
A	test/registered/unit/models/test_deepseek_v4_tbo_collectives.py
M	test/registered/unit/models/test_deepseek_v4_unified_fp8_q_pair.py
M	test/registered/unit/models/test_draft_entry_hook_parity.py
M	test/registered/unit/models/test_falcon_h1_mixer_sums.py
A	test/registered/unit/models/test_fused_set_kv_buffer_gate.py
M	test/registered/unit/models/test_glm5_next_bfg_fusion.py
M	test/registered/unit/models/test_glm5_next_modelopt.py
M	test/registered/unit/models/test_glm_nextn_moe_ptpc.py
M	test/registered/unit/models/test_glmmoedsa_correction_bias_fp32.py
M	test/registered/unit/models/test_gpt_oss_runai_ownership.py
M	test/registered/unit/models/test_granitemoe_split_expert_loading.py
M	test/registered/unit/models/test_hunyuan_v3_nextn_weight_loading.py
M	test/registered/unit/models/test_hunyuan_v4.py
M	test/registered/unit/models/test_hunyuan_v4_nextn_weight_loading.py
A	test/registered/unit/models/test_inkling_aux_hidden_states.py
D	test/registered/unit/models/test_inkling_graph_fusion_gates.py
M	test/registered/unit/models/test_interns1pro_processor.py
M	test/registered/unit/models/test_iquest_q1.py
M	test/registered/unit/models/test_kimi_k25.py
M	test/registered/unit/models/test_kimi_k3_bfa_overlap.py
M	test/registered/unit/models/test_kimi_k3_packed_modules_mapping.py
M	test/registered/unit/models/test_kimi_k3_pipeline_layers.py
M	test/registered/unit/models/test_kimi_k3_rocm_quant.py
M	test/registered/unit/models/test_kimi_k3_vision.py
M	test/registered/unit/models/test_kimi_vl.py
M	test/registered/unit/models/test_kimi_vl_moonvit.py
M	test/registered/unit/models/test_llada2.py
M	test/registered/unit/models/test_llava.py
M	test/registered/unit/models/test_llava_processor_pool.py
M	test/registered/unit/models/test_locate_anything.py
M	test/registered/unit/models/test_longcat_flash_router_hpc_gemm.py
R094	test/srt/models/test_mamba2.py	test/registered/unit/models/test_mamba2.py
M	test/registered/unit/models/test_mellum.py
A	test/registered/unit/models/test_minimax_m31_model.py
M	test/registered/unit/models/test_mistral3_vision_feature.py
M	test/registered/unit/models/test_model_registry.py
M	test/registered/unit/models/test_moss_vl_processor.py
M	test/registered/unit/models/test_nano_nemotron_vl.py
A	test/registered/unit/models/test_native_moe_routing_policy.py
M	test/registered/unit/models/test_nemotron_h_mtp.py
M	test/registered/unit/models/test_nemotron_h_shared_add.py
M	test/registered/unit/models/test_nemotron_h_weight_loading.py
M	test/registered/unit/models/test_nope_mha_k_cast.py
M	test/registered/unit/models/test_qwen3_5_mtp_quant_config.py
M	test/registered/unit/models/test_qwen3_5_packed_weight_loader.py
M	test/registered/unit/models/test_qwen3_embedding_registration.py
M	test/registered/unit/models/test_qwen4_exp_ple_table.py
M	test/registered/unit/models/test_radio.py
M	test/registered/unit/models/test_rocm_hisparse_fused_kv.py
M	test/registered/unit/models/test_rocm_mla_absorb_fusion_gate.py
M	test/registered/unit/models/test_shared_experts_fusion_gates.py
M	test/registered/unit/models/test_solar_pipeline_split.py
M	test/registered/unit/models/test_vit_pos_embed_interpolate.py
M	test/registered/unit/models/test_xllm.py
M	test/registered/unit/models/test_zaya_mod_tp.py
A	test/registered/unit/moe/test_ascend_ops.py
M	test/registered/unit/multimodal/rust/inkling/test_bindings.py
M	test/registered/unit/multimodal/rust/qwen/test_e2e_parity.py
M	test/registered/unit/multimodal/rust/qwen/test_preprocess.py
M	test/registered/unit/multimodal/rust/qwen/test_token_layout_mrope.py
M	test/registered/unit/multimodal/rust/shared/test_fetch.py
M	test/registered/unit/multimodal/rust/shared/test_image_decode.py
M	test/registered/unit/multimodal/rust/shared/test_rust_mm_gate.py
M	test/registered/unit/multimodal/rust/shared/test_rust_server_extension.py
M	test/registered/unit/multimodal/rust/shared/test_wrap_encoded.py
M	test/registered/unit/multimodal/test_bailing_mm_processor.py
M	test/registered/unit/multimodal/test_base_processor_bad_input.py
M	test/registered/unit/multimodal/test_base_processor_image_decode.py
M	test/registered/unit/multimodal/test_cuda_ipc_transport.py
M	test/registered/unit/multimodal/test_cuda_vmm_transport.py
M	test/registered/unit/multimodal/test_glm4v_mixed_offsets.py
M	test/registered/unit/multimodal/test_gpu_feature_transport.py
A	test/registered/unit/multimodal/test_inkling_media_errors.py
M	test/registered/unit/multimodal/test_kimi_image_identity.py
M	test/registered/unit/multimodal/test_media_artifact_processor.py
M	test/registered/unit/multimodal/test_media_url_security.py
M	test/registered/unit/multimodal/test_minicpmv4_7_canvas_mrope.py
A	test/registered/unit/multimodal/test_minimax_m31_processor.py
M	test/registered/unit/multimodal/test_mrope_encoder_utils.py
M	test/registered/unit/multimodal/test_nano_nemotron_vl_processor.py
M	test/registered/unit/multimodal/test_pixtral_processor.py
M	test/registered/unit/multimodal/test_precomputed_embedding_validation.py
M	test/registered/unit/multimodal/test_processor_async_call_sites.py
M	test/registered/unit/multimodal/test_processor_clone_isolation.py
M	test/registered/unit/multimodal/test_processor_device_selection.py
M	test/registered/unit/multimodal/test_vit_npu_graph_runner.py
A	test/registered/unit/multimodal_gen/test_flux3_swiglu_fp8.py
A	test/registered/unit/multimodal_gen/test_image_preprocessing.py
M	test/registered/unit/multimodal_gen/test_lora_merge_cache_key.py
M	test/registered/unit/npu/attention/test_npu_dcp.py
M	test/registered/unit/npu/graph_runner/test_npu_cudagraph_backend.py
M	test/registered/unit/npu/graph_runner/test_npu_graph_runner.py
A	test/registered/unit/npu/test_npu_gemma_rmsnorm.py
A	test/registered/unit/npu/test_npu_scatter_pa_kv_cache.py
M	test/registered/unit/npu/test_sparsity_driven_kv_offload_config.py
M	test/registered/unit/observability/test_cpu_monitor.py
M	test/registered/unit/observability/test_forward_pass_metrics.py
M	test/registered/unit/observability/test_func_timer.py
A	test/registered/unit/observability/test_http_request_metrics.py
M	test/registered/unit/observability/test_label_transform.py
M	test/registered/unit/observability/test_metrics_utils.py
A	test/registered/unit/observability/test_prealloc_token_usage.py
M	test/registered/unit/observability/test_ray_wrappers.py
M	test/registered/unit/observability/test_req_time_stats.py
M	test/registered/unit/observability/test_request_metrics_exporter.py
M	test/registered/unit/observability/test_rust_metrics_fixtures.py
M	test/registered/unit/observability/test_scheduler_stage_metrics.py
M	test/registered/unit/observability/test_startup_func_log_and_timer.py
M	test/registered/unit/observability/test_stat_loggers_di.py
M	test/registered/unit/observability/test_trace.py
A	test/registered/unit/parser/inkling_tmlv0_golden.json
A	test/registered/unit/parser/test_beam_response_template.py
M	test/registered/unit/parser/test_chat_parsing.py
M	test/registered/unit/parser/test_code_completion_parser.py
M	test/registered/unit/parser/test_conversation.py
A	test/registered/unit/parser/test_deepseek_recipe_renderer.py
A	test/registered/unit/parser/test_glm53_chat_template.py
M	test/registered/unit/parser/test_harmony_parser.py
M	test/registered/unit/parser/test_hunyuan_reasoning.py
M	test/registered/unit/parser/test_inkling_renderer.py
M	test/registered/unit/parser/test_iquest_q1_reasoning_parser.py
M	test/registered/unit/parser/test_jinja_template_utils.py
M	test/registered/unit/parser/test_k2_v3_reasoning_parser.py
M	test/registered/unit/parser/test_kimik3_reasoning_parser.py
M	test/registered/unit/parser/test_reasoning_parser.py
A	test/registered/unit/parser/test_response_template.py
M	test/registered/unit/parser/test_template_manager.py
M	test/registered/unit/platforms/test_platform_interface.py
M	test/registered/unit/plugins/test_hook_registry.py
M	test/registered/unit/plugins/test_load_plugins.py
M	test/registered/unit/sampling/test_ascend_sampler.py
M	test/registered/unit/sampling/test_penaltylib.py
M	test/registered/unit/sampling/test_sampling_batch_info.py
M	test/registered/unit/sampling/test_sampling_metadata_staging.py
M	test/registered/unit/sampling/test_sampling_params.py
M	test/registered/unit/sampling/test_torch_sampling_top_k_bound.py
M	test/registered/unit/scripted_runtime/test_tree_core_inspection.py
M	test/registered/unit/server_args/test_attn_dp_size.py
M	test/registered/unit/server_args/test_compute_world_size.py
M	test/registered/unit/server_args/test_declared_fallbacks.py
M	test/registered/unit/server_args/test_mamba_extra_buffer_platform.py
M	test/registered/unit/server_args/test_model_config_cache.py
M	test/registered/unit/server_args/test_page_major_backend_allowlist.py
M	test/registered/unit/server_args/test_platform_prefill_cp_deprecation.py
M	test/registered/unit/server_args/test_record_holds_the_raw_input.py
M	test/registered/unit/server_args/test_resolution_declarations.py
M	test/registered/unit/server_args/test_resolution_hook_registry.py
M	test/registered/unit/server_args/test_resolution_is_reproducible.py
M	test/registered/unit/server_args/test_server_args.py
M	test/registered/unit/server_args/test_server_args_docs_coverage.py
M	test/registered/unit/server_args/test_unified_memory_spec_gate.py
M	test/registered/unit/server_args/test_unified_prefill_cuda_graph_gate.py
M	test/registered/unit/server_args/test_unified_tbo_gate.py
M	test/registered/unit/spec/test_adaptive_runtime_state.py
M	test/registered/unit/spec/test_adaptive_spec_params.py
M	test/registered/unit/spec/test_block_verification_args.py
M	test/registered/unit/spec/test_decode_bookkeeping_ownership.py
D	test/registered/unit/spec/test_decoupled_spec_io.py
M	test/registered/unit/spec/test_dflash_domino.py
A	test/registered/unit/spec/test_dflash_dp_attention.py
A	test/registered/unit/spec/test_dflash_draft_graph_eligibility.py
M	test/registered/unit/spec/test_dflash_extra_buffer_lazy.py
M	test/registered/unit/spec/test_dflash_logits.py
A	test/registered/unit/spec/test_dflash_mla.py
A	test/registered/unit/spec/test_dflash_model_config.py
M	test/registered/unit/spec/test_dflash_overlap_hostsync.py
M	test/registered/unit/spec/test_dp_spec_prefill_coordination.py
M	test/registered/unit/spec/test_draft_construction_isolation.py
M	test/registered/unit/spec/test_draft_tp_context.py
M	test/registered/unit/spec/test_dspark_accept_tail.py
A	test/registered/unit/spec/test_dspark_blockwise_rejection.py
A	test/registered/unit/spec/test_dspark_deferred_commit.py
M	test/registered/unit/spec/test_dspark_dp_spec_prefill_coordination.py
M	test/registered/unit/spec/test_dspark_draft_block.py
M	test/registered/unit/spec/test_dspark_greedy_draft_accept.py
A	test/registered/unit/spec/test_dspark_sim_draw_ahead.py
A	test/registered/unit/spec/test_dspark_step_inputs_once.py
M	test/registered/unit/spec/test_dspark_target_hidden_projection.py
M	test/registered/unit/spec/test_eagle_draft_cuda_graph_runner.py
M	test/registered/unit/spec/test_eagle_draft_extend_logits.py
A	test/registered/unit/spec/test_eagle_draft_sampling.py
M	test/registered/unit/spec/test_eagle_gate_routing.py
A	test/registered/unit/spec/test_eagle_post_verify_graphs.py
M	test/registered/unit/spec/test_eagle_seeded_coins.py
M	test/registered/unit/spec/test_eagle_worker_v2_topk1_fastpath.py
M	test/registered/unit/spec/test_fast_prefill_plan.py
A	test/registered/unit/spec/test_minimax_dflash.py
A	test/registered/unit/spec/test_move_accept_tokens_targets.py
M	test/registered/unit/spec/test_multimodal_draft_embeddings.py
M	test/registered/unit/spec/test_ngram_corpus.py
M	test/registered/unit/spec/test_ngram_mamba_verify_update.py
M	test/registered/unit/spec/test_ngram_worker_prompt.py
M	test/registered/unit/spec/test_oot_dflash_hooks.py
A	test/registered/unit/spec/test_oot_eagle_hooks.py
M	test/registered/unit/spec/test_plugin_hook_signatures.py
M	test/registered/unit/spec/test_pp_draft_embedding.py
M	test/registered/unit/spec/test_resolve_swa_kv_pool.py
M	test/registered/unit/spec/test_spec_cpu_overlap_constraint.py
M	test/registered/unit/spec/test_spec_tp_sync.py
M	test/registered/unit/spec/test_spec_utils_reexport_surface.py
D	test/registered/unit/spec/test_suffix_attention_merge_dispatch.py
M	test/registered/unit/spec/test_uno_request_validation.py
M	test/registered/unit/spec/test_uno_tree_config.py
M	test/registered/unit/spec/test_uno_tree_sparse_sampling.py
M	test/registered/unit/spec/test_weight_update_runners.py
A	test/registered/unit/speculative/test_dspark_engram_commit_in_graph.py
A	test/registered/unit/speculative/test_dspark_sampling_accept_graph.py
M	test/registered/unit/speculative/test_pd_eagle_draft_proposal.py
M	test/registered/unit/state_capturer/test_npu_indexer_topk_capturer.py
M	test/registered/unit/state_capturer/test_routed_experts_scattered_a2a.py
M	test/registered/unit/test_check_env.py
M	test/registered/unit/test_checkpoint_quantization.py
A	test/registered/unit/test_dsa_dcp_backend_defaults.py
M	test/registered/unit/test_dsa_tilelang_fp8_validation.py
M	test/registered/unit/test_flashinfer_sparse_mla.py
M	test/registered/unit/test_global_config_read_ratchet.py
M	test/registered/unit/test_import_without_flashinfer.py
M	test/registered/unit/test_model_overrides.py
M	test/registered/unit/test_mxfp8_moe_runner_backend.py
M	test/registered/unit/test_parallel_consumers.py
M	test/registered/unit/test_platform_context.py
M	test/registered/unit/test_quantization_post_load.py
M	test/registered/unit/test_rocm_pageable_h2d_staging.py
M	test/registered/unit/test_runai_utils.py
M	test/registered/unit/test_runtime_context.py
M	test/registered/unit/test_runtime_context_override.py
M	test/registered/unit/test_server_args_cli_metadata.py
M	test/registered/unit/test_server_args_migration.py
M	test/registered/unit/test_server_args_namespaces.py
M	test/registered/unit/test_token_scoring.py
M	test/registered/unit/tokenizer/test_mistral_empty_assistant.py
M	test/registered/unit/utils/test_common.py
M	test/registered/unit/utils/test_cpp_extension_loader.py
M	test/registered/unit/utils/test_custom_op_eager_direct.py
M	test/registered/unit/utils/test_diffusion_torch_fallback.py
M	test/registered/unit/utils/test_gauge_histogram.py
M	test/registered/unit/utils/test_hf_transformers.py
M	test/registered/unit/utils/test_hf_transformers_loading.py
D	test/registered/unit/utils/test_http_server_auth.py
M	test/registered/unit/utils/test_invariants.py
M	test/registered/unit/utils/test_json_response.py
M	test/registered/unit/utils/test_make_pp_layers.py
A	test/registered/unit/utils/test_network.py
M	test/registered/unit/utils/test_patch_tokenizer.py
M	test/registered/unit/utils/test_profile_merger.py
A	test/registered/unit/utils/test_request_logger_redaction.py
M	test/registered/unit/utils/test_safe_unpickler.py
M	test/registered/unit/utils/test_subprocess_watchdog.py
M	test/registered/unit/utils/test_weight_checker.py
M	test/registered/unit/utils/test_weight_checker_comparator.py
A	test/registered/unit/utils/test_weight_checker_cpu.py
M	test/registered/unit/utils/test_weight_versions.py
M	test/registered/utils/test_network_address.py
M	test/registered/utils/test_numa_utils.py
M	test/registered/utils/test_phase_checker.py
M	test/registered/utils/test_socket_utils.py
M	test/registered/utils/test_stale_shm_cleanup.py
M	test/registered/vlm/test_paddleocr_vl_server.py
M	test/registered/vlm/test_rust_native_mm_e2e.py
M	test/registered/vlm/test_rust_native_mm_mmmu.py
M	test/registered/vlm/test_token_id_retokenize_e2e.py
M	test/registered/vlm/test_unlimited_ocr_server.py
M	test/registered/vlm/test_video_utils.py
M	test/registered/vlm/test_vision_chunked_prefill.py
M	test/registered/vlm/test_vision_openai_server_a.py
M	test/registered/vlm/test_vision_openai_server_extra.py
M	test/registered/vlm/test_vlm_input_format.py
M	test/registered/vlm/test_vlm_tp4.py
M	test/registered/xpu/test_breakable_cuda_graph.py
A	test/registered/xpu/test_disaggregation_staging_xpu.py
M	test/registered/xpu/test_dsa_indexer_xpu.py
A	test/registered/xpu/test_hicache_transfer_round_trip_xpu.py
M	test/registered/xpu/test_numa_utils_xpu.py
M	test/registered/xpu/test_qsa_indexer_xpu.py
M	test/registered/xpu/test_topk.py
M	test/registered/xpu/test_xpu_flux2_dev.py
A	test/registered/xpu/test_xpu_gemma3_bidirectional_attn.py
M	test/run_suite.py
M	tools/sglang-simulator/README.md
M	tools/sglang-simulator/pyproject.toml
M	tools/sglang-simulator/setup.py
M	tools/sglang-simulator/src/sglang_simulator/simulation/sglang/hicache_storage.py
M	tools/sglang-simulator/src/sglang_simulator/time_predictor/ml.py
A	tools/sglang-simulator/src/sglang_simulator/time_predictor/ml_features.py
A	tools/sglang-simulator/src/sglang_simulator/tools/__init__.py
A	tools/sglang-simulator/src/sglang_simulator/tools/collect_baseline/README.md
A	tools/sglang-simulator/src/sglang_simulator/tools/collect_baseline/__init__.py
A	tools/sglang-simulator/src/sglang_simulator/tools/collect_baseline/collect.py
A	tools/sglang-simulator/src/sglang_simulator/tools/collect_baseline/hooks.py
A	tools/sglang-simulator/src/sglang_simulator/tools/train_ml_time_predictor/README.md
A	tools/sglang-simulator/src/sglang_simulator/tools/train_ml_time_predictor/__init__.py
A	tools/sglang-simulator/src/sglang_simulator/tools/train_ml_time_predictor/example_steps.jsonl
A	tools/sglang-simulator/src/sglang_simulator/tools/train_ml_time_predictor/train.py
A	tools/sglang-simulator/test/test_baseline_collection_sglang.py
A	tools/sglang-simulator/test/test_train_latency_model.py
```

</details>

## Act 3 — Build: P0 independent review

Primary-agent review verified the three conflict resolutions and both fixture changes. The production control, transfer, priority and weight-lifecycle implementations retain their pre-merge semantics. Independent six-file pytest: 62 passed, 25 subtests passed (28.68 s); independent locked Router/indexer cargo check passed on task-local Rust 1.99.1. Proof logs: /tmp/sglang-rfc-build/p0-parent-tests.log and p0-parent-router-check.log. P0 is approved for the merge commit under the user instruction to commit each tested feature. No Codex co-author is added.

## Act 3 — Build: P1 protocol and model foundation (2026-11-04)

Baseline: parent commit `46ba2e84b9`. Scope is P1 only; no commits, pushes,
subdelegation, model pinning, Router runtime integration, or P2–P8 physical
features. P9 remains a later expansion. D1–D7 remain adopted. Parent review and
commit are pending.

Python adds version-1.0 PREFETCH/DEMOTE/PIN schemas on envelope 0.1. PREFETCH and
DEMOTE normalize to existing CopyTo/MoveTo only when the live executor advertises
the direction/backend. PIN is validated but always unsupported; schema discovery
is separate from executable capabilities. PrefixTarget validates bounded nested
compatibility, aligned definite boundaries, token material, ordered page
observations, dependencies, and canonical decimal-string i64 event hashes.
Execution requires complete token material from zero and matching identity/layout;
no truncated hash is used to reconstruct a storage key. Raw salt reaches the
existing executor namespace transform unchanged. Unknown consumer actions and
legacy action/target semantics remain intact.

New public_result is absent on old action results. Its ready/preserved/reclaimed
ranges derive from per-target publication/reclamation evidence, alongside
requested/accepted/missing/blocked scopes, owner/cache/result versions and reason
codes. Queued I/O and generic RUNNING do not establish readiness or scoped
acceptance. Relative timeout requests cancellation; resource drain still governs
quiescence. Claim state, protection version, revision and tombstone fields are
separate schemas, with no claim executor or fake protection.

The independent `sgl-kv-orchestrator` crate contains protocol parsing, shared
checked-in JSON schemas, salt encoding, i64 string serialization, and pure
session/branch/observation/placement/claim/budget models. It has no dependency on
Router or indexer and no runtime integration. Cargo.lock adds only the new local
package; all dependency versions were already present. Schema fixtures are drift
checked against Python; action, result/claim and salt vectors run in both
languages. Exact wire contract: `sgl-router/sgl-kv-orchestrator/README.md`.

Validation used existing `.venv`, `PYTHONPATH=python`, and the user-provided
cargo/rustup/cache environment. Tests remain in the existing registered CPU test
files. New coverage includes invalid i64/bounds/dependencies/layouts, Unicode and
null/empty salts, old selectors, no false capabilities, replay conflicts,
attached and in-flight timeout cancellation, old result serialization, separate
preservation/reclamation, and real CPU pool/file-store publication under
controlled DMA. No GPU or live Mooncake claim is made. One inherited unknown
native-action test used `sglang.pin` as a placeholder; it now uses
`sglang.future_action`, with PIN covered explicitly by the new tests.

Exact requested proof commands:

```sh
PYTHONPATH=python .venv/bin/python -m pytest -q --tb=short test/registered/unit/managers/test_kv_control.py test/registered/unit/managers/test_kv_policy.py test/registered/unit/mem_cache/test_kv_control_cache.py test/registered/unit/mem_cache/test_kv_control_service.py test/registered/unit/mem_cache/test_kv_control_transfers.py test/registered/unit/mem_cache/test_kv_policy_runtime.py
cargo test --manifest-path sgl-router/Cargo.toml -p sgl-kv-orchestrator
cargo check --locked --manifest-path sgl-router/Cargo.toml -p sgl-router -p sgl-kv-indexer
```

Proof logs: `/tmp/sglang-rfc-build/p1-python-tests.log`,
`/tmp/sglang-rfc-build/p1-rust-test.log`,
`/tmp/sglang-rfc-build/p1-router-check.log`. Relevant output:

```text
72 passed, 17 warnings, 66 subtests passed in 14.99s
Rust: test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
Rust doc tests: 0 passed; 0 failed
Router/indexer: Finished `dev` profile [unoptimized + debuginfo]
```

Python warnings are dependency deprecations from the existing FlashInfer/CUTLASS
installation. `cargo fmt` and `git diff --check` are clean. P1 uses the existing
file and host/GPU executor contracts; Mooncake adapters (P2), expanded DEMOTE (P3),
PIN execution (P4), Router graph/routing (P5/P6), recovery feedback (P7), and joint
GPU/Mooncake acceptance (P8) remain pending. `source_observed_at` stays null until
a physical observation source supplies it. Public outcomes are historical
operation evidence, not permanent residency guarantees. `allow_partial=false`
prevents requested clipping, without promising transactional physical I/O.

P1 changed-file inventory (excluding temporary proof logs):

```text
PLAN-REVIEW-LOG.md
PLAN.md
python/sglang/srt/managers/kv_control/contracts.py
python/sglang/srt/managers/kv_control/controller.py
python/sglang/srt/managers/kv_control/public.py
python/sglang/srt/managers/kv_control/registry.py
python/sglang/srt/managers/kv_control/service.py
sgl-router/Cargo.lock
sgl-router/Cargo.toml
sgl-router/sgl-kv-orchestrator/Cargo.toml
sgl-router/sgl-kv-orchestrator/README.md
sgl-router/sgl-kv-orchestrator/fixtures/actions.json
sgl-router/sgl-kv-orchestrator/fixtures/claim.schema.json
sgl-router/sgl-kv-orchestrator/fixtures/demote.schema.json
sgl-router/sgl-kv-orchestrator/fixtures/pin.schema.json
sgl-router/sgl-kv-orchestrator/fixtures/prefetch.schema.json
sgl-router/sgl-kv-orchestrator/fixtures/result.schema.json
sgl-router/sgl-kv-orchestrator/fixtures/results.json
sgl-router/sgl-kv-orchestrator/fixtures/salt.json
sgl-router/sgl-kv-orchestrator/src/lib.rs
sgl-router/sgl-kv-orchestrator/src/models.rs
sgl-router/sgl-kv-orchestrator/src/protocol.rs
test/registered/unit/managers/test_kv_control.py
test/registered/unit/mem_cache/test_kv_control_transfers.py
```

### P1 independent review and proof

Reviewed public normalization, capability gating, timeout cancellation/drain, result projection, shared schemas and Rust validation. Independent Python proof: 72 passed, 66 subtests passed (13.55 s); independent locked Rust protocol proof: 4 passed. Source logs: /tmp/sglang-rfc-build/p1-parent-tests.log and p1-parent-rust.log. PIN remains unsupported until its executor is integrated. Public state_version currently orders operation evidence, not KV location-stream observations; P7 must attach an actual cache-state watermark before cross-stream reconciliation. P1 approved for commit.

### P4a L2 protection foundation — independent review

Added owner-thread FULL host-prefix protection with split-aware native receipts, shared physical-slot budgets, per-caller/task/incarnation claim revisions, finite TTL, tombstones and reset invalidation. Public PIN execution is still pending. Independently reviewed both new files; protection/cache/transfer regression: 33 passed, 18 subtests passed (12.14 s). After integration with P1, focused protection suite: 12 passed. Logs: /tmp/sglang-rfc-build/p4a-parent-regression.log and p4a-integrated-tests.log. Approved for feature commit; runtime wiring follows separately.

## P2 backend/PREFETCH implementation evidence — 2026-11-04

Base: P1 `e96321c648` plus P4a `b3bccb5015`. No commits/pushes or subdelegation.
`protection.py` and public PIN execution are unchanged. Parent review/commit is
pending; P3, P4 wiring and P5+ remain separate work.

Added `unified_cache/storage_execution.py`: explicit file/Mooncake adapter registry,
backend instance/layout validation, declared copy/move paths, stable backend
namespace and writer-derived object descriptions. The existing HiCache worker
queues, whole-scope approval, owner-thread host publication, exact-prefix shared
interests and cancel/drain/reference mechanisms continue to perform physical I/O.
Mooncake support is restricted to single TP/PP/CP/DCP, FULL MHA,
`page_first_direct`, without split heads or unified storage suffixes. The upstream
external layout reporter's MHA/multi-rank restriction remains unchanged.

Capabilities under `public_protocol` now include:

```json
{
  "storage_backend": "mooncake",
  "storage_profile": "tp1_full_mha_page_first_direct_v1",
  "backend_namespace": "sglang.storage.v1:<64 hex SHA256>",
  "backend_namespace_selection": true,
  "storage_targets": true,
  "paths": {
    "sglang.prefetch": [["host","gpu"],["storage","host"],["storage","gpu"]],
    "sglang.demote": [["gpu","host"],["gpu","storage"]]
  },
  "pin_protections": [],
  "hash_only_execution": false
}
```

File advertises backend `file`, profile `native_full_file_v1`. Paths remain the
intersection with the live cache executor. Unsupported/detached backends expose
no storage paths. A named namespace must equal the bound adapter's fingerprint;
mismatch returns `incompatible_namespace`. Omission preserves legacy local-backend
selection. The fingerprint excludes engine identity/epoch, host pointers and pool
capacity; it includes actual endpoint/tenant/key builders and byte layout. File
also includes canonical directory and hostname. Endpoint aliases are not inferred.
Storage result locations contain backend + backend_namespace and no engine_id.
The Location schema gains an optional backend_namespace, with absent defaults
omitted; shared result schema/fixtures were regenerated and Rust-tested.

P4 prerequisite query uses the existing trusted control endpoint and context:
`{kind:"storage_targets", context, target, location:{tier:"storage",backend,
backend_namespace}}`. Target must resolve to one definite, complete, aligned token
prefix. Response fields: description_version="1.0", namespace, compatibility
(raw salt), key_salt_encoding, page_size, location, existence="unknown", pages.
Each page has start_token/end_token, component="full", full storage_digest,
keys, and (Mooncake) parallel object_bytes. It creates no operation/claim or
network existence check. Actual keys use `get_storage_hash_str` and the real
writer's `_tag_keys`/`_batch_preprocess` (K/V/rank), or file `_get_component_keys`.
No event i64 is inverted. The service namespace/salt transform is applied once.
Exact payload example and reproducible live command are in
`sgl-router/sgl-kv-orchestrator/README.md`.

The live test used Mooncake SDK **0.3.14** from
`/home/youwei/.cache/sglang-rfc-tools/mooncake`, master `127.0.0.1:52091`, metadata
`http://127.0.0.1:52093/metadata`, metrics/admin port `52092`, tenant `default`, TCP,
standalone=false/check_server=false. GPU 2 was NVIDIA A100 80GB PCIe, torch
2.14.1+cu130. Two independent cache owners with distinct pools/epochs shared the
backend. A unique weight namespace isolated each run. No reference-server cache
was read or changed; GPU 0/1 servers were not stopped.

Live evidence: ordinary insertion with service-encoded Unicode salt produced the
same full digests as both owners' public descriptions; all eight physical objects
(four pages × K/V, two layers/two heads) were absent before writing and existed
after acknowledged writing. The reader had no radix prefix or session association.
Independent public L3→L2 restored exact host bytes, joined/cancelled an independent
interest without losing the remaining reader, and loaded exact bytes back to GPU.
Readiness was absent at submission and present with actual host registration.
This is real GPU/host/Mooncake I/O; it is not model-generation or Router acceptance.

Found and fixed a production completion issue: Mooncake accepted any positive
object read count, including short reads. Completion now requires all expected
bytes and rejects missing receipts; partial write receipts retain failures. CPU
regressions exercise real key/buffer metadata with controlled SDK responses,
plus existing file partial/cancel/publication tests. The registered GPU test now
uses `tree_slots`, allocator `.state`, and explicit `attn_dcp_rank` fixture APIs;
its original host/file/preparation assertions remain.

Proof commands retain the requested six CPU files and GPU test. All Python runs
prepend `/home/youwei/.cache/sglang-rfc-tools/mooncake:python` to PYTHONPATH. GPU runs
use `CUDA_VISIBLE_DEVICES=2`; the focused live run sets
`KV_CONTROL_MOONCAKE_CONFIG` to the JSON documented in the README.

```text
Six CPU files: 74 passed, 17 warnings, 66 subtests passed in 14.78s
Registered GPU test: 1 passed, 1 skipped, 17 warnings, 2 subtests passed in 14.09s
Focused live Mooncake: 1 passed, 1 deselected, 17 warnings in 13.44s
Mooncake group/tenant regression: 21 passed, 17 warnings in 11.47s
Locked Rust shared fixtures: 4 passed; 0 failed
```

The normal GPU command skips only the opt-in live test; the focused live command
passes it against the supplied master. Logs are `/tmp/sglang-rfc-build/p2-`:
`cpu-tests.log`, `gpu-tests.log`, `mooncake-live.log`, `mooncake-regression.log`,
`rust-fixtures.log`, and `environment.log`. Warnings are existing dependency
deprecations. Changed Python files pass `ruff --select F`; `git diff --check` is
clean. No production reference constraints were bypassed. Remaining validation:
model-generation/Router joint acceptance (P8), other host layouts/MLA/multiple
ranks (P9), expanded DEMOTE (P3), PIN (P4), Router/recovery (P5–P7).

### P2 independent review and proof

Reviewed registered adapters, namespace/key descriptions, native read/write completion and short-read checks. Independent seven-file CPU regression (including protection foundation): 86 passed, 77 subtests passed (15.19 s). Independent real GPU host/file and live Mooncake tests on GPU1: 2 passed, 2 subtests passed (13.78 s). Both restored K/V bytes and post-publication GPU load-back verified. Shared exact-scope cancellation preserves the other interest. Locked Rust protocol fixtures passed. Logs: /tmp/sglang-rfc-build/p2-parent-{cpu,gpu,rust}.log. Initial supported Mooncake path is TP1 FULL MHA page_first_direct; other physical layouts are not advertised. P2 approved for commit; model-level two-worker acceptance remains P8.

## P3 DEMOTE outcome/reclamation evidence — 2026-11-04

Base: clean `69de4c3e82` (P2). No commits, pushes or delegation. P4 registry/runtime
and Router/L3 owner wiring are untouched. Source changes are contracts/controller,
`control_transfers.py` and `control_storage.py`; regressions extend the existing
registered manager/transfer suites and the existing opt-in live GPU test.

Public DEMOTE carries an internal owner-selected flag after normalization to
MoveTo; no public field/version changes. This separates its bounded source-load
handling and reason codes from legacy copy/move jobs. It reserves per-page outcome
capacity before starting so concurrent splits and distinct blocker reasons cannot
exceed the result budget. Both cold paths retain native completion, shared exact
scope jobs, capacity checks and reference drain. Host readiness is fence/publish
based; storage uses the existing verified contiguous backup ACK. Existing host
copies and native file/Mooncake skip-write paths avoid redundant data writes.

Reclamation is shared by host/storage paths: release only this job's GPU holds,
check actual pending transfers, remaining references and device children, call
native tree demote, capture the real returned free-token count, and drain frees
through the cache allocator before recording source_reclaimed. Generic
`source_busy` remains on legacy results; public blocked scope uses active_reference,
inflight_backup, inflight_load, structural_dependency, or conservative
not_reclaimable. Failure/cancellation keeps the actual failure/cancel code on
preserved-but-unreclaimed ranges. There is no L1 PIN or inferred pin_conflict.
Incomplete host/storage backup keeps all source data. A completed cold backup can
reclaim an eligible suffix while retaining an ancestor. Active references terminate
as partial without waiting for request completion; owned I/O still requires drain.

Cancellation of one shared host DEMOTE caller now snapshots already-published
cold coverage before dropping its interest, leaving the other caller's physical
job intact. Last-caller host/storage cancellation drains and reports destination
publication without initiating reclamation. New tests assert allocator capacity,
source/host contents, per-range public projection, released references, immutable
historical results, and retries on retained prefixes. Partial storage ACKs and host
submission failures preserve the ready prefix and all GPU source. Concurrent load
and storage backup tests verify bounded rejection/partial results without cancelling
the foreign demand. File reuse rejects any redundant os.replace; Mooncake reuse
rejects any redundant SDK put when physical keys already exist.

Legacy SetPriority/CopyTo/MoveTo parsing now rejects non-null backend_namespace on
GPU/host with invalid_payload, before dispatch. Old payloads remain accepted.
Host→storage public DEMOTE remains unsupported and is explicitly regression-tested.

Proof (PYTHONPATH=/home/youwei/.cache/sglang-rfc-tools/mooncake:python):

```text
Seven CPU suites: 96 passed, 17 warnings, 81 subtests passed in 29.60s
cargo test --locked --manifest-path sgl-router/Cargo.toml -p sgl-kv-orchestrator:
  4 passed; 0 failed; doc tests 0 passed; 0 failed
ruff check --select F on changed Python files: All checks passed!
git diff --check: clean
```

CPU command is the P2 six-file pytest plus
`test/registered/unit/mem_cache/test_kv_control_protection.py` with `-q --tb=short`.
Logs: `/tmp/sglang-rfc-build/p3-cpu-tests.log`, `p3-rust-fixtures.log`.
Dependency deprecation warnings are unchanged.

Fresh real GPU/file/Mooncake proof is **pending a parent-approved GPU 2 window**.
An async request was made; no window was confirmed during this implementation.
No GPU test, model server, benchmark or process termination was run. The existing
live test now additionally checks the writer's reclaimed scope, removed GPU values
and four returned allocator slots alongside real physical-key existence and K/V
bytes. These added GPU assertions were syntax/lint checked, not executed here.
P2's prior parent proof (2 GPU tests + 2 subtests, SDK 0.3.14) remains historical
baseline evidence, not a fresh P3 result. Reproduction command with 64 MiB segment
is saved in `/tmp/sglang-rfc-build/p3-report.txt` for the scheduled window.

Remaining: P4 execution wiring, Router/L3 ownership and P5–P8 integration; P9 expanded
profiles. No host-source reclamation or L1 PIN was added. Storage confirmation is
point-in-time evidence with retained host data during reclamation, not a persistent
backend lease. Full-scope retries must respect current source residency; the tested
bounded retry targets the retained complete prefix after suffix reclamation.

### P3 independent review and proof

Reviewed real cold preservation/source allocator reclamation, public-only blocker classification, shared-job key separation, partial backup and cancellation effects, and namespace field validation. Independent CPU7: 96 passed,81 subtests passed(22.05s); independent GPU host/file+live Mooncake writer reclamation/loadback:2 passed,2 subtests passed(18.48s). Logs p3-parent-cpu.log,p3-parent-gpu.log in /tmp/sglang-rfc-build. No new path support advertised beyond tested gpu->host/storage. P3 approved for commit.

### P5 bounded session/branch state — independent review

Integrated frozen branch/session/request targets, shared suffix material, explicit cleanup receipts and typed observation/query fences. Independent isolated-worktree and integrated-root crate proofs pass4 protocol+18 behavioral tests. Reviewed ownership/GC, version-domain separation, no pending-to-confirmed promotion, and session associations surviving physical removal. Existing provider remains authoritative; P6 host adapter still pending. Logs p5-parent-tests.log,p5-integrated-tests.log. Approved for commit.

### P4c Mooncake protection owner — independent review

Reviewed async single-writer actor, shared object aggregation, explicit backend_soft_pin contract, revision/tombstone fencing, conservative TTL and unknown-result recovery barriers. Independent worktree proof passed4 protocol+21 backend tests with real admin pin/unpin; integrated crate proof passed4 protocol+21 backend+18 state tests. One earlier rerun found the long-lived private seed key evicted under concurrent performance load(-704), so final proof seeds a fresh private key immediately before the real test and keeps its client alive through completion. This is an environment precondition failure, retained in parent logs, not interpreted as protection success. Logs /tmp/sglang-rfc-build/p4c-parent-tests.log and p4c-integrated-tests.log. P6 host wiring remains pending. Approved for commit.

### P4b L2 PIN runtime — independent review and integration

Reviewed native registry dispatch, independent claim/query/expiry lifetime, attached request binding/fanout, successful checkpoint activation, abort handling, caller cleanup, pre-reset invalidation and deadline-aware idle sleep. Independent isolated final7suite97passed80subtests. Integrated with P2/P3/P5/P4c:109passed84subtests(28.43s), Rust4protocol+20backend+18state passed(optional live backend test excluded here; independently tested in P4c). Regenerated merged shared schemas after drift test caught P2 backend_namespace missing from P4b claim schema. Evidence p4b-parent-final-tests.log,p4b-integrated-tests.log,p4b-integrated-rust.log. GPU correctness already verified via p4b-gpu-pin.py. Approved for commit.

## Late native PREFETCH / ordinary demand coalescing — 2026-11-04

Base `f6ec0941235f9711735bbc5c93082449fc368c73`, clean at entry. No commits/pushes,
subdelegation, global config or P7 event-publication changes. P1–P5 behavior and
first-profile physical limits are retained. This is a targeted P8 optimization;
model-level performance acceptance remains pending parent review/restart/replay.

Evidence read before production edits: `perf-optimization-notes.txt`,
`perf-findings-p2.json`, `bench-p2-matrix-v1/{manifest,summary}.json`, ready-window
summary, and `engine-b.log` in `/tmp/sglang-rfc-build`. Original benchmark artifacts
were read only. Both results remain evidence:
- P2 matrix: 36/36 valid; 1s wait / 0.5s lead. Mean TTFT 2K 336→138ms,
  8K 804→755ms, 16K 1469→2503ms, with 16K ready-before 0/6.
- Ready-window: 16/16 valid; 3s wait / 2.5s lead. Mean TTFT 8K 630→96ms,
  16K 996→161ms, all prefetch cases ready before resume.
No failed/slow model samples were removed or reclassified.

Mechanism: logs show native loaded=16384 followed by ordinary matched=16384,
loaded=0. Source inspection found separate per-handle storage submissions without
native/demand matching. A pre-edit deterministic real-file/pool test counted TWO
storage hit queries and TWO backend page GET calls; the second request reported
(0,None) loaded span. The same interleaving after the change counts ONE query/GET
and reports (4,0). Logs: `prefetch-before-mechanism.log` and
`prefetch-after-mechanism.log`. Event gates control correctness interleavings,
not elapsed-time performance; no artificial delay was used for a performance claim.
The original model run has no packet-level GET trace, so its log alone is not
claimed as proof that every additional millisecond was duplicate network I/O.

Implementation: `StorageCacheReads` indexes only active native storage→host jobs
by exact complete tokens + actual encoded salt/extra-key. Normal persistent-host,
non-bigram `wait_complete` requests join through the existing cache prefetch entry.
Each request has an incarnation-keyed bounded demand interest. Router cancel keeps
I/O alive while demand exists; request abort removes only that request. Publication
and native drain remain prerequisites for admission progress. Shared holds survive
native-result retirement until each request takes its loaded span or aborts, then
normal match/load-back/request locking takes over. Partial/failing reads unblock
with their actual prefix, without marking an unfilled tail ready. Published-before-
arrival data stays host-attributed; joined newly loaded data is storage-attributed.
No new PIN, bypassed capacity check, GPU reservation or global inference barrier.

Bounds use existing max_prepared_prefixes for demand handles and max_scope_tokens
for material. Overflow falls back to the ordinary path. Key lookup is indexed;
readiness polls are O(1) and do not walk/hash the prefix. Source inspection showed
active native I/O polls already check quiescence rather than repeatedly traversing
16K tokens; no unproven quadratic-polling refactor was added. Fixed counters and a
waiter gauge are exposed through existing prefetch stats; no invented timing/bytes
or success/unused estimates. `native_demand_consumed` means admission pop, not an
inference-hit guarantee. Disabled control and nonmatching scopes/policies retain
the existing path. Reset/storage detach cannot discard retained demand receipts.

Verification:
- Seven control suites: 113 passed, 86 subtests; final log `prefetch-cpu-tests.log`.
- Storage lifecycle + scheduler HiCache events/held-row admission/chunked gate/
  runtime attach: 24 passed, 17 subtests (`prefetch-admission-tests.log`).
- GPU1 real host/file + Mooncake coalesced demand: 2 passed, 2 subtests
  (`prefetch-gpu-tests.log`), SDK0.3.14, supplied TCP master, private test namespace,
  64MiB segment. `prefetch-gpu-idle-check.json` records both model servers with
  zero running/waiting/queued requests immediately before the tiny test. Existing
  servers/source snapshots were not changed; GPUs2/3 processes were untouched.
- Rust locked tests with CARGO_BUILD_JOBS=8: 4 protocol + 20 backend + 18 state
  passed; optional live admin soft-pin test ignored (`prefetch-rust-tests.log`).
  Live storage data I/O was separately exercised by the GPU test.
- Changed Python `ruff --select F` and `git diff --check` passed.
Dependency deprecation warnings remain. A test-fixture-only rerun initially used
msgspec.replace on a dataclass config; corrected to dataclasses.replace and rerun.
The failed log is retained as `prefetch-cpu-budget-fixture-failure.log`.

New deterministic cases cover active GET joins, duplicate submissions, independent
abort, Router cancel while demand waits, both consumers cancelled/drain, token/
length/salt/encoded-namespace mismatches, per-cache budget exhaustion, partial read,
post-publication host attribution, disabled-control fallback and no leaked holds.
Existing idle native completion, allocator/headroom and PIN lifetime suites pass.
The live test joins an actual request handle before owner publication, verifies its
storage span and confirms host and GPU K/V bytes through the unchanged load-back.

Limitations: exact page-aligned scope only, wait_complete only; no superset-prefix
matching or latency policy/EMA. Native cancellation can remain draining until the
shared demand read finishes. Normal admission/eviction rules apply after span pop.
No new Qwen TTFT result is claimed. Parent must restart the working snapshot and
repeat the immutable 0.5s and 2.5s lead matrices, keeping every failed/slow sample,
before deciding whether this fixes the observed model-level regression.

### P8 native/demand coalescing — independent review

Reviewed indexed exact-prefix demand joins, owner publication/quiescence, cancellation/drain, request abort, loaded-span attribution and teardown guards. Independent final CPU proof113passed86subtests(21.46s); independent GPU host/file/live Mooncake join2passed2subtests(18.92s). Prior parent run encountered newly edited fixture using msgspec.replace on dataclass Limits; final corrected fixture proof is recorded, no failed production result hidden. Deterministic before/after I/O count2queries+2GETs ->1query+1GET; model TTFT benefit still awaits snapshot restart. Other scopes/stop policies retain ordinary path. Approved for commit.

### P8 reproducible benchmark harness

Added paired Engine HTTP benchmark with actual cache-tier checks, raw failure retention, deterministic outputs, trace provenance, historical cost EMA, bootstrap intervals and optional background/host-pressure load. PIN readiness uses current claim contract, not transfer ready_scope. Independently ran8tests including local two-engine mock HTTP and recorded real SSE replay; all passed. New load/trace options are mock-validated only until separate real runs. Script files and input fixture committed; real P2 matrices remain immutable under /tmp/sglang-rfc-build and will be included in final result artifacts.

## P6 Router orchestration acceptance — 2026-11-04

Optional authenticated Router control/lifecycle actor, bounded exact-job/shared-interest bookkeeping, immutable global selectors, both HTTP routing paths, live scoped-residency advisory, finite L2 claim cleanup and actual Mooncake L3 owner/resolver are integrated. Normal selection retains load/admission/affinity ownership. Source/operation completion alone cannot mark current residency. Configuration/runbook: sgl-router/KV_ORCHESTRATION.md.

Parent independent checks: 12 real HTTP proxy tests passed; 44 orchestrator tests passed with 1 optional live backend test skipped. Builder complete serial proxy proof: 236 passed/1 existing ignored; Router unit proof: 1113 passed. Parent independent unit rerun logged p6-parent-unit-tests.log. Parent reviewed contracts, actor admission/refresh/cancel/claim paths and routing hooks. Real model Router B3 and final event-watermark integration remain P7/P8 work. Evidence: /tmp/sglang-rfc-build/p6-report.md and p6-parent-*.log.

## P7a feedback acceptance — 2026-11-04

Engine bounded owner-thread feedback polling plus optional separate nonblocking ZMQ/replay publisher, actual native mutation watermarks, request-use facts and explicit association, claim/invalidation events, legacy location/session watermark preservation are integrated. Parent review covered journal byte/record bounds, source/cursor/gap handling, request accounting and lifecycle hooks.

Parent independent isolated tests145passed90subtests; integrated with native/demand sharing149passed92subtests(33.96s), Router wire24passed. A prior integrated attempt hit the existing random-port probe bind race; isolated KV events7passed4subtests and final full rerun passed, no production change hidden. Evidence p7a-parent-tests.log, p7a-integrated-final-tests.log, p7a-port-recheck.log, p7a-parent-wire-tests.log and p7-report.md under /tmp/sglang-rfc-build. P7b indexer source fences and P7c Router consumption are separate commits.

## P7b Indexer recovery acceptance — 2026-11-04

Registered publisher source epochs, compare-and-swap lifecycle authority, bounded retired epochs/chunk fingerprints, atomic placement+query evidence, native cache generation/version, session label preservation, bridge split-batch coordinates and Indexer restart UUID fences are integrated. Legacy sources remain advisory/unknown; trusted empty-cut reconciliation and external invalidation are explicit, no durable replay/HA advertised.

Parent reviewed source gate/reconcile/clear paths, placement lock atomicity, split bridge shape, client confirmation filters and protobuf reserved fields. Independent indexer156passed; combined main indexer156+orchestrator44passed with1optional liveignored. P6/P7a retained. Evidence: /tmp/sglang-rfc-build/p7b-report.md, p7b-parent-tests.log, p7b-integrated-tests.log.

## P8 Router benchmark harness acceptance — 2026-11-04

Explicit router_bench.py B2/B3 pair uses the same Router lifecycle preparation targeting B, fixed tool deadlines, both workers cold after verified shared publication, actual x-sgl-worker-url attribution, tier/output consistency and stable task cancellation/drain/ack. Direct B2 and freely routed B3 are distinguished; route-change negatives remain in raw. Auth values are absent from raw.

Parent independent integrated16unittests passed(11.953s), reviewed source setup, timer/observer separation, worker/epoch checks, cleanup and accounting. P6 real-model experiment running separately; no GPU benefit inferred from HTTP fixtures. Evidence /tmp/sglang-rfc-build/p8-router-harness-report.md and p8-router-integrated-tests.log.

## P8 measured Engine benchmark archive — 2026-11-04

Six completed runs/202 scenarios archived under benchmark/session_kv_control/results/2026-11-04 with immutable raw JSONL gzip, launch/runtime provenance, original negative samples, schedules/manifests, paired CI/derived CSV and static figures. Parent independently rebuilt derived artifacts offline and verified SHA256/raw/schedule/result consistency. CSV writer uses LF for repository whitespace checks; captured source raw unchanged.

All202scenario validations passed. Short-window optimized TTFT means2K194.97->70.85ms,8K484.93->137.00ms,16K1190.62->864.61ms (N10 each). Entire-cycle gains are not universal. Original16K negative and hostpressure pre-fix regression, background completion throughput decline are explicitly retained. Cross-snapshot comparisons contain P4/P5 and harness/N confounds, not isolated causal proof. Mechanistic2GET->1GET proof is separate. B3 and capacity correction are later additive records.

## P7c Router feedback reconciliation acceptance — 2026-11-04

Trusted discovered Engine journal/cache identities, bounded feedback cursors/diagnostic facts, pre/post scoped-query watermarks, gap/restart/invalidation barriers and current shared routing publication checks are integrated. Historical operation outcomes never install location; L3 uncertainty suppresses positive cached receipts without releasing sibling interests. Unhinted Engine caller=request facts remain outside configured-caller polling; rendered request association remains explicit. No global snapshot/HA claim.

Parent independent17HTTP passed; integrated full serial Router proxy241passed/1existingignored(34.78s). Final integration caught three old test-literal shapes after P7a wire/P7b protobuf extensions; fixed absent session/default legacy source fields in two existing Router fixtures, then full suite passed. Builder Router lib1113passed; main wire24passed separately. Evidence p7c-report.md, p7c-parent-http-tests.log, p7c-integrated-http-tests.log(failed intermediate fixture compile retained), p7c-integrated-proxy-tests.log. Corrected local tokenizer CLI path to tokenizer.json.

## P8 native host-pressure admission correction — 2026-11-04

Measured full-host trials identified a free-only native PREFETCH admission gate that prevented ordinary unpinned host reclamation. Native admission now checks destination/transfer budgets before calling existing evict_host for its positive shortfall under requested-prefix holds. PIN/in-flight refs remain native exclusions. Insufficient scans share100ms monotonic backoff; already freed capacity proceeds immediately; all expired interests cannot introduce this new side effect. Whole-node granularity may reclaim beyond the shortfall; missing backend data can still cause predictive displacement.

Five real-cache regressions cover exactshortfall/existingprefix, PIN/sharedtimeout/cancel, destination/rate refusal, immediate use of freed capacity, expired caller. Parent independent126/90 priorcaseversion and final integrated154passed96subtests(22.56s); independentGPU host/file/liveMooncake2passed2subtests(20.77s). Builder final127passed90subtests. Baseline new testcase fails before fix. Repeated existing event-publisher fixture port failures traced to async ZMQ probe teardown; fixture now uses synchronous TCP probe release, preserving rank-offset/event assertions. Final combined run passed after this test-only correction. Reports/logs: /tmp/sglang-rfc-build/p8-capacity-report.md, p8-capacity-parent-tests.log, p8-capacity-parent-gpu-tests.log, final-control-fixed-port-tests.log; earlier failing port logs retained. Real pressure before/after is still required after runtime snapshot restart.

## P8 final integrated acceptance — 2026-11-04

Runtime snapshot f65f0393c4 (Router Rust equivalent35a18bf008), two real Qwen3-8B/Mooncake Engines. Final CPU154passed96subtests; GPU host/file/liveMooncake2passed2subtests; Router lib1114passed serial39.47s; proxy241passed1existingignored; indexer156+orchestrator44passed1optionalignored; benchmark16passed; scheduler11passed4subtests. Default parallel Router unit attempt hit three existing subscriber timing timeouts, full failed transcript retained and complete serial proof passed. Full build and targeted Rustfmt pass. No known functional validation failure remains.

Final hostpressure6/6valid (N3pairs): TTFT591.20->96.41ms(-83.7%), complete foreground cycle5782.69->4874.56ms(-15.7%),3/3fullhostready. Final short-window18/18valid:2K297.07->170.74ms,8K807.69->422.53ms,16K998.78->468.90ms.8K/16K were0/3ready at resume and share in-flight storage reads; attribution stays storage.2Kcycle+1.5% and initial background throughput costs remain explicit; no universal throughput claim.

Real P6Router7/7valid; final legacy3/3/reorg2/2valid, formal Router picked preparedB and actualhostreuse verified. Readonly feedback proof confirms both currentowners/nativewatermarks and scoped-only recovery. Real finalL2claimactive/released; two sharedMooncakeclaimsactive, firsttaskreleasepreserves sibling, bothterminal backendquiescent. Startup-before-ready connection refusal produced no trial and is retained. Total238effective scenarios plus independent protection proofs, immutable compressedraw/schedules/manifests, negative results and source/hardware provenance are archived. Offline initial6run verifier and additive36trial/protection verifier both passed; SHA256SUMS fully verified.

Docs PLAN.md, Router runbook and final report now describe completed P0–P8 first profile and deferred P9. Report: benchmark/session_kv_control/results/2026-11-04/FINAL_ACCEPTANCE.md. Captured source/reflog preserve their original whitespace as evidence; newly generated CSV usesLF. Own model/Router/Mooncake ports51801/51803/12910/12911/52091–52093 stopped after idle proofs; externaljobs untouched. Retained scratch worktrees support review. No push performed; commits have no Codex coauthor.

## mem_fraction_static=0.8 — 持续压力测试工具

用户要求提升实际显存预算、扩大任务数量并延长测试。两台任务 Engine 使用 `mem_fraction_static=0.8`、无 `max_total_tokens` 上限、32 请求上限、HiCache ratio=2、Mooncake TCP 每 Engine 128GB。B 实测 L1=346944、L2=693952 tokens；A L1=346688，自动分配差异 0.074%。测试工具只对无显式 cap 的自动容量容许 1% 差异，并保留实际数值。服务源代码保持不可变 `f65f0393c4`，用于后续优化前对照。

沿用 codex-build 工作流；固定任务书 `/tmp/sglang-rfc-build/mem08/spec/STRESS_PLAN.md`。独立工作树实现后，root 阅读全部新增实现与测试，检查 L2 salt 隔离、并发/失败计账、固定恢复截止时间和 PIN 清理。真实冷请求暴露 `cached_tokens=0/details=null` 的工具校验问题，补充回归后通过。root 独立运行 `python -W error::ResourceWarning -m unittest -v test_stress_bench test_session_kv_bench`：20 项通过，2.107s。

真实 Mooncake 双 Engine 冒烟：L3 两组共 4 条恢复请求、L2 两组共 14 条生成请求全部有效，PIN release/query 无清理失败。初次无效 source smoke 和原 calibration 的自动容量预检拒绝保留为启动观察，不混入正式性能结果。资源采样器已实际读取双 Engine metrics/loads、Master metrics 和 GPU0/1 显存/利用率。

正式矩阵采用 64×16K 会话、两次遍历、3 轮配对，扫描 8/16 并发；L2 使用 4×16K 热点和 53×16K 冷数据冲刷、3 轮配对。无每请求 cache flush；L3 两组均在请求完成后 DEMOTE，显式测量 host/storage 行为。收益尚待正式数据判定。按用户已有授权提交工具，不推送。

### 0.8 压力口径补充：自然 GPU 驱逐

显式 DEMOTE 会释放 GPU 前缀，无法单独证明大 L1 KV 池受到容量压力，因此补充 `--retention-mode natural`。L2 热点仍按工具等待语义 DEMOTE/PIN，冷压力仅 COPY 到 host、保留 GPU KV；L3 正式请求后不主动回收 GPU，让原生 LRU 与 write_through 工作。默认 demote 保留已有口径，manifest/arm/summary 明确标记模式。root 审阅全部变更并独立测试：23 项通过（日志 harness-natural-tests-root.log）。运行中的 L2 demote 对照继续使用已加载的 08be8787e6 工具，不改动其解释器或 Engine。后续主矩阵使用 natural，按实际 GPU/host 使用量证明压力。

### 有界 native L3→L2 流水

固定规格 `/tmp/sglang-rfc-build/mem08/spec/NATIVE_PIPELINE_PLAN.md`；隔离实现后由 root 阅读全部生产与测试 diff。`max_concurrent_storage_reads` 默认 4，范围 1–64；不同有效 salt 可流水，同 salt 串行、GPU 目标独占，完整 host resident 目标直接完成。查询阶段预留完整缺口，真实 host allocation 后停止重复计账，取消/超时保留到实际 drain；PIN、原生引用和 eviction backoff 不变。

root 审阅发现较早 GPU waiter 可能被后续 host job 持续填充挤占，补固定 admission 顺序与 live-interest 屏障；取消及过期解除屏障，已有完整 host 目标仍可完成。未增加底层 GET worker。CPU fake-driver fixture 的注销路径按模拟 CUDA 注册配对，运行环境用不存在的 CUDA 设备 999 隐藏 GPU；仓库 test_utils 不能接受空 CUDA_VISIBLE_DEVICES。

root 独立 proof：最终 transfers 42 passed、25 subtests，16.19s；cache/service/protection/feedback/controller 76 passed、68 subtests，22.29s。测试涵盖并发上限、相同 salt、容量重复批准、allocation 后不重复扣减、PIN 压力与退避、取消/期限/query miss/partial、精确共享与 demand 所有权、GPU DMA fence 及 FIFO 饥饿边界。日志 native-fifo-transfers-root.log、native-regressions-root-v2.log。当前性能长测仍由不可变 f65 源执行；本提交的真实收益待同配置长测比较。按用户授权测试通过后提交，不推送。

### 0.8 离线证据复核

新增 analyze_stress.py 与验证测试。root 审阅全部实现及测试，补充“恢复完成但 cache metadata 不可验证”仍保留延迟与吞吐的回归。独立验证共 39 项通过（1.915s）；对已完成显式 L2 六组实际记录运行离线 CLI，366 条请求、12 条释放 tombstone、3 组各 4 条 active/full PIN 均通过，数值重算及 SHA-256 无错误。

监控语义经源码核对：token_usage/kv_used_tokens 仅为 active，不能冒充全部缓存占用。分开记录 active/evictable/free，并用实际启动容量减 free 推导 resident/reserved；自然驱逐实验已观察到 free=2880/346944，即约 99.17% 分配占用。资源为 2s 采样、边界估计，物理 I/O 与精确峰值不推断。失败/未完成实验及负收益保持可见。

### 计时复核：实验重置成本

自然 L2 首轮两组 flush 分别约 11.9s/59.3s，足以改变整组吞吐。分析器增加全部初始 flush 原始时长、保留 inclusive wall，同时仅扣除实验 flush 计算 work throughput；capability refresh、真实发布/压力/恢复及控制清理仍计入。资源窗口从 raw arm_result 时间减 arm.wall_ms 推导，注明未知 logging/validation gap，防止用 earliest request 加 full wall 把 59s 重置误移到下一组。

root 阅读全部差异并独立验证：40 项通过（analysis-reset-tests-root.log）。新用例两组实际工作均 150s、flush 为 10s/60s，inclusive 吞吐不同而 work 吞吐相同；原始窗口无跨组重叠。已完成显式 L2 的 work 总时长按需 421.662s、PIN 422.218s，仍为约持平，未借调整口径抹去负差异。

### 持续压力发现：排队 native job 与正式请求重复读取

真实 C8 基线首轮 PREFETCH 128 条均有效，提前就绪仅 2 条，整组约 196s；同轮按需组约 157s。受控 CPU file-store 复现确认 queued native job 未加入 _joinable，正式请求会独立恢复，两个范围发生 3 次 query/GET。host exact job 在注册 interest 后立即可发现，保持原有 salt/完整 prefix/extra_key/stop-policy 约束，修复后物理 query/GET 为 2 次，普通 demand 实际 span 同为 (4,0)。

保护边界：全 PIN 空间无法批准时，已有 demand 的 queued job 真实 DEFERRED，正常进度接口消费零新增 storage span 后允许计算；GPU/FIFO barrier 阻止新 join，join 后出现 GPU load 也 DEFERRED，原 DMA/prefill fence 保留。native-only 仍等待，取消/期限不撤销有效 demand，abort 清理后未启动任务不产生 I/O。

root 阅读全部生产与测试差异并独立运行 transfers、cache、service、protection、feedback、controller：123 passed、96 subtests，24.59s（native-queued-regressions-root.log）。生产文件测试 SHA256=daff864598fe3e266ed5a7352d3d6e5e9267c9f3f7255334fcb5fd885160778a；源码、命令、前后机制证据见 native-proof-manifest.json 与 native_queued_demand_repro.py。运行中的基线 f65 不变；更新尚未启动的优化 runtime 后，以相同 0.8/大池配置完成真实 GPU/Mooncake 验证及 8/16/32 长测。按用户授权提交，不推送。

### 长测监控归档规模

2s 完整 Prometheus/loads 采样持续增长，归档器支持有序 gzip parts 与逐文件 SHA-256，避免单文件超过常用 Git 托管上限。单文件与 parts 同时出现时拒绝，重复/缺失/路径越界/哈希不匹配时拒绝，不丢弃指标。独立测试 41 项通过，新增用例证明两片采样都计入且改写任一片会被检出；原单文件与压缩 cells 保持兼容。

### 0.8 L2 性能阶段归档

两种 L2 口径共 732 次真实生成均有效；完整 HTTP/SSE/control、schedule、manifest、scope/claim/tombstone、启动命令、分阶段测试与 2s 原始监控已压缩归档到 benchmark/session_kv_control/results/2026-11-04-mem08。保留压缩前索引与压缩后物理 SHA，root 对实际打包目录离线重算：2 cells 全部 verified/valid，732 原始调用与完成数一致。SHA256SUMS 全部通过。

结果：natural 热点 TTFT 4552.012→586.192ms（约−87.1%）；仅排除实验 flush 后，全工作耗时 453.716/451.510s，总吞吐约+0.49%。demote 热点 3630.534→573.820ms（约−84.2%），工作耗时 421.662/422.218s，总吞吐约−0.13%。L1/L2 natural 采样分配占用约99.24%/99.75%；热点收益与整体吞吐区分，全部负差异及59.3s flush 干扰保留。N=3配对轮次，热点每组N=12，不能宣称稳定尾延迟或跨环境吞吐。L3长测尚在继续。

## 0.8 显存预算强化长测完成

正式执行时段 2026-11-04 08:07:53–10:24:54 UTC，约137分钟，含数据准备、实验重置和快照切换。5724次正式生成全部完成，93879792 input tokens（含缓存）、150060 output tokens；4656条相同输入恢复跨快照/负载输出一致。两个Engine均mem_fraction_static0.8，无max_total_tokens cap，B L1346944/L2693952，actualnatural分配占用约99.24%/99.75%；MooncakeTCP128GiB每Engine，64×16K工作集、C8/C16/C32，3轮配对、两次遍历。

源码快照f65与ff764308fb独立保存。新C8对同轮按需组：平均TTFT−47.96%、work吞吐+11.47%、p95−28.31%；C16平均TTFT−9.94%、work吞吐−1.05%；C32平均TTFT−28.17%、work吞吐+13.92%、p95+16.18%。该工作吞吐包含3s工具等待、响应与控制收尾，仅排除初始实验flush；HTTP指标另列，不能冒充生产无等待GPU容量。跨快照按需基线自身也改变，存储历史/时段未隔离，不能把绝对加速全归因代码。L2热点−87.1%，混合吞吐约持平。

原C32保留1条FAILED/incomplete_storage_read，普通请求日志先完整L3load随后准入范围丢失并重算；新C32保留4条DEFERRED/whole_scope_capacity，ready空，正常计算退回，全部输出一致。它们计入所有完成请求，不筛掉。严格每次PF成功矩阵为6/8，离线verified为8/8。Host驱逐仅处理evicted且无引用的host叶节点，GPU副本与在途引用可能限制L2可回收空间；本轮保持该语义，不以驱逐GPU热数据换取所有PF成功。

CPU123测试/96子测试、工具41测试、真实GPU/file/Mooncake2测试/2子测试通过。实际CPU file-store前后独立重放3query/3GET→2/2，span相同。0.8实际GPU证明在旧矩阵完成且队列为空后执行；优化runtime正确性通过后启动。结束捕获idle/caps/serverinfo，停止且核实本任务Engine/Master及采样器；端口51801/51803/52091/52092/52093无监听，GPU0/1回到130/78MiB，GPU2/3未操作。

归档共8正式矩阵与启动观察、4570条完整2s监控，以9个gzip parts无损重建原SHA；各cell保留原始与压缩后checksum，CPU进程树计数、全部命令/环境/源hash/正确性日志、静态PNG/SVG/CSV均保存。verify_acceptance.py实际通过：所有文件hash、5724调用、4656输出、已知1+4未完成控制及4570采样无损重建。32并发p95回退、CPU成本、N3与合成轨迹/单机TCP限制写入最终报告。性能原始数据与限制见 benchmark/session_kv_control/results/2026-11-04-mem08/FINAL_ACCEPTANCE.zh-CN.md。按授权分功能提交，未推送。

### 生成归档格式核查

CSV 标准 CRLF、matplotlib SVG path 换行及一份原始测试输出含生成器/捕获空白。通过结果目录局部 .gitattributes 标明 CSV 的 cr-at-eol，保留生成 SVG 和原始 .log 字节，未放宽源码或测试检查。补充物理 SHA 后，归档 diff whitespace 检查与 SHA256SUMS 通过；性能数据未改写。

## Act 3 — Build：生命周期优化 O0（2026-11-04）

实现依据为 SESSION_KV_OPTIMIZATION_PLAN.zh-CN.md，基线 f9943a83fa。用户授权按功能测试后提交，并要求控制改动范围与维护成本。沿用 codex-build 的实现、完整差异审阅、parent 独立复测流程，使用隔离 worktree，提交由 parent 执行。

O0 增加普通恢复与 native 读取的查询/GET 累积调用、时间及 anchor KV payload tokens/bytes；wire bytes 保持 unknown。native 观测按物理 job 计数，逻辑兴趣与真实 demand 分开；容量与预留 gauge 经已有 loads/capacity 查询返回，不修改公共三动作 schema。

parent 已阅读完整五文件差异并独立运行 transfers/cache/feedback/service：76 passed、39 subtests，22.22s。仓库要求的 ruff F401/F821/F823/UP037 检查通过。日志为 /tmp/sglang-rfc-build/lifecycle-opt/o0-parent-tests.log。本提交不宣称性能改善；后续同二进制配对将验证观测成本与策略收益。

### O3a — host→storage DEMOTE

增加 FULL host-only 叶节点/可回收后缀的保存后定向回收，复用 HiCache 原生写入及完整前缀确认。owner 在 ACK 后释放自有引用，再检查 PIN/计算/DMA/结构依赖。定向 helper 不递归回收范围外祖先；GPU 驻留的 host 副本仍保留并报告 source_gpu_resident。legacy File 正向 metadata 缓存不能提供新鲜确认时，不声明该破坏性路径。

parent 完整审阅生产差异和新增测试，并独立运行 host_demote/transfers/protection/service/manager：121 passed、107 subtests，23.42s。实际 CPU K/V 恢复及 H2D 内容一致；write_through/write_back、部分写入、共享前缀、split、引用和取消均覆盖。selected ruff 通过；真实 GPU/Mooncake 与性能评估在后续阶段执行。日志：/tmp/sglang-rfc-build/lifecycle-opt/o3a-parent-tests.log。

### O1 — native 预算与真实需求优先

增加默认关闭、page 对齐的 native 在途预算、speculative 在途预算、demand host 余量与可选需求优先。普通读取占用参与 native 准入判断，普通推理保留原恢复路径。按成功准入计数，每八次 demand 给背景一次机会；保留 GPU FIFO、同 salt 约束。复核发现 >64 个受阻背景任务可遮住可运行需求，已修复为背景轮转加一次有界机会，并添加 68-job 测试。

parent 审阅完整差异并独立执行 transfers/service/cache/prefetch lifecycle：88 passed、66 subtests，63.38s；活性修复后独立复测 transfers：54 passed、42 subtests，23.25s。日志为 lifecycle-opt/o1-parent-tests.log 和 o1-parent-fairness-tests.log。预算不分配新 slots，未释放取消后的在途预留，不扩大 GPU 驱逐范围。

### O2 — 分组准备成本与有界提前量

新增独立、有界的 preparation 模块，按实际选中 worker/epoch/兼容布局/backend namespace 与前缀大小分组，使用最近成本分位及最大提前量。完整 L3→L2 准备样本与未完成操作的耗时下界区分；其他 tier path 不混入。成本包括实际 admission/control/publication，不能当作纯 I/O。未配置 preparation_policy 时保留旧策略。

复核修正了预选 worker 与实际执行 worker 不一致、H2D 样本混入、后端身份变化，以及 tool_done 提前结束时遗留旧 due 的问题；旧分支完成记录不会代替增长后的当前范围。parent 已审阅完整差异。core 48 项测试通过（另 1 项 live 后端用例按原配置忽略），Router HTTP 19 项通过，覆盖早到、增长范围、late 部分重叠、旧策略与反馈边界。日志为 lifecycle-opt/o2-core-tests.log、o2-router-tests.log。Rust fmt 通过；本机精确 1.92.0 工具链已补齐，后续将执行该版本检查。

### O3a — 实际 GPU/Mooncake 字节验收

实际 GPU/file/Mooncake 原有 2 项测试及 2 子测试通过。新增非恒定、分层 K/V 数据的 host DEMOTE→L3→GPU 回读测试，实际 host slots 释放及逐元素 K/V 一致检查通过，18.79s；测试采用原生 JSON action 封装及实际 tensor dtype，未使用模型输出代替搬运字节证明。

真实 Qwen3 2K 控制观察保留于 lifecycle-opt/live-host-demote-proof.json：首次完整 prefill 与恢复生成的一位数字不同。对照 v2 证明不执行 DEMOTE 的 GPU 暖缓存也出现同样差异；GPU 暖缓存、仅 host 恢复、L3 回收后恢复的输出完全一致，实际释放/恢复 2048 tokens，后续请求 host 命中 2048。该观察用于区分计算形状数值变化与 KV 数据错误，不删除首次观察。日志：gpu-mooncake-tests.log、gpu-host-demote-exact-tests-v4.log、live-host-demote-v2-proof.log。

### O5 — 重复生命周期的 wire 身份

发现并修复持续恢复中的历史结果重放：Router ack 忘记本地 Job 后，新任务的相同目标会重建相同 Engine message ID；在 Engine 历史 COMPLETED 尚未过期且实际副本已驱逐时，只重放旧结果。现按 Router actor incarnation 与单调序号生成新 episode 的 wire ID，存活的逻辑 job 仍共享，单次提交丢响应仍重试同一 wire ID。

parent 完整审阅 17 行 runtime 改动与回归，独立运行 HTTP orchestration 20 项通过，1.31s。测试保留旧 Engine COMPLETED、清除实际 residency，在任务切换和 Router 重启后核验三个独立 wire ID 与真实重新准备。日志：lifecycle-opt/o5-wire-parent-tests.log。

### O5/O7 — 持续生命周期实验工具

新增 additive lifecycle_bench/analyze_lifecycle 与文档/测试，复用原 HTTP/SSE、控制、配置对齐和统计工具，原 stress/router 入口保持原口径。按需、固定预取、生命周期策略复用同一固定任务计划；控制收尾有界异步，背压及期末 drain 计入，保留发送迟到、所有完成响应和无效路径。支持共享祖先/salt、预测误差、取消、路由归属和真实压力采样；模拟 rollout 与真实 trainer 指标分开。

parent 审阅完整工具与测试，修正 tool_wait RPC 耗时未从 resume_expected 剩余时间扣除的问题，并增加确定性时钟测试。每组保存实际 capacity 成本快照，缺失明确 unknown；源首次生成及暖缓存参考都保留并计入源成本。独立复测新增 14 项通过，7.266s；旧 stress/analyze/router 41 项在首次组合复测通过。Ruff、格式及文档链接检查通过。日志：lifecycle-opt/o5-harness-parent-final-tests.log；正式 GPU 结果另行归档。

### O4 — 有界准备后 PIN

在原准备 Job 内挂载一次有限 L2 PIN 子操作，默认 TTL=0 关闭；已有兴趣共享决定清理，失效不自动重获。保守 token 预算、操作/claim 计数和父子 prefix/payload material 均受限；未知申请或释放保留预算并阻止 ack。内部子 claim 身份绑定 wire episode；公开准备观察与 auto_pin 结果分开。host 来源的生命周期 DEMOTE 仅在新策略配置后开放。

parent 阅读完整差异与测试，要求将无 claim 查询的拒绝快捷路径限制为内部新 claim，防止显式失败续期掩盖旧有效 claim。修正后 parent 独立复测 HTTP orchestration 27 项通过，3.14s；core 49 项通过，另 1 项 live 测试按原配置忽略。真实请求发送不会释放保护；共享清理、丢响应、未知释放、TTL、缺失窗口及 material 拒绝均覆盖。Rustfmt 1.92.0 通过；Clippy 无新增诊断，原 backend_protection/observations/protocol 的七项诊断独立记录。日志：lifecycle-opt/o4-parent-http-tests.log、o4-parent-core-tests.log。

### O5 — 避免短工具等待中的升降往返

新增最小预期冷层驻留窗口 min_cold_residency_ms，默认 3000ms。新策略仅在 host 来源的生命周期 tool_wait 中，比较剩余等待减准备提前量；窗口不足则保留 host 副本。显式 submit 指令及旧策略保持原行为。该门槛避免仅因压力就 DEMOTE 随即 PREFETCH 同一短期上下文。

parent 独立 HTTP 回归 28 项通过；新增测试同时验证近恢复跳过、较长等待提交实际 DEMOTE，包含保护及旧行为回归。日志：lifecycle-opt/o5-cold-window-parent-tests.log。

### O5 — 有效观测去重与消费期轮询

4-case 启动观察保留了普通/候选组的数字输出差异和负性能结果。GPU 字节证明与暖路径对照已通过；三组重启统一启用上游 deterministic inference，继续保持 0.8、Mooncake、TP1、自然缓存容量，后续配置另列，不与旧非确定性结果混作代码净收益。

启动 trace 显示完整 ready 后仍有重复 query 与正式 decode 重叠。新策略在原 poll freshness、反馈水位及 worker 健康条件内复用已完成观测，取消及 PIN 子操作仍独立推进；PIN missing 明确撤销父 readiness。harness 完整观测或请求开始后停止主动轮询，消费结束再收尾，保留观测年龄。大段服务/成本快照仅保留逐组证据文件，汇总和 console 不重复展开。

parent 独立 HTTP orchestration 29 项通过，4.71s；harness 15 项通过，8.777s。覆盖窗口内去重、过期/失败后重新查询、取消推进，以及完整准备后无持续轮询。日志：lifecycle-opt/o5-observation-parent-tests.log、o5-observation-harness-tests.log。正式性能使用 release Router，源码与配置另行记录。

### O7 — 完整前缀的恢复输入与启动验证

恢复输入在发布前缀后追加一个固定合成工具结果 token；源参考使用相同完整输入，发布目标保持页对齐前缀。这样按需与准备组均可复用完整前缀，避免将生成 logits 必须重算的末页计为缓存恢复缺失。request_observed 报告真实完整源输入，Router 继续按既有语义关联完整页；mock 同步采用所声明页大小。全部输入规则、初次/暖参考及发送记录保留。

parent 独立 lifecycle/Router 工具 23 项通过，19.739s；Ruff 和 diff 检查通过。deterministic+release 的 4-case 启动验证三组均 4/4 完成、输出相同且 control drain 完成，重复前缀 prefill 为零。生命周期平均 TTFT 107ms 对按需 302ms，工作吞吐 +4.9%；仅为小工作集配置验证，不作为正式容量压力结论。证据：lifecycle-opt/smoke-deterministic-release，离线复核随后统一归档。

### Rust 1.92.0 兼容与静态检查

修正 core 中七项 Clippy 诊断：actor 大请求经 Box 传递，内部 mutation 元组增加私有类型别名，整数对齐检查使用 is_multiple_of。范围限定为三个文件，没有变更 wire schema、配置或调度语义。

parent 阅读完整差异，并用仓库要求的精确 Rust 1.92.0 独立执行 core 全目标 Clippy（-D warnings）及 core 测试：49 项通过，另 1 项 live 后端测试按原配置忽略。日志：lifecycle-opt/core-clippy-parent.log、core-1.92-parent-tests.log。该 CPU 检查窗口单独记录，不作为正式性能样本。

### O7 — 配对轮次的不可变源复用

新增默认关闭的 reuse-published-sources：轮内共用 salt、源输出参考和一次实际发布，轮间重新隔离，后续组只重置目标 B。历史引用明确标记 reference_only，不伪造重复 RPC 或当前 L3 驻留。源 GPU 调用及执行墙钟按实际计数，公共准备成本等分进入完整作业口径；恢复请求数、输入和压力门槛保持原定义。L3 容量竞争与保护继承实验继续使用逐组发布。

parent 审阅完整四文件差异，并独立复测 lifecycle/Router 工具 25 项通过，23.186s；格式和 selected Ruff 通过。新增 64-case、两轮、三组测试核对 256 次真实源调用与 384 次恢复、各组成本分摊及篡改来源检测。日志：lifecycle-opt/source-reuse-parent-tests.log。正在运行的原 C8 筛选沿用已加载的逐组发布代码。

### O5 — 终态回执复用

C8 压力筛选三组 384 次恢复输出一致，生命周期组合组工作吞吐降低 21.6%。注册前等待均值 1143ms，对照为 84ms；task_started 的 HTTP 均值 990ms，对照为 41ms。生命周期组与固定预取组 GET payload 完全相同；Engine 请求总量明显增加，证据支持先减少 actor 串行 RPC 等待，不能将退化直接归因于容量或带宽。

新策略下复用已终结且 quiescent 的操作回执，仍独立验证当前 capabilities、feedback、residency/claim；QUEUED 即使 quiescent 也继续查询。首次提交和 PIN release 新消息至少查询一次。parent 阅读完整差异并独立 HTTP 回归 32 项通过，2.79s；精确 Rust 1.92 格式检查通过。日志：lifecycle-opt/terminal-parent-http-tests.log。正在运行的 C16/C32 保留旧二进制，后续版本重新评估，不混合为同一配置结果。

### O7 — 尚未获得 Engine 回执的计划任务

C16 首轮筛选出现六次工具观测异常：合法 scheduled Job 的 result 为 null，客户端统计 child status 时调用了 dict.get。恢复输出仍全部一致，控制最终均 quiescent；这些原始异常及该轮未通过验收的状态保留。

修正为将未获得回执的 child status 记为 unknown，完整准备保持 false，消费后仍按原路径清理。parent 新增直接覆盖该真实边界的回归，独立 lifecycle 工具 18 项通过，14.824s；selected Ruff、格式及 diff 检查通过。日志：lifecycle-opt/null-receipt-parent-tests.log。正在执行的旧进程不改写，后续新进程采用修复。

### O7 — 成本与资源离线复算

新增独立 analyze_lifecycle_costs 入口，复用既有资源解析和轮次统计。按实际工作区间筛选资源采样，GPU 容量读取唯一 TP1 internal state，host/gpu 分配与活动 token 分开；CPU 仅按稳定 PID/create-time 区间计量，缺口、重启及捕获边界未知显式保留。存储增量需要同 epoch 的前后请求及完整响应证据，gauge 不差分，缺失/回退为 unknown；GET payload 与 wire bytes 分开。

parent 阅读完整生产代码与测试差异，独立新增及既有分析 30 项通过，2.229s；覆盖缺失统计、计数重置、多池、PID 复用、坏记录、压缩分片、所有完成请求尾延迟与归档重放一致。日志：lifecycle-opt/lifecycle-costs-parent-tests.log。该工具只读离线证据，不改变运行负载或接受门槛。

### O5 — 有界只读批量查询（Engine）

新增经能力协商的 read_batch 1.0，最多八个非递归只读命令，复用 capabilities/query/residency/claim/events。执行任何 child 前验证全部 context 与顶层一致，复用既有 owner、校验、错误及传输通道；三动作 schema 保持原样。Events 可以推进状态，文档明确不承诺全局原子快照，Router 仍需验证水位和反馈失效。

parent 阅读完整生产、文档及新测试差异，并独立执行 read_batch/service/feedback/protection CPU/HTTP 回归：49 项及 47 子测试通过；selected Ruff、isort、格式和 diff 检查通过。日志：lifecycle-opt/read-batch-parent-tests.log。覆盖数量/版本/递归/写命令拒绝、全量 context 预检、隔离、HTTP/msgpack、原错误和批内 generation 变化。Router 接入另提交，正在运行的旧测试进程保持原版本。

### O5 — Router 协商批量观察

新策略且 Engine 明确支持 read_batch 时，将能力、反馈、操作及 scope/claim 观察合入一次 RPC；初次发现/提交、取消与释放仍单独执行。抽取既有能力绑定及反馈应用，串行与批量共用操作身份、ticket、水位、范围和 claim 验证。日志分页未追上、gap、错误/超限回复或失效都撤销 readiness，按既有单查询路径修复后再尝试；不依赖批量“原子快照”。旧 Engine 和未配置策略保留原路径。

parent 阅读完整三个文件差异，并独立 HTTP orchestration 40 项通过，3.46s；精确 Rust 1.92 fmt 全 workspace 检查通过。新增回归核验持续 partial 观察实际只发一个 HTTP RPC、终态与 queued、PIN TTL/新 release 消息、协商回退、坏/缺/超限结果、epoch/generation/gap、分页及失效恢复。日志：lifecycle-opt/read-batch-parent-http-tests.log。后续真实压力需统一切换两端再评估，原筛选负结果完整保留。

### O7 — 显式零命中与完整重算

C32 原始响应中，按需十条、生命周期九条 cached_tokens 明确为整数 0，cached_tokens_details 为 null。上游 output_streamer 的零命中分支返回 None；旧工具将此合法响应标为详情异常，原 repeated_prefill 汇总漏掉对应完整重算。原始 invalid、异常和验收失败状态保持原样，不改写证据。

strict 模式仅将明确整数 0 解释为各层零命中，保留 details_present=false 与 explicit_zero_total 来源；缺少总量、非零或非法类型继续 unknown。非严格旧调用保持兼容。parent 阅读完整差异并独立 session/stress/analyze/lifecycle 62 项通过，17.561s，diff 检查通过；新 HTTP 回归验证完整前缀计入重算。日志：lifecycle-opt/zero-cache-parent-tests.log。只读逐条证明为 zero-cache-c32-proof.json：对应前缀重算至少 163840/147456 tokens，固定预取未出现该类响应；不能继续将旧汇总的零值作为无重算证明。

### Rust Router 静态检查

精确 Rust 1.92.0 release Clippy 检出反馈构造器原有 min/max 形式的 manual_clamp。单行替换为等价 clamp，未改变合法 limit 或协议语义。parent 独立 release lib Clippy（-D warnings）通过，HTTP orchestration 40 项复测通过，3.55s。日志：lifecycle-opt/router-clippy-1.92-final-parent.log、router-clamp-parent-http-tests.log。运行中的压力进程继续使用 61284379d0，后续二进制切换在配对实验边界进行。

### O5 — 入口绝对恢复窗口与过期推测退回

新版 C8 的 30 条准备响应晚于真实恢复时刻，18 次完整 native GET 发布时已存在整个前缀，约 43.49GB API payload 没有新增 L2。内部消息未记录入口时间，出队和 worker 发现后用相对等待计算 due，延长了预测窗口。

现内部消息携带入口 Instant，Intent 保存绝对窗口，按实际 worker 提前量计算 due。原始 Value 仍用于指纹和重放。所有存活兴趣窗口均过期时，仅未尝试的准备以 late_before_submit 正常退回；任一未来或显式无窗口兴趣可继续。已尝试/丢响应动作沿原 wire ID 收尾，已有 PIN 保留正常 TTL，新 PIN 首次写前重查窗口。恢复从未启动的共享工作会撤销旧本地 quiescence，取消仍不可逆。

parent 审阅完整 runtime/core/HTTP 差异并独立复测：48 HTTP 项通过，5.24s；精确 Rust 1.92 核心 50 项通过、1 live 项按原配置忽略。精确 1.92 release lib Clippy（-D warnings）及全 workspace fmt 通过。日志：lifecycle-opt/resume-window-parent-http-tests.log、resume-window-parent-core-tests.log、resume-window-parent-clippy.log。仅将旧回归中无法跨真实 HTTP 存活的 1ms 窗口调整为仍小于估计提前量的 100ms；新增排队/发现、共享/取消/重放、PIN 子发现期间到期等真实等待回归。

### O7 — 延迟观测到的正常退回

工具将中间 query 或最终 drain 才出现的 late_before_submit 标记累计到 policy_fallback，保留原始响应及初始退回状态，不把正常策略退回记作异常。已创建 intent 的 admitted 状态不变，继续完整 cancel/query/ack，未取得 Engine 回执仍为 unknown。

parent 审阅完整两文件差异，并独立 lifecycle 工具 20 项通过，14.756s；selected Ruff、格式及 diff 检查通过。新增回归分别在 query/drain 阶段给出退回标记，证明各执行一次 cancel/ack、最终 quiescent 且不产生错误；无标记的 null 回执仍计零。日志：lifecycle-opt/fallback-parent-tests.log。
