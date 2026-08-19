# Remove Prefill Context Parallel V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove prefill context parallel v1 and make the unversioned strategy-based CP implementation the only supported path.

**Architecture:** Canonical server arguments initialize one context-parallel strategy. The model runner owns pre-model sharding and post-model gathering, while CP strategies and attention backends own layout- and cache-specific operations; modeling code no longer contains prefill-CP branches.

**Tech Stack:** Python, PyTorch, SGLang runtime configuration, unittest/pytest, registered CUDA CI tests.

**Spec:** `docs/superpowers/specs/2026-08-15-remove-prefill-cp-v1-design.md`

## Global constraints

- Delete `SGLANG_ENABLE_CP_V2`; do not retain an environment compatibility shim.
- Delete the old prefill CP CLI aliases and fields; retain only canonical CP arguments.
- Delete `python/sglang/srt/layers/utils/cp_utils.py`; do not move its v1 implementation elsewhere.
- Remove prefill CP sharding, gathering, metadata construction, and branching from `python/sglang/srt/models/`.
- Rename only CP-specific v2 identifiers; preserve model versions and unrelated identifiers such as `DRAFT_EXTEND_V2`.
- Preserve DCP behavior and ordinary non-CP execution for ineligible batches.
- All surviving tests in `test/registered/cp` must pass on their registered CPU/GPU environments.

---

### Task 1: Remove the environment gate and legacy server arguments

**Files:**
- Modify: `python/sglang/srt/environ.py`
- Modify: `python/sglang/srt/server_args.py`
- Modify: `python/sglang/srt/arg_groups/overrides.py`
- Modify: `python/sglang/srt/arg_groups/deepseek_v4_hook.py`
- Modify: `python/sglang/srt/runtime_context.py`
- Modify: `test/registered/unit/server_args/test_server_args.py`
- Modify: `test/registered/unit/test_runtime_context.py`

**Interfaces:**
- Consumes: canonical `ServerArgs.enable_prefill_cp`, `ServerArgs.cp_strategy`, and `ServerArgs.attn_cp_size`.
- Produces: canonical configuration with no CP version gate or legacy alias fields.

- [ ] **Step 1: Replace legacy server-argument tests with canonical contract tests**

Add assertions equivalent to:

```python
args = parser.parse_args(
    ["--model", "dummy", "--enable-prefill-cp", "--cp-strategy", "interleave"]
)
server_args = ServerArgs(
    model_path="dummy",
    enable_prefill_cp=args.enable_prefill_cp,
    cp_strategy=args.cp_strategy,
    attn_cp_size=2,
)
server_args._handle_context_parallelism()
self.assertTrue(server_args.enable_prefill_cp)
self.assertEqual(server_args.cp_strategy, "interleave")
self.assertFalse(hasattr(server_args, "enable_prefill_context_parallel"))
```

Delete tests that expect legacy fields or `_handle_legacy_cp_arguments()` translations.

- [ ] **Step 2: Run the focused tests to expose the old surface**

Run:

```bash
python -m pytest -q test/registered/unit/server_args/test_server_args.py test/registered/unit/test_runtime_context.py
```

Expected before implementation: failure because the legacy dataclass fields and compatibility handler still exist.

- [ ] **Step 3: Delete the environment variable and compatibility configuration**

Remove `SGLANG_ENABLE_CP_V2`, CP v2 model allowlist auto-enabling, legacy CP dataclass fields/constants, deprecated parser aliases, `_handle_legacy_cp_arguments()`, and every call to that handler. Update canonical validation and overrides to read only `enable_prefill_cp` and `cp_strategy`.

- [ ] **Step 4: Run canonical configuration checks**

Run:

```bash
python -m compileall -q python/sglang/srt/environ.py python/sglang/srt/server_args.py python/sglang/srt/arg_groups python/sglang/srt/runtime_context.py
python -m pytest -q test/registered/unit/server_args/test_server_args.py test/registered/unit/test_runtime_context.py
```

Expected: compile checks and available tests pass; if the local environment lacks test dependencies, record the import failure for devbox execution.

- [ ] **Step 5: Commit canonical configuration**

```bash
git add python/sglang/srt/environ.py python/sglang/srt/server_args.py python/sglang/srt/arg_groups/overrides.py python/sglang/srt/arg_groups/deepseek_v4_hook.py python/sglang/srt/runtime_context.py test/registered/unit/server_args/test_server_args.py test/registered/unit/test_runtime_context.py
git commit -m "refactor: remove prefill CP v1 configuration"
```

### Task 2: Make the strategy CP API unversioned

**Files:**
- Modify: `python/sglang/srt/layers/cp/base.py`
- Modify: `python/sglang/srt/layers/cp/utils.py`
- Modify: `python/sglang/srt/layers/cp/bcg.py`
- Modify: `python/sglang/srt/layers/cp/cp_decode_attn_tp.py`
- Modify: `python/sglang/srt/model_executor/forward_batch_info.py`
- Modify: `python/sglang/srt/model_executor/model_runner.py`
- Modify: `python/sglang/srt/model_executor/runner/eager_runner.py`
- Modify: `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`
- Modify: `test/registered/cp/test_cp_strategy_unit.py`

**Interfaces:**
- Consumes: `get_cp_strategy() -> Optional[ContextParallelStrategy]`.
- Produces: `is_cp_active(forward_batch) -> bool`, `_execute_extend_cp(...)`, `cp_input_ids`, `cp_round_robin_input_ids(...)`, and `enable_cp_bcg_capture(server_args) -> bool`.

- [ ] **Step 1: Rewrite CP strategy tests against unversioned APIs**

Replace version-gate tests with behavior such as:

```python
active_batch = SimpleNamespace(
    forward_mode=_ExtendMode(),
    input_ids=torch.arange(8),
)
with patch("sglang.srt.layers.cp.utils.get_cp_strategy", return_value=strategy):
    self.assertTrue(is_cp_active(active_batch))
```

Remove all mocks of `envs.SGLANG_ENABLE_CP_V2`, rename CP-v2 test classes and fields, and preserve BCG bucket/overflow coverage.

- [ ] **Step 2: Run the renamed unit test and verify it fails first**

Run:

```bash
python -m pytest -q test/registered/cp/test_cp_strategy_unit.py
```

Expected before implementation: import or attribute failure for `is_cp_active` and unversioned BCG names.

- [ ] **Step 3: Rename APIs and remove the version gate**

Implement `is_cp_active()` using forward mode, strategy availability, input availability, and `strategy.can_apply()`. Remove `enable_cp_v2()`, `CP_V2_DEFAULT_MODEL_CLASSES`, and the legacy fallback in `cp_materialize_global_token_order()`. Rename all CP-specific fields, variables, methods, comments, and exports without changing unrelated v2 identifiers.

- [ ] **Step 4: Run CP API compile and unit checks**

Run:

```bash
python -m compileall -q python/sglang/srt/layers/cp python/sglang/srt/model_executor
python -m pytest -q test/registered/cp/test_cp_strategy_unit.py
```

Expected: unversioned CP API compiles and the CPU CP strategy suite passes in a test-capable environment.

- [ ] **Step 5: Commit the unversioned CP API**

```bash
git add python/sglang/srt/layers/cp python/sglang/srt/model_executor test/registered/cp/test_cp_strategy_unit.py
git commit -m "refactor: make prefill CP strategy API unversioned"
```

### Task 3: Remove prefill CP behavior from modeling

**Files:**
- Modify: `python/sglang/srt/models/deepseek_v2.py`
- Modify: `python/sglang/srt/models/deepseek_nextn.py`
- Modify: `python/sglang/srt/models/deepseek_v4.py`
- Modify: `python/sglang/srt/models/deepseek_v4_nextn.py`
- Modify: `python/sglang/srt/models/qwen2_moe.py`
- Modify: `python/sglang/srt/models/qwen3_moe.py`
- Modify: `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`
- Modify: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- Modify: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py`
- Modify: `python/sglang/srt/models/utils.py`
- Modify: `python/sglang/srt/layers/attention/index_topk_share.py`
- Modify: `test/registered/cp/test_cp_strategy_unit.py`

**Interfaces:**
- Consumes: rank-local input tensors prepared by `_execute_extend_cp()` and attention backend cache materialization.
- Produces: model bodies with no prefill CP metadata construction, sharding, gathering, or version selection.

- [ ] **Step 1: Capture the modeling boundary as a failing static check**

Run:

```bash
rg -n 'layers\.utils\.cp_utils|is_cp_(v2_)?active|prepare_context_parallel_metadata|cp_split_and_rebuild|cp_all_gather_rerange|mla_use_prefill_cp|is_prefill_context_parallel_enabled' python/sglang/srt/models
```

Expected before implementation: matches in the listed model files.

- [ ] **Step 2: Delete model-level v1 and version-selection branches**

Remove legacy metadata creation, input/position splitting, output gathering, and `use_cp_v1` branches. Remove v1 MLA KV rebuild branches because the attention strategy materializes full KV. Keep DCP-only logic and ordinary model execution unchanged.

- [ ] **Step 3: Move index-topk publication gathering to the sharing owner**

Make `IndexTopKShareState.update()` gather `topk_indices` through `cp_gather_after_forward()` when `is_cp_active(self._forward_batch)` and publication is required. Add a unit test that patches the gather helper, calls `update()`, and asserts the gathered tensor is stored. Remove `_gather_dsa_topk_indices_for_cp()` and CP predicates from `deepseek_nextn.py`.

- [ ] **Step 4: Verify the model boundary and syntax**

Run:

```bash
! rg -n 'layers\.utils\.cp_utils|is_cp_(v2_)?active|prepare_context_parallel_metadata|cp_split_and_rebuild|cp_all_gather_rerange|mla_use_prefill_cp|is_prefill_context_parallel_enabled' python/sglang/srt/models
python -m compileall -q python/sglang/srt/models
```

Expected: no prefill CP helper match and all model files compile.

- [ ] **Step 5: Commit the model boundary cleanup**

```bash
git add python/sglang/srt/models
git commit -m "refactor: decouple modeling from prefill CP"
```

### Task 4: Collapse attention and DSA onto the strategy implementation

**Files:**
- Modify: `python/sglang/srt/layers/attention/flashattention_backend.py`
- Modify: `python/sglang/srt/layers/attention/trtllm_mha_backend.py`
- Modify: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`
- Modify: `python/sglang/srt/layers/attention/dsa_backend.py`
- Modify: `python/sglang/srt/layers/attention/dsa/utils.py`
- Modify: `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
- Modify: `python/sglang/srt/layers/attention/dsa/dsa_npu_indexer.py`
- Modify: `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- Modify: `python/sglang/srt/hardware_backend/musa/attention/flashattention_backend.py`
- Modify: `python/sglang/kernels/ops/attention/dsv4/metadata_kernel.py`

**Interfaces:**
- Consumes: `is_cp_active(forward_batch)` and `get_cp_strategy()`.
- Produces: one CP attention/cache path, with non-CP handling only when the batch is inactive.

- [ ] **Step 1: Identify version-selection branches as the failing condition**

Run:

```bash
rg -n 'SGLANG_ENABLE_CP_V2|is_cp_v2_active|CP-v[12]|cp_all_gather_rerange_output|cp_split_and_rebuild_position' python/sglang/srt/layers/attention python/sglang/srt/hardware_backend python/sglang/kernels/ops/attention/dsv4
```

Expected before implementation: legacy/versioned matches in FlashAttention, TRT-LLM, DSA, and hardware backends.

- [ ] **Step 2: Collapse backend branches**

Rename active predicates, keep strategy calls for active batches, and delete old all-gather/reindex alternatives. Make DSA token padding use active CP metadata when active and the normal attention-TP rule otherwise; remove flag-dependent padding logic.

- [ ] **Step 3: Remove DSA v1 indexer and cache behavior**

Keep `ContextParallelStrategy.materialize_full_indexer_k_cache()` and related strategy primitives. Delete `cp_all_gather_rerange_output()` fallbacks and old round-robin helpers from DSA/NPU paths.

- [ ] **Step 4: Verify backend syntax and version-free scans**

Run:

```bash
python -m compileall -q python/sglang/srt/layers/attention python/sglang/srt/hardware_backend python/sglang/kernels/ops/attention/dsv4
! rg -n 'SGLANG_ENABLE_CP_V2|is_cp_v2_active|CP-v[12]' python/sglang/srt/layers/attention python/sglang/srt/hardware_backend python/sglang/kernels/ops/attention/dsv4
```

Expected: compilation succeeds and no CP-version marker remains.

- [ ] **Step 5: Commit the backend cleanup**

```bash
git add python/sglang/srt/layers/attention python/sglang/srt/hardware_backend python/sglang/kernels/ops/attention/dsv4
git commit -m "refactor: remove prefill CP v1 backend paths"
```

### Task 5: Delete the legacy utility and migrate remaining consumers

**Files:**
- Delete: `python/sglang/srt/layers/utils/cp_utils.py`
- Modify: `python/sglang/srt/layers/communicator.py`
- Modify: `python/sglang/srt/layers/communicator_dsa_cp.py`
- Modify: `python/sglang/srt/layers/cp/padding.py`
- Modify: `python/sglang/srt/managers/schedule_policy.py`
- Modify: `python/sglang/srt/model_executor/model_runner.py`
- Modify: `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`
- Modify: `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`
- Modify: `python/sglang/srt/model_executor/forward_batch_info.py`
- Delete: `test/registered/kernels/ops/attention/test_cp_prefix_len_fa3_parity.py`
- Delete: `test/registered/kernels/ops/attention/test_mla_cp_fa3_parity.py`

**Interfaces:**
- Consumes: canonical runtime context and the unversioned strategy API.
- Produces: no import of `sglang.srt.layers.utils.cp_utils` anywhere in Python or tests.

- [ ] **Step 1: Run the legacy import census**

Run:

```bash
rg -n 'sglang\.srt\.layers\.utils\.cp_utils' python test
```

Expected before implementation: all remaining callers and v1 parity tests are listed.

- [ ] **Step 2: Replace configuration-only helpers**

Use canonical runtime state (`get_parallel().enable_prefill_cp`) or strategy state (`is_cp_enabled()`/`is_cp_active()`) for scheduler, communicator, and graph-runner decisions. Do not preserve old split/gather helpers.

- [ ] **Step 3: Delete the legacy file and legacy kernel tests**

Delete `cp_utils.py` and the two parity tests that construct or validate legacy metadata.

- [ ] **Step 4: Verify import closure and syntax**

Run:

```bash
! rg -n 'sglang\.srt\.layers\.utils\.cp_utils' python test
test ! -e python/sglang/srt/layers/utils/cp_utils.py
python -m compileall -q python/sglang/srt
```

Expected: no legacy import, legacy utility absent, and runtime sources compile.

- [ ] **Step 5: Commit the legacy implementation deletion**

```bash
git add -A python/sglang/srt test/registered/kernels/ops/attention
git commit -m "refactor: delete prefill CP v1 implementation"
```

### Task 6: Update tests, examples, and documentation to canonical CP

**Files:**
- Delete: `test/registered/cp/test_dsa_prefill_cp_legacy.py`
- Modify: `test/registered/cp/test_deepseek_v3_cp_single_node.py`
- Modify: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`
- Modify: `test/registered/cp/test_dsa_prefill_cp.py`
- Modify: `test/registered/cp/test_gqa_prefill_cp.py`
- Modify: every tracked test/manual/docs file found by the legacy CP alias and `SGLANG_ENABLE_CP_V2` scans
- Modify: `docs/docs/references/environment_variables.mdx`
- Modify: `docs/docs/hardware-platforms/ascend-npus/optimization/parameter_tuning.mdx`

**Interfaces:**
- Consumes: canonical `--enable-prefill-cp --cp-strategy ...` CLI.
- Produces: registered tests and documentation with no v1/version-gate configuration.

- [ ] **Step 1: Make surviving registered tests use canonical configuration**

Remove all `SGLANG_ENABLE_CP_V2` environment entries, rename `TestDSACPV2Interleave` to `TestDSACPInterleave`, and replace the DeepSeek V3 legacy flag with:

```python
"--enable-prefill-cp",
"--cp-strategy",
"zigzag",
```

- [ ] **Step 2: Delete the legacy registered test and stale v1 references**

Delete `test_dsa_prefill_cp_legacy.py`. Update tracked manuals, registered tests outside the CP directory, and docs to canonical flags where they still describe supported CP; remove stale environment-variable documentation.

- [ ] **Step 3: Run repository-wide static acceptance checks**

Run:

```bash
! rg -n 'SGLANG_ENABLE_CP_V2' python test docs
! rg -n '\b(CP_V2[A-Za-z0-9_]*|[A-Za-z0-9_]*cp_v2[A-Za-z0-9_]*)\b|CP-v[12]' python test docs
! rg -n -- '--enable-(dsa-|nsa-)?prefill-context-parallel|--(dsa-|nsa-)?prefill-cp-mode' python test docs
test ! -e test/registered/cp/test_dsa_prefill_cp_legacy.py
```

Expected: every command succeeds with no forbidden match.

- [ ] **Step 4: Run formatting and registry checks**

Run:

```bash
git diff --check
pre-commit run --all-files
python scripts/ci/check_ci_register.py
```

Expected: formatting, static ratchets, and registered-test registry validation pass.

- [ ] **Step 5: Commit test and documentation migration**

```bash
git add -A test docs python
git commit -m "test: retire prefill CP v1 coverage"
```

### Task 7: Validate every surviving registered CP test

**Files:**
- Verify: `test/registered/cp/test_cp_strategy_unit.py`
- Verify: `test/registered/cp/test_deepseek_v3_cp_single_node.py`
- Verify: `test/registered/cp/test_dsa_prefill_cp.py`
- Verify: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`
- Verify: `test/registered/cp/test_gpt_oss_4gpu_mxfp4_cp.py`
- Verify: `test/registered/cp/test_gqa_prefill_cp.py`

**Interfaces:**
- Consumes: the completed implementation branch.
- Produces: recorded passing results for every surviving registered CP file.

- [ ] **Step 1: Run local final checks**

Run:

```bash
python -m compileall -q python/sglang/srt
git diff --check HEAD~1
git status --short
```

Expected: compilation and diff checks pass; only intentional uncommitted verification artifacts may remain.

- [ ] **Step 2: Run the CPU CP suite on a test-capable devbox**

Run:

```bash
python -m pytest -q test/registered/cp/test_cp_strategy_unit.py
```

Expected: pass.

- [ ] **Step 3: Run H200 CP tests on an 8x H200 devbox**

Run:

```bash
python -m pytest -q test/registered/cp/test_deepseek_v3_cp_single_node.py test/registered/cp/test_dsa_prefill_cp.py
```

Expected: both files pass.

- [ ] **Step 4: Run B200 CP tests on a 4x B200 devbox**

Run:

```bash
python -m pytest -q test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py test/registered/cp/test_gpt_oss_4gpu_mxfp4_cp.py
```

Expected: both files pass.

- [ ] **Step 5: Run GQA CP tests on a 4x H100 devbox**

Run:

```bash
python -m pytest -q test/registered/cp/test_gqa_prefill_cp.py
```

Expected: pass.

- [ ] **Step 6: Review the final branch against the spec**

Run:

```bash
git diff --stat clean-dsv4...HEAD
git diff --name-status clean-dsv4...HEAD
```

Confirm every deleted surface, preserved non-goal, and devbox result in `docs/superpowers/specs/2026-08-15-remove-prefill-cp-v1-design.md`.
