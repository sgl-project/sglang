# Kimi-Linear CP-v2 Implementation Plan

> **For Codex:** Execute this plan test-first in the current branch. Keep the
> production change limited to the Kimi-Linear layer transitions and CP-v2
> activation hooks.

**Goal:** Run Kimi-Linear MLA layers on zigzag CP shards while running KDA layers
on complete token batches replicated across the four TP ranks.

**Architecture:** A Kimi-specific CP-v2 layer communicator converts
`hidden_states` and `residual` at KDA/MLA boundaries using the active
`ContextParallelStrategy`. `KimiDecoderLayer` invokes it before input RMSNorm
and after MLA attention, before the TP MLP. The existing CP-v2 eager runner
continues to own model-entry splitting and model-exit gathering.

**Tech Stack:** Python, PyTorch distributed collectives, SGLang CP-v2 strategy
API, `unittest`, four NVIDIA GB300 GPUs, GSM8K evaluation.

---

### Task 1: Specify layer-transition behavior with a failing unit test

**Files:**

- Create: `test/registered/cp/test_kimi_linear_cp_v2.py`
- Test: `test/registered/cp/test_kimi_linear_cp_v2.py`

**Step 1: Write the first failing test**

Create a CPU-registered `CustomTestCase` with a recording CP strategy. The first
test constructs a communicator for the first KDA layer, patches CP-v2 active,
and asserts that its rank-local `hidden_states` are passed through
`gather_hidden_states` while `residual=None` is preserved.

```python
communicator = KimiLinearCPV2LayerCommunicator(
    is_kda_layer=True,
    previous_is_kda_layer=None,
)
hidden_states, residual = communicator.prepare_attn(
    hidden_states, None, forward_batch
)
self.assertEqual(strategy.gather_calls, 1)
self.assertIsNone(residual)
```

**Step 2: Run the focused test and confirm RED**

Run:

```bash
python -m unittest test.registered.cp.test_kimi_linear_cp_v2 -v
```

Expected: failure because the communicator module/class does not exist.

### Task 2: Implement the minimal CP-v2 communicator

**Files:**

- Create: `python/sglang/srt/layers/cp/kimi_linear.py`
- Modify: `test/registered/cp/test_kimi_linear_cp_v2.py`
- Test: `test/registered/cp/test_kimi_linear_cp_v2.py`

**Step 1: Implement first-KDA gather**

Define `KimiLinearCPV2LayerCommunicator` with static layer-type inputs and a
`prepare_attn` method. Gate it with `is_cp_v2_active`; obtain the strategy with
`get_cp_strategy`; gather the model-entry shard for a first KDA layer.

**Step 2: Run the focused test and confirm GREEN**

Run the same unit-test command and require it to pass.

**Step 3: Add one failing transition test at a time**

Add and run tests for:

- KDA to MLA: shard `hidden_states` and `residual`.
- MLA attention to MLP: gather `hidden_states` and `residual`.
- KDA to KDA: identity/no strategy calls.
- CP-v2 inactive: identity/no strategy calls.

For each case, first observe failure, then implement the smallest transition
logic needed to pass it.

**Step 4: Add a real zigzag round-trip test**

Use `ZigzagCPStrategy` metadata and the existing fake CP group pattern to prove
that a full tensor split for MLA and gathered before its MLP returns to original
token order, including its residual tensor.

**Step 5: Run communicator and existing strategy tests**

```bash
python -m unittest \
  test.registered.cp.test_kimi_linear_cp_v2 \
  test.registered.cp.test_cp_strategy_unit -v
```

Expected: all pass.

### Task 3: Wire the communicator into Kimi-Linear

**Files:**

- Modify: `python/sglang/srt/models/kimi_linear.py`
- Modify: `test/registered/cp/test_kimi_linear_cp_v2.py`
- Test: `test/registered/cp/test_kimi_linear_cp_v2.py`

**Step 1: Add a failing wiring test**

Verify the model exposes a communicator configured from each layer's current
and previous KDA/MLA type, including `previous_is_kda_layer=None` for layer zero.
Use a minimal Kimi config or constructor patching so the test remains CPU-only.

**Step 2: Construct and invoke the communicator**

In `KimiDecoderLayer.__init__`, compute current and previous layer types from
`KimiLinearConfig.is_kda_layer`. Construct the communicator. At the very start
of `forward`, call `prepare_attn` before input RMSNorm.

After MLA attention, call `prepare_mlp` before post-attention RMSNorm and the
TP MLP. Shard the final full layer output so the generic model-exit gather keeps
its CP-v2 contract.

**Step 3: Run the focused test and confirm GREEN**

```bash
python -m unittest test.registered.cp.test_kimi_linear_cp_v2 -v
```

### Task 4: Enable Kimi-Linear in the CP-v2 eager path

**Files:**

- Modify: `python/sglang/srt/layers/cp/utils.py`
- Modify: `python/sglang/srt/models/kimi_linear.py`
- Modify: `test/registered/cp/test_kimi_linear_cp_v2.py`

**Step 1: Add failing activation/accessor assertions**

Assert that `KimiLinearForCausalLM` is in `CP_V2_DEFAULT_MODEL_CLASSES` and
that its `get_input_embeddings()` accessor returns `model.embed_tokens`.

**Step 2: Add the activation and embedding hooks**

Add the architecture string to the default class set and the standard accessor
to `KimiLinearForCausalLM`.

**Step 3: Run the focused CP tests**

```bash
python -m unittest \
  test.registered.cp.test_kimi_linear_cp_v2 \
  test.registered.cp.test_cp_strategy_unit -v
```

Expected: all pass.

### Task 4b: Keep KDA on global tensor parallelism

Use global TP rank/size for KDA projections, recurrent cache shape, head
partitioning, and `A_log`/`dt_bias` weight loading. Add a regression test where
`tp_rank` differs from `attn_tp_rank`.

Add the FlashInfer MLA CP-v2 path because it is the GB300 default backend, and
test its latent-KV materialization plus zigzag dispatch.

### Task 5: Local static and regression verification

**Files:**

- Verify all modified Python and test files.

**Step 1: Run formatting and lint checks**

```bash
pre-commit run --files \
  python/sglang/srt/layers/cp/kimi_linear.py \
  python/sglang/srt/layers/cp/utils.py \
  python/sglang/srt/models/kimi_linear.py \
  test/registered/cp/test_kimi_linear_cp_v2.py
```

**Step 2: Run CP-focused test suites**

Run the new unit test, existing CP strategy unit test, and any applicable
server-argument tests selected by the diff.

**Step 3: Inspect the final local diff**

Require `git diff --check`, review all changes against the design, and confirm
no unrelated user changes are present.

### Task 6: GB300 end-to-end verification

**Files:**

- Remote checkout and logs on `baizhou-dev-2`.
- No generated benchmark artifacts committed to the repository.

**Step 1: Prepare the devbox**

Use `rx devbox run baizhou-dev-2`. Clone or update SGLang, fetch the implementation
branch, pull the latest stacked base, and install the current editable Python and
kernel dependencies before launching a job.

**Step 2: Locate or download the model**

Use `moonshotai/Kimi-Linear-48B-A3B-Instruct` from a shared cache if present;
otherwise download it to devbox-attached persistent storage.

**Step 3: Launch the requested configuration**

Start SGLang with:

```bash
--tp 4 --attn-cp-size 4 --enable-prefill-cp --cp-strategy zigzag
```

Capture the exact command, commit, model path, backend selection, and complete
server log.

**Step 4: Run GSM8K**

Use the repository evaluation command against the live endpoint. Record sample
count, accuracy, and any mismatch/error output. If no established Kimi-Linear
threshold exists, compare against a TP4 non-CP run using the same model,
tokenizer, prompts, and decoding settings.

**Step 5: Diagnose until verified**

If the server or evaluation fails, preserve the first failure signature, add a
focused regression test where feasible, and repeat local plus GB300 validation.

### Task 7: Review, commit, push, and open the stacked PR

**Files:**

- Review: all files changed from the merged PR #31619 implementation on `main`.

**Step 1: Re-run verification before claiming completion**

Capture fresh output for the focused tests, pre-commit checks, and GSM8K result.

**Step 2: Commit intentionally**

Keep the design document commit and create logically scoped implementation/test
commits. Do not fold in changes from the stacked base or unrelated worktree
state.

**Step 3: Push to the authenticated fork**

Push `codex/kimi-linear-cp-v2` to `Fridge003/sglang`.

**Step 4: Open the PR**

PR #31619 merged before publication, so rebase onto current
`sgl-project/sglang:main` and target `main`. Include the layout transition
table, unit-test commands, exact GB300 launch command, and GSM8K result.
