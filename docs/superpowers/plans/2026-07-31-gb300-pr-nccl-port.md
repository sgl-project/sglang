# GB300 PR Test NCCL Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every distributed server launch in GB300 pull-request tests an explicit, unique NCCL rendezvous port below the runner's ephemeral port range.

**Architecture:** Keep port selection local to the three GB300 PR test files. Each file derives a reserved port block from `DEFAULT_PORT_FOR_SRT_TEST_RUNNER`, and each server launch passes a distinct value through `--nccl-port`; production allocation and nightly tests remain untouched.

**Tech Stack:** Python, unittest-based registered CI tests, Ruff, pre-commit, GitHub Actions `/rerun-test`.

## Global Constraints

- Do not modify `get_free_port()` or production port-allocation logic.
- Do not modify GB300 nightly tests.
- Cover exactly seven server launches across the three GB300 PR server tests.
- Use CI ports `10100`, `10101`, `10110`–`10112`, `10120`, and `10121`.
- Keep the existing DeepSeek CuteDSL assignments unchanged.

---

### Task 1: Prove the configuration gap

**Files:**
- Inspect: `test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py`
- Inspect: `test/registered/ep/test_flashinfer_a2a.py`
- Inspect: `test/registered/disaggregation/test_disaggregation_aarch64.py`

**Interfaces:**
- Consumes: Python AST for the three registered test modules.
- Produces: A red/green static regression command that requires one explicit `--nccl-port` per server launch.

- [ ] **Step 1: Run the failing static regression check**

```bash
python3 - <<'PY'
import ast
from pathlib import Path

expected = {
    "test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py": (2, 100),
    "test/registered/ep/test_flashinfer_a2a.py": (3, 110),
    "test/registered/disaggregation/test_disaggregation_aarch64.py": (2, 120),
}
for filename, (launch_count, base_offset) in expected.items():
    tree = ast.parse(Path(filename).read_text(), filename=filename)
    launches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"popen_launch_server", "popen_launch_pd_server"}
    ]
    nccl_flags = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value == "--nccl-port"
    ]
    bases = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "NCCL_PORT_BASE"
            for target in node.targets
        )
    ]
    assert len(launches) == launch_count, (filename, len(launches))
    assert len(nccl_flags) == launch_count, (filename, len(nccl_flags))
    assert len(bases) == 1, (filename, len(bases))
    assert isinstance(bases[0].value, ast.BinOp)
    assert isinstance(bases[0].value.right, ast.Constant)
    assert bases[0].value.right.value == base_offset
PY
```

Expected: FAIL for `test_flashinfer_a2a.py` because it has zero explicit NCCL flags instead of three.

### Task 2: Assign ports to FlashInfer A2A launches

**Files:**
- Modify: `test/registered/ep/test_flashinfer_a2a.py`

**Interfaces:**
- Consumes: `DEFAULT_PORT_FOR_SRT_TEST_RUNNER`.
- Produces: `NCCL_PORT_BASE` at offset `+110`, with three launch arguments at offsets `+0`, `+1`, and `+2`.

- [ ] **Step 1: Add the test-local base**

Add `DEFAULT_PORT_FOR_SRT_TEST_RUNNER` to the existing `test_utils` import and define:

```python
# Keep rendezvous ports below the ephemeral range on the 4-GPU GB300 runner.
NCCL_PORT_BASE = DEFAULT_PORT_FOR_SRT_TEST_RUNNER + 110
```

- [ ] **Step 2: Pass a distinct port to each launch**

Add these argument pairs to the three `other_args` lists:

```python
"--nccl-port",
str(NCCL_PORT_BASE),
```

```python
"--nccl-port",
str(NCCL_PORT_BASE + 1),
```

```python
"--nccl-port",
str(NCCL_PORT_BASE + 2),
```

### Task 3: Assign ports to disaggregation launches

**Files:**
- Modify: `test/registered/disaggregation/test_disaggregation_aarch64.py`

**Interfaces:**
- Consumes: `DEFAULT_PORT_FOR_SRT_TEST_RUNNER`.
- Produces: `NCCL_PORT_BASE` at offset `+120`, with prefill and decode launch arguments at offsets `+0` and `+1`.

- [ ] **Step 1: Add the test-local base**

Add `DEFAULT_PORT_FOR_SRT_TEST_RUNNER` to the existing `test_utils` import and define:

```python
# Keep rendezvous ports below the ephemeral range on the 4-GPU GB300 runner.
NCCL_PORT_BASE = DEFAULT_PORT_FOR_SRT_TEST_RUNNER + 120
```

- [ ] **Step 2: Pass distinct ports to prefill and decode**

Add to `prefill_args`:

```python
"--nccl-port",
str(NCCL_PORT_BASE),
```

Add to `decode_args`:

```python
"--nccl-port",
str(NCCL_PORT_BASE + 1),
```

### Task 4: Verify, publish, and exercise the GB300 job

**Files:**
- Verify: all three affected test files
- Update: PR #33044

**Interfaces:**
- Consumes: the seven explicit port assignments.
- Produces: a pushed PR revision and terminal GitHub Actions results.

- [ ] **Step 1: Re-run the static regression check**

Run the exact command from Task 1.

Expected: exit code 0, with launch counts `2`, `3`, and `2`, and base offsets `100`, `110`, and `120`.

- [ ] **Step 2: Run formatting, lint, and compilation**

```bash
ruff check \
  test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py \
  test/registered/ep/test_flashinfer_a2a.py \
  test/registered/disaggregation/test_disaggregation_aarch64.py
ruff format --check \
  test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py \
  test/registered/ep/test_flashinfer_a2a.py \
  test/registered/disaggregation/test_disaggregation_aarch64.py
python3 -m py_compile \
  test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py \
  test/registered/ep/test_flashinfer_a2a.py \
  test/registered/disaggregation/test_disaggregation_aarch64.py
```

- [ ] **Step 3: Run repository checks**

```bash
pre-commit run --all-files
git diff --check
```

- [ ] **Step 4: Commit and push**

```bash
git add \
  docs/superpowers/plans/2026-07-31-gb300-pr-nccl-port.md \
  test/registered/ep/test_flashinfer_a2a.py \
  test/registered/disaggregation/test_disaggregation_aarch64.py
git commit -m "Pin NCCL ports across GB300 PR tests"
git push origin codex/pin-gb300-nccl-ports
```

- [ ] **Step 5: Trigger the same GB300 rerun**

Post this exact PR comment:

```text
/rerun-test test/registered/utils/test_numa_utils.py test/registered/ep/test_flashinfer_a2a.py test/registered/disaggregation/test_disaggregation_aarch64.py test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py
```

- [ ] **Step 6: Monitor and diagnose**

Poll the bot result comment and every linked workflow until terminal. For any
failure, inspect the earliest failing test and full traceback; distinguish port
collisions from unrelated model, runner, or test failures.
