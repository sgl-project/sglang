# Pattern 0: Profile-Driven Identification + Semantic Gate (Detail)

This pattern is a **hard gate** — if semantic validation fails, STOP immediately.

## 0a. Run kernel stack analysis

Given the user provides a profiling trace path and an overlap kernel description,
use the dedicated stack analysis script in **two steps**: first discover actual
kernel names, then extract full Python call stacks.

### Step 1: Discover actual kernel names

User-provided kernel names (e.g. `reduce_scatter`, `add_`) often don't
exactly match the canonical names in the trace. Use `--list-kernels` to
discover the real names first:

```bash
python3 .claude/skills/sglang-overlap-integration/scripts/trace_kernel_stack.py \
  --input <trace_path> \
  --list-kernels \
  --kernel-filter <keyword1> <keyword2> ...
```

This outputs:
- All GPU kernels sorted by GPU time, with `[MATCH]` tags on those matching
  any filter substring
- Matching CPU op names (important for ops that appear only at the CPU op
  level)
- Suggested `--kernel-filter` values for Step 2

### Step 2: Extract full Python call stacks

Use the discovered kernel name substrings from Step 1:

```bash
python3 .claude/skills/sglang-overlap-integration/scripts/trace_kernel_stack.py \
  --input <trace_path> \
  --kernel-filter <discovered_substring1> <discovered_substring2> ... \
  --stack-depth 0 \
  --format chain
```

| Flag | Purpose |
|------|---------|
| `--list-kernels` | List all GPU kernel names sorted by GPU time; with `--kernel-filter`, mark matches with `[MATCH]` and show matching CPU ops |
| `--kernel-filter` | One or more case-insensitive substrings; matches both GPU kernel names **and** CPU op names |
| `--stack-depth` | Max Python stack frames to display. `0` = unlimited (full call stack). Default: `0`. |
| `--format table` | Per-kernel stack report with GPU time, location, and full call stack. |
| `--format chain` | Operator-chain summary for Pattern 0c (grouped by source location). |
| `--format json` | Machine-readable JSON output for downstream tooling. |
| `--output <path>` | Write to file instead of stdout. |

This script is self-contained (does not depend on the general-purpose triage
scripts) and provides **unlimited stack depth** by default, unlike the
4-frame cap in the general triage output.

For a full triage report (all kernels, overlap opportunities, fuse opportunities),
use the general-purpose script instead:

```bash
python3 .claude/skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py \
  --input <trace_path>
```

## 0b. Locate target operators

From the overlap kernel description, identify each operator in the trace:
- Match GPU kernels (`cat=kernel` events) by name substring
- Match CPU ops (`cat=cpu_op` events) by op name
- Use `python_function` events for source scope
- Confirm: non-trivial GPU time, same CUDA stream (sequential), temporal ordering

## 0c. Build operator chain + source mapping

For each operator, record: GPU kernel name, CPU op name, Python source location,
input tensor shapes, stream, temporal order, per-layer GPU time.

Output the chain in temporal order and the source mapping for each step.

## 0d. Semantic equivalence validation

Must verify all of the following:

1. **Compute equivalence** — same math operation (reduction type, scaling application, dtype handling)
2. **Communication equivalence** — same collective type, data partitioning, reduction operation
3. **Shape invariance** — same input/output/intermediate shapes
4. **Semantic ordering** — same operation order (order matters when inputs differ across ranks)
5. **Scaling factor compatibility** — same timing and mechanism of scaling

**If any check fails**: output the mismatch reason and **STOP**. Do not proceed.

## 0e. Summary output

```
=== Operator Chain to Replace ===
Chain: <op1> → <op2> → <op3>
Total GPU time / Per-layer / Stream / Source mapping
Semantics: VALIDATED or MISMATCH (with reason)
```

## 0f. Source-mapping target level

When building the source mapping in Step 0c, the target location must be the
**model-level replacement point** — the highest-level function in the model
code that orchestrates the entire compute+communication chain — not the
low-level communication primitive.

**Rule**: Walk up the Python call stack from each GPU kernel until you reach
the model layer (typically a `forward()` method or similar orchestrator on an
`nn.Module` subclass in the model files, or a model-layer helper function in
the layer modules). That model-level function is the insertion point where
Pattern 5 and Pattern 6 will inject the `_maybe_fused_*()` fast-path and
bypass redundant collectives.

**Why**: Pattern 5 needs to intercept the full operator chain (compute +
elementwise + communication) at a level where all inputs and the fused
output are accessible, and Pattern 6 needs to skip the surrounding
collective call. Low-level wrappers in communication modules cannot serve as
insertion points because they lack access to the compute inputs needed by
the fused kernel.

**Stack-level heuristic** — classify each frame in the call stack:

| Stack level | How to identify | Use as target? |
|---|---|---|
| **Low** | Communication primitive wrappers (e.g., `reduce_scatter_tensor`, `all_reduce`) in distributed/parallel modules | No — no access to compute inputs |
| **Mid** | Layer-internal helpers (e.g., elementwise fuse functions, quantization helpers) that handle only one step of the chain | Partial — only covers part of the chain |
| **High** | Model-layer orchestrator functions (e.g., `forward()` or a helper called from it) that invoke the full compute→elementwise→communication sequence | **Yes** — all inputs and the collective call are in scope |

The low-level primitive is still useful for confirming the GPU kernel
identity (e.g., verifying the exact NCCL/CUDA kernel name), but the source
mapping for the operator chain should point to the model-level function that
will be modified.
