---
name: sglang-overlap-integration
description: >
  Integrate an existing overlap kernel (compute-communication fusion) into the SGLang serving
  framework. Requires a profiling trace file path and the overlap kernel source code from the
  user — ask for them if not provided. This skill locates target operators via profiler analysis,
  validates semantic equivalence, and performs framework-level plumbing — kernel placement,
  communicator setup, env-var dispatch, and model code replacement. Use this skill when the
  user wants to merge a pre-written overlap kernel into SGLang's distributed inference pipeline,
  wire up symmetric-memory-based communication for a custom kernel, or add an overlap kernel
  as a drop-in replacement for existing compute+communication operators. Also trigger when the
  user mentions: "integrate overlap kernel into SGLang", "add overlap layer to SGLang",
  "hook overlap kernel into model_runner", "SGLang overlap kernel integration",
  "wire up overlap kernel with nvshmem/torch_symm_mem in SGLang", or describes a step-by-step
  plan to add a fused compute-comm kernel to the SGLang model execution flow.
---

# SGLang Overlap Kernel Integration

## Overview

This skill guides the integration of an existing overlap kernel (compute-communication fusion)
into the SGLang LLM serving framework. It does NOT generate kernel code — it assumes the
overlap kernel already exists and handles identification, validation, and framework-level plumbing.

## Prerequisites — Required User Inputs

This skill requires two mandatory inputs. **If either is missing, stop and ask the user — do not proceed with guesswork or code inspection alone.**

| # | Input | Format | Used By |
|---|-------|--------|---------|
| 1 | **Profiling trace file path** | `.json` or `.json.gz` from `torch.profiler.profile(with_stack=True, record_shapes=True)` on a distributed run (TP≥2) | Pattern 0 — locates target compute/communication operators and maps them to source code |
| 2 | **Overlap kernel source code** | File path or pasted code | Pattern 1 — determines compute op, communication collective, symm-mem mechanism, I/O shapes, and context state |

When asking the user, request:
1. The absolute path to the profiling trace (must include CUDA kernel events and communication collectives).
2. The overlap kernel source — file path or pasted code.

Do not skip Pattern 0, fabricate operator chains from code inspection, or begin code changes (Patterns 2–6) before both inputs are provided.

## Minimal-Invasion Principle

All integration changes must obey these four rules:

1. **Do not touch unrelated kernels.** Only modify the code path that the overlap kernel
   directly replaces. Other kernels' computation logic and input parameters should remain
   completely unchanged.

2. **Do not introduce unnecessary computation.** The fused path should replace existing
   work, not add new work. If the overlap kernel subsumes a step, skip that step — do not
   compute it redundantly "just in case."

3. **Use communicator per-iteration state to control bypass.** 
   Add a flag (e.g., `comm.enable_<kernel_name>_fused_kernel`) in the communicator instance to control
   whether the fused kernel is enabled in the current iteration or not. Its value is usually 
   decided before the beginning of the model's decoder layer loop.
   When you need to tell a downstream function "skip this step because the fused kernel is 
   enabled and already handled it," try to get the communication group and check the communicator 
   instance's per-iteration decided flag (e.g., `comm.enable_<kernel_name>_fused_kernel`) at the bypass site 
   directly. This avoids redundant checks in the fused path.

These rules exist because SGLang's model code is shared across many configurations.
Spreading overlap-specific logic into unrelated code paths creates maintenance burden
and subtle correctness risks for non-overlap users.

---

## Core Integration Patterns

The integration consists of **seven patterns**, each building on the previous one:

| # | Pattern | Purpose |
|---|---------|---------|
| 0 | Profile-driven identification + semantic gate | Locate target operators via profiling, validate equivalence; **stop if mismatched** |
| 1 | Kernel understanding | Identify compute/communication/symm-mem mechanism in the kernel |
| 2 | Self-contained placement | Place kernel under `symm_mem_kernels/` with no external deps |
| 3 | Communicator setup | Extend existing communicator or create new one; lazy-init context helpers with shape-key caching |
| 4 | Env-var gating | Add `SGLANG_OPT_USE_*` env var via `environ.py` for opt-in control |
| 5 | Fused fast-path + fallback | Add `_maybe_fused_*()` with `None`-return fallback; upstream bypass if needed |
| 6 | Redundant communication bypass | Skip model-layer collective when fused kernel handles it internally |

---

## Pattern 0: Profile-Driven Identification + Semantic Gate

This pattern is a **hard gate** — if semantic validation fails, STOP immediately.

Use the profiler analysis script to locate target operators in the trace, build an
operator chain with source mapping, and validate semantic equivalence between the
overlap kernel and the original sequential operators.

**Read [`references/pattern0-profiler-analysis.md`](references/pattern0-profiler-analysis.md)**
for the full step-by-step procedure (script commands, flags, operator chain format,
semantic validation checklist, and source-mapping target level heuristic).

Key points:
- Run `trace_kernel_stack.py` in two steps: `--list-kernels` first, then `--format chain`
- Map to **model-level** replacement points (not low-level primitives) — see Pattern 0f in the reference
- If semantic validation fails → output mismatch reason and **STOP**

---

## Pattern 1: Kernel Understanding

Before touching SGLang code, understand the overlap kernel:

- **Compute operator**: GEMM / attention / normalization / reduction / ...
- **Communication collective**: all-reduce / reduce-scatter / all-gather / ...
- **Symmetric memory mechanism**:
  - `torch.distributed._symmetric_memory` → `torch_symm_mem.empty()`, `multimem_all_reduce_`, etc.
  - `nvshmem4py` → `nvshmem.core.tensor()`, `nvshmem.bindings.mc_ptr()`, etc.
- **I/O shapes and dtypes**
- **Context/state needed at launch** (buffers, barriers, handles)
- **Target call sites** in SGLang model code (from Pattern 0 source mapping)

---

## Pattern 2: Self-Contained Kernel Placement

**Rule: overlap kernel code must be self-contained — no external third-party dependency.**

Place kernel files under `symm_mem_kernels/<kernel_name>_symm_mem.py`. Each kernel
exposes: a Context class (with `finalize()`), a context factory, and an op function.

**Read [`references/pattern2-kernel-placement.md`](references/pattern2-kernel-placement.md)**
for placement conventions, context class patterns, and op function conventions.

---

## Pattern 3: Communicator Setup

**Principle: symmetric memory initialization does NOT happen in `parallel_state.py`.**
Context objects are created lazily on first use inside communicator methods.

This pattern has two branches depending on whether a matching communicator already exists.

### Branch A: Extend existing communicator

If the overlap kernel uses a symmetric memory mechanism that already has a communicator
(e.g., `torch.distributed._symmetric_memory` → `TorchSymmMemCommunicator`):

#### Communicator creation condition (static, at startup)

In `parallel_state.py`, extend the communicator's creation condition to include the new
fused kernel env var (OR with existing conditions). This ensures that **as long as the
fused kernel env var is enabled**, the communicator must be guaranteed to be instantiated 
if the fused kernel env var is on — regardless of whether the fused kernel will actually 
run in every iteration.

Do NOT initialize any symm-mem buffers or contexts here — only ensure the communicator
object itself is created. The ctx is lazy-initialized on first use (see below).

#### Lazy-init context helper

Add `get_or_create_<name>_ctx()` to the existing communicator class:

- Store `self._<name>_ctx` and `self._<name>_key` as instance attributes
- Build a **shape key** tuple from model-structure parameters only (dims fixed
  after model init, dtype, world_size, etc.) — **do NOT include per-batch
  parameters (num_tokens, etc.)** in the key
- If key matches and context exists → return cached context immediately
- If key differs → call `finalize()` on stale context, then create new one via factory
- Derive the buffer's max token budget from `server_args` — allocate at the
  **theoretical maximum** so the buffer accommodates any batch size without
  reallocation
- Return `None` when the relevant signal variable indicates the path is unavailable

The ctx is created lazily on first actual invocation of the fused kernel path
(i.e., when `enable_fused_kernel` is `True` for the first time), not at startup.

**Critical: allocate once, reuse forever.** Never include per-batch values in the
shape key. Allocate at the theoretical maximum from `server_args`. The shape key
should only contain values fixed after model initialization.

```python
def _get_max_forward_tokens(self) -> int:
    server_args = get_global_server_args()
    chunked = server_args.chunked_prefill_size
    if chunked is not None and chunked > 0:
        return chunked
    return server_args.max_prefill_tokens
```

### Branch B: Create new communicator

If the overlap kernel uses a symmetric memory mechanism that does NOT yet have a communicator
in SGLang (e.g., `nvshmem4py`, `cuda.core.experimental`), a new communicator class must
be created and registered.

#### New communicator class

Create a new communicator file under `device_communicators/` (e.g., `<mechanism>_communicator.py`).
Follow the existing communicator interface conventions:

1. **Constructor** — validate device capability and world size; set `self.unavailable = True`
   on failure and return early; on success, allocate symmetric memory buffers, perform
   rendezvous, then set `self.unavailable = False`
2. **Signal variables** — env-var-driven booleans to gate different execution paths:
   - One env var → one signal variable (e.g., `self.fused_kernel_enabled`)
   - Only when multiple env vars control different paths within the same communicator,
     introduce additional signal variables (e.g., `self.allreduce_enabled` for the base
     allreduce env var, `self.fused_kernel_enabled` for the fused kernel env var)
   - All path gating should use these signal booleans, NOT raw env var reads at call sites
3. **Lazy-init context helpers** — same shape-key-cached pattern as Branch A
4. **Runtime mode flags** — add as needed (e.g., `use_cp`)
5. **Cleanup** — release symmetric memory resources on destruction

#### Register in GroupCoordinator

In `parallel_state.py`, `GroupCoordinator`:

1. Add a class-level type annotation: `<name>_comm: Optional[<CommunicatorClass>]`
2. In `__init__`, add the creation condition gated by env var and `world_size > 1`
3. Do NOT initialize any symm-mem buffers here — only instantiate the communicator

### Communicator access from model code (both branches)

Access via `get_tp_group().<name>_comm`. Add a model-class helper that checks all
prerequisites (comm not None, not unavailable, signal variables enabled, `enable_fused_kernel`
active, runtime mode active) and returns `None` otherwise.

### Signal variable pattern (both branches)

Communicator availability is determined by env-var-driven signal booleans:

- `self.unavailable` — communicator is not usable (device/hardware prerequisites not met)
- Additional signal variables are added **only** when multiple env vars control different
  execution paths within the same communicator (e.g., one env var for allreduce, another
  for fused kernel — then use two separate signals instead of a single `unavailable`)

Do NOT hardcode fixed flag pairs (like `disabled`/`allreduce_disabled`) as a universal
pattern. Instead, add signal variables that correspond one-to-one with the env vars that
gate each path.

### Per-iteration `enable_fused_kernel` flag (both branches)

The integration uses **two-level gating**:

| Level | Controls | Determined when | Location |
|-------|----------|----------------|----------|
| **Static** (env var) | Fused path *available* — communicator instantiated | Server startup | `environ.py` + `parallel_state.py` |
| **Dynamic** (flag) | Fused path *active* for current batch | Each forward pass | `comm.enable_fused_kernel` |

#### Design

1. **Define** on communicator: `self.enable_fused_kernel: bool = False`

2. **Set per-iteration** before the layer loop — **all conditions evaluated here**:
   ```python
   comm = get_tp_group().<name>_comm
   if comm is not None and not comm.disabled:
       comm.enable_fused_kernel = (
           <phase_condition>(forward_batch.forward_mode)
           and <runtime_mode_condition>  # e.g., CP mode active, a2a backend check
       )
   ```
   This is the **single point of truth** for whether the fused path runs. Incorporate
   every relevant condition here:
   - Forward mode (prefill/decode/both)
   - Runtime mode (CP mode, TP mode, a2a backend, etc.)
   - Any other prerequisite that the original code path checks

   **Do NOT** leave conditions to be checked later by reusing existing variables
   (like `use_reduce_scatter`, `_use_cp`) at the fused-path entry site. All conditions
   are folded into this one flag assignment.

3. **Reset** after the layer loop: `comm.enable_fused_kernel = False`

4. **Check** in `_maybe_fused_*()` — only the per-iteration flag, nothing else:
   ```python
   def _maybe_fused_<name>(self, ...):
       comm = get_tp_group().<name>_comm
       if comm is None or comm.disabled or not comm.enable_fused_kernel:
           return None
       ctx = comm.get_or_create_<name>_ctx(...)  # lazy-init on first actual use
       if ctx is None:
           return None
       # ... proceed with fused kernel
   ```

#### Two-level interaction flow

```
Env var OFF → communicator not created → fused path never entered
Env var ON  → communicator created (ctx deferred)
  └─ Each iteration:
       ForwardMode matches → enable_fused_kernel = True → ctx lazy-created/reused
       ForwardMode mismatches → enable_fused_kernel = False → original path runs
```

The `enable_fused_kernel` flag is independent of `change_state()` — it is managed
exclusively by model-level forward dispatch code.

---

## Pattern 4: Env-Var Gating

Follow the `env-var-conventions` skill strictly.

### Definition

In `environ.py`, add to `Envs` class:
- `SGLANG_OPT_` prefix for optimization toggles
- `EnvBool(False)` — default off, opt-in
- Do NOT use `os.getenv`, `get_bool_env_var`, or `SGL_` prefix

### Access at call sites

Use `envs.SGLANG_OPT_USE_*_FUSED_KERNEL.get()` — never read the env var directly.

### Runtime mode flag (optional)

If the kernel only runs under specific conditions (e.g., CP mode prefill):
- Add a runtime flag on the communicator
- Set it `True` before the layer loop when condition is active, reset `False` after
- Check it in the model's comm accessor helper

---

## Pattern 5: Fused Fast-Path + Fallback

The core integration pattern — replace sequential compute+communication with the fused kernel.

### Find target call sites

Use source mapping from Pattern 0. Compute steps live in kernel launch functions;
auxiliary steps are typically elementwise PyTorch ops in combine/output paths;
collective steps are called via `get_tp_group()` methods.

### Add `_maybe_fused_*()` method

Add to the relevant model class (MoE, MLP, attention). This method:

1. Checks all prerequisites (including `comm.enable_fused_kernel`) → returns `None` if any unmet
2. Calls comm's lazy-init context helper → returns `None` if context unavailable
3. Calls the overlap op function
4. Returns result on success, `None` on failure

The prerequisite check must include the per-iteration phase flag:
```python
comm = get_tp_group().<name>_comm
if comm is None or comm.disabled or not comm.enable_fused_kernel:
    return None
```

### Insert with transparent fallback

In the model's forward method, call `_maybe_fused_*()` **before** the original code:

```
fused_out = self._maybe_fused_*(...)
if fused_out is not None:
    return fused_out
# Original sequential path — unchanged, serves as fallback
```

**The entry condition for the fused path must ONLY depend on `comm.enable_fused_kernel`**
(checked inside `_maybe_fused_*()`). Do NOT gate the call with existing variables like
`use_reduce_scatter`, `_use_cp`, or any other original-path variable. Those conditions
are already folded into the per-iteration flag at the Model-level set point.

This guarantees:
- Zero overhead when env var is off
- Graceful fallback on any failure
- Original code path never deleted
- No coupling between fused-path routing and original-path variables

### Upstream bypass (when kernel subsumes an upstream step)

If the overlap kernel handles a step that upstream normally does, the bypass site
must query the communicator's per-iteration state directly — **never add a new
parameter to the function signature** to thread bypass intent through the call chain.

Correct pattern:
```python
# Inside the function that normally performs the step:
comm = get_tp_group().<name>_comm
if comm is not None and comm.enable_fused_kernel:
    pass  # fused kernel already handled this step
else:
    <original_step>
```

Forbidden pattern:
```python
# DO NOT do this — pollutes function signatures with overlap-specific flags
def some_upstream_fn(x, ..., skip_because_fused=False):
    if not skip_because_fused:
        <original_step>
```

This keeps function signatures stable and ensures unrelated callers are never
affected by overlap integration.

### Branch enforcement (when fused path requires a specific downstream branch)

When the fused fast-path depends on the downstream execution taking a
particular branch (e.g., outplace mode instead of inplace, skip-reduce
instead of reduce), the fused-path code **must actively force that branch**
rather than assuming the default configuration happens to match.

Relying on an `assert` is insufficient — it only crashes at runtime and
provides no recovery.  Instead:

1. **Before entering the fused path**, override the relevant config fields
   to the values the fused path requires.
2. **Always restore** the original values afterward, even on exception
   (`try/finally`).
3. **In the downstream dispatch logic**, unsupported combinations should
   gracefully fall through to the correct branch (e.g., force outplace
   when incompatible flags are set) rather than assert.

This makes the fused path self-contained — it works regardless of the
user's default runner configuration.

---

## Pattern 6: Redundant Communication Bypass

When the fused kernel handles a collective internally, the surrounding model code must
skip its own collective step.

### Bypass pattern

Query the communicator's per-iteration state at the bypass site — do NOT pass a flag
through the call chain:

```python
if <condition_for_collective>:
    comm = get_tp_group().<name>_comm
    if comm is None or not comm.enable_fused_kernel:
        result = <collective_op>(result)
    # else: fused kernel already performed the collective internally
```

**Critical**: the bypass condition must be exactly correct. If the collective is skipped
when the fused kernel is NOT handling it, correctness breaks. Add a comment explaining
why the bypass is safe.

**Do NOT** add helper functions that thread bypass flags through multiple call layers.
The communicator instance is globally accessible via `get_tp_group()` — read it directly
at the point where the decision is made.

---

## Verification

After all patterns are applied:

1. **Import check** — all new imports resolve
2. **Env var off** — original path taken, no behavior change
3. **Env var on** — fused kernel called, correct output
4. **Fallback** — comm unavailable → graceful fallback to original
5. **Runtime mode** — flag set/cleared correctly if applicable
6. **Shape key caching** — context reused across same-shape calls; shape key does NOT include per-batch parameters
7. **Context allocation** — symm-mem context allocated once at max token budget (from `server_args`), never recreated on varying batch sizes
8. **Branch enforcement** — fused path actively forces required downstream branches (no asserts); original config always restored on exit
9. **Communicator initialization check** — when fused kernel env var is on:
   comm must not be `None` and `comm.disabled` must be `False`; if not,
   emit `logger.warning(...)` at startup explaining why the fused path is unavailable
10. **Per-iteration phase gating** — `enable_fused_kernel` set/reset correctly per
    `ForwardMode`; fused path never invoked in unsupported phase
11. **Minimal-invasion check** — confirm:
    - No computation added that didn't exist before (only replaced or skipped)
    - Bypass decisions read `comm.enable_fused_kernel` at the bypass site, not threaded flags
    - No existing variables (e.g., `use_reduce_scatter`, `_use_cp`, `no_combine`) reused
      to gate the fused path — only the per-iteration flag is consulted
    - No other kernel's config/mode temporarily modified to produce intermediate results
      for the overlap kernel (e.g., forcing `no_combine=True` on experts is forbidden)

---

## Reference Files

| File | Purpose |
|------|---------|
| `python/sglang/srt/environ.py` | Env var definitions (`Envs` class) |
| `python/sglang/srt/distributed/parallel_state.py` | Communicator creation (`GroupCoordinator`) |
| `python/sglang/srt/distributed/device_communicators/torch_symm_mem.py` | `TorchSymmMemCommunicator` — reference for communicator interface patterns |
| `python/sglang/srt/distributed/device_communicators/symm_mem_kernels/` | Self-contained overlap kernel directory |
| `python/sglang/srt/model_executor/forward_batch_info.py` | `ForwardMode` enum — `is_extend()`, `is_decode()` for phase detection |
| `python/sglang/srt/layers/communicator.py` | `LayerCommunicator` — reference for per-iteration context and scatter mode patterns |
| `.claude/skills/sglang-overlap-integration/scripts/trace_kernel_stack.py` | Targeted kernel stack extraction — filter by kernel name, unlimited stack depth |
| `.claude/skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py` | General-purpose profiler triage (all kernels, overlap, fuse) |
| `.claude/skills/env-var-conventions/SKILL.md` | Env var naming conventions |

---

## Integration Checklist

- [ ] **P0**: Run profiler analysis, locate operators, build chain + source mapping (map to **model-level** replacement points, not low-level primitives — see Pattern 0f)
- [ ] **P0**: Validate semantic equivalence; if mismatched → STOP
- [ ] **P1**: Understand kernel (compute/comm/symm-mem/call sites)
- [ ] **P2**: Migrate kernel to `symm_mem_kernels/` (self-contained, no external deps)
- [ ] **P2**: Add exports to `__init__.py`; implement context class with `finalize()`
- [ ] **P3**: If matching communicator exists → extend with `get_or_create_*_ctx()` lazy-init helper; else → create new communicator class and register in `GroupCoordinator`
- [ ] **P3**: Extend communicator creation condition in `parallel_state.py` (or add new comm registration)
- [ ] **P3**: Set up signal variables (one per env var gating a distinct path)
- [ ] **P3**: Define `enable_fused_kernel` flag on communicator; confirm user-specified target phase(s) (prefill/decode/both); implement per-iteration set/reset logic in model forward; **fold ALL conditions (forward mode, runtime mode, CP/TP) into the single flag assignment**
- [ ] **P4**: Define `SGLANG_OPT_USE_*` env var in `environ.py`
- [ ] **P5**: Add `_maybe_fused_*()` fast-path + fallback in model forward; entry condition ONLY checks `comm.enable_fused_kernel` — no existing variables reused
- [ ] **P5**: Add upstream bypass using communicator per-iteration state (NOT function parameter propagation)
- [ ] **P5**: Force required downstream branches when entering fused path (override config, try/finally restore); downstream dispatch must fall through gracefully instead of asserting
- [ ] **P6**: Bypass redundant collective by reading communicator state at bypass site (NOT call-chain flags)
- [ ] Verify: env-var-off path, env-var-on path, fallback, runtime mode, shape caching, no-per-batch key, max-budget allocation, branch enforcement
- [ ] Verify: minimal-invasion — no unrelated kernel signatures changed, no unnecessary computation added, no call-chain parameter pollution, no existing variables reused for fused-path routing, no other kernel's config temporarily modified
- [ ] Verify: communicator initialization check — env var on but comm unavailable → warning log emitted at startup
- [ ] Verify: per-iteration phase gating — `enable_fused_kernel` set/reset correctly per `ForwardMode`; fused path never invoked in unsupported phase