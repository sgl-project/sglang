# Attention Backend Unit Test Suite — Onboarding

This document is for new developers joining the attention-backend test
effort. It summarizes the suite's purpose, layout, conventions, and known
limitations, with enough pointers to get productive in a single sitting.

If you only have time for the executive summary, read **§1 (What this suite
covers)** and **§4 (Adding a new test)**, then skim **§7 (Known limitations)**.

---

## Table of Contents

1. [What this suite covers](#1-what-this-suite-covers)
2. [Why a separate "unit" suite when there's also `test/registered/attention/`](#2-why-a-separate-unit-suite)
3. [Layout — fixtures, runners, and per-method test files](#3-layout)
4. [Adding a new test](#4-adding-a-new-test)
5. [Adding a new attention backend or method](#5-adding-a-new-backend-or-method)
6. [Conventions worth knowing before you change anything](#6-conventions)
7. [Known limitations (production-side bugs and structural rejects)](#7-known-limitations)
8. [Running tests + debugging tips](#8-running--debugging)
9. [Hardware coverage today](#9-hardware-coverage)

---

## 1. What this suite covers

The suite verifies that **every SGLang attention backend produces correct
output for every supported `(attention method, forward mode, input shape,
runner mode, speculative kind, cache layout)` combination**.

Concretely it tests:

- **Attention methods**: standard MHA / GQA / MQA, sliding-window (SWA),
  MLA (DeepSeek), DSA (DeepSeek sparse), DSV4 (DeepSeek V4 compressed),
  dual-chunk, plus the linear / state-space family GDN / KDA / Lightning /
  Mamba2.
- **Backends**: `torch_native`, `triton`, `flashinfer`, `fa3`, `fa4`,
  `flex_attention`, `flashmla`, `cutlass_mla`, `trtllm_mha`, `trtllm_mla`,
  `tokenspeed_mla`, `dsa`, `dsv4`, `dual_chunk_flash_attn`, plus the linear-
  attention backends GDN / KDA / Lightning / Mamba2.
- **Forward modes**: `DECODE`, `EXTEND`, `MIXED`, `TARGET_VERIFY`,
  `DRAFT_EXTEND`, `DRAFT_EXTEND_V2`.
- **Runner modes**: eager, CUDA graph (capture + replay with distinct data),
  PCG / BCG split-op, plus the production speculative graph runners
  (`EAGLEDraftCudaGraphRunner`, `EAGLEDraftExtendCudaGraphRunner`,
  `FrozenKVMTPCudaGraphRunner`).
- **Speculative kinds**: EAGLE chain (topk=1) + tree (topk=2),
  Frozen-KV MTP, DFlash, NGRAM.
- **Input shapes**: page size 1, page-boundary triplet (seq_len ∈
  {page−1, page, page+1}), prefix-exact-page, total-exact-page, cross-page,
  ragged batches, decode with nonzero prefix, bsz=1 decode, SWA window-edge
  triplet.
- **Cache layouts (`loc_layout`)**: `contiguous` (baseline regression),
  `shuffled_pages` (default everywhere), `interleaved_pages`,
  `non_monotonic_extend`. See §6 — every test exercises a non-tidy layout
  by default.

**Counts (latest H200/SM 9.0)**: 176 test methods, ~600+ subtests, ~40 s
full sweep, 30 skipped (hardware gates + documented production limitations),
0 failures.

The suite intentionally **does not** test: RoPE math (orthogonal,
pre-processing), DSV4 `Compressor`/`C4Indexer` math (model-owned `nn.Module`s
whose outputs flow into the backend), or Phase 4b worker-level integration
(needs a real `TpModelWorker` and draft-model load — belongs in
e2e/registered tests). See `PLAN.md` "Resolved Decisions" and "Deferred
follow-ups".

## 2. Why a separate "unit" suite

`test/registered/attention/` contains **end-to-end / server-level** tests
that launch a full SGLang server, load a model, and run accuracy /
benchmark workloads. They're slow (minutes) and coarse — they verify the
attention backend works *in production* but don't isolate which
`(backend × shape × mode × layout)` combination broke.

This suite is **module-level**: every test instantiates a tiny attention
module backed by the real attention-backend class, exercises one shape /
mode / layout combination, and asserts against an **independent
PyTorch reference**. A failure here pinpoints the exact backend method and
input shape that broke.

The fixtures use real `ModelRunner` subclasses, real `ReqToTokenPool`, and
real KV pools — so `isinstance` checks, metadata layouts, and pool sizing
match production. Only the model and the server are mocked away.

## 3. Layout

```
test/manual/attention/unittest/
├── PLAN.md                            (design + arc history; not in final PR)
├── KNOWN_FAILURES.md                  (every production-side limitation — read first)
├── MUTATION_FIXES.md                  (mutation-testing campaign history)
├── ONBOARDING.md                      (this file)
├── common/
│   ├── attention_methods/             (one fixture file per attention method)
│   │   ├── dense_attention.py         (MHA/GQA/MQA + shared loc-layout helper)
│   │   ├── mla_attention.py           (DeepSeek-shaped MLA)
│   │   ├── gdn_attention.py           (GDN linear)
│   │   ├── kda_attention.py           (Kimi delta attention)
│   │   ├── lightning_attention.py     (Bailing seg_la)
│   │   ├── mamba2_attention.py        (Mamba-2 SSM)
│   │   ├── dsa_attention.py           (DeepSeek sparse + dense fallback)
│   │   ├── dsv4_attention.py          (DeepSeek V4 SWA + C4 + C128)
│   │   └── dual_chunk_attention.py    (dual-chunk packed-query + sparse all-column)
│   └── runner_modes/                  (one orchestrator per runner mode)
│       ├── cuda_graph_decode_runner.py    (decode CG capture/replay)
│       ├── split_op_runner.py             (PCG/BCG extend through RadixAttention split-op)
│       ├── speculative_cuda_graph_runner.py     (shared CG capture/replay for spec)
│       ├── speculative_target_verify_runner.py  (TARGET_VERIFY chain + tree)
│       ├── speculative_draft_runner.py          (production EAGLEDraftCudaGraphRunner)
│       └── speculative_draft_extend_runner.py   (production EAGLEDraftExtendCudaGraphRunner + V2)
└── <method>/                          (one folder per attention method)
    ├── README.md                      (per-method coverage matrix + production constraints)
    └── test_<backend>.py              (one test file per backend × method combination)
```

**Three layers in a single test call**:
1. The **per-backend test file** (e.g. `mla/test_triton.py`) enumerates cases
   and forwards them to a helper.
2. The **attention-method fixture** (e.g. `mla_attention.py::run_mla_attention_case`)
   builds the runner, backend, module, reference, and forward batch, then
   runs the eager path and compares against the reference.
3. The **runner-mode orchestrator** (e.g. `cuda_graph_decode_runner.py`)
   owns the CG / split-op / spec-runner lifecycle and calls into the
   attention-method fixture via adapter callbacks.

A test file's job is **only case enumeration**. The fixture and orchestrator
own the runner contract. Don't reimplement capture/replay in a per-backend
test file — add an adapter to the orchestrator instead.

## 4. Adding a new test

### 4a. Adding a new case to an existing backend × method

The cheap, common case. Open the per-backend test file (e.g.
`mla/test_triton.py`) and add a `MLAAttentionCase(...)` entry to the
appropriate `CASES = (...)` tuple. The test method iterates the tuple.

```python
MLAAttentionCase(
    name="mla_decode_my_new_layout",     # descriptive; appears in subTest
    backend="triton",
    forward_mode=ForwardMode.DECODE,
    num_heads=4,
    page_size=16,
    prefix_lens=(14, 15, 16),            # one entry per request
)
```

Then run `python <test_file> -v` and verify the new subtest passes.

### 4b. Adding a new test method (new runner mode, new spec kind)

Add a new method on the test class:

```python
def test_runner_mode_my_new_thing_cases(self):
    for case in self.MY_NEW_CASES:
        with self.subTest(case=case.name, backend=case.backend):
            run_my_helper(self, case)
```

The helper (`run_my_helper`) lives in `common/runner_modes/` if the runner
contract is reusable across attention methods, or in
`common/attention_methods/<method>_attention.py` if it's method-specific.

### 4c. Adding the layout-robustness pattern

If your new helper threads a `loc_layout` parameter (it should), then add
`test_layout_robustness_cases` to opt into the more aggressive non-tidy
layouts beyond the default `shuffled_pages`. Template (from
`dense/test_triton.py`):

```python
LAYOUT_ROBUSTNESS_CASES = (
    # ... extend + decode cases for layout testing
)

def test_layout_robustness_cases(self):
    for case in self.LAYOUT_ROBUSTNESS_CASES:
        for layout in ("interleaved_pages", "non_monotonic_extend"):
            if layout == "non_monotonic_extend" and case.forward_mode.is_decode():
                continue
            with self.subTest(case=case.name, layout=layout):
                run_dense_attention_case(self, case, loc_layout=layout)
```

If a layout fails on your backend, record it in a `LAYOUT_KNOWN_FAILURES`
dict on the class with the production-side cause, and `skipTest` from inside
the loop. See `dense/test_fa3.py` for the canonical example.

## 5. Adding a new backend or method

### 5a. Adding a new backend to an existing method (e.g. `intel_amx` to dense)

Create `<method>/test_<backend>.py`, import the method's fixture +
runners, define `CASES = (...)`, and write the per-test methods. If the
backend has a hardware gate (e.g. requires SM ≥ X), add a `_supported()`
helper and `@unittest.skipIf(not _SUPPORTED, _SKIP_REASON)` on the class.
See `mla/test_cutlass_mla.py` for the canonical hardware-gated example.

### 5b. Adding a new attention method (e.g. a new linear-attention variant)

Substantial work. The path:

1. Create `common/attention_methods/<method>_attention.py` with:
   - A `@dataclass(frozen=True) class <Method>AttentionCase`
   - A `Tiny<Method>ModelConfig` mimicking the production HF config
   - A `Mock<Method>ModelRunner(ModelRunner)` overriding `__init__` to skip
     server startup (see existing fixtures for the standard set of mocked
     attributes)
   - A `_make_forward_batch(case, runner, *, max_context_len, device,
     loc_fn=None)` that builds a `ForwardBatch` honoring `loc_fn` (see §6.4)
   - A reference module (independent pure-PyTorch, with copied random weights
     — must NOT call `RadixAttention` or any SGLang kernel)
   - A `build_<method>_attention_fixture(...)` returning a fixture dataclass
   - A `run_<method>_attention_case(testcase, case, *, loc_layout=...)`
     entry point that does fixture build + eager + reference + assert_close

2. Add `<method>/README.md` with a coverage matrix and any
   production-unsupported constraints.

3. Add `<method>/test_<backend>.py` per supported backend, starting with a
   minimal `test_projected_<method>_attention_cases`. Expand from there.

4. If the method supports runner modes beyond eager, add adapter callbacks
   for `cuda_graph_decode_runner.py`, `split_op_runner.py`, etc. The
   adapter pattern is documented inline in each orchestrator file.

## 6. Conventions

### 6.1. References must be independent

The expected path is an **HF-style pure-PyTorch reference** that may
receive random weights *copied* from the SGLang module but must not call
`RadixAttention`, `RadixLinearAttention`, attention-backend wrappers,
Triton/FlashInfer/FLA kernels, or any SGLang helper that encodes backend-
specific math. Backend-against-backend comparisons hide correlated bugs.

### 6.2. RoPE is out of scope

These tests feed post-RoPE-equivalent Q/K tensors. RoPE math belongs in
focused rotary tests, not in every backend × shape case.

### 6.3. ForwardContext must be active before backend metadata init

```python
with forward_context(ForwardContext(attn_backend=backend)):
    backend.init_forward_metadata(forward_batch)
    output = ...  # module forward
```

The fixtures handle this in their `run_*_eager` paths; if you're writing a
new runner-mode helper, mirror the pattern.

### 6.4. `loc_layout` and the `shuffled_pages` default

Every fixture's case runner accepts `loc_layout: str` (default
`"shuffled_pages"`). The four layouts are:

| Layout | What it does | Catches |
|---|---|---|
| `contiguous` | `_token_loc(req_idx, pos) = page_size + req_idx*max_ctx + pos`. Original tidy mapping. | Regression baseline. |
| `shuffled_pages` (**default**) | Per-request page order randomly permuted. | Backends that assume `req_to_token[req_idx, pos]` increases monotonically with `pos`. |
| `interleaved_pages` | Pages from different requests interleaved in physical-slot order. | Backends that assume a request's pages occupy a contiguous physical range. |
| `non_monotonic_extend` | Extend-token slots scattered within each request. | Backends that assume `out_cache_loc[i+1] == out_cache_loc[i] + 1` within an extend. |

The default is `shuffled_pages` (not `contiguous`) because production
allocators produce non-monotonic per-request page assignments after
fragmentation, and `contiguous` silently masks page-table-derivation bugs.
**If you add a new test that fails under the default, suspect a backend
bug before suspecting the test.**

If your backend can't handle a layout, record it in
`LAYOUT_KNOWN_FAILURES` on the test class with a `skipTest` referencing the
production-side cause. See §C.1 in `KNOWN_FAILURES.md`.

### 6.5. Skip cleanly when a backend is unavailable

```python
try:
    backend = ATTENTION_BACKENDS[case.backend](runner)
except (AssertionError, ImportError, ModuleNotFoundError) as exc:
    testcase.skipTest(f"{case.backend} backend is not available: {exc}")
```

Every fixture does this. The pattern lets you run the suite against any
container without crashing on missing optional kernels.

### 6.6. Runner orchestration must be shared across attention methods

If you're adding a runner contract (CG, split-op, spec-runner) that's
reusable across multiple attention methods, put the lifecycle in
`common/runner_modes/` with an **adapter callback** contract. Backend or
attention-method tests should call that helper and pass callbacks, not
reimplement capture/replay or eager-vs-graph comparison locally. The
existing `cuda_graph_decode_runner.py`, `speculative_*_runner.py`, and
`split_op_runner.py` are the canonical examples.

## 7. Known limitations

Read `KNOWN_FAILURES.md` before extending the suite. It catalogs:

- **Container dependency gaps** (re-image to fix): `flash_attn` SM10.x
  wheel missing, tilelang `wait_wgmma` missing.
- **Hardware-architecture gates** (correctly skipped, no action): every
  SM-gated MLA backend, FA3, DSA `fa3`/`trtllm` impls.
- **Backend bugs needing production fixes** (some gated, some no-test):
  - 9 layout-handling bugs across FA3 / FA4 / FlashInfer MLA / FlashMLA /
    dual_chunk under `LAYOUT_KNOWN_FAILURES`.
  - Speculative-mode rejects (Mamba2 tree, MLA non-EAGLE, SWA non-EAGLE,
    KDA non-EAGLE drift, Lightning tree, FA tree drift, DSV4 tree).
  - Graph-runner rejects (FlashMLA `DRAFT_EXTEND` CG buffer-layout
    mismatch, HybridLinearAttn `DRAFT_EXTEND` CG `ValueError`, DSV4
    `compress_ratio≠0` draft-extend production-unreachable).
  - Split-op rejects (Lightning shape, Mamba2 projection, DSV4 indices,
    DSA MHA_ONE_SHOT K-slice).
  - Sparse-kernel bugs (dual_chunk vertical-buffer overflow,
    `cudaErrorIllegalAddress` in `_vertical_slash_sparse_attention`,
    Triton dense `DRAFT_EXTEND` eager mismatch).
  - DSA-specific gaps (tree draft needs parent-indices; HiSparse needs
    coordinator).
- **Production-design constraints** (intentional, not bugs): page-size
  hard-pins per MLA backend, speculative `topk` rejects, KV-cache dtype
  restrictions.

When in doubt: if a test fails and the symptom matches an entry in
`KNOWN_FAILURES.md`, you're hitting a documented production issue, not a
regression.

## 8. Running + debugging

### 8.1. Run the full sweep

```bash
python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py'
```

Expect ~40 s on H200, ~3-4 min on GB300. Test count + skip count should
match `KNOWN_FAILURES.md` "Reference runs" — divergence means a regression.

### 8.2. Run a single backend or method

```bash
python test/manual/attention/unittest/mla/test_triton.py -v
```

Each test file is executable directly and prints per-subtest progress.

### 8.3. Run one test method

```bash
python test/manual/attention/unittest/dense/test_triton.py \
  TestTritonDenseAttentionBackendCorrectness.test_layout_robustness_cases -v
```

### 8.4. Debugging a failure

1. Check `KNOWN_FAILURES.md` first.
2. Re-run with `-v` and `--locals` (or insert `breakpoint()` in the
   relevant `run_*_attention_case`) to see which case in the tuple fails.
3. The reference is in the same fixture file as the actual path —
   walk both side-by-side. The K/V projection weights are the same
   (`_copy_*_weights` enforces this), so divergence is purely in the
   attention math or metadata interpretation.
4. If you suspect a layout issue, re-run with
   `loc_layout="contiguous"` to confirm it's a layout-handling bug, then
   record it in `LAYOUT_KNOWN_FAILURES` if production-side.

### 8.5. CUDA-graph debugging

CG tests capture against a fixed padded batch and replay against distinct
metadata/input tensors. If replay fails but eager passes:
- Suspect the backend's `init_forward_metadata_replay_cuda_graph` — does
  it restore every buffer the kernel reads?
- Suspect shared CG input buffers leaking between independent runner
  instances — the runner mode helpers reset shared buffers between captures.

### 8.6. Hardware gating quick-reference

| If you see... | It means... |
|---|---|
| `skipped 'cutlass_mla requires SM 10.0+'` | You're not on Blackwell B200 |
| `skipped 'trtllm_mla requires SM 12.0a / 12.1a'` | You're not on Blackwell B200 NVL |
| `skipped 'FlashMLA decode/target-verify requires SM90a'` | You're on Blackwell (Hopper-only kernel) |
| `skipped 'dsa fa3 requires SM9.x (Hopper)'` | You're on Blackwell or older than Hopper |

All of these are correct — they fire the documented hardware gates. See
§B in `KNOWN_FAILURES.md`.

## 9. Hardware coverage

| Hardware | Backends covered | Skipped backends |
|---|---|---|
| H200 (SM 9.0) | torch_native, triton, flashinfer, fa3, fa4, flex_attention, flashmla (full), dsa (full incl. fa3 impl), dsv4 + SWA, dual_chunk (with sgl-kernel FA3), GDN/KDA/Lightning/Mamba2 | cutlass_mla, trtllm_mla, tokenspeed_mla, trtllm_mha (prefill) |
| B200 (SM 10.0) | + cutlass_mla, + trtllm_mha (prefill), + dsa trtllm impl. FlashMLA decode/verify becomes gated, container blockers (dual_chunk flash_attn / dsa tilelang) become live | trtllm_mla, tokenspeed_mla |
| GB300 (SM 10.3) | All SM-10.0-exact gates (cutlass_mla, dsa trtllm) skip until binaries land for sm_103; container blockers active for dual_chunk + dsa tilelang | Same as B200 plus the SM-10.0-exact gates |

The suite is designed to **skip cleanly** on any hardware where a backend
isn't supported — a 0-failure sweep on each platform means the gate is
correctly aligned with what production actually runs.

---

## Where to learn more

- **`PLAN.md`** — design history and the full arc of how the suite was
  built. Reads as a journal; skim "Resolved Decisions" for rationale.
- **`KNOWN_FAILURES.md`** — the canonical reference for every backend
  issue, organized by action needed (re-image / hardware gate / production
  fix / production-design constraint).
- **`<method>/README.md`** — per-method coverage matrix and constraints.
  Start here when you're working in a specific attention method.
- **`MUTATION_FIXES.md`** — mutation-testing campaign that hardened
  several backend metadata derivations; mostly historical.
