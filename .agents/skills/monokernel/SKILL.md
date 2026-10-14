---
name: monokernel
description: Work on SGLang's monokernels (RFC #43949), persistent kernels that run most of a decoder layer in one or two launches, kept as a plugin in python/sglang/kernels/ops/mono. Use when fixing, tuning or validating a monokernel, adding a model's spec, or repairing the plugin after an SGLang change broke it.
---

# Monokernels

[RFC #43949](https://github.com/sgl-project/sglang/issues/43949) defines one
framework for monokernels. They are written and maintained mostly by agents, so
the design keeps them in one box: everything lives in
`python/sglang/kernels/ops/mono/`, and SGLang only calls the plugin's hooks at a
few fixed points. `kernels/README.md` states this as the one exception to "runtime
integration stays in `srt`". A monokernel change touches `srt/` only where it
must: a new hook call, or a cache feature a kernel needs (for example a new state
dtype), in its own PR.

## Layout

```
python/sglang/kernels/ops/mono/
  __init__.py              re-exports the calls SGLang makes: handle_monokernel,
                           load_tensor, attach, route, kernel
  guards.py                before any process starts: handle_monokernel refuses
                           an unsupported setup with every reason and sets the
                           decode capture sizes; every wait's deadline
  runtime.py               in each process: load_tensor, attach (refuse with the
                           loaded weights, build, compare the ranks' plans),
                           route, kernel
  spec.py                  what a model declares (MonoSpec, Span, Feature,
                           Numerics) and the registry, by HF architecture
  common_hip/              the shared device runtime: resident grid, mailboxes,
                           guarded waits (guard.py), peer memory (peer.py)
  <model>_<vendor>/    one model: its kernels, its host runner, weight
                           layout helpers, and spec.py (the spec and its glue)
```

## The socket in srt

These are the only SGLang lines a monokernel depends on. Keep them unchanged
while you fix, tune or extend the plugin:

- `arg_groups/fields/exec_.py`: the `enable_monokernel` flag.
- `arg_groups/pipeline.py` and `resolution_hooks.py`: the `handle_monokernel` call
  and its whitelist entry.
- Each covered model, through `from sglang.kernels.ops import mono`:
  `mono.load_tensor()` in `load_weights()`, `mono.attach()` in
  `precompile_kernels_after_loading()`, one `mono.route()` a forward, and
  `mono.kernel(plan, layer_id, point)` where a span starts. For DeepSeek-V4.1:
  `models/deepseek_v4.py`, `models/deepseek_common/amd/deepseek_v4_hip.py` and
  `deepseek_v4_fused_mhc.py`.

Covering a new model adds that model's seams. A span that starts at a new point
adds one seam. A cache feature a kernel needs goes to `srt/` in its own PR.
Nothing else in `srt/` changes.

## Invariants

- A span takes and returns exactly what the code it replaces takes and returns,
  and writes KV and state as that code does, so covered and normal layers can
  alternate in one step.
- Refuse, never fall back silently. Configuration reasons come from the spec's
  `config_disable_reasons` before any process starts, and model reasons from
  `model_disable_reasons` at attach. Every rank refuses together.
- Every wait on another workgroup or GPU runs under the guard
  (`common_hip/guard.py`), with a deadline below `--watchdog-timeout`. A failed
  wait flags every peer and stops its launch with a fault at address 0x0, and a
  wait that sees a flag stops too. Never let a launch run on after a failed wait.
  There is no host-side check: a rank finishes a step only with every peer's data.
- Kernels on other streams (the overlap scheduler's, a model's side streams) only
  delay a launch, since they finish on their own. A kernel that waits on another
  GPU, or another process on the same GPU, can keep a workgroup from starting
  until the deadline stops the server: never put one beside a monokernel.
- Never quantize more than the normal path. Bound each launch output in the spec's
  `NUMERICS`; the kernel checks read the bounds from there.
- Glue runs while a CUDA graph is captured, not per step: no host syncs, and
  nothing that only an eager forward does.
- Every SGLang name a plugin module reads goes in that module's `SGLANG_READS`:
  definitions, attributes, parameters as `f(p)`, string literals as `"x"`.
  `test/registered/unit/mono/test_mono_contract.py` pins them, and fails when a
  module imports from `sglang.srt` a name it has not declared.

## Common tasks

**Repair after an SGLang change.** `test_mono_contract.py` names the file and
the missing names. Find what SGLang renamed or moved, update the glue and its
`SGLANG_READS`, and validate. Change the seams in `srt/` only if the replaced code
changed what it takes or returns.

**Fix or tune a kernel.** Change the kernel and its runner in
`<model>_<vendor>/`. Run the kernel checks, then the full validation. Update
`test/manual/kernels/mono/baselines.json` only with a measured improvement.

**Add a model.** Add `<model>_<vendor>/` with its kernels, runner and
`spec.py`: spans, `NUMERICS`, refusal reasons, `build`, and `SGLANG_READS`. Add
one line to `_REGISTRY` in `spec.py`. Add the model's seams in `srt/`, and the
model's checks under `test/registered/kernels/ops/mono/` and `e2e/mono/`.
Reuse `common_hip/` for every wait.

## Validate

```bash
python test/manual/kernels/mono/validate.py
```

It runs the CPU tests, the kernel checks, the serving test (GSM8K and KL
consistency across the switch points, and a fault on one rank that must stop
every rank without sending a token), and a paired decode A/B with and without
`--enable-monokernel` at 1, 4, 8 and 16 concurrent requests. Each step prints
PASS or FAIL, and decode speed is compared with `baselines.json`. Use `--steps`
for a subset. CPU-only work needs `--steps unit`.

The scripts serve `DEEPSEEK_V41_FLASH_MODEL_PATH`, by default the Hugging Face id,
which downloads the checkpoint: point it at a local copy.

Follow `kernel-organization` for placement and `write-sglang-test` for tests.
