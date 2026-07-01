---
name: large-class-style
description: 'Code style for SGLang large classes `Scheduler`, `TokenizerManager`, and `ModelRunner`: frozen-code conventions and `__init__` orchestration style. Use when modifying any of these three classes or reviewing changes to them.'
---

# Code Style for Scheduler / TokenizerManager / ModelRunner

Conventions for SGLang's three large classes:

- `Scheduler` — `python/sglang/srt/managers/scheduler.py`
- `TokenizerManager` — `python/sglang/srt/managers/tokenizer_manager.py`
- `ModelRunner` — `python/sglang/srt/model_executor/model_runner.py`

## 1. Frozen Code

Some core files are **frozen**: they are *orchestration-only* — a thin composition root that constructs collaborators, wires them, delegates to them, and coordinates the calls. They must stay that way. **Domain logic does not belong in a frozen file**; it lives in a collaborator class in its own module.

### Why

- The file is a thin orchestrator over collaborator classes; freezing keeps it that way and stops it growing back into a god class.
- Keeping domain logic in collaborators (their own files) is what makes per-file code ownership, single responsibility, and unit testing possible.
- The orchestrator is the composition root: it may know about every collaborator, because wiring and sequencing them is its job. Coordination stays here — domain logic does not.

### Frozen files

- `python/sglang/srt/model_executor/model_runner.py`

### Allowed: orchestration

Every statement refers to a collaborator and is one of:

1. **Construct** — a short `init_<thing>` helper whose body is essentially a single construction (follows §2); use `maybe_init_<thing>` with a one-line gate when conditional.
2. **Wire** — a short call that runs the helper from the orchestrator (e.g. in `__init__`).
3. **Delegate** — calls to a collaborator's methods at the necessary call sites (`self.foo.run(...)`).
4. **Coordinate** — the minimal control flow that *selects or orders* the above: an `if` choosing whether / which collaborator to wire or call, the order of calls, threading one call's result into the next.

Heuristic: a statement is allowed only if it constructs, wires, delegates, or selects/orders those — never if it *computes or transforms* a value beyond passing arguments and results through.

```python
# model_runner.py — orchestration only.
def init_foo(self):                        # construct
    self.foo = FooManager(server_args=self.server_args, device=self.device)

self.init_foo()                            # wire (in __init__)

if self.server_args.enable_bar:            # coordinate: select
    self.bar.prepare(forward_batch)        # delegate
out = self.foo.run(forward_batch)          # delegate
self.baz.consume(out)                      # coordinate: thread result into next delegate
```

### Not allowed: domain logic

Config building, data transformation, algorithm bodies, math, post-processing — any branch or loop that *computes* rather than *coordinates*. It belongs in the collaborator.

```python
# NOT allowed in a frozen file: domain logic inlined.
self.foo = None
if self.server_args.enable_foo:
    config = build_foo_config(self.model_config, self.device)   # config logic in frozen file
    self.foo = FooManager(config)                               # inline construction, not via (maybe_)init_foo
    out = [step(x) for x in batch]                              # computation, not coordination
```

Move that body into `FooManager` (its `__init__` or a factory) plus a `(maybe_)init_foo` helper.

### Where coordination logic goes

1. **Default: extract.** Pull cohesive coordination into a low-coupling collaborator (an initializer, a forward pipeline) and delegate to it.
2. **Residue stays.** Coordination that can't be cohesively extracted may remain — but only the minimal **Coordinate** form above, kept pseudocode-readable. This is the explicit exception, not a fallback; note why it stays.

When the residue outgrows pseudocode, that is the signal to extract a dedicated coordinator — not to keep inlining.

### Pass what the collaborator needs, not the god object

When you extract domain logic into a collaborator (a factory, an initializer, a pipeline) in its own module, give it the **specific values** it needs — `model_config`, `device`, the sizes — not the whole frozen object (`ModelRunner`, `Scheduler`). Passing the god object back into the collaborator re-creates the coupling the split was meant to remove: the module still reads dozens of attributes off it, can't be unit-tested without building the whole class, and every field rename ripples back in.

- Default to **narrow, keyword args**. Reference shape: `layer_setup.resolve_layer_indices(*, model, model_config, is_draft_worker, spec_algorithm)`.
- Return a small **frozen struct** and let the orchestrator assign it onto its own fields. The collaborator should not reach back in and mutate the god object.
- If a leaf genuinely needs the live object — its constructor contract already takes the runner, or it reads state that mutates after init — confine that dependency to the **smallest leaf** and pass narrow args everywhere above it. Note why it can't be narrowed.

### If you do pass the god object, keep it read-only

When a callee genuinely takes the live object, it should **read** fields off it and **return** results — and avoid **writing** fields back into it unless there is genuinely no other way. Let the orchestrator own the assignment onto its own fields. A callee that mutates the god object scatters that object's writes across other modules: you can no longer see what `ModelRunner` owns by reading `model_runner.py`, the hidden writes race with the orchestrator's own ordering, and the callee silently depends on being invoked at exactly the right moment.

```python
# Good — callee reads the runner and returns; the orchestrator owns the writes.
# model_runner.py
class ModelRunner:
    def bar(self):
        self.a, self.b, self.c = foo(self)

# another_file.py
def foo(model_runner):
    return xxx, yy, zz

# Avoid — callee reaches back in and writes the runner's fields.
# model_runner.py
class ModelRunner:
    def bar(self):
        foo(self)

# another_file.py
def foo(model_runner):
    model_runner.a = xx
    model_runner.b = yy
    model_runner.c = zz
```

## 2. `__init__` style

Apply when modifying the `__init__` of the three classes above.

### Why

- Downstream forks override one piece (tokenizer, KV cache, IPC, …).
- Inline logic forces them to copy the whole `__init__`, which rots against upstream.
- Splitting into `init_*` helpers lets them override exactly what they need.
- Reference shape: `TokenizerManager.__init__` in `python/sglang/srt/managers/tokenizer_manager.py`.

### Rules

- **`__init__` is an orchestrator.** Sequence of `self.init_*(...)` calls + minimal glue. No non-trivial construction inlined.
- **One helper per overridable unit.** Each `init_*` = one concern a subclass might swap. Don't lump.
- **Naming:** `init_<thing>` (snake_case, names the component). Conditional construction → `maybe_init_<thing>`, gate inside the helper.
- **No silent state coupling.** A helper only reads `self.*` set by earlier helpers. Ordering lives in `__init__`. Shared intermediates → pass as args, not via `self.*`.
- **New logic = new helper.** Default to adding `init_<thing>`, not another inline block. One-line `self.foo = server_args.foo` is fine; structured logic is not.
- **Preserve override points.** Prefer additive changes to existing `init_*` signatures. Breaking changes → call out in PR.

### Scope

Only the three classes listed above. Not other manager-style classes, not small dataclass/utility constructors.
