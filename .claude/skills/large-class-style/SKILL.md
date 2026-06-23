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

Some core files are **frozen** while they are being decomposed: their internal logic must not grow. A frozen file may be touched only to construct, wire, and delegate to a collaborator — never to add logic.

### Why

- The file is a god class being decomposed; freezing stops it accumulating more logic while that work is in flight.
- Forcing new logic into collaborator classes (their own files) is what makes per-file code ownership, single responsibility, and unit testing possible.
- Without a freeze, features land in the god class faster than it can be split, so it never shrinks.

### Frozen files

- `python/sglang/srt/model_executor/model_runner.py`

### Allowed edits

Put any new logic in a **collaborator class in its own module**. The frozen file then references it in only these three ways:

1. **Construct** — a short `init_<thing>` helper whose body is essentially a single construction (follows §2); use `maybe_init_<thing>` with a one-line gate when construction is conditional.
2. **Wire** — a short call that runs the helper from the orchestrator (e.g. in `__init__`).
3. **Delegate** — short calls to the collaborator's methods at the necessary call sites (`self.foo.bar(...)`), with no new surrounding logic.

```python
# In model_runner.py — the only edits allowed: construct, wire, delegate.
def init_foo(self):                                              # construct
    self.foo = FooManager(server_args=self.server_args, device=self.device)

self.init_foo()                                                  # wire (in __init__)

self.foo.bar(forward_batch)                                      # delegate (at the call site)
```

### Not allowed

Anything beyond construct / wire / delegate — config building, branching, loops, warmup orchestration, post-processing. That logic lives in the collaborator, not the frozen file.

```python
# NOT allowed in a frozen file: construction / logic inlined instead of delegated.
self.foo = None
if self.server_args.enable_foo:
    config = build_foo_config(self.model_config, self.device)   # config logic in frozen file
    self.foo = FooManager(config)                               # inline construction, not via (maybe_)init_foo
```

Move that body into `FooManager` (its `__init__` or a factory) plus a `(maybe_)init_foo` helper; the frozen file keeps only construct + wire + delegate.

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
