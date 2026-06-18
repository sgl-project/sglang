---
name: env-var-conventions
description: Conventions for SGLang environment variables — where to define, how to access, how to name, and how to deprecate. Use when adding, renaming, or reviewing any `SGLANG_*` environment variable (or migrating a legacy `SGL_*` alias), or when touching `python/sglang/srt/environ.py`.
---

# Environment Variables — Conventions

Apply this skill when adding, renaming, or reviewing any sglang-owned environment variable (`SGLANG_*`, or a legacy `SGL_*` alias being phased out), or when touching `python/sglang/srt/environ.py`.

## Rule 1 — Define in the `Envs` class in `python/sglang/srt/environ.py`

All sglang-owned env vars live as `EnvField` descriptors on the `Envs` class. Never add a new `os.getenv("SGLANG_...")`, `get_bool_env_var("SGLANG_...")`, or `get_int_env_var("SGLANG_...")` call site — the helpers in `python/sglang/srt/utils/common.py` carry an explicit `FIXME: move your environment variable to sglang.srt.environ` and exist only for pre-existing call sites.

Group the new entry under an existing section comment (e.g. `# Logging Options`, `# Scheduler: recv interval`, `# Flashinfer`). Add a new section comment only when none fits — never drop a new entry at the bottom of an unrelated block.

### Decision table: register in `Envs` or use `os.getenv`?

| Variable | Owner | Goes through `Envs`? |
|---|---|---|
| `SGLANG_*` | sglang | **Always.** The canonical prefix for all new entries. |
| `MOONCAKE_*`, `ASCEND_*`, `DEEP_NORMAL_*`, `IS_H200`, `USE_TRITON_W8A8_FP8_KERNEL`, `HF_HUB_DISABLE_XET`, `DISABLE_OPENAPI_DOC` | Upstream/vendor alias that sglang wants to centralize | **Yes** — register in `Envs` so `.get()` / `.override()` work uniformly. Keep the upstream prefix. |
| `CUDA_*`, `NCCL_*`, `TORCH_*`, `OMP_*`, `HF_HUB_*` (raw upstream) | External tooling | **No.** Read with `os.getenv` — they're set by the launcher / driver, not by sglang. |
| `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, `HOME`, `PATH` | Distributed launcher / OS | **No.** `os.getenv` only. |
| Test runner internals (`PYTEST_CURRENT_TEST`, etc.) | Test framework | **No.** `os.getenv` only. |

`SGL_*` is **not** a parallel valid prefix — it's a deprecated legacy alias. `_convert_SGL_to_SGLANG` rewrites `SGL_*` to `SGLANG_*` at import time with a `DeprecationWarning`. Never define a new `SGL_*` descriptor or `os.getenv("SGL_...")` call site; if you see one in code, it's tech debt to migrate.

The rule of thumb: if the value's lifecycle is owned by sglang code (we read it, we may want to override it in tests, we may want to rename it), put it in `Envs`. If the value is set by something outside sglang and we only consume it as-is, use `os.getenv`.

## Rule 2 — Pick the typed descriptor

| Type | Use for |
|---|---|
| `EnvBool(default)` | boolean flag |
| `EnvInt(default)` | integer |
| `EnvFloat(default)` | float |
| `EnvStr(default)` | string |
| `EnvTuple(())` | comma-separated list, parsed via `s.split(",")` and stripped |

Default value:
- For a knob whose "unset" state must be distinguishable from any concrete value, use `None` as the default (e.g. `EnvStr(None)`, `EnvInt(None)`). The descriptor handles set-to-None correctly via `_set_to_none`.
- For a feature flag, the default encodes the production behavior. `ENABLE_FOO = EnvBool(False)` means foo is off in prod; `DISABLE_FOO = EnvBool(False)` means foo is on in prod. See Rule 4 on picking the verb.

### IntEnum for multi-state knobs

When a knob has more than two discrete states (e.g. an off / soft / strict ladder), define an `IntEnum` next to `Envs` and pass the enum member as the `EnvInt` default. Callers compare against the enum, not raw integers:

```python
class ToolStrictLevel(IntEnum):
    OFF = 0
    FUNCTION = 1
    PARAMETER = 2

SGLANG_TOOL_STRICT_LEVEL = EnvInt(ToolStrictLevel.OFF)

# At the call site:
if envs.SGLANG_TOOL_STRICT_LEVEL.get() >= ToolStrictLevel.PARAMETER:
    ...
```

Don't pile `SGLANG_ENABLE_FOO_STRICT` / `SGLANG_ENABLE_FOO_OFF` boolean knobs that fight each other — one ordered integer is cleaner.

## Rule 3 — Access via the `EnvField` API, never raw `os.environ`

```python
from sglang.srt.environ import envs

if envs.SGLANG_FOO.get():
    ...
```

`envs.SGLANG_FOO` is the descriptor itself; its `__bool__` and `__len__` raise on purpose so that `if envs.SGLANG_FOO:` fails loudly instead of silently reading as truthy. The `.get()` is mandatory at every read site.

### Full API surface

| Method | Use |
|---|---|
| `.get()` | Read value (parsed). Returns `default` if unset, or `None` if explicitly set to None. |
| `.set(value)` | Set value. `set(None)` flips the internal `_set_to_none` flag so the next `.get()` returns `None`, not `default`. |
| `.clear()` | Unset entirely. Next `.get()` returns `default`. |
| `.is_set()` | True iff the key is present in `os.environ` (regardless of value, including the explicit-None case). |
| `.override(value)` | Context manager — set on enter, restore exactly what was there on exit. Use this in tests. |

The `_set_to_none` distinction matters: `clear()` → `is_set()=False, get()=default`; `set(None)` → `is_set()=True, get()=None`. A few descriptors (e.g. `SGLANG_TEST_MAX_RETRY = EnvInt(None)`, `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE = EnvInt(None)`) rely on this to distinguish "user opted out" from "default applies".

### Test overrides

```python
with envs.SGLANG_TEST_RETRACT.override(True):
    ...
```

Don't mutate `os.environ` directly in tests — `override` restores the original cleanly, including the explicit-None state, even if the block raises.

For multiple overrides composed dynamically, use `ExitStack`:

```python
with ExitStack() as stack:
    for name, value in test_envs.items():
        stack.enter_context(getattr(envs, name).override(value))
    run_test()
```

### Subprocess inheritance

`override` mutates the real `os.environ`, so child processes spawned **inside** the `with` block inherit the override. This is the supported way to seed a `subprocess.Popen`:

```python
with envs.SGLANG_TEST_RETRACT.override(True):
    subprocess.Popen([...]).wait()
```

A child started outside the `with` block sees the original value.

### `temp_set_env` is for non-sglang keys only

The module-level `temp_set_env(**env_vars)` helper exists for overriding **non-sglang** env vars (e.g. `CUDA_LAUNCH_BLOCKING`, `NCCL_DEBUG`) in tests. It explicitly rejects `SGLANG_*` / `SGL_*` keys:

```python
# Wrong — raises ValueError
with temp_set_env(SGLANG_TEST_RETRACT="true"): ...

# Right — sglang keys go through the descriptor
with envs.SGLANG_TEST_RETRACT.override(True): ...

# Right — non-sglang keys go through temp_set_env
with temp_set_env(CUDA_LAUNCH_BLOCKING="1"): ...
```

The `allow_sglang=True` escape hatch exists for the rare case where you must bypass `Envs` (e.g. setting an env var **name** that's only constructed at runtime); don't use it just to skip writing a descriptor.

## Rule 4 — Naming: `SGLANG_` prefix + verb category

Prefix is always `SGLANG_` for new entries. `SGL_*` is auto-translated to `SGLANG_*` with a `DeprecationWarning` in `_convert_SGL_to_SGLANG`; never add a new `SGL_*` key.

The second token signals intent. Pick the right verb up front — renames require an alias entry (Rule 5).

| Verb | Meaning | Example |
|---|---|---|
| `ENABLE_FOO` | Knob that turns feature foo on/off. Default in the `EnvBool` encodes prod behavior. | `SGLANG_ENABLE_TORCH_COMPILE`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` |
| `DISABLE_FOO` | Kill-switch. `DISABLE_FOO=True` turns foo off. | `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP` |
| `USE_FOO` | Selects which implementation / backend | `SGLANG_USE_AITER`, `SGLANG_USE_DEEPGEMM_BMM` |
| `FORCE_FOO` | Overrides autodetection | `SGLANG_FORCE_FP8_MARLIN`, `SGLANG_FORCE_STREAM_INTERVAL` |
| `LOG_FOO` | Logging-only knob | `SGLANG_LOG_GC`, `SGLANG_LOG_MS` |
| `TEST_FOO` | Test-only hook | `SGLANG_TEST_RETRACT`, `SGLANG_TEST_MAX_RETRY` |
| `DEBUG_FOO` | Debug-only instrumentation | `SGLANG_DEBUG_MEMORY_POOL`, `SGLANG_DEBUG_SYMM_MEM` |
| `OPT_FOO` | Perf-optimization toggle (heavily used by DSV4 work) | `SGLANG_OPT_USE_FUSED_HASH_TOPK`, `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2` |

Picking between `ENABLE_FOO` and `DISABLE_FOO`: both verbs are valid. The only forbidden combination is `DISABLE_FOO = EnvBool(True)`, because it produces a true double-negative at the call site (`if not envs.SGLANG_DISABLE_FOO.get():` reads as "if not disabled"). All other combinations are fine:

| Pattern | Call site | Verdict |
|---|---|---|
| `ENABLE_FOO = EnvBool(False)` | `if envs.SGLANG_ENABLE_FOO.get():` | OK — opt-in feature |
| `ENABLE_FOO = EnvBool(True)`  | `if envs.SGLANG_ENABLE_FOO.get():` | OK — on in prod, user opts out via `False` |
| `DISABLE_FOO = EnvBool(False)` | `if not envs.SGLANG_DISABLE_FOO.get():` | OK — single negation, reads as "if enabled" |
| `DISABLE_FOO = EnvBool(True)`  | `if not envs.SGLANG_DISABLE_FOO.get():` | **Forbidden** — true double-negative |

`SGLANG_*` is the canonical sglang prefix. Vendor-integration keys (`MOONCAKE_*`, `ASCEND_*`, `DEEP_NORMAL_*`, `IS_H200`) keep their upstream prefix and live in the same `Envs` class — these are integration aliases, not sglang-owned feature flags.

## Rule 5 — Renames go through `*WithAlias` or `_print_deprecated_env`

For a rename where the old key must keep working with a warning:

```python
SGLANG_DSA_FUSE_TOPK = EnvBoolWithAlias(True, deprecated_name="SGLANG_NSA_FUSE_TOPK")
```

Use `EnvBoolWithAlias` / `EnvIntWithAlias`. The fallback emits a `DeprecationWarning` and copies the old value over.

For a full removal where the env var is going away, add to `_convert_SGL_to_SGLANG`:

```python
_print_deprecated_env("SGLANG_OLD_NAME", "SGLANG_NEW_NAME")  # mapped to a replacement
_print_deprecated_env("SGLANG_OLD_NAME")                     # no replacement, gone
```

For env-var to CLI-flag migration, add at module top-level:

```python
_warn_deprecated_env_to_cli_flag(
    "SGLANG_FOO",
    "Please use '--foo' instead.",
)
```

Don't silently flip a default during a rename. If the new default disagrees with the old one, that's a behavior change — call it out in the PR body separately from the rename.

## Rule 6 — Env var vs CLI flag

| If the knob is… | Goes in |
|---|---|
| User-facing (documented, expected to flip per deployment) | `server_args.py` CLI flag |
| Expert toggle, A-B kill-switch, vendor integration | `environ.py` env var |
| Test / debug hook | `environ.py` env var with `TEST_` / `DEBUG_` prefix |
| Temporary env var rolling out to a CLI flag | env var first, then migrate via `_warn_deprecated_env_to_cli_flag` |

Don't add a CLI flag that just forwards to an env var, and don't add an env var that duplicates an existing CLI flag. Pick one surface.

## Out of scope

- **External / vendor env vars consumed raw** (`HF_HUB_*`, `CUDA_*`, `NCCL_*`, `TORCH_*`, `OMP_*`, `RANK`, `MASTER_ADDR`, etc.): see the decision table in Rule 1 — `os.getenv` is correct, don't pull them into `Envs`.
- **Pre-existing `get_bool_env_var(...)` / `get_int_env_var(...)` call sites**: leave them as is; new code shouldn't add more, but mass-migration is out of scope for a feature PR.
- **Upstream-aliased keys already in `Envs`** (`MOONCAKE_*`, `ASCEND_*`, `DEEP_NORMAL_*`, `IS_H200`, `USE_TRITON_W8A8_FP8_KERNEL`, `HF_HUB_DISABLE_XET`, `DISABLE_OPENAPI_DOC` — see Rule 1 decision table): the `SGLANG_` prefix rules in Rule 4 don't apply — the upstream prefix is the canonical name.
