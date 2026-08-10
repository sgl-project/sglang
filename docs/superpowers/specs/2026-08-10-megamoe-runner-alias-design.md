# MegaMoE Runner Alias Design

## Context

The DeepSeek V4 Pro GB300 nightly uses MegaMoE through
`--moe-a2a-backend megamoe`. MegaMoE is an all-to-all execution mode rather
than a standalone MoE kernel runner, but operators also expect to be able to
select it through `--moe-runner-backend`.

The nightly's high-throughput variant already sets
`SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320`. The environment
descriptor still defaults to 1024, which is too small for common DeepSeek V4
prefill budgets when a process does not inherit the test-specific override.

## Goals

- Accept `--moe-runner-backend megamoe` as an operator-facing compatibility
  alias for `--moe-a2a-backend megamoe`.
- Keep the internal MoE runner selection on its existing automatic path.
- Raise the default MegaMoE per-rank token capacity from 1024 to 8192.
- Exercise the compatibility alias in the GB300 nightly while keeping its
  explicit 8320 capacity override.

## Non-goals

- Introduce a new MegaMoE kernel runner implementation.
- Change the meaning of existing non-MegaMoE runner backends.
- Add the alias to `--speculative-moe-runner-backend`.
- Change MegaMoE buffer allocation or fallback behavior beyond the requested
  default capacity increase.

## Design

### CLI choices

Expose `megamoe` only in the choices for the target model's
`--moe-runner-backend`. Keep the underlying list of real runner backends
separate so `--speculative-moe-runner-backend` does not accidentally accept an
alias that is not normalized for the speculative model.

Out-of-tree extensions added through `add_moe_runner_backend_choices` must
remain visible to both the real-runner list and the target-model CLI list.

### Input normalization

Normalize the alias at the beginning of `ServerArgs.__post_init__`, before the
dummy-model early return and before model-specific overrides inspect the
backend fields:

```text
moe_runner_backend=megamoe
    -> moe_runner_backend=auto
    -> moe_a2a_backend=megamoe
```

This makes the alias work for both CLI construction and direct `ServerArgs`
construction. Resetting the runner to `auto` gives the same downstream state as
the canonical `--moe-a2a-backend megamoe` invocation and prevents `megamoe`
from reaching kernel-runner compatibility checks.

If the user also supplies a conflicting A2A backend, the alias wins and a
warning explains that the A2A backend was changed to `megamoe`. This follows
the requested rule that detecting the runner alias must set the A2A backend.

### Environment default

Change
`SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` from `EnvInt(1024)` to
`EnvInt(8192)`. Explicit environment values continue to take precedence.

This is an intentional behavior and memory-footprint change for MegaMoE users
who do not set the variable explicitly.

### Nightly coverage

In `test/registered/gb300/test_deepseek_v4_pro_fp4.py`:

- keep the high-throughput environment override at 8320;
- select MegaMoE through `--moe-runner-backend megamoe` instead of the
  canonical A2A flag.

The nightly therefore covers the new alias on the real DeepSeek V4 Pro launch
path while retaining the extra token headroom requested for this test.

## Tests

Add CPU unit coverage before implementation for:

- `megamoe` being accepted by `--moe-runner-backend`;
- normalization to runner `auto` plus A2A `megamoe`;
- the alias overriding a conflicting A2A backend;
- `megamoe` remaining invalid for `--speculative-moe-runner-backend`;
- the environment descriptor returning 8192 when the variable is unset and
  still honoring an explicit override.

Run the focused CPU tests, formatting/lint checks for changed files, and a
static inspection of the GB300 test configuration. The GPU nightly itself is
not expected to run locally.

## Compatibility and Rollback

Existing invocations are unchanged. The new spelling is additive, and the
canonical A2A spelling remains supported. Reverting the alias normalization and
CLI choice removes the compatibility spelling; reverting the `EnvInt` change
restores the previous buffer default independently.
