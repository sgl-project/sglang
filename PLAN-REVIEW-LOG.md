# Programmable KV implementation and review log

## Scope and reference contract

Implement the accepted SGLang programmable KV RFC on branch `rfc-kvc`, starting
at `e4a171d4a0`. The user authorized local commits after each complete, verified
feature. GPU validation is explicitly deferred; this run performs CPU validation.
Do not modify the sibling reference checkout or terminate existing GPU processes.
No commit includes Codex as a co-author.

The implementation follows the SGLang RFC and its confirmed differences:
no public Pin, explicit cache priority above the session heuristic in the opt-in
profile, and copy-scoped claims with explicit renewal. Native and opaque actions
have separate interpretation owners. Policies propose; existing cache/pool owners
approve resources and execute.

References reviewed include the SGLang and vLLM RFCs, the current reference
`vllm/v1/kv_hints` implementation and README, the October 18/19 development reports,
the October 20 acceptance report, and
`../vllm-rfc/docs/programmable-kvc/experiment-reports/2026-10-20-000635-snapshot/technical-report.md`.
The accepted implementation lessons are whole-prefix preparation holds,
per-admission capacity/match revalidation, independent shared-operation lifetimes,
ACK/replay preserving failed outcomes, low-cost unconsumed signals, and movement
gated by pressure, pause/transfer cost and actual admission opportunities.

Implementation units cover control/logical lifecycle, serving/cache priority,
real movement adapters, composable policies and scheduling, then integration and
CPU acceptance. Unsupported component/topology combinations must be explicit.
This is an implementation sequence, not a claim that all units are already done.

## Act 3 — Build

### Unit 1, initial build

The delegated builder produced a native control package and a registered CPU
behavior suite. No serving path or real cache execution is claimed by this unit.
The parent independently read all new files and ran:

```text
PYTHONPATH=python CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest test/registered/unit/managers/test_kv_control.py -q
Running 69 items in this shard
.....................................................................    [100%]
69 passed in 8.51s
```

Review did not approve the initial build for commit. Two additional reproductions
identified lifecycle gaps:

- Native task finish left a session-addressed attached operation queued. Its
  result remained `QUEUED` and active count remained one, allowing later dispatch
  for a finished task. Cleanup must abandon only the removed task incarnations'
  interests and preserve independent users of shared content.
- An executor submission exception produced `CANCELLING / executor_error`, but
  the drain response changed it to `CANCELLED` with an empty reason. The eventual
  result and ACK must retain `FAILED` and the original cause, while still waiting
  for resource quiescence.

The builder was asked to add failing-before regressions and fix these two cases.
The active delegated session is `01a159fe-6599-7621-9b0c-c83bb11a2cce`; one of the
two delegated fix rounds for this unit has been requested. Parent review and
proof remain required before commit.

### Unit 1, review round 1 and parent verdict

The regressions reproduced before the fix (`11 failed, 67 deselected`) and passed
afterward (`11 passed, 67 deselected`). Native task finish now cancels interests
matched by frozen task incarnation, including same-envelope prepared actions;
external tasks and raw logical-index snapshots remain independent. Executor
exceptions retain their first cause and observed progress during draining, then
become `FAILED` after quiescence. ACK/replay preserve that failure.

The parent read the complete correction diff and independently reran the full
unit proof:

```text
Running 78 items in this shard
........................................................................ [ 92%]
......                                                                   [100%]
78 passed in 7.88s
```

Parent isort, Ruff syntax/name checks, formatting and diff-whitespace checks
passed. Imports are checked with the repository's separate isort configuration;
Ruff's additional unconfigured `I` rule classified the source differently and is
not substituted for that repository configuration. The builder also verified
CPU CI registration and the pytest entrypoint. One delegated fix round was used.
Verdict: approve this control/logical-lifecycle feature for its local commit.
Serving, real priority/eviction, movement and policies remain subsequent features.

### CPU environment

The repository has an isolated ignored `.venv`. It reads existing reference
site-packages through `reference-runtime.pth`; additional packages are installed
only into this repository's environment. The reference environment is unchanged.
Initial tools: Python 3.12.7, pytest 9.1.1, msgspec 0.22.0, ruff 0.17.1,
isort 8.0.1. The metadata-only unit needs no model/kernel imports or import stubs.
Full runtime CPU integration will have separate dependency and test records.

For upcoming real-cache tests, `gguf==0.19.0` was installed into the local
environment with `--no-deps`. The existing session UnifiedRadixCache suite passed
on actual CPU pools: `4 passed, 16 warnings in 9.51s`. Its test-utils port default
cannot parse an empty `CUDA_VISIBLE_DEVICES`; the CPU run used the nonexistent
device `999` and first asserted `torch.cuda.device_count() == 0`. Production and
test-utils source were not changed for this environment issue.

### Unit 2a — Real cache priorities, parent takeover

The builder added optional copy-scoped side metadata, real UnifiedRadixCache
execution/inspection, FULL/SWA/Mamba geometry and lifecycle hooks, and explicit
Rust-to-Python selection. The delegated run stopped on its usage limit before
producing a completion report. The parent took over without retrying delegation,
read all changes and independently ran the initial proof: 100 passed, 16 warnings,
9 subtests passed in 13.68s.

Parent review reproduced quadratic candidate selection: reclaiming 32 leaves
computed 528 ordering keys. The correction uses one candidate heap per walk,
updates exposed parents and rebuilds at claim-edit or TTL boundaries. Regressions
cover key-count growth on real FULL/SWA/Mamba pools, expiration and claim edits
during a walk, and the explicitly selected profile's session migration. The
profile uses LRU without claims, as specified; disabled caches retain legacy
session ordering. No claim/scoring metadata is added to ordinary node records.

Final parent proof ran each file in its own process with accelerators hidden and
the Python TreeCore selected: control 78 passed; real priorities 21 passed with
18 subtests; session 4 passed; joint allocation/eviction 15 passed with 45
subtests; lock refs 1 passed; Mamba path cap 5 passed, 5 GPU cases skipped, 2
subtests; SWA locked recovery 8 passed. Total: 132 passed, 5 skipped. The original
Rust-toolchain failure remains a limitation, not a successful Rust test.
Repository isort, Ruff formatting/syntax/name checks and git diff whitespace
checks pass. The key-count regression permits at most 64 evaluations for 32
victims in all three component paths, with and without a live claim.
Verdict: approve the real-cache priority feature for a local commit.
Serving transport, request callbacks, transfers and policies are still pending.
