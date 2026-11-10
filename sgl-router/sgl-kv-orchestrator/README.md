# KV orchestration models

This crate defines the Router's versioned PIN, PREFETCH, and DEMOTE protocol,
session and branch associations, immutable target resolution, and independent
interests in shared work. The Router actor owns network I/O and worker
selection; Engine/HiCache owns the physical operations.

- `models` and `protocol` validate the action and prefix descriptors. Layout,
  namespace, page size, and token material must match the executing Engine.
- `session_index` and `target_resolver` track logical request and branch
  identities. A resolved scope does not assert residency.
- `operations` tracks the interests sharing a physical job. Releasing one
  interest does not cancel work that another live interest still needs.
- `observations` keeps readiness evidence scoped to the worker and its epoch.
  A stale or incomplete observation must not be used as a ready prefix.

See [Router configuration and lifecycle examples](../KV_ORCHESTRATION.md).
