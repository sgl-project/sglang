# PD Flip Prefill Manager Reuse Design

## Problem

With runtime role switching enabled, every worker already owns both decode and
prefill disaggregation queues.  A decode-to-prefill migration currently creates
a second prefill KV manager for the migration source.  Both prefill managers
register the same bootstrap HTTP address, so the migration manager overwrites
the normal prefill manager's advertised ZMQ rank port.  Migration traffic works,
but requests admitted after the role flip use the original manager and can no
longer complete the bootstrap handshake.

## Required behavior

- If `disagg_prefill_bootstrap_queue.kv_manager` exists, PD migration source
  transfers must reuse that exact manager.
- The manager is cached in `pd_flip_source_kv_manager` for existing callers.
- A dedicated prefill manager is created only for legacy/non-dual-queue decode
  workers that have no initialized prefill manager.
- No migration, rollover, or role-switch safety condition is relaxed.

## Acceptance

- A unit test proves the existing prefill manager is returned without invoking
  the manager constructor.
- A fallback unit test proves manager construction remains available when no
  prefill queue exists.
- The four-node trace completes 40/40 with node2 in prefill role and no
  post-flip bootstrap timeout or worker crash.
