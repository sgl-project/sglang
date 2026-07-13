# PD Flip Rollover Blocker Diagnostics Design

## Purpose

Identify the exact internal predicate that prevents node2 from rolling a completed first migration session into the second session. The diagnostic must also distinguish this failure from the valid case where no requests remain after the observation window.

## Constraints

- Do not relax or reorder migration safety checks.
- Do not clear or archive a session automatically.
- Do not expose request prompts, generated text, credentials, or KV contents.
- Keep the existing boolean rollover API compatible.
- Emit bounded, structured data suitable for controller logs and status sampling.

## Design

Add a pure scheduler helper:

`_pd_flip_rollover_blockers(session, next_role) -> List[Dict[str, Any]]`

Each returned item contains a stable `code`, plus `rid` when the failure belongs to an entry. Session-level codes cover role, pending count, failed count, and terminal state. Source-entry codes cover base metadata release, initial transfer, final owner, delta transfer, and delta metadata release. Target-entry codes mirror the current target rollover checks.

`_pd_flip_can_rollover_session` becomes a compatibility wrapper that returns whether the blocker list is empty. Rejected source/target starts include the compact blocker list in the response message. Migration status includes the same `rollover_blockers` array for the session's current role.

## Data flow

1. The controller calls second `source/start`.
2. The worker evaluates the existing session with the pure helper.
3. If blockers exist, behavior remains a rejection.
4. The response and status snapshot record exact blocker codes and RIDs.
5. The experiment runner already archives controller and worker logs, so no runner format change is required.

## Testing

- A source-released session with a missing initial transfer flag reports `source_entry_not_transferred`.
- A non-noop delta missing metadata release reports `source_delta_metadata_not_freed`.
- A safe source-released session reports no blockers and remains rollover-capable.
- A conflicting second start contains the structured blocker in its message/status.
- A zero-remaining-request controller path is checked separately from worker rollover diagnostics; no-request completion must not be labeled as a session conflict.

## Success criteria

The next 40-request run either completes the second session or produces an exact blocker code and RID. The output must answer whether the remaining request existed at second-start time. No migration behavior is changed by this diagnostic patch.
