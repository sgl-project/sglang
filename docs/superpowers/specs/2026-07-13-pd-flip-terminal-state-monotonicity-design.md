# PD Flip Source Terminal-State Monotonicity Design

## Problem

After `finish_pd_flip_migration_source` sets a source session to `source_released`, a later migration-status poll calls the source delta pump. The pump sees zero pending delta requests and overwrites the terminal state with `source_delta_transferred`. This makes rollover reject the next session and makes the source appear non-idle even when running and waiting request lists are empty.

## Required behavior

- `source_released` and `source_aborted` are terminal and monotonic.
- `active` and `target_aborted` are target terminal states and monotonic.
- Transfer polling may continue to refresh counters and timing, but must never replace either terminal state.
- Non-terminal sessions keep the existing transitions to `source_delta_transferred` or `source_failed`.
- No force role switch and no relaxation of source-idle or rollover safety checks.

## Implementation

In `_pd_flip_source_pump_delta_transfer`, capture whether the session is already terminal before applying the final state transition. Only non-terminal sessions may transition to `source_failed` or `source_delta_transferred`.

Apply the same rule to both target polling layers. Once the target is `active` or `target_aborted`, neither the initial target pump nor target delta pump may replace that state with `target_transferred`, `target_delta_transferred`, or `target_failed`. Counters may still refresh.

The change is intentionally local: controller behavior, status polling, transfer counters, metadata release, and output relay are unchanged.

## Verification

- A released source session with completed non-noop delta remains `source_released` after a pump/status poll.
- An aborted source session remains `source_aborted`.
- A non-terminal completed delta still becomes `source_delta_transferred`.
- A non-terminal failed delta still becomes `source_failed`.
- An active target remains `active` after both initial and delta status pumps.
- An aborted target remains `target_aborted`.
- Real trace acceptance requires node2 to become Prefill, 40/40 requests with zero errors, and all workers healthy after the post-workload observation period.
