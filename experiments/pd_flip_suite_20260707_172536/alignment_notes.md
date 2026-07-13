# Alignment Notes

This suite includes raw files plus derived alignment helpers.

- `aligned_timeline.csv`: merged request, controller, and worker-status events for the state-machine 200-request run.
- Time zero is the replay runner `run_started_wall` from `10_ab_200_state_machine_two_phase/state_machine/summary.json`.
- Request `first_token` wall time is reconstructed as `start_wall + ttft_s`.
- Controller action wall time is anchored to the suite log line `forcing two-phase D->P`; per-action times are cumulative `elapsed_seconds`.
- Worker status rows use sampler `ts_wall`; sampler interval was 0.2s, so status-stage boundary precision is about one polling interval.
- Worker `timing_debug` values are preserved as raw JSON in `details_json`. Many of those values are worker-local monotonic times, so use them as intra-worker relative timing unless wall-clock fields are added later.

Primary A/B result:

- Baseline no-state-machine: see `09_ab_200_baseline_no_state_machine/baseline/summary.json`.
- State-machine two-phase: see `10_ab_200_state_machine_two_phase/state_machine/summary.json`.

Important caveat: runs 01-08 and the first 10 run have request/controller raw data, but their sampler was started with an old node spec and did not collect worker status. The rerun of `10_ab_200_state_machine_two_phase` fixed this and is the aligned chain-measurement source.
