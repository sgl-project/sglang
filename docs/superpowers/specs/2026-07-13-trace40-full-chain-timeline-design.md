# Trace40 SLO-Driven PD Full-Chain Timeline Design

## Goal

Provide one experiment entry point on cloud-099 that runs the existing 40-request interleaved workload through the real request-level SLO-driven progressive D-to-P state machine on cloud-099 through cloud-102, while preserving the enabled dual-source HiCache stitch attempt and measuring every action, including stitch failure and source-Decode full fallback.

## Fixed topology

- cloud-099 / node0 / `192.168.0.42`: initial Prefill, Mooncake services, router, controller, workload, and measurement coordinator.
- cloud-100 / node1 / `192.168.0.40`: Decode.
- cloud-101 / node2 / `192.168.0.39`: Decode migration source and eventual Prefill candidate.
- cloud-102 / node3 / `192.168.0.41`: Decode migration target.
- Initial roles are `1P3D`; a committed run ends as `node0,node2=P` and `node1,node3=D`.

The experiment fixes the migration pair to `node2 -> node3` so repeated runs are comparable. The first migration selects the first `N=floor(running_requests * ratio)` requests. If the target lacks KV capacity or request slots, `N` is repeatedly halved until feasible or zero.

## Workload and SLO input

The run uses 40 requests, strictly alternating 20 long 10,000-character prompts and 20 short 1,000-character prompts. Each trace record carries its own TTFT and TPOT requirement. The replay process appends live request progress to a JSONL SLO ledger consumed by the controller.

The preferred trace spreads arrivals so that fresh requests and TPOT intervals continue to appear during the observation window. The all-at-once burst trace is retained for stress testing but is not the primary full-chain trace because it cannot reliably supply fresh post-migration TTFT samples.

The controller starts D-to-P only when all conditions hold:

1. Minimum Prefill and Decode sample counts are satisfied.
2. Prefill TTFT attainment is below the configured threshold.
3. Decode TPOT attainment is above the configured threshold.
4. node2 has migratable running requests.
5. node3 can accept at least one request after repeated halving.

After the first migration, the SLO monitor resets its logical window so the observation decision uses only fresh samples.

## Stitch and fallback behavior

Dual-source stitching remains enabled. For every selected request the target attempts:

1. Match L1 GPU, L2 host, and L3 Mooncake prefix coverage.
2. Compute candidate stitch boundary `H`.
3. Start L3 prefetch and L2-to-L1 load-back.
4. Re-match and validate actual continuous target coverage.
5. Receive the source Decode suffix `[H,C0)` when stitch is viable.

If target restore coverage is less than the announced `H`, the experiment records the stitch failure as an expected observed branch, not as the end of the experiment. It then records the fallback handshake and transfers the full committed KV range `[0,C0)` from node2. Target activation and request continuation proceed only after fallback coverage is complete.

The run is unsuccessful if stitch failure is hidden, fallback repeats for the same request, target coverage contains a gap, a request is lost or duplicated, or node3 cannot continue generation.

## Timeline event model

Every active operation emits a structured JSONL event with:

- `run_id`, `session_id`, optional `rid`, batch ordinal, node, role, process, and phase.
- `event`, `status`, failure reason, and relevant request/KV counts.
- wall-clock epoch in nanoseconds for cross-node ordering.
- process monotonic timestamp in nanoseconds for precise same-process durations.
- operation or correlation ID linking request, controller action, transfer, and fallback.

Each completed operation records either explicit start/end events or one event containing start, end, and elapsed duration. Cross-process summaries use wall-clock timestamps; process-local latency uses monotonic timestamps. The report labels 50 ms polling-derived intervals as observed bounds rather than exact internal latency.

Before traffic, the coordinator saves `date +%s%N`, timezone, `chronyc tracking` or equivalent, and host identity from all four nodes. A missing clock synchronization tool is recorded as a caveat, not silently ignored.

## Required measured phases

### Setup and readiness

- Mooncake metadata, master, and store start-to-ready.
- Every Worker start-to-health and role/event-loop verification.
- Router start-to-ready and worker discovery.
- Workload process start, first dispatch, and ledger readiness.
- Monitor process start and first valid SLO window.

### Trigger and selection

- Every SLO sample window and attainment result.
- Threshold first crossed and trigger accepted.
- Source/target selection.
- Router drain and source admission pause.
- Source running-request snapshot.
- Ratio-to-N calculation.
- Every target-capacity check and each halving decision.

### First migration and stitch attempt

- Source base-freeze start/end and `C0` per request.
- Target receiver creation and request-slot/KV-slot reservation.
- Prefix query start/end with L1/L2/L3 hit lengths and candidate `H`.
- L3 prefetch registration, progress completion, and loaded-token count.
- Host-to-GPU load-back start/end.
- Restore re-match, actual coverage, and stitch success/failure decision.
- Source suffix transfer `[H,C0)` start/end when attempted.
- Time from stitch attempt start to failure detection.

### Full source fallback

- Target fallback-required event and reason.
- Controller detection latency bounded by sampling interval.
- Source fallback command start/response.
- Target full-fallback preparation.
- Full `[0,C0)` KV transfer start/end, tokens, and available byte metrics.
- Target fallback receive completion and coverage validation.
- Total incremental cost added by the failed stitch before fallback succeeds.

### Activation, observation, and final decision

- Initial delta/quiesce boundaries where applicable.
- Target commit, source finish, target activate, and first continued target token.
- Observation-window exact start/deadline/end.
- Every fresh observation SLO sample.
- Recovery or persistent-risk decision.
- SAFE recovery admission/router resume, or second-migration start.

### Second migration and role switch

- Remaining-request snapshot and capacity decisions.
- Base/delta transfer and atomic activation stages.
- Source residual-request verification.
- Worker runtime role change and active-event-loop convergence.
- Router role update and admission resume.
- Final topology verification.

### Shutdown and artifact production

- Workload completion per request and overall.
- Metrics sampler stop and log collection duration.
- Summary generation and archive creation.

## Components

### Controller support

Expose the progressive monitor through a CLI entry point that accepts a trace ledger, fixed source, and fixed target. The CLI must execute `monitor_progressive`, not the older two-phase monitor. It must pass the configured first ratio, observation duration, sample minima, and threshold into the existing progressive evaluator.

### Trace SLO monitor

Add logical window reset support without truncating the raw ledger. Reset records a monotonic cutoff; later reads ignore earlier events. The raw ledger remains append-only for auditability.

### Timeline instrumentation

Extend existing controller action records and migration measurements rather than creating an unrelated timing system. Worker-internal stitch and fallback events supply exact internal timestamps; the 50 ms sampler supplies external detection bounds and router/role transitions.

### Four-node runner

Add a Bash coordinator intended to run on cloud-099 with commands:

- `preflight`: read-only validation; never starts services.
- `run`: starts the configured experiment and saves the run ID.
- `status`: reads Worker, router, workload, controller, and sampler state.
- `collect`: collects logs and generates summaries without stopping shared services.
- `stop`: stops only containers/processes tagged with the current run ID; it never uses broad process killing.

The runner refuses to overwrite an artifact directory, refuses placeholder credentials, verifies the repository/image/trace on every required host, and records the exact commands and resolved non-secret configuration. It never writes secrets into artifacts.

## Failure handling

- Preflight failure prevents all experiment starts.
- Partial startup records successfully started run-owned processes and provides scoped cleanup commands.
- Controller failure triggers its existing safe admission/router cleanup and preserves the session journal.
- Stitch failure proceeds to exactly one full-source fallback per RID.
- Migration or fallback failure prevents role commit.
- Log collection runs even after workload/controller failure.
- Shared Mooncake or Worker containers are not stopped automatically at the end.

## Artifacts and report

The run directory contains the original trace checksum, resolved redacted configuration, request/response raw JSONL, per-request SLO ledger, Controller JSON and journal, 50 ms cluster events, four Worker logs, router and Mooncake logs, pre/post/final role snapshots, clock-sync evidence, and container/image/source metadata.

The generated report provides:

- request completion and TTFT/TPOT attainment;
- the state-machine path and final topology;
- per-stage exact or observed-bound latency;
- per-request P/H/C0/C1 and transfer ranges;
- stitch attempt duration, failure reason, fallback duration, and combined migration duration;
- observation duration and samples used for the final decision;
- raw artifact paths and explicit measurement caveats.

## Local verification before cluster use

Local tests must prove:

1. The progressive CLI selects `monitor_progressive` and forwards the trace ledger and fixed pair.
2. Trace-window reset excludes old records while preserving the ledger.
3. D-to-P requires low Prefill attainment and healthy Decode attainment.
4. Timeline start/end events produce correct durations and correlation fields.
5. Stitch failure and full fallback appear as separate measured phases.
6. The runner's preflight, command rendering, scoped cleanup, redaction, and dry-run behavior work without SSH or Docker side effects.

No cluster process is started during implementation or local verification.
