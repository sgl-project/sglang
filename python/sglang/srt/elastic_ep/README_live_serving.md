# Live elastic EP scale-up

Live scale-up prepares one joining cohort while the original ranks serve. Mooncake
remains the process-group backend and NIXL EP handles dispatch/combine. Admission
and topology changes still run synchronously on the scheduler; this feature does
not promise zero latency degradation.

## Configuration

Set `SGLANG_ELASTIC_LIVE_SCALE=split` uniformly on all ranks, including scale and recovery joiners. Set
`--elastic-ep-reuse-cuda-graph` and a common `--max-ep-size` to retain the original
decode graphs and capture only joining ranks before admission. Start all joiners
before posting `/scale_elastic_ep`; a cohort is admitted together after preparation.

Without the reuse flag, `split` interleaves expanded eager serving with individual
CUDA graph capture steps. `SGLANG_ELASTIC_CAPTURE_SERVICE_STEPS` controls the number
of scheduler turns between captures (default 32). `prepare` overlaps only NIXL
connection preparation and retains synchronous graph recapture. An unset live-scale
environment variable keeps the existing scale path. Eager serving is supported.

## Protocol and storage

1. Old ranks agree on the target using the existing DP metadata gather. Each rank
   starts an operation-scoped background connection with NIXL `activate=False`.
2. The joining cohort initializes locally. With graph reuse, remote peers stay
   masked while joiners capture decode graphs. All ranks validate the same
   capacity, dispatch layout, graph configuration, and reuse policy.
3. At a forward boundary, original ranks admit the complete cohort into Mooncake,
   install expert routing, activate NIXL peers, and complete the ready barrier.
   Graph reuse retains graph, parameter, routing, and expert-counter storage.
4. The expanded cohort serves. HTTP readiness follows graph preparation; an
   intermediate eager-serving notification does not finish a split-capture scale.

Repeated growth uses a new operation namespace. Scale-down and recovery use their
existing protocols: recovery regrowth does not require a live preparation manifest
and recaptures graphs. Recovery joiners use ordinary synchronous admission, then join the same live
control protocol as the original serving ranks.

## Current scope

- Non-overlap scheduler; attention TP/CP/DCP/PP all 1; Mooncake PG and NIXL EP.
- Graph reuse additionally requires dense TP 1 and DP LM head.
- No speculative decoding, two-batch overlap, graph-pool borrowing, or LP dispatch
  solver with live graph reuse.
- NIXL must support `connect_ranks(activate=False)` and release the Python GIL while
  connecting. Same-node CUDA IPC setup can still compete for CUDA driver locks.
- Once transport preparation starts, connection errors/timeouts fail closed.
  Earlier request deadlines fail the scale operation without terminating serving. An in-flight connection cannot be safely
  cancelled; restart before retrying. Operation metadata remains in TCPStore for
  the process lifetime to avoid early-deletion races with slow participants.
- This port has CPU protocol tests. Distributed serving/accuracy and failure tests
  must be rerun on this exact base before claiming production readiness. Results
  from the earlier development branch are not validation of this rebased branch.

CPU tests: `test/registered/unit/elastic_ep/test_live_scale.py`.
Tensor/CUDA regressions: `test/manual/ep/test_live_graph_reuse_tensors.py`.
