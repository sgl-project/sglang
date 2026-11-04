# Sustained L2/L3 benchmark at `mem_fraction_static=0.8`

`stress_bench.py` compares paired arms while retaining the native cache state during each arm. It uses two already-running Engines and never starts, stops or reconfigures serving processes. Both Engines must have HiCache and Mooncake enabled, the same model/cache/control layout, no implicit KV policy, `mem_fraction_static=0.8`, and no explicit `max_total_tokens` cap.

Use `--retention-mode natural` for primary tests of ordinary cache pressure at the full GPU capacity. Native GPU residency accumulates across jobs and native LRU eviction determines device, host and storage reuse. Capture Engine GPU KV occupancy metrics during the complete run and report observed peak used GPU KV tokens with the actual boot capacity. `nvidia-smi` allocated memory alone cannot prove KV occupancy.

`--retention-mode demote` remains the default for compatibility with the explicit L2/L3 path experiment. It deliberately removes each relevant prefix from GPU after generation, reducing the role of GPU LRU capacity. Keep results from the two profiles separate and record the selected mode in every report; the manifest, arm files and summary capture it.

The static memory fraction is a GPU memory budget, not measured GPU busy time. Save actual Engine boot capacities and resource telemetry separately. Supply the measured destination host token capacity with `--host-capacity-tokens`. The formal working set must exceed that capacity. The `--allow-small-working-set` flag permits functional smoke tests and labels the manifest accordingly.

The harness compares all configuration fingerprint fields exactly, except automatically derived `max_total_num_tokens`: up to a 1% difference is accepted and recorded when neither Engine has an explicit token cap. This covers small boot allocation differences between GPU models. It does not equalize different GPU hardware; preserve hardware identities and compare arms on the same destination Engine.

For the measured example, Engine B has 346,944 GPU KV tokens and 693,952 host KV tokens. These numbers must be refreshed after a new boot. A 64-session dataset of 16,384-token prefixes contains 1,048,576 logical tokens, about 1.51 times this host pool. A 53-prefix churn phase contains 868,352 logical tokens, about 1.25 times the pool. Token counts describe logical materialization; they do not establish a simultaneous resident byte peak.

## L2: retain selected prefixes while unrelated work displaces host KV

```bash
.venv/bin/python benchmark/session_kv_control/stress_bench.py l2 \
  --engine-a http://127.0.0.1:51801 \
  --engine-b http://127.0.0.1:51803 \
  --model /home/youwei/bzh/model/Qwen/Qwen3-8B \
  --host-capacity-tokens 693952 \
  --length 16384 --hot-sessions 4 --pressure-prefixes 53 \
  --rounds 3 --concurrency 8 --output-tokens 32 --retention-mode natural \
  --pin-ttl-s 900 --timeout-s 120 \
  --output /tmp/session-kv-mem08-l2
```

Each round runs `on_demand` and `pin` in seeded random order. Each arm flushes B once, primes hot sessions, confirms full GPU-to-storage publication and GPU-to-host DEMOTE, runs unrelated real generation and host materialization, samples hot host residency, then resumes all hot sessions with the same suffix. The candidate acquires explicit L2 eviction-exclusion claims and verifies their complete active scopes again before resume. Baseline host misses are observed rather than required. Hot priming always includes DEMOTE to establish the host-resident set, followed by PIN in the candidate. For pressure jobs, `natural` uses GPU-to-host COPY and confirms full scope without requesting GPU reclamation; device KV remains available for native LRU eviction. The `demote` profile explicitly DEMOTEs every pressure prefix. Neither profile silently changes the pressure workload size.

Hot and pressure salts include run, round and arm, so the second arm cannot reuse the first arm's newly published storage objects. Token material, output count, workload sizes and control paths remain matched. Salts differ intentionally; resumed output IDs and text must match by logical session across arms and rounds.

The native default `ProtectionLimits.max_protected_tokens` is 65,536. The default four hot sessions at 16K use this entire budget. Eight hot sessions at 16K require an explicit larger Engine `kv_control_config.protection_limits.max_protected_tokens`. The harness reads that setting from server metadata when available, otherwise uses this source snapshot's default. It rejects oversized sets and never silently reduces the workload. Actual acquire responses remain authoritative. PIN TTL is bounded and claims are explicitly released and queried for terminal tombstones even when inference or later controls fail.

Report both hot-session TTFT and whole-arm cost. Whole-arm time includes flush, priming/publication, claim acquisition, all pressure generation and host-materialization work, residency checks, foreground resume and claim cleanup. Pressure latency is reported independently. A hot TTFT reduction can coexist with slower pressure processing or lower whole-service throughput.

## L3: cyclic working set with genuine tool-wait overlap

```bash
.venv/bin/python benchmark/session_kv_control/stress_bench.py l3 \
  --engine-a http://127.0.0.1:51801 \
  --engine-b http://127.0.0.1:51803 \
  --model /home/youwei/bzh/model/Qwen/Qwen3-8B \
  --host-capacity-tokens 693952 \
  --sessions 64 --length 16384 --passes 2 \
  --blocks 3 --concurrency 8 --output-tokens 32 --retention-mode natural \
  --tool-wait-s 3 --lead-s 2.5 --timeout-s 120 \
  --output /tmp/session-kv-mem08-l3-c8
```

A publishes the full unique-salt dataset once before comparison; every publication must confirm full scope. Each paired block runs `on_demand` and `prefetch` in seeded random order. The same seeded dataset permutation and fixed per-client lanes apply to both arms. B flushes once at arm start and then retains ordinary native cache behavior. Increase `--passes` or `--blocks` for longer runs; scan concurrency 8 and 16 with independent output directories and preserve the same seed and workload dimensions.

Each closed-loop client starts a real tool wait. The prefetch arm submits asynchronous storage-to-host work at `deadline - min(lead, wait)`. Generation is sent at the fixed tool deadline without waiting for readiness. Late completion and client dispatch delay remain visible. The control future drains after generation under the existing bounded operation timeout/cancellation protocol. There is no implicit PIN. In `natural` mode neither arm requests DEMOTE after response. Native write-through, GPU residency and GPU/host eviction remain active, and actual device hits are retained in accounting. In `demote` mode both arms explicitly DEMOTE the prefix to host after each response to isolate host/storage reuse. Control drain and configured retention work contribute to client-cycle and arm time. The `demote` profile intentionally changes GPU residency and must not be presented as ordinary application cache pressure.

On-demand requests may reuse L2 content retained by prior requests. Those hits remain in the result. Actual device, host and storage token attribution comes from generation metadata. Output IDs and text are checked for every logical dataset session across arms, passes and blocks. Cache metadata or control failure invalidates the relevant job without discarding a completed generation's latency or output throughput.

Source publication cost is separate in `source-publication.json`. Arm time includes the initial flush, all tool waits, inference, control drain and any configured DEMOTE. Compare TTFT, complete client-cycle latency, whole-arm completed requests/s and output tokens/s together. Concurrent requests are correlated samples; three paired blocks do not establish stable p99 or a general throughput guarantee.

A small smoke example adds `--sessions 2 --length 2048 --blocks 1 --passes 1 --concurrency 2 --output-tokens 4 --allow-small-working-set`. It validates execution and accounting only.

## Evidence and verification

Each invocation requires a new output directory and writes:

- `manifest.json`: arguments, complete metadata/capabilities, both configuration fingerprints, input token material and hashes, source revision and Python module hashes.
- `schedule.json`: exact paired arm order and per-block dataset permutation. L3 client lanes are the permutation/pass list sliced by client index. Each arm also records its retention mode.
- `raw.jsonl`: complete HTTP requests, streamed events, control replies, periodic metrics/load samples, phase metadata, per-request L3 outcomes, arm results and run failures. Salts, operation references and monotonic timestamps support attribution.
- `b*-*.json`: each arm's complete rows, full control receipts, timings, tier tokens, failures and cleanup evidence.
- `summary.json`: all planned/completed/failed/unattempted counts, all completed-request latency distributions, TTFT p50/p95/p99, tier token sums, repeated prefill, actual wall throughput, readiness/late/status/unused-work accounting and paired block deltas.
- `checksums.json`: SHA-256 for produced evidence files, excluding itself.

Periodic `/metrics` and `/v1/loads` calls use one bounded thread and a short independent request timeout. Full `/server_info` is captured at startup and arm boundaries, outside measured arm time. GPU busy time, physical Mooncake I/O bytes, host byte peak and service CPU require external owner telemetry; observed peak used GPU KV tokens should be derived from sampled Engine pool metrics and sampling cadence disclosed; unknown quantities remain absent or null. Optional telemetry failure does not become a serving failure.

A failed generation remains in the planned denominator. A later control failure retains the completed generation in latency and output counts while marking its cache-path validation invalid. Whole-arm failures preserve unattempted planned requests. Exit status is 0 only when every arm and output/control/cleanup check passes, 2 for completed comparisons containing invalid arms, and 1 for run setup or orchestration failure. Read the failure counts alongside latency: completed-only timing cannot describe requests that failed to complete.

```bash
cd benchmark/session_kv_control
/home/youwei/bzh/project/TokenCake/RFC/sglang-rfc/.venv/bin/python \
  -W error::ResourceWarning -m unittest -v test_stress_bench test_session_kv_bench
```

## Native read pipeline setting

Engine `--kv-control-config` accepts `"cache_limits":{"max_concurrent_storage_reads":4}` (default 4, range 1–64). Different effective cache salts may pipeline native storage-to-host reads. Same-salt scopes retain serialization and exact identical scopes share one physical read. GPU destinations retain exclusive admission; an older live GPU waiter blocks newer cold jobs until its turn, cancellation or expiry. Complete host-resident targets can finish immediately. Setting 1 serializes the cold stage for comparisons. This does not add Mooncake GET worker threads or widen supported topology.

Queries reserve their full missing host span before asynchronous allocation; after real allocation, free-pool accounting owns that space. Cancellation retains the reservation until physical drain. PIN and native references still constrain host reclamation. More admitted reads can increase scratch occupancy and contention; report readiness, total cycle and throughput against the same setting-1 or previous-source control. A larger admission bound alone does not establish a gain.

## Offline verification and comparison

```sh
.venv/bin/python benchmark/session_kv_control/analyze_stress.py \
  --results /tmp/sglang-rfc-build/mem08 --output /tmp/mem08-analysis
```

The output directory must be new. Completed formal cells require a manifest, full paired schedule, arm files, summary and SHA-256 index. Live partial cells, functional smoke and failed setup remain separately listed. The verifier recomputes numeric summaries, checks actual tier metadata/output equality/full ready scopes and raw PIN acquisition/release proofs. All completed generations contribute even when a later control fails. A completed request with unverifiable cache metadata stays in throughput and latency and is flagged explicitly. Source publication, foreground recovery and pressure requests remain separate. JSON/gzipped JSON and raw JSONL/gzipped JSONL are accepted; regenerate the physical-file checksum index after packing and preserve the original index as provenance.

`resources.jsonl` from `observe_resources.py` supplies sampled GPU and cache facts. `kv_used_tokens` and `token_usage` count active slots and exclude evictable radix cache. Total allocated/reserved GPU KV slots use boot capacity minus `kv_available_tokens`; this includes transfer allocations. The report discloses 2s sampling and approximate arm bounds rather than claiming exact byte peaks. Standalone CSV/PNG export uses an already-installed matplotlib; dependencies are not installed by this tool.

The analyzer reports inclusive arm time and work time excluding only the initial laboratory `/flush_cache` RPC. It lists every flush duration and leaves capability refresh, source/priming work, pressure requests, waits, inference and cleanup in the work cost. A reset can be much slower than inference; inclusive differences alone do not establish service throughput gains. Resource windows use raw `arm_result` timestamps and measured arm duration with an explicitly unknown validation/logging gap.

Exact ordinary demand can also join a native host-read job while it is queued, before storage I/O starts. An unrelated blocked GET followed by a second queued scope and its matching demand required three query/GET calls before this change and two after, with the same delivered span. A joined queued job unable to obtain whole-scope host capacity returns DEFERRED so the normal request can continue computing. GPU barriers decline new joins or defer joined queued work, retaining ordinary fallback and DMA fences. Native-only waiting behavior and exact prefix/salt/extra-key/stop-policy checks remain unchanged.

Large observer captures can be archived in ordered gzip parts. `resources-index.json` lists `parts` entries with `file` and physical `sha256`; the verifier rejects missing, duplicate, unsafe or mismatched parts before parsing every sampled row. A single `resources.jsonl[.gz]` and an indexed capture cannot coexist at the same result root. This keeps individual evidence files bounded without dropping metrics.

Measured 0.8 results, raw files, sampled CPU costs and limits are in [the acceptance report](results/2026-11-04-mem08/FINAL_ACCEPTANCE.zh-CN.md). The archive intentionally contains one baseline failed prefetch and four optimized capacity deferrals, so the analyzer returns exit 2 while reporting all 8 cells verified and all 5,724 calls completed.
