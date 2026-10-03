# Proactive resident HiCache restore (experimental)

Restore an **exact token prefix** before an ordinary continuation reaches `/generate`.
Requires a fixed text-generation model, TP1/PP1/DP1, FULL-only resident (`cache`)
HiCache, and the file backend. LoRA, speculation, disaggregation, sidecars, SWA,
embedding models, and multimodal models are rejected. This control does not send a
model request or move KV to the GPU.

```sh
curl -s http://localhost:30000/hicache/prefetch -H 'Content-Type: application/json' \
 -d '{"operation_id":"tool-1","input_ids":[1,2,3,4],"ttl_ms":10000}'
curl -s http://localhost:30000/hicache/prefetch -H 'Content-Type: application/json' \
 -d '{"operation_id":"tool-1","action":"status"}'
curl -s http://localhost:30000/hicache/prefetch -H 'Content-Type: application/json' \
 -d '{"operation_id":"tool-1","action":"cancel"}'
```

Use a prefix at least as long as the storage prefetch threshold (the four-token
example only illustrates the request shape). Prefixes are aligned down to complete
cache pages. For an exact continuation prompt, normally omit the final token and
align the remainder, matching ordinary prompt-cache lookup. Include the same
`cache_salt` in both requests when used; there is no implicit session lookup.
Existing admin API authentication applies.

Only one restore is active; 32 recent outcomes are retained. Repeating an ID with
the same normalized prefix/salt/TTL is idempotent; a different payload is rejected.
Accepted operations begin `RUNNING` and finish `SUCCESS`, `MISS`, `FAILURE`,
`CANCELLED`, or `EXPIRED`. `CACHED` is a no-I/O result, `DECLINED` means existing
controller admission did not start the restore. A matching early continuation
joins the in-flight restore instead of submitting duplicate storage reads. It
then uses ordinary prefix matching and H2D. Other requests remain ordinary.

TTL cancels pending work; it does **not** create a resident cache lease. Published
pages use normal eviction. Cancelling a completed restore does not invalidate
shared KV. Finite failures use the separate lifecycle fix; a backend call that
never returns cannot be reclaimed safely. Cancelled allocated work retains its
existing ownership until terminal ACK. Engine pause cancels this control and
continues draining its ACKs.

## Benchmark

```sh
python benchmark/hicache/bench_proactive_prefetch.py \
 --model-path /path/to/model --work-dir /tmp/hicache-bench \
 --results-dir /tmp/hicache-results --repetitions 3
```

A: recompute with empty L3; B: request-time file L3 restore; C: control restore
before arrival. Each trial has a fresh process/cache, identical model/server/
sampling configuration, and no injected I/O delay. Five tool gaps: 0/100/500/1000/
3000 ms. Client TTFT comes from the first SSE token event. Existing plugin hooks
record actual storage reads, publication, host occupancy/evictions, and H2D;
all three modes have identical instrumentation. Source file KV is copied into
B/C trial directories; this measures a local file backend with potentially warm
OS page cache. It is not a remote-storage latency claim.

Zero-gap includes control RPC latency; `actual_signal_to_arrival_ms` records that
overhead. Compare observed TTFT savings with measured hidden restore work rather
than assuming an exact nominal zero. Results include raw per-trial timings,
identical output-token verification, duplicate reads, and a no-continuation
experiment followed by ordinary cache flush to verify reclaimability.
