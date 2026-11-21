# Member load

`GET /internal/member_load` reports this router's load per worker role, so a
gateway can weigh members (one router and its engines) against each other. It
is read-only: each request takes one pass over the worker registry and the
engine-reported load the router already holds, starts no polling, and changes
nothing about routing. Successful requests are logged at DEBUG, like the
probes.

```json
{
  "contract": "member-load/v1",
  "observed_at_ms": 1794904913200,
  "router": {"instance": "<pod name>", "version": "<router version>"},
  "roles": {
    "decode": {
      "workers_total": 3,
      "workers_ready": 3,
      "workers_reporting": 3,
      "running_reqs": 842,
      "waiting_reqs": 15,
      "kv_used_tokens": 123,
      "kv_capacity_tokens": 456,
      "kv_usage_mean": 0.81,
      "kv_usage_max": 0.94,
      "waiting_uncached_tokens": 0,
      "est_prefill_queue_ms_min": 0.0,
      "est_prefill_queue_ms_max": 0.0,
      "max_running_reqs": 384,
      "running_frac_max": 0.92
    }
  }
}
```

- `roles` has a `plain`, `prefill` or `decode` key iff the registry holds at
  least one registered worker (one in a model pool) of that role.
- `workers_total`: registered workers of the role. `workers_ready`: those the
  router would send a request now: the circuit breaker allows it, a prefill
  knows its bootstrap port, and with `--load-monitor` its load report is fresh
  (the chat handler's freshness gate). `workers_reporting`: in-service workers
  whose every reported DP rank is fresh (and, with `--load-monitor`, whose
  latest pull succeeded with non-zero capacity).
- In service means routing could use the worker's shape: a prefill without a
  bootstrap port is left out. The circuit breaker is not consulted, so a
  worker failing under load still counts, and `workers_reporting` can exceed
  `workers_ready` while breakers are open.
- `running_reqs`, `waiting_reqs`, `kv_used_tokens`, `kv_capacity_tokens`: sums
  over reporting workers (each worker summed over its DP ranks), capped at
  2^63-1.
- `kv_usage_mean` / `kv_usage_max`: mean and max of each reporting worker's
  `kv_used_tokens / kv_capacity_tokens`, over workers with a known (non-zero)
  capacity; `null` when there are none.
- The fields below come from the extended load newer engines send: five
  fields appended to `LoadStat` after `attn_dp_rank`
  (`num_waiting_uncached_tokens, num_total_tokens, max_running_requests,
  total_prefill_uncached_tokens, total_prefill_busy_us`), the same names in
  `GET /v1/loads?include=core`. A worker counts only when every DP rank's
  latest report has all five; older engines leave them `null`. Reading them
  changes nothing about routing.
- `waiting_uncached_tokens`: uncached prompt tokens waiting for prefill,
  summed over those workers.
- `est_prefill_queue_ms_min` / `_max`: min / max over those workers of the
  expected wait before a request sent now starts prefill: the worker's
  uncached tokens waiting ÷ its prefill throughput (summed over DP ranks).
  Throughput is uncached tokens per second of prefill *busy* time (from the
  cumulative counters), as a moving average over the last ~10 busy seconds;
  idle time does not age it. Warm-up: the first report seeds it with the
  engine's lifetime average, and a rank that has prefilled less than 1 busy
  second in total has no rate yet. A worker with nothing waiting reads 0 ms
  even without a rate; one with tokens waiting and no rate (any rank cold)
  gives no estimate, so it can never read as idle. An engine restart
  (counters going backwards) restarts the average.
- `max_running_reqs`: sum of the engines' `max_running_requests` over DP ranks
  and over those workers that report a non-zero limit. `running_frac_max`:
  max over the same workers of `running / max_running_requests`, each summed
  over the worker's DP ranks (can pass 1 by the engine's own accounting).
  Both `null` when no worker reports a limit.
- Load comes from the engines' load socket (engines that advertise a load
  port) and, with `--load-monitor`, from pulls. With neither, every role
  reports `workers_reporting: 0`, zero sums and `null` usage.
- Load from workers not in the registry is ignored. `observed_at_ms` is wall
  clock epoch milliseconds. `router.instance` is the pod identity the access
  log uses (`POD_NAME`, else `HOSTNAME`, else `unknown`).
