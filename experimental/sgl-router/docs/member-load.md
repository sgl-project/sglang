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
      "waiting_uncached_tokens": null,
      "est_prefill_queue_ms_min": null,
      "est_prefill_queue_ms_max": null
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
- `waiting_uncached_tokens`, `est_prefill_queue_ms_min` / `_max`: always
  `null` on this router. Its engine load (`LoadStat` on the load socket,
  `GET /v1/loads?include=core` with `--load-monitor`) carries no uncached-token
  or prefill-throughput fields.
- Load comes from the engines' load socket (engines that advertise a load
  port) and, with `--load-monitor`, from pulls. With neither, every role
  reports `workers_reporting: 0`, zero sums and `null` usage.
- Load from workers not in the registry is ignored. `observed_at_ms` is wall
  clock epoch milliseconds. `router.instance` is the pod identity the access
  log uses (`POD_NAME`, else `HOSTNAME`, else `unknown`).
