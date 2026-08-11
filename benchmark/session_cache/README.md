# Session cache priority pressure benchmark

This benchmark measures whether demoting low-value radix-native sessions protects other sessions during severe device-KV pressure on one engine.

Start a fresh server for each arm. Set `--max-total-tokens` low enough to make the working set exceed device KV capacity:

```bash
python -m sglang.launch_server \
  --model-path /path/to/model \
  --tp 4 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-session-radix-cache \
  --radix-eviction-policy priority \
  --max-total-tokens 131072 \
  --port 30000
```

Run the protected control arm:

```bash
python benchmark/session_cache/bench_priority_pressure.py \
  --arm protected \
  --output /tmp/session-priority-protected.json
```

Restart the server, then run the demoted arm:

```bash
python benchmark/session_cache/bench_priority_pressure.py \
  --arm demoted \
  --output /tmp/session-priority-demoted.json
```

The default workload primes eight low-value sessions, primes six high-value sessions so they are newer, then grows only the low-value sessions past the configured device-KV capacity. The expected signal is a higher `high_cached_fraction_mean` in the demoted arm with `all_requests_succeeded=true` in both arms. For a stable comparison, run fresh-server A/B and B/A pairs.
