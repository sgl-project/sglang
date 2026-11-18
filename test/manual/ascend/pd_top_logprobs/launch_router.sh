#!/usr/bin/env bash
# Standard PD router command used by the manual test (sglang-router 0.3.2).
# Launch both workers first, then this script, then client.py.
set -eo pipefail
exec python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill "${PD_PREFILL_URL:-http://127.0.0.1:17710}" "${PD_BOOTSTRAP_PORT:-17730}" \
    --decode "${PD_DECODE_URL:-http://127.0.0.1:17720}" \
    --host "${PD_ROUTER_HOST:-127.0.0.1}" --port "${PD_ROUTER_PORT:-17700}" \
    --prometheus-port "${PD_ROUTER_METRICS_PORT:-17701}" \
    --request-timeout-secs 120 --disable-retries --disable-circuit-breaker \
    "$@"
