#!/bin/bash
# Cross-node PD: router. Runs on any node that can reach both PREFILL_IP
# and DECODE_IP. Exposes the OpenAI-compatible endpoint on $HOST_IP:8000.
set -ex

: "${HOST_IP:=0.0.0.0}"
: "${PREFILL_IP:?PREFILL_IP required}"
: "${DECODE_IP:?DECODE_IP required}"

# --disable-circuit-breaker: with a single P/D pair there is no failover target,
# so the breaker only hurts -- a slow large-prefix warmup trips it and the router
# then 503s ("all circuits open") against a perfectly healthy prefill worker.
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --host "$HOST_IP" \
  --port 8000 \
  --policy round_robin \
  --prefill-policy round_robin \
  --decode-policy round_robin \
  --disable-circuit-breaker \
  --request-timeout-secs 1800 \
  --prefill "http://${PREFILL_IP}:30025" 8998 \
  --decode  "http://${DECODE_IP}:30100"
