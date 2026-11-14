#!/usr/bin/env bash
# collect_telemetry.sh — sample NPU and server state during a benchmark run.
#
# Usage:
#   bash bench/collect_telemetry.sh start        # begins a background sampler
#   bash bench/collect_telemetry.sh stop
#   bash bench/collect_telemetry.sh snapshot     # one-off labelled snapshot
#
# Writes results/telemetry/<timestamp>.csv with: ts, device_count, hbm_used_mb (per-device columns),
# plus results/telemetry/server-metrics.log (Prometheus /metrics samples) when the server exposes it.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJ=$(cd "$HERE/.." && pwd)
# shellcheck source=../env/00_common.sh
source "$PROJ/env/00_common.sh"

TEL_DIR="$RESULTS_DIR/telemetry"
mkdir -p "$TEL_DIR"
PIDFILE="$TEL_DIR/sampler.pid"
: "${INTERVAL:=2}"

sample_once() {
  local out=$1 ts
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if command -v npu-smi >/dev/null 2>&1; then
    # One row per sample; npu-smi's table is flattened into a single CSV field to stay robust
    # across driver versions, and duplicated raw for parsing later.
    {
      printf '%s,' "$ts"
      npu-smi info 2>/dev/null | tr '\n' ';' | sed 's/;/ /g; s/  */ /g'
    } >> "$out"
    [ -f "$out.raw" ] || : > "$out.raw"
    { echo "### $ts"; npu-smi info 2>/dev/null; } >> "$out.raw"
  else
    echo "$ts,npu-smi-unavailable" >> "$out"
  fi
  if curl -sf -m 2 "http://127.0.0.1:$PORT/metrics" >/dev/null 2>&1; then
    { echo "### $ts"; curl -s -m 2 "http://127.0.0.1:$PORT/metrics" | grep -E 'sglang_|npu_|cache|running|waiting' ; } \
      >> "$TEL_DIR/server-metrics.log"
  fi
}

case "${1:-}" in
  start)
    [ -f "$PIDFILE" ] && { echo "already sampling (pid $(cat "$PIDFILE"))"; exit 0; }
    CSVP="$TEL_DIR/telemetry-$(date +%Y%m%d-%H%M%S).csv"
    echo "csv=$CSVP"
    ( while :; do sample_once "$CSVP"; sleep "$INTERVAL"; done ) &
    echo $! > "$PIDFILE"
    echo "sampler started (pid $!, interval ${INTERVAL}s)"
    ;;
  stop)
    if [ -f "$PIDFILE" ]; then kill "$(cat "$PIDFILE")" 2>/dev/null || true; rm -f "$PIDFILE"; echo "sampler stopped"; else echo "not running"; fi
    ;;
  snapshot)
    OUT="$TEL_DIR/snapshot-$(date +%Y%m%d-%H%M%S).txt"
    { echo "=== $OUT ==="; npu-smi info 2>/dev/null || echo "npu-smi unavailable";
      echo; echo "=== server health ==="; curl -s -o /dev/null -w '%{http_code}\n' "http://127.0.0.1:$PORT/health" || true;
      echo; echo "=== metrics excerpt ==="; curl -s -m 3 "http://127.0.0.1:$PORT/metrics" 2>/dev/null | head -40 || true; } > "$OUT"
    echo "snapshot: $OUT"
    ;;
  *) echo "usage: $0 start|stop|snapshot"; exit 1 ;;
esac
