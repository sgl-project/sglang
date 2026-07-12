#!/usr/bin/env bash
set -euo pipefail

: "${STORE_HOST:?STORE_HOST is required}"
: "${REMOTE_ENV_FILE:?REMOTE_ENV_FILE is required}"
: "${STORE_GENERATION_TOKEN:?STORE_GENERATION_TOKEN is required}"
: "${STORE_GENERATION_FILE:?STORE_GENERATION_FILE is required}"

NEW_PID=$(ssh "$STORE_HOST" bash -s -- "$REMOTE_ENV_FILE" <<'REMOTE'
set -euo pipefail
source "$1"
for pid_file in /tmp/pd-flip-store.pid /tmp/pd-flip-store-health.pid; do
  if test -s "$pid_file"; then
    old_pid=$(cat "$pid_file")
    kill "$old_pid" 2>/dev/null || true
    while kill -0 "$old_pid" 2>/dev/null; do sleep 0.1; done
  fi
done
rm -rf /tmp/pd-flip-store-health
mkdir -p /tmp/pd-flip-store-health
nohup env \
  MOONCAKE_LOCAL_HOSTNAME="${MOONCAKE_MASTER%:*}" \
  MOONCAKE_GLOBAL_SEGMENT_SIZE=4gb \
  MOONCAKE_LOCAL_BUFFER_SIZE=0 \
  python3 -m mooncake.mooncake_store_service --port="$MOONCAKE_STORE_PORT" \
  >/tmp/pd-flip-store.log 2>&1 &
store_pid=$!
echo "$store_pid" >/tmp/pd-flip-store.pid
until timeout 1 bash -c "</dev/tcp/127.0.0.1/$MOONCAKE_STORE_PORT"; do
  kill -0 "$store_pid"
  sleep 0.2
done
printf '%s\n' "$store_pid" >/tmp/pd-flip-store-health/generation
nohup python3 -m http.server 18081 --directory /tmp/pd-flip-store-health \
  >/tmp/pd-flip-store-health.log 2>&1 &
echo $! >/tmp/pd-flip-store-health.pid
until timeout 1 bash -c '</dev/tcp/127.0.0.1/18081'; do sleep 0.1; done
printf '%s\n' "$store_pid"
REMOTE
)

printf '{"token":"%s","pid":"%s"}\n' \
  "$STORE_GENERATION_TOKEN" "$NEW_PID" >"$STORE_GENERATION_FILE"
