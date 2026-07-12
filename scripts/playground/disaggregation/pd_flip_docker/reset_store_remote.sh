#!/usr/bin/env bash
set -euo pipefail

: "${STORE_HOST:?STORE_HOST is required}"
: "${REMOTE_ENV_FILE:?REMOTE_ENV_FILE is required}"
: "${STORE_GENERATION_TOKEN:?STORE_GENERATION_TOKEN is required}"
: "${STORE_GENERATION_FILE:?STORE_GENERATION_FILE is required}"

REMOTE_PROOF=$(ssh "$STORE_HOST" bash -s -- "$REMOTE_ENV_FILE" <<'REMOTE'
set -euo pipefail
source "$1"
state_dir="${XDG_STATE_HOME:-$HOME/.local/state}/sglang-pd-flip"
mkdir -p "$state_dir"
chmod 700 "$state_dir"
store_pid_file="$state_dir/store.pid"
health_pid_file="$state_dir/health.pid"
health_dir="$state_dir/health"

stop_owned_process() {
  pid_file=$1
  expected=$2
  if ! test -s "$pid_file"; then return; fi
  pid=$(cat "$pid_file")
  case "$pid" in *[!0-9]*|'') return 1 ;; esac
  test "$(ps -o uid= -p "$pid" | tr -d ' ')" = "$(id -u)"
  ps -o args= -p "$pid" | grep -F -- "$expected" >/dev/null
  kill "$pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 0.1; done
}

for spec in "$store_pid_file:mooncake_store_service" "$health_pid_file:http.server 18081"; do
  pid_file=${spec%%:*}
  expected=${spec#*:}
  if test -s "$pid_file"; then
    stop_owned_process "$pid_file" "$expected"
  fi
done
rm -rf "$health_dir"
mkdir -p "$health_dir"
nohup env \
  MOONCAKE_LOCAL_HOSTNAME="${MOONCAKE_MASTER%:*}" \
  MOONCAKE_GLOBAL_SEGMENT_SIZE=4gb \
  MOONCAKE_LOCAL_BUFFER_SIZE=0 \
  python3 -m mooncake.mooncake_store_service --port="$MOONCAKE_STORE_PORT" \
  >"$state_dir/store.log" 2>&1 &
store_pid=$!
echo "$store_pid" >"$store_pid_file"
until ss -ltnp "sport = :$MOONCAKE_STORE_PORT" | grep -F "pid=$store_pid," >/dev/null; do
  kill -0 "$store_pid"
  sleep 0.2
done
starttime=$(awk '{print $22}' "/proc/$store_pid/stat")
generation=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
printf '{"pid":"%s","starttime":"%s","generation":"%s"}\n' \
  "$store_pid" "$starttime" "$generation" >"$health_dir/generation"
nohup python3 -m http.server 18081 --directory "$health_dir" \
  >"$state_dir/health.log" 2>&1 &
health_pid=$!
echo "$health_pid" >"$health_pid_file"
until ss -ltnp 'sport = :18081' | grep -F "pid=$health_pid," >/dev/null; do
  kill -0 "$health_pid"
  sleep 0.1
done
printf '%s %s %s\n' "$store_pid" "$starttime" "$generation"
REMOTE
)

read -r NEW_PID NEW_STARTTIME NEW_GENERATION <<EOF
$REMOTE_PROOF
EOF
test -n "$NEW_PID" && test -n "$NEW_STARTTIME" && test -n "$NEW_GENERATION"
printf '{"token":"%s","pid":"%s","starttime":"%s","generation":"%s"}\n' \
  "$STORE_GENERATION_TOKEN" "$NEW_PID" "$NEW_STARTTIME" "$NEW_GENERATION" \
  >"$STORE_GENERATION_FILE"
