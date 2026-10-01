#!/usr/bin/env bash
# Sourced by run_probe.sh / run_harness.sh: launch one validation arm of the
# #39342 reproduction in its own session and tear down exactly that process
# group.
#
#   start_server <label> <A|B|C> <eager|graph> <logname> [chunked_prefill_size]
#   stop_server
#
# Environment: MODEL_PATH (required), WS (default /tmp/mixed_chunk_mamba),
# PORT (default 30000), HEALTH_TIMEOUT_S (default 600). SERVER_CMD overrides
# the launched command and SETSID_CMD the session wrapper (both for testing
# the launcher itself with a dummy server).
#
#   A: mixed ON,  radix ON (extra_buffer)   B: mixed OFF, radix ON   C: mixed ON, radix OFF

WS=${WS:-/tmp/mixed_chunk_mamba}
PORT=${PORT:-30000}
HEALTH_TIMEOUT_S=${HEALTH_TIMEOUT_S:-600}
HANDSHAKE_TIMEOUT_S=${HANDSHAKE_TIMEOUT_S:-15}
SERVER_PGID=""
SERVER_PID=""
SERVER_LOGNAME=""

_proc_pgrp_sid() {
  # pgrp and session id of one pid from /proc, independent of ps formatting.
  local stat rest
  stat=$(cat "/proc/$1/stat" 2>/dev/null) || return 1
  rest=${stat##*) }
  # shellcheck disable=SC2086
  set -- $rest
  printf '%s %s\n' "$3" "$4"
}

# The caller's own group; nothing below may ever signal it.
read -r _CALLER_PGID _CALLER_SID <<<"$(_proc_pgrp_sid $$)"

_port_in_use() {
  # A listener on the port answers the connect; refuse rather than kill it.
  (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null
}

_group_is_foreign() {
  # True when the group id is a live session leader that is not the caller's.
  local pgid=$1
  [ -n "$pgid" ] || return 1
  [[ "$pgid" =~ ^[0-9]+$ ]] || return 1
  [ "$pgid" != "$_CALLER_PGID" ] || return 1
  [ "$pgid" != "$$" ] || return 1
  [ "$pgid" != "1" ] || return 1
  return 0
}

stop_server() {
  local pgid=$SERVER_PGID
  SERVER_PGID=""
  SERVER_PID=""
  if [ -z "$pgid" ]; then return 0; fi
  if ! _group_is_foreign "$pgid"; then
    echo "stop_server: refusing to signal process group $pgid (caller's group or invalid)" >&2
    return 1
  fi
  # The whole group: launcher, scheduler, detokenizer and any Triton compile helpers.
  kill -TERM -- "-$pgid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 -- "-$pgid" 2>/dev/null; then return 0; fi
    sleep 0.5
  done
  kill -KILL -- "-$pgid" 2>/dev/null || true
  for _ in $(seq 1 10); do
    if ! kill -0 -- "-$pgid" 2>/dev/null; then return 0; fi
    sleep 0.2
  done
  echo "stop_server: process group $pgid still alive" >&2
  return 1
}

_abort_launch() {
  # Handshake failed: signal only the single pid we may know, never a group.
  local pid=$1
  if [ -n "$pid" ] && [[ "$pid" =~ ^[0-9]+$ ]] && [ "$pid" != "$$" ]; then
    kill -TERM "$pid" 2>/dev/null || true
  fi
  SERVER_PGID=""
  SERVER_PID=""
}

start_server() {
  local label=$1 arm=$2 graph=$3 logname=$4 chunk=${5:-2048}
  local here
  here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  mkdir -p "$WS/logs" "$WS/results"
  if [ -n "$SERVER_PGID" ]; then
    echo "start_server: a server started by this invocation is still running (pgid $SERVER_PGID)" >&2
    return 1
  fi
  if _port_in_use; then
    echo "start_server: port $PORT is already in use; refusing to start (nothing was killed)" >&2
    return 1
  fi
  SERVER_LOGNAME=$logname
  echo "label=$label sha=$(git -C "$here" rev-parse HEAD 2>/dev/null || echo unknown) arm=$arm graph=$graph chunk=$chunk port=$PORT" | tee "$WS/logs/$logname.meta"
  # Non-empty when the probe instrumentation (or anything else) is applied on top of HEAD.
  git -C "$here" diff --stat 2>/dev/null | tail -1 >> "$WS/logs/$logname.meta"

  local cmd
  if [ -n "${SERVER_CMD:-}" ]; then
    cmd=(bash -c "$SERVER_CMD")
  else
    : "${MODEL_PATH:?set MODEL_PATH to the local model directory}"
    local armflags graphflags
    case "$arm" in
      A) armflags=(--enable-mixed-chunk --mamba-radix-cache-strategy extra_buffer) ;;
      B) armflags=(--mamba-radix-cache-strategy extra_buffer) ;;
      C) armflags=(--enable-mixed-chunk --disable-radix-cache) ;;
      *) echo "start_server: bad arm $arm" >&2; return 1 ;;
    esac
    case "$graph" in
      eager) graphflags=(--disable-prefill-cuda-graph --disable-decode-cuda-graph) ;;
      graph) graphflags=(--cuda-graph-backend-prefill breakable) ;;
      *) echo "start_server: bad graph mode $graph" >&2; return 1 ;;
    esac
    cmd=(env SGLANG_MAMBA_DEBUG_ASSERTS=1 python3 -m sglang.launch_server
         --model-path "$MODEL_PATH" --trust-remote-code --tp-size 1 --port "$PORT" --host 127.0.0.1
         --mem-fraction-static 0.80 --attention-backend triton --linear-attn-backend triton
         --mamba-backend triton --sampling-backend pytorch --mamba-ssm-dtype float32
         --chunked-prefill-size "$chunk" --max-running-requests 48 --enable-metrics
         --log-level info --schedule-policy fcfs "${armflags[@]}" "${graphflags[@]}")
  fi

  # Child-side handshake: the child reports its own pid / pgrp / session from
  # inside the new session, so the parent never reads a group id that a
  # not-yet-completed setsid could still leave equal to the caller's.
  local hs="$WS/logs/$logname.handshake"
  rm -f "$hs" "$hs.tmp"
  local setsid_cmd
  read -r -a setsid_cmd <<<"${SETSID_CMD:-setsid}"
  "${setsid_cmd[@]}" env HANDSHAKE_FILE="$hs" bash -c '
    stat=$(cat /proc/$$/stat); rest=${stat##*) }
    read -r _state _ppid pgrp sid _ <<<"$rest"
    printf "%s %s %s\n" "$$" "$pgrp" "$sid" > "$HANDSHAKE_FILE.tmp" && mv "$HANDSHAKE_FILE.tmp" "$HANDSHAKE_FILE"
    exec "$@"' _ "${cmd[@]}" > "$WS/logs/$logname.log" 2>&1 < /dev/null &
  local spawned=$!

  local waited=0 cpid="" pgrp="" sid=""
  while [ ! -s "$hs" ]; do
    if [ "$waited" -ge "$((HANDSHAKE_TIMEOUT_S * 10))" ]; then
      echo "start_server: no handshake from the launched server within ${HANDSHAKE_TIMEOUT_S}s" >&2
      _abort_launch "$spawned"
      return 1
    fi
    sleep 0.1
    waited=$((waited + 1))
  done
  read -r cpid pgrp sid < "$hs"
  if ! [[ "$cpid" =~ ^[0-9]+$ && "$pgrp" =~ ^[0-9]+$ && "$sid" =~ ^[0-9]+$ ]]; then
    echo "start_server: malformed handshake '$cpid $pgrp $sid'" >&2
    _abort_launch "$spawned"
    return 1
  fi
  # A fresh session: the child leads both its group and its session.
  if [ "$pgrp" != "$cpid" ] || [ "$sid" != "$cpid" ] || ! _group_is_foreign "$pgrp"; then
    echo "start_server: launched process is not a session leader of a new group (pid=$cpid pgrp=$pgrp sid=$sid caller pgrp=$_CALLER_PGID); aborting without signaling any group" >&2
    _abort_launch "$cpid"
    return 1
  fi
  if ! kill -0 "$cpid" 2>/dev/null; then
    echo "start_server: launched server (pid $cpid) exited before the health check" >&2
    tail -20 "$WS/logs/$logname.log" >&2
    return 1
  fi
  SERVER_PID=$cpid
  SERVER_PGID=$pgrp
  echo "$SERVER_PGID" > "$WS/logs/$logname.pgid"

  waited=0
  while [ "$waited" -lt "$HEALTH_TIMEOUT_S" ]; do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "server up after ${waited}s (pid $SERVER_PID, pgid $SERVER_PGID)"
      return 0
    fi
    if ! kill -0 -- "-$SERVER_PGID" 2>/dev/null; then
      echo "start_server: server exited during startup" >&2
      tail -40 "$WS/logs/$logname.log" >&2
      SERVER_PGID=""
      SERVER_PID=""
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "start_server: no /health within ${HEALTH_TIMEOUT_S}s; stopping process group $SERVER_PGID" >&2
  tail -40 "$WS/logs/$logname.log" >&2
  stop_server
  return 1
}

# Success, failure and interruption all tear down the group this invocation started.
trap 'stop_server' EXIT
trap 'stop_server; trap - INT; kill -INT $$' INT
trap 'stop_server; trap - TERM; kill -TERM $$' TERM
