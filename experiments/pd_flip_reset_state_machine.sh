#!/usr/bin/env bash
set -euo pipefail

SUITE_DIR="${1:?usage: pd_flip_reset_state_machine.sh <suite-dir>}"
REPO="${REPO:-/root/sglang}"
DOCKER_DIR="${REPO}/scripts/playground/disaggregation/pd_flip_docker"
ENV_FILE="${DOCKER_DIR}/env.local"

source "${ENV_FILE}"

HOSTS=(192.168.0.42 192.168.0.40 192.168.0.39 192.168.0.41)
SESSIONS=(pd-node0 pd-node1 pd-node2 pd-node3)
ROLES=(prefill prefill decode decode)
URLS=("${NODE0}" "${NODE1}" "${NODE2}" "${NODE3}")

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

log "reset workers start"
for i in "${!HOSTS[@]}"; do
  host="${HOSTS[$i]}"
  session="${SESSIONS[$i]}"
  role="${ROLES[$i]}"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${host}" \
    "if command -v tmux >/dev/null 2>&1; then tmux kill-session -t '${session}' 2>/dev/null || true; fi; if [ -f '${DOCKER_DIR}/${session}.pid' ]; then kill \$(cat '${DOCKER_DIR}/${session}.pid') 2>/dev/null || true; rm -f '${DOCKER_DIR}/${session}.pid'; fi; fuser -k ${PORT}/tcp 2>/dev/null || true; cd '${DOCKER_DIR}'; rm -f worker.resume_state_machine.log; if command -v tmux >/dev/null 2>&1; then tmux new -d -s '${session}' 'cd ${DOCKER_DIR}; ENV_FILE=${ENV_FILE} ENABLE_PD_FLIP_STATE_MACHINE=1 ENABLE_PD_RUNTIME_ROLE_SWITCH=1 ./run_worker.sh ${role} 0.0.0.0 2>&1 | tee worker.resume_state_machine.log'; else nohup bash -lc 'cd ${DOCKER_DIR}; ENV_FILE=${ENV_FILE} ENABLE_PD_FLIP_STATE_MACHINE=1 ENABLE_PD_RUNTIME_ROLE_SWITCH=1 ./run_worker.sh ${role} 0.0.0.0' > worker.resume_state_machine.log 2>&1 < /dev/null & echo \$! > '${session}.pid'; fi"
done

deadline=$((SECONDS + 2400))
while (( SECONDS < deadline )); do
  ok=1
  for url in "${URLS[@]}"; do
    curl -fsS "${url}/server_info" >/dev/null 2>&1 || ok=0
  done
  if (( ok == 1 )); then
    log "workers ready"
    break
  fi
  sleep 10
done

log "reset router start"
ssh -o BatchMode=yes -o StrictHostKeyChecking=no 192.168.0.42 \
  "if command -v tmux >/dev/null 2>&1; then tmux kill-session -t pd-router 2>/dev/null || true; fi; if [ -f '${DOCKER_DIR}/pd-router.pid' ]; then kill \$(cat '${DOCKER_DIR}/pd-router.pid') 2>/dev/null || true; rm -f '${DOCKER_DIR}/pd-router.pid'; fi; fuser -k ${ROUTER_PORT}/tcp 2>/dev/null || true; cd '${DOCKER_DIR}'; rm -f router.resume_state_machine.log; if command -v tmux >/dev/null 2>&1; then tmux new -d -s pd-router 'cd ${DOCKER_DIR}; ENV_FILE=${ENV_FILE} ./run_router.sh 2>&1 | tee router.resume_state_machine.log'; else nohup bash -lc 'cd ${DOCKER_DIR}; ENV_FILE=${ENV_FILE} ./run_router.sh' > router.resume_state_machine.log 2>&1 < /dev/null & echo \$! > pd-router.pid; fi"

deadline=$((SECONDS + 900))
while (( SECONDS < deadline )); do
  if curl -fsS "http://127.0.0.1:${ROUTER_PORT}/pd_flip/router/workers" >/dev/null 2>&1; then
    log "router ready"
    break
  fi
  sleep 5
done

curl -fsS "http://127.0.0.1:${ROUTER_PORT}/pd_flip/router/workers" \
  > "${SUITE_DIR}/roles_after_manual_reset.json"
log "reset complete"
