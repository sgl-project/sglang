# Progressive four-node PD flip runbook

This runbook is the deployment contract for a progressive `1P3D -> 2P2D`
Mooncake/HiCache experiment. Run it on four physical 8-GPU nodes at the same
SGLang commit. Do not treat a controller JSON result alone as success: worker
role, worker active event loop, router role, ownership, and client output must
all agree.

## Verification status of this runbook change

The contract and local source regressions can be run without a cluster, but the
four-node acceptance below still requires external infrastructure. On the
Windows authoring host used for this change (2026-07-13), the Docker CLI was
installed but its Linux daemon was not running, only one 8 GiB GPU was present,
and no worker/router listened on ports 30000/8000. Therefore the real four-node
Mooncake run, all three stitch modes, and both SLO decision paths were **not
executed on that host**. No experimental result or artifact was fabricated;
operators must complete every acceptance check below on the target cluster.

## Topology and invariants

Initial topology:

```text
node0: prefill
node1: decode
node2: decode source selected for D-to-P
node3: decode target
```

The successful commit topology is `node0/node2 prefill, node1/node3 decode`.
Node2 is the only role-flip source. Node3 temporarily owns migrated decode
requests and remains decode. A recovery run keeps the initial `1P3D` topology.

The progressive controller migrates a capacity-safe first batch, observes only
post-first-migration SLO samples for 10 seconds, then takes one of two paths:

- **SLO recovery without role flip:** keep the activated first batch on node3,
  resume node2 decode admission, and leave all runtime/router roles unchanged.
- **persistent prefill risk with successful D-to-P:** migrate the remaining
  atomic batch, activate it on node3, change node2's worker role and active
  event loop to prefill, then update the router and resume admission.

A source or target must never activate a strict subset of the journaled batch.
On zero hit, target KV capacity must cover the full committed prefix before
migration starts. Output relay accepts only sequences above the last sequence
emitted by node2; duplicate or missing output is a failed run.

## Configuration

On every node, create a private local configuration (never commit its secret):

```bash
cd scripts/playground/disaggregation/pd_flip_docker
cp env.example env.local
chmod 600 env.local
vi env.local
export ENV_FILE="$PWD/env.local"
# task11-clean-shell-smoke-begin
set -a
source "$ENV_FILE"
set +a
case "$ADMIN_API_KEY" in
  ""|replace-with-*|changeme|CHANGE_ME) echo "unsafe ADMIN_API_KEY" >&2; exit 2 ;;
esac
: "${SGLANG_REPO:?}" "${MODEL_PATH:?}" "${NODE0:?}" "${NODE1:?}" "${NODE2:?}" "${NODE3:?}"
[[ "$MOONCAKE_GLOBAL_SEGMENT_SIZE" == "0" ]] || {
  echo "workers must use the dedicated store (MOONCAKE_GLOBAL_SEGMENT_SIZE=0)" >&2
  exit 2
}
# task11-clean-shell-smoke-end
```

Set the same `MODEL_PATH`, `MODEL_ID`, `TOKENIZER_PATH`, node URLs, Mooncake
addresses, TP/DP sizes, and `ADMIN_API_KEY` everywhere. Keep
`TP_SIZE * DP_SIZE == 8`. The checked-in progressive defaults are:

```text
PD_FLIP_FIRST_MIGRATION_RATIO=0.5
PD_FLIP_OBSERVATION_SECONDS=10
PD_FLIP_SLO_THRESHOLD=0.9
PD_FLIP_MIN_PREFILL_SLO_SAMPLES=20
PD_FLIP_MIN_DECODE_SLO_SAMPLES=20
HICACHE_WRITE_POLICY=write_through
```

All workers are launched with the Mooncake transfer backend, state machine,
runtime role switching, decode radix cache, hierarchical cache, Mooncake
HiCache storage, `write_through`, prefix stitching, and an admin API key. The
controller forwards that key as a Bearer token and uses the same ratio,
observation, SLO sample, and durable session-journal settings. Do not put the
real key in `EXTRA_SGLANG_ARGS` or a checked-in file.
`PD_FLIP_ROUTER_ADMIN_API_KEY` defaults to `ADMIN_API_KEY` and must currently
match it because the controller uses one credential for router and workers.

## Mooncake prerequisites

Before workers start, verify RDMA routing/device names on all nodes and start
one reachable metadata service, one master, and at least one store segment.
The following commands run on the service host; replace addresses and device
names in `env.local` first:

```bash
python3 -m mooncake.http_metadata_server >mooncake-metadata.log 2>&1 &
mooncake_master --eviction_high_watermark_ratio=0.95 >mooncake-master.log 2>&1 &
MOONCAKE_LOCAL_HOSTNAME=10.0.0.10 \
MOONCAKE_TE_META_DATA_SERVER=http://10.0.0.10:8080/metadata \
MOONCAKE_MASTER=10.0.0.10:50051 \
MOONCAKE_PROTOCOL=rdma MOONCAKE_DEVICE=mlx5_0 \
MOONCAKE_GLOBAL_SEGMENT_SIZE=4gb MOONCAKE_LOCAL_BUFFER_SIZE=0 \
python3 -m mooncake.mooncake_store_service --port=8081 \
  >mooncake-store.log 2>&1 &
```

Preflight must show metadata HTTP, master RPC, and store ports reachable from
every worker. Master/store logs must show the segment registered before launch:

```bash
curl --fail --silent "${MOONCAKE_TE_META_DATA_SERVER%/metadata}/metadata" >/dev/null
timeout 3 bash -c "</dev/tcp/${MOONCAKE_MASTER%:*}/${MOONCAKE_MASTER##*:}"
timeout 3 bash -c "</dev/tcp/${MOONCAKE_MASTER%:*}/${MOONCAKE_STORE_PORT}"
ibv_devices
```

This runbook uses one topology only: the dedicated store contributes the 4 GiB
segment shown above and every worker has `MOONCAKE_GLOBAL_SEGMENT_SIZE=0`.
Preflight fails if a worker advertises a contributed segment or the dedicated
store is absent. HiCache must stay `write_through`: the hit-mode acceptance
checks below depend on completed prefix publication, not eventual write-back.

## Start and health checks

Start workers in this exact order, one command on each host:

```bash
# node0
ENV_FILE=$PWD/env.local "$SGLANG_REPO/scripts/playground/disaggregation/pd_flip_docker/run_worker.sh" prefill 0.0.0.0 |& tee node0-worker.log
# node1
ENV_FILE=$PWD/env.local "$SGLANG_REPO/scripts/playground/disaggregation/pd_flip_docker/run_worker.sh" decode 0.0.0.0 |& tee node1-worker.log
# node2 (D-to-P source)
ENV_FILE=$PWD/env.local "$SGLANG_REPO/scripts/playground/disaggregation/pd_flip_docker/run_worker.sh" decode 0.0.0.0 |& tee node2-worker.log
# node3 (migration target)
ENV_FILE=$PWD/env.local "$SGLANG_REPO/scripts/playground/disaggregation/pd_flip_docker/run_worker.sh" decode 0.0.0.0 |& tee node3-worker.log
```

Then start the router and controller-side monitor:

```bash
ENV_FILE=$PWD/env.local "$SGLANG_REPO/scripts/playground/disaggregation/pd_flip_docker/run_router.sh" |& tee router.log
ENV_FILE=$PWD/env.local "$SGLANG_REPO/scripts/playground/disaggregation/pd_flip_docker/run_controller.sh" metrics | tee controller-metrics.json
```

Use the admin credential on every worker control/status call:

```bash
AUTH=(-H "Authorization: Bearer ${ADMIN_API_KEY}")
ROUTER_AUTH=(-H "Authorization: Bearer ${PD_FLIP_ROUTER_ADMIN_API_KEY:-$ADMIN_API_KEY}")
for url in "$NODE0" "$NODE1" "$NODE2" "$NODE3"; do
  curl --fail --silent "${AUTH[@]}" "$url/health" >/dev/null
  curl --fail --silent "${AUTH[@]}" "$url/pd_flip/runtime_role/status" | jq -e \
    'all((if type == "array" then .[] else . end);
      .success == true and .status.runtime_role_switch_enabled == true and
      .status.event_loop_dynamic == true and
      .status.role == .status.active_event_loop_role)'
  curl --fail --silent "${AUTH[@]}" "$url/pd_flip/migration/status" | jq .
done
curl --fail --silent "${ROUTER_AUTH[@]}" \
  "http://${ROUTER_HOST}:${ROUTER_PORT}/pd_flip/router/workers" | jq .
```

Before traffic, assert roles rather than inspecting them visually:

```bash
for item in "$NODE0 prefill" "$NODE1 decode" "$NODE2 decode" "$NODE3 decode"; do
  set -- $item
  curl --fail --silent "${AUTH[@]}" "$1/pd_flip/runtime_role/status" |
    jq -e --arg role "$2" 'all((if type == "array" then .[] else . end);
      .status.role == $role and .status.active_event_loop_role == $role)'
done
DIRECTION=d_to_p SOURCE_NAME=node2 "$SGLANG_REPO/scripts/playground/disaggregation/pd_flip_docker/run_controller.sh" dry-run |
  tee "${SGLANG_REPO}/pd-flip-artifacts/dry-run.json"
```

The dry-run must select node2 and node3, show adequate request slots and KV
tokens, and contain no POST error. Do not execute if worker/controller/router
names or bootstrap ports disagree.

## Running the two decision paths

Use a mixed trace containing short prefill-heavy prompts and long streaming
decode requests. Save request IDs, prompt kind, arrival offset, HTTP status,
TTFT, per-token latency, complete token/text output, and output sequence. Start
the measurement sidecar before the controller:

```bash
cd "$SGLANG_REPO"
PATH_KIND=${PATH_KIND:-recovery} # recovery or commit
MODE=${MODE:-full}               # full, partial, or zero
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${PATH_KIND}"
HOST_ARTIFACT_DIR="${SGLANG_REPO}/pd-flip-artifacts/${RUN_ID}"
CONTAINER_ARTIFACT_DIR="/sgl-workspace/sglang/pd-flip-artifacts/${RUN_ID}"
mkdir -p "$HOST_ARTIFACT_DIR"
set -a; source "$ENV_FILE"; set +a
cp "$ENV_FILE" "$HOST_ARTIFACT_DIR/env.resolved.private"
git rev-parse HEAD >"$HOST_ARTIFACT_DIR/git-commit.txt"
docker image inspect "$IMAGE" --format '{{json .RepoDigests}}' >"$HOST_ARTIFACT_DIR/image-digests.json"
ADMIN_API_KEY="$ADMIN_API_KEY" \
ROUTER_ADMIN_API_KEY="${PD_FLIP_ROUTER_ADMIN_API_KEY:-$ADMIN_API_KEY}" \
EVENTS="$HOST_ARTIFACT_DIR/migration_events.jsonl" \
ROUTER_URL="http://${ROUTER_HOST}:${ROUTER_PORT}" \
NODE0="$NODE0" NODE1="$NODE1" NODE2="$NODE2" NODE3="$NODE3" python3 - <<'PY' &
import json, os, time, urllib.request
from scripts.playground.disaggregation import pd_flip_migration_measure as m
class AuthClient(m.HttpClient):
    def __init__(self, key, timeout_seconds=3):
        super().__init__(timeout_seconds=timeout_seconds)
        self.key = key
    def get_json(self, url):
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + self.key})
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
worker_client = AuthClient(os.environ["ADMIN_API_KEY"])
router_client = AuthClient(os.environ["ROUTER_ADMIN_API_KEY"])
nodes = [{"name": "node%d" % i, "worker_url": os.environ["NODE%d" % i]} for i in range(4)]
names = {node["worker_url"].rstrip("/"): node["name"] for node in nodes}
deadline = time.monotonic() + 420
with open(os.environ["EVENTS"], "w", encoding="utf-8") as output:
    while time.monotonic() < deadline:
        started = time.monotonic()
        m.write_event(output, m.collect_router_event(router_client, os.environ["ROUTER_URL"], names))
        for node in nodes:
            for event in m.collect_worker_events(worker_client, node):
                m.write_event(output, event)
        time.sleep(max(0, 0.25 - (time.monotonic() - started)))
PY
MEASURE_PID=$!
READY_MARKER="$HOST_ARTIFACT_DIR/workload.ready.json"
CONTROLLER_DONE_MARKER="$HOST_ARTIFACT_DIR/controller.done"
python3 "$SGLANG_REPO/scripts/playground/disaggregation/pd_flip_progressive_workload.py" \
  --base-url "http://${ROUTER_HOST}:${ROUTER_PORT}" \
  --source-url "$NODE2" --target-url "$NODE3" \
  --admin-api-key-env ADMIN_API_KEY --model "$MODEL_ID" \
  --ready-marker "$READY_MARKER" \
  --controller-done-marker "$CONTROLLER_DONE_MARKER" \
  --mode "$MODE" --decision-path "$PATH_KIND" \
  --output-dir "$HOST_ARTIFACT_DIR" \
  --ttft-slo-seconds "$TTFT_SLO_SECONDS" --tpot-slo-seconds "$TPOT_SLO_SECONDS" &
WORKLOAD_PID=$!
until test -s "$READY_MARKER"; do sleep 0.1; done
```

The producer's `recovery` profile stops short-request pressure early; its
`commit` profile continues pressure past the observation window. These are
inputs, not proof of the decision. Run the controller and retain stderr/stdout:

```bash
PD_FLIP_ARTIFACT_DIR="$CONTAINER_ARTIFACT_DIR" \
PD_FLIP_MONITOR_ITERATIONS=120 "$SGLANG_REPO/scripts/playground/disaggregation/pd_flip_docker/run_controller.sh" monitor |& \
  tee "$HOST_ARTIFACT_DIR/controller-monitor.log"
CONTROLLER_EXIT=${PIPESTATUS[0]}
touch "$CONTROLLER_DONE_MARKER"
wait "$WORKLOAD_PID"; WORKLOAD_EXIT=$?
test "$CONTROLLER_EXIT" -eq 0
test "$WORKLOAD_EXIT" -eq 0
```

Execute all six input/path combinations with the checked-in matrix driver. It
refuses an empty reset command, waits for the dedicated store health endpoint,
starts measurement and workload in the background, waits for the workload RID
ready marker, runs the controller, releases the workload with the done marker,
and summarizes every case. The five command/endpoint variables may use `MODE`,
`DECISION_PATH`, `HOST_CASE_DIR`, and `CONTAINER_CASE_DIR` from the driver;
controller paths are container-visible while collection/summary paths are host-visible:

```bash
STORE_HOST=cloud-099
REMOTE_ENV_FILE=/root/sglang/scripts/playground/disaggregation/pd_flip_docker/env.local
PD_FLIP_STORE_READY_URL=http://10.0.0.10:18081/generation
PD_FLIP_RESET_STORE_CMD='"$SGLANG_REPO/scripts/playground/disaggregation/pd_flip_docker/reset_store_remote.sh"'
PD_FLIP_MEASURE_COMMAND='python3 "$SGLANG_REPO/scripts/playground/disaggregation/pd_flip_migration_measure.py" sample \
  --router-url "http://${ROUTER_HOST}:${ROUTER_PORT}" \
  --node "name=node0,worker_url=$NODE0" --node "name=node1,worker_url=$NODE1" \
  --node "name=node2,worker_url=$NODE2" --node "name=node3,worker_url=$NODE3" \
  --api-key-env ADMIN_API_KEY --router-api-key-env PD_FLIP_ROUTER_ADMIN_API_KEY \
  --duration-seconds 3600 --output-events "$MIGRATION_EVENTS"'
PD_FLIP_CONTROLLER_COMMAND='PD_FLIP_ARTIFACT_DIR="$CONTAINER_CASE_DIR" \
  PD_FLIP_SESSION_JOURNAL_PATH="$CONTAINER_CASE_DIR/pd_flip_session.json" \
  PD_FLIP_SESSION_ID_PREFIX="$PD_FLIP_SESSION_ID_PREFIX" \
  "$SGLANG_REPO/scripts/playground/disaggregation/pd_flip_docker/run_controller.sh" monitor'
PD_FLIP_SUMMARIZE_COMMAND='python3 "$SGLANG_REPO/scripts/playground/disaggregation/pd_flip_migration_measure.py" summarize \
  --events-jsonl "$MIGRATION_EVENTS" --controller-log "$HOST_CASE_DIR/controller.log" \
  --request-metrics-jsonl "$HOST_CASE_DIR/request_metrics.jsonl" \
  --errors-jsonl "$HOST_CASE_DIR/errors.jsonl" --output-dir "$HOST_CASE_DIR/summary"'
export STORE_HOST REMOTE_ENV_FILE PD_FLIP_STORE_READY_URL PD_FLIP_RESET_STORE_CMD
export PD_FLIP_MEASURE_COMMAND PD_FLIP_CONTROLLER_COMMAND PD_FLIP_SUMMARIZE_COMMAND

python3 "$SGLANG_REPO/scripts/playground/disaggregation/pd_flip_progressive_matrix.py" \
  --base-url "http://${ROUTER_HOST}:${ROUTER_PORT}" \
  --router-url "http://${ROUTER_HOST}:${ROUTER_PORT}" --prefill-url "$NODE0" \
  --other-decode-url "$NODE1" \
  --source-url "$NODE2" --target-url "$NODE3" --model "$MODEL_ID" \
  --admin-api-key-env ADMIN_API_KEY \
  --router-admin-api-key-env PD_FLIP_ROUTER_ADMIN_API_KEY \
  --output-root "$SGLANG_REPO/pd-flip-artifacts/progressive-matrix" \
  --reset-store-cmd "$PD_FLIP_RESET_STORE_CMD" \
  --store-ready-url "$PD_FLIP_STORE_READY_URL" \
  --measure-command "$PD_FLIP_MEASURE_COMMAND" \
  --controller-command "$PD_FLIP_CONTROLLER_COMMAND" \
  --summarize-command "$PD_FLIP_SUMMARIZE_COMMAND"
```

Each case uses `pd-matrix-<mode>-<path>-<case-token>` as its controller session
prefix; `-first` and `-final` sessions therefore cannot reuse manifests from a
previous case. After workload completion the matrix drains and idles all decode
workers, restores node1/node2/node3 to active-loop `decode`, restores router
roles/drain state and admission, then requires the next case to start at 1P3D.

For **SLO recovery without role flip**, accept only when prefill pressure first
causes a first migration, post-first-migration samples recover, and:

Accept recovery only if the first-batch journal reaches `target_active`, node2
remains decode with admission resumed, the first batch remains owned/active on
node3, no runtime/router role changes occur, and all four
`role == active_event_loop_role` checks still match `1P3D`.

For **persistent prefill risk with successful D-to-P**, use
`PATH_KIND=commit`; accept only when the post-first-migration samples remain
risky while decode attainment remains healthy.

Accept commit only if the journal ends `target_active`, node3 owns every
journaled request exactly once, node2 has no migrated residual request, and:

```bash
for item in "$NODE0 prefill" "$NODE1 decode" "$NODE2 prefill" "$NODE3 decode"; do
  set -- $item
  curl --fail --silent "${AUTH[@]}" "$1/pd_flip/runtime_role/status" |
    jq -e --arg role "$2" 'all((if type == "array" then .[] else . end);
      .status.role == $role and .status.active_event_loop_role == $role)'
done
"$SGLANG_REPO/scripts/playground/disaggregation/pd_flip_docker/run_controller.sh" metrics | jq -e \
  '[.[] | {name, worker_role, router_role}] | length == 4'
```

The metrics output and router worker endpoint must also report final
`node0/node2 prefill, node1/node3 decode`. A role mismatch is not transient
success; abort admission and reconcile it.

## Manufacturing and accepting HiCache modes

Run each mode in an isolated Mooncake namespace/store or clear the test store
between cases. Keep prompts page-aligned where noted and confirm publication
has completed before starting the active long decode request. The producer's
`--mode` selects only a warm-up/input strategy; it cannot force or prove a
cache hit. Accept the mode only from worker `request_measurements.stitch_mode`.

1. `full_prefix_stitch`: send a warm-up request whose prefix is P, wait for
   write-through completion, then start a long request with the same P and a
   unique suffix. Trigger migration while it is decoding. Accept only
   `stitch_mode=full_prefix_stitch`, `H == floor(P/page_size)*page_size`,
   `0 < H <= C0 <= C1`, and source transfer covers only `[H,C0)`.
2. `partial_prefix_stitch`: warm exactly the first K full pages of P, then use
   P plus a never-seen suffix for the active request. Choose
   `0 < K < floor(P/page_size)*page_size`. Accept only
   `stitch_mode=partial_prefix_stitch`, `H == K`, `0 < H < P`, and stitched
   Mooncake plus source ranges cover C0 without overlap or a gap.
3. `source_decode_full_fallback`: use a cryptographically random, never-warmed
   page-aligned prefix in an empty test namespace. Accept only `H == 0`,
   `stitch_mode=source_decode_full_fallback`, target capacity reserved for at
   least C0 before source start, and source bytes/tokens cover `[0,C0)`.

Here P is `p_tokens`, H is `h_tokens`, C0 is `c0_tokens` at the base freeze,
and C1 is `c1_tokens` after delta synchronization. For every mode, require
`final_owner=target`, a nondecreasing C0/C1 boundary, no failed request, no
duplicate output, and exact final output equality with a no-migration control
for deterministic decoding.

Mooncake currently may not expose byte counts in every transport build. This
is an explicitly known limitation: when bytes are unavailable, require
`mooncake_bytes=null` and `mooncake_bytes_available=false`; do not invent zero
bytes and do not fail an otherwise token-verified run solely for this field.

## Status, abort, and reconciliation

Inspect the same `SESSION_ID` on both nodes:

```bash
curl --fail --silent "${AUTH[@]}" "$NODE2/pd_flip/migration/status?session_id=$SESSION_ID" | jq .
curl --fail --silent "${AUTH[@]}" "$NODE3/pd_flip/migration/status?session_id=$SESSION_ID" | jq .
```

Before source finish/target activation, abort target then source with the full
journaled session (never a RID subset):

```bash
curl --fail --silent "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SESSION_ID\",\"reason\":\"operator abort\"}" \
  "$NODE3/pd_flip/migration/target/abort" | jq .
curl --fail --silent "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SESSION_ID\",\"reason\":\"operator abort\"}" \
  "$NODE2/pd_flip/migration/abort" | jq .
```

For a controller interruption, run durable journal reconciliation from the
repo root. The command calls `reconcile_session` and uses the archived journal:

```bash
set -a; source "$SGLANG_REPO/scripts/playground/disaggregation/pd_flip_docker/env.local"; set +a
SESSION_ID="$SESSION_ID" JOURNAL="$HOST_ARTIFACT_DIR/pd_flip_session.json" python3 - <<'PY'
import json, os
from scripts.playground.disaggregation.pd_flip_controller import (
    HttpClient, PDClusterConfig, PDFlipController, PDNode,
)
nodes = [PDNode(f"node{i}", os.environ[f"NODE{i}"], os.environ[f"NODE{i}"], int(os.environ["BOOTSTRAP_PORT"])) for i in range(4)]
config = PDClusterConfig(
    router_url=f"http://{os.environ['ROUTER_HOST']}:{os.environ['ROUTER_PORT']}",
    nodes=nodes, session_journal_path=os.environ["JOURNAL"],
)
result = PDFlipController(config, HttpClient(api_key=os.environ["ADMIN_API_KEY"])).reconcile_session(os.environ["SESSION_ID"])
print(json.dumps(result.__dict__, indent=2, default=str))
raise SystemExit(0 if result.success else 1)
PY
```

If reconciliation reports `abort_incomplete` or operator recovery, keep both
workers drained, preserve the journal/status snapshots, and do not issue manual
role/router changes until ownership is established.

## Artifacts and acceptance schema

Stop the sidecar and summarize only after traffic and final consistency checks:

```bash
kill "$MEASURE_PID"; wait "$MEASURE_PID" || true
python3 "$SGLANG_REPO/scripts/playground/disaggregation/pd_flip_migration_measure.py" summarize \
  --events-jsonl "$HOST_ARTIFACT_DIR/migration_events.jsonl" \
  --controller-log "$HOST_ARTIFACT_DIR/controller-monitor.log" \
  --request-metrics-jsonl "$HOST_ARTIFACT_DIR/request_metrics.jsonl" \
  --errors-jsonl "$HOST_ARTIFACT_DIR/errors.jsonl" \
  --output-dir "$HOST_ARTIFACT_DIR/summary"
tar -C "$(dirname "$HOST_ARTIFACT_DIR")" -czf "$HOST_ARTIFACT_DIR.tar.gz" \
  "$(basename "$HOST_ARTIFACT_DIR")"
```

Each run is a **configuration archive** containing the private resolved env
(share only after redaction), commit, image digest, worker/router/Mooncake logs,
controller output, session journal, exact workload/seed, raw outputs, and
sidecar data. Required summary files include `migration_status_samples.csv`,
`migration_request_samples.jsonl`, `migration_request_samples.csv`,
`router_worker_samples.csv`, `worker_pd_flip_samples.csv`,
`worker_load_samples.csv`, `controller_actions.csv`,
`controller_state_trace.csv`, and `request_impact_by_stage.csv`.

The request schema must include `rid`, P/H/C0/C1 token fields, `stitch_mode`,
`mooncake_bytes`, `mooncake_bytes_available`, source/delta bytes and durations,
held/freeze/commit/activate timestamps, source queue, final owner, output
boundary, and rollback reason. The controller state schema must include SLO
good/total counts, configured/effective ratio, capacity fallback count, state,
reason, source/target, and role before/after. Status samples must retain pending,
transferred, released, failed, held, waiting, index-debug, timing-debug, and
error fields.

A run passes only when all requested modes/path assertions hold, client outputs
contain no missing/duplicate sequence, every request has one final owner, the
session batch is atomic, SLO decisions use post-first-migration samples, and
worker role, active event loop, router role, and controller journal agree.
