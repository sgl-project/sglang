#!/usr/bin/env bash
# OpenHands multi-turn concurrency sweep of Muse Spark Open 816B NVFP4 on one 8 x B300 node (TP8/EP8, DFlash2 draft with
# forced acceptance 4.5, four tokenizer workers behind Granian via --enable-http2). One script, resumable:
#
#   serving environment (setup_server.sh) -> client venv -> dataset -> rungs 128 64 48 32 16 8 4 1 -> results/results.tsv
#
# Each rung boots its own server (launch_server.sh hicache|plain), checks the served shape, warms up with two unrecorded
# conversations, flushes the cache, runs the measured ladder at parallel=cc over 2*cc fresh conversations, and tears the
# server down. HiCache rungs (cc >= 16) are re-run until the client's connections spread across the four tokenizer
# workers (largest worker <= GATE_FACTOR x cc/4, measured 15 s into the ladder). If the final retry is still imbalanced,
# complete that client pass and label it OK_UNBALANCED; continue the remaining rungs. A guard kills server and client when
# host-memory headroom or free disk drops below the limits. Exit 0 when every rung is OK or OK_UNBALANCED.
set -uo pipefail
set +m
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1

MODEL_PATH=${MODEL_PATH:-meta-models/Muse-Spark-Open-816B-A42B-NVFP4}
DRAFT_PATH=${DRAFT_PATH:-RadixArk/Muse-Spark-Open-DFlash2}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-meta-models/Muse-Spark-Open-816B-A42B-NVFP4}
PORT=${PORT:-30100}
SERVER_PYTHON=${SERVER_PYTHON:-python3}
CONCURRENCIES=${CONCURRENCIES:-"128 64 48 32 16 8 4 1"}
HICACHE_MIN_CC=${HICACHE_MIN_CC:-16}
NUM_CONVERSATIONS=${NUM_CONVERSATIONS:-1024}
GATE_FACTOR=${GATE_FACTOR:-1.25}              # 0 disables the split gate
GATE_MAX_ATTEMPTS=${GATE_MAX_ATTEMPTS:-6}
BOOT_TIMEOUT_S=${BOOT_TIMEOUT_S:-3600}        # a fresh JIT cache compiles kernels for ~25 min on the first boot
GUARD_MIN_HEADROOM_GIB=${GUARD_MIN_HEADROOM_GIB:-300}
GUARD_MIN_DISK_GIB=${GUARD_MIN_DISK_GIB:-1927}
HICACHE_GB=${HICACHE_GB:-200}
GUARD_MAX_ALLOCATED_BYTES=${GUARD_MAX_ALLOCATED_BYTES:-1200000000000}
export MODEL_PATH DRAFT_PATH SERVED_MODEL_NAME PORT SERVER_PYTHON HICACHE_GB

TOKENIZER_WORKERS=4; TURNS=13; NUM_MULT=2; WARMUP_CONVS=2; OSL=220; N_GPUS=8
VENV=$HERE/venv; EVALSCOPE=$VENV/bin/evalscope
DATASET=$HERE/datasets/openhand-muse-$NUM_CONVERSATIONS.json
RESULTS=${RESULTS_DIR:-$HERE/results}; TSV=$RESULTS/results.tsv; OWNED=$RESULTS/owned-pgids
URL=http://127.0.0.1:$PORT
CACHE_DIR=${CACHE_DIR:-$HERE/cache}
export CACHE_DIR
export HF_HOME=${HF_HOME:-$CACHE_DIR/huggingface}
GIB=1073741824
SERVER_PID=""; CLIENT_PID=""; GUARD_PID=""; MAIN_PID=$BASHPID; ANY_FAIL=0
export TZ=UTC

say() { echo "=== $(date -u +%FT%TZ) $* ==="; }
die() { say "FATAL: $*"; exit 1; }

# ---------------------------------------------------------------- host metrics
cgroup_headroom_gib() {
    local max pinned
    max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null) && [[ $max =~ ^[0-9]+$ ]] || return 1
    pinned=$(awk -v G=$GIB '/^anon |^shmem |^kernel /{s+=$2; n++} END{if(n!=3) exit 1; printf "%d", s/G}' /sys/fs/cgroup/memory.stat) || return 1
    echo $(( max / GIB - pinned ))
}
headroom_gib() { local c h; h=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo) || return 1; c=$(cgroup_headroom_gib) || c=$h; [ "$c" -lt "$h" ] && echo "$c" || echo "$h"; }
disk_free_gib() { df -B1 --output=avail "$1" | awk 'NR==2{printf "%d", $1/1073741824}'; }
gpus_idle() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000{busy=1} END{exit busy}'; }
wait_gpus_idle() { local i; for i in $(seq 1 36); do gpus_idle && return 0; sleep 5; done; return 1; }
port_free() { ! (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; }

# ---------------------------------------------------------------- process ownership
own() { echo "$1" >> "$OWNED"; }
kill_group() {
    local pid=${1:-} i; [[ $pid =~ ^[1-9][0-9]*$ ]] || return 0
    kill -TERM -- "-$pid" 2>/dev/null
    for i in 1 2 3 4 5 6 7 8; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$pid" 2>/dev/null; return 0
}
teardown() { kill_group "$CLIENT_PID"; kill_group "$SERVER_PID"; CLIENT_PID=""; SERVER_PID=""; : > "$OWNED"; }
kill_owned() { local p; for p in $(cat "$OWNED" 2>/dev/null); do kill -TERM -- "-$p" 2>/dev/null; done; sleep 3; for p in $(cat "$OWNED" 2>/dev/null); do kill -KILL -- "-$p" 2>/dev/null; done; }
cleanup() {
    local rc=$? had_server=$SERVER_PID; trap - EXIT
    [ -n "$GUARD_PID" ] && kill "$GUARD_PID" 2>/dev/null
    teardown
    if [ -n "$had_server" ]; then
        wait_gpus_idle || { say "WARNING: GPUs still busy after teardown"; [ $rc = 0 ] && rc=1; }
    fi
    exit $rc
}
trap cleanup EXIT
trap 'say "interrupted"; exit 130' INT TERM
trap 'ANY_FAIL=1; say "GUARD TRIPPED: stopping the sweep"; exit 75' USR1

guard() {
    local h d allocated tick=0
    while :; do
        if [ $((tick % 30)) = 0 ]; then
            allocated=$(du -sx --block-size=1 "${ALLOCATION_PATHS[@]}" | awk '{s+=$1} END{printf "%.0f", s}') || allocated=$GUARD_MAX_ALLOCATED_BYTES
            if [ "$allocated" -ge "$GUARD_MAX_ALLOCATED_BYTES" ]; then
                echo "$(date -u +%FT%TZ) OWN ALLOCATION GUARD: $allocated bytes" >> "$RESULTS/guard.log"
                touch "$RESULTS/GUARD_TRIPPED"; kill_owned; kill -USR1 "$MAIN_PID"; exit 1
            fi
        fi
        tick=$((tick + 1))
        h=$(headroom_gib) || h=0; d=$(for p in "${ALLOCATION_PATHS[@]}"; do disk_free_gib "$p" || exit 1; done | sort -n | head -1) || d=0
        if [ "$h" -lt "$GUARD_MIN_HEADROOM_GIB" ] || [ "$d" -lt "$GUARD_MIN_DISK_GIB" ]; then
            echo "$(date -u +%FT%TZ) GUARD TRIPPED headroom=${h}GiB disk=${d}GiB: killing server and client" | tee -a "$RESULTS/guard.log" >&2
            touch "$RESULTS/GUARD_TRIPPED"; kill_owned; kill -USR1 "$MAIN_PID"; exit 1
        fi
        sleep 2
    done
}

# ---------------------------------------------------------------- environment, client, dataset
ensure_client() {
    "$VENV/bin/python" -c "import evalscope.perf.plugin.datasets.swe_smith" 2>/dev/null && return 0
    say "installing the evalscope client into $VENV (pinned, no dependency resolution)"
    rm -rf "$VENV"
    python3 -m venv "$VENV" && "$VENV/bin/pip" install --no-deps -r "$HERE/client-requirements.txt" || return 1
    "$VENV/bin/python" -c "import evalscope.perf.plugin.datasets.swe_smith"
}
ensure_dataset() {
    [ -s "$DATASET" ] && return 0
    say "building the dataset: $NUM_CONVERSATIONS conversations x $TURNS turns with the $MODEL_PATH tokenizer (one-time; streams two HF datasets)"
    mkdir -p "$HERE/datasets"
    env -u HF_HUB_OFFLINE "$VENV/bin/python" "$HERE/build_openhands_padded_dataset.py" --model "$MODEL_PATH" --pad-source openscience \
        --first-turn-length 74160 --subsequent-turn-length 753 --num-turns "$TURNS" --number "$NUM_CONVERSATIONS" \
        --output-path "$DATASET.partial" || return 1
    mv "$DATASET.partial" "$DATASET"
}
dataset_conversations() {
    "$VENV/bin/python" - "$DATASET" "$TURNS" <<'PY'
import json, sys
convs = json.load(open(sys.argv[1]))["conversations"]; turns = int(sys.argv[2])
assert all(len(c) == turns and all(t.get("messages") for t in c) for c in convs), "malformed conversation"
print(len(convs))
PY
}

# ---------------------------------------------------------------- server and client per rung
start_server() {   # $1 hicache|plain  $2 cc  $3 rung dir
    setsid bash "$HERE/launch_server.sh" "$1" "$2" > "$3/server.log" 2>&1 < /dev/null &
    SERVER_PID=$!; own "$SERVER_PID"
}
wait_health() {   # $1 server log
    local i
    for i in $(seq 1 $(( BOOT_TIMEOUT_S / 10 ))); do
        curl -fsS --max-time 5 "$URL/health" >/dev/null 2>&1 && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "server exited during boot; tail of $1:"; tail -n 8 "$1"; return 1; }
        if tr '\r' '\n' < "$1" | grep -av "Warning: Target model" | grep -aqE "CUDA error|Scheduler hit an exception|RuntimeError|ValueError|out of memory|Traceback"; then
            echo "server log reports an error during boot:"; tr '\r' '\n' < "$1" | grep -aE "CUDA error|Scheduler hit an exception|RuntimeError|ValueError|out of memory" | head -n 3; return 1
        fi
        [ -f "$RESULTS/GUARD_TRIPPED" ] && return 1
        sleep 10
    done
    echo "server not healthy after ${BOOT_TIMEOUT_S}s"; return 1
}
verify_shape() {   # $1 server-info.json  $2 cc  $3 hicache gb
    "$VENV/bin/python" - "$@" "$MODEL_PATH" "$DRAFT_PATH" "$SERVER_PID" <<'PY'
import json, sys
info, cc, hicache, model, draft = json.load(open(sys.argv[1])), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
info = info.get("server_args", info)
from pathlib import Path
raw = dict(x.split(b"=",1) for x in Path(f"/proc/{sys.argv[6]}/environ").read_bytes().split(b"\0") if b"=" in x)
expected_sim = {"SGLANG_SIMULATE_ACC_LEN":"4.5", "SGLANG_SIMULATE_ACC_METHOD":"match-expected", "SGLANG_SIMULATE_ACC_TOKEN_MODE":"real-draft-token"}
expected_sim["SGLANG_SIMULATE_ACC_TOKEN_MODE"] = __import__("os").environ.get("SIM_ACC_TOKEN_MODE", "real-draft-token")
actual_sim = {k.decode():v.decode() for k,v in raw.items() if k.startswith(b"SGLANG_SIMULATE_")}
assert actual_sim == expected_sim, actual_sim
Path(sys.argv[1]).with_name("simulation-env.json").write_text(json.dumps(actual_sim,indent=2)+"\n")
want = {"tp_size": 8, "ep_size": 8, "max_running_requests": cc, "cuda_graph_max_bs_decode": cc, "context_length": 131072,
        "quantization": "modelopt_fp4", "moe_runner_backend": "flashinfer_trtllm", "attention_backend": "trtllm_mha",
        "page_size": 128, "mem_fraction_static": 0.9, "enable_layernorm_sp": True, "enable_megamoe_for_layernorm_sp": True,
        "speculative_algorithm": "DFLASH", "speculative_num_draft_tokens": 8, "speculative_use_block_verification": False, "kv_cache_dtype": "fp8_e4m3", "speculative_draft_attention_backend": "fa4",
        "speculative_draft_model_path": draft, "model_path": model, "tokenizer_worker_num": 4, "enable_http2": True,
        "enable_hierarchical_cache": hicache > 0, "enable_cache_report": True, "sampling_defaults": "model"}
if hicache:
    want["hicache_size"] = hicache
bad = {k: (info.get(k), v) for k, v in want.items() if info.get(k) != v}
print("served:", {k: info.get(k) for k in ("kv_cache_dtype", "max_total_num_tokens", "chunked_prefill_size", "version")})
if bad:
    print("MISMATCH (got, wanted):", bad)
sys.exit(1 if bad else 0)
PY
}
start_client() {   # $1 name  $2 parallel  $3 number  $4 dataset offset  $5 out dir  $6 log
    setsid "$EVALSCOPE" perf --model "$SERVED_MODEL_NAME" --url "$URL/v1/chat/completions" --api openai \
        --dataset swe_smith --dataset-path "$DATASET" --max-tokens "$OSL" --multi-turn --temperature "$TEMPERATURE" \
        --number "$3" --parallel "$2" --dataset-offset "$4" --extra-args "$EXTRA_ARGS" \
        --connect-timeout 600 --read-timeout 3600 --total-timeout 21600 \
        --name "$1" --outputs-dir "$5" --no-timestamp > "$6" 2>&1 < /dev/null &
    CLIENT_PID=$!; own "$CLIENT_PID"
}
wait_client() { local rc=0; wait "$CLIENT_PID" || rc=$?; CLIENT_PID=""; return $rc; }
check_summary() {   # $1 evalscope out dir  $2 expected requests -> OK | INCOMPLETE(...) | NO_SUMMARY
    "$VENV/bin/python" - "$1" "$2" <<'PY'
import glob, json, sys
files = glob.glob(f"{sys.argv[1]}/parallel_*/benchmark_summary.json")
if len(files) != 1:
    print("NO_SUMMARY"); sys.exit(0)
s = json.load(open(files[0])); expected = int(sys.argv[2])
ok = s.get("Total Requests") == expected and s.get("Failed Requests", 0) == 0
print("OK" if ok else f"INCOMPLETE(total={s.get('Total Requests')},failed={s.get('Failed Requests')},expected={expected})")
PY
}
gate_limit() { python3 -c "import math; print(math.ceil($1 / $TOKENIZER_WORKERS * $GATE_FACTOR))"; }

run_rung() {   # $1 cc
    local cc=$1 mode hicache gated=0 limit attempt unbalanced=0 rd status="" split="n/a" boot_s="" ladder_s="" t0 rc
    rd=$RESULTS/c$cc
    case "$(cat "$rd/status" 2>/dev/null)" in OK|OK_UNBALANCED) say "c=$cc already complete in $rd, skipping"; return 0;; esac
    if [ "$cc" -ge "$HICACHE_MIN_CC" ]; then mode=hicache; hicache=$HICACHE_GB; else mode=plain; hicache=0; fi
    [ "$mode" = hicache ] && awk -v f="$GATE_FACTOR" 'BEGIN{exit !(f>0)}' && gated=1
    limit=$(gate_limit "$cc")
    for attempt in $(seq 1 "$GATE_MAX_ATTEMPTS"); do
        if [ -d "$rd" ]; then mkdir -p "$RESULTS/preserved"; mv "$rd" "$RESULTS/preserved/c$cc-before-a$attempt-$(date +%s%N)"; fi; mkdir -p "$rd"; : > "$OWNED"
        say "c=$cc attempt $attempt: $mode server (HiCache $hicache GB/rank), $(( NUM_MULT * cc )) conversations, $(( TURNS * NUM_MULT * cc )) requests$([ $gated = 1 ] && echo ", gate: largest worker <= $limit connections")"
        t0=$(date +%s)
        start_server "$mode" "$cc" "$rd"
        wait_health "$rd/server.log" || { status=BOOT_FAIL; break; }
        boot_s=$(( $(date +%s) - t0 ))
        curl -fsS --max-time 30 "$URL/get_server_info" > "$rd/server-info.json" || { status=VERIFY_FAIL; break; }
        verify_shape "$rd/server-info.json" "$cc" "$hicache" || { status=VERIFY_FAIL; break; }
        say "c=$cc booted in ${boot_s}s; warm-up: $WARMUP_CONVS conversations at c=1 from the dataset tail"
        start_client warmup 1 "$WARMUP_CONVS" $(( NCONV - WARMUP_CONVS )) "$rd" "$rd/warmup.log"
        wait_client && [ "$(check_summary "$rd/warmup" $(( TURNS * WARMUP_CONVS )))" = OK ] || { tail -n 5 "$rd/warmup.log"; status=WARMUP_FAIL; break; }
        curl -fsS --max-time 30 "$URL/get_server_info" > "$rd/server-info-warmup.json" || { status=VERIFY_FAIL; break; }
        "$VENV/bin/python" - "$rd/server-info-warmup.json" <<'PYAL' || { status=SIMULATION_VERIFY_FAIL; break; }
import json,sys
s=json.load(open(sys.argv[1]));states=s.get("internal_states") or []
a=states[0].get("avg_spec_accept_length") if states else None
assert a is not None and 4.3 <= float(a) <= 4.7, ("Expected observed simulated AL near4.5 after warmup",a)
print("Observed warmup simulated acceptance:",a)
PYAL
        curl -fsS --max-time 120 -X POST "$URL/flush_cache" > /dev/null || { status=FLUSH_FAIL; break; }
        sleep 3
        say "c=$cc measured ladder: parallel=$cc number=$(( NUM_MULT * cc ))"
        t0=$(date +%s)
        start_client "c$cc" "$cc" $(( NUM_MULT * cc )) 0 "$rd" "$rd/evalscope.log"
        if [ $gated = 1 ]; then
            sleep 15
            split=$(python3 "$HERE/conn_split.py" "$PORT" "$TOKENIZER_WORKERS" 3 3) || { status=GATE_ERROR; break; }
            printf 'c%s\tattempt %s\tsplit %s\tlargest %s\tlimit %s\t' "$cc" "$attempt" "${split##* }" "${split%% *}" "$limit" >> "$RESULTS/gate.log"
            if [ "${split%% *}" -gt "$limit" ]; then
                if [ "$attempt" -lt "$GATE_MAX_ATTEMPTS" ]; then
                    echo rejected >> "$RESULTS/gate.log"
                    say "c=$cc attempt $attempt REJECTED: connection split ${split##* } (largest worker ${split%% *} > $limit); rebooting"
                    teardown; mkdir -p "$RESULTS/rejected"; mv "$rd" "$RESULTS/rejected/c$cc-a$attempt-$(date +%s%N)"
                    wait_gpus_idle || die "GPUs not released after the rejected attempt"
                    continue
                fi
                echo accepted_unbalanced_final_attempt >> "$RESULTS/gate.log"
                unbalanced=1
                printf 'attempt=%s\nconnection_split=%s\nmaximum_per_worker=%s\nreason=balance retries exhausted; complete the final client pass\n' "$attempt" "${split##* }" "$limit" > "$rd/balance-fallback.txt"
                say "c=$cc final attempt $attempt is UNBALANCED (${split##* }, limit $limit); retaining the full pass and continuing the sweep"
            else
                echo accepted >> "$RESULTS/gate.log"
                say "c=$cc connection split ${split##* } accepted (limit $limit)"
            fi
        fi
        wait_client; rc=$?
        ladder_s=$(( $(date +%s) - t0 ))
        status=$(check_summary "$rd/c$cc" $(( TURNS * NUM_MULT * cc )))
        [ $rc = 0 ] || { tail -n 5 "$rd/evalscope.log"; [ "$status" = OK ] && status=CLIENT_EXIT_$rc; }
        if [ "$status" = OK ] && [ "$unbalanced" = 1 ]; then status=OK_UNBALANCED; fi
        curl -fsS --max-time 30 "$URL/get_server_info" > "$rd/server-info-after.json"
        break
    done
    teardown
    wait_gpus_idle || die "GPUs not released after rung c=$cc"
    [ -n "$status" ] || status=GATE_EXHAUSTED
    mkdir -p "$rd"
    "$VENV/bin/python" "$HERE/summarize_rung.py" row "$rd" "$cc" "$hicache" "$status" "$boot_s" "$ladder_s" "$attempt" "${split##* }" >> "$TSV"
    echo "$status" > "$rd/status"
    say "c=$cc done: $status (boot ${boot_s:-?}s, ladder ${ladder_s:-?}s, attempts $attempt)"
    [[ "$status" = OK || "$status" = OK_UNBALANCED ]] || ANY_FAIL=1
}

# ---------------------------------------------------------------- main
mkdir -p "$RESULTS" || exit 1
exec > >(tee -a "$RESULTS/sweep.log") 2>&1
say "SWEEP rungs=[$CONCURRENCIES] hicache=${HICACHE_GB}GB/rank for cc>=$HICACHE_MIN_CC model=$MODEL_PATH draft=$DRAFT_PATH gate=$GATE_FACTOR"
# Locks cover the whole run, including teardown. Use the same lock path for all sweeps on this node.
exec 9>"${GPU_LOCK_FILE:-/tmp/muse-openhands-gpu.lock}"
flock -n 9 || die "another OpenHands sweep owns the GPU lock"
for p in "$MODEL_PATH" "$DRAFT_PATH"; do
    [ -d "$p" ] || die "set MODEL_PATH and DRAFT_PATH to downloaded checkpoint directories"
done
[ -f "$MODEL_PATH/generation_config.json" ] || die "generation_config.json is required"
for value in "$HICACHE_GB" "$GATE_MAX_ATTEMPTS" "$NUM_CONVERSATIONS" $CONCURRENCIES; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "expected a positive integer, got $value"
done
mkdir -p "$CACHE_DIR" "$HF_HOME"
ALLOCATION_PATHS=("$HERE" "$RESULTS" "$MODEL_PATH" "$DRAFT_PATH" "$CACHE_DIR" "$HF_HOME")
gpus_idle || die "GPUs are in use; this script never kills processes it did not start"
port_free || die "port $PORT is busy"
NEED=$(( N_GPUS * HICACHE_GB + 240 + GUARD_MIN_HEADROOM_GIB )); HR=$(headroom_gib)
[ "$HR" -ge "$NEED" ] || die "host memory headroom ${HR} GiB < ${NEED} GiB needed for HiCache ${HICACHE_GB} GB/rank on $N_GPUS ranks"
DFREE=$(disk_free_gib "$HERE"); [ "$DFREE" -ge "$GUARD_MIN_DISK_GIB" ] || die "free disk ${DFREE} GiB < ${GUARD_MIN_DISK_GIB} GiB"
: > "$OWNED"
bash "$HERE/setup_server.sh" || die "serving environment setup failed"
"$SERVER_PYTHON" "$HERE/sampling_defaults.py" "$MODEL_PATH" "$RESULTS/sampling.json" || die "invalid checkpoint sampling defaults"
TEMPERATURE=$("$SERVER_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["request"].get("temperature", 1.0))' "$RESULTS/sampling.json")
EXTRA_ARGS=$("$SERVER_PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1]))["request"]; print(json.dumps(d))' "$RESULTS/sampling.json")
ensure_client || die "client setup failed"
ensure_dataset || die "dataset build failed"
NCONV=$(dataset_conversations) || die "dataset validation failed"
MAXCC=0; for cc in $CONCURRENCIES; do [[ $cc =~ ^[0-9]+$ ]] || die "bad concurrency '$cc'"; [ "$cc" -gt "$MAXCC" ] && MAXCC=$cc; done
[ "$NCONV" -ge $(( NUM_MULT * MAXCC + WARMUP_CONVS )) ] || die "$NCONV conversations < $(( NUM_MULT * MAXCC + WARMUP_CONVS )) needed"
for p in "$MODEL_PATH" "$DRAFT_PATH"; do case $p in /*) [ -d "$p" ] || die "missing $p";; esac; done

{
    echo "started_at=$(date -u +%FT%TZ) host=$(hostname)"
    echo "gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | awk '{$1=$1};1' | paste -sd,)"
    echo "sglang=$("$SERVER_PYTHON" -c 'import sglang; print(sglang.__version__, sglang.__file__)')"
    echo "model=$MODEL_PATH"
    echo "draft=$DRAFT_PATH"
    echo "dataset=$DATASET sha256=$(sha256sum "$DATASET" | cut -d' ' -f1) conversations=$NCONV"
    echo "evalscope=$("$VENV/bin/python" -c 'import importlib.metadata as m, json; d = m.distribution("evalscope"); u = json.loads(d.read_text("direct_url.json") or "{}"); print(d.version, u.get("vcs_info", {}).get("commit_id", ""))')"
    echo "rungs=$CONCURRENCIES hicache_min_cc=$HICACHE_MIN_CC gate_factor=$GATE_FACTOR num_mult=$NUM_MULT warmup_convs=$WARMUP_CONVS osl=$OSL"
    echo "acceptance=SGLANG_SIMULATE_ACC_LEN=4.5 match-expected ${SIM_ACC_TOKEN_MODE:-real-draft-token}; draft block size 8; tokenizer workers $TOKENIZER_WORKERS; --enable-http2"
    echo "preflight=headroom ${HR} GiB, free disk ${DFREE} GiB"
} >> "$RESULTS/run-metadata.txt"
[ -s "$TSV" ] || "$VENV/bin/python" "$HERE/summarize_rung.py" header > "$TSV"
guard & GUARD_PID=$!
for cc in $CONCURRENCIES; do
    kill -0 "$GUARD_PID" 2>/dev/null || die "guard is not running; refusing to continue unguarded"
    run_rung "$cc"
    if [ "$ANY_FAIL" != 0 ]; then say "Stopping after failed rung; evidence preserved"; break; fi
done
echo "finished_at=$(date -u +%FT%TZ) any_fail=$ANY_FAIL" >> "$RESULTS/run-metadata.txt"
say "SWEEP COMPLETE (any_fail=$ANY_FAIL) -> $TSV"
"$VENV/bin/python" "$HERE/summarize_rung.py" table "$TSV"
exit $ANY_FAIL
