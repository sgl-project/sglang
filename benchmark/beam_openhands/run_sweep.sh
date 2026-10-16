#!/usr/bin/env bash
# OpenHands multi-turn concurrency sweep against an already-running SGLang endpoint (BASE_URL in site.env).
#
#   preflight: client pins, tokenizer view, dataset, server profile, run identity
#   per rung (default 1 4 8 16 32):
#     server profile check -> wait idle -> warm-up: 2 conversations at c=1 from the dataset tail -> validate
#     -> wait idle -> POST /flush_cache -> ladder: parallel=cc, number=2*cc from the dataset head -> validate
#     -> results/c<cc>/rung.json and results/results.tsv
#
# Unlike the reference, which boots one server per rung with --max-running-requests set to cc, one server stays up
# for every rung and only the client concurrency changes. This script never starts, stops or signals the server.
# The cache is flushed only after /v1/loads reports the server idle, and the server itself refuses a flush while
# busy.
#
# The client is evalscope perf, run through evalscope_client.py. That wrapper registers one API plugin, which strips
# special-token spellings from the replayed history and keeps the server's usage record; see its docstring.
#
# Resumable. A rung whose status is OK is kept. Any other rung directory is moved to incomplete/ and the rung is
# rerun. Each invocation writes its preflight to attempts/<time>/, and a new run promotes that to the results
# directory. A changed dataset, client, static server configuration or script is refused, so a new RESULTS_DIR is
# needed; the accepted run's files stay as they were.
#
#   SITE_ENV=/path/to/site.env [CONCURRENCIES="128 64 ..."] [RESULTS_DIR=...] bash run_sweep.sh
set -uo pipefail
set +m
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh" || exit 1

CONCURRENCIES=${CONCURRENCIES:-"1 4 8 16 32"}
STOP_ON_FAIL=${STOP_ON_FAIL:-1}
IDLE_TIMEOUT_S=${IDLE_TIMEOUT_S:-600}
LOAD_SAMPLE_S=${LOAD_SAMPLE_S:-5}      # /v1/loads polling during the ladder (foreign-load check); 0 disables
MIN_FREE_GIB=${MIN_FREE_GIB:-20}
CLIENT_PID=""
SAMPLER_PID=""
ANY_FAIL=0
export HF_HUB_OFFLINE=1

# --------------------------------------------------------------------------- own processes only
kill_group() {
    local pid=${1:-} i
    [[ $pid =~ ^[1-9][0-9]*$ ]] || return 0
    kill -TERM -- "-$pid" 2>/dev/null
    for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$pid" 2>/dev/null || return 0; sleep 1; done
    kill -KILL -- "-$pid" 2>/dev/null
    return 0
}
stop_sampler() { kill_group "$SAMPLER_PID"; [ -n "$SAMPLER_PID" ] && wait "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=""; }
cleanup() { local rc=$?; trap - EXIT; stop_sampler; kill_group "$CLIENT_PID"; exit $rc; }
trap cleanup EXIT
trap 'say "interrupted"; exit 130' INT TERM

# --------------------------------------------------------------------------- helpers
tool() { "$PY" "$HERE/sweep_tool.py" "$@"; }
free_gib() { df -B1 --output=avail "$1" | awk 'NR==2{printf "%d", $1/1073741824}'; }
extra_args() {
    "$PY" -c 'import json, sys; k = json.loads(sys.argv[1]); d = {"ignore_eos": True}
if k: d["chat_template_kwargs"] = k
print(json.dumps(d))' "$CHAT_TEMPLATE_KWARGS"
}

start_client() {   # $1 name  $2 parallel  $3 number  $4 dataset offset  $5 outputs dir  $6 log
    # --stream is evalscope's default, spelled out: the replayed history is the streamed content + reasoning_content.
    local cmd=("$PY" "$HERE/evalscope_client.py" --history-tokenizer "$TOKENIZER_VIEW"
        --model "$SERVED_MODEL_NAME" --url "$BASE_URL/v1/chat/completions" --api openai_openhands_replay
        --dataset swe_smith --dataset-path "$DATASET" --max-tokens "$OSL" --multi-turn --stream --temperature 0
        --number "$3" --parallel "$2" --dataset-offset "$4" --extra-args "$EXTRA_ARGS"
        --connect-timeout 600 --read-timeout 3600 --total-timeout 21600
        --name "$1" --outputs-dir "$5" --no-timestamp)
    printf '%q ' "${cmd[@]}" >> "$(dirname "$6")/commands.txt"; echo >> "$(dirname "$6")/commands.txt"
    setsid "${cmd[@]}" > "$6" 2>&1 < /dev/null &
    CLIENT_PID=$!
}
wait_client() { local rc=0; wait "$CLIENT_PID" || rc=$?; CLIENT_PID=""; return $rc; }

validate_run() {   # $1 evalscope outputs dir  $2 offset  $3 number  $4 out  [$5 loads file  $6 cc]
    "$PY" "$HERE/summarize_rung.py" validate --evalscope-dir "$1" --dataset-stats "$RESULTS_DIR/dataset-stats.json" \
        --history-tokenizer "$TOKENIZER_VIEW" --offset "$2" --number "$3" --turns "$TURNS" --osl "$OSL" --context-length "$CONTEXT_LENGTH" \
        ${5:+--loads "$5" --cc "$6"} --out "$4"
}

summarize() {   # $1 cc  $2 rung dir  $3 status -> prints the status row confirms (OK only with complete metrics)
    "$PY" "$HERE/summarize_rung.py" row --rung-dir "$2" --cc "$1" --status "$3" \
        --expected-requests $(( TURNS * NUM_MULT * $1 )) --n-gpus "$N_GPUS" --out "$2/rung.json"
}
finish_rung() {   # $1 cc  $2 rung dir  $3 status
    local status
    status=$(summarize "$1" "$2" "$3") || true
    [ -n "$status" ] || status=SUMMARY_FAIL
    if ! "$PY" "$HERE/summarize_rung.py" table --results-dir "$RESULTS_DIR" > /dev/null; then
        say "c=$1: results.tsv could not be rebuilt"
        if [ "$status" = OK ]; then status=SUMMARY_FAIL; summarize "$1" "$2" "$status" > /dev/null; fi
    fi
    echo "$IDENTITY_SHA" > "$2/identity.sha256"
    echo "$status" > "$2/status"
    say "c=$1 done: $status"
    [ "$status" = OK ] || ANY_FAIL=1
    [ "$status" = OK ]
}

run_rung() {   # $1 cc
    local cc=$1 rd=$RESULTS_DIR/c$1 status="" rc code
    if [ "$(cat "$rd/status" 2>/dev/null)" = OK ] && [ "$(cat "$rd/identity.sha256" 2>/dev/null)" = "$IDENTITY_SHA" ]; then
        say "c=$cc already OK in $rd, keeping it"
        return 0
    fi
    if [ -e "$rd" ]; then
        mkdir -p "$RESULTS_DIR/incomplete"
        mv "$rd" "$RESULTS_DIR/incomplete/c$cc-$(date -u +%Y%m%dT%H%M%SZ)"
        say "c=$cc: moved the previous unfinished attempt to $RESULTS_DIR/incomplete/"
    fi
    mkdir -p "$rd"
    [ "$(free_gib "$RESULTS_DIR")" -ge "$MIN_FREE_GIB" ] || { finish_rung "$cc" "$rd" DISK_LOW; return 1; }

    tool server-check --base-url "$BASE_URL" --profile "$SERVER_PROFILE" --model-path "$MODEL_PATH" \
        --served-model-name "$SERVED_MODEL_NAME" --n-gpus "$N_GPUS" --cc "$cc" --out "$rd/server-info-before.json" \
        || { finish_rung "$cc" "$rd" VERIFY_FAIL; return 1; }
    tool wait-idle --base-url "$BASE_URL" --timeout-s "$IDLE_TIMEOUT_S" || { finish_rung "$cc" "$rd" SERVER_BUSY; return 1; }

    say "c=$cc warm-up: $WARMUP_CONVS conversations at c=1 from the dataset tail (offset $(( NCONV - WARMUP_CONVS )))"
    start_client warmup 1 "$WARMUP_CONVS" $(( NCONV - WARMUP_CONVS )) "$rd/warmup" "$rd/warmup.log"
    wait_client || status=WARMUP_FAIL
    validate_run "$rd/warmup" $(( NCONV - WARMUP_CONVS )) "$WARMUP_CONVS" "$rd/warmup-validation.json" || status=WARMUP_FAIL
    [ -z "$status" ] || { tail -n 5 "$rd/warmup.log"; finish_rung "$cc" "$rd" "$status"; return 1; }

    tool wait-idle --base-url "$BASE_URL" --timeout-s "$IDLE_TIMEOUT_S" || { finish_rung "$cc" "$rd" SERVER_BUSY; return 1; }
    code=$(curl -sS --max-time 120 -X POST -o "$rd/flush.txt" -w '%{http_code}' "$BASE_URL/flush_cache") || code=000
    [ "$code" = 200 ] || { say "flush_cache returned $code: $(cat "$rd/flush.txt" 2>/dev/null)"; finish_rung "$cc" "$rd" FLUSH_FAIL; return 1; }
    sleep 3

    say "c=$cc ladder: parallel=$cc number=$(( NUM_MULT * cc )) ($(( TURNS * NUM_MULT * cc )) requests)"
    if [ "$LOAD_SAMPLE_S" != 0 ]; then
        # Its own process group, without the lock fd or the log pipe, so stopping it cannot leave either held.
        setsid "$PY" "$HERE/sweep_tool.py" sample-loads --base-url "$BASE_URL" --interval-s "$LOAD_SAMPLE_S" \
            --out "$rd/loads.jsonl" < /dev/null > /dev/null 2>> "$rd/sampler.err" 9>&- &
        SAMPLER_PID=$!
    fi
    start_client "c$cc" "$cc" $(( NUM_MULT * cc )) 0 "$rd/ladder" "$rd/evalscope.log"
    wait_client; rc=$?
    stop_sampler
    curl -fsS --max-time 30 "$BASE_URL/server_info" > "$rd/server-info-after.json" 2>/dev/null
    if validate_run "$rd/ladder" 0 $(( NUM_MULT * cc )) "$rd/validation.json" "$rd/loads.jsonl" "$cc"; then
        status=OK
    else
        status=INVALID
    fi
    [ "$rc" = 0 ] || { tail -n 5 "$rd/evalscope.log"; [ "$status" = OK ] && status=CLIENT_EXIT_$rc; }
    curl -fsS --max-time 30 -o /dev/null "$BASE_URL/health" || status=SERVER_UNHEALTHY
    finish_rung "$cc" "$rd" "$status"
}

# --------------------------------------------------------------------------- preflight
mkdir -p "$RESULTS_DIR" || exit 1
exec 9> "$RESULTS_DIR/.lock"
flock -n 9 || die "another sweep is using $RESULTS_DIR"
exec > >(tee -a "$RESULTS_DIR/sweep.log") 2>&1
say "SWEEP rungs=[$CONCURRENCIES] endpoint=$BASE_URL model=$SERVED_MODEL_NAME dataset=$DATASET results=$RESULTS_DIR"

MAXCC=0
for cc in $CONCURRENCIES; do
    [[ $cc =~ ^[1-9][0-9]*$ ]] || die "bad concurrency '$cc'"
    [ "$cc" -gt "$MAXCC" ] && MAXCC=$cc
done
[ -x "$PY" ] || die "no client venv at $VENV; run setup_client.sh"
[ -f "$DATASET" ] || die "no dataset at $DATASET; run prepare_dataset.sh"
ATTEMPT=$RESULTS_DIR/attempts/$(date -u +%Y%m%dT%H%M%SZ)-$$
mkdir -p "$ATTEMPT"
say "preflight -> $ATTEMPT"
tool client-check --requirements "$HERE/client-requirements.txt" --out "$ATTEMPT/client-manifest.json" \
    || die "client venv does not match client-requirements.txt"
tool tokenizer-view --model-path "$MODEL_PATH" --view "$TOKENIZER_VIEW" --revision "${MODEL_REVISION:-}" \
    --out "$ATTEMPT/tokenizer-view.json" || die "tokenizer view check failed"
CONTEXT_LENGTH=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["resolved_args"]["context_length"])' "$SERVER_PROFILE")
tool dataset-check --dataset "$DATASET" --tokenizer-manifest "$ATTEMPT/tokenizer-view.json" \
    --chat-template-kwargs "$CHAT_TEMPLATE_KWARGS" --need $(( NUM_MULT * MAXCC + WARMUP_CONVS )) --turns "$TURNS" \
    --first-turn-length "$FIRST_TURN_TOKENS" --subsequent-turn-length "$SUBSEQUENT_TURN_TOKENS" --osl "$OSL" \
    --render-sample 2 --out "$ATTEMPT/dataset-stats.json" || die "dataset check failed"
tool server-check --base-url "$BASE_URL" --profile "$SERVER_PROFILE" --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" --n-gpus "$N_GPUS" --cc "$MAXCC" --out "$ATTEMPT/server-info-start.json" \
    || die "the endpoint does not match $SERVER_PROFILE"
EXTRA_ARGS=$(extra_args)
CONFIG=$("$PY" -c 'import json, sys; print(json.dumps(dict(zip(sys.argv[1::2], sys.argv[2::2]))))' \
    turns "$TURNS" first_turn_tokens "$FIRST_TURN_TOKENS" subsequent_turn_tokens "$SUBSEQUENT_TURN_TOKENS" osl "$OSL" \
    num_mult "$NUM_MULT" warmup_convs "$WARMUP_CONVS" extra_args "$EXTRA_ARGS" base_url "$BASE_URL" \
    served_model_name "$SERVED_MODEL_NAME" server_source "${SERVER_SOURCE:-}" model_revision "${MODEL_REVISION:-}" \
    n_gpus "$N_GPUS" dataset "$DATASET")
tool accept-run --results-dir "$RESULTS_DIR" --attempt-dir "$ATTEMPT" --package-dir "$HERE" --config "$CONFIG" \
    || die "this results directory belongs to a different run (details above; its files are unchanged); use a new RESULTS_DIR"
IDENTITY_SHA=$(sha256sum "$RESULTS_DIR/run-identity.json" | cut -d' ' -f1)
NCONV=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["num_conversations"])' "$RESULTS_DIR/dataset-stats.json")
"$PY" -c 'import json, sys, time
row = dict(zip(sys.argv[2::2], sys.argv[3::2])); row["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
open(sys.argv[1], "a").write(json.dumps(row) + "\n")' "$RESULTS_DIR/invocations.jsonl" \
    host "$(hostname)" attempt "$ATTEMPT" identity_sha256 "$IDENTITY_SHA" concurrencies "$CONCURRENCIES" \
    extra_args "$EXTRA_ARGS" model_path "$MODEL_PATH" model_revision "${MODEL_REVISION:-}" server_source "${SERVER_SOURCE:-}" \
    dataset_sha256 "$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["sha256"])' "$RESULTS_DIR/dataset-stats.json")"

for cc in $CONCURRENCIES; do
    run_rung "$cc" && continue
    [ "$STOP_ON_FAIL" = 1 ] && { say "stopping at the first failed rung (STOP_ON_FAIL=1)"; break; }
done
say "SWEEP COMPLETE (any_fail=$ANY_FAIL) -> $RESULTS_DIR/results.tsv"
"$PY" "$HERE/summarize_rung.py" table --results-dir "$RESULTS_DIR"
exit $ANY_FAIL
