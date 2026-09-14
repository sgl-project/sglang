#!/bin/bash
# Usage: CONC=24 TAG=x [DURATION=3600 PORT=30000 AIPERF_BIN=... AIPERF_EXTRA=...] bash bench_any.sh
: "${CONC:?}"; : "${TAG:?}"
: "${DURATION:=3600}"
: "${PORT:=30000}"
: "${TOKENIZER:=/scratch/models/MiniMax-M3-MXFP4}"
: "${AIPERF_EXTRA:=}"
RESULT_DIR=/scratch/results/aiperf_${TAG}_c${CONC}
[ -d "$RESULT_DIR" ] && mv "$RESULT_DIR" "${RESULT_DIR}_prev_$(date -u +%H%M%S)"
mkdir -p $RESULT_DIR
# an orphaned client on this port would corrupt the measurement, so refuse to start
LOCK=/scratch/run/aiperf_port${PORT}.lock
if [ -f "$LOCK" ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "ERROR: aiperf client pid $(cat $LOCK) for port $PORT still alive"; exit 2; fi
echo $$ > $LOCK
trap 'rm -f $LOCK' EXIT
URL=http://127.0.0.1:$PORT
curl -s $URL/metrics > $RESULT_DIR/server_metrics_before.prom
curl -s $URL/get_server_info > $RESULT_DIR/server_info.json
# GPU util/mem and CPU sampler every 10 s
( while true; do echo "$(date -u +%FT%TZ) $(rocm-smi --showuse --showmemuse --csv 2>/dev/null | grep -E '^card[0-3]' | tr '\n' ';') cpu=$(top -bn1 | grep 'Cpu(s)' | awk '{print $2}')"; sleep 10; done ) > $RESULT_DIR/resources.log 2>&1 &
SAMPLER=$!
# server /metrics snapshot every 60 s
( while true; do sleep 60; curl -s $URL/metrics > $RESULT_DIR/server_metrics_$(date -u +%H%M%S).prom; done ) &
SNAP=$!
START=$(date -u +%FT%TZ)
${AIPERF_BIN:-/scratch/aiperf-venv/bin/aiperf} profile \
  --scenario inferencex-agentx-mvp \
  --url "$URL" \
  --endpoint /v1/chat/completions \
  --endpoint-type chat \
  --streaming \
  --model MiniMax-M3 \
  --tokenizer "$TOKENIZER" \
  --tokenizer-trust-remote-code \
  --apply-chat-template \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --num-dataset-entries 393 \
  --concurrency "$CONC" \
  --benchmark-duration $DURATION \
  --random-seed 42 \
  --use-server-token-count \
  --ui simple \
  --output-artifact-dir "$RESULT_DIR" $AIPERF_EXTRA 2>&1 | tee $RESULT_DIR/aiperf_stdout.log
RC=${PIPESTATUS[0]}
END=$(date -u +%FT%TZ)
kill $SAMPLER $SNAP 2>/dev/null
curl -s $URL/metrics > $RESULT_DIR/server_metrics_after.prom
echo "start=$START end=$END rc=$RC conc=$CONC duration=$DURATION tag=$TAG" > $RESULT_DIR/run_meta.txt
echo "DONE rc=$RC -> $RESULT_DIR"
