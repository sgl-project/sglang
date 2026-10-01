#!/usr/bin/env bash
# One warmup run followed by three measured, identically seeded repetitions.
set -euo pipefail
if [[ $# != 5 ]]; then
    echo "Usage: $0 MODEL_PATH INFERENCEX_CHECKOUT CONC BASE_URL NEW_OUTPUT_DIR" >&2
    exit 2
fi
sr_model=$1
sr_infx=$2
sr_conc=$3
sr_url=${4%/}
sr_out=$5
case "$sr_conc" in 1|2|4|8|16|32) ;; *) exit 2;; esac
test ! -e "$sr_out"
sr_here=$(cd "$(dirname "$0")" && pwd)
sr_requests=160
if [[ "$sr_conc" == 32 ]]; then sr_requests=320; fi
test "$(git -C "$sr_infx" rev-parse HEAD)" = 127c84be90f1536a92c1bd62f7f8b3b071d615fe
test -z "$(git -C "$sr_infx" status --porcelain)"
python3 "$sr_here/run_inferencex.py" --inferencex-dir "$sr_infx" \
    --checkpoint "$sr_model" --self-test
mkdir -p "$sr_out"
curl --fail --silent --show-error "$sr_url/get_server_info" -o "$sr_out/server-info.json"
for sr_phase in warmup repeat-01 repeat-02 repeat-03; do
    sr_dest="$sr_out/$sr_phase"
    mkdir "$sr_dest"
    curl --fail --silent --show-error "$sr_url/metrics" -o "$sr_dest/metrics.before.prom"
    python3 "$sr_here/run_inferencex.py" --inferencex-dir "$sr_infx" \
        --checkpoint "$sr_model" --thinking thinking --reasoning-effort 100 \
        --request-manifest "$sr_dest/requests.json" \
        --model "$sr_model" --tokenizer "$sr_model" \
        --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
        --backend vllm --base-url "$sr_url" --endpoint /v1/completions \
        --dataset-name random --random-input-len 8192 --random-output-len 1024 \
        --random-range-ratio .8 --random-prefix-len 0 --random-num-workers 1 \
        --num-prompts "$sr_requests" --num-warmups "$((2 * sr_conc))" \
        --max-concurrency "$sr_conc" --request-rate inf --seed 0 --ignore-eos \
        --use-chat-template --trust-remote-code \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 \
        --save-result --save-detailed --result-dir "$sr_dest" --result-filename benchmark.json \
        2>&1 | tee "$sr_dest/client.log"
    curl --fail --silent --show-error "$sr_url/metrics" -o "$sr_dest/metrics.after.prom"
done
python3 "$sr_here/summarize_perf.py" "$sr_out"
