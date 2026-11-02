#!/bin/bash

set -euo pipefail

# =============================================================================
# 用法 (Usage):
#   $0 [output_len] [host] [port] [log_file]
#
# 说明:
#   分两个阶段:
#   1) 搜索阶段: 固定 input_len=16384、output_len=1，按 --request-rate 发压，
#      先倍增/倍减找到包住目标 TTFT 的区间，再二分，找到使 Mean TTFT 落在
#      TARGET_TTFT_MS * (1 ± TTFT_TOL)（默认 1000ms ± 5%）内的 request-rate。
#      若迭代次数用完仍未落入区间，取实测 TTFT 最接近目标的 rate 并给出警告。
#   2) 扫描阶段: 以搜索得到的 base_rate 按输入长度反比缩放:
#        rate(L) = base_rate * 16384 / L   （保留 3 位小数）
#      例: base_rate=4.5 → 8K 用 9.000，32K 用 2.250。
#      依次压测下方 input_len 数组，num 与 bench_sglang.sh 一致（与 rate 同比
#      缩放，使每组压测时长大致相同）。不设置 --max-concurrency。
#
#   所有轮次随机种子固定为 12345，每轮 --flush-cache；warmup 仅第一轮执行。
#   结束后打印 CSV:
#     input_len,num,request_rate,Request throughput,Input token throughput,
#     Mean TTFT,Median TTFT,P99 TTFT
#   CSV 同时保存为 <log_file 去掉 .log>.csv（仅在指定 log_file 时）。
#
# 参数说明（均可选）:
#   output_len   输出 token 数，默认 1
#   host         压测目标地址，默认 10.83.149.81
#   port         压测目标端口，默认 9001
#   log_file     压测日志落盘路径，默认自动创建临时文件（脚本退出时删除）
#
# 环境变量（均可选）:
#   TARGET_TTFT_MS   目标 Mean TTFT，默认 1000
#   TTFT_TOL         相对容差，默认 0.05
#   INIT_RATE        搜索初始 request-rate，默认 4
#   SEARCH_NUM       搜索阶段每轮 num-prompts，默认 512
#   MAX_ITERS        搜索最大迭代次数，默认 14
#   BASE_RATE        直接指定 16K 的 base rate，跳过搜索阶段
#   WARMUP           首轮 warmup 请求数，默认 4（仅首轮执行）
#
# 示例:
#   $0
#   $0 1 10.34.69.50 9001 sweep.log
#   BASE_RATE=4.5 $0 1 10.34.69.50 9001 sweep.log   # 跳过搜索
# =============================================================================

input_len=(4096 8192 16384 32768 65536 131072 262144 524288 786432 1048000)
num=(1536 1024 512 256 128 64 32 16 10 8)

base_len=16384
seed=12345
model=/mnt/aie-shared-models-n-datasets-wlcb/wangfan8/models/mimo-v2.6-flash/global_step_11
dataset_path=/mnt/dataset/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json

# bench_serving 压完后会读 /get_server_info 的 internal_states 取 spec accept length，
# 经 router 访问时返回体没有该字段会 KeyError 导致整轮无结果。这里包一层:
# 返回体缺 internal_states 时视作非 200，accept_length 记为 None，其余逻辑不变。
bench_entry='
import runpy, requests
_get = requests.get
def _patched_get(url, *a, **kw):
    r = _get(url, *a, **kw)
    if str(url).endswith("/get_server_info") and r.status_code == 200:
        try:
            j = r.json()
            j = j["decode"][0] if "decode" in j else j
            j["internal_states"][0]
        except Exception:
            r.status_code = 599
    return r
requests.get = _patched_get
runpy.run_module("sglang.bench_serving", run_name="__main__", alter_sys=True)
'

o="${1:-1}"
host="${2:-10.83.149.81}"
port="${3:-9001}"
log_file="${4:-}"

target_ttft="${TARGET_TTFT_MS:-1000}"
ttft_tol="${TTFT_TOL:-0.05}"
init_rate="${INIT_RATE:-4}"
search_num="${SEARCH_NUM:-512}"
max_iters="${MAX_ITERS:-14}"
base_rate="${BASE_RATE:-}"

# 未指定日志路径时使用临时文件，脚本退出时清理；指定则保留。
cleanup_log=0
csv_file=""
if [[ -z "$log_file" ]]; then
    log_file="$(mktemp)"
    cleanup_log=1
else
    csv_file="${log_file%.log}.csv"
fi
run_out="$(mktemp)"
trap '((cleanup_log)) && rm -f "$log_file"; rm -f "$run_out"' EXIT

if ((${#input_len[@]} != ${#num[@]})); then
    echo "Error: input_len and num must have the same length" >&2
    exit 1
fi

: > "$log_file"

# 首轮 warmup（预热服务端 / CUDA graph 等），之后置 0
warmup="${WARMUP:-4}"

# 浮点计算统一走 awk
calc() { awk "BEGIN { $1 }"; }

# rate 保留 3 位小数，最小 0.001
fmt_rate() { calc "r = $1; if (r < 0.001) r = 0.001; printf \"%.3f\", r"; }

# 从单轮输出中取 "<label>: value" 的值
get_field() {
    awk -v label="$1" '
        { sub(/\r$/, "") }
        index($0, label) == 1 {
            v = $0; sub(/^[^:]*:/, "", v); gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
            val = v
        }
        END { print val }
    ' "$run_out"
}

# run_bench <input_len> <num> <rate> <tag>
# 结果写入全局变量 RUN_OK / RUN_REQ_TPUT / RUN_IN_TPUT / RUN_MEAN_TTFT / RUN_MEDIAN_TTFT / RUN_P99_TTFT
run_bench() {
    local cur_len="$1" cur_num="$2" cur_rate="$3" tag="$4"
    {
        echo "============================================================"
        echo ">> [${tag}] input_len=${cur_len} output_len=${o} num_prompts=${cur_num} request_rate=${cur_rate} warmup=${warmup} host=${host}:${port}"
        echo "============================================================"
    } | tee -a "$log_file"

    : > "$run_out"
    RUN_OK=1
    if ! python3 -c "$bench_entry" \
        --backend sglang \
        --model "$model" \
        --host "$host" \
        --port "$port" \
        --dataset-name random \
        --dataset-path "$dataset_path" \
        --warmup-requests "$warmup" \
        --random-input "$cur_len" \
        --random-output "$o" \
        --random-range-ratio 1.0 \
        --flush-cache \
        --seed "$seed" \
        --num-prompts "$cur_num" \
        --request-rate "$cur_rate" 2>&1 | tee -a "$log_file" "$run_out"; then
        echo ">> [warn] input_len=${cur_len} request_rate=${cur_rate} 压测异常退出" | tee -a "$log_file" >&2
        RUN_OK=0
    fi
    warmup=0

    RUN_REQ_TPUT="$(get_field 'Request throughput (req/s):')"
    RUN_IN_TPUT="$(get_field 'Input token throughput (tok/s):')"
    RUN_MEAN_TTFT="$(get_field 'Mean TTFT (ms):')"
    RUN_MEDIAN_TTFT="$(get_field 'Median TTFT (ms):')"
    RUN_P99_TTFT="$(get_field 'P99 TTFT (ms):')"
    [[ -n "$RUN_MEAN_TTFT" ]] || RUN_OK=0
}

# -----------------------------------------------------------------------------
# 阶段 1: 搜索 16K 下 Mean TTFT ≈ target 的 request-rate
# -----------------------------------------------------------------------------
if [[ -z "$base_rate" ]]; then
    ttft_lo="$(calc "print $target_ttft * (1 - $ttft_tol)")"
    ttft_hi="$(calc "print $target_ttft * (1 + $ttft_tol)")"
    echo ">> 搜索阶段: input_len=${base_len} 目标 Mean TTFT ∈ [${ttft_lo}, ${ttft_hi}] ms" | tee -a "$log_file"

    lo=""          # TTFT 偏低的最大 rate
    hi=""          # TTFT 偏高的最小 rate
    rate="$(fmt_rate "$init_rate")"
    best_rate="" best_diff=""
    search_hist=()

    for ((it = 1; it <= max_iters; it++)); do
        run_bench "$base_len" "$search_num" "$rate" "search ${it}/${max_iters}"
        if ((!RUN_OK)); then
            # 失败按 TTFT 过高处理（通常是压力过大）
            echo ">> [search] rate=${rate} 无有效结果，视为 TTFT 过高" | tee -a "$log_file"
            ttft="inf"
            hi="$rate"
        else
            ttft="$RUN_MEAN_TTFT"
            diff="$(calc "d = $ttft - $target_ttft; print (d < 0 ? -d : d)")"
            if [[ -z "$best_diff" ]] || calc "exit !($diff < $best_diff)"; then
                best_rate="$rate" best_diff="$diff"
            fi
            if calc "exit !($ttft >= $ttft_lo && $ttft <= $ttft_hi)"; then
                search_hist+=("${rate},${ttft},hit")
                echo ">> [search] rate=${rate} Mean TTFT=${ttft} ms 命中目标区间" | tee -a "$log_file"
                base_rate="$rate"
                break
            elif calc "exit !($ttft < $ttft_lo)"; then
                lo="$rate"
            else
                hi="$rate"
            fi
        fi
        search_hist+=("${rate},${ttft},$([[ "$hi" == "$rate" ]] && echo high || echo low)")
        echo ">> [search] rate=${rate} Mean TTFT=${ttft} ms → 区间 lo=${lo:-?} hi=${hi:-?}" | tee -a "$log_file"

        # 下一个 rate: 未成区间时倍增/倍减，否则二分
        if [[ -z "$hi" ]]; then
            next="$(fmt_rate "$(calc "print $rate * 2")")"
        elif [[ -z "$lo" ]]; then
            next="$(fmt_rate "$(calc "print $rate / 2")")"
        else
            next="$(fmt_rate "$(calc "print ($lo + $hi) / 2")")"
        fi
        if [[ "$next" == "$rate" || "$next" == "$lo" || "$next" == "$hi" ]]; then
            echo ">> [search] rate 精度已达 0.001，停止搜索" | tee -a "$log_file"
            break
        fi
        rate="$next"
    done

    if [[ -z "$base_rate" ]]; then
        if [[ -z "$best_rate" ]]; then
            echo "Error: 搜索阶段没有任何有效结果" >&2
            exit 1
        fi
        base_rate="$best_rate"
        echo ">> [warn] 未找到落入目标区间的 rate，使用最接近目标的 rate=${base_rate}（|ΔTTFT|=${best_diff} ms）" | tee -a "$log_file" >&2
    fi

    echo
    echo "============================================================" | tee -a "$log_file"
    echo ">> 搜索记录 (request_rate,Mean TTFT,result)" | tee -a "$log_file"
    printf '%s\n' "${search_hist[@]}" | tee -a "$log_file"
fi

base_rate="$(fmt_rate "$base_rate")"
echo ">> base_rate(${base_len})=${base_rate}" | tee -a "$log_file"

# -----------------------------------------------------------------------------
# 阶段 2: 按 rate(L) = base_rate * 16384 / L 扫描各输入长度
# -----------------------------------------------------------------------------
total=${#input_len[@]}
csv_rows=()
for ((i = 0; i < total; i++)); do
    cur_len="${input_len[$i]}"
    cur_num="${num[$i]}"
    cur_rate="$(fmt_rate "$(calc "print $base_rate * $base_len / $cur_len")")"
    run_bench "$cur_len" "$cur_num" "$cur_rate" "sweep $((i + 1))/${total}"
    if ((RUN_OK)); then
        csv_rows+=("${cur_len},${cur_num},${cur_rate},${RUN_REQ_TPUT},${RUN_IN_TPUT},${RUN_MEAN_TTFT},${RUN_MEDIAN_TTFT},${RUN_P99_TTFT}")
    else
        echo ">> [warn] input_len=${cur_len} 无有效结果，继续下一组" | tee -a "$log_file" >&2
    fi
done

echo
echo "============================================================"
echo ">> 结果 (base_rate@${base_len}=${base_rate}, target Mean TTFT=${target_ttft}ms)"
echo "============================================================"

{
    echo "input_len,num,request_rate,Request throughput,Input token throughput,Mean TTFT,Median TTFT,P99 TTFT"
    if ((${#csv_rows[@]})); then printf '%s\n' "${csv_rows[@]}"; fi
} | if [[ -n "$csv_file" ]]; then tee "$csv_file"; else cat; fi

((${#csv_rows[@]})) || { echo "Error: no complete benchmark results" >&2; exit 1; }
