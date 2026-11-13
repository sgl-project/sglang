#!/usr/bin/env bash
# Write the all-reduce selection, then tune the backends in that file.
#
# Selection flags are consumed here and are not forwarded. The tuner ignores
# them once --selection-config is set.
#   --qr-quant INT4
#   --qr-cast-bf16-to-fp16 0|1
#   --ignore-acr-max-size 0|1
#   --ignore-scr-max-size 0|1
#   --selection-out PATH     JSON to write (default: <output-dir>/selection_config.json)
#   --nproc-per-node N       torchrun processes (default: --tp)
#
# Every other argument is passed to run_allreduce. --tp and --sizes are required.
#
#   /home/oshkarav/sglang/python/sglang/auto_tune/run_allreduce.sh \
#       --tp 8 --qr-quant INT4 --qr-cast-bf16-to-fp16 1 \
#       --median 1 -n 200 --output-dir /tmp/ar-tune \
#       --sizes 16384 65536 131072

set -euo pipefail

qr_quant="INT4"
qr_cast="1"
ignore_acr="0"
ignore_scr="0"
selection_out=""
output_dir="allreduce_tune_out"
nproc=""
tp=""
tuner_args=()

usage() {
    cat <<'EOF'
Write the all-reduce selection, then tune the backends in that file.

Selection flags are consumed here and are not forwarded. The tuner ignores
them once --selection-config is set.
  --qr-quant INT4                  HIP QuickReduce codec. Default INT4.
  --qr-cast-bf16-to-fp16 0|1       Cast bf16 to fp16 inside HIP QuickReduce. Default 1.
  --ignore-acr-max-size 0|1        Let AITER run past its max size. Default 0.
  --ignore-scr-max-size 0|1        Let SGLang custom all-reduce run past its max size. Default 0.
  --selection-out PATH             JSON to write. Default: <output-dir>/selection_config.json
  --nproc-per-node N               torchrun processes. Default: --tp
  -h, --help                       Show this text.

Every other argument is passed to run_allreduce. --tp and --sizes are required.
Tuner flags are listed by: python -m sglang.auto_tune.run_allreduce --help

  /home/oshkarav/sglang/python/sglang/auto_tune/run_allreduce.sh \
      --tp 8 --qr-quant INT4 --qr-cast-bf16-to-fp16 1 \
      --ignore-acr-max-size 1 --ignore-scr-max-size 0 \
      --median 1 -n 200 --output-dir /tmp/ar-tune \
      --sizes 16384 65536 131072
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --qr-quant)
            qr_quant="$2"
            shift 2
            ;;
        --qr-cast-bf16-to-fp16)
            qr_cast="$2"
            shift 2
            ;;
        --ignore-acr-max-size)
            ignore_acr="$2"
            shift 2
            ;;
        --ignore-scr-max-size)
            ignore_scr="$2"
            shift 2
            ;;
        --selection-out)
            selection_out="$2"
            shift 2
            ;;
        --selection-config)
            echo "run_allreduce.sh writes the selection itself; drop --selection-config" >&2
            exit 2
            ;;
        --nproc-per-node)
            nproc="$2"
            shift 2
            ;;
        --output-dir)
            output_dir="$2"
            tuner_args+=("$1" "$2")
            shift 2
            ;;
        --tp|--tp-size)
            tp="$2"
            tuner_args+=("$1" "$2")
            shift 2
            ;;
        *)
            tuner_args+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$tp" ]]; then
    echo "run_allreduce.sh: --tp is required" >&2
    usage >&2
    exit 2
fi
if [[ -z "$nproc" ]]; then
    nproc="$tp"
fi
if [[ -z "$selection_out" ]]; then
    selection_out="${output_dir}/selection_config.json"
fi

mkdir -p "$(dirname "$selection_out")"

python3 -m sglang.auto_tune.generate_selection_config \
    --output "$selection_out" \
    --qr-quant "$qr_quant" \
    --qr-cast-bf16-to-fp16 "$qr_cast" \
    --ignore-acr-max-size "$ignore_acr" \
    --ignore-scr-max-size "$ignore_scr"

exec torchrun --nproc_per_node="$nproc" -m sglang.auto_tune.run_allreduce \
    --selection-config "$selection_out" \
    "${tuner_args[@]}"
