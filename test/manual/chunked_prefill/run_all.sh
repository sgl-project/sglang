#!/usr/bin/env bash
# Sequentially run every chunked-prefill manual fixture in this directory.
#
# Each fixture's stdout/stderr is captured to results/<fixture>.log and its
# mixed-prefix gsm8k metrics (when produced) to results/<fixture>.json. A
# brief summary table is printed at the end.
#
# Usage:
#   bash test/manual/chunked_prefill/run_all.sh                  # all
#   bash test/manual/chunked_prefill/run_all.sh --only a,e,i     # subset
#   bash test/manual/chunked_prefill/run_all.sh --skip d,b       # complement
#   RESULTS_DIR=/tmp/x bash test/manual/chunked_prefill/run_all.sh
#
# Fixtures are not registered with CI; this script is the only entry point.

set -uo pipefail

cd "$(dirname "$0")/../../.."

ALL_LETTERS=(a b c d e f g h i j k)
SKIP=""
ONLY=""
while (( "$#" )); do
  case "$1" in
    --skip) SKIP="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

declare -A LETTER_TO_MODULE=(
  [a]="test_feature_a_pp"
  [b]="test_feature_b_disagg"
  [c]="test_feature_c_hybrid_swa"
  [d]="test_feature_d_hisparse"
  [e]="test_feature_e_spec"
  [f]="test_feature_f_radix"
  [g]="test_feature_g_priority"
  [h]="test_feature_h_piecewise_cuda_graph"
  [i]="test_feature_i_lora"
  [j]="test_feature_j_lora_overlap"
  [k]="test_feature_k_dp_attention"
)

# Resolve which letters to run.
if [[ -n "$ONLY" ]]; then
  IFS=',' read -ra letters <<< "$ONLY"
else
  letters=("${ALL_LETTERS[@]}")
fi
if [[ -n "$SKIP" ]]; then
  IFS=',' read -ra skipset <<< "$SKIP"
  filtered=()
  for l in "${letters[@]}"; do
    in_skip=0
    for s in "${skipset[@]}"; do [[ "$l" == "$s" ]] && in_skip=1 && break; done
    [[ "$in_skip" -eq 0 ]] && filtered+=("$l")
  done
  letters=("${filtered[@]}")
fi

RESULTS_DIR="${RESULTS_DIR:-test/manual/chunked_prefill/results}"
mkdir -p "$RESULTS_DIR"
export CHUNKED_PREFILL_RESULTS_DIR="$RESULTS_DIR"

declare -A STATUS
declare -A ELAPSED

for letter in "${letters[@]}"; do
  module="${LETTER_TO_MODULE[$letter]:-}"
  if [[ -z "$module" ]]; then
    echo "[skip] unknown letter '$letter'"
    continue
  fi
  log="$RESULTS_DIR/${module}.log"
  echo "=== [$letter] $module === (log: $log)"
  t0=$(date +%s)
  python3 -m unittest "test.manual.chunked_prefill.$module" -v \
    > "$log" 2>&1
  rc=$?
  t1=$(date +%s)
  ELAPSED[$letter]=$((t1 - t0))
  if [[ $rc -eq 0 ]]; then
    STATUS[$letter]="PASS"
  else
    STATUS[$letter]="FAIL($rc)"
  fi
  echo "    $letter: ${STATUS[$letter]} (${ELAPSED[$letter]}s)"
done

echo
echo "==================== SUMMARY ===================="
printf "%-3s %-40s %-12s %-8s\n" "" "fixture" "status" "elapsed"
echo "-------------------------------------------------"
for letter in "${letters[@]}"; do
  module="${LETTER_TO_MODULE[$letter]:-?}"
  printf "%-3s %-40s %-12s %-8s\n" "$letter" "$module" "${STATUS[$letter]:-SKIP}" "${ELAPSED[$letter]:-0}s"
done
echo "================================================="
echo "Per-fixture JSON metrics (if produced): $RESULTS_DIR/*.json"
