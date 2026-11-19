#!/usr/bin/env bash
# The logprob comparison with --ignore-eos (every prompt decodes 32 tokens), three rounds
# of one server per half node. GPUs 0-3: sgl-kernel, flashinfer, flashinfer-small (tie-break
# without deterministic mode); GPUs 4-7: flashinfer-small-det, flashinfer-large-det,
# flashinfer-unfused-small-det. Results in logprob2/.
set -uo pipefail
ROOT=/hai-workspace/dsv41-fi-topk
cd "$ROOT"
export LOGPROB_DIR=logprob2 COLLECT_ARGS=--ignore-eos
until [[ -z $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1024') ]]; do
  sleep 10
done
FI=(-- --dsa-topk-backend flashinfer)
SMALL_DET=(SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK=small SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC=1)
echo "$(date -u +%FT%TZ) round 1 start"
bash tools/run_logprob.sh sgl-kernel 0,1,2,3 30000 &
bash tools/run_logprob.sh flashinfer-small-det 4,5,6,7 30001 "${SMALL_DET[@]}" "${FI[@]}" &
wait
echo "$(date -u +%FT%TZ) round 2 start"
bash tools/run_logprob.sh flashinfer 0,1,2,3 30000 "${FI[@]}" &
bash tools/run_logprob.sh flashinfer-large-det 4,5,6,7 30001 \
  SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK=large SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC=1 "${FI[@]}" &
wait
echo "$(date -u +%FT%TZ) round 3 start"
bash tools/run_logprob.sh flashinfer-small 0,1,2,3 30000 SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK=small "${FI[@]}" &
bash tools/run_logprob.sh flashinfer-unfused-small-det 4,5,6,7 30001 SGLANG_DSA_FUSE_TOPK=0 "${SMALL_DET[@]}" "${FI[@]}" &
wait
L=logprob2
pairs=()
for c in sgl-kernel flashinfer flashinfer-small flashinfer-small-det flashinfer-large-det flashinfer-unfused-small-det; do
  pairs+=("$L/$c/pass1.jsonl:$L/$c/pass2.jsonl")
done
pairs+=(
  "$L/sgl-kernel/pass1.jsonl:$L/flashinfer/pass1.jsonl"
  "$L/sgl-kernel/pass1.jsonl:$L/flashinfer-small/pass1.jsonl"
  "$L/flashinfer-small-det/pass1.jsonl:$L/flashinfer-large-det/pass1.jsonl"
  "$L/flashinfer-small-det/pass1.jsonl:$L/flashinfer-unfused-small-det/pass1.jsonl"
  "$L/sgl-kernel/pass1.jsonl:$L/flashinfer-small-det/pass1.jsonl"
  "$L/sgl-kernel/pass1.jsonl:$L/flashinfer-large-det/pass1.jsonl"
)
python3 tools/logprob_equivalence.py compare --out $L/report.md "${pairs[@]}" > $L/compare.log 2>&1
echo "$(date -u +%FT%TZ) logprob2 finished"
