#!/usr/bin/env bash
# Entry point intended to run inside an already-created 4x GB300 Slurm allocation.
set -euo pipefail

MODEL="${MODEL:?set MODEL to a fully cached model id or local checkpoint path}"
LONGBENCH_SUBSET="${LONGBENCH_SUBSET:?set LONGBENCH_SUBSET to the frozen JSON subset}"
GPQA_DATASET="${GPQA_DATASET:?set GPQA_DATASET to the frozen GPQA-Diamond CSV}"
FLASHINFER_SOURCE_DIR="${FLASHINFER_SOURCE_DIR:?set FLASHINFER_SOURCE_DIR to the final source checkout}"
FLASHINFER_HEAD="${FLASHINFER_HEAD:?set FLASHINFER_HEAD to the exact 40-character source commit}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to a new evidence directory}"
EXPECTED_TVM_FFI_VERSION="${EXPECTED_TVM_FFI_VERSION:?set the compatibility baseline TVM-FFI version}"
GPQA_SCORE_TOLERANCE="${GPQA_SCORE_TOLERANCE:?set the explicit GPQA noninferiority margin}"
LONGBENCH_SCORE_TOLERANCE="${LONGBENCH_SCORE_TOLERANCE:?set the explicit LongBench-v2 noninferiority margin}"
MINFER_FMHA_CACHE_DIR="${MINFER_FMHA_CACHE_DIR:?set an allocation-local baseline JIT cache}"
BASELINE_FMHA_PRECOMPILE_RECEIPT="${BASELINE_FMHA_PRECOMPILE_RECEIPT:?set a new precompile receipt path}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" benchmark/minimax_m3/precompile_fmha_sm100.py \
  --cache-dir "${MINFER_FMHA_CACHE_DIR}" \
  --output "${BASELINE_FMHA_PRECOMPILE_RECEIPT}"

exec "${PYTHON_BIN}" benchmark/minimax_m3/run_msa_ab_repetitions.py \
  --model "${MODEL}" \
  --longbench-subset "${LONGBENCH_SUBSET}" \
  --gpqa-dataset "${GPQA_DATASET}" \
  --flashinfer-source-dir "${FLASHINFER_SOURCE_DIR}" \
  --expected-flashinfer-head "${FLASHINFER_HEAD}" \
  --output-root "${OUTPUT_ROOT}" \
  --expected-tvm-ffi-version "${EXPECTED_TVM_FFI_VERSION}" \
  --python "${PYTHON_BIN}" \
  --server-timeout "${SERVER_TIMEOUT:-7200}" \
  --gpqa-score-tolerance "${GPQA_SCORE_TOLERANCE}" \
  --longbench-score-tolerance "${LONGBENCH_SCORE_TOLERANCE}" \
  --start-repetition "${START_REPETITION:-1}" \
  --min-median-output-throughput-gain "${MIN_MEDIAN_OUTPUT_THROUGHPUT_GAIN:-0}"
