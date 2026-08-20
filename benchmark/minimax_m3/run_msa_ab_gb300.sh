#!/usr/bin/env bash
# Entry point intended to run inside an already-created 4x GB300 Slurm allocation.
set -euo pipefail

MODEL="${MODEL:?set MODEL to a fully cached model id or local checkpoint path}"
LONGBENCH_SUBSET="${LONGBENCH_SUBSET:?set LONGBENCH_SUBSET to the frozen JSON subset}"
GPQA_DATASET="${GPQA_DATASET:?set GPQA_DATASET to the frozen GPQA-Diamond CSV}"
FLASHINFER_SOURCE_DIR="${FLASHINFER_SOURCE_DIR:?set FLASHINFER_SOURCE_DIR to the final source checkout}"
FLASHINFER_HEAD="${FLASHINFER_HEAD:?set FLASHINFER_HEAD to the exact 40-character source commit}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to a new evidence directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" benchmark/minimax_m3/run_msa_ab_repetitions.py \
  --model "${MODEL}" \
  --longbench-subset "${LONGBENCH_SUBSET}" \
  --gpqa-dataset "${GPQA_DATASET}" \
  --flashinfer-source-dir "${FLASHINFER_SOURCE_DIR}" \
  --expected-flashinfer-head "${FLASHINFER_HEAD}" \
  --output-root "${OUTPUT_ROOT}" \
  --python "${PYTHON_BIN}" \
  --min-median-output-throughput-gain "${MIN_MEDIAN_OUTPUT_THROUGHPUT_GAIN:-0}"
