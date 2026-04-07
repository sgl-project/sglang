#!/bin/bash
cd "$(dirname "$0")/../../.." || exit 1
python -m sglang.srt.speculative.tests.benchmark_triton_sampling --metric-only
