#!/usr/bin/env bash
# Reproduce the fixed-input framework benchmark on the existing 151 container.
# Override NPU_DEVICE if device 7 is occupied. Pass --output and case arguments.
set -eo pipefail
TASK_ROOT=${TASK_ROOT:-/home/y00970600/personal/hisparse-lim-framework-20261020}
CONTAINER=${CONTAINER:-qry-20261019-cann9d1}
NPU_DEVICE=${NPU_DEVICE:-7}
docker exec -e ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICE" -e OMP_NUM_THREADS=2 \
  -e SOURCE_ROOT="$TASK_ROOT/src" \
  "$CONTAINER" bash -lc '
    set -e
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export PYTHONPATH="$SOURCE_ROOT/python:$PYTHONPATH"
    exec python "$SOURCE_ROOT/benchmark/hisparse/bench_lim_framework.py" \
      "$@"
  ' bash "$@"
