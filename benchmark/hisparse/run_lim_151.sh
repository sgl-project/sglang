#!/usr/bin/env bash
# Reproduce the fixed-input framework benchmark on the existing 151 container.
# Override NPU_DEVICE if device 7 is occupied. Pass --output and case arguments.
set -eo pipefail
TASK_ROOT=${TASK_ROOT:-/home/y00970600/personal/hisparse-lim-framework-20261020}
CONTAINER=${CONTAINER:-qry-20261019-cann9d1}
NPU_DEVICE=${NPU_DEVICE:-7}
PROBE_ROOT=${PROBE_ROOT:-/home/y00970600/personal/hisparse-fused-probe-151-20261019}
docker exec -e ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICE" -e OMP_NUM_THREADS=2 \
  -e SOURCE_ROOT="$TASK_ROOT/src" -e PROBE_ROOT="$PROBE_ROOT" \
  "$CONTAINER" bash -lc '
    set -e
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source "$PROBE_ROOT/custom_ops/vendors/custom_transformer/bin/set_env.bash"
    export PYTHONPATH="$SOURCE_ROOT/python:$PYTHONPATH"
    export SGLANG_NPU_LIM_LIBRARY="$PROBE_ROOT/torch-probe/hisparse_fused_probe.so"
    exec python "$SOURCE_ROOT/benchmark/hisparse/bench_lim_framework.py" \
      --registered-host-helper "$PROBE_ROOT/host-view-build" "$@"
  ' bash "$@"
