#!/bin/bash
# TEMPORARY: runs probe.py in each receiver mode, three times each.
cd "$(dirname "$0")"
cat /opt/rocm/.info/version 2>/dev/null
python -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_properties(0).gcnArchName)"
pip list 2>/dev/null | grep -iE "torch.memory.saver|^sglang "
PRELOAD=$(python -c "from torch_memory_saver.utils import get_binary_path_from_package as p; print(p('torch_memory_saver_hook_mode_preload'))")
echo "preload: ${PRELOAD}"

run() {
  local name=$1
  shift
  for r in 1 2 3; do
    echo "===== ${name} (run ${r})"
    env "$@" LD_PRELOAD="${PRELOAD}" timeout 300 python probe.py 2>&1 | grep -v "^\s*$" | grep -vi warn
    echo "exit ${PIPESTATUS[0]}"
  done
}

for mode in base pre_ipc self_ipc post_resume_dummy scratch_first sync_after_open touch_before copy_twice recv_1gpu; do
  run "${mode}" MODE="${mode}"
done
run "base AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3" MODE=base AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3
run "base HSA_XNACK=1" MODE=base HSA_XNACK=1
