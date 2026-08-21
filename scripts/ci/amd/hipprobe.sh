#!/bin/bash
echo "===== container view ====="
echo "kernel: $(uname -r)"
ls -l /dev/kfd /dev/dri 2>&1 | head -6
echo "kfd topology nodes visible: $(ls /sys/class/kfd/kfd/topology/nodes/ 2>/dev/null | wc -l)"
echo "kfd properties readable: $(cat /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null | wc -l) lines"

echo
echo "===== raw HIP, bypassing torch ====="
python3 - <<'PY'
import ctypes, ctypes.util
for name in ("libamdhip64.so", "/opt/rocm/lib/libamdhip64.so"):
    try:
        hip = ctypes.CDLL(name)
        break
    except OSError as e:
        print("load failed", name, e)
else:
    raise SystemExit("no libamdhip64")
n = ctypes.c_int(-1)
rc = hip.hipGetDeviceCount(ctypes.byref(n))
print("hipGetDeviceCount rc=%d count=%d" % (rc, n.value))
hip.hipGetErrorString.restype = ctypes.c_char_p
print("rc meaning:", hip.hipGetErrorString(rc).decode())
rt = ctypes.c_int(0)
if hasattr(hip, "hipRuntimeGetVersion"):
    hip.hipRuntimeGetVersion(ctypes.byref(rt)); print("hip runtime version:", rt.value)
PY

echo
echo "===== HSA agent enumeration (control) ====="
/opt/rocm/bin/rocm_agent_enumerator 2>&1 | head -10 || echo "rocm_agent_enumerator rc=$?"

echo
echo "===== unfiltered HIP log, last 60 lines ====="
AMD_LOG_LEVEL=4 python3 -c "import torch; print('torch count', torch.cuda.device_count())" 2>&1 | tail -60
