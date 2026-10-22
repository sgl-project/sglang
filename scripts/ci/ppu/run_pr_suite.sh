#!/usr/bin/env bash
# PR-only entry point; one installation per worker pod, no deterministic retries.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
lane="${1:?cpu|smoke|body|multi}"
export PPU_EVIDENCE_DIR="${PPU_EVIDENCE_DIR:?run/attempt/job-specific directory required}"
mkdir -p "$PPU_EVIDENCE_DIR"
export PYTHONPATH="$ROOT/python:$ROOT/test${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_TEST_MAX_RETRY=0
export SGLANG_IS_IN_CI=true
# The worker cannot append to the orchestration container's summary path.
export GITHUB_STEP_SUMMARY="$PPU_EVIDENCE_DIR/test-summary.log"
start=$SECONDS
finish() {
    status=$?
    printf '{"exit_code":%s,"elapsed":%s}\n' "$status" "$((SECONDS-start))" > "$PPU_EVIDENCE_DIR/worker.json"
    python3 scripts/ci/ppu/pr_report.py sanitize --directory "$PPU_EVIDENCE_DIR"
    exit "$status"
}
trap finish EXIT
case "$lane" in
    cpu) hw=cpu; suite=stage-a-test-cpu-ppu; cap=300 ;;
    smoke) hw=ppu; suite=stage-a-test-1-gpu-ppu; cap=300 ;;
    body) hw=ppu; suite=stage-b-test-1-gpu-ppu; cap=1080 ;;
    multi) hw=ppu; suite=stage-c-test-2-gpu-ppu; cap=600 ;;
    *) exit 2 ;;
esac
if [[ "$lane" == cpu ]]; then
    bash scripts/ci/ppu/install_pr_cpu.sh > "$PPU_EVIDENCE_DIR/setup.log" 2>&1
else
    # Match vLLM's published-action policy: shared NAS mounts remain writable.
    # Keep routine writes in /ppu-pr-cache and /ppu-evidence, but these paths
    # are destinations, not a filesystem access-control boundary.
    export PPU_INSTALL_PROFILE=pr
    export GITHUB_WORKSPACE="$ROOT"
    export PIP_CACHE_DIR=/ppu-pr-cache/pip
    export SGL_KERNEL_WHEEL_CACHE_DIR=/ppu-pr-cache/aot/OAM-810E
    export PPU_JIT_CACHE_SHARED=/mnt/wl_nas/cache/sglang-jit-ci
    export PPU_JIT_CACHE_LOCAL=/tmp/ppu-pr-tvm-ffi
    export SGLANG_JIT_CACHE_DIR=/tmp/ppu-pr-jit
    export DG_JIT_CACHE_DIR=/tmp/ppu-pr-deep-gemm
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python3 python/sglang/test/ppu_pr_utils.py > "$PPU_EVIDENCE_DIR/model.json"
    bash scripts/ci/ppu/ppu_install_dependency.sh > "$PPU_EVIDENCE_DIR/setup.log" 2>&1
    source scripts/ci/ppu/use_local_jit_cache.sh >> "$PPU_EVIDENCE_DIR/setup.log" 2>&1
fi
printf '{"setup_seconds":%s,"lane":"%s"}\n' "$((SECONDS-start))" "$lane" > "$PPU_EVIDENCE_DIR/setup.json"
python3 - <<'PY' > "$PPU_EVIDENCE_DIR/environment.json"
import json, os, platform, subprocess, torch, sglang
from pathlib import Path
assert Path(sglang.__file__).resolve().is_relative_to(Path.cwd() / 'python'), 'Not importing tested checkout'
print(json.dumps({
    'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'python': platform.python_version(), 'torch': torch.__version__, 'checkout_import': sglang.__file__,
    'image': os.environ.get('PPU_BASE_IMAGE_REF'), 'digest': os.environ.get('PPU_BASE_IMAGE_DIGEST'),
    'sdk': os.environ.get('PPU_SDK_VERSION'),
    'backend': 'cpu' if os.environ.get('PPU_INSTALL_PROFILE') != 'pr' else 'FA3 serving; Triton offline smoke; PyTorch sampling probes',
    'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}, indent=2))
PY
tests_start=$SECONDS
runner_args=()
# Collect evidence for every CPU file in one attempt; any failure still fails CI.
[[ "$lane" != cpu ]] || runner_args+=(--continue-on-error)
python3 test/run_suite.py --hw "$hw" --suite "$suite" --pytest --require-nonempty \
    --junit-dir "$PPU_EVIDENCE_DIR/$suite" --timeout-per-file "$cap" --suite-timeout "$cap" \
    "${runner_args[@]}"
if [[ "$lane" == body && "${PPU_RUN_KERNEL:-false}" == true ]]; then
    remaining=$((1080 - (SECONDS-tests_start)))
    ((remaining > 0))
    ((remaining <= 180)) || remaining=180
    python3 test/run_suite.py --hw ppu --suite stage-b-kernel-test-1-gpu-ppu --pytest --require-nonempty \
        --junit-dir "$PPU_EVIDENCE_DIR/stage-b-kernel-test-1-gpu-ppu" \
        --timeout-per-file "$remaining" --suite-timeout "$remaining"
fi
# The EXIT trap records completion; the orchestration job then validates all evidence.
