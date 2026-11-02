#!/usr/bin/env bash
# Execute inside the pinned Linux Runpod image only; does not start/stop a Pod.
set -euo pipefail
mode=${1:-prepare}
case "$mode" in prepare|run) ;; *) echo 'Usage: bash prepare_per_request_spec_opt_out_e2e.sh [prepare|run]' >&2; exit 2;; esac
test "$(uname -s)" = Linux
expected=${OPT_OUT_EXPECTED_COMMIT:?Set OPT_OUT_EXPECTED_COMMIT to the reviewed full PR commit SHA}
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
cd "$repo"
test "$(git rev-parse HEAD)" = "$expected" || { echo 'Unexpected PR commit: review before running.' >&2; exit 2; }
test -z "$(git status --porcelain --untracked-files=no)" || { echo 'Tracked source changes detected.' >&2; exit 2; }
export PYTHONPATH="$repo/python:$repo/test/manual/spec"
export SGLANG_CACHE_DIR=/workspace/sglang-47965-cache
export HF_HOME=/workspace/sglang-47965-hf
export CARGO_BUILD_JOBS=4
mkdir -p /workspace/sglang-47965-results

if [[ "$mode" = prepare ]]; then
  exec > >(tee /workspace/sglang-47965-results/preflight.log) 2>&1
  date -u
  git rev-parse HEAD
  python --version
  command -v cargo
  command -v rustup
  command -v c++
  # Match rust/rust-toolchain.toml, including for the excluded radix-tree crate.
  rustup toolchain install 1.92 --profile minimal
  export RUSTUP_TOOLCHAIN=1.92
  cargo --version
  c++ --version
  python -m pip install pytest pytest-subtests
  python -c 'import torch, sglang; print("torch", torch.__version__, "CUDA", torch.version.cuda); print("source", sglang.__file__)'
  # Use the production wrapper: it supplies PyTorch ABI, C++20, compatibility
  # header and rpath settings. Do not copy an unrelated bundled .so or invoke
  # the generic Rust loader directly for this extension.
  SGLANG_RUST_BUILD_MODE=auto timeout --kill-after=15s 600 python -c 'from sglang.srt.mem_cache.rust_tree_core.extension import bindings; print("RUST_BUILD_OK", bindings.__file__)'
  # A separate interpreter proves loading works without a new Cargo build.
  SGLANG_RUST_BUILD_MODE=never python -c 'from sglang.srt.mem_cache.rust_tree_core.extension import bindings; print("RUST_CACHE_LOAD_OK", bindings.__file__)'
  SGLANG_RUST_BUILD_MODE=never python -c 'from test_per_request_spec_opt_out_e2e import install_probes; install_probes(); print("PROBE_IMPORT_OK")'
  timeout --kill-after=15s 600 python - <<'PY'
import os
from huggingface_hub import snapshot_download
models = os.environ.get("OPT_OUT_MODELS", "Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-0.5B-Instruct").split(",")
for model in models:
    path = snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    print("MODEL_CACHE_READY", model, path, flush=True)
PY
  echo PREFLIGHT_PASS
  exit 0
fi

export RUSTUP_TOOLCHAIN=1.92
export SGLANG_RUST_BUILD_MODE=never
# Fail before loading models if preparation/cache did not survive shutdown.
python -c 'from sglang.srt.mem_cache.rust_tree_core.extension import bindings; print("RUST_CACHE_LOAD_OK", bindings.__file__)'
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
nvidia-smi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPT_OUT_RESULTS
OPT_OUT_RESULTS=$(mktemp -d /workspace/sglang-47965-results/e2e-XXXXXXXX)
echo "Results: $OPT_OUT_RESULTS"
# The harness starts servers in separate process groups. Kill only this run's
# bootstrap groups on timeout/interruption; GNU timeout alone cannot reach them.
cleanup() {
  python - <<'PY'
import os, signal, subprocess
needle = os.environ["OPT_OUT_RESULTS"] + "/serve_probe.py"
rows = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
for line in rows.splitlines():
    pid_s, _, args = line.strip().partition(" ")
    if needle not in args:
        continue
    pid = int(pid_s)
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
PY
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
timeout --kill-after=15s 900 python -u test/manual/spec/test_per_request_spec_opt_out_e2e.py 2>&1 | tee "$OPT_OUT_RESULTS/driver.log"
test -s "$OPT_OUT_RESULTS/summary.json"
cat "$OPT_OUT_RESULTS/summary.json"
echo 'Save the complete results directory off-Pod, then STOP the Pod. Script completion does not stop billing.'
