#!/bin/bash
# Run one single-board serving performance entry inside the worker pod.
#
# This is the body every entry of test-ppu-perf.yml executes on its board,
# held here rather than in the workflow for the reason the Answer board script
# records: that workflow needs one job id per concurrent entry, because
# flytiger-eco/ppu-distributed-action derives both its NAS staging path and its
# K8s job name from $GITHUB_JOB, and a local composite action is not available to
# those jobs -- their runner group drives the job through a container hook, so the
# workspace the checkout populates is inside the container while the runner
# resolves `uses: ./` against its own filesystem (measured, run 34085102675).
#
# A separate script from run_answer_suite_board.sh rather than a shared one with
# a flag: the two suites read different environment variables on purpose, so a
# change made for one cannot move the other's config, and the two lines can share
# a board without reading each other's settings.
#
# Reads:
#   SGLANG_PPU_PERF_TEST_CONFIG  absolute path of the reviewed config, and the
#                                one source for the visible devices and the
#                                checkpoint -- a second copy in the workflow
#                                could drift from what the test reads
#   PERF_SUITE                   registered suite name for run_suite.py
#   PERF_TIMEOUT_PER_FILE        seconds run_suite.py gives the registered file
#
# Usage: bash /workspace/source/scripts/ci/ppu/run_perf_suite_board.sh
set -e

mkdir -p ~/.pip && cat > ~/.pip/pip.conf << 'PIPEOF'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
PIPEOF

: "${SGLANG_PPU_PERF_TEST_CONFIG:?the reviewed config path is required}"
: "${PERF_SUITE:?the registered suite name is required}"
: "${PERF_TIMEOUT_PER_FILE:?the per-file timeout is required}"

# Constant for every entry, so they are set here rather than repeated in each
# job. HF_HUB_OFFLINE keeps the tokenizer load off the network on a cluster that
# has none, and this suite loads a tokenizer twice over -- once in the server and
# once in bench_serving, which encodes the synthetic prompts; HF_HUB_CACHE is the
# shared read-only cache the checkpoints' companion repositories were fetched
# into. PPU_SUPPORTS_FP8=0 reflects the board, not a preference.
export SGLANG_IS_IN_CI=true
export CPLUS_INCLUDE_PATH=/usr/local/PPU_SDK/targets/x86_64-linux/include
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=/nas_aisw/datasets/hf_cache/hub
export PPU_SUPPORTS_FP8=0

# v0.5.18 asserts sglang-kernel >= 0.4.6.post1 / flashinfer >= 0.6.17 whenever it
# sees CUDA, and PPU presents itself as CUDA. The image carries the vendor's PPU
# builds (0.4.3+v0.1.0.ppu2.1.1 / 0.6.12), which sit on a different release train
# and have no upgrade path here, so the comparison only blocks the launch --
# measured on the accuracy line, run 34900233869.
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

# _handle_ppu_backends enables SGLANG_SAIL_PLA_CUDA on PPU, which sends the GDN
# linear-attention decode into `pla.decode`. `pla` is absent from the v2.1.1
# image and nothing installs it, so the Qwen3.5 and Qwen3-Next entries die in
# CUDA graph capture with `No module named 'pla'` (measured on the accuracy
# line, run 34908645516). Off takes the community Triton kernel the same
# function falls through to. Drop this once the image ships `pla`.
export SGLANG_SAIL_PLA_CUDA=0

# Gloo carries the CPU side of every process group SGLang creates, and left to
# itself it picks its address by resolving the pod's own hostname, which on
# several of these nodes has no address: ranks either fall back to loopback or
# raise out of the resolver at `new_group` before a weight is read (measured on
# the Answer line, run 34085800820, nodes swu10/swu12/swu15). `lo` is correct
# here and only here: this script serves the single-pod entries, whose ranks are
# processes in one network namespace. The multi-node entries must not copy it --
# they derive the interface facing their peers instead.
export GLOO_SOCKET_IFNAME=lo

cd /workspace/source
if [[ -z "${PPU_E2E_PLAN:-}" ]]; then
  git config --global --add safe.directory /workspace/source
fi
SGLANG_PPU_SOURCE_REVISION=$(git rev-parse HEAD)
export SGLANG_PPU_SOURCE_REVISION

# The device list and the checkpoint come out of the config the test will read.
# nproc_per_node is a resource request and not device isolation -- a pod that
# asks for two PPUs still sees all eight and torch still reports device_count 8
# -- so the visible set has to be narrowed here for the preflight, which requires
# the visible count to equal the configured one, to mean anything. It is what
# makes a tp 2 measurement a tp 2 measurement rather than a tp 2 server on a
# board the test never checked it had to itself.
CUDA_VISIBLE_DEVICES=$(python3 -c "import json, os; print(','.join(str(int(device)) for device in json.load(open(os.environ['SGLANG_PPU_PERF_TEST_CONFIG']))['hardware']['visible_devices']))")
export CUDA_VISIBLE_DEVICES
PERF_MODEL_PATH=$(python3 -c "import json, os; print(json.load(open(os.environ['SGLANG_PPU_PERF_TEST_CONFIG']))['model']['path'])")
echo "source revision: $SGLANG_PPU_SOURCE_REVISION"
echo "test config:     $SGLANG_PPU_PERF_TEST_CONFIG"
echo "suite:           $PERF_SUITE"
echo "visible devices: $CUDA_VISIBLE_DEVICES"
echo "checkpoint:      $PERF_MODEL_PATH"

bash scripts/ci/ppu/ppu_install_dependency.sh

# The JIT cache is keyed on source mtime, and a fresh checkout stamps every file
# with the checkout time, so without this every run recompiles.
JIT_MTIME=$(git log -1 --format=%ct 2>/dev/null || echo 0)
if [ "$JIT_MTIME" != 0 ] && [ -d "python/sglang/kernels/jit" ]; then
  find python/sglang/kernels/jit -type f -exec touch -d @"$JIT_MTIME" {} +
  echo 'Pinned JIT source mtimes'
fi

# The JIT build directory moves off the shared NAS cache before anything can
# compile; the script it comes from records why.
# shellcheck source=scripts/ci/ppu/use_local_jit_cache.sh
. scripts/ci/ppu/use_local_jit_cache.sh

# The warm sits immediately before the load, on the node that will do the
# loading, which no step of the orchestration shell could do; the measured host
# reclaims the cache within thirty minutes, so a warm separated from its load is
# wasted machine time. It is the startup budget this protects, not the numbers:
# every measurement runs against an already-loaded server.
bash scripts/ci/warm_page_cache.sh "$PERF_MODEL_PATH"

# Routed through run_suite.py so the executed set is exactly what
# register_ppu_ci declares. No --continue-on-error: one registered file runs
# here, and a suite that cannot measure has to turn the run red -- what this line
# never does is judge a number it did measure.
if [[ -n "${PPU_E2E_PLAN:-}" ]]; then
  exec python3 scripts/ci/ppu/pr_e2e_worker.py suite
fi
cd test
python3 run_suite.py \
  --hw ppu \
  --suite "$PERF_SUITE" \
  --nightly \
  --timeout-per-file "$PERF_TIMEOUT_PER_FILE"
