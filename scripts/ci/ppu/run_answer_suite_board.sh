#!/bin/bash
# Run one single-board Answer entry inside the worker pod.
#
# This is the body every entry of test-ppu-answer.yml executes on its board,
# held here rather than in the workflow because that workflow needs one job id
# per concurrent entry -- flytiger-eco/ppu-distributed-action derives both its
# NAS staging path and its K8s job name from $GITHUB_JOB, which every leg of one
# matrix shares -- and a local composite action is not available to those jobs:
# their runner group drives the job through a container hook, so the workspace
# the checkout populates is inside the container while the runner resolves
# `uses: ./` against its own filesystem, where nothing was ever written
# (measured, run 34085102675). Twelve copies of this script inlined into twelve
# jobs would be the only other shape, so what does not vary between entries
# lives here and what does arrives in the environment.
#
# Reads:
#   SGLANG_PPU_ANSWER_TEST_CONFIG  absolute path of the reviewed config, and the
#                                  one source for the visible devices and the
#                                  checkpoint -- a second copy in the workflow
#                                  could drift from what the test reads
#   ANSWER_SUITE                   registered suite name for run_suite.py
#   ANSWER_TIMEOUT_PER_FILE        seconds run_suite.py gives the registered file
#
# Usage: bash /workspace/source/scripts/ci/ppu/run_answer_suite_board.sh
set -e

mkdir -p ~/.pip && cat > ~/.pip/pip.conf << 'PIPEOF'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
PIPEOF

: "${SGLANG_PPU_ANSWER_TEST_CONFIG:?the reviewed config path is required}"
: "${ANSWER_SUITE:?the registered suite name is required}"
: "${ANSWER_TIMEOUT_PER_FILE:?the per-file timeout is required}"

# Constant for every entry, so they are set here rather than repeated in each of
# the twelve jobs. HF_HUB_OFFLINE keeps the tokenizer load off the network on a
# cluster that has none; HF_HUB_CACHE is the shared read-only cache the
# checkpoints' companion repositories were fetched into. PPU_SUPPORTS_FP8=0
# reflects the board, not a preference.
export SGLANG_IS_IN_CI=true
export CPLUS_INCLUDE_PATH=/usr/local/PPU_SDK/targets/x86_64-linux/include
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=/nas_aisw/datasets/hf_cache/hub
export PPU_SUPPORTS_FP8=0
export SGLANG_PPU_ANSWER_INCLUDE_RAW_OUTPUTS=1

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

# Gloo carries the CPU side of every process group SGLang creates, and it picks
# its address by resolving the pod's own hostname. On four of the eight nodes
# this batch landed on, that hostname has no address: six ranks logged torch's
# "Unable to resolve hostname to a (local) address ... Manually set the network
# interface to bind to with GLOO_SOCKET_IFNAME" and fell back to loopback, while
# the other two raised out of the fallback instead, killing three entries at
# `new_group` before any weight was read (measured, run 34085800820, nodes
# swu10/swu12/swu15). Naming the interface takes the resolver out of the path,
# which torch's own message prescribes. `lo` is correct here and only here: this
# script serves the single-pod entries, whose ranks are processes in one network
# namespace. The multi-node entries must not copy it: they derive the interface
# facing their peers with answer_gloo_iface.sh instead.
export GLOO_SOCKET_IFNAME=lo

cd /workspace/source
if [[ -z "${PPU_E2E_PLAN:-}" ]]; then
  git config --global --add safe.directory /workspace/source
fi
SGLANG_PPU_SOURCE_REVISION=$(git rev-parse HEAD)
export SGLANG_PPU_SOURCE_REVISION

# The device list and the checkpoint come out of the config the test will read.
# nproc_per_node is a resource request and not device isolation -- a pod that
# asks for one PPU still sees all eight nodes and torch still reports
# device_count 8 -- so the visible set has to be narrowed here for the
# preflight, which requires the visible count to equal the configured one, to
# mean anything.
CUDA_VISIBLE_DEVICES=$(python3 -c "import json, os; print(','.join(str(int(device)) for device in json.load(open(os.environ['SGLANG_PPU_ANSWER_TEST_CONFIG']))['hardware']['visible_devices']))")
export CUDA_VISIBLE_DEVICES
ANSWER_MODEL_PATH=$(python3 -c "import json, os; print(json.load(open(os.environ['SGLANG_PPU_ANSWER_TEST_CONFIG']))['model']['path'])")
echo "source revision: $SGLANG_PPU_SOURCE_REVISION"
echo "test config:     $SGLANG_PPU_ANSWER_TEST_CONFIG"
echo "suite:           $ANSWER_SUITE"
echo "visible devices: $CUDA_VISIBLE_DEVICES"
echo "checkpoint:      $ANSWER_MODEL_PATH"

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
# wasted machine time.
bash scripts/ci/warm_page_cache.sh "$ANSWER_MODEL_PATH"

# Routed through run_suite.py so the executed set is exactly what
# register_ppu_ci declares. No --continue-on-error: an Answer-quality regression
# must turn the run red.
if [[ -n "${PPU_E2E_PLAN:-}" ]]; then
  export SGLANG_PPU_ANSWER_INCLUDE_RAW_OUTPUTS=0
  exec python3 scripts/ci/ppu/pr_e2e_worker.py suite
fi
cd test
python3 run_suite.py \
  --hw ppu \
  --suite "$ANSWER_SUITE" \
  --nightly \
  --timeout-per-file "$ANSWER_TIMEOUT_PER_FILE"
