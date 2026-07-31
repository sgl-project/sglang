#!/usr/bin/env bash
set -euo pipefail

SGLANG_ROOT=/workspace/sglang
WORK_ROOT=/work
WEBUI_DIR="${SGLANG_ROOT}/python/sglang/multimodal_gen/apps/realtime_webui"
MODEL_STAMP="${WORK_ROOT}/model/.ready-9be09c5352-sL6CTylRv4QWY98mVTkuLoe5REbKHlvd"
API_PORT="${MINWM_API_PORT:-30000}"
WEBUI_PORT="${MINWM_WEBUI_PORT:-18080}"
RUN_WEBUI="${MINWM_RUN_WEBUI:-true}"

python3 - <<'PY'
from pathlib import Path

path = Path("/workspace/sglang/python/pyproject.toml")
text = path.read_text()
block = """[[tool.setuptools-rust.ext-modules]]
target = "sglang.srt.grpc._core"
path = "../rust/sglang-grpc/Cargo.toml"
binding = "PyO3"
"""
if block in text:
    path.write_text(text.replace(block, ""))
PY

python3 -m pip install -e "${SGLANG_ROOT}/python[diffusion]" \
  --root-user-action=ignore
python3 -m pip uninstall -y peft

if command -v node >/dev/null 2>&1; then
  node "${WEBUI_DIR}/playback_controller_test.js"
  node "${WEBUI_DIR}/realtime_low_latency_defaults_test.js"
fi
# Keep validation on the parity profile.  The serving process below deliberately
# uses the faster non-bitwise profile, and model globals are read at import time.
MINWM_ATTENTION_IMPL=packed \
MINWM_PACKED_ATTENTION_DETERMINISTIC=true \
MINWM_NATIVE_COMPONENTS=text_encoder,vae \
MINWM_SEGMENT_COMPILE=true \
python3 -m pytest -q \
  "${SGLANG_ROOT}/python/sglang/multimodal_gen/test/unit/realtime/test_realtime_output_transport.py" \
  "${SGLANG_ROOT}/python/sglang/multimodal_gen/test/unit/realtime/test_realtime_runtime.py" \
  "${SGLANG_ROOT}/python/sglang/multimodal_gen/test/unit/realtime/test_minwm_realtime.py"

if [[ ! -f "${MODEL_STAMP}" ]]; then
  [[ -f "${WORK_ROOT}/checkpoint/model.pt" ]]
  [[ "$(stat -c '%s' "${WORK_ROOT}/checkpoint/model.pt")" == "20014120667" ]]
  echo "18a48a2709d74b93ce26f0b808f381d191553853aae81dd72d2438430251d379  ${WORK_ROOT}/checkpoint/model.pt" \
    | sha256sum --check
  [[ -d /donor/transformer ]]

  mkdir -p "${WORK_ROOT}/model"
  python3 \
    "${SGLANG_ROOT}/python/sglang/multimodal_gen/tools/convert_minwm_checkpoint.py" \
    --minwm-checkpoint "${WORK_ROOT}/checkpoint/model.pt" \
    --donor-diffusers-dir /donor \
    --output-dir "${WORK_ROOT}/model" \
    --link-donor \
    --source-uri "s3://leap-world-us-west-2/world-model/minwm/checkpoints/run-archive/rolling/Wan21/Action2V/bidirectional/wan22-5B-varlen-multishot-texiao-0725detailed-mix-dccb050-dmd-0724-5eba381389f-merge/global_step_010000/generator/model.pt" \
    --source-version-id "sL6CTylRv4QWY98mVTkuLoe5REbKHlvd" \
    --action-type auto \
    --local-attn-size 32 \
    --sink-size 8 \
    --sliding-window-num-frames 32 \
    --rope-position-mode block_relative \
    --rope-max-frame-gap 12 \
    --prompt-first-frame-pin-enabled
  touch "${MODEL_STAMP}"
fi

# Keep MinWM's 24 FPS contract while retaining the low-latency transport,
# reduced preview size, smaller encoded batches, and stale-frame trimming from
# the shared WebUI implementation.
sed -i \
  -e 's/const DEFAULT_TARGET_FPS = 16;/const DEFAULT_TARGET_FPS = 24;/' \
  -e 's/fps: 16/fps: 24/g' \
  -e 's/"0.05\/frame"/"checkpoint-relative"/g' \
  -e 's/"4deg\/frame"/"checkpoint-relative"/g' \
  -e 's/"6deg\/frame"/"checkpoint-relative"/g' \
  "${WEBUI_DIR}/app.js"
sed -i \
  -e 's/\(Realtime Studio\)\( · MinWM gap12 · B200 Spot\)*/\1 · MinWM gap12 · B200 Spot/g' \
  -e 's/id="fps" type="number" value="16"/id="fps" type="number" value="24"/' \
  -e 's/id="guidance" type="number" value="1"/id="guidance" type="number" value="0"/' \
  -e 's/id="sinkSize" type="number" value="9"/id="sinkSize" type="number" value="8"/' \
  -e 's/id="windowFrames" type="number" value="18"/id="windowFrames" type="number" value="32"/' \
  -e 's/<b>16 fps<\/b>/<b>24 fps<\/b>/' \
  -e 's/<div class="section-title">LingBot<\/div>/<div class="section-title">MinWM<\/div>/' \
  "${WEBUI_DIR}/index.html"

WEBUI_PID=""
if [[ "${RUN_WEBUI}" == "true" ]]; then
  REALTIME_UPSTREAM_HTTP="http://127.0.0.1:${API_PORT}" \
  REALTIME_UPSTREAM_WS="ws://127.0.0.1:${API_PORT}" \
  WEBUI_PORT="${WEBUI_PORT}" \
    python3 "${WEBUI_DIR}/server.py" >"${WORK_ROOT}/logs/webui-${WEBUI_PORT}.log" 2>&1 &
  WEBUI_PID=$!
fi
cleanup() {
  [[ -z "${SERVER_PID:-}" ]] || kill -TERM "${SERVER_PID}" 2>/dev/null || true
  [[ -z "${WEBUI_PID}" ]] || kill -TERM "${WEBUI_PID}" 2>/dev/null || true
  [[ -z "${SERVER_PID:-}" ]] || wait "${SERVER_PID}" 2>/dev/null || true
  [[ -z "${WEBUI_PID}" ]] || wait "${WEBUI_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Product serving profile.  Packed FA4 avoids the severe KV-length scaling of
# dense attention on the gap12 T2V checkpoint.  Keep the small dynamic segment
# compiles, but leave whole-DiT compilation off: graph-breaking around FA4 is
# correct but substantially slower than eager execution for this workload.
# The checkpoint's window=32/sink=8/gap12/prompt-pin contract is unchanged.
MINWM_ATTENTION_IMPL="${MINWM_ATTENTION_IMPL:-packed}" \
MINWM_PACKED_ATTENTION_DETERMINISTIC="${MINWM_PACKED_ATTENTION_DETERMINISTIC:-false}" \
MINWM_NATIVE_COMPONENTS="${MINWM_NATIVE_COMPONENTS:-}" \
MINWM_SEGMENT_COMPILE="${MINWM_SEGMENT_COMPILE:-false}" \
MINWM_CACHE_ROTATED_K="${MINWM_CACHE_ROTATED_K:-true}" \
MINWM_PRECOMPUTE_CACHE_ROPE="${MINWM_PRECOMPUTE_CACHE_ROPE:-true}" \
MINWM_CACHE_PACKED_METADATA="${MINWM_CACHE_PACKED_METADATA:-true}" \
python3 -m sglang.multimodal_gen.runtime.launch_server \
  --model-path "${WORK_ROOT}/model" \
  --pipeline-class-name MinWMCausalDMDPipeline \
  --attention-backend fa \
  --performance-mode speed \
  --num-gpus 1 \
  --sp-degree 1 \
  --enable-cfg-parallel false \
  --enable-torch-compile "${MINWM_ENABLE_TORCH_COMPILE:-false}" \
  --warmup-mode off \
  --realtime-causal-sink-size 8 \
  --realtime-causal-kv-cache-num-frames 32 \
  --host 0.0.0.0 \
  --port "${API_PORT}" &
SERVER_PID=$!
wait "${SERVER_PID}"
