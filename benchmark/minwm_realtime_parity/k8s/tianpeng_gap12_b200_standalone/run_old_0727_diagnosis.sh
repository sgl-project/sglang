#!/usr/bin/env bash
set -euo pipefail

SGLANG_ROOT=/workspace/sglang
WORK_ROOT=/work
CHECKPOINT="${WORK_ROOT}/checkpoint/model.pt"
MODEL_STAMP="${WORK_ROOT}/model/.ready-8de158c6e9"

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

python3 -m pip install -e "${SGLANG_ROOT}/python[diffusion]" --root-user-action=ignore
python3 -m pip uninstall -y peft
python3 -m pip install --force-reinstall --no-deps \
  --index-url https://flashinfer.ai/whl/cu130 \
  'flashinfer-jit-cache==0.6.12+cu130'

test -f "${CHECKPOINT}"
test "$(stat -c '%s' "${CHECKPOINT}")" = "20014135255"
echo "6fa23f07a9b912c76724d14d2b217904bba3854c026d7e0dffcd43861e7c4486  ${CHECKPOINT}" \
  | sha256sum --check --status
if [[ ! -f "${MODEL_STAMP}" ]]; then
  rm -rf "${WORK_ROOT}/model"
  mkdir -p "${WORK_ROOT}/model"
  python3 "${SGLANG_ROOT}/python/sglang/multimodal_gen/tools/convert_minwm_checkpoint.py" \
    --minwm-checkpoint "${CHECKPOINT}" \
    --donor-diffusers-dir /donor \
    --output-dir "${WORK_ROOT}/model" \
    --link-donor \
    --source-uri "s3://leap-world-us-west-2/world-model/minwm/checkpoints/run-archive/rolling/Wan21/Action2V/bidirectional/wan22-5B-varlen-multishot-texiao-addsplithq-da25148-dmd-0724-5eba381389f-merge/global_step_011000/generator/model.pt"
  touch "${MODEL_STAMP}"
fi

mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
python3 -m sglang.multimodal_gen.runtime.launch_server \
  --model-path "${WORK_ROOT}/model" \
  --pipeline-class-name MinWMCausalDMDPipeline \
  --attention-backend fa \
  --performance-mode speed \
  --num-gpus 1 \
  --sp-degree 1 \
  --enable-cfg-parallel false \
  --enable-torch-compile true \
  --warmup-mode off \
  --host 0.0.0.0 \
  --port 30070
