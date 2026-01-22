#!/usr/bin/env bash
# set -euo pipefail

# 1) 环境
WORKDIR=$(realpath $(dirname $0))
cd "${WORKDIR}"

# Assert branch clean
if [[ -n $(git status --porcelain) ]]; then
    echo "Error: Git branch is not clean. Please commit or stash your changes."
    exit 1
fi

export PYTHONPATH=${WORKDIR}/python
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 2) 模型路径（按需替换）
MODEL_PATH="/inspire/ssd/project/embodied-multimodality/public/openveo3/checkpoints/A14B-360p-wsd-105000"

# 3) 输入/输出
REF_IMAGE="../mossVG/assets/trump.jpeg"
REF_VIDEO="../mossVG/data/samples/trump2.mp4"
OUT_DIR="../mossVG/data/samples"

GIT_HASH=$(git rev-parse --short HEAD)
DATETIME=$(date +%Y%m%d_%H%M%S)
OUT_NAME="trump3_${GIT_HASH}_${DATETIME}.mp4"

# 4) Prompt（与 diffusers_example.sh 同款）
PROMPT=$(cat <<'EOF'
The clip shows a clean, studio-style moment against a plain, softly lit, patriotic blue backdrop. Donald Trump stands centered in frame, facing the camera, wearing his signature ill-fitting navy suit, a crisp white shirt, and an extra-long bright
red tie, with an American flag pin visible on his lapel.

The shot begins with a brief close-up on his face and upper chest. He looks directly into the lens with a confident, pursed-lip expression and says: “Make America Great Again, everyone. I am the greatest player, nobody knows basketball better than
me. I love winning, deals, walls, basketball, Music!” As he emphasizes the last word, a heavy, energetic background track starts playing underneath the scene. The framing then shifts back to a slightly wider, eye-level medium shot, and the camera
subtly pulls back to reveal more of his torso and the open space around him. From the left edge of the screen, a basketball suddenly flies into frame toward the center. He tracks it with his eyes, lifts his hands, and catches it cleanly at chest
height with a soft, muted thump. Without pausing, he adjusts his stance, lowers the ball, and begins playing basketball—moving into a rhythmic dribble that contrasts with his formal suit, the crisp bounce sounds layering over the background music
as he drops into a focused posture.
EOF
)

# 5) 运行（SGLang native）
python -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
    --backend sglang \
    --pipeline-class-name MoVA \
    --model-path "${MODEL_PATH}" \
    --num-gpus 4 \
    --prompt "${PROMPT}" \
    --image-path "${REF_IMAGE}" \
    --output-path "${OUT_DIR}" \
    --output-file-name "${OUT_NAME}" \
    --height 352 \
    --width 640 \
    --num-frames 193 \
    --fps 24 \
    --seed 42 \
    --num-inference-steps 25 \
    --save-output

# 6) Compute PSNR
echo "Computing PSNR against ${REF_VIDEO}..."
ffmpeg -i "${OUT_DIR}/${OUT_NAME}" -i "${REF_VIDEO}" -lavfi psnr -f null - 2>&1 | grep "PSNR"
