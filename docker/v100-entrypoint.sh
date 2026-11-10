#!/usr/bin/env bash
# Container entrypoint. The first argument picks a model recipe; each runs the
# same serve script as a host install. Anything else runs as a command.
#
#   glm53            GLM-5.3-Flash, 8 GPUs, port 11435   (GLM53_MODEL)
#   qwen38 [mtp|target]
#                    Qwen3.8-Flash-Next, 4 GPUs, 11435   (FLASH_NEXT_MODEL)
#   dsv41            DeepSeek-V4.1-Flash, 8 GPUs, 11435  (MODEL_PATH)
#   h3 [fl2va|ref2va]
#                    MiniMax-H3 video, 4 GPUs, 30010     (H3_MODEL)
#   smoke            kernel registration check only
#   --flag ...       plain `python -m sglang.launch_server --flag ...`
set -Eeuo pipefail

cd /opt/sglang

smoke() {
  if [[ "${SGLANG_V100_SKIP_STARTUP_CHECK:-0}" != 1 ]]; then
    bash scripts/smoke_v100.sh
  fi
}

case "${1:-}" in
  glm53)
    shift; smoke
    exec bash scripts/serve_glm53_flash_nvfp4_v100.sh "$@" ;;
  qwen38)
    shift; smoke
    [[ $# -gt 0 ]] || set -- mtp
    exec bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh "$@" ;;
  dsv41)
    shift; smoke
    exec bash scripts/serve_dsv41_v100.sh "$@" ;;
  h3)
    shift; smoke
    exec bash scripts/serve_minimax_h3_v100.sh "$@" ;;
  smoke)
    exec bash scripts/smoke_v100.sh ;;
  "" | -*)
    smoke
    exec python -m sglang.launch_server "$@" ;;
  *)
    exec "$@" ;;
esac
