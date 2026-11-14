#!/usr/bin/env bash
# 01_container.sh — start the SGLang NPU container for an N-die slice.
#
# The docker run below mirrors the documented invocation in
# docs/docs/hardware-platforms/ascend-npus/getting-started/installation.mdx and
# model-deployment/tutorials/deepseek_v4_flash.mdx, trimmed to the devices this slice actually has.
#
# Usage:  bash deploy/01_container.sh            # interactive shell
#         bash deploy/01_container.sh --detach   # background, then docker exec
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJ=$(cd "$HERE/.." && pwd)
# shellcheck source=../env/00_common.sh
source "$PROJ/env/00_common.sh"

DETACH=0
[ "${1:-}" = "--detach" ] && DETACH=1

command -v docker >/dev/null 2>&1 || { echo "docker not found. The meetup box normally provides the image; if you are outside the container, ask the TA for the docker run line."; exit 1; }

# ---- devices ---------------------------------------------------------------
DEV_ARGS=()
shopt -s nullglob
for d in /dev/davinci[0-9]*; do DEV_ARGS+=(--device="$d:$d"); done
[ ${#DEV_ARGS[@]} -gt 0 ] || { echo "no /dev/davinci* devices — cannot start a useful container"; exit 1; }
DEV_ARGS+=(--device=/dev/davinci_manager --device=/dev/hisi_hdc)

# ---- weights mount (read-write: some paths create a few cache files) ------
MOUNT_ARGS=()
if [ -d "$(dirname "$MODEL_DIR")" ]; then
  MOUNT_ARGS+=(-v "$(dirname "$MODEL_DIR"):$(dirname "$MODEL_DIR")")
else
  echo "NOTE: $(dirname "$MODEL_DIR") does not exist on the host; creating it so weights can be downloaded."
  mkdir -p "$(dirname "$MODEL_DIR")"
  MOUNT_ARGS+=(-v "$(dirname "$MODEL_DIR"):$(dirname "$MODEL_DIR")")
fi

RUN_ARGS=(
  --privileged=true --net=host --ipc=host --shm-size=16g
  --name "$CONTAINER_NAME"
  -v /var/queue_schedule:/var/queue_schedule
  -v /etc/ascend_install.info:/etc/ascend_install.info
  -v /usr/local/sbin:/usr/local/sbin
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware
  -v "$PROJ:$PROJ"
  "${MOUNT_ARGS[@]}"
  "${DEV_ARGS[@]}"
  -e ASCEND_RT_VISIBLE_DEVICES="$(seq -s, 0 $((NPUS - 1)))"
  -e ASCEND_USE_FIA=1
  -e PORT="$PORT"
  --entrypoint=bash
)

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "container '$CONTAINER_NAME' already exists; starting it"
  docker start "$CONTAINER_NAME" >/dev/null
  if [ "$DETACH" -eq 1 ]; then docker exec -it "$CONTAINER_NAME" bash; fi
  exit 0
fi

echo "image: $IMAGE"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "image not present locally; pulling (this can be slow and needs ~30 GB free disk)"
  docker pull "$IMAGE"
fi

echo "starting container..."
if [ "$DETACH" -eq 1 ]; then
  docker run -itd "${RUN_ARGS[@]}" "$IMAGE" >/dev/null
  echo "started in background: $CONTAINER_NAME"
  echo "enter with: docker exec -it $CONTAINER_NAME bash"
  echo "then: source $PROJ/env/00_common.sh && bash $PROJ/deploy/02_weights.sh"
else
  echo "run this inside the container once it opens:"
  echo "  source $PROJ/env/00_common.sh"
  echo "  bash $PROJ/deploy/02_weights.sh && bash $PROJ/deploy/03_launch.sh"
  exec docker run -it "${RUN_ARGS[@]}" "$IMAGE"
fi
