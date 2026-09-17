#!/bin/bash
# Build + push the MiniMax-M3 MI355X engine image from a laptop/devbox with `rx` logged in and the AWS `radixark` profile.
# Follows rdxa-skills/build-sglang-images-cpu-vm (engine overlay recipe). Usage: bash build_on_cpu_vm.sh [ctxdir] [tag]
set -euo pipefail
BC=${1:-$(dirname "$0")}; TAG=${2:-m3-mi355x-$(date -u +%Y%m%d)-$(cut -c1-10 "$BC/SGLANG_COMMIT")}
VM=${VM:-m3-image-build}; REG=976589843892.dkr.ecr.us-west-2.amazonaws.com; REPO=radixark/sglang-ext; REMOTE=$REG/$REPO:$TAG
PROFILE=radixark; REGION=us-west-2
rx whoami >/dev/null || { echo "run: rx login --google"; exit 1; }
rx devbox status "$VM" >/dev/null 2>&1 || rx devbox acquire --cloud ec2 --name "$VM" --ttl 4h --disk 300
until rx devbox status "$VM" 2>/dev/null | grep -qi running; do sleep 15; done
rx devbox run "$VM" -- bash -c 'docker version --format "{{.Server.Version}}"; docker buildx version; nproc; df -h / | tail -1'
COPYFILE_DISABLE=1 tar --no-xattrs -czf /tmp/m3ctx.tgz -C "$BC" .
base64 < /tmp/m3ctx.tgz | rx devbox run "$VM" -- bash -c 'cd "$HOME" && rm -rf ctx && mkdir -p ctx && base64 -d | tar xzf - -C ctx && echo FILES=$(find ctx -type f | wc -l)'
# the 26 GB base pull dominates; run detached and poll for the exit sentinel
rx devbox run "$VM" -- bash -c "cd \$HOME/ctx || exit 1
nohup bash -c 'docker build --build-arg SGLANG_COMMIT=\$(cat SGLANG_COMMIT) -t sglang-ext:$TAG . > \$HOME/build.log 2>&1; echo BUILD_EXIT=\$? >> \$HOME/build.log' >/dev/null 2>&1 </dev/null &
echo BUILD_STARTED"
until rx devbox run "$VM" -- bash -c 'grep -m1 "^BUILD_EXIT=" "$HOME/build.log"' 2>/dev/null | grep -q BUILD_EXIT; do sleep 30; done
rx devbox run "$VM" -- bash -c 'tail -5 "$HOME/build.log"'; rx devbox run "$VM" -- bash -c 'grep -q "^BUILD_EXIT=0" "$HOME/build.log"'
# verify the overlay took: our modules import and the detector carries the stream-end flush
rx devbox run "$VM" -- bash -c "docker run --rm --entrypoint python sglang-ext:$TAG -c 'import sglang.srt.function_call.minimax_m3 as m; assert hasattr(m.MinimaxM3Detector, \"finish\"); import sglang.srt.environ as e; assert hasattr(e.Envs, \"SGLANG_ENABLE_STRICT_MODEL_NAME\"); print(\"OVERLAY_OK\")'"
rx devbox run "$VM" -- bash -c "docker run --rm --entrypoint bash sglang-ext:$TAG -c 'cd /sgl-workspace/aiter && git diff --stat | tail -1; ls aiter/configs/model_configs/ | grep minimax_m3_gfx950'"
aws --profile $PROFILE --region $REGION ecr describe-repositories --repository-names "$REPO" >/dev/null 2>&1 || aws --profile $PROFILE --region $REGION ecr create-repository --repository-name "$REPO" --image-tag-mutability MUTABLE
aws --profile $PROFILE --region $REGION ecr get-login-password | rx devbox run "$VM" -- bash -c "docker login --username AWS --password-stdin $REG"
rx devbox run "$VM" -- bash -c "docker tag sglang-ext:$TAG $REMOTE || exit 1
nohup bash -c 'docker push $REMOTE > \$HOME/push.log 2>&1; echo PUSH_EXIT=\$? >> \$HOME/push.log' >/dev/null 2>&1 </dev/null &
echo PUSH_STARTED"
until rx devbox run "$VM" -- bash -c 'grep -m1 "^PUSH_EXIT=" "$HOME/push.log"' 2>/dev/null | grep -q PUSH_EXIT; do sleep 30; done
rx devbox run "$VM" -- bash -c 'tail -3 "$HOME/push.log"; grep -q "^PUSH_EXIT=0" "$HOME/push.log"'
aws --profile $PROFILE --region $REGION ecr describe-images --repository-name "$REPO" --image-ids imageTag="$TAG" --query 'imageDetails[0].{tags:imageTags,pushedAt:imagePushedAt,digest:imageDigest}'
echo "IMAGE: $REMOTE"; echo "release the VM when done: rx devbox release $VM"
