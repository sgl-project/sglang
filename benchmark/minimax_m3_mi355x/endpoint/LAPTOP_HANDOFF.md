# Finish the MiniMax-M3 MI355X image from a laptop (fresh agent instructions)

State left behind on 2026-09-17 (devbox `kevin-mi-mi350x-8gpu-amd`, the M3-perf checkout at `/sgl-workspace/sglang`):

- Branch `M3-perf`, head `5f94c88e5e` (EAGLE3 GQA, `SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=1`, lenient tool parser, endpoint package in this directory). Not yet pushed to `github.com/kevin-mii/sglang` because the devbox's GitHub token had expired.
- Build context for that head: `/scratch/m3ctx` on the devbox (13 MB) and, once the branch is pushed, reproducible with `bash benchmark/minimax_m3_mi355x/endpoint/make_context.sh <dir>`.
- A CPU build VM `m3-image-build` (rx, EC2 c7i.4xlarge, owner kevin-mi) with the context at `~/ctx` and `docker build -t sglang-ext:m3-mi355x-20260917-5f94c88e5e` running or finished; log at `~/build.log` ending in `BUILD_EXIT=<code>`. It auto-releases 2026-09-18 00:45 UTC.

## 0. Prerequisites on the laptop

```bash
curl -sfL https://nodes.radixark.ai/install-rx.sh | sh && export PATH="$HOME/.local/bin:$PATH" && rx login --google && rx whoami
brew install awscli            # v2; SSO needs v2
aws configure sso --profile radixark   # sso-session rdxa, the org's start URL, region us-west-2, account 976589843892, role DevOpsLead (or an ECR-push role)
aws sso login --profile radixark && aws --profile radixark sts get-caller-identity   # expect account 976589843892
gh auth login -h github.com -p https --web
```

## 1. Refresh the skill and read it

The recipe is `build-sglang-images-cpu-vm` in the `rdxa-skills` plugin (https://github.com/radixark/rdxa_skills, `plugins/rdxa-skills/skills/build-sglang-images-cpu-vm/SKILL.md`). In Claude Code: `/plugin` -> update `rdxa-skills` (or `claude plugin update rdxa-skills`), then invoke the skill. Everything below is that skill's engine-overlay recipe, already instantiated.

## 2. Push the branch (from the devbox checkout, or a clone)

```bash
# on the devbox, after `gh auth login` there, or from any clone with the commits:
git push https://github.com/kevin-mii/sglang.git M3-perf
```

## 3a. If the VM build already finished: verify and push

```bash
VM=m3-image-build; TAG=m3-mi355x-20260917-5f94c88e5e
REG=976589843892.dkr.ecr.us-west-2.amazonaws.com; REPO=radixark/sglang-ext; REMOTE=$REG/$REPO:$TAG
rx devbox run $VM -- bash -c 'grep "^BUILD_EXIT=" ~/build.log; docker images sglang-ext'
rx devbox run $VM -- bash -c "docker run --rm --entrypoint python sglang-ext:$TAG -c 'import sglang.srt.function_call.minimax_m3 as m; assert hasattr(m.MinimaxM3Detector, \"finish\"); import sglang.srt.environ as e; assert hasattr(e.Envs, \"SGLANG_ENABLE_STRICT_MODEL_NAME\"); print(\"OVERLAY_OK\")'"
rx devbox run $VM -- bash -c "docker run --rm --entrypoint bash sglang-ext:$TAG -c 'cd /sgl-workspace/aiter && git diff --stat | tail -1; ls aiter/configs/model_configs | grep minimax_m3_gfx950; grep -c INDEX_TOPK_FREQ:-1 /sgl-workspace/sglang/benchmark/minimax_m3_mi355x/endpoint/serve_endpoint.sh'"
aws --profile radixark --region us-west-2 ecr describe-repositories --repository-names $REPO >/dev/null 2>&1 || aws --profile radixark --region us-west-2 ecr create-repository --repository-name $REPO --image-tag-mutability MUTABLE
aws --profile radixark --region us-west-2 ecr get-login-password | rx devbox run $VM -- bash -c "docker login --username AWS --password-stdin $REG"   # foreground: it reads stdin
rx devbox run $VM -- bash -c "docker tag sglang-ext:$TAG $REMOTE && nohup bash -c 'docker push $REMOTE > ~/push.log 2>&1; echo PUSH_EXIT=\$? >> ~/push.log' >/dev/null 2>&1 </dev/null & echo PUSH_STARTED"
until rx devbox run $VM -- bash -c 'grep -m1 "^PUSH_EXIT=" ~/push.log' | grep -q PUSH_EXIT; do sleep 30; done
rx devbox run $VM -- bash -c 'tail -3 ~/push.log'
aws --profile radixark --region us-west-2 ecr describe-images --repository-name $REPO --image-ids imageTag=$TAG --query 'imageDetails[0].{tags:imageTags,pushedAt:imagePushedAt,digest:imageDigest}'
rx devbox release $VM
```

## 3b. If the VM is gone or the branch moved: rebuild

```bash
git clone -b M3-perf https://github.com/kevin-mii/sglang && cd sglang
bash benchmark/minimax_m3_mi355x/endpoint/make_context.sh /tmp/m3ctx
bash /tmp/m3ctx/build_on_cpu_vm.sh /tmp/m3ctx        # acquires the VM, ships the context, builds, verifies, logs in, pushes, prints the digest
rx devbox release m3-image-build
```

`build_on_cpu_vm.sh` follows the skill step by step (detached build with a `BUILD_EXIT` sentinel, foreground `docker login`, detached push with `PUSH_EXIT`). Base image is pinned by digest to `lmsysorg/sglang-rocm:v0.5.19-rocm724-mi35x-20260911`; the overlay is pure Python plus the aiter FlyDSL patch, so no kernel build happens.

## 4. Run the endpoint from the image

See `ENDPOINT.md` in this directory: `docker run ... -v /models:/models -e MODEL_ROOT=/models -e SGLANG_API_KEY=<key> <image> bash /sgl-workspace/sglang/benchmark/minimax_m3_mi355x/endpoint/serve_endpoint.sh` (models `amd/MiniMax-M3-MXFP4` and `Inferact/MiniMax-M3-EAGLE3-GQA` under `/models`). `SPEC=none` is the shape that passes every vendor gate measured; the default EAGLE3 shape is faster but fails the aime25 non-stop gate.
