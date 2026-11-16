# syntax=docker/dockerfile:1
# Beam serving image (CUDA). Build from the repository root; application source
# is cloned from upstream:
#   docker buildx build --platform linux/amd64 --load \
#     -f docker/beam/beam.Dockerfile -t beam:local-test .
FROM lmsysorg/sglang:v0.5.22@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Replace the stock SGLang with the upstream beam branch. Pass a commit SHA as
# BEAM_REF to pin the source of a published image.
ARG BEAM_REF=beam
WORKDIR /sgl-workspace
RUN git init -q /sgl-workspace/sglang-beam \
    && git -C /sgl-workspace/sglang-beam fetch -q --depth 1 \
        https://github.com/sgl-project/sglang.git "${BEAM_REF}" \
    && git -C /sgl-workspace/sglang-beam checkout -q --detach FETCH_HEAD \
    && uv pip uninstall sglang \
    && uv pip install --no-deps -e /sgl-workspace/sglang-beam/python

# Pins the beam branch adds on top of the v0.5.22 release: OpenAI 2.8.0 for the
# Responses shell types and the cuTile compiler for the Beam router GEMM. Record
# the new versions so later installs keep them.
RUN sed -i -E '/^(openai|nvidia-cuda-tileiras)==/d' /sgl-workspace/constraints.txt \
    && uv pip install -c /sgl-workspace/constraints.txt \
        openai==2.8.0 nvidia-cuda-tileiras==13.4.92 \
    && uv pip freeze | grep -E '^(openai|nvidia-cuda-tileiras)==' \
        >> /sgl-workspace/constraints.txt \
    && python3 -c "from importlib.metadata import version; assert version('openai') == '2.8.0'; assert version('nvidia-cuda-tileiras') == '13.4.92'"

WORKDIR /sgl-workspace/sglang-beam
