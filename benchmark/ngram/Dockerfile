# Optional named environment; run_container.sh can use the base directly.
# Build from the repository root. Source and data are mounted by the launcher.
ARG SGLANG_BASE_IMAGE=docker.io/lmsysorg/sglang:v0.5.19
FROM ${SGLANG_BASE_IMAGE}

COPY benchmark/ngram/check_env.py /opt/code-edit/check_env.py
RUN python /opt/code-edit/check_env.py --packages-only

ENV PYTHONPATH=/workspace/sglang/python \
    PYTHONDONTWRITEBYTECODE=1 \
    SGLANG_RUST_BUILD_MODE=never
WORKDIR /workspace/sglang
CMD ["/bin/bash"]
