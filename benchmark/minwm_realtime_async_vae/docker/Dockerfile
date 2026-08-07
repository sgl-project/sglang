ARG PYTHON_IMAGE
ARG GPU_IMAGE

FROM ${PYTHON_IMAGE} AS cpu-runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/sglang/python \
    SGLANG_LIGHTWEIGHT_RUNTIME=1
COPY benchmark/minwm_realtime_async_vae/docker/requirements-cpu.lock /tmp/requirements-cpu.lock
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
      --require-hashes \
      --no-deps \
      --no-cache-dir \
      -r /tmp/requirements-cpu.lock
COPY python/sglang /opt/sglang/python/sglang
WORKDIR /opt/sglang
USER 65532:65532

FROM cpu-runtime AS gateway
LABEL org.opencontainers.image.title="minwm-realtime-gateway"
ENTRYPOINT ["python", "-m", "sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server"]

FROM cpu-runtime AS coordinator
LABEL org.opencontainers.image.title="minwm-realtime-coordinator"
ENTRYPOINT ["python", "-m", "sglang.multimodal_gen.runtime.entrypoints.realtime_coordinator_server"]

FROM ${GPU_IMAGE} AS gpu-runtime
ARG TAEHV_REVISION=093b918971d59001a0bad6dfd6e0409b5e1752cf
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/sglang/python \
    TAEHV_CHECKPOINT_PATH=/opt/taehv/taew2_2.pth
COPY . /opt/sglang
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 - <<'PY'
from pathlib import Path

path = Path("/opt/sglang/python/pyproject.toml")
text = path.read_text()
block = '''[[tool.setuptools-rust.ext-modules]]
target = "sglang.srt.grpc._core"
path = "../rust/sglang-grpc/Cargo.toml"
binding = "PyO3"
'''
if text.count(block) != 1:
    raise RuntimeError("expected exactly one SGLang gRPC extension block")
path.write_text(text.replace(block, ""))
PY
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -e '/opt/sglang/python[diffusion,tracing]' \
      --root-user-action=ignore && \
    python3 -m pip uninstall -y peft && \
    python3 -m pip install --no-cache-dir --no-deps \
      "taehv @ git+https://github.com/madebyollin/taehv.git@${TAEHV_REVISION}" \
      --root-user-action=ignore
COPY benchmark/minwm_realtime_async_vae/docker/requirements-aws.lock /tmp/requirements-aws.lock
RUN python3 -m pip install \
      --require-hashes \
      --no-deps \
      --no-cache-dir \
      -r /tmp/requirements-aws.lock && \
    python3 -c 'import boto3, botocore, jmespath, s3transfer'
RUN install -d -m 0755 /opt/taehv
ADD --checksum=sha256:d053e216ca50e2bb837bbcd79b85f0366bea00e5938025572382a773b74c559a \
    https://raw.githubusercontent.com/madebyollin/taehv/093b918971d59001a0bad6dfd6e0409b5e1752cf/taew2_2.pth \
    /opt/taehv/taew2_2.pth
WORKDIR /opt/sglang

FROM gpu-runtime AS denoiser
LABEL org.opencontainers.image.title="minwm-realtime-denoiser"

FROM gpu-runtime AS vae
LABEL org.opencontainers.image.title="minwm-realtime-vae"
ENTRYPOINT ["python3", "-m", "sglang.multimodal_gen.runtime.entrypoints.realtime_vae_server"]
