ARG CUDA_VERSION=12.1.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04 AS base

ARG PYTHON_VERSION=3

ENV DEBIAN_FRONTEND=noninteractive

RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-pip \
    && if [ "${PYTHON_VERSION}" != "3" ]; then update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1; fi \
    && python3 --version \
    && python3 -m pip --version \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt-get update -y \
    && apt-get install -y git curl sudo

WORKDIR /sgl-workspace

RUN pip3 --no-cache-dir install --upgrade pip \
    && pip3 --no-cache-dir install --upgrade setuptools wheel \
    && git clone --depth=1 https://github.com/sgl-project/sglang.git \
    && cd sglang \
    && pip --no-cache-dir install -e "python[all]" \
    && pip3 --no-cache-dir install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

ENV DEBIAN_FRONTEND=interactive

# build the wheel
FROM base AS build

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install build

WORKDIR /sgl-workspace/sglang/python
RUN python3 -m build

# minimal sglang srt image
FROM python:3.10-slim AS srt

RUN --mount=type=bind,from=build,src=/sgl-workspace/sglang/python/dist,target=/sgl-workspace/dist \
    --mount=type=cache,target=/root/.cache/pip \
    pip install --verbose $(find /sgl-workspace/dist -name "*.whl" -printf "%p[srt] ") && \
    pip install --index https://flashinfer.ai/whl/cu121/torch2.4/ flashinfer

ENTRYPOINT [ "python3", "-m", "sglang.launch_server" ]
