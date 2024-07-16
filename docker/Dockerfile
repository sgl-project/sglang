ARG CUDA_VERSION=12.4.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3

ENV DEBIAN_FRONTEND=noninteractive

RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv python3-pip \
    && if [ "${PYTHON_VERSION}" != "3" ]; then update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1; fi \
    && python3 --version \
    && python3 -m pip --version \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt-get update -y \
    && apt-get install -y python3-pip git curl sudo

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

WORKDIR /sgl-workspace

RUN pip3 --no-cache-dir install --upgrade pip \
    && pip3 --no-cache-dir install "sglang[all]" \
    && pip3 --no-cache-dir uninstall -y triton triton-nightly \
    && pip3 --no-cache-dir install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly \
    && pip3 --no-cache-dir install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/

ENV DEBIAN_FRONTEND=interactive
