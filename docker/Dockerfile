# Build stage
FROM nvcr.io/nvidia/tritonserver:24.04-py3-min AS builder

ARG CUDA_VERSION=12.5.1
ARG BUILD_TYPE=all
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=America/Los_Angeles

RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections && \
    echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    curl \
    git \
    sudo \
    libibverbs-dev \
    rdma-core \
    infiniband-diags \
    openssh-server \
    perftest \
    ibverbs-providers \
    libibumad3 \
    libibverbs1 \
    libnl-3-200 \
    libnl-route-3-200 \
    librdmacm1 && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10 && \
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm -rf /var/lib/apt/lists/* get-pip.py && \
    apt-get clean

# Install Python packages
WORKDIR /sgl-workspace

ARG CUDA_VERSION
RUN pip3 install --no-cache-dir \
    datamodel_code_generator \
    setuptools \
    wheel \
    html5lib \
    six && \
    git clone --depth=1 https://github.com/sgl-project/sglang.git && \
    case "${CUDA_VERSION}" in \
      "12.1.1") export CUINDEX=121 ;; \
      "12.4.1"|"12.5.1") export CUINDEX=124 ;; \
      "11.8.0") \
        export CUINDEX=118 && \
        pip3 install --no-cache-dir sgl-kernel -i https://docs.sglang.ai/whl/cu118 ;; \
      *) echo "Unsupported CUDA version: ${CUDA_VERSION}" && exit 1 ;; \
    esac && \
    pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu${CUINDEX} && \
    pip3 install --no-cache-dir Pillow==11.2.1 && \
    cd sglang && \
    pip3 install --no-cache-dir -e "python[${BUILD_TYPE}]" --find-links https://flashinfer.ai/whl/cu${CUINDEX}/torch2.5/flashinfer-python


RUN rm -f \
  /etc/ssh/ssh_host_ecdsa_key \
  /etc/ssh/ssh_host_ed25519_key \
  /etc/ssh/ssh_host_rsa_key \
  /etc/ssh/ssh_host_ecdsa_key.pub \
  /etc/ssh/ssh_host_ed25519_key.pub \
  /etc/ssh/ssh_host_rsa_key.pub

# Final stage
FROM nvcr.io/nvidia/tritonserver:24.04-py3-min

ENV DEBIAN_FRONTEND=interactive \
    PYTHONUNBUFFERED=1 \
    TZ=America/Los_Angeles

COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /sgl-workspace/sglang /sgl-workspace/sglang

WORKDIR /sgl-workspace
