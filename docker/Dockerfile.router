######################## BASE IMAGE ##########################
FROM ubuntu:24.04 AS base

ARG PYTHON_VERSION=3.12

# set the environment variables
ENV PATH="/root/.local/bin:${PATH}"
ENV DEBIAN_FRONTEND=noninteractive

# uv environment variables
ENV UV_HTTP_TIMEOUT=500
ENV VIRTUAL_ENV="/opt/venv"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_LINK_MODE="copy"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


# install dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt update -y \
    && apt install -y curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# install python
RUN uv venv --python ${PYTHON_VERSION} --seed ${VIRTUAL_ENV}

FROM scratch AS local_src
COPY . /src

######################### BUILD IMAGE #########################
FROM base AS build-image

ARG SGLANG_REPO_REF=main
ARG BRANCH_TYPE=remote

# set the environment variables
ENV PATH="/root/.cargo/bin:${PATH}"

# install dependencies
RUN apt update -y \
    && apt install -y git build-essential libssl-dev pkg-config protobuf-compiler \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

# install rustup from rustup.rs
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rustc --version && cargo --version && protoc --version

# pull the github repository or use local source
COPY --from=local_src /src /tmp/local_src
RUN if [ "$BRANCH_TYPE" = "local" ]; then \
        cp -r /tmp/local_src /opt/sglang; \
    else \
        cd /opt \
        && git clone --depth=1 https://github.com/sgl-project/sglang.git \
        && cd /opt/sglang \
        && git checkout ${SGLANG_REPO_REF}; \
    fi \
    && rm -rf /tmp/local_src

# working directory
WORKDIR /opt/sglang/sgl-router

# build the rust dependencies
RUN cargo clean \
    && rm -rf dist/ \
    && cargo build --release \
    && uv build \
    && rm -rf /root/.cache

######################### ROUTER IMAGE #########################
FROM base AS router-image

# Copy the built package from the build image
COPY --from=build-image /opt/sglang/sgl-router/dist/*.whl dist/

# Build the package and install
RUN uv pip install --force-reinstall dist/*.whl

# Clean up unnecessary files to reduce the image size
RUN rm -rf /root/.cache \
    && apt purge -y --auto-remove curl

# Set the entrypoint to the main command
ENTRYPOINT ["python3", "-m", "sglang_router.launch_router"]
