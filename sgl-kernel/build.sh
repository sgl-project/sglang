#!/bin/bash

set -ex

docker run --rm -it \
    -v "$(pwd)":/sgl-kernel \
    pytorch/manylinux-builder:cuda12.1 \
    bash -c "
    pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 && \
    export TORCH_CUDA_ARCH_LIST='7.5 8.0 8.9 9.0+PTX' && \
    cd /sgl-kernel && \
    python setup.py bdist_wheel
    "
