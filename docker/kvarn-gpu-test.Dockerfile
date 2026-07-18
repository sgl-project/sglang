# GPU test container for KVarN integration with SGLang.
# Uses NVIDIA PyTorch base image for GPU + torch + triton support.
#
# Build:  docker build -t kvarn-gpu-test -f docker/kvarn-gpu-test.Dockerfile .
# Run:    docker run --gpus all --rm -v $(pwd):/workspace/sglang kvarn-gpu-test \
#             python -m pytest tests/kvarn/ -v --tb=short

FROM nvcr.io/nvidia/pytorch:25.05-py3

WORKDIR /workspace/sglang

# Install sglang dependencies (lightweight — the base image already has torch + triton)
RUN pip install --no-cache-dir \
    pytest==8.3.4 \
    requests \
    pybase64 \
    transformers \
    huggingface_hub \
    aiohttp

# Install sglang in editable mode
COPY python /workspace/sglang/python
COPY sgl-kernel /workspace/sglang/sgl-kernel
COPY scripts /workspace/sglang/scripts

ENV PYTHONPATH=/workspace/sglang/python

CMD ["bash"]