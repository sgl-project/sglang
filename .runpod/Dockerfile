FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev \
        python3.11-distutils python3-setuptools build-essential \
        # cuda-cudart-dev-12-1 cuda-compiler-12-1 \
        curl ca-certificates && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# RUN ldconfig /usr/local/cuda-12.1/compat/

# Provide libcuda stub for linking CUDA extensions during build
# RUN ln -sf /usr/local/cuda-12.1/targets/x86_64-linux/lib/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so
# ENV LD_LIBRARY_PATH="/usr/local/cuda-12.1/targets/x86_64-linux/lib/stubs:${LD_LIBRARY_PATH}"

# install uv
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# install general dependencies
COPY .runpod/requirements.txt /
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /requirements.txt

# install sglang + flashinfer-python
# see https://docs.sglang.ai/start/install.html#method-1-with-pip-or-uv
RUN uv pip install --system "sglang[all]>=0.4.6.post3" && \
    uv pip install --system flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.5

# copy files
COPY .runpod /

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=1 

ENV PYTHONPATH="/:/sglang-workspace"

RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
        export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
        python3 /download_model.py; \
    fi



# Start the handler
CMD ["python3", "/handler.py"]
