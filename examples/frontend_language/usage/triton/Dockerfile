FROM nvcr.io/nvidia/tritonserver:24.01-py3

WORKDIR /opt

RUN git clone https://github.com/sgl-project/sglang.git

WORKDIR /opt/sglang
RUN pip install --upgrade pip && \
    pip install -e "python[all]" && \
    pip install datasets
