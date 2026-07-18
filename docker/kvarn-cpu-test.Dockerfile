# Minimal CPU container for KVarN unit tests (config, sinkhorn, store, dequant).
# No sglang installation required — the test conftest.shim handles imports.
FROM python:3.12-slim

WORKDIR /workspace/sglang

RUN pip install --no-cache-dir --timeout 120 --retries 5 \
    torch==2.6.0 \
    pytest==8.3.4 \
    numpy && \
    pip install --no-cache-dir --timeout 120 --retries 5 triton==3.2.0

CMD ["bash"]