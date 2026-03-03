# Dev Dockerfile - builds on top of existing sglang base image
# For faster iteration during development
ARG BASE_IMAGE=harbor.xa.xshixun.com:7443/hanfeigeng/lmsysorg/sglang:kv-cache-logging-dev-otel-0.8
FROM ${BASE_IMAGE}

# Copy local sglang source (with your changes)
COPY python /opt/sglang/python
COPY test /opt/sglang/test

# Uninstall base image's sglang and reinstall from local source
WORKDIR /opt/sglang
RUN pip uninstall -y sglang && \
    pip install --no-cache-dir setuptools>=61.0 wheel && \
    pip install transformers==4.57.1 && \
    pip install -e python --no-deps && \
    pip install --no-cache-dir \
    'opentelemetry-sdk>=1.26.0,<1.27.0' \
    'opentelemetry-api>=1.26.0,<1.27.0' \
    'opentelemetry-exporter-otlp>=1.26.0,<1.27.0' \
    'opentelemetry-semantic-conventions-ai>=0.4.1,<0.5.0'

