# Dev Dockerfile - builds on top of existing sglang base image
# For faster iteration during development
ARG BASE_IMAGE=lmsysorg/sglang:latest
FROM ${BASE_IMAGE}

# Copy local sglang source (with your changes)
COPY python /opt/sglang/python

# Uninstall base image's sglang and reinstall from local source
WORKDIR /opt/sglang
RUN pip uninstall -y sglang && \
    pip install transformers==4.57.1 && \
    pip install -e python --no-deps

# Reset workdir
WORKDIR /
