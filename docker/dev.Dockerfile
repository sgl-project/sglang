# Dev Dockerfile - builds on top of existing sglang base image
# For faster iteration during development
ARG BASE_IMAGE=lmsysorg/sglang:latest
FROM ${BASE_IMAGE}

# Copy local sglang source (with your changes)
COPY python /opt/sglang/python

# Reinstall sglang from local source
WORKDIR /opt/sglang
RUN pip install -e python --no-deps

# Reset workdir
WORKDIR /

# Default entrypoint
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
