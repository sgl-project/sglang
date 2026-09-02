# Bake GLM-5.3-Flash tokenizer + chat template into the sgl-router image.
#
# The router auto-downloads tokenizer.json from HF at startup unless
# --tokenizer-path is given; we pre-bake it (and the chat template) so the
# image is hermetic and starts instantly without an outbound HF call.
#
# Build (from this directory, after building the base router image):
#   docker build -f ../sgl-router.Dockerfile \
#     -t sgl-router:combine-router-admission-http2-loadaware ../..
#   curl -sL -o tokenizer.json \
#     https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/tokenizer.json
#   docker build -f Dockerfile -t sgl-router:glm53-flash-combine .
#
# tokenizer.json is ~20 MB and is NOT committed; fetch it from HF before
# building. tokenizer_config.json already has the chat_template inlined —
# see README.md.
#
# Usage at runtime:
#   sgl-router --model-id glm-5.3-flash \
#     --tokenizer-path /tokenizer/tokenizer.json \
#     --worker-urls http://worker-1:8000 ...

ARG ROUTER_BASE=sgl-router:combine-router-admission-http2-loadaware
FROM ${ROUTER_BASE}

# Switch to root just to drop the files in, then back to nonroot (matches
# upstream's USER nonroot).
USER root
COPY tokenizer.json        /tokenizer/tokenizer.json
COPY tokenizer_config.json /tokenizer/tokenizer_config.json
COPY chat_template.jinja   /tokenizer/chat_template.jinja
USER nonroot:nonroot

# Convenience envs so the deployment can wire these via $(VAR) substitution.
ENV SGL_ROUTER_TOKENIZER_PATH=/tokenizer/tokenizer.json
ENV SGL_ROUTER_CHAT_TEMPLATE=/tokenizer/chat_template.jinja
