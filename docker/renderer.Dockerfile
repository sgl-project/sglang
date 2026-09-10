# syntax=docker/dockerfile:1

# Keep the compiler aligned with rust/rust-toolchain.toml. Pin image indexes
# rather than individual architecture manifests so both platforms use this file.
FROM rust:1.92.0-slim-bookworm@sha256:f1f73538ebe623fd3673a35aff3df358ae1084c64c55646516e5b17b321b6c9b AS build

ARG TARGETARCH
ARG CARGO_BUILD_JOBS=4
ENV RUSTUP_TOOLCHAIN=1.92.0 \
    CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS} \
    PCRE2_SYS_STATIC=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY rust/Cargo.toml rust/Cargo.lock rust/rust-toolchain.toml rust/
# Cargo loads every workspace member even when building only the renderer.
COPY rust/sglang-grpc/Cargo.toml rust/sglang-grpc/
COPY rust/sglang-grpc/src/ rust/sglang-grpc/src/
COPY rust/sglang-mm/Cargo.toml rust/sglang-mm/
COPY rust/sglang-mm/src/ rust/sglang-mm/src/
COPY rust/sglang-server/Cargo.toml rust/sglang-server/
COPY rust/sglang-server/src/ rust/sglang-server/src/
COPY rust/sglang-renderer/Cargo.toml rust/sglang-renderer/
COPY rust/sglang-renderer/src/ rust/sglang-renderer/src/

# Avoid rustup downloading development components from rust-toolchain.toml,
# but fail if the image's compiler and the workspace toolchain drift apart.
RUN channel=$(sed -n 's/^channel = "\([^"]*\)"/\1/p' rust/rust-toolchain.toml) \
    && case "${RUSTUP_TOOLCHAIN}" in "$channel"|"$channel".*) ;; *) exit 1 ;; esac

RUN --mount=type=cache,id=renderer-registry-${TARGETARCH},target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=renderer-git-${TARGETARCH},target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=renderer-target-${TARGETARCH},target=/build/rust/target,sharing=locked \
    cargo build --manifest-path rust/Cargo.toml -p sglang-renderer \
        --bin sglang-renderer --release --features http --locked \
    && install -D rust/target/release/sglang-renderer /out/sglang-renderer

# Run the existing unit suite in the same Linux toolchain used for the image.
# This sibling stage is selected by CI and is not a dependency of the runtime.
FROM build AS test
COPY rust/sglang-renderer/tests/ rust/sglang-renderer/tests/
COPY experimental/sgl-router/tests/fixtures/tiny_tokenizer.json experimental/sgl-router/tests/fixtures/tiny_tokenizer.json
RUN --mount=type=cache,id=renderer-registry-${TARGETARCH},target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=renderer-git-${TARGETARCH},target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=renderer-target-${TARGETARCH},target=/build/rust/target,sharing=locked \
    cargo test --manifest-path rust/Cargo.toml -p sglang-renderer --features http --locked

FROM debian:bookworm-slim@sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171 AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgcc-s1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 65532 sglang \
    && useradd --uid 65532 --gid 65532 --no-log-init --create-home \
        --home-dir /home/sglang --shell /usr/sbin/nologin sglang \
    && mkdir -p /home/sglang/.cache/huggingface \
    && chown -R 65532:65532 /home/sglang

COPY --from=build /out/sglang-renderer /usr/local/bin/sglang-renderer
COPY LICENSE /usr/share/licenses/sglang-renderer/LICENSE

# Metadata changes must not invalidate compilation.
ARG SGLANG_BUILD_COMMIT=unknown
ARG SGLANG_BUILD_URL=
ARG SGLANG_IMAGE_TAG=local/sglang-renderer:dev
ENV HOME=/home/sglang \
    HF_HOME=/home/sglang/.cache/huggingface \
    SGLANG_BUILD_COMMIT=${SGLANG_BUILD_COMMIT} \
    SGLANG_BUILD_URL=${SGLANG_BUILD_URL} \
    SGLANG_IMAGE_TAG=${SGLANG_IMAGE_TAG}
LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision="${SGLANG_BUILD_COMMIT}" \
      org.opencontainers.image.version="${SGLANG_IMAGE_TAG}" \
      org.opencontainers.image.url="${SGLANG_BUILD_URL}" \
      ai.sglang.build.commit="${SGLANG_BUILD_COMMIT}" \
      ai.sglang.build.url="${SGLANG_BUILD_URL}" \
      ai.sglang.image.tag="${SGLANG_IMAGE_TAG}"

USER 65532:65532
WORKDIR /home/sglang
EXPOSE 30000
# The renderer's existing graceful shutdown handler listens for Ctrl-C.
STOPSIGNAL SIGINT
ENTRYPOINT ["/usr/local/bin/sglang-renderer"]
CMD ["--help"]
