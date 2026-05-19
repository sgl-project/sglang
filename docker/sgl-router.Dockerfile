# Multi-stage build for sgl-router.
#
# Stage 1 (builder): Compile a release binary against Rust 1.90 (matching
#   `experimental/sgl-router/rust-toolchain.toml`).
# Stage 2 (runtime): Distroless cc-debian12 base — provides libc + minimal
#   shared libs that reqwest's TLS stack needs (we use rustls-only by
#   default, so the surface stays small).
#
# Build:
#   docker build -f docker/sgl-router.Dockerfile -t sgl-router:dev .
# Run:
#   docker run --rm -p 8090:8090 -v $(pwd)/config:/etc/sgl-router \
#       sgl-router:dev --config /etc/sgl-router/sgl-router.yaml
#
# Image budget: < 100 MB stripped (M6 acceptance). Verify with
#   `docker image inspect sgl-router:dev --format '{{.Size}}'`.

ARG RUST_VERSION=1.90
ARG DEBIAN_VERSION=bookworm

######################## STAGE 1 — builder ##########################
FROM rust:${RUST_VERSION}-${DEBIAN_VERSION} AS builder

# Cache dependency builds: copy manifests first, fetch deps, then copy
# sources. A code-only change reuses the cached deps layer.
WORKDIR /work
COPY experimental/sgl-router/Cargo.toml experimental/sgl-router/Cargo.lock ./
COPY experimental/sgl-router/rust-toolchain.toml ./

# Pre-fetch the dependency graph against the pinned toolchain so the
# layer cache survives source-only changes.
RUN mkdir -p src && echo "fn main() {}" > src/main.rs \
    && echo "" > src/lib.rs \
    && cargo fetch --locked \
    && rm -rf src

# Now bring in real source. tests/ and benches/ aren't needed for
# `cargo build --release --bin sgl-router`.
COPY experimental/sgl-router/src ./src

# `--frozen` would refuse a Cargo.lock update; we explicitly want the
# pinned lockfile to drive deps, so use `--locked` instead.
RUN cargo build --release --locked --bin sgl-router \
    && strip target/release/sgl-router

######################## STAGE 2 — runtime ##########################
FROM gcr.io/distroless/cc-debian12:nonroot AS runtime

# distroless is opinionated about location; we install at /usr/local/bin/
# so a hand-written exec invocation finds it without PATH gymnastics.
COPY --from=builder /work/target/release/sgl-router /usr/local/bin/sgl-router

# Default config path; mount your own via `-v <host-path>:/etc/sgl-router`.
ENV SGL_ROUTER_CONFIG=/etc/sgl-router/sgl-router.yaml
EXPOSE 8090

# distroless `nonroot` runs as uid 65532. The router doesn't need root.
USER nonroot:nonroot

# `--config` is the only CLI flag; the env var above feeds it via clap's
# `#[arg(long, env = "SGL_ROUTER_CONFIG")]`.
ENTRYPOINT ["/usr/local/bin/sgl-router"]
