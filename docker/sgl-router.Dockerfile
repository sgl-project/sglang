# Multi-stage build for sgl-router.
#
# Three stages, each scoped to its caching contract:
#   1. chef   — generate a `recipe.json` describing the dep graph.
#   2. builder — compile deps from the recipe, then the workspace.
#   3. runtime — distroless cc-debian12 with the stripped binary.
#
# The `cargo-chef` indirection is the canonical Rust multi-stage cache
# pattern: the recipe step's inputs are JUST `Cargo.toml` + `Cargo.lock`,
# so a source-only change produces a recipe-layer cache hit and the
# heavy `cook --release` step is reused untouched. A naive "copy
# manifests → cargo fetch → copy src" approach caches only the fetched
# registry; every source change still recompiles every dep.
#
# `Cargo.lock` is gitignored repo-wide (root .gitignore "# Rust lib"
# block), so we generate it inside the chef stage with `cargo
# generate-lockfile` and propagate that lockfile to the builder via
# `COPY --from=chef`. Both stages thus build against the same lockfile,
# preserving --locked semantics within a single Docker build.
#
# Build (from the repo root):
#   docker build -f docker/sgl-router.Dockerfile -t sgl-router:dev .
# Run:
#   docker run --rm -p 8090:8090 \
#       -v $(pwd)/docker/sgl-router.sample.yaml:/etc/sgl-router/sgl-router.yaml \
#       sgl-router:dev --config /etc/sgl-router/sgl-router.yaml
#
# Image budget: < 100 MB stripped (M6 acceptance). Verify with
#   `docker image inspect sgl-router:dev --format '{{.Size}}'`.

ARG RUST_VERSION=1.90
ARG DEBIAN_VERSION=bookworm

######################## STAGE 1 — chef recipe ##########################
FROM rust:${RUST_VERSION}-${DEBIAN_VERSION} AS chef
RUN cargo install cargo-chef --locked --version ^0.1
WORKDIR /work
COPY experimental/sgl-router/Cargo.toml ./
COPY experimental/sgl-router/rust-toolchain.toml ./
# Stub a minimal src tree so cargo can resolve the workspace, generate
# the lockfile (gitignored upstream), then prepare the chef recipe.
RUN mkdir -p src && echo "fn main() {}" > src/main.rs \
    && echo "" > src/lib.rs \
    && cargo generate-lockfile \
    && cargo chef prepare --recipe-path recipe.json \
    && rm -rf src

######################## STAGE 2 — builder ##############################
FROM rust:${RUST_VERSION}-${DEBIAN_VERSION} AS builder
RUN cargo install cargo-chef --locked --version ^0.1
WORKDIR /work
COPY --from=chef /work/recipe.json ./recipe.json
COPY --from=chef /work/Cargo.lock ./Cargo.lock
COPY experimental/sgl-router/rust-toolchain.toml ./

# Cook (compile + cache) the dep graph from the recipe. This layer's
# inputs are recipe.json + the toolchain — code changes in src/ do NOT
# invalidate it.
RUN cargo chef cook --release --recipe-path recipe.json

# Now bring in the real sources and the manifest they need.
COPY experimental/sgl-router/Cargo.toml ./
COPY experimental/sgl-router/src ./src

# --locked is intentionally omitted: the lockfile is generated in-container
# (gitignored upstream) and `cargo chef cook` may have mutated it during the
# dep-cook step, so a strict --locked check would spuriously fail.
RUN cargo build --release --bin sgl-router \
    && strip target/release/sgl-router

######################## STAGE 3 — runtime ##############################
FROM gcr.io/distroless/cc-debian12:nonroot AS runtime

COPY --from=builder /work/target/release/sgl-router /usr/local/bin/sgl-router

# Default config path; mount your own via `-v <host-path>:/etc/sgl-router`.
ENV SGL_ROUTER_CONFIG=/etc/sgl-router/sgl-router.yaml
EXPOSE 8090

# distroless `nonroot` runs as uid 65532. The router doesn't need root.
USER nonroot:nonroot

ENTRYPOINT ["/usr/local/bin/sgl-router"]
