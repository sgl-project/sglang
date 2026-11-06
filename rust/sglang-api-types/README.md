# sglang-api-types crate

Rust bindings for SGLang's `sglang.api.v1` API: the prost messages, the tonic
client and server for `SglangService`, and the JSON serde used by SGLang's HTTP
API. The bindings are generated from `proto/sglang/api/v1` by `cargo gen-api`
and checked in, so building the crate needs no `protoc` or build script.
SGLang's own Rust server uses this same crate.

```toml
[dependencies]
sglang-api-types = "0.1"
```

The gRPC client lives at
`sglang_api_types::api::v1::sglang_service_client::SglangServiceClient`.
The public API uses tonic 0.14 and prost 0.14 types.

## Releases and versions

`sglang-api-types` is versioned on its own semver track, independent of SGLang releases.
Releases are made manually:

1. Bump `version` in `Cargo.toml`: the minor version (for example 0.1 to 0.2)
   for any schema, generated-API, or dependency change, since even a new field
   can break Rust callers; the patch version for documentation or metadata only.
2. Once the bump is on main, a maintainer can tag that commit
   `api-types-v<version>` and push the tag. That tag starts
   `release-crate-api-types.yml`, which refuses a tag that differs from the
   `Cargo.toml` version, runs `cargo-semver-checks` against the latest release
   on crates.io, and then uploads the crate.
