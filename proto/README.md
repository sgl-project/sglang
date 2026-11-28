# SGLang protobuf schemas

The protobuf definitions of SGLang's gRPC APIs. CI publishes this directory to
the Buf Schema Registry as
[`buf.build/sgl-project/sglang`](https://buf.build/sgl-project/sglang).

| Package | Contents |
|---------|----------|
| `sglang.api.v1` | The native `SglangService` gRPC API, which shares its request and response messages with the HTTP API. |
| `sglang.json.v1` | Custom options that define the HTTP JSON form of `sglang.api.v1` messages. |
| `sglang.runtime.v1` | The gRPC bridge protocol. |

## Generated SDKs

Use the [generated SDKs](https://buf.build/sgl-project/sglang/sdks) instead of
copying these files. For example, in Go:

```bash
go get buf.build/gen/go/sgl-project/sglang/protocolbuffers/go@vX.Y.Z
go get buf.build/gen/go/sgl-project/sglang/grpc/go@vX.Y.Z
```

Each published commit is labeled with its Git commit SHA. Commits on `main`
are also labeled `main`, and release tags are labeled with their tag name (for
example `v0.5.24`). Pin a release tag or commit SHA rather than `main`.

## gRPC wire format only

The generated SDKs are meant for gRPC (the binary protobuf wire format). The
HTTP JSON form of these messages is **not** the canonical proto3 JSON mapping:
the `sglang.json.v1` options change it (bare lists, positional tuples,
scalar-or-list unions, embedded raw JSON). Do not use `protojson` or
`json_format` on these types to build or parse SGLang HTTP request and
response bodies.

## Changing the schemas

- `buf lint` and `buf breaking` (the `FILE` rules in `buf.yaml`) run on every
  pull request that touches this directory.
- A deliberate breaking change needs maintainer sign-off: a maintainer applies
  the `buf skip breaking` label to the pull request. The same check runs again
  before publishing from `main`, and it honors the same label.
- The checked-in Rust and Python bindings are generated from these files. After
  editing a `.proto` file, run `cd rust && cargo gen-api` and commit the
  regenerated output with the schema change.
