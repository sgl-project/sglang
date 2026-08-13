# gRPC protocol

This directory is the canonical source for SGLang's gRPC schema.

The schema is published to `buf.build/sgl-project/sglang`:

- A daily workflow publishes the latest Git `main` schema to the `nightly` label.
- The workflow can be run manually to retry nightly publication.
- Tags matching `v*` update the Buf `main` label and publish the corresponding release label.
- Buf commits and generated SDK versions are immutable and can be pinned by consumers.

Repository setup requires the public Buf module and a `BUF_TOKEN` GitHub Actions secret with permission to push it. Register the Rust Prost/Tonic and Python Protobuf/gRPC generated SDKs for the `main` and `nightly` labels once so subsequent pushes generate them automatically.
