# Buf publication and generated SDKs

The root `buf.yaml` declares `proto/` as the public `buf.build/sgl-project/sglang` module. Run all Buf commands from the repository root.

## Local validation

```bash
buf format --diff --exit-code
buf build
buf lint
buf breaking --against '<base-repository-url>#format=git,commit=<base-commit>'
```

Pull requests that change `proto/**`, `buf.yaml`, or `.github/workflows/buf.yml` run the same checks. Breaking-change detection compares with the pull request's target commit.

## Publication policy

- The scheduled workflow publishes current Git `main` to `nightly` every day at 08:00 UTC.
- A manual workflow run is a retry of the nightly publication and is accepted only from Git `main`.
- A protected `v*` Git tag must point to a commit contained in Git `main`. Its workflow publishes one schema commit with both `main` and the exact Git tag as BSR labels.
- Publication runs only in `sgl-project/sglang`, includes a source link to the Git commit, is serialized, and is never canceled in flight.
- CI deliberately omits `buf push --create`. The public BSR module must exist before the first publication.

Treat release-tag labels as append-once. The Git tag ruleset must prevent tag updates and deletion. Buf commits themselves are immutable, but labels, including `main`, `nightly`, and release labels, are mutable pointers; production consumers must pin the BSR commit ID or exact generated-SDK version.

## One-time repository setup

1. Create the public BSR module `buf.build/sgl-project/sglang` with default label `main`.
2. Create a repository-scoped BSR token that can push only this module and store it in the GitHub Actions secret `BUF_TOKEN`.
3. Protect `v*` Git tags so only the release workflow or designated release maintainers can create them, and prevent updates and deletion.
4. Register the generated SDK plugins below. Enable generation for `nightly` after the first nightly push and for `main` after the first release push.

## Generated SDK registrations

The selected Rust plugin versions match the in-tree Prost 0.13 and Tonic 0.12 runtime line. Revisions are pinned so a plugin rebuild cannot silently change generated output.

| SDK             | BSR plugin                              | Version   | Revision | Labels            |
| --------------- | --------------------------------------- | --------- | -------- | ----------------- |
| Rust messages   | `buf.build/community/neoeinstein-prost` | `v0.4.0`  | `1`      | `nightly`, `main` |
| Rust gRPC       | `buf.build/community/neoeinstein-tonic` | `v0.4.1`  | `1`      | `nightly`, `main` |
| Python messages | `buf.build/protocolbuffers/python`      | `v31.1`   | `1`      | `nightly`, `main` |
| Python gRPC     | `buf.build/grpc/python`                 | `v1.72.1` | `1`      | `nightly`, `main` |

The generated package names follow the BSR conventions:

- Cargo: `sgl-project_sglang_community_neoeinstein-prost` and `sgl-project_sglang_community_neoeinstein-tonic`.
- Python: `sgl-project-sglang-protocolbuffers-python` and `sgl-project-sglang-grpc-python`.

Use the generated SDK page to resolve an immutable module commit to an exact package version. Do not express production dependencies as `latest` or as a mutable BSR label. The BSR embeds the plugin version, plugin revision, and schema commit identity in each generated-SDK version; see the [Cargo](https://buf.build/docs/bsr/generated-sdks/cargo/) and [Python](https://buf.build/docs/bsr/generated-sdks/python/) SDK documentation.
