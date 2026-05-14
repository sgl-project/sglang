# sgl-router CI Parity with SMG

> Surveyed against `.github/workflows/pr-test-rust.yml` (SMG's main CI workflow).
> Rust toolchain pinned in `sgl-model-gateway/rust-toolchain.toml`: **1.90 (minimal + clippy)**.

## SMG job inventory

Jobs confirmed by parsing the workflow YAML:

```
build-wheel
python-unit-tests
unit-tests
gateway-e2e
docker-build-test
k8s-integration
finish
summarize-benchmarks
```

## M0 job parity decision

| SMG job | sgl-router M0 status | Reason |
|---|---|---|
| `build-wheel` | **Skip in v1** | Per spec §9 Fork D1: Docker-only distribution, no Python wheel. |
| `python-unit-tests` | **Skip in v1** | Depends on `build-wheel`; no Python bindings in v1. |
| `unit-tests` (clippy + fmt + `cargo test`) | **Mirror in M0** | Tier-1 + tier-2: lint, format, and unit tests are mandatory from day one. |
| `gateway-e2e` | **Placeholder in M0; real in M1** | No HTTP routes exist yet to exercise end-to-end. |
| `docker-build-test` | **Placeholder in M0; real in M6** | Production Dockerfile is deferred to M6. |
| `k8s-integration` | **Placeholder in M0; real in M2** | Service discovery is deferred to M2. |
| `finish` | **Mirror in M0** | Aggregator job required for branch protection to have a stable required-check name. |
| `summarize-benchmarks` | **Skip until M7** | No benchmarks until M7. |

**Placeholder jobs** emit `echo "placeholder – not yet implemented"` and succeed unconditionally.
This keeps their required-check names stable in branch protection from M0 onward.

## Notable SMG CI implementation details

These facts are worth carrying into sgl-router's workflow design:

- **sccache scoping**: SMG pins `RUSTC_WRAPPER=sccache` only on the `build-wheel` and `unit-tests`
  jobs (GitHub-hosted runners that have sccache). Setting it workflow-wide leaks it to self-hosted
  GPU runners that lack sccache, which breaks any `pip install` that compiles a Rust extension.
  Apply the same scoping discipline in sgl-router.

- **Rust toolchain pin**: 1.90 via `rust-toolchain.toml`. sgl-router will use its own
  `rust-toolchain.toml`; do not assume a channel from the environment.

- **nightly fmt**: `cargo +nightly fmt -- --check` requires a nightly toolchain install step
  alongside the stable pin. Mirror this pattern; do not assume `rustfmt` on the stable toolchain
  is sufficient.

- **gateway-e2e matrix**: four matrix entries across 2-GPU and 4-GPU H100 runners, with a
  `concurrency` group per runner type + matrix name to serialize across in-flight PRs rather than
  cancel. This is the correct pattern for scarce self-hosted GPU runners; reuse it when sgl-router
  adds real E2E jobs in M1.

- **k8s-integration uses a kind cluster** with mock worker pods (no GPU). The test binary avoids
  importing the parent `e2e_test/conftest.py` via `--confcutdir` to dodge heavy infra deps. Keep
  the same isolation pattern in M2.

- **`run-ci` label gate**: most jobs are gated on the `run-ci` PR label in addition to push/dispatch.
  Mirror this so costly jobs don't run on every draft PR.

- **apt mirror flakiness**: `ci_install_gateway_dependencies.sh` retries `apt-get` up to 5 times
  with backoff for Azure Ubuntu mirror flakiness. Use the same retry script or the same pattern in
  sgl-router's dependency install step.

## Coverage gaps to address later

SMG does **not** run these checks; sgl-router should.

| Gap | Priority | Decision |
|---|---|---|
| `cargo deny` — license + advisory supply-chain lint | **Close in M0** | Cheap to add (one TOML + one CI step); closes a real supply-chain gap SMG leaves open. |
| `cargo audit` — known CVE scan against `Cargo.lock` | **Close in M0** | Cheap to add; catches CVEs in transitive deps that `cargo deny` advisory DB may not cover. |
| `cargo machete` — unused-dependency lint | Defer (nice-to-have) | Low risk in a greenfield crate; revisit when dep count grows. |

## M0 gap-closure decisions

Both `cargo deny` and `cargo audit` are added to **tier-1** (fast, GitHub-hosted, no GPU) in M0:

- They are cheap: each is a single `cargo install` + one-line invocation, no extra runner cost.
- They close a real supply-chain gap SMG has left open since its inception.
- Establishing them in M0 means every subsequent milestone is audited from the start rather than
  retroactively cleaning up deps in M6.

Task 6 (workflow file implementation) will add the concrete CI steps for both checks.
