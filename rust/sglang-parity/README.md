# SGLang parity

A Rust library and command-line runner for deterministic end-to-end parity tests
between SGLang's Python and Rust serving implementations. The first suite covers
native HTTP `POST /generate`, including JSON and SSE responses.

## Run

Prepare a SGLang Python environment that can launch both serving implementations,
including the Rust HTTP extension and a model supported by your inference backend.
The runner uses the same environment, model, hardware options, and request bytes
for both implementations. It does not install dependencies or select a device.

From the `rust/` workspace:

```sh
cargo build -p sglang-parity
cp sglang-parity/examples/run.json /path/to/run.json
# Edit the Python executable, model, and shared backend settings.
cargo run -p sglang-parity -- --config /path/to/run.json --describe
cargo run -p sglang-parity -- --config /path/to/run.json
```

Paths in `run.json` are relative to the invocation directory unless absolute.
`server.working_dir` optionally selects the servers' working directory. On a Mac,
use an environment and model supported by SGLang's MLX backend, set
`server.env.SGLANG_USE_MLX` to `"1"`, and include `--mlx-enable-sampling` in
`server.args` for the default sampling and output-logprob cases. Backend support
for deterministic inference is required; the runner never retries with it disabled.

For the current MLX path, also set `--prefill-attention-backend torch_native` and
`--decode-attention-backend torch_native` in the shared arguments. These select the
non-Triton cache-allocation path while generation uses MLX and the deterministic
flag remains enabled. Leaving those phases unspecified can select a CUDA backend
during deterministic configuration and fail readiness on a Mac.

The runner sets `--enable-deterministic-inference`, `--disable-radix-cache`,
`--random-seed` (default 42), host `127.0.0.1`, and the configured port. These controls
cannot be overridden through extra arguments or environment settings. Metrics and
speculative decoding are outside this baseline. Requests are serial and never
retried; the request timeout covers the whole response, including a stream.

Python starts first with `SGLANG_RUST_SERVER=0`, runs each case twice, and stops.
Rust then runs the same cases with `SGLANG_RUST_SERVER=1`. Readiness uses
`/health_generate`. Normal shutdown sends TERM, then KILL after the configured
grace period (at most 60 seconds). Cancelling the run kills the managed process
group and reaps the direct child. The CLI handles both SIGINT and SIGTERM through
this cleanup path. Managed execution supports macOS and Linux.

The default suite has five scenarios, each with streaming off and on: greedy
generation, one-token limit, a two-prompt batch, output logprobs, and fixed-seed
sampling. Ten cases, repeated twice on each side, produce 40 generation requests
plus readiness probes. This is a finite baseline, not exhaustive API coverage.

## Review the test contract

Start with [`suites/native_generate/suite.json`](suites/native_generate/suite.json).
It is the single source for request bodies, expected statuses, equivalence groups,
and comparison rules. The compiled default and an external specification use the
same loader:

```sh
cargo run -p sglang-parity -- --config /path/to/run.json \
  --suite native_generate --suite-file /path/to/suite.json --describe
```

`--describe` validates and prints the resolved requests, capture modes, comparison
scopes, complete rules and exception reasons, repeats, response implementation,
and cumulative/incremental output mode. It starts no services and makes no network
requests. Execution and reporting use that same resolved object. Runtime settings
cannot override comparison rules, and there are no per-case comparison overrides.

Comparison is strict over the entire JSON tree: object key order is irrelevant;
keys, array order and length, all values, and missing versus `null` matter. There
are no tolerances, text normalization, or implicit field exclusions. The default
specification permits only these per-result scalar values to vary:

| Pointer | Requirement | Reason |
| --- | --- | --- |
| `/meta_info/id` | Nonempty string | Each request has its own identifier. |
| `/meta_info/e2e_latency` | Finite, nonnegative number | Elapsed time varies. |

Every declared path must exist on every result, even if both implementations omit
it. The core validates values before replacing them in comparison copies. Original
responses retain every field. Unknown rules, invalid/duplicate pointers, and empty
exception reasons are configuration errors. The supported rule vocabulary is
`exact_json`, `non_empty_string`, and `non_negative_number`; pointers are literal
JSON Pointers, with no wildcards, scripts, or expressions.

The suite's Rust response policy validates the native protocol and reconstructs
full final JSON. It cannot add comparison exceptions. Cumulative streams use final
snapshots; incremental streams accumulate text, output tokens, and output
logprobs. Batch results return to input order. The policy checks indices, per-result
IDs, sequence progression, terminal results, and the final `[DONE]`. It rejects
in-band errors and truncated streams. Legitimate event coalescing and interleaving
across batch results do not need to match between implementations. Raw events are
retained for diagnosis.

First, each side must produce valid responses and repeat its own final result.
Unstable results are marked `UNSTABLE` and skip a definite parity conclusion.
Stable results are compared between Python and Rust. Declared equivalence groups
also compare streaming and nonstreaming final results within each implementation,
reusing the captured responses. Two repeats are a finite stability check, not a
proof of determinism under all scheduling conditions.

Negative HTTP tests use a separate specification with explicit 4xx/5xx statuses,
JSON capture, and an empty exception list. Successful and error contracts cannot
be mixed in one specification. An error inside a successful SSE response remains
a protocol failure.

## Results

Each run creates a unique directory under `output_dir` (default `target/parity`):

```text
<run-id>/
  effective_suite.json
  report.json
  python/
    server.log
    <case>/<repeat>/
      request.json
      response.body
      events.json          # SSE captures, including partial captures
      final.json           # complete result after successful API validation
  rust/
    ...
```

`report.json` records attempts, validation errors, repeatability, parity and
equivalence differences, and artifact paths. Each difference has a JSON path,
kind, and both values. Runtime failures and unexecuted attempts remain visible.
Interrupted runs preserve a partial report and the bytes received so far.

| Exit code | Meaning |
| --- | --- |
| 0 | All responses valid, stable, and equivalent. |
| 1 | Protocol validation or parity/equivalence failure. |
| 2 | Configuration, runtime, artifact I/O, cancellation, or repeatability problem. |

Code 2 takes precedence over code 1. Missing dependencies are not passing results.
Differences exposed in the existing servers should be investigated independently;
the test does not widen its exceptions to conceal them.

## Library and module boundaries

Call `sglang_parity::run(&config, &suite, &policy).await` from Rust. A caller supplies
`RunConfig`, a resolved `HttpSuite`, and a `ResponsePolicy` that returns complete
final JSON or violations. The library owns process cleanup, HTTP capture, declared
comparison rules, and all artifacts; these do not depend on the CLI.

| Location | Responsibility |
| --- | --- |
| `src/runner.rs` | Lifecycle order, repeated execution, final comparisons, reports. |
| `src/process.rs` | Shared SGLang configuration and managed service lifetime. |
| `src/http.rs`, `src/sse.rs` | HTTP capture and generic SSE framing. |
| `src/compare.rs` | Strict JSON differences and declared scalar-value exceptions. |
| `src/artifacts.rs` | Artifact storage, without test decisions. |
| `suites/native_generate/` | API cases, native response validation and reconstruction. |
| `cli/main.rs` | Configuration loading, suite selection, presentation, exit status. |

`src/` does not reference concrete suites. The CLI compiles native generation via
an explicit module path, and the suite depends on public core APIs. Response
fixtures in the suite are inputs for parser/reconstruction unit tests; e2e tests
always compare live responses from the two implementations, not stored answers.

To support another HTTP API, add a suite specification and response policy, then
register it in the CLI. Future gRPC support will add a transport, request and
observation variants, and the runner's single capture dispatch. It must establish
the actual comparable serving paths and readiness contract; today's HTTP switch
does not imply support for every Python/Rust gRPC combination. No gRPC client or
placeholder plugin framework is included.

## Development

From `rust/`:

```sh
cargo fmt --all -- --check
cargo clippy -p sglang-parity --all-targets -- -D warnings
cargo test -p sglang-parity
cargo doc -p sglang-parity --no-deps
```

CPU tests cover the comparison rules, SSE framing, API reconstruction, and managed
runner failure paths. Integration tests use lightweight local services; the
default suite against a real model is a separate acceptance run. Follow the
repository contribution guide and run `pre-commit run --all-files` before submitting.
