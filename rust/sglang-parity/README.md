# SGLang parity

A Rust library and command-line runner for deterministic end-to-end parity tests
between SGLang's Python and Rust serving implementations. The first suite covers
native HTTP `POST /generate`, including JSON and SSE responses.

## Run

Use the committed configuration for your platform directly. Both select a public
Qwen3 model at a fixed revision and define every startup profile used by the suite;
no model path edits or local configuration copy are needed. The runner prepares one
Python 3.12.8 environment and the Rust HTTP extension for both implementations.
It tests a detached snapshot of the calling checkout's exact `HEAD`; commit source
changes first. Python and Rust use the same source, dependencies, model, hardware
options, and request bytes.

From the `rust/` workspace:

```sh
# Apple Silicon / MLX
cargo run --locked -p sglang-parity -- --config sglang-parity/configs/mlx.json

# Linux x86_64 / NVIDIA CUDA
cargo run --locked -p sglang-parity -- --config sglang-parity/configs/cuda.json
```

Append `--describe` to inspect the resolved plan without downloading models,
installing environments or starting services. First execution requires network
access for dependencies and model downloads; later runs reuse verified environment
and Hugging Face caches. No authentication is required for the selected models.

| Configuration | Model | Pinned model revision |
| --- | --- | --- |
| [`configs/mlx.json`](configs/mlx.json) | [mlx-community/Qwen3-0.6B-4bit](https://huggingface.co/mlx-community/Qwen3-0.6B-4bit/tree/73e3e38d981303bc594367cd910ea6eb48349da8) | `73e3e38d981303bc594367cd910ea6eb48349da8` |
| [`configs/cuda.json`](configs/cuda.json) | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) | `c1899de289a04d12100db370d81485cdf75e47ca` |

The two platforms use different model formats; parity compares Python and Rust
on the same platform, not MLX against CUDA. Both configs bound context length to
2048, the token pool to 4096 and concurrent requests to four. CUDA requires the
host prerequisites below; two-device DP cases remain uncovered on a single GPU.
The CUDA config explicitly selects Triton attention, which supports both
deterministic inference and radix caching; automatic backend selection can disable
the cache needed by the cache-hit cases.
Real NVIDIA acceptance of the CUDA configuration is still pending.

The defaults contain no personal paths or forced offline settings. Model caching
uses the standard Hugging Face cache and honors `HF_HOME` / `HF_HUB_CACHE`.
For custom settings, copy the relevant config and pass its path with `--config`.
When changing models, update the `--revision` in every overridden argument list
and review model-specific cases such as the Qwen3 reasoning checks.

The command runs the complete [`native_generate` suite](suites/native_generate/suite.json),
including metadata checks and both streaming modes. No `--suite-file` is needed.
Existing configurations must define every profile referenced by the suite;
missing profiles are configuration errors, not a request to run fewer tests.

Relative `server.python`, `server.working_dir`, and `output_dir` paths resolve from
the invocation directory; a bare executable name such as `python3` uses `PATH`.
`server.working_dir` optionally selects the servers' working directory, where the
server resolves relative model paths and file paths in `server.args`. On a Mac,
use a model supported by SGLang's MLX backend and include `--mlx-enable-sampling` in
`server.args` for the default sampling and output-logprob cases. Backend support
for deterministic inference is required; the runner never retries with it disabled.

For the current MLX path, also set `--prefill-attention-backend torch_native` and
`--decode-attention-backend torch_native` in the shared arguments. These select the
non-Triton cache-allocation path while generation uses MLX and the deterministic
flag remains enabled. Leaving those phases unspecified can select a CUDA backend
during deterministic configuration and fail readiness on a Mac.

The runner sets `--enable-deterministic-inference`, `--random-seed` (default 42),
host `127.0.0.1`, and the configured port. It sets `--disable-radix-cache` unless
the profile declares `radix_cache: true`. These controls cannot be overridden
through extra arguments or environment settings. Metrics and speculative
decoding are outside the current suite. Requests are serial and never
retried; the request timeout covers the whole response, including a stream.

Within each profile, Python starts first with `SGLANG_RUST_SERVER=0`, runs each
case twice, and stops. Rust then runs the same cases with `SGLANG_RUST_SERVER=1`.
Cases can request a fresh process for each attempt. Readiness uses
`/health_generate`. Normal shutdown sends TERM, then KILL after the configured
grace period (at most 60 seconds). Cancelling the run kills the managed process
group and reaps the direct child. The CLI handles both SIGINT and SIGTERM through
this cleanup path. Managed execution supports macOS and Linux.

The suite defines 24 cases: JSON/SSE pairs for greedy generation, one-token
limits, batches, fixed-seed sampling, input/output logprobs, cache hits, weight
versions, reasoning, DP ranks 0 and 1, and retractions. Explicit cumulative and
incremental profile bindings produce 48 case executions. Each available execution
runs twice per implementation, in addition to prerequisites and readiness probes.
On MLX, 32 executions are available and 16 require CUDA. These are the currently
declared scenarios; additional API behavior needs additional cases.
Custom request fields are sent unchanged and their responses are compared in full.
New API features may also require extending the suite's protocol validation and
unit tests; matching responses alone do not prove every requested option was honored.

## Multiple startup profiles in one run

A **profile** names a SGLang startup configuration. A case explicitly lists its
profiles; there is no implicit Cartesian product. One invocation runs all bound
profile/case instances and produces one JSON/HTML report. Python and Rust receive
the same settings and request bytes within each profile. Comparisons and JSON/SSE
equivalence groups never cross profiles.

`server` remains the base configuration and the implicit `default` profile.
Named profiles override only `model`, `seed`, `args`, `env`, and `radix_cache`:

```json
{
  "server": {"model": "/path/to/fixed-model-snapshot", "args": []},
  "profiles": {
    "cached": {"server": {"radix_cache": true}},
    "incremental": {"server": {"args": ["--incremental-streaming-output"]}},
    "dp": {"server": {"args": ["--dp-size", "2"]},
           "requires": {"backends": ["cuda"], "min_cuda_devices": 2}}
  }
}
```

`args` replaces the entire base argument list; `env` overlays keys. Other settings
inherit from the base. There is no profile-to-profile inheritance. `default` is
reserved. Unknown/duplicate bindings and incomplete equivalence groups are
configuration errors. Unreferenced profiles are not started. Execution follows
`default`, then named profiles in key order, with cases in specification order.
The response policy is compiled against each profile's actual streaming mode.

A native case can declare prerequisites and scenario expectations:

```json
{
  "name": "cached_json",
  "profiles": ["cached"],
  "isolation": "fresh_process",
  "before_each": [{
    "body": {"text": "A sufficiently long fixed prefix...",
             "sampling_params": {"temperature": 0, "max_new_tokens": 1}},
    "expect_status": 200
  }],
  "body": {"text": "A sufficiently long fixed prefix...",
           "sampling_params": {"temperature": 0, "max_new_tokens": 8}, "stream": false},
  "expect_status": 200,
  "expectations": [{"check": "cached_tokens", "positive": true}]
}
```

Omitted `profiles` means `["default"]`. `shared` isolation (the default) reuses a
service across consecutive cases. `fresh_process` restarts it for **each attempt**,
then runs `before_each` in order and sends the measured request. Prerequisites
undergo the same protocol validation, without measured-case assertions or
comparison. A failed prerequisite prevents that measured request; its response
and diagnostics are retained. There are no automatic retries or recursive steps.

Cases may also declare `requires`; it intersects the profile's backend requirement
and can raise its device minimum. Backend exclusions are identified before setup;
CUDA device counts come from the environment probe. Unavailable cases remain in
the report as uncovered, with their reason and exit code 2.

### Scenario coverage

All scenarios and their metadata assertions live in
[`suites/native_generate/suite.json`](suites/native_generate/suite.json).
Use [`configs/mlx.json`](configs/mlx.json) on Mac or
[`configs/cuda.json`](configs/cuda.json) on CUDA. Both include every required
profile, including cumulative and incremental settings.

| Scenario | Required evidence | Availability |
| --- | --- | --- |
| Greedy generation | Explicit zero/null defaults, positive timestamp, default weight version and spans | MLX, CUDA |
| One-token limit | A request limited to one generated token | MLX, CUDA |
| Batch | Two prompts with independent results and stream routing | MLX, CUDA |
| Sampling | Fixed-seed sampling with nonzero temperature and top-p | MLX, CUDA |
| Output logprobs | Empty input top logprobs; positive length matching the output logprob array | MLX, CUDA |
| Input logprobs | At least one input top-logprob entry with an actual numeric probability | CUDA; MLX currently rejects prompt logprobs |
| Cache hit | Positive cached tokens and cache details after an identical long-prefix warmup | MLX, CUDA; fresh process per attempt |
| Weight version | `parity-v1` and matching contiguous spans through the final token | MLX, CUDA |
| Reasoning | Positive reasoning count, no larger than completion count | Qwen3 with its reasoning parser |
| DP routing | Explicit rank 0 and rank 1 requests return their selected rank | CUDA, at least two visible devices |
| Retraction | At least one member of a four-request batch has a positive retraction count | CUDA; forced test retraction, fresh process per attempt |

Every scenario includes JSON and SSE and explicitly binds both output modes.
The MLX configuration retains CUDA-only cases as unavailable coverage, so a full
invocation on Mac intentionally exits 2. This does not replace CUDA
acceptance. To run a deliberately narrower suite, edit the explicit bindings and
cases; do not relabel unavailable checks as passing.

Expectations are a finite native-suite vocabulary, implemented in
[`expectations.rs`](suites/native_generate/expectations.rs), not a generic
expression language. They run on the reconstructed response **before** value
exceptions. A Python assertion passing on both attempts proves the target
condition occurred. Rust assertions are checked independently. A scenario
failure does not suppress comparison of a valid response: Rust can fail the
positive-value assertion and also show the corresponding missing-field parity
failure. Existing server differences are results, not expected passing answers.

Library callers use `run_plan(&config, &plan)` for profile bindings;
`run(config, suite, policy)` is the default-only convenience entry into the same
executor.

### Profile artifacts and review

The source snapshot is pinned once for the whole run. Installations are shared by
the existing commit/platform/lock/build-environment key, never by profile name.
Each service combines that prepared interpreter with its own profile settings.
A later installation reads its lock from the pinned snapshot, even if the
original checkout has since changed branches.

```text
<run>/
  effective_plan.json
  report.json / report.html
  source.log
  environments/<installation-key>/...
  profiles/<profile>/<python|rust>/
    logs/server-<number>.log
    <case>/<attempt>/
      request.json / response.body / final.json / events.json
      before_each/<step>/...
```

Each attempt records its actual service log. Reports group by profile and retain
the five-line default case summary. Scenario results remain separate from
response validation and parity. `--case profile/case` selects an execution;
an unqualified name works only when unique. Old reports and the original
single-profile artifact layout remain readable.

## Reproducible environments

Review [`environments/profiles.json`](environments/profiles.json) and the generated
[`mlx.lock`](environments/mlx.lock) / [`cuda.lock`](environments/cuda.lock) for the
installation contract. These are separate from the API suite specification.

| Backend | Host prerequisites | Dependency inputs |
| --- | --- | --- |
| `mlx` | Apple Silicon, macOS 14+, Xcode command-line tools | `python/pyproject_other.toml` base + expanded `srt_mps`; tokenizer compatibility constraint from the default manifest. |
| `cuda` | Linux x86_64, glibc 2.31+, NVIDIA driver supporting CUDA 13.0, C compiler | Default `python/pyproject.toml`; PyTorch `cu130` wheels. |

Both profiles include the local package's declared build dependencies. Install
**uv 0.11.14**, Git, and the Rust toolchain specified by `rust/rust-toolchain.toml`
first. The runner downloads Python **3.12.8** through uv when needed; it does not
install system drivers or compilers. Third-party packages must have suitable
wheels. The CUDA profile adds NVIDIA's official package index with uv's
`first-index` policy, which obtains the genuine `cuda-tile` wheels instead of
building its PyPI downloader stub. Lock generation and installation use the same
index settings, and every installed distribution must match a recorded hash.

Optional `environment` settings in `run.json`:

```json
{
  "environment": {
    "source_root": "/path/to/sglang",
    "backend": "auto",
    "cache_dir": "/path/to/parity-cache",
    "setup_timeout_secs": 1800
  }
}
```

Omit `source_root` to discover the repository from the invocation directory.
`auto` selects MLX on Apple Silicon and CUDA on Linux x86_64. The default cache is
`<source_root>/rust/target/parity-environments`. Relative paths resolve from the
invocation directory. `--describe` prints the commit, profile, lock digest, cache
paths, and effective suite without downloading, installing, building, or starting
services.

The CLI prints progress to stderr by default: source revision, artifact directory,
environment setup or reuse, server readiness, and each case/repeat. Stages that
take longer than ten seconds print periodic elapsed-time updates and the log path.
Detailed installation/build output stays in `setup.log`; service output stays in
each implementation's `server.log`. Set `RUST_LOG=warn` to hide routine progress.
Library callers can collect the same events with their own `tracing` subscriber.
`--describe` keeps stdout as JSON and does not emit execution progress.

The runner rejects staged or unstaged tracked changes and untracked files under
`python/` or `rust/`. Ignored build outputs are permitted. It installs and runs
against a detached Git worktree, checks its revision and cleanliness before and
after each implementation, and removes inherited Python import overrides.
Changing branches in the development checkout after preparation does not change
the tested source.

Environments are keyed by source commit, profile, lock digest, and shared build
settings. Source and environment leases serialize reuse and remain held through
both implementations.
The venv is created at its final path; only successful verification writes the
completion marker. The next attempt rebuilds an incomplete managed venv.
Completed environments are validated without reinstalling dependencies; failed
verification stops the run. Each run checks actual package
versions, source import locations, a device operation, and the Rust loader's
source fingerprint and extension path. Rust builds use the existing loader and
`Cargo.lock`.

To use an existing interpreter, set `server.python`. The runner validates its
Python version and installed third-party packages against the selected lock and
does not install into that environment. SGLang imports still come from the fixed
source snapshot; Rust artifacts use the build cache. On MLX, installed SGLang
metadata can still declare CUDA dependencies, so validation checks the selected
platform lock and real imports instead of running a global dependency check.
Extra installed packages are permitted and included in the recorded inventory;
every applicable locked package must have the required version.

### Why commit dependency locks?

SGLang's existing Python installation flow resolves dependencies from project
declarations at installation time. Parity adds complete third-party dependency
locks so that rebuilding on the same supported platform selects the same package
versions and verifies downloaded distributions against recorded hashes. The
existing declarations remain the inputs; there is no second hand-maintained
dependency list.

Both implementations already share one prepared environment, so full locks are
not required for a fair comparison within a run. They prevent dependency drift
between fresh environments created by different developers or at different times.
Source snapshots and import/build verification separately ensure that both
implementations come from the requested commit.

The cost is larger generated diffs and lock maintenance when dependency inputs
change. Most lockfile content consists of distribution hashes. We retain the
resolver's generated output rather than adding custom artifact filtering. Review
the dependency inputs, selected versions, and platform configuration together;
ordinary test runs never resolve newer versions. Locks fix Python dependencies,
but do not guarantee identical generation results across hardware.

### Updating dependencies

From the desired checkout, with uv 0.11.14 available:

```sh
cargo run -p sglang-parity -- --update-env-lock --backend mlx
cargo run -p sglang-parity -- --update-env-lock --backend cuda
```

These commands resolve the repository declarations and atomically replace one
lock. Ordinary runs only install the locked versions with hash verification;
they never re-resolve dependencies. Dependency/profile changes invalidate the
input digest and require a lock update. Ordinary source changes do not. Commit
updated locks alongside dependency changes. Generated files contain no local
paths, timestamps, or source commit, so regeneration is reviewable.

Model files and GPU drivers are outside the Python lock. The committed run configs
pin model revisions through `--revision`, which is retained in the effective plan.
Custom runs should likewise use a pinned revision or a local snapshot with recorded
provenance. Resolving dependencies successfully is
not evidence of successful device execution or Python/Rust parity.

## Review the test contract

Start with [`suites/native_generate/suite.json`](suites/native_generate/suite.json).
It is the single source for request bodies, profile bindings, prerequisites,
scenario assertions, expected statuses, equivalence groups, and comparison rules.
`--suite-file` optionally replaces this specification for custom tests; it does
not add to the built-in suite. Both use the same loader:

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

| Pointer | Presence | Requirement | Reason |
| --- | --- | --- | --- |
| `/meta_info/id` | Required | Nonempty string | Each request has its own identifier. |
| `/meta_info/e2e_latency` | Required | Finite, nonnegative number | Elapsed time varies. |
| `/meta_info/response_sent_to_client_ts` | Optional | Finite, nonnegative number | Send times vary; streaming results retain this field from the first data event when provided. |

Exception rules default to `"presence": "required"`: missing paths fail validation.
With `"presence": "optional"`, absent fields remain absent; present values still
must satisfy `require`. The core replaces validated values only in comparison
copies. Two absent fields compare equally; presence on only one side remains a
structural difference. `null` is a present value and must pass the declared type
check. Original responses retain every field. `--describe` and the saved effective
suite include the resolved presence policy.

Unknown rules or presence policies, invalid/duplicate pointers, and empty
exception reasons are configuration errors. The supported comparison vocabulary is
`exact_json`, `non_empty_string`, and `non_negative_number`; pointers are literal
JSON Pointers, with no wildcards, scripts, or expressions.

The suite's `streaming.fields` declaration assigns a lifecycle to each complete
response field, using per-result JSON Pointers. The native suite interprets these
rules; the core stores the effective configuration without interpreting it.

| Rule | Native fields / behavior |
| --- | --- |
| `first` | Retain the first-event timestamp; reject late or repeated occurrences. |
| `terminal` | Latency and weight-version spans appear only when the result finishes. |
| `constant` | Request ID and prompt count stay identical within the result. |
| `counter` | Completion, reasoning, cache, retraction and output-logprob counts are nonnegative integers that cannot decrease. |
| `snapshot` | Cache details, DP rank and current weight version are validated per event; retain the terminal snapshot. |
| `text`, `tokens`, `output_logprobs` | Validate cumulative prefix extension or concatenate incremental content. |
| `input_logprobs` | Reconstruct input data without appending repeated prompt logprobs. |
| `finish_reason`, `index` | Validate termination and batch routing. |

Constant, counter and snapshot fields must retain their presence throughout a
result. A field absent from the entire result stays absent, so a Python/Rust
presence mismatch remains a parity difference. Required fields and value
exceptions keep their existing requirements. Unknown streaming fields produce
`streaming rule not covered` with the field path and event index; parity is
`SKIPPED` until the suite declares the lifecycle. This is a missing test rule,
not evidence of a server bug. Members inside declared objects remain intact and
are compared strictly. Custom streaming suites must supply `streaming.fields`;
unknown rule names and invalid pointers fail before services start.

Each batch result has its own first and terminal events. A single data event can
be both. The suite validates all events and the final `[DONE]`, rejects in-band
errors and truncated streams, and checks token/logprob consistency and contiguous
weight-version spans. These rules apply to the current metrics-disabled native
configuration; other APIs need their own response policies.

`final.json` is the complete reconstructed result before value exceptions, not a
copy of the last SSE event. It includes declared first-event fields and accumulated
content. Legitimate event coalescing and interleaving across batch results do not
need to match between implementations. Intermediate snapshots are validated but
are not matched by event ordinal; their terminal values are compared. The raw
body and events remain unchanged as evidence.

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

Each run creates a unique directory under `output_dir` (default `target/parity`),
using the profile layout shown above. Custom single-profile runs retain this
layout:

```text
<run-id>/
  effective_suite.json
  report.json
  report.html              # standalone human-readable diagnostics
  setup.log                # installation/build/verification output
  environment.lock         # exact dependency lock used
  environment-probe.json   # installed packages, device and Rust artifact evidence
  environment.json         # source/profile/cache and successful verification record
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

`report.json` records environment evidence, attempts, validation errors,
repeatability, parity and equivalence differences, field origins, and artifact paths. Each
difference has a JSON path, kind, and both values. Runtime failures and unexecuted
attempts remain visible.
Streaming field origins map reconstructed JSON Pointers to zero-based indices in
`events.json`; nested differences inherit the nearest ancestor's source. Reports
show reconstructed values separately from comparison values and can expand the
source events. Source records contain indices, not copies of events. Old reports
without these records remain readable and keep their saved verdicts.

Interrupted runs preserve a partial report and the bytes received so far.

Reports start with the saved suite, streaming output mode, commit, backend, and
check totals, then group diagnostics by test case. Each case shows its request,
response validation, Python and Rust repeatability, parity, and related case
equivalence. A valid response and a stable implementation can still disagree
with the other implementation. Default CLI output uses five lines per case:

```text
greedy_stream · POST /generate · SSE · cumulative · expected HTTP 200
  Response PASS · Repeat Python PASS / Rust PASS
  Parity FAIL: 8 missing in Rust
  Equivalence: 2/2 passed
  Details: --case greedy_stream · report.html#case-1
```

The compact view includes check statuses, a short parity reason, and a detail
pointer. Long reason previews are explicitly marked as truncated; complete
field differences, response diagnostics, and evidence remain available through
`--case <name>` and HTML. Each case's HTML anchor refers to the report linked at
the end of the CLI output. Long names and paths may wrap in narrow terminals.
Parity difference counts exclude repeatability and equivalence differences.
Equivalence is linked from both related cases but counted once; its HTML details
live under the recorded left-hand case. Counts describe difference occurrences,
not independent bugs. Skipped and unfinished checks include diagnostic reasons.

Open `report.html` for the case directory, expanded failure details, requests,
reconstructed final JSON, recorded comparison rules, and raw response/SSE/log
links. Comparison values are labeled separately from reconstructed values:
value exceptions may replace timestamps with `0`, but a missing field remains
`<missing>`, distinct from JSON `null`. A missing artifact is shown as unavailable
and does not change the recorded test verdict. JSON previews are limited to
64 KiB, with links to complete artifacts. The HTML uses system fonts, follows
the system's light/dark theme, and stacks comparison columns on narrow screens.
Status labels remain meaningful without color. The page has no external
dependencies; copy the entire run directory to preserve relative evidence links.
Each report describes one run. Profile sections identify their cumulative or
incremental output mode; a run can contain both. Missing saved metadata is shown
as unavailable.

Existing results can be viewed without preparing an environment or starting a
service, including after moving the run directory:

```sh
sglang-parity --report target/parity/<run-id>/report.json
sglang-parity --report target/parity/<run-id>/report.json --case greedy_json
```

Both commands regenerate the complete `report.html`; `--case` keeps the run totals
and displays only the selected case, with untruncated comparison and reconstructed
values, source event indices, value-exception rules, and evidence paths. They use saved results
and rules, not the current suite. They return the recorded test exit code (or `2`
if reading/writing the report fails). An unknown case is an error. Report options
cannot be combined with run or environment-maintenance options. Redirected
output and `NO_COLOR` disable terminal colors. Progress remains on stderr; the
final summary goes to stdout. `--describe` remains a JSON-only review operation.

| Exit code | Meaning |
| --- | --- |
| 0 | All responses valid, stable, and equivalent. |
| 1 | Protocol validation, scenario assertion, or parity/equivalence failure. |
| 2 | Configuration, runtime, artifact I/O, cancellation, repeatability, prerequisite, or unavailable-coverage problem. |

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
| `src/plan.rs` | Named startup overrides, requirements, explicit compiled profile/case plans. |
| `src/runner.rs` | Lifecycle order, repeated execution, final comparisons, reports. |
| `src/process.rs` | Shared SGLang configuration and process-group ownership for setup and services. |
| `src/environment.rs` | Source snapshots, platform selection, environment preparation, cache leases and provenance. |
| `src/environment/lock.rs` | Shared dependency expansion, semantic input digests, lock validation and generation. |
| `environments/` | Platform profiles, generated locks and the Python verification probe. |
| `src/http.rs`, `src/sse.rs` | HTTP capture and generic SSE framing. |
| `src/compare.rs` | Strict JSON differences and declared scalar-value exceptions. |
| `src/artifacts.rs` | Atomic artifact storage, without test decisions. |
| `src/report.rs` | Case diagnostics, terminal summaries, and standalone HTML from recorded results. |
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
python3 -m unittest discover -s sglang-parity/environments -p test_probe.py
```

The CPU tests exercise separate contracts:

| Location | Contract |
| --- | --- |
| `src/compare.rs`, `src/sse.rs` | Exact comparison and SSE framing, independent of any API. |
| `suites/native_generate/tests.rs` | Valid native responses reconstruct correctly; malformed responses fail with useful diagnostics. |
| `src/process.rs` | Managed configuration, setup/service subprocesses, readiness, cancellation and cleanup. |
| `src/environment/`, `environments/test_probe.py` | Dependency contracts, cache lifecycle and provenance validation without a GPU. |
| `tests/harness.rs` | Real HTTP capture, repeatability and parity verdicts, equivalence groups, exit codes, and artifacts. |

Parser fixtures are inputs to unit tests, while integration tests use lightweight
local services. These tests check whether the checker can accept valid results and
detect deliberate faults. The default suite against a real model is a separate
acceptance run that compares the actual Python and Rust implementations. Follow
the repository contribution guide and run `pre-commit run --all-files` before submitting.
