# SGLang parity

A Rust library and command-line runner for deterministic end-to-end parity tests
between SGLang's Python and Rust serving implementations. The first suite covers
native HTTP `POST /generate`, including JSON and SSE responses.

## Run

Use the committed configuration for your platform directly. Each selects an embedded
environment with a public Qwen3 model at a fixed revision;
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

### Four explicit run choices

The entire run configuration is:

```json
{
  "environment": "mlx",
  "suites": ["native_generate", "openai_http"],
  "check": "generated-content",
  "output_dir": "target/parity"
}
```

All four fields are required. `environment` is `mlx` or `cuda`; `suites` is a
nonempty list of distinct built-in names, in execution order. `check` applies to
all selected suites. A relative `output_dir` resolves from the invocation directory.
Unknown fields or names, omitted choices, and empty paths are errors.
The committed configs explicitly select `native_generate` and `full-response`.
Copy one to a local run file to change these four choices, then pass `--config`.

Platform settings live in [`configs/environments/`](configs/environments): model,
revision, device and attention settings, resources, port, seed, and timeouts.
Scenario settings live in [`suites/profiles.json`](suites/profiles.json).
These definitions and the suite specifications are compiled into the binary;
changing the invocation directory cannot select different resources. There are no
external definition references, config inheritance, or field overrides. Add new
platforms/scenarios in their owning definition and rebuild. The environment loader
continues to locate source from the calling checkout and use verified caches.
Model downloads use the standard Hugging Face cache (`HF_HOME` / `HF_HUB_CACHE`).

Old run files with inline `server`/`profiles` are rejected with a migration message.
Local model paths, custom startup arguments, Python paths and timeouts in those
files cannot be copied into the four-field format; update the appropriate built-in
definition or use the Rust library's resolved configuration. No old settings are
silently discarded. Historical report JSON remains readable.

### CLI reference

| Options | Purpose |
| --- | --- |
| `--config <path>` | Required run file, including for `--describe`. |
| `--describe` | Print the selection, environment, resolved profiles, requests, rules and service arguments without installation or service startup. |
| `--report <report.json>` | Read recorded verdicts, regenerate HTML and print a summary; no test execution. |
| `--case <name>` | Expand one case, only with `--report`. |
| `--update-env-lock` | Regenerate one dependency lock; cannot be combined with run/report options. |
| `--backend <mlx\|cuda>` | Required only with `--update-env-lock`; does not override a run. |
| `--help`, `-h` | Print usage. |

`--suite`, `--suite-file`, and `--check` are removed. There is no version flag or
environment-variable override for the four run choices. `RUST_LOG` controls logging
and `NO_COLOR` controls presentation, without changing the selected tests.

The runner sets `--enable-deterministic-inference`, `--random-seed` (42 in both environments),
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
Custom request fields are sent unchanged; `full-response` compares the complete response.
New API features may also require extending the suite's protocol validation and
unit tests; matching responses alone do not prove every requested option was honored.

## Choose the check

Both built-in suites support two explicit checks. Set `"check": "full-response"`
or `"check": "generated-content"` in the run file. Neither is implicitly selected.

Append `--describe` to review the selected check without starting services.
The check is recorded in the effective suite and report and applies to
repeatability, Python/Rust parity, and declared JSON/SSE equivalence.

| Check | Compared values | Validation and scenario checks |
| --- | --- | --- |
| `full-response` | Complete JSON or reconstructed SSE response, with the suite's declared value exceptions and precision rules | Full response validation and all configured scenario assertions |
| `generated-content` | Native text; OpenAI Completion choice index and text; Chat choice index, text, reasoning, refusal, and tool-call type, name and raw arguments | Output integrity and content, reasoning, tool-call and refusal assertions |

Generated-content checks omit token IDs, all logprobs, usage, cache statistics,
weight versions, timing, response IDs and other metadata. Empty, null and absent
optional payloads are treated as no generated content. Finish reasons establish
completion and error integrity but their exact values, and matched-stop metadata,
are not compared. Tool arguments remain exact strings; no tool is executed.
Metadata assertions are omitted from this check. An HTTP error or incomplete
output still cannot count as matching generated content.

Generated-content reports explicitly show `Check: Generated content parity` and
`Metadata: Not checked`. A passing content check proves agreement only for the
generated payload. Use `full-response` to check the complete response contract.

## Multiple startup profiles in one run

A **profile** names a SGLang startup configuration. A case explicitly lists its
profiles; there is no implicit Cartesian product. Each selected suite runs all bound
profile/case instances and produces its own JSON/HTML report. Python and Rust receive
the same settings and request bytes within each profile. Comparisons and JSON/SSE
equivalence groups never cross profiles.

Every profile is defined once in [`suites/profiles.json`](suites/profiles.json):

```json
{
  "default": {"args": []},
  "incremental": {"args": ["--incremental-streaming-output"]},
  "cached_incremental": {
    "args": ["--enable-cache-report", "--incremental-streaming-output"],
    "radix_cache": true
  }
}
```

Cases must explicitly name their profiles. `default` is an ordinary empty scenario,
not an automatically inserted or reserved profile. There are no suite-local
profile maps or fallback definitions. Only referenced profiles run, sorted by
name; their cases retain specification order.

Profile `args` **append** to the environment's base arguments. An exact duplicate
option or an option-name prefix ambiguity is rejected, including `--key=value`
versus `--key value`. Profile `env` adds variables and rejects base-key collisions.
An explicit `radix_cache` selects that switch; model and seed remain owned by the
environment. Profiles may require a backend or minimum device count. There are
no generic overrides, argument replacement, or profile inheritance. Controlled
server options still pass through the existing launch validation.
The response policy uses each resolved profile's actual streaming mode.

A native case can declare prerequisites and scenario expectations:

```json
{
  "name": "cached_json",
  "profiles": ["cached_cumulative"],
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

Omitted `profiles` is an error. `shared` isolation (the default) reuses a
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
[`configs/cuda.json`](configs/cuda.json) on CUDA. The suite resolves every bound profile from the shared catalog, including
cumulative and incremental settings.

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
`run(config, suite, policy)` is the single-server convenience entry into the same
executor. Library callers provide explicit resolved conditions; the CLI run file
is a separate, smaller input type.

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
      output.json          # generated-content checks instead of final.json
      before_each/<step>/...
```

Each attempt records its actual service log. Reports group by profile and retain
the five-line default case summary. Scenario results remain separate from
response validation (or output integrity) and parity. `--case profile/case` selects an execution;
an unqualified name works only when unique. Old reports and the original
single-profile artifact layout remain readable.

## Multiple suites

List both built-in names in `suites` to run Native and OpenAI sequentially.
All plans are validated before preparation; one observed commit and one leased
source snapshot are shared throughout. Each suite starts and stops its own
Python/Rust processes. Compatible environment/build/model caches are reused,
but service state and comparisons never cross suites.

A parity or validation failure does not prevent the next suite from running.
Source invalidation, unsafe cleanup, cancellation, or critical artifact I/O errors
stop the batch; remaining suites are `NOT_RUN`. Exit precedence is `2 > 1 > 0`.

A single suite keeps its existing artifact layout. Multiple suites add an index:

```text
<output_dir>/<batch-id>/
  summary.json / index.html
  native_generate/<run-id>/report.json / report.html / ...
  openai_http/<run-id>/report.json / report.html / ...
```

The index links to self-contained suite directories using relative paths.
Each report records the expanded settings and rules, not only their names.
Use `--report` with an individual `report.json` to inspect saved cases.

## Reproducible environments

Review [`environments/profiles.json`](environments/profiles.json) and the generated
[`mlx.lock`](environments/mlx.lock) / [`cuda.lock`](environments/cuda.lock) for the
installation contract. These are separate from the API suite specification.

| Backend | Host prerequisites | Dependency inputs |
| --- | --- | --- |
| `mlx` | Apple Silicon, macOS 14+, Xcode command-line tools | `python/pyproject_other.toml` base + expanded `srt_mps`; tokenizer compatibility constraint from the default manifest. |
| `cuda` | Linux x86_64, glibc 2.31+, NVIDIA driver supporting CUDA 13.0, C compiler | Default `python/pyproject.toml`; PyTorch `cu130` wheels. |

Both dependency profiles include the local package's declared build dependencies. Install
**uv 0.11.14**, Git, and the Rust toolchain specified by `rust/rust-toolchain.toml`
first. The runner downloads Python **3.12.8** through uv when needed; it does not
install system drivers or compilers. Third-party packages must have suitable
wheels. The CUDA profile adds NVIDIA's official package index with uv's
`first-index` policy, which obtains the genuine `cuda-tile` wheels instead of
building its PyPI downloader stub. Lock generation and installation use the same
index settings, and every installed distribution must match a recorded hash.
The selected Python environment's executable directory is prepended to the
server `PATH`, so its locked tools are available to subprocesses. CUDA setup
checks that `ninja` runs before compiling the Rust extension or starting services.

The built-in environments fix the backend and preparation timeout. Source is
located from the invocation directory; the cache uses
`<source_root>/rust/target/parity-environments`. `--describe` prints the actual
commit, dependency lock digest, cache paths and resolved plans without creating
an environment or starting services. These preparation settings belong to the
embedded environment definition, not to the four-field run file.

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
Python bytecode is stored in a sibling `<installation-key>.pycache` directory via
`PYTHONPYCACHEPREFIX`. Environment verification warms the modules it imports;
later service processes reuse them and cache additional modules on first import.
The cache shares the installation's commit, Python, dependency and build identity,
and Python validates cached files against their source before reuse. This also
accelerates existing installations without reinstalling packages. Source snapshots
and user-provided Python environments receive no bytecode files. Removing the
bytecode directory only causes it to be regenerated. Its path is recorded in
`environment.json`; `--describe` does not create it.
The venv is created at its final path; only successful verification writes the
completion marker. The next attempt rebuilds an incomplete managed venv.
Completed environments are validated without reinstalling dependencies; failed
verification stops the run. Each run checks actual package
versions, source import locations, a device operation, and the Rust loader's
source fingerprint and extension path. Rust builds use the existing loader and
`Cargo.lock`.

Rust library callers can use an existing interpreter through `RunConfig.server.python`. The runner validates its
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

Model files and GPU drivers are outside the Python lock. The built-in environments
pin model revisions through `--revision`, which is retained in the effective plan.
Custom runs should likewise use a pinned revision or a local snapshot with recorded
provenance. Resolving dependencies successfully is
not evidence of successful device execution or Python/Rust parity.

## Review the test contract

Start with [`suites/native_generate/suite.json`](suites/native_generate/suite.json).
It is the single source for request bodies, profile bindings, prerequisites,
scenario assertions, expected statuses, equivalence groups, and comparison rules.
The OpenAI API contract is in [`suites/openai_http/suite.json`](suites/openai_http/suite.json).
Change the owning built-in specification and rebuild to add cases or rules.

`--describe` validates and prints the resolved requests, capture modes, comparison
scopes, complete rules and exception reasons, repeats, response implementation,
and cumulative/incremental output mode. It starts no services and makes no network
requests. Execution and reporting use that same resolved object. Runtime settings
cannot override comparison rules, and there are no per-case comparison overrides.

For `full-response`, comparison is strict over the entire JSON tree: object key order is irrelevant;
keys, array order and length, all values, and missing versus `null` matter. There
are no tolerances, text normalization, or implicit field exclusions. Suites may
explicitly declare numeric precision rules below. The native specification
permits only these per-result scalar values to vary:

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

Optional `comparison.per_result_numeric_rules` select numeric values to round
before exact comparison. Each rule declares `path`, `precision: "float32"`, and
a reviewable `reason`. Unlike scalar exceptions, these pointer patterns permit
`*` as a whole path component to select array elements or object members.
Only the selected numbers are converted through `f32`; no epsilon is applied.
Missing paths and `null` stay distinct and unchanged; presence and API types are
still validated by the suite. Non-numeric selected values and numbers that
overflow to infinity fail validation. Tokens, counts, and other unlisted fields
remain exact. Empty rules preserve the original strict behavior.

Precision rules apply to repeatability, Python/Rust parity, and semantic
equivalence. For a semantic projection, paths are relative to its root and
dynamic-value exceptions are not applied. `final.json` and `equivalence.json`
retain the values before rounding; reports show the applied rule alongside
comparison values. Saved reports retain their original verdicts.

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

Each run creates a unique directory under the required `output_dir` (`target/parity` in the committed configs),
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
      output.json          # generated payload after output integrity checks
  rust/
    ...
```

Only the artifact for the selected check is required: `final.json` for
`full-response`, or `output.json` for `generated-content`. Raw response bytes and
SSE events remain available for both checks.

`report.json` records the selected check, environment evidence, attempts, validation errors,
repeatability, parity and equivalence differences, field origins, and artifact paths. Each
difference has a JSON path, kind, and both values. Runtime failures and unexecuted
attempts remain visible.
Streaming field origins map reconstructed JSON Pointers to zero-based indices in
`events.json`; nested differences inherit the nearest ancestor's source. Reports
show reconstructed values separately from comparison values and can expand the
source events. Source records contain indices, not copies of events. Old reports
without these records remain readable and keep their saved verdicts. Reports
without a `check` field default to `full-response`; viewing a report never
reprojects its responses or recomputes its comparisons.

Interrupted runs preserve a partial report and the bytes received so far.

Reports start with the saved suite, selected check, streaming output mode, commit, backend, and
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

For generated-content runs, the same five-line layout uses `Output integrity`
and `Content parity` in place of `Response` and `Parity`. Totals and expanded
details use `Output integrity` instead of `Response validation`, and the report
marks metadata as not checked.

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
reconstructed final JSON or generated output JSON, recorded comparison rules, and raw response/SSE/log
links. Comparison values are labeled separately from reconstructed values:
value exceptions may replace timestamps with `0`, but a missing field remains
`<missing>`, distinct from JSON `null`. Content reports label values from
`output.json` as generated output and retain their source-event origins.
A missing artifact is shown as unavailable
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
`RunConfig`, a resolved `HttpSuite`, and a `ResponsePolicy` that prepares values
for the selected check or returns violations. Use `run_plan` for explicit profiles
and `run_suites` for sequential, independent suites pinned to one source revision. The library owns process cleanup, HTTP capture, declared
comparison rules, and all artifacts; these do not depend on the CLI.

| Location | Responsibility |
| --- | --- |
| `src/config.rs`, `configs/` | Four-field run selection and embedded platform definitions. |
| `suites/profiles.rs`, `suites/profiles.json` | One catalog of scenarios and explicit Case/Profile resolution. |
| `src/plan.rs` | Append-only startup composition, requirements, resolved profile/case plans. |
| `src/runner.rs` | One executor for single/multiple suites, source pinning, lifecycle order, comparisons and reports. |
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
detect deliberate faults. The built-in suites against real models are separate
acceptance run that compares the actual Python and Rust implementations. Follow
the repository contribution guide and run `pre-commit run --all-files` before submitting.

### OpenAI HTTP generation

Set `"suites": ["openai_http"]` in the four-field run file, or include it alongside
`native_generate`. Run with `--config`; append `--describe` for resolved requests
without environment preparation or service startup.

`suites/openai_http/suite.json` defines both `POST /v1/completions` and
`POST /v1/chat/completions`: JSON/SSE, greedy and seeded sampling, multiple
choices, batch prompts, echo, multi-turn chat, logprobs, usage options, and
invalid token limits. It also binds the cached, versioned, reasoning and tool
profiles in both backend streaming modes. Ordinary chat cases disable thinking;
the reasoning cases explicitly enable it. Multimodal inputs, Responses and
embeddings remain outside this suite.

#### Empty and populated response scenarios

The same specification includes `expectations` for real-service scenario checks.
For `full-response`, these run on the original, reconstructed response before comparison exceptions.
A missing positive field or an untriggered stop/cache/tool condition fails the
scenario assertion; it never counts as verified coverage. Protocol-valid responses
remain available for parity even when a scenario assertion fails.

| Field / behavior | Inactive or empty scenario | Populated scenario |
|---|---|---|
| `choices[].logprobs` | Greedy generation without logprobs | Logprobs enabled, with sampled-token probabilities |
| `top_logprobs` | `logprobs=0` / `top_logprobs=0`; no alternatives | Up to two alternatives per generated token |
| Generated text | Explicit stop at the first output text | Greedy generation requires nonempty text |
| `matched_stop` | One token with EOS ignored requires length finish and null | Explicit stop strings require a matching stop and stop finish |
| `message/delta.reasoning_content` | Thinking disabled | Qwen3 reasoning parser; nonempty reasoning required |
| `usage.reasoning_tokens` | Zero when thinking is disabled | Positive count, bounded by generated tokens |
| `usage.prompt_tokens_details` | Cache reporting disabled | Radix cache plus `--enable-cache-report`; warm the exact request prefix before each measured attempt, require positive cached tokens |
| `message/delta.tool_calls` | No tools and `tool_choice=none` | Qwen25 parser, one named function; require the name, parsed arguments and tool-call finish reason |
| `metadata` | Default weight version (already populated) | Explicit `parity-v1`; verify version and nonempty spans in JSON responses |
| Chat logprob `token_id` | Not applicable when logprobs are disabled | Require a nonnegative integer for every sampled token; zero is a valid ID |
| `refusal`, `logprobs.refusal` | Verify no structured refusal in ordinary chat | **Unavailable:** neither current service emits structured refusal payloads; a natural-language refusal is ordinary content |

An inactive feature may legitimately have no payload (missing, null, or an empty
collection). Scenario checks establish that no payload was generated; the full
parity comparison still distinguishes every wire representation. Empty objects
are not fabricated for metadata or usage: these objects have required members
when present. Weight metadata is currently emitted on JSON responses, not SSE.
The paired versioned SSE requests still participate in full parity and equivalence.

Tool arguments are reconstructed by tool-call index. No external tool is executed.
The named call uses an enum-constrained argument to make the scenario reproducible;
if generation does not satisfy it, the report records a scenario failure.
Only the single call's ID is a declared value exception; names and arguments
remain exact. Adding multi-call cases requires declaring the corresponding ID
exceptions as well.

This is a coverage **specification**, not a claim that all server behavior passes.
Check response validation and scenario assertions in each real run. A skipped,
invalid or untriggered case is unverified coverage; unit fixtures cannot establish
E2E acceptance. The nonempty refusal state remains explicitly unsupported until
a service implements it.

OpenAI SSE always contains **deltas**, even when the backend profile uses
cumulative native output. The suite validates every event and reconstructs each
choice by index; different legal event fragmentation is not a parity failure.
Unknown stream semantics fail validation rather than silently disappearing.

The `id: chat_constant` rule enforces a single response ID only for Chat
Completions. For `/v1/completions`, including multiple choices and batch prompts,
IDs may vary across events; each must still be a nonempty string. Reconstruction
keeps the first ID and its event origin, while raw events retain every ID.
Choices are always reconstructed by `index`, never by ID. Creation timestamps
and the other declared constant fields retain their existing checks.

With `"check": "full-response"`, there are two deliberately separate comparisons:

- **Python/Rust parity and repeatability** use the complete JSON or reconstructed
  SSE result in `final.json`. Only declared ID/time values are replaced;
  missing fields, nulls and empty values remain distinct.
- **JSON/SSE equivalence** uses `equivalence.json`: model, indexed content,
  reasoning/refusal text, function tool calls, finish reasons, token logprobs and
  final usage. Absent/null optional text and call collections mean no generated
  payload in this view; their exact shapes remain in full parity. IDs and wire wrappers such as
  message/delta are excluded only from this semantic view. The paired streaming
  request requires final usage. Other fields remain visible to full parity.

With `"check": "generated-content"`, parity and repeatability read `output.json`;
JSON/SSE equivalence uses the same generated-payload contract. It contains
Completion text or Chat content, reasoning/refusal text and function calls,
routed by choice index. Tool-call IDs, logprobs, usage and other metadata are
excluded. Only content-related scenario assertions run. Output integrity still
checks that the response completed and the generated payload can be reconstructed.

Allowing Completion IDs to vary does not relax JSON/SSE result equivalence:
every declared pair still compares each indexed choice's content, reasoning, tool
calls, refusal, finish reason and logprobs, along with the model and complete final usage. Independent JSON
and SSE requests need not produce the same literal ID.

For full-response checks, the OpenAI suite specification explicitly compares log probabilities at `float32`
precision: Chat content/refusal logprobs and their top alternatives, and
Completion `token_logprobs` and `top_logprobs` values. This removes differences
such as `-0.24555965` versus `-0.24555964767932892`, which map to the same `f32`.
Different `f32` values still fail; `token_id`, offsets, usage, nulls and missing
fields keep their exact comparisons.

Reports link each comparison to its own values and event origins. Older reports
continue using their recorded full-response checks and equivalence. Real service
differences within the selected check remain failures; a successful environment
setup does not imply protocol or parity success.
