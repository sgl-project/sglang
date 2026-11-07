# Source-bound native staged execution

This opt-in route uses the public SimOpus UMD prepared-forward API for the
existing DeepSeek V4.1 Flash regional realization. The minimal ordinary
SimOpus provider remains separate. Both use `predictor.name: "simopus"` and
require the current `simopus` package; no old package or selector is accepted.

## Ownership and scope

The simulator owns actual request selection, allocation, overlap/FutureMap
publication, cancellation, output clipping and its explicitly selected progress
policy. SimOpus owns source/profile/FIT loading, ordered P/Q projection, graph
selection, operation lookup, all numerical cost resolution and structured query
evidence. No simulator cost graph or legacy cost fallback is installed.

One worker event requests four EXTEND or five DECODE regional stage results via
`UMDStaticModel.estimate_prepared_forward_result`. Every stage must resolve and
retain its declared partial worker scope before progress is drawn or published.
The adapter checks SDK source, runtime digest, exact ordered inputs and complete
phase coverage, then converts milliseconds to seconds once. Missing or invalid
costs leave the clock, forward ID, history and modeled draft buffer unchanged.
Failure after externally visible publication is terminal; restart the run.

The current bounded route models TP2/TP4, attention-DP1, PP1/EP1/CP1 and static
draft Q5/target Q6. It preserves original physical source/capture identities;
running the simulator on a newer CPU host does not qualify that host's kernels
as new timing evidence. Joint/ADP2, legacy proposal cost engines, dynamic widths,
grammars, stop/EOS-dependent output content and retractions remain excluded.
Synthetic output token IDs require length-only completion with `ignore_eos`.

The factory starts a **cold** modeled draft buffer. It does not accept an
arbitrary warm capacity or the historical `native_initial_draft_buffer` input.
Historical warm/joint experiments remain reproducible from their archived
source; this bounded upstream contribution does not silently import them.
Unavailable cold cost cells remain explicit data gaps.

## Configuration

Install SimOpus with its native prepared-region public API; the integration was
checked against main `e1e0d7a2a89bcd5d1098e2807f591394c020ac21`. Pin the
installed revision in `provider_revision`. Use a reviewed local model config
matching the SDK's authoritative model revision, without downloading weights.

The `predictor` object selects these inputs:

| Input | Meaning |
| --- | --- |
| `name`, `model_id` | `simopus`, `deepseek-ai/DeepSeek-V4.1-Flash` |
| `model_revision`, `provider_revision` | Full lowercase checkpoint and installed SDK commits |
| `systems_root`, `database_mode` | Explicit slice root; `SILICON`, or explicit `HYBRID` with its matching analytic asset |
| `native_segment_profile_id` | Exact original native region profile SHA256 |
| `native_capture_profile` | Original capture profile `{path, sha256}` |
| `native_source_binding` | Optional SDK-verified `{root, profile, cohort, stack}` originals |
| `native_analytic_model` | Optional original `{path, sha256}`; required only for `HYBRID` |
| `native_source_work_policy` | Optional existing versioned SDK policy; does not qualify costs |
| `progress_kind` | Explicit `empirical`, `source_golden351` or `given_mean_acceptance` |
| `progress_artifact` | Original empirical training/heldout `{path, sha256}`; natural TP2 only |
| `given_acceptance` | Authored given-mean condition `{path, sha256}`; only with its named policy |
| `model_seed` | Explicit nonnegative seed for golden/given policies; empirical uses its artifact seed |
| `native_cost_evidence_path` | Optional new exclusive JSONL file for full evaluated SDK documents |
| `model_config_root`, `output_token_id` | Optional local config root and in-vocabulary synthetic token |

Relative asset paths resolve against the configuration directory. Unknown fields
are rejected. TP4 requires source-verified golden or given progress material;
selecting a policy cannot manufacture that material. Given mean acceptance is
an authored condition, not independently predicted acceptance. Cost lookup does
not draw progress. One successful decode event shares its one declared draw
across the batch, retaining native tail clipping separately.

Launch through the normal simulator entry point, with explicit CPU device,
DSPARK, draft block size 5, target width 6, positive max-running-requests,
explicit draft path/revision, static ragged verification and physical graphs
disabled. The simulator uses the original overlap loop; modeled TP is supplied
in `scheduler`, while the host remains one CPU process. The CPU argument adapter
does not relax native calls outside the two explicit simulator bootstrap flags.

## Evidence and verification

`simopus` metrics label the native staged contract and progress identity.
`iteration.jsonl` retains each committed native event with its exact step,
per-stage source references, progress provenance and immutable full evaluated
SDK documents. Evaluated documents remain `committed_event: false`; the separate
event record links them to the actual committed forward ID. Costs evaluated
before a later failure are not counted as committed worker events. An optional
exclusive cost JSONL also preserves those failed-attempt evaluations.

CPU controls include real public SDK projection, unequal ordered P/Q, real native
result/reserve/FutureMap interfaces, result-owned accounting, FIFO and cancelled
tail drains. Synthetic leaf values exercise software contracts only. Neither
passing these controls nor initialization grants timing, silicon or accuracy
acceptance. Use the exact matching qualified data and separately declared
evaluation protocol for numerical accuracy work.

For bounded CPU HTTP controls, select SGLang's existing Python TreeCore with
`SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`. A complete source checkout may
otherwise build its optional Rust extension at startup. This selection tests
the native Python cache policy and does not validate Rust/JIT execution or
represent a performance comparison. Install the current SimOpus SDK explicitly
for its integration controls; the simulator does not install an older SDK.
