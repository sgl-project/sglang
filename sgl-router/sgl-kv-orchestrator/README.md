# KV public protocol 1.0 (P1)

This crate contains pure schemas, validated action parsing, salt encoding and
logical models. It imports no Router/indexer runtime and performs no I/O, PIN,
routing, session mutation or protection aggregation. `protocol::parse_action`
validates the three new public actions. Other consumers' actions are returned
unchanged; unsupported `sglang.*` names/versions return `Unsupported`. Existing
Engine action parsers and semantics remain in Python.

Envelope stays `0.1`; all three new actions use action version `1.0`.
`fixtures/{prefetch,demote,pin,result,claim}.schema.json` are generated from Python
msgspec contracts. The registered Python manager tests compare these schemas to
current contracts. Both languages consume `actions.json`, `salt.json`, and
`results.json`. Range/dependency rules supplement JSON Schema; use `parse_action`
for validation, rather than deserializing Rust model structs directly.

A complete Engine submission (replace owner/epoch/namespace/layout/page values
with discovery results):

```json
{
  "kind": "submit",
  "context": {
    "engine_id": "worker-b",
    "engine_epoch": "epoch-from-capabilities",
    "namespace": "weights-v1",
    "caller_id": "router-owner"
  },
  "envelope": {
    "protocol_version": "0.1",
    "message_id": "prepare-17",
    "actions": [{
      "action_id": "a",
      "action_type": "sglang.prefetch",
      "action_version": "1.0",
      "payload": {
        "target": {
          "kind": "prefix",
          "compatibility": {
            "namespace": "weights-v1",
            "cache_salt": null,
            "layout_id": "python:full:p4:w0",
            "page_size": 4,
            "components": ["full"],
            "hash_scheme": "sglang.sha256.u32le.i64.v1"
          },
          "start_token": 0,
          "end_token": 4,
          "token_ids": [1, 2, 3, 4]
        },
        "source": {"tier": "storage", "backend": "file"},
        "destination": {"tier": "host", "engine_id": "worker-b"},
        "allow_partial": false,
        "timeout_ms": 5000
      }
    }]
  }
}
```

`sglang.demote` has the same payload fields. PREFETCH permits ascending paths
host→gpu, storage→host/gpu. DEMOTE permits descending paths gpu→host/storage;
host→storage is schema-valid but not executable. Actual paths are the intersection
with the live executor's directions under capabilities `public_protocol.paths`.
The corresponding `public_protocol.actions` map lists only executable actions.
`schema_versions` includes PIN for parsing/discovery, independently of execution.
Storage is currently **file only**; Mooncake execution remains P2. Public action
normalization uses existing `CopyTo`/`MoveTo`; old actions and versions are unchanged.

Locations have `tier` plus nullable `engine_id`, `backend`, `backend_namespace`.
Engine tiers cannot name a backend; storage cannot name an engine. A supplied
backend must match the executor. Non-null `backend_namespace` is unsupported in
P1. Remote Engine IDs are rejected. Raw salt must be sent to the Engine; its
existing transform applies namespace encoding once. The Rust helper matches
`encode_cache_salt`: SHA256 of compact UTF-8 JSON
`["sglang.kv.weight.v1",namespace,original_salt_or_null]`; empty salt maps to null.
Discovery's existing `cache_salt_encoding` distinguishes namespace encoding from
owner assertion. Never encode twice or infer encoding from a salt's shape.

Prefix fields:

- Required compatibility fields: namespace, cache_salt (string/null), layout_id,
  page_size (1–256), unique components (1–64), hash_scheme.
- Required bounds: nonnegative `start_token < end_token`, both page aligned,
  at most 2^53−1. Execution requires start zero and tokens covering exactly
  `[0,end_token)`. Token IDs are u32, at most 65,536 tokens.
- Optional `blocks` (at most 4,096): each has `block_hash` (canonical signed-i64
  decimal string), component, start_token, end_token, dependency_start (default
  zero). Each block spans one page, lies within the prefix, follows earlier
  blocks of its component without overlap, and has an aligned dependency at or
  before its start. These are observations, not authoritative storage keys.
- Token-less descriptors and nonzero-start ranges can be parsed but execution is
  unsupported. Hash-only material never reconstructs full storage digests.
- Legacy Content/Tokens/Session targets and `request_cached` also remain accepted
  selectors. The latter requires a request binding and resolves after checkpoint.

`allow_partial` defaults false. It gates requested logical clipping (legacy
`allow_clip` requires true); it does not make I/O transactional. Physical partial
completion, cancellation and blocked reclamation always report observed effects.
`timeout_ms` is optional, integer 1–86,400,000; omission adds no control deadline.
The owner converts it to a local monotonic deadline. Expiry requests cancellation;
quiescence still requires the executor's drain receipt. No absolute deadline is
accepted. Existing resource limits bound native envelopes, tokens, descriptors,
operations and replay. `MetadataLimits` names future Router metadata budgets;
P1 creates no Router state or runtime configuration defaults.

PIN payload: `mode` acquire/renew/release, `target`, `location`, `claim_id`,
`revision` (0–2^53−1), `ttl_seconds` (finite positive, at most 31,536,000, required
for acquire/renew and absent/null for release), `required_protection`
(eviction_exclusion by default, or backend_soft_pin). Valid PIN always returns
unsupported; it never creates a claim or changes priority. `ClaimState` is a
separate schema with caller_id, task_incarnation, claim_id, revision, state,
owner_epoch, protection_version=1.0, tombstone, effective_protection and
remaining_ttl. These are P4 hooks, with no lifecycle executor.

New action results retain `ref`, `sequence`, `update.status`, `update.quiescent`,
`completed_steps` and target outcomes, and add `public_result`:

- result_version=1.0, requested_scope, accepted_scope, ready_scope,
  ready_location, preserved_scope, reclaimed_scope, missing_scope, blocked_scope;
- owner_epoch, state_version and outcome_sequence (owner registry sequence),
  cache_generation when known, source_observed_at (null until an observation
  source supplies it), and claim (null in P1).

Scope entries use the existing TargetOutcome shape: coverage, branch_index,
component/content identity, status, actual location, completed_steps, code/reason.
Accepted scope requires target-specific evidence and may be empty during RUNNING.
Ready/preserved requires per-target `destination_ready`; reclaimed additionally
requires `source_reclaimed`. Queuing and generic RUNNING never imply readiness.
DEMOTE may report preserved and blocked ranges together. These are historical
operation observations, not current residency leases; P7 will add invalidation
feedback. Old action results omit `public_result` entirely.

Run from the repository root:

```sh
cargo test --manifest-path sgl-router/Cargo.toml -p sgl-kv-orchestrator
PYTHONPATH=python .venv/bin/python -m pytest -q test/registered/unit/managers/test_kv_control.py
```
