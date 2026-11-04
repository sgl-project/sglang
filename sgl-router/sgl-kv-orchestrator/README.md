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
Storage adapters support **file** and the P2 **TP1 FULL MHA, page_first_direct Mooncake** profile. Public action
normalization uses existing `CopyTo`/`MoveTo`; old actions and versions are unchanged.

Locations have `tier` plus nullable `engine_id`, `backend`, `backend_namespace`.
Engine tiers cannot name a backend; storage cannot name an engine. A supplied
backend must match the executor. A supplied `backend_namespace` must equal the live capability value; mismatches
return `incompatible_namespace`. Omission retains the legacy bound-backend behavior. Remote Engine IDs are rejected. Raw salt must be sent to the Engine; its
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


## P2 storage execution and physical target descriptions

`public_protocol` additionally exposes `backend_namespace`,
`backend_namespace_selection`, `storage_targets`, and `storage_profile`.
The Mooncake profile is `tp1_full_mha_page_first_direct_v1`; file uses
`native_full_file_v1`. Existing `paths` remain authoritative: PREFETCH supports
storage→host/gpu and host→gpu; DEMOTE reuses gpu→host/storage. There is no new
host reclamation, PIN, remote engine transfer or Router integration.

The backend namespace is `sglang.storage.v1:<sha256>`. It fingerprints actual
backend configuration and byte layout, independently of engine ID/epoch and pool
capacity. Mooncake includes master address, metadata address, tenant, writer key
prefix/suffixes and object sizes; file includes hostname, canonical directory and
writer suffixes. Both include pool class/layout/dtype/page size/layer interval,
bytes per token and head geometry. Identical compatible workers derive the same
value. Endpoint aliases are conservatively distinct; this is configuration
identity, not automatic cluster alias discovery. Send the source worker's namespace
when sharing content. Weight namespace and raw cache salt remain content identity.

The existing trusted Engine `POST /v1/kv/control` also accepts:

```json
{
  "kind": "storage_targets",
  "context": {
    "engine_id": "worker-b",
    "engine_epoch": "epoch-from-capabilities",
    "namespace": "weights-v1",
    "caller_id": "router-owner"
  },
  "target": {
    "kind": "tokens",
    "namespace": "weights-v1",
    "token_ids": [1, 2, 3, 4],
    "cache_salt": "original-tenant-salt"
  },
  "location": {
    "tier": "storage",
    "backend": "mooncake",
    "backend_namespace": "sglang.storage.v1:<value-from-capabilities>"
  }
}
```

PrefixTarget is also accepted with the P1 nested compatibility fields. The target
must resolve to one complete, aligned token prefix. `request_cached`, hash-only,
unsupported layouts, foreign epochs/namespaces and mismatched backends are rejected.
Existing control authentication/context validation applies; this is a trusted
control query, without a new public authorization mechanism.

The result has `description_version: "1.0"`, `namespace`, `compatibility` (raw
salt), `key_salt_encoding` (`sglang.kv.weight.v1` or `owner_assertion`), `page_size`,
`location` (storage backend and backend_namespace, no engine_id),
`existence: "unknown"`, and `pages`. Each page contains `start_token`, `end_token`,
`component: "full"`, full `storage_digest`, and `keys`; Mooncake also returns
`object_bytes` parallel to `keys`. These are physical object descriptions, with no
residency assertion, allocation, claim, network query or operation record.

Keys use the actual storage namespace hash chain and writer helpers:
Mooncake `_tag_keys`/`_batch_preprocess`, including rank and K/V pieces; file
`_get_component_keys`. Event i64 hashes are never expanded into storage digests.
The query applies the service's existing salt transform once, like ordinary
request insertion. Returned compatibility retains raw salt for safe reuse.

Live verification uses actual CUDA pools and Mooncake I/O, with no model download.
The reader starts without radix nodes or session metadata. Set
`KV_CONTROL_MOONCAKE_CONFIG` to backend JSON, prepend the desired SDK directory to
`PYTHONPATH`, and run the existing registered GPU test. For the supplied test master:

```sh
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/home/youwei/.cache/sglang-rfc-tools/mooncake:python
export KV_CONTROL_MOONCAKE_CONFIG='{"master_server_address":"127.0.0.1:52091","metadata_server":"http://127.0.0.1:52093/metadata","master_metrics_port":52092,"local_hostname":"127.0.0.1","protocol":"tcp","standalone_storage":false,"check_server":false,"tenant_id":"default","global_segment_size":33554432}'
.venv/bin/python -m pytest -q --tb=short test/registered/hicache/test_programmable_kv_cache.py
```

Without the opt-in variable, the live test skips and the existing GPU/host/file
roundtrip still runs. Each live run uses a unique weight namespace. The test checks
write→resolved-key existence, cross-worker key equality, host registration, shared
interest cancellation and GPU load-back bytes. Model generation, TP>1, MLA,
other Mooncake host layouts and Router-driven recovery remain separate validation.


## P3 DEMOTE observations

The action/envelope versions and payload fields are unchanged. Public DEMOTE
remains GPU→host/storage; host→storage DEMOTE and L1 PIN are unsupported.
Legacy copy/move behavior and soft priority retain their existing meaning.

A DEMOTE confirms the cold copy before reclaiming any source. Host completion
requires publication after the DMA fence; storage requires the native writer's
verified contiguous prefix receipt. Existing host backups and file/Mooncake
objects use native skip-write paths. An incomplete backup preserves all GPU source
ranges and reports confirmed cold ranges separately from missing ranges.

Reclamation runs through tree-core demote and allocator frees, deepest eligible
nodes first. Public target outcomes use these blocked reason codes:

- `active_reference`: the GPU component still has a correctness reference after
  this job releases its own temporary hold.
- `inflight_backup` / `inflight_load`: a matching pending transfer or storage job
  still holds the affected source.
- `structural_dependency`: a child still has GPU KV, preventing ancestor removal.
- `not_reclaimable`: conservative fallback for another non-evictable state.
- `cancelled` or the actual backup failure code when that operation deliberately
  retains the source after publication.

No code infers `pin_conflict` from ordinary references or soft priority. Request
references do not cause an indefinite retry loop: the result is partial with the
preserved/reclaimed/blocked scopes. A foreign source load returns partial when
cold data is already ready, otherwise deferred. Owned I/O still drains before
quiescence. A new bounded attempt may target the retained complete prefix; already
reclaimed suffixes do not become sources again.

Cancelled shared callers retain publication evidence already observed at their
cancellation boundary; the other caller keeps its own interest/holds. Last-caller
cancellation waits for its I/O drain and reports resulting cold data without
initiating source reclamation. Outcome records are historical evidence and do not
promise continuing storage residency. Per-page result capacity is approved before
public DEMOTE starts, allowing distinct blockers after concurrent radix splits.

Legacy primitive locations now reject a non-null `backend_namespace` on GPU/host
with `invalid_payload`; absent/null values and storage namespaces retain the P2
contract. No new public schema field or action version was introduced in P3.
