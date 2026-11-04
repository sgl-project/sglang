# KV public protocol, session state, and cache protection

This crate contains pure schemas, validated action parsing, salt encoding and
logical models, plus the P5 bounded association and evidence foundation. It imports
no Router/indexer runtime. Its backend protection module performs bounded asynchronous Mooncake I/O; Engine L2 PIN execution remains in Python.
`protocol::parse_action`
validates the three new public actions. Other consumers' actions are returned
unchanged; unsupported `sglang.*` names/versions return `Unsupported`. Existing
Engine action parsers and semantics remain in Python.

The optional Router `preparation_policy` config enables bounded preparation cost
samples grouped by worker URL, owner epoch, compatibility and prefix size bucket.
For example, `{"min_samples":4,"quantile":0.9,"max_lead_ms":10000}` uses a recent
tail estimate with a bounded lead; missing samples use the configured initial
cost. `max_groups` defaults to 128 and `sample_window` to 32. Samples describe
complete observed preparation including control polling and admission, not pure
I/O. Confirmed storage-to-host completion supplies measured samples. An incomplete
terminal preparation contributes its observed elapsed time as a lower bound;
`incomplete_samples` distinguishes those observations from complete samples.
The reported quantile can therefore be a lower bound on full preparation time.
A positive remaining wait can overlap part of a late preparation. Missing or
zero remaining wait retains on-demand recovery. Omitting this config preserves
the existing global cost/lead policy. No new Engine action fields are required.

`preparation_policy.ready_pin_ttl_ms` optionally submits a separate finite host
PIN after confirmed lifecycle storage-to-host preparation. It defaults to zero
(disabled), and is limited to 60,000ms. `auto_pin_max_tokens` defaults to 65,536
and conservatively reserves token ranges across this Router's child claims;
overlapping jobs can be overcharged, while the Engine enforces physical dedup
and its own protection budget. Each preparation episode makes one lease attempt,
without automatic reacquisition after expiry. Parent cancellation/cleanup releases
only its own claim when no shared interests remain; unknown child outcomes block
acknowledgement. The query exposes `auto_pin` independently from `ready_scope`.
With this policy configured, `tool_wait` can explicitly select a host source for
L2-to-L3 DEMOTE; the default source remains GPU.

P5 is a pure API; Router runtime integration remains P6. `session_index` provides
`SessionIndex`, explicit `SessionLimits`, canonical `ContentKey`, parent-linked
`ContentRegistration`, immutable `BranchRevision`, request incarnation bindings
and scoped cleanup receipts. Caller identity comes from trusted admission context.
Only `start_session` creates a task; finish/cancel use `finish_task` (non-cascading)
or explicit `finish_task_with_cascade`. The receipt lists exact task-owned claim
and intent IDs to revoke. It does not call Engine's legacy recursive cleanup or
remove physical KV. A retired task cannot be recreated, including by late events.

Adapters register page-aligned content nodes with suffix-only token material;
shared ancestors store tokens once. Digest/key construction, namespace/salt
translation and physical layout validation stay in adapters. A token checksum
survives material eviction to reject conflicting refills; it is not a storage key.
`resolve(Selector)` binds session, branch/latest, request or registered explicit
prefix to a `FrozenTarget`, with known/missing scopes, materialization gaps and
the current session revision. Later branches cannot expand that target.
`materialize` checks current material and constructs the P1 token prefix only at
submission time. Intermediate ranges remain unsupported. Missing Stored events
do not prevent rendered requests or confirmed cache hits from creating associations.

All persistent collections and identifiers are bounded. `collect_garbage` drops
unreferenced history/content; request bindings, current branches and externally
held snapshots/content handles retain their dependencies. Hold `content()` handles
when assembling several unassociated nodes. Reserve room for the incoming node
before replacing an active branch. `evict_material` preserves associations.
Capacity errors leave active state intact. Tombstones are never automatically
evicted; exhausting their budget refuses further cleanup until the fenced owner
domain is rotated. P6 must separately bound external operation/view handles.

`observations` provides injected `LocationProvider` with separate `confirmed`,
`expected` and `unknown` scopes. `ObservationFence` stores bounded owner/tier
version fences only; existing indexer/local confirmed trees remain the residency
source. Adapters bind epochs from trusted discovery, mark stream gaps, and obtain
query tickets before actual scope queries. `confirm_query` checks the response
epoch and unchanged ticket. Only a complete owner/tier snapshot repairs stream
completeness; partial scope queries can confirm their actual ranges independently.
Retained views must call `revalidate` before consumption. Every newer event
invalidates retained observations conservatively. Owner epochs, instance domains
and cache-event watermarks fence restarts and delayed responses. P1
`PublicResult.state_version` is an **operation evidence sequence**, represented by
`OperationEvidenceVersion`; it cannot be used as `CacheEventVersion` or establish
current residency. P6 must query the authoritative provider after operation results.

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
(eviction_exclusion by default, or backend_soft_pin), and optional `owner_task`.
The native FULL-only persistent host cache executes local host
`eviction_exclusion`. GPU PIN, L3 PIN and `backend_soft_pin` are unsupported.
Discovery advertises only live executable protection under `pin_protections`.
PIN supports independent submissions and attached hints. Attached explicit
prefixes validate and activate existing host content at dispatch, using the same
executor as independent PIN. An accepted explicit claim survives the inference
request's later completion/abort until release, TTL or invalidation.

Attached `request_cached` acquire is queued without claiming readiness. It binds
only to the owner's committed token prefix after a successful request checkpoint,
before request correctness references are released. Abort, failed checkpoint or
missing committed material cannot activate it. The host replica must already be
complete and published at activation; PIN does not silently copy GPU data or wait
for a pending host backup. The finite TTL starts at successful activation. A
nonterminal checkpoint does not execute the queued action. Independent
`request_cached` acquire is invalid because it has no request binding.

Each attached claim uses a deterministic effective ID derived from the caller,
unique request binding and original payload claim_id. This isolates sampling
branches and reused request IDs. ClaimState.caller_id retains the original caller
(`request` for hints) and ClaimState.claim_id returns the effective ID. Attached
renew/release with the same bound context and original claim_id address that
same claim. They do not resolve target bytes/session history. Independent
follow-ups/query use the returned effective claim_id with the same caller's
unbound context; a new sampling binding cannot release another branch by reusing
the original payload. No additional protocol field is required. Acquire and
renew message replay preserve their historical receipt and never extend TTL.
Operation `lookup` with the unbound `request` caller and message_id discovers all
sampling bindings and their effective claim IDs. Queued/failed operations have
no active claim receipt. Bound contexts remain forbidden for independent submit;
use an unbound context and the effective ID for external follow-ups.

Discovery sets `pin_submission=["independent","attached"]`,
`pin_request_cached="successful_checkpoint_existing_host_prefix"` and
`pin_attached_claim_id="returned_effective_claim_id_per_binding"`.

`owner_task` is an explicit local logical-index identity
`{"session_id":"task","incarnation":7}`. Acquire/renew validate it against
this Engine's live index; it is not a global Router incarnation. Selecting a
session/content prefix never establishes demand ownership. Omitting owner_task
creates a caller-owned claim keyed by caller_id + claim_id, so an explicit
prefix works on a worker with no local session/index association. The Router
must release these caller-owned claims when its demand ends. Legacy finish_task
retains its recursive index behavior, and releases only the finishing caller's
claims explicitly owned by the removed local task identities. Other callers'
claims and unowned claims remain protected.

Acquire validates the entire existing page-aligned FULL host prefix at existing
radix boundaries, including ancestor dependencies and the D2H publication fence.
It creates no missing cache content and performs no split. Admission checks
claim/record/token/scope/result budgets before installing native host locks.
Overlapping physical host slots are charged once. Split-aware receipts preserve
protection across later radix splits. TTL starts after installing those locks.
Ordinary hits, completed operations and repeated messages do not renew claims.
Same revision + same request is idempotent; conflicting or older revisions fail.
Release creates a tombstone even when acquire was not observed. Renew/release
address the claim key without resolving target bytes or a removed session.
Released/expired keys require a new claim ID. Terminal records are never silently
evicted: `protection_limits.max_records` exhaustion rejects new keys until an
explicit owner epoch rotation. Other protection_limits bound active claims,
physical tokens, scope tokens/prefixes, TTL and retained feedback events.

Independent query through POST `/v1/kv/control`:

```json
{"kind":"claim","context":{"engine_id":"worker-b","engine_epoch":"epoch-from-capabilities","namespace":"weights-v1","caller_id":"router-owner"},"claim_id":"keep-prefix","owner_task":null}
```

It returns ClaimState with caller_id, task_incarnation, claim_id, revision, state,
owner_epoch, protection_version=1.0, tombstone, effective_protection, remaining_ttl,
owner_task, actual_scope and location. The compatibility field task_incarnation
is the decimal local incarnation when owner_task is present, otherwise `"caller"`.
Only active claims report positive remaining_ttl and eviction_exclusion.
actual_scope records the accepted coverage; terminal scope is historical and
does not assert current residency. Operation public_result.claim is the receipt
at operation completion. Use the independent query for current state.

`{"kind":"claim_events","context":...,"after_sequence":0,"limit":256}`
returns owner_epoch, gap, items (`sequence`, `claim`), and next_sequence. The
bounded per-epoch journal contains only this caller's changes, including expiry
with no inference traffic. A gap requires querying the caller's known claim keys.
This is polling feedback, without a durable journal or push subscription. Cache
reset/flush, host-resource teardown and weight namespace changes invalidate
claims and release references before destroying their cache ownership. Epoch
rotation rejects old contexts; the journal is not retained across that rotation.
Storage-only detach preserves independent L2 claims. Accepted claims are never
revoked due to ordinary admission pressure. Valid TTLs schedule owner ticks,
remain separate from in-flight operations, and bound idle sleeping; normal
decode steps avoid scanning protection records before the nearest deadline.

New action results retain `ref`, `sequence`, `update.status`, `update.quiescent`,
`completed_steps` and target outcomes, and add `public_result`:

- result_version=1.0, requested_scope, accepted_scope, ready_scope,
  ready_location, preserved_scope, reclaimed_scope, missing_scope, blocked_scope;
- owner_epoch, state_version and outcome_sequence (owner registry sequence),
  cache_generation when known, source_observed_at (null until an observation
  source supplies it), and the historical claim receipt for PIN.

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

Regenerate the shared schemas and default-expanded claim vectors from the root:

```sh
PYTHONPATH=python .venv/bin/python - <<'PY'
import json
from pathlib import Path
import msgspec
from sglang.srt.managers.kv_control.contracts import Prefetch, Demote, Pin, ClaimState, PublicResult, validated
root = Path("sgl-router/sgl-kv-orchestrator/fixtures")
for name, schema in {"prefetch": Prefetch, "demote": Demote, "pin": Pin, "claim": ClaimState, "result": PublicResult}.items():
    (root / f"{name}.schema.json").write_text(json.dumps(msgspec.json.schema(schema), ensure_ascii=False, indent=2) + "\n")
values = json.loads((root / "results.json").read_text())
values["claims"] = [msgspec.to_builtins(validated(v, ClaimState)) for v in values["claims"]]
(root / "results.json").write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n")
PY
```


## Native PREFETCH and arriving demand

With KV control enabled, an ordinary `wait_complete` request can join an active
native storage→host restore when the complete page-aligned token scope and
already-encoded salt/extra-key match exactly. Matching is indexed by immutable
prefix identity; each readiness poll is a handle lookup. Unequal scopes,
best_effort/timeout, buffer-only/bigram and disabled-control paths retain their
existing behavior. No global prefill barrier is added for these host restores.

Demand interests are bounded by `max_prepared_prefixes`, with scope material
bounded by `max_scope_tokens`. A join allocates no additional storage bounce or
GPU memory. Existing native approvals, host/GPU correctness references, backend
workers and owner-thread publication remain authoritative. Router cancellation
cannot terminate a read still needed by ordinary demand; its receipt drains while
the request waits. Request abort drops only its demand. Once all interests are
cancelled, normal terminate/drain applies.

After publication/drain, the request resumes through the existing
`check_prefetch_progress` / `pop_prefetch_loaded_span` interface. The pop releases
the temporary shared holds into the normal synchronous match/load-back admission
path; ordinary eviction semantics then apply. Reset/detach cannot discard live
demand holds. This adds no PIN or retention claim. Partial/missing reads unblock
admission to use the real published prefix and compute the remainder.

The loaded span is clipped against data already resident when the request joined.
Joined reads attribute newly published data to storage; requests arriving after
host publication keep normal host attribution. Existing cached-token tier splitting
and clipping are reused. Shared reads do not fabricate an extra physical lookup.

`prefetch_outcome_stats_snapshot()` adds fixed counters `native_demand_coalesced`,
`native_demand_consumed`, `native_demand_aborted`, `native_demand_limit_declined`,
and gauge `native_demand_waiters`. Consumed means the admission span was taken,
not proof of eventual inference use. Existing `issued` counts actual controller
submissions. No latency/byte EMA or unused-work estimate is invented. Model-level
benefit must be measured after restarting the benchmark snapshot; correctness
proof and fewer backend calls alone do not establish a TTFT improvement.
