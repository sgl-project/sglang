# Programmable KV implementation and review log

## Scope and reference contract

Implement the accepted SGLang programmable KV RFC on branch `rfc-kvc`, starting
at `e4a171d4a0`. The user authorized local commits after each complete, verified
feature. GPU validation is explicitly deferred; this run performs CPU validation.
Do not modify the sibling reference checkout or terminate existing GPU processes.
No commit includes Codex as a co-author.

The implementation follows the SGLang RFC and its confirmed differences:
no public Pin, explicit cache priority above the session heuristic in the opt-in
profile, and copy-scoped claims with explicit renewal. Native and opaque actions
have separate interpretation owners. Policies propose; existing cache/pool owners
approve resources and execute.

References reviewed include the SGLang and vLLM RFCs, the current reference
`vllm/v1/kv_hints` implementation and README, the October 18/19 development reports,
the October 20 acceptance report, and
`../vllm-rfc/docs/programmable-kvc/experiment-reports/2026-10-20-000635-snapshot/technical-report.md`.
The accepted implementation lessons are whole-prefix preparation holds,
per-admission capacity/match revalidation, independent shared-operation lifetimes,
ACK/replay preserving failed outcomes, low-cost unconsumed signals, and movement
gated by pressure, pause/transfer cost and actual admission opportunities.

Implementation units cover control/logical lifecycle, serving/cache priority,
real movement adapters, composable policies and scheduling, then integration and
CPU acceptance. Unsupported component/topology combinations must be explicit.
This is an implementation sequence, not a claim that all units are already done.

## Act 3 — Build

### Unit 1, initial build

The delegated builder produced a native control package and a registered CPU
behavior suite. No serving path or real cache execution is claimed by this unit.
The parent independently read all new files and ran:

```text
PYTHONPATH=python CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest test/registered/unit/managers/test_kv_control.py -q
Running 69 items in this shard
.....................................................................    [100%]
69 passed in 8.51s
```

Review did not approve the initial build for commit. Two additional reproductions
identified lifecycle gaps:

- Native task finish left a session-addressed attached operation queued. Its
  result remained `QUEUED` and active count remained one, allowing later dispatch
  for a finished task. Cleanup must abandon only the removed task incarnations'
  interests and preserve independent users of shared content.
- An executor submission exception produced `CANCELLING / executor_error`, but
  the drain response changed it to `CANCELLED` with an empty reason. The eventual
  result and ACK must retain `FAILED` and the original cause, while still waiting
  for resource quiescence.

The builder was asked to add failing-before regressions and fix these two cases.
The active delegated session is `01a159fe-6599-7621-9b0c-c83bb11a2cce`; one of the
two delegated fix rounds for this unit has been requested. Parent review and
proof remain required before commit.

### Unit 1, review round 1 and parent verdict

The regressions reproduced before the fix (`11 failed, 67 deselected`) and passed
afterward (`11 passed, 67 deselected`). Native task finish now cancels interests
matched by frozen task incarnation, including same-envelope prepared actions;
external tasks and raw logical-index snapshots remain independent. Executor
exceptions retain their first cause and observed progress during draining, then
become `FAILED` after quiescence. ACK/replay preserve that failure.

The parent read the complete correction diff and independently reran the full
unit proof:

```text
Running 78 items in this shard
........................................................................ [ 92%]
......                                                                   [100%]
78 passed in 7.88s
```

Parent isort, Ruff syntax/name checks, formatting and diff-whitespace checks
passed. Imports are checked with the repository's separate isort configuration;
Ruff's additional unconfigured `I` rule classified the source differently and is
not substituted for that repository configuration. The builder also verified
CPU CI registration and the pytest entrypoint. One delegated fix round was used.
Verdict: approve this control/logical-lifecycle feature for its local commit.
Serving, real priority/eviction, movement and policies remain subsequent features.

### CPU environment

The repository has an isolated ignored `.venv`. It reads existing reference
site-packages through `reference-runtime.pth`; additional packages are installed
only into this repository's environment. The reference environment is unchanged.
Initial tools: Python 3.12.7, pytest 9.1.1, msgspec 0.22.0, ruff 0.17.1,
isort 8.0.1. The metadata-only unit needs no model/kernel imports or import stubs.
Full runtime CPU integration will have separate dependency and test records.

For upcoming real-cache tests, `gguf==0.19.0` was installed into the local
environment with `--no-deps`. The existing session UnifiedRadixCache suite passed
on actual CPU pools: `4 passed, 16 warnings in 9.51s`. Its test-utils port default
cannot parse an empty `CUDA_VISIBLE_DEVICES`; the CPU run used the nonexistent
device `999` and first asserted `torch.cuda.device_count() == 0`. Production and
test-utils source were not changed for this environment issue.

### Unit 2a — Real cache priorities, parent takeover

The builder added optional copy-scoped side metadata, real UnifiedRadixCache
execution/inspection, FULL/SWA/Mamba geometry and lifecycle hooks, and explicit
Rust-to-Python selection. The delegated run stopped on its usage limit before
producing a completion report. The parent took over without retrying delegation,
read all changes and independently ran the initial proof: 100 passed, 16 warnings,
9 subtests passed in 13.68s.

Parent review reproduced quadratic candidate selection: reclaiming 32 leaves
computed 528 ordering keys. The correction uses one candidate heap per walk,
updates exposed parents and rebuilds at claim-edit or TTL boundaries. Regressions
cover key-count growth on real FULL/SWA/Mamba pools, expiration and claim edits
during a walk, and the explicitly selected profile's session migration. The
profile uses LRU without claims, as specified; disabled caches retain legacy
session ordering. No claim/scoring metadata is added to ordinary node records.

Final parent proof ran each file in its own process with accelerators hidden and
the Python TreeCore selected: control 78 passed; real priorities 21 passed with
18 subtests; session 4 passed; joint allocation/eviction 15 passed with 45
subtests; lock refs 1 passed; Mamba path cap 5 passed, 5 GPU cases skipped, 2
subtests; SWA locked recovery 8 passed. Total: 132 passed, 5 skipped. The original
Rust-toolchain failure remains a limitation, not a successful Rust test.
Repository isort, Ruff formatting/syntax/name checks and git diff whitespace
checks pass. The key-count regression permits at most 64 evaluations for 32
victims in all three component paths, with and without a live claim.
Verdict: approve the real-cache priority feature for a local commit.
Serving transport, request callbacks, transfers and policies are still pending.

### Unit 2b — Serving service and request lifecycle

The parent implemented the serving unit directly after the builder's quota
failure, without further delegation. The service uses the real executor,
RuntimeContext's opt-in Memory configuration, typed BaseReq IPC, existing
tokenizer communicator and admin HTTP policy, plus sync/async Engine methods.
Only single-owner TP/PP/CP/DP=1 is enabled; unsupported topologies fail explicitly.
This avoids rank-local random clocks/IDs being misrepresented as TP support.

Attached hints survive session reconstruction and native/OpenAI adapters.
Consumed actions are removed from a fresh per-request envelope; opaque action
objects remain intact. Message lookup discovers independent fan-out bindings.
The completion hook runs after real checkpoint and before freeing/unlocking the
request. Queued and grammar aborts cancel work without requiring a KV row.
Intermediate checkpoints/preemption do not complete it. Active work blocks
destructive flush, and successful flush rotates/rebinds the cache epoch.
Correlated communicator replies reject late responses after RPC cancellation;
send failures clean up the waiter and inline replies are not lost.

Parent proof, each serving/core regression file run separately except the
existing config-bag group: control 78; service 12; communicator/config bags 26;
request construction 2; StreamingSession 11; OpenAI chat 159; completions 21.
All 309 passed. Service cases use real CPU pools/Req/checkpoint/release,
in-process HTTP and typed msgpack IPC; no model/cache import stubs.
An initial HTTP fixture used TestClient's other thread and correctly hit the
executor owner check; ASGI transport now runs on the actual owner. The release
test caught use of an obsolete Req field; it now uses authoritative
full_untruncated_fill_ids and verifies both committed suffix and cache salt.

Ruff's existing whole-file findings in large runtime modules include custom
Arg annotations and existing late imports/lambdas. The parent compared all
changed files to HEAD using identical E4/E7/E9/F/UP037 rules: no added findings.
New files pass those rules without exemptions; repository isort, Ruff format,
and git whitespace checks pass. User edits to the Chinese RFC/progress report
are retained and excluded from implementation commits.

### User-authorized small GPU validation

The user made a GPU available and authorized small single-card validation,
superseding the earlier CPU-only limit. GPU 1 was idle; all other processes were
preserved. Real CUDA tensor arithmetic passed. The initial kernel import failed
because the borrowed PyTorch 2.13 ABI lacked CUDAErrorLogCapture. The target
virtualenv alone was updated to torch 2.14.1 and sglang-kernel 0.4.10, then
torchvision 0.29.1 to correct its nms ABI; reference packages are untouched.
After the environment correction, the parent reran control 78, real-cache 21,
service 12 and communicator 3: all 114 passed on 2.14.1+cu130.

The new manual GPU smoke script passed on NVIDIA A100-SXM4-80GB: real CUDA
allocator/copies, priority victim choice, retained KV bytes, request-lock
protection, reclaim under pressure, copy-lifetime/replay and full pool balance.
It allocated 8,704 peak tensor bytes and ran its cache body in 0.2165 seconds;
these are fixture measurements, not serving performance results.

A real tiny-Llama server attempt is still unsuccessful before cache init at
DualGemm's cutlass.cute import. Local nvidia-cutlass-dsl 4.8.0 plus its pinned
base/cu12/cu13 libraries do not expose that module. Earlier ABI/torchvision
failures and the final startup failure are preserved in /tmp/sglang-kv-*-failed.log;
the successful cache output is /tmp/sglang-kv-gpu-cache-smoke-passed.log.
No version checks or model kernels were bypassed to manufacture a successful
model result. Further model validation remains necessary.

Verdict: approve the serving/priority entry-point feature with this explicit
model-validation limitation. Real transfers and composable policies remain work.

### Unit 3a — Persistent-host HiCache execution

Implemented actual FULL-prefix GPU→host COPY/MOVE and host→GPU COPY using the
existing cache/controller allocation, queue, fence and publication paths. Jobs
coalesce compatible logical scopes and retain independent cancellation/results.
Whole-scope holds protect both pre-existing and arriving prefix members. Unique
physical fragments are charged once across branches; per-branch allocation
rechecks actual remaining capacity. A scheduler callback uses its existing
prefill budget and decode-growth estimate, including fixed-length requests and
chunked-request identity deduplication, without carrying a second reservation into
normal request admission.

Native loads install indices before DMA completion in the existing HiCache path.
The supported non-overlap scheduler therefore waits for native H2D completion
before fresh prefill matching. Idle progress drains the real completion queues.
Normal request load thresholds remain unchanged; explicitly approved loads use a
separate minimum threshold. COPY publishes after fences; MOVE reclaims only
eligible source leaves and reports retained ranges as partial. Last-waiter cancel
retains buffers until quiescence. Partial-submit failure causes and actual effects
survive cancellation, ACK/replay and independent retry. Completed outcomes merge
adjacent logical ranges and hash only the final runs.

Review corrections included: distinguish terminal DEFERRED from resource QUEUED;
reject scopes larger than the entire pool without leaving an active idle record;
reserve traversal budget for ordinary page splits during DMA; validate every
joiner's result budget; normalize equivalent local tier addresses for coalescing;
record reclamation only after allocator release; preserve failure over cancel;
and retain observed destination effects if cancellation races with completion.
The real request/host locks and tree split fixups were exercised, not simulated
as an independent residency database.

Current capabilities are explicit: one owner, built-in FULL, persistent local
host, token-backed complete prefixes at page/radix boundaries, and no overlap.
Hybrid transfers, descriptor-only/range transfers, L3/linker paths and multi-rank
publication remain pending. Existing weight update entry points can mutate before
checking idleness, so this profile now returns a typed refusal before accessing
runners/collectives. Proper weight-namespace isolation is a required later unit;
weight updates are not advertised as supported.

Parent CPU proof ran each file separately with accelerators hidden and Python
TreeCore, preserving full output in /tmp/sglang-kv-host-cpu-proof.txt:

- control: 78 passed;
- real-cache priority: 21 passed, 18 subtests;
- serving/request lifecycle: 12 passed;
- native host transfer: 13 passed;
- existing HiCache transfer lifecycle: 8 passed;
- existing prefill adder: 53 passed, 54 subtests;
- existing schedule policy: 11 passed, 4 subtests;
- weight-update runner selection: 1 passed, 9 subtests;
- scheduler weight-version tracking: 20 passed.

Total: 217 passed. After strengthening the byte fixtures, the 13 host cases
passed again. The host suite uses actual CPU pools and HybridCacheController;
only the L2 DMA/fence boundary is controlled. It verifies ordinary splits during
DMA, independent shared cancellation, complete-scope capacity, partial MOVE,
partial submission failure and result/replay preservation, no-allocation rejection,
terminal impossible-scope deferral, and running-growth accounting. Recovery tests
poison freed device buffers before restoring, so stale contents cannot satisfy
the byte checks. Default prefill behavior keeps the same growth formula.

Real GPU 1 HiCache direct I/O passed on A100-SXM4-80GB with PyTorch 2.14.1+cu130:
GPU→host MOVE stored nonzero K/V before reclaim; host→GPU COPY restored them
after all free GPU rows were poisoned. Pool availability was 64 after MOVE and
60 after COPY (four valid cached tokens); references, pending queues and native
jobs/interests retired. Peak fixture tensor allocation was 9,728 bytes. The tiny
head emits an optional JIT-row-size diagnostic; this proof explicitly uses the
direct backend and makes no claim about the kernel backend.

A model execution path now passes as well. A local OPT-125m configuration with
--load-format dummy avoids the separate tiny-Llama CUTLASS import failure and
executes real GPU forward/cache/control code. The reusable server smoke covers
attached priority, bounded result discovery, independent full-prefix MOVE,
GPU-absence/host-presence inspection, independent COPY, fresh priority zero,
loaded-prefix reuse on repeated generation (3 cached tokens, equal outputs),
flush epoch rotation and HTTP 409 for stale references. Dummy outputs are not a
model-quality or throughput result. The failed Llama attempt remains recorded.

GPU proof artifacts include /tmp/sglang-kv-gpu-hicache-poison-smoke.log and
/tmp/sglang-kv-opt-hicache-final-smoke/{server.log,result.json}. The manual server
harness uses an owned process group, checks its generated owner identity and
retains separate run artifacts. Only its own processes are terminated. A final
identity-isolated harness run is recorded in the validation report.

### Unit 3a — Final review additions

The byte fixtures now seed canonical overlapping prefixes before insertion and
poison freed device buffers before H2D. CPU host cases passed again. Review also
found that rebuilding an executor after cache flush could drop its scheduler
capacity callback. The service preserves that callback across epoch rebind;
a regression calls it after reset and the server harness checks the approval
capability after flush. The host suite is now 14 tests; host plus service passed
26 tests. Unique test count across this unit's nine relevant suites is 218.
The validation record is docs/programmable-kvc/validation-host-control.md.

The default prefill growth formula is unchanged. New files pass strict Ruff
E4/E7/E9/F/UP037; changed existing files have no added findings against HEAD under
identical rules. Repository isort/format, AST syntax and diff-whitespace checks
are required before the local feature commit. Storage and policies remain the
next implementation units; this entry does not declare the full RFC complete.

Final identity-isolated GPU server run passed in
/tmp/sglang-kv-opt-hicache-rebind-smoke/. It additionally asserts scheduler-budget
approval before and after flush. The poisoned-byte CUDA test passed in
/tmp/sglang-kv-gpu-hicache-poison-smoke.log. Final isort, formatting, whitespace,
and no-added-Ruff-finding checks passed. Verdict: approve this bounded
persistent-host transfer feature for its local commit.

### Unit 5a — Namespace encoding for actual request/cache/store keys

Implemented bounded, version-tagged namespace/salt encoding shared by ordinary
requests and independent control. Actual serving enables it even without hints;
the low-level callable cache mode retains its explicit owner-assertion contract.
Request completion binds logical salts, so attached selectors do not encode an
already encoded salt. Discovery reports the scheme. Namespace rebind is an owner
boundary guarded by idleness, pending transfers, StreamingSession private KV and
correctness references; it rotates epoch/logical metadata while old physical
entries can remain inaccessible under old keys. This does not yet enable the
parameter mutation APIs.

Review found that preparation inside an overrideable request constructor could
be bypassed by the documented subclass factory. A caller wrapper now owns the
namespace/hint preparation while the original constructor remains an extension
point. Tests use an actual subclass overriding only that constructor. Ordinary,
session and unknown-session construction all use the wrapper.

The eleven new real-CPU cases pass, including unhinted checkpoint/release versus
independent lookup, Unicode and hash-looking logical salts, no double encoding,
generated suffix binding, old live physical entries after namespace rebind,
distinct actual storage hash chains, stale contexts, active rows/locks,
StreamingSession-owned KV and callback preservation after flush. Existing core,
priority, service, host transfer and constructor regressions passed (127 tests)
before the constructor-wrapper correction; focused final proof is recorded below.
The namespaced OPT dummy-weight HiCache server proof also passed with three
cached tokens and equal output, plus independent MOVE/COPY and epoch checks.

Final wrapper proof passed 24 tests (namespace 10, service 12, constructor 2);
the additional wrong-thread regression then passed with all 11 namespace cases.
Wrong-owner namespace change is refused before changing any records. Total
unique relevant cases are 138 including the existing 127. GPU server proof after
the wrapper correction passed at
/tmp/sglang-kv-opt-namespaced-wrapper-smoke/{server.log,result.json}, including
independent lookup against actual namespaced ordinary-generation keys and
loaded-prefix reuse. No added Ruff findings, isort/format and diff whitespace
checks pass. Verdict: approve the namespace-key foundation for a local commit;
parameter mutation transactions remain the next required work.

### Unit 5b — Guarded parameter mutation and version publication

Replaced the temporary serving refusal with owner-fenced disk/IPC updates and
begin/tensor/distributed/end sessions. Namespace/key/epoch isolation precedes any
runner packing or load. New inference is rejected directly before request/session
construction or forwarding while the fence is open; native submissions return
busy. Parameter work synchronizes before requested cache flush and publication.
No-flush updates retain old physical entries only under incompatible old keys.
Provided new version labels derive compatible namespaces; unknown generations
are fresh. Failed results/rollback keep an isolated namespace; failed parts are
not converted into successful transaction publication by end. Unknown exceptions
or synchronization failures retain the fence; failed final flush can retry end
without repeating parameter mutation.

Audit also covered debug reset_tensors and explicit weight-version assertion,
which now use the same guarded boundary. Memory ownership changes refuse open
weight fences rather than tripping the scheduler's legacy idle assertion. Their
typed refusals propagate through existing IPC. Offloaded resources refuse weight
mutation. Python Engine now forwards optional version labels/selectors through
real tensor serialization and typed requests.

New CPU tests mutate actual tensors behind the existing manager methods and use
real cache keys/metadata. They cover standalone and multipart updates, early
refusal, no-flush isolation, rollback/partial failure, exception fencing, retries,
debug reset, version assertion/P2P end, selectors, wrong thread, memory ownership
and Python Engine serialization. Twenty-three cases pass; relevant existing CPU
suite results are retained in /tmp/sglang-kv-weight-cpu-proof.log.

A real GPU OPT dummy-model reset changed attention LayerNorm bias tensors. Engine
checksums changed, the model namespace changed, first new-version generation
reported 0 cached tokens, and repeated generation reported 3 cached tokens with
equal output IDs. This is a parameter/cache-isolation smoke test, not a model
quality result. Artifacts: /tmp/sglang-kv-opt-weights-smoke/{server.log,result.json}.

An existing GPU weight-checker suite was initially invoked with GPUs hidden:
42 cases failed on No CUDA GPUs, and a minimal failed reproduction is preserved
at /tmp/sglang-kv-weight-checker-cpu-env-failed.log. On GPU 1 it reports 63 passed,
1 failed; the remaining FP8 lazy-pair case cannot import deep_gemm in this
host environment. Full output is /tmp/sglang-kv-weight-checker-gpu-proof.log.
That dependency failure is not counted as passing or hidden by modifying the
existing checker suite. FP8/model-parallel validation remains outside this
single-owner FP16 fixture.

Final review added transient memory availability checks and typed memory/version
refusals, so offloaded KV cannot execute native operations and offloaded model
resources cannot receive parameter writes. Inference also refuses unavailable
weights/cache. Namespace/budget/availability callbacks survive rebind. Public
metadata version assertions participate in the same fence; P2P/RDMA labels remain
unpublished until end, and unchanged committed labels are no-ops outside an
update. Wrong-thread end refuses before touching a runner.

The final new weight suite is 24 tests, all passing; weight/service/host suites
passed together (50 cases). Relevant per-file CPU proof passed 143 cases with
2 existing skips before the final availability case; no added Ruff findings,
isort/format and diff-whitespace checks pass. The real GPU parameter-isolation
proof passed; the separate existing FP8 dependency failure remains recorded.
Verdict: approve guarded parameter lifecycle for its local feature commit.
Real storage execution and composable policy/admission work remain outstanding.

## Unit 6a — Native persistent-host file storage writes

Implemented real single-prefix COPY GPU/host→file storage and MOVE GPU→file
storage. GPU staging uses existing host transfers and whole-scope approval;
compatible storage jobs share one worker operation with independent interests.
The worker receives an immutable complete hash chain and pinned real host
indices. Owner ACK consumption retains facts only for bounded active native
writes. Ordinary backup behavior remains enabled by default; explicit native
host-only copies suppress automatic cold writes so the requested destination
owns its side effect.

Review identified that successful batch receipts do not certify joint prefix
availability in an evicting backend. Native writes now request a worker-side
contiguous-prefix verification before ACK. Short writes preserve the GPU source
and report verified coverage with failure cause; full writes can reclaim GPU
copies only through existing eligibility after releasing the native GPU hold.
Another request's lock gives PARTIAL/source_busy. Cancellation drains worker
access, never deletes shared cold objects, and never turns a failed result into
successful cancellation. Existing radix splits preserve captured worker buffers
and logical outcomes. Attach/detach refuses pending native work; idle changes
refresh adapter capabilities.

New CPU file-storage suite: 10 passed. Includes actual disk bytes, pending ACKs,
one/all shared-waiter cancellation, cancel during DMA, short/evicted-prefix
writes, failure ACK summaries/retry, external request refs and radix splitting.
With existing priority/cache/HiCache lifecycle cases: 39 passed plus 18 subtests
(/tmp/sglang-kv-storage-cpu-proof.log). Serving/namespace/weight/host regression
suites: 61 passed (/tmp/sglang-kv-storage-regressions.log).

Existing file path/LRU tests passed 37 cases. Four existing backfill tests were
also attempted with GPUs hidden and failed at fixture get_device() before the
feature path; these fixtures select an accelerator. Their failure is retained
at /tmp/sglang-kv-storage-backend-proof.log and requires the GPU-specific rerun.

The same four backfill tests subsequently passed on GPU 1 with their existing
64-token fixture (/tmp/sglang-kv-storage-backfill-gpu-proof.log). The hidden-GPU
attempt remains recorded rather than being counted as passing CPU coverage.

Real GPU 1 A100 FP16 proof passed after the final prefix-verification change:
/tmp/sglang-kv-gpu-file-storage-final-smoke.log. The --storage manual fixture
writes nonzero 17/18 K/V through real direct DMA and file workers, reads exact
file bytes, waits before GPU reclamation, poisons free GPU rows then restores
from host, and checks allocator/lock/job retirement. Peak device allocation is
9,728 bytes for the tiny fixture. The optional JIT reports unsupported 16-byte
rows; the explicitly selected direct-I/O backend passes. No JIT or model-quality
claim follows. Other GPU processes were untouched.

Parent read the complete implementation and diff. New strict Ruff checks and
existing-file finding deltas, isort, formatting and whitespace checks pass.
Storage reads, multi-branch cold jobs, other backends/linkers and policy/admission
work remain separate implementation units; discovery does not claim them.

## Unit 6b — Exact native restoration from file storage

Implemented COPY file-storage→host/GPU for one complete token-backed prefix,
including absent physical radix nodes. Native operations reuse real async hit
queries, hit-sized host allocations, file reads and host insertion, with a
per-operation minimum size; normal prefetch thresholds and stop policies remain
unchanged. Complete missing host scope and scheduler GPU copy headroom pass
before lookup; queued capacity waits retain no holds. Distinct cold scopes
serialize and re-evaluate shared host prefixes, while identical destinations
coalesce with independent interests. Actual hit allocation requires a full fit.

Owner cache publication acquires native prefix holds before ordinary admission
can evict the inserted rows. Host publication waits for completed_req ACK after
all worker access; H2D then uses actual HiCache fences and the existing native
prefill barrier. Query-only misses/revocations have a separate query-consumed
quiescence boundary. Last cancellation drains workers/DMA and scratch releases;
partial reads never start an incomplete GPU restore. Native per-request metrics,
loaded-token and retry state retire with the job. COPY holds end on completion,
so future admission preparation remains a policy responsibility.

Review found two integration mistakes during the first probes: CPU scratch
allocation inherited pin_memory=True without a CUDA allocator; the controlled
CPU fixture now explicitly uses unpinned scratch pages. Native prefetch also
passed list keys into a radix tree using array('q'); it now uses that canonical
representation. Initial GPU timeout is retained at
/tmp/sglang-kv-gpu-file-roundtrip-key-type-failed.log, not counted as passing.
Real low-level GPU 1 roundtrip then passed at
/tmp/sglang-kv-gpu-file-roundtrip-smoke.log: both GPU and host copies evicted,
nonzero files restored into poisoned GPU rows, all actual bytes/refs verified.

Audit also found that cancellation could hide a composite child staging failure.
Write/read adapters now propagate failed child results and already-observed
short-read ACK causes through late cancellation. Dedicated tests exercise both.

New read CPU suite: 11 passed, all with actual namespace-encoded file keys.
Existing-host suffix read has only two free host rows for two missing rows,
which proves the existing two-row prefix isn't charged twice. Namespace change
cannot read old-generation files. Disk and GPU cancellation, capacity no-I/O,
miss/short reads and failed-ACK cancellation are covered. Write suite now has
11 passing cases including the composite-failure regression. Final 22-case
proof: /tmp/sglang-kv-storage-read-final-cpu-proof.log. Earlier combined read/
write/host/service proof: 46 passed; HiCache lifecycle/namespace/weight regressions:
43 passed (/tmp/sglang-kv-storage-read-regressions.log). Two existing real GPU
FULL_ps4 L3 write/prefetch tests passed unchanged in
/tmp/sglang-kv-storage-read-existing-gpu-proof.log.

Actual model-serving proof passed on GPU 1 with OPT-125m configuration and dummy
weights: /tmp/sglang-kv-opt-file-storage-roundtrip-smoke/{server.log,result.json}.
It generates/sets priority, independently MOVEs to file storage, flushes physical
cache (new control epoch), proves both GPU/host absent, independently COPYs
storage→GPU, checks fresh priority, and repeats generation with 3 cached input
tokens and equal output IDs. Final flush rejects stale refs. Low-level nonzero
byte evidence complements the dummy-weight interface check; no quality claim.
Only the owned server process group was terminated. Parent reviewed all changes;
strict new-file Ruff, no added old-file findings, isort/format/whitespace pass.
Verdict: approve local exact file-restoration feature. Linker/multi-branch cold
work and composable policies with actual admission cooperation remain pending.

## Unit 7a — Explicit authority over internal policy claims

Added internal ExecutionRequest origin (public ingress always explicit), origin-
scoped revision identities, and live explicit-before-policy effective priorities.
Independent explicit claims still merge by max; clearing/expiring all explicit
claims reveals still-live policy claims. The actual radix copy metadata, splits,
expiry and eviction ordering use that rule. Policy revisions have a separate
bounded per-copy budget (default four) and cannot consume explicit revision
slots; total copy-unit capacity remains shared. Configuration and payload
validation retain their boundaries; caller-name conventions do not grant policy
origin. Policies are still disabled by default; this is their authority seam.

Three new actual CPU-cache tests cover negative explicit intent vs larger policy
priority, independent revisions and metadata slots, expiry/clears/stale replays,
split TTLs, caller-name spoofing and rejection of a public origin payload.
Existing and new control/cache/service tests pass together (114 cases plus 18
subtests), /tmp/sglang-kv-policy-priority-cpu-proof.log. Initial test assertions
used a tuple-containing Python object instead of JSON and expected a different
existing capacity status; corrected to exercise the real JSON boundary and
existing REJECTED metadata-capacity semantics. No production behavior was
changed to satisfy those incorrect expectations. Ruff/isort/format/whitespace
checks pass. Parent reviewed the complete diff; approve this local feature unit.
Composable runtime, observations and scheduler admission holds remain ongoing.
