# CI / Test Action Items

Patterns that have been applied to this repo's tests and CI. Each entry is a general
rule plus a way to spot it in this repo; the PR numbers are evidence, not definitions,
and are expected to be replaced as newer examples appear.

Sections follow the order of questions an audit asks about a test: should it exist, is
it in the right place, does it test the right thing, is it stable. The remaining
sections are about the pipeline around the tests.

---

## A. Should this test exist in this form?

**A1. Every case answers "what future diff would turn this red?"; if the only answer is editing the test, delete it.**
The admission criteria in `.claude/rules/unit-test-admission.md` apply to existing cases as
much as new ones. Spot: a test for a code path or kernel that no longer exists;
assertions that only check a mock was called; a stress loop that cannot reproduce the
failure it claims to guard; a case whose every assertion is also made by another case in
the group. Not this pattern: the rule's own carve-outs -- an external-source literal, or
a completeness / negative-branch contract -- are bookkeeping, even when the code is a
one-liner; and a path that was renamed is not a path that no longer exists, so check
before declaring a test orphaned. Examples: #34464, #34667, #39013, #38881, #38259.

**A2. A test earns its cost by what it protects; a guard on a standalone, rarely touched, rarely used surface can be deleted outright.**
Tests exist to stop other people's changes from breaking code. That value scales with
how often the code changes, how many users hit it, and how entangled it is with other
paths. Spot: the module under test has had no non-test commits in months; it has few
importers; the feature sits behind a flag almost nobody enables or serves one model few
people run. The heavier the test (server launch, multi-GPU), the higher this bar.
Not this pattern: low churn alone. A stable, load-bearing path (cache eviction, PD
transfer) is where a regression hurts most; all three signals -- little churn, few
users, little entanglement -- must hold before deleting on this ground.
Examples: none applied on this ground yet; #37990 (cases for models nobody runs) and
#38011 (low-signal model tests demoted from the PR gate) are partial.

**A3. A variant whose configuration is a strict subset of another variant's is redundant.**
Spot: several classes in one file whose launch configs differ by a single flag or env
var. Holds when three things are true: the configs differ by one dimension; that
dimension is additive (it adds load or a knob position rather than gating a path or skipping
assertions); and the heavier variant's assertions cover the lighter one's. Watch for
variants that override a test method, and for looser expectations in the heavier variant
(a longer timeout or relaxed bound), which make the lighter one still worth keeping.
Not this pattern (and this one has the highest false-positive rate of the catalog):
sharing a helper or fixture, which does not make one variant a subset of another; a
variant whose flipped flag is itself the feature under test (e.g. an overlap-loading
toggle), which fails the additive condition; variants that drive different integration
layers of the same model (engine input format vs. server endpoint), so neither's
assertions cover the other's; or a lighter class that is skip-decorated and launches
nothing, so it costs nothing to keep and its skip is an A6 question instead.
Examples: #38093, #39544, #33745, #33586, #34070, #33763, #34464, #33752, #39013, #34882.

**A4. Classes that can run against one server configuration share the launch.**
Spot: several classes in a group each launching a server in `setUpClass` with the same
or request-equivalent arguments. Launch cost usually dominates the cases themselves, so
this saves more than the case count suggests. Not this pattern: a flag that changes
server state (attention backend, page size, speculative decoding on or off) needs its
own launch; and sharing couples the classes, so one crash takes the others down with it.
Examples: #33641, #33756, #33944, #36736, #38014, #37252.

**A5. An end-to-end matrix whose only varying dimension is one layer belongs in a layer-level unit test.**
Spot: a full server launch matrix that exists to select a kernel or backend. Not this
pattern: the one end-to-end smoke per backend that stays -- the unit test covers the
numerics, but backend selection can still change integration (graph capture, memory pool
layout). Examples: #33596, #33611.

**A6. Skipped, disabled, or unreachable registrations are dead coverage; each carries a reason and an expiry, or is deleted.**
Spot: `disabled=` registrations or `skipTest` calls with no reference to what would
un-skip them; skips older than the issue they were waiting on; registrations naming a
retired `runner_config`; registered files whose TestCase classes never execute. Not
this pattern: a hardware-conditional skip (`skipUnless` on compute capability), which is
a gate; or a `disabled=` whose linked issue is still open, which is doing its job.
Examples: #32324, #39368, #38585, #34377, #33772, #34779, #33654, #34070.

---

## B. Is it in the right place, at the right cost?

**B1. A test runs on the stage and runner its resources actually require.**
Spot: CPU-only tests registered to a GPU `runner_config`; multi-GPU registrations for a
test that launches one server. Not this pattern: a test that does not use a GPU but
imports a CUDA-only module or `sgl_kernel` at import time; check before moving it.
Examples: #34074, #33654, #33605, #34913, #35220.

**B2. Trigger frequency matches signal per unit of cost: per-commit for fast, high-signal checks; nightly for model-level accuracy and perf; weekly or a shadow cadence for slow-moving coverage.**
Spot: a multi-GPU accuracy suite registered per-commit whose failures have never been
PR-specific; a nightly job that has been green for weeks, occupies several GPUs per run,
and whose coverage a weekly run would hold just as well; a retired platform still
running daily. Demoting is not
deleting -- the nightly and weekly grids exist to hold this coverage at the right cost.
Examples: #38011, #37990, #33809, #36814, #37532, #34204, #37586, #31749.

**B3. Declared cost matches measured cost; shard counts and timeouts derive from it.**
Spot: an `est_time` off from the measured run by an order of magnitude (the partitioner
packs shards with it when live stats are missing); a shard count or job timeout that has
not moved while the suite's contents have. Not this pattern: small drift on a file that
already has live stats, which override `est_time`. Examples: #35407, #36242,
#38238, #32408, #37435, #37452, #39194, #37532.

---

## C. Does it test the right thing?

**C1. Fixtures go through the real production path; the oracle is written independently.**
The logic that builds a fixture should stay inside the tested surface, and the reference
value must not come from the implementation under test, or a shared bug makes both sides
agree. Spot: weights copied into parameters by hand instead of through the weight
loader; a config assembled without the override entry points; literals that duplicate
what the model or config already declares; a mock missing a field the code under test
reads; a hand-built fake of an internal structure that has been patched in several
fix-up PRs, which is the signal to construct it through the real constructor or a
shared helper. Not this pattern: a deliberately minimal fake that isolates one unit --
the signal is repeated patching, not the existence of a fake. The oracle is the one thing
that must stay independent; harnesses, fixtures and eval entry points are shared (C6).
Examples: #33615, #37339, #34100, #33509, #33179, #34746, #36424, #37148, #37182,
#38314, #38315, #38418, #39019, #39100, #39101.

**C2. The test, not a default or the environment, decides which path it exercises.**
Spot: a test that relies on defaults to select an execution mode, capture range or memory
fraction, so a default change silently moves what is covered; a job-level env var that
changes which strategy a class exercises, so the class tests the wrong thing on every run
of that lane; a filter or setting leaking from one suite into the next; cached process
state that an env pin cannot override. Not this pattern: a test that exists to check the
defaults work; pinning it would delete that coverage. Examples: #34146, #33776, #33847,
#38221, #33772, #34300, #39411.

**C3. The assertion matches the semantics under test.**
Spot: exit-code assertions on a process expected to abort; equality assertions where the
contract allows ties; an absolute tolerance on a value whose error scales with magnitude.
Examples: #34017, #37873, #32410, #34272, #35795.

**C4. Thresholds come from a measured distribution; derived metrics are gated on the primary one.**
Spot: round-number thresholds with no recorded basis; a derived quantity asserted without
the accuracy result it depends on. Not this pattern: a deliberately loose bound whose job
is "the model did not fall apart", which is a design choice; flag the ones with no margin
or that keep being adjusted. Examples: #36570, #34145, #31702, #31748, #38725, #36290.

**C5. Randomness is removed before a tolerance is loosened.**
Spot: a kernel test with unseeded random inputs whose bound was widened to make it pass;
a comparison that could be bit-exact but asserts within an epsilon. Seed the inputs, pin
the RNG, or make the guard exact. Not this pattern: a kernel that is itself
nondeterministic (atomic or split-K reduction order), where seeding does not help and a
tolerance is the correct assertion. Examples: #32126, #37343, #30026, #35787, #34356,
#34607.

**C6. Shared test infrastructure has one implementation.**
Spot: the same eval, mixin or fixture reimplemented locally in several files, each with
its own threshold or drift. Not this pattern: a reference implementation written
independently inside a test, which C1 requires; this is about harnesses, not oracles.
Examples: #34477, #36979, #39906.

---

## D. Is it stable?

**D1. Resources are reused only after they are actually released, not after release was requested.**
Spot: a launch that follows a kill without waiting for the device memory to come back; a
teardown that sends SIGKILL without a graceful shutdown first; a container sharing the
host IPC namespace so `/dev/shm` leaks across jobs. Examples: #39545, #37194, #31871,
#32829, #36605, #37485, #38244, #38638.

**D2. Each role in a test, and each concurrent run, has its own exclusive resources.**
Spot: two roles in one test sharing a device, NIC or port by default; concurrent runs
sharing a mount or namespace. Examples: #39584, #33044, #34755, #39611, #35500.

**D3. Side artifacts do not decide the correctness verdict.**
Spot: a run that fails because a profile, metrics upload or perf dump was missing or
unreadable; profiling enabled in a run whose purpose is pass/fail. Examples: #33832,
#35489, #38629.

---

## E. Pipeline structure

**E1. Jobs and reruns come from one declarative definition, not hand-written or duplicated ones.**
Spot: a workflow where adding a runner means editing a matrix by hand; sharding sized by
a constant rather than by measured partition data; a rerun workflow that declares its own
image, deps or env instead of reusing the stage's. Examples: #34186, #33329, #31792,
#36257, #36775, #34324, #34195.

**E2. An expensive artifact is built once per run and reused downstream.**
Spot: the same build or setup block appearing in several jobs of the same run; the same
compile happening in every job of a matrix. Examples: #33461, #33384, #33597, #33460,
#37258.

**E3. Runner-pool availability is a single configuration switch, not a recurring workflow edit.**
Spot: the same job disabled and re-enabled by separate PRs, each touching the workflow
files, several times in a quarter. Examples: #31764, #32719, #35627, #35607, #36981,
#37522, #38770, #38842, #39044.

**E4. Operational commands are gated on trust, scoped narrowly, and always answer.**
Spot: a command any commenter can trigger; a command that reruns far more than the caller
needs; an unrecognized command that is skipped silently; a reply that names the wrong
backend or run. Examples: #35750, #31980, #34057, #37618, #38734, #38736, #36778.

**E5. Fail-fast cascades link only jobs whose failures are correlated.**
Spot: a scheduled or manually dispatched run whose later jobs are cancelled by an earlier
unrelated failure; another platform's lane cancelled by a CUDA failure. Examples:
#35392, #35238, #36146.

**E6. Platform-specific CI triggers only on paths that affect it.**
Spot: a platform workflow that runs on every PR regardless of what changed; a lint stage
that runs a whole workspace's tests when none of its files moved. Examples: #36100, #34864.

---

## F. Install and cache cost

**F1. Build caches persist across jobs and runs, and every cache has a miss path.**
Spot: caches rebuilt per job; a cache miss that fails the job instead of falling back; a
cache key that does not change when its inputs do. Examples: #33361, #33619, #33460,
#34231, #33597, #35337, #33512, #32243.

**F2. The environment is pinned: toolchain and critical dependencies are fixed, survive later installs, and the checkout is what runs.**
Spot: build steps with no explicit toolchain, so an image bump silently changes output; a
transitive dependency that a later `pip install` upgrades or downgrades; an image built
from whatever main was at build time rather than the workflow commit; a site-packages
copy of the package under test taking precedence over the working tree. Examples:
#33437, #37508, #34768, #34322, #33738, #33441.

**F3. Repeated work inside the test process is cut.**
Spot: a checker that re-parses the whole package on every scan; a fixture that reloads a
tokenizer or model per case; independent subprocesses run serially; lint environments
rebuilt on every run. Examples: #36240, #36235, #36241, #37203, #34309.

**F4. Work whose precondition already holds is skipped: installs, disk reclamation, cache refreshes.**
Spot: unconditional installs; a skip check whose matching is too strict to fire on real
package name variants; unconditional disk reclamation; cache-base refreshes on every ref.
Examples: #33757, #37689, #33637, #33644, #33904.

**F5. A published package is preferred over a source clone; a required clone authenticates and retries.**
Spot: anonymous `git clone` in an install script; a dependency built from source that
ships a wheel. Examples: #37672, #37504, #37647, #34635.

**F6. Build outputs match the platforms and ABIs of the runner fleet.**
Spot: single-architecture images consumed by heterogeneous runners; an extension built
for one ABI when jobs run on two. Examples: #34276, #34253, #33619, #37820.

---

## G. Observability

**G1. Job names and status surfaces state the configuration that actually ran.**
Spot: an image version or platform variant only discoverable in the log; a status block
that omits a lane the PR gate depends on. Examples: #35686, #34813, #34877.

**G2. CI reporting carries queue time and runner health alongside pass/fail, and is shaped to be read.**
Spot: notifications that only say whether a run passed; a scheduled job missing from the
monitor that tracks its peers; a digest with no local timestamps, no filtering to
exceptions, and no trend over time. Examples: #37881, #33753, #37884, #38380.

---

## H. Guardrails

**H1. Every convention and every silent failure mode has a check that can fail.**
Spot: a rule that lives only in a review comment or a doc; a registered file whose tests
never run still reporting success; a suite whose job is never dispatched; a coverage job
that has failed on every run without anyone noticing; a sweep touching hundreds of test
files with no before/after registration count per backend. Silent non-execution is more
dangerous than a red test. Examples: #32735, #34074, #34147, #39697, #38581, #37436.
