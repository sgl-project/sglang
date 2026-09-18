# CI / Test Action Items

Patterns that have been applied to this repo's tests and CI. Each entry is a general
rule plus a way to spot it in this repo; the PR numbers are evidence, not definitions,
and are expected to be replaced as newer examples appear.

Sections follow the order of questions an audit asks about a test: should it exist, is
it in the right place, does it test the right thing, is it stable. The remaining
sections are about the pipeline around the tests.

---

## A. Should this test exist in this form?

**A1. A variant whose configuration is a strict subset of another variant's is redundant.**
Holds when three things are true: the configs differ by one dimension; that dimension is
additive (it adds load or a knob position rather than gating a path or skipping
assertions); and the heavier variant's assertions cover the lighter one's. Watch for
variants that override a test method, and for looser expectations in the heavier variant
(a longer timeout or relaxed bound), which make the lighter one still worth keeping.
Examples: #38093, #39544, #33745, #33586, #34070, #33763, #34464, #33752, #39013, #34882.

**A2. Suites that differ only in launch arguments should share one server or engine.**
Spot: several classes in a group whose `setUpClass` differs by a flag. Launch cost usually
dominates the cases themselves, so this saves more than the case count suggests.
Examples: #33641, #33756, #33944, #36736, #38014, #37252.

**A3. An end-to-end matrix whose only varying dimension is one layer belongs in a layer-level unit test.**
Spot: a full server launch matrix that exists to select a kernel or backend. Examples: #33596, #33611.

**A4. A skip or disable carries a reason and an expiry; long-lived ones are dead coverage.**
Spot: `disabled=` registrations or `skipTest` calls with no reference to what would
un-skip them; skips older than the issue they were waiting on. Examples: #32324, #39368,
#38585, #34377, #33772, #34779.

**A5. A registration with no live runner or no reachable test is noise.**
Spot: registrations naming a retired `runner_config`; registered files whose TestCase
classes never execute. Examples: #33654, #34070.

---

## B. Is it in the right place, at the right cost?

**B1. A test runs on the stage and runner its resources actually require.**
Spot: CPU-only tests registered to a GPU `runner_config`; multi-GPU registrations for a
test that launches one server; large-model accuracy suites gating every PR when a
scheduled run would do. Examples: #34074, #33654, #33605, #34913, #33809, #36814,
#37532, #38011, #37990, #35220.

**B2. Declared cost matches measured cost; shard counts and timeouts derive from it.**
Spot: an `est_time` off from the measured run by more than a small factor (the
partitioner packs shards with it when live stats are missing); a shard count or job
timeout that has not moved while the suite's contents have. Examples: #35407, #36242,
#38238, #32408, #37435, #37452, #39194, #37532.

---

## C. Does it test the right thing?

**C1. Fixtures go through the real production path; the oracle is written independently.**
Weights through the real weight loader, configs through the real override entry points,
so loader logic stays inside the tested surface. The reference value must not come from
the implementation under test, or a shared bug makes both sides agree. Mocks must carry
every field the code under test reads -- and a hand-built fake of an internal structure
that needs patching after every refactor is the signal to stop hand-building it and
construct it through the real constructor or a shared helper. Examples: #33615, #37339,
#34100, #33179, #34746, #36424, #37148, #37182, #38314, #38315, #38418, #39019, #39100,
#39101.

**C2. The test is pinned to the path it means to exercise.**
Spot: a test that relies on defaults to select an execution mode, capture range or memory
fraction, so a default change silently moves what is covered. Examples: #34146, #33776,
#33847, #38221, #33772.

**C3. The test does not depend on ambient runner state.**
Spot: a job-level env var that changes which strategy a class exercises, so the class
tests the wrong thing on every run of that lane; a filter or setting leaking from one
suite into the next; cached process state that an env pin cannot override. Examples:
#34300, #39411.

**C4. The assertion matches the semantics under test.**
Spot: exit-code assertions on a process expected to abort; equality assertions where the
contract allows ties; an absolute tolerance on a value whose error scales with magnitude.
Examples: #34017, #37873, #32410, #34272, #35795.

**C5. Thresholds come from a measured distribution; derived metrics are gated on the primary one.**
Spot: round-number thresholds with no recorded basis; a derived quantity asserted without
the accuracy result it depends on. Examples: #36570, #34145, #31702, #31748, #38725, #36290.

**C6. Randomness is removed before a tolerance is loosened.**
Spot: a kernel test with unseeded random inputs whose bound was widened to make it pass;
a comparison that could be bit-exact but asserts within an epsilon. Seed the inputs, pin
the RNG, or make the guard exact. Examples: #32126, #37343, #30026, #35787, #34356, #34607.

**C7. Inputs derive from the object under test rather than being hardcoded.**
Spot: literals that duplicate something the model or config already declares. Examples: #33509.

**C8. One metric has one implementation.**
Spot: the same eval computed by more than one local implementation, each with its own
threshold. Examples: #34477, #36979, #39906.

---

## D. Is it stable?

**D1. Resources are handed off on real release, not on the signal that asks for it.**
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

**D4. Scheduled and sanity failures are cleared in batches, on a cadence.**
Spot: a scheduled suite that has been red long enough for the cause to be forgotten.
Examples: #34637, #34523, #39892, #32091, #35044, #36413.

---

## E. Pipeline structure

**E1. Jobs are generated from declarative config, not hand-written lists.**
Spot: a workflow where adding a runner means editing a matrix by hand; sharding sized by
a constant rather than by measured partition data. Examples: #34186, #33329, #31792,
#36257, #36775, #34324.

**E2. An expensive artifact is built once per run and reused downstream.**
Spot: the same build or setup block appearing in several jobs of the same run; the same
compile happening in every job of a matrix. Examples: #33461, #33384, #33597, #33460, #37258.

**E3. Availability of a runner pool is one switch, not a workflow edit each time.**
Spot: the same job disabled and re-enabled by separate PRs, each touching the workflow
files, several times in a quarter. Examples: #31764, #32719, #35627, #35607, #36981,
#37522, #38770, #38842, #39044.

**E4. Operational commands are gated on trust, scoped narrowly, and always answer.**
Spot: a command any commenter can trigger; a command that reruns far more than the caller
needs; an unrecognized command that is skipped silently; a reply that names the wrong
backend or run. Examples: #35750, #31980, #34057, #37618, #38734, #38736, #36778.

**E5. Fast-fail cascades link only jobs whose failures are correlated.**
Spot: a scheduled or manually dispatched run whose later jobs are cancelled by an earlier
unrelated failure; another platform's lane cancelled by a CUDA failure. Examples:
#35392, #35238, #36146.

**E6. Recurring maintenance runs on a schedule, placed to avoid contention.**
Spot: upkeep that only happens when someone remembers; nightlies that collide on the same
runner pool at the same hour. Examples: #34380, #31749, #32011, #37586.

**E7. Platform-specific CI triggers only on paths that affect it.**
Spot: a platform workflow that runs on every PR regardless of what changed; a lint stage
that runs a whole workspace's tests when none of its files moved. Examples: #36100, #34864.

**E8. Rerun entry points use the same environment definition as the stage they rerun.**
Spot: a rerun workflow that declares its own image, deps or env. Examples: #34195.

---

## F. Install and cache cost

**F1. Build caches persist across jobs and runs, and every cache has a miss path.**
Spot: caches rebuilt per job; a cache miss that fails the job instead of falling back; a
cache key that does not change when its inputs do. Examples: #33361, #33619, #33460,
#34231, #33597, #35337, #33512, #32243.

**F2. The toolchain and the dependencies that matter are pinned, and checked after later installs.**
Spot: build steps with no explicit toolchain, so an image bump silently changes output; a
transitive dependency that a later `pip install` upgrades or downgrades; an image built
from whatever main was at build time rather than the workflow commit. Examples: #33437,
#37508, #34768, #34322, #33738.

**F3. Repeated work inside the test process is cut.**
Spot: a checker that re-parses the whole package on every scan; a fixture that reloads a
tokenizer or model per case; independent subprocesses run serially; lint environments
rebuilt on every run. Examples: #36240, #36235, #36241, #37203, #34309.

**F4. Install steps are idempotent -- satisfied requirements are skipped.**
Spot: unconditional installs; a skip check whose matching is too strict to fire on real
package name variants. Examples: #33757, #37689, #33637.

**F5. A published package beats a source clone; a required clone authenticates and retries.**
Spot: anonymous `git clone` in an install script; a dependency built from source that
ships a wheel. Examples: #37672, #37504, #37647, #34635.

**F6. Build outputs cover every platform and ABI the runners use, and callers can name the output.**
Spot: single-architecture images consumed by heterogeneous runners; a build workflow with
a hardcoded output tag. Examples: #34276, #34253, #33619, #37820.

**F7. Heavyweight operations are conditional.**
Spot: unconditional disk reclamation; cache-base refreshes on every ref. Examples: #33644, #33904.

**F8. Nothing installed shadows the checkout.**
Spot: a site-packages copy of the package under test taking precedence over the working tree.
Examples: #33441.

---

## G. Observability

**G1. Job names and status surfaces state the configuration that actually ran.**
Spot: an image version or platform variant only discoverable in the log; a status block
that omits a lane the PR gate depends on. Examples: #35686, #34813, #34877.

**G2. Queue time and runner health are reported alongside pass/fail.**
Spot: notifications that only say whether a run passed; a scheduled job missing from the
monitor that tracks its peers. Examples: #37881, #33753.

**G3. Notifications are tuned for reading: local timestamps, exceptions only, trends over logs.**
Spot: a digest nobody opens. Examples: #37884, #38380.

---

## H. Guardrails

**H1. Every convention and every silent failure mode has a check that can fail.**
Spot: a rule that lives only in a review comment or a doc; a registered file whose tests
never run still reporting success; a suite whose job is never dispatched; a coverage job
that has failed on every run without anyone noticing. Silent non-execution is more
dangerous than a red test. Examples: #32735, #34074, #34147, #39697.

**H2. A large consolidation carries a guard that registrations and coverage did not drop.**
Spot: a sweep touching hundreds of test files with no before/after registration count per
backend; a coverage dip on the day it merged. Examples: #38581, #37436.
