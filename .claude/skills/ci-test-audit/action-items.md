# CI / Test Action Items

Patterns that have been applied to this repo's tests and CI. Each entry is a general
rule plus a way to spot it; the PR numbers are evidence, not definitions, and are
expected to be replaced as newer examples appear.

---

## A. Orchestration structure

**A1. Put a test on the stage and runner its resources actually require.**
Spot: CPU-only tests registered to a GPU `runner_config`; multi-GPU registrations for a
test that launches one server. Examples: #34074, #33654, #33605, #34913.

**A2. Generate jobs from declarative config instead of hand-written job lists.**
Spot: a workflow where adding a runner means editing a matrix by hand; sharding sized by
a constant rather than by measured partition data. Examples: #34186, #33329.

**A3. Factor a step repeated across jobs into one reusable workflow or a single upstream job.**
Spot: the same build or setup block appearing in several jobs of the same run.
Examples: #33461, #33384, #33597.

**A4. Keep rerun entry points on the same environment definition as the stage they rerun.**
Spot: a rerun workflow that declares its own image, deps or env. Examples: #34195.

**A5. Gate operational slash commands on trust, and delete ones whose blast radius is too coarse.**
Spot: a command any commenter can trigger; a command that reruns far more than the caller needs.
Examples: #35750, #31980.

**A6. Give recurring repo maintenance to a scheduled workflow.**
Spot: upkeep that only happens when someone remembers. Examples: #34380.

---

## B. Cost and caching

**B1. Make install steps idempotent -- skip when the requirement is already satisfied.**
Spot: unconditional installs; a skip check whose matching is too strict to fire on real
package name variants. Examples: #33757, #37689, #33637.

**B2. Persist build caches across jobs and runs, and give every cache a miss path.**
Spot: caches rebuilt per job; a cache miss that fails the job instead of falling back.
Examples: #33361, #33619, #33460, #34231, #33597.

**B3. Build an expensive artifact once per run and download it downstream.**
Spot: the same compile happening in every job of a matrix. Examples: #33384, #33460.

**B4. Make heavyweight operations conditional.**
Spot: unconditional disk reclamation, cache-base refreshes on every ref. Examples: #33644, #33904.

**B5. Build against a pinned toolchain rather than the image default.**
Spot: build steps with no explicit toolchain, so an image bump silently changes output.
Examples: #33437.

**B6. Remove installed remnants that shadow the checkout.**
Spot: a site-packages copy of the package under test taking precedence over the working tree.
Examples: #33441.

**B7. Prefer a published package over a source clone; when a clone is required, authenticate and retry.**
Spot: anonymous `git clone` in an install script; a dependency built from source that
ships a wheel. Examples: #37672, #37504, #37647.

**B8. Cover every target platform and ABI the runners use, and let callers name the output.**
Spot: single-architecture images consumed by heterogeneous runners; a build workflow with
a hardcoded output tag. Examples: #34276, #34253, #33619.

---

## C. Matrix reduction

**C1. Drop a variant whose config is a strict subset of another variant's.**
Holds when three things are true: the configs differ by one dimension; that dimension is
additive (it adds load or a knob position rather than gating a path or skipping
assertions); and the heavier variant's assertions cover the lighter one's. Watch for
variants that override a test method, and for looser expectations in the heavier variant
(a longer timeout or relaxed bound), which make the lighter one still worth keeping.
Examples: #38093, #39544, #33745, #33586, #34070.

**C2. Merge suites that differ only in launch arguments so they share one server or engine.**
Spot: several classes in a group whose `setUpClass` differs by a flag. Launch cost usually
dominates the cases themselves, so this saves more than the case count suggests.
Examples: #33641, #33756, #33944.

**C3. Delete registrations with no live runner or no reachable test.**
Spot: registrations naming a retired `runner_config`; registered files whose TestCase
classes never execute. Examples: #33654, #34070.

**C4. Size shard counts from measured duration.**
Spot: a shard count that has not moved while the suite's contents have. Examples: #35407.

---

## D. Test shape

**D1. Replace an end-to-end matrix with a layer-level unit test when the thing under test is one layer.**
Spot: a full server launch matrix whose only varying dimension is a kernel or backend
selection. Examples: #33596, #33611.

**D2. Build fixtures through the real production path, and write the oracle independently.**
Weights should go through the real weight loader and configs through the real override
entry points, so loader logic stays inside the tested surface. The reference value must
not come from the implementation under test, or a shared bug makes both sides agree.
Mocks must carry every field the code under test actually reads.
Examples: #33615, #37339, #34100.

**D3. Route evals through one shared entry point.**
Spot: the same metric computed by more than one local implementation, each with its own
threshold. Examples: #34477, #36979, #39906.

---

## E. Reliability

**E1. Hand off resources on real release, not on the signal that asks for it.**
Spot: a launch that follows a kill without waiting for the device memory to come back; a
teardown that sends SIGKILL without a graceful shutdown first. Examples: #39545, #37194.

**E2. Give each role in a test its own exclusive resources.**
Spot: two roles in one test sharing a device, NIC or port by default. Examples: #39584.

**E3. Judge by the signal that matches the semantics under test.**
Spot: exit-code assertions on a process expected to abort; equality assertions where the
contract allows ties. Examples: #34017, #37873.

**E4. Set thresholds from a measured distribution, and gate derived metrics on the primary one.**
Spot: round-number thresholds with no recorded basis; a derived quantity asserted without
the accuracy result it depends on. Examples: #36570, #34145.

**E5. Pin the test to the path it means to exercise.**
Spot: a test that relies on defaults to select an execution mode or capture range, so a
default change silently moves what is covered. Examples: #34146, #33776, #33847.

**E6. Derive test inputs from the object under test rather than hardcoding them.**
Spot: literals that duplicate something the model or config already declares. Examples: #33509.

**E7. Clear scheduled and sanity failures in batches, on a cadence.**
Spot: a scheduled suite that has been red long enough for the cause to be forgotten.
Examples: #34637, #34523, #39892.

---

## F. Observability

**F1. Report queue time and runner health alongside pass/fail.**
Spot: notifications that only say whether a run passed. Examples: #37881.

**F2. Tune notifications for reading: local timestamps, exceptions only, trends over logs.**
Spot: a digest nobody opens. Examples: #37884, #38380.

---

## G. Guardrails

**G1. Add a lint for failure modes that are silent.**
Spot: a registered file whose tests never run still reports success. Silent non-execution
is more dangerous than a red test. Examples: #32735.

**G2. Ship every new CI convention with something that can fail when it is violated.**
Spot: a rule that lives only in a review comment or a doc. Examples: #32735, #34074.
