import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Link,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasState,
} from "cursor/canvas";

/**
 * Data below is measured, not estimated:
 *  - 423 PRs merged into sgl-project/sglang between 2026-08-07 and
 *    2026-08-14 18:39 UTC, joined against the Actions API by merge-commit SHA.
 *  - The AMD PR gate changed mid-window (PR #34204, 2026-08-12 10:31 UTC
 *    swapped it from pr-test-amd.yml to pr-test-amd-rocm720.yml), so each PR
 *    is scored against whichever workflow was the gate on its merge date.
 *    Both carry an identical on.pull_request.paths list.
 *  - Path predictions were reproduced with dorny/paths-filter's own matcher
 *    (picomatch with { dot: true }); the model agreed with reality on
 *    419 of 423 PRs.
 */

type Outcome = {
  label: string;
  all: number;
  eligible: number | null;
  tone: "success" | "danger" | "warning" | "info" | "neutral";
  note: string;
};

const OUTCOMES: Outcome[] = [
  {
    label: "Never triggered",
    all: 66,
    eligible: null,
    tone: "neutral",
    note: "No changed file matched on.pull_request.paths — docs-only and similar. Correct behaviour.",
  },
  {
    label: "Triggered, every job skipped",
    all: 11,
    eligible: null,
    tone: "warning",
    note: "Run created, no check-changes flag matched. Reports green while testing nothing.",
  },
  {
    label: "Blocked at pr-gate",
    all: 71,
    eligible: 71,
    tone: "danger",
    note: "Draft, missing run-ci label, or rate-limit cooldown. Zero GPU jobs.",
  },
  {
    label: "Reached a GPU, then cancelled",
    all: 196,
    eligible: 196,
    tone: "danger",
    note: "Killed mid-flight, a median of 1 minute after the PR merged.",
  },
  {
    label: "Ran to completion, red",
    all: 73,
    eligible: 73,
    tone: "warning",
    note: "A real verdict, but a failing one.",
  },
  {
    label: "Ran to completion, green",
    all: 4,
    eligible: 4,
    tone: "success",
    note: "The only PRs that merged on a passing AMD result.",
  },
];

const GATE_ONE_PATHS: Array<[string, string]> = [
  ["python/**", "Any Python change, including files no AMD filter later matches."],
  ["scripts/ci/**", "All CI scripts, not just the AMD ones."],
  ["test/**", "The whole test tree."],
  ["python/sglang/kernels/aot/**", "AOT kernel sources."],
  [".github/workflows/pr-test-amd-rocm720.yml", "The workflow editing itself."],
  ["docker/rocm.Dockerfile", "Creates a run, but matches no filter — see the gap below."],
];

const GATE_THREE: Array<[string, string, string, string]> = [
  [
    "main_package",
    "python/sglang/** (minus multimodal_gen and kernels/aot), python/pyproject_other.toml, scripts/ci/amd/*, scripts/ci/utils/*, test/**",
    "stage-a / b / c — 33 GPU jobs",
    "284",
  ],
  [
    "sgl_kernel",
    "python/sglang/kernels/aot/**",
    "2 kernel unit jobs, plus all of stage-a/b/c",
    "5",
  ],
  [
    "jit_kernel",
    "python/sglang/kernels/**, test/registered/kernels/**",
    "2 JIT unit + benchmark jobs",
    "80",
  ],
  [
    "multimodal_gen",
    "python/sglang/multimodal_gen/**, python/sglang/cli/**, srt/observability/**, kernels/ops/diffusion/**",
    "8 diffusion jobs",
    "101",
  ],
];

const POOLS: Array<[string, string, string, string, string]> = [
  ["linux-mi300-1gpu-sglang", "120", "19", "1443", "45 min"],
  ["linux-mi300-2gpu-sglang", "27", "2", "217", "33 min"],
  ["linux-mi300-4gpu-sglang", "2", "1", "25", "26 min"],
  ["linux-mi300-8gpu-sglang", "19", "4", "124", "18 min"],
  ["linux-mi35x-gpu-1", "4", "1", "77", "2 min"],
  ["linux-mi35x-gpu-8", "7", "5", "197", "35 min"],
  ["linux-mi35x-gpu-8.fabric", "1", "1", "57", "174 min"],
];

function Header() {
  return (
    <Stack gap={6}>
      <H1>AMD ROCm CI on upstream SGLang</H1>
      <Text tone="secondary">
        What actually triggers the AMD gate, and what happened to it across one week of merges into{" "}
        <Code>sgl-project/sglang</Code>. Window: 2026-08-07 to 2026-08-14, 423 merged PRs.
      </Text>
    </Stack>
  );
}

function Summary() {
  return (
    <Grid columns={4} gap={16}>
      <Stat value="423" label="PRs merged in the week" />
      <Stat value="346" label="Eligible for AMD GPU jobs" tone="info" />
      <Stat value="77" label="Got a complete verdict" tone="warning" />
      <Stat value="4" label="Merged on a green AMD run" tone="danger" />
    </Grid>
  );
}

function TriggerRules() {
  return (
    <Stack gap={12}>
      <H2>What triggers AMD CI</H2>
      <Text>
        Three gates, all of which must pass before a single GPU job starts. The first two are path
        checks that use different syntax and different path lists, which is where the coverage holes
        come from.
      </Text>

      <H3>Gate 1 — the run is created</H3>
      <Text tone="secondary">
        <Code>on.pull_request.paths</Code>. No <Code>types</Code> is declared, so only{" "}
        <Code>opened</Code>, <Code>synchronize</Code> and <Code>reopened</Code> fire — adding the{" "}
        <Code>run-ci</Code> label or marking a draft ready does <Text weight="semibold">not</Text>{" "}
        dispatch a run. There is no <Code>branches</Code> restriction.
      </Text>
      <Table
        headers={["Path pattern", "Effect"]}
        rows={GATE_ONE_PATHS.map(([p, note]) => [<Code>{p}</Code>, <Text size="small">{note}</Text>])}
      />

      <H3>Gate 2 — pr-gate.yml</H3>
      <Text tone="secondary">
        Called with all defaults, and evaluated at job runtime against a live API read, so a rerun
        picks up the current state. The PR must not be a draft and must carry the{" "}
        <Code>run-ci</Code> label. Actors without write access get a 120-minute cooldown — computed
        from <Code>pr-test.yml</Code> run history, so the cooldown is shared with CUDA CI.
      </Text>

      <H3>Gate 3 — check-changes selects the job groups</H3>
      <Text tone="secondary">
        A second, independent <Code>dorny/paths-filter</Code> evaluation. The count column is how
        many of the 357 triggered PRs matched each flag this week.
      </Text>
      <Table
        headers={["Flag", "Matches", "Dispatches", "PRs"]}
        columnAlign={["left", "left", "left", "right"]}
        rows={GATE_THREE.map(([flag, matches, jobs, n]) => [
          <Code>{flag}</Code>,
          <Text size="small" tone="secondary">
            {matches}
          </Text>,
          <Text size="small">{jobs}</Text>,
          <Text size="small">{n}</Text>,
        ])}
      />

      <Callout tone="warning" title="The gap between gate 1 and gate 3">
        <Text>
          Gate 1 admits <Code>scripts/ci/**</Code> and <Code>docker/rocm.Dockerfile</Code>, but gate
          3 only matches <Code>scripts/ci/amd/*</Code> and <Code>scripts/ci/utils/*</Code> — both
          single-level — and never matches the Dockerfile at all. Anything in that gap creates a run
          in which every job skips, and since nothing fails, the run reports green. Eleven PRs
          landed that way this week, including a ROCm Dockerfile change and a{" "}
          <Code>python/pyproject.toml</Code> bump. Also caught: <Code>scripts/ci/cuda/</Code>,{" "}
          <Code>scripts/ci/npu/</Code>, <Code>scripts/ci/xpu/</Code>, <Code>scripts/ci/slurm/**</Code>,{" "}
          <Code>scripts/ci/utils/diffusion/**</Code> and <Code>scripts/ci/runner_configs.yml</Code>.
        </Text>
      </Callout>
    </Stack>
  );
}

function Funnel() {
  const [scope, setScope] = useCanvasState<"all" | "eligible">("funnel-scope", "all");
  const eligibleOnly = scope === "eligible";
  const rows = eligibleOnly ? OUTCOMES.filter((o) => o.eligible !== null) : OUTCOMES;
  const total = eligibleOnly ? 346 : 423;

  return (
    <Stack gap={12}>
      <Row gap={8} align="center">
        <H2>Where the week's merges landed</H2>
      </Row>
      <Row gap={8} align="center">
        <Pill active={scope === "all"} onClick={() => setScope("all")}>
          All 423 merged
        </Pill>
        <Pill active={eligibleOnly} onClick={() => setScope("eligible")}>
          346 eligible for GPU jobs
        </Pill>
        <Text size="small" tone="tertiary">
          measured on each PR's merge commit
        </Text>
      </Row>

      <Table
        headers={["Outcome", "PRs", "Share", "What it means"]}
        columnAlign={["left", "right", "right", "left"]}
        rowTone={rows.map((o) => o.tone)}
        rows={rows.map((o) => {
          const n = eligibleOnly ? (o.eligible ?? 0) : o.all;
          return [
            <Text weight="medium">{o.label}</Text>,
            <Text>{n}</Text>,
            <Text tone="secondary">{Math.round((100 * n) / total)}%</Text>,
            <Text size="small" tone="secondary">
              {o.note}
            </Text>,
          ];
        })}
      />
      <Text size="small" tone="tertiary">
        Outcome of the AMD PR gate on each PR&rsquo;s merge commit &middot; GitHub Actions API joined
        by merge-commit SHA &middot; 423 PRs merged 2026-08-07 to 2026-08-14 18:39 UTC
      </Text>

      <Text tone="secondary">
        Of the 346 PRs that should have run AMD GPU jobs, 273 reached a GPU but only{" "}
        <Text weight="semibold">77 produced a complete verdict</Text> before merging — and 73 of
        those 77 were red. Put the other way: 95% of finished AMD runs this week were failing, and
        the remaining 269 eligible PRs merged with no AMD answer at all.
      </Text>
    </Stack>
  );
}

function WhyRunsDie() {
  return (
    <Stack gap={12}>
      <H2>The dominant failure mode is timing, not labels</H2>
      <Text>
        Cancellation, not the <Code>run-ci</Code> gate, is what removes most AMD signal. Of the 196
        cancelled runs, <Text weight="semibold">190 (97%) were killed at or after the merge</Text>,
        a median of one minute afterwards — that is{" "}
        <Code>cancel-pr-workflows-on-close.yml</Code> reacting to{" "}
        <Code>pull_request_target: closed</Code>. When killed, a median of only 25% of their GPU jobs
        had passed.
      </Text>

      <Stack gap={16}>
        <Card>
          <CardHeader>AMD run wall time vs time to merge</CardHeader>
          <CardBody>
            <BarChart
              categories={["AMD run", "To merge"]}
              series={[{ name: "Median elapsed (minutes)", data: [389, 210], tone: "danger" }]}
              valueSuffix=" min"
              horizontal
              height={150}
            />
            <Text size="small" tone="tertiary">
              Median elapsed minutes &middot; x-axis: minutes, y-axis: interval measured &middot;
              &ldquo;AMD run&rdquo; = wall time of runs that ran to completion (n=77);
              &ldquo;To merge&rdquo; = run start to PR merge, for runs that reached a GPU (n=273)
              &middot; GitHub Actions API, 2026-08-07 to 2026-08-14
            </Text>
            <Text size="small" tone="secondary">
              The AMD suite needs roughly twice the time it is allowed to exist. The merge routinely
              wins the race.
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>GPU job outcomes across the week</CardHeader>
          <CardBody>
            <BarChart
              categories={["Passed", "Cancelled", "Failed"]}
              series={[{ name: "GPU jobs (count)", data: [3225, 1962, 923] }]}
              valueSuffix=" jobs"
              horizontal
              height={180}
            />
            <Text size="small" tone="tertiary">
              Job count &middot; y-axis: job conclusion, x-axis: number of GPU jobs &middot; all
              6,110 AMD GPU jobs dispatched by the 273 PRs that reached a GPU &middot; GitHub
              Actions API, 2026-08-07 to 2026-08-14
            </Text>
            <Text size="small" tone="secondary">
              Roughly a third of all AMD GPU time this week produced a result that was thrown away.
            </Text>
          </CardBody>
        </Card>
      </Stack>

      <Callout tone="danger" title="The loop">
        <Text>
          Queues run hours deep, so a run finishes long after its PR is ready. The PR merges, the run
          is cancelled, and the capacity spent on it is wasted — which lengthens the queue for the
          next run. Adding runners alone does not break this; shortening the wall-clock time of a run
          until it beats the merge does.
        </Text>
      </Callout>
    </Stack>
  );
}

function Capacity() {
  return (
    <Stack gap={12}>
      <H2>Why a run takes hours</H2>
      <Text>
        One PR with only <Code>main_package</Code> set dispatches 33 GPU jobs. Pool sizes and load
        are from a 12-hour window ending 2026-08-14 05:02 UTC.
      </Text>
      <Table
        headers={["Runner pool", "Runners", "Jobs per PR run", "Jobs in 12h", "Avg queue wait"]}
        columnAlign={["left", "right", "right", "right", "right"]}
        rowTone={[
          undefined,
          undefined,
          "warning",
          undefined,
          undefined,
          "warning",
          "danger",
        ]}
        rows={POOLS.map(([pool, runners, per, jobs, wait]) => [
          <Code>{pool}</Code>,
          <Text>{runners}</Text>,
          <Text>{per}</Text>,
          <Text tone="secondary">{jobs}</Text>,
          <Text tone="secondary">{wait}</Text>,
        ])}
      />
      <Text size="small" tone="tertiary">
        Effective runner counts and mean queue wait per pool &middot; source:{" "}
        <Code>scripts/ci/utils/runner_utilization_report.py</Code> &middot; 12-hour window ending
        2026-08-14 05:02 UTC
      </Text>
      <Text tone="secondary">
        The binding constraint is <Code>linux-mi35x-gpu-8.fabric</Code>: one runner, one
        disaggregation job per <Code>main_package</Code> PR, a 60-minute timeout. It can serve at
        most about a dozen runs a day against a demand of roughly 40. Worst observed wait was 660
        minutes. The <Code>max-parallel</Code> throttles compound this — the 14-way stage-b matrix is
        capped at 4, and several other stages at 1 — which converts a wide run into a long one, and
        long runs are exactly what the merge cancels.
      </Text>
    </Stack>
  );
}

function OpenWork() {
  return (
    <Stack gap={12}>
      <H2>Open pull requests</H2>
      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader trailing={<Pill size="sm">verified live</Pill>}>
            #34812 — gate AMD tests on rocm.Dockerfile
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text size="small">
                Adds <Code>docker/rocm.Dockerfile</Code> to <Code>main_package</Code> and removes a
                dead <Code>python/pyproject_rocm.toml</Code> rule. The pins in that file are not
                inert: <Code>amd_ci_install_dependency.sh</Code> greps{" "}
                <Code>AITER_COMMIT_DEFAULT</Code> and <Code>MORI_COMMIT</Code> at job time and
                rebuilds against them, because the image is never rebuilt for a PR.
              </Text>
              <Text size="small" tone="secondary">
                Nine commits in 60 days touched only this file. One of them, a MoRI bump, reported a
                green AMD check with all 17 GPU jobs skipped.
              </Text>
              <Link href="https://github.com/sgl-project/sglang/pull/34812">View PR #34812</Link>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={<Pill size="sm">verified live</Pill>}>
            #34813 — show AMD state in the PR body
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text size="small">
                <Code>pr-states.yml</Code> writes a CI-states block into every PR description but
                only queries <Code>pr-test.yml</Code> and <Code>pr-test-extra.yml</Code>. A merge
                oncall reading the description cannot see AMD at all. Adds a third row plus a{" "}
                <Code>notify-pr-states</Code> job so reruns refresh it.
              </Text>
              <Text size="small" tone="secondary">
                Costs no runner capacity. The rendering cannot be tested by a PR run, since{" "}
                <Code>pull_request_target</Code> always executes the base-branch copy.
              </Text>
              <Link href="https://github.com/sgl-project/sglang/pull/34813">View PR #34813</Link>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Divider />

      <H3>Not yet addressed</H3>
      <Text>
        Both open PRs together cover the 11 skipped-everything PRs and the 71 gate-blocked ones. They
        do nothing for the 196 cancelled runs, which is the largest bucket. That one needs the run to
        get shorter than the time-to-merge: move the disaggregation and DeepSeek-V4 accuracy jobs off
        the per-PR path onto nightly or the existing <Code>run-ci-extra</Code> opt-in, which frees
        the single fabric runner and five of the seven MI35x 8-GPU slots per run, then relax the{" "}
        <Code>max-parallel</Code> caps so the remaining suite runs wide instead of long.
      </Text>
    </Stack>
  );
}

export default function AmdCiCoverage() {
  return (
    <Stack gap={28} style={{ padding: 4 }}>
      <Header />
      <Summary />
      <TriggerRules />
      <Funnel />
      <WhyRunsDie />
      <Capacity />
      <OpenWork />
    </Stack>
  );
}
