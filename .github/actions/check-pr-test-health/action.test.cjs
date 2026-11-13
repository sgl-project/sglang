const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');

const yaml = fs.readFileSync(path.join(__dirname, 'action.yml'), 'utf8');
const script = yaml.split('        script: |\n')[1]
  .split('\n').map(line => line.replace(/^          /, '')).join('\n');
// `require` and GITHUB_WORKSPACE mirror what actions/github-script injects.
const WORKSPACE = path.join(__dirname, '..', '..', '..');
const run = new (Object.getPrototypeOf(async function () {}).constructor)(
  'github', 'context', 'core', 'process', 'require', script,
);

async function check({
  snapshot = [], live = [], lint = 'success', event = 'pull_request',
  failedJobNames = ['model-test'], jobs, independentTestJobPrefix = '', skipHealthCheck = 'false',
} = {}) {
  const failures = [];
  let labelReads = 0;
  let jobReads = 0;
  const labels = names => names.map(name => ({ name }));
  const github = {
    rest: {
      checks: { listForRef: async () => ({ data: { check_runs: [
        { app: { slug: 'github-actions' }, status: 'completed', conclusion: lint },
      ] } }) },
      pulls: { get: async args => {
        assert.equal(args.pull_number, 42);
        labelReads++;
        return { data: { labels: labels(live) } };
      } },
      repos: { listPullRequestsAssociatedWithCommit: async () => {
        labelReads++;
        return { data: [{ labels: labels(live) }] };
      } },
      actions: { listJobsForWorkflowRun: () => {} },
    },
    paginate: async () => {
      jobReads++;
      return jobs || failedJobNames.map(name => ({
        name, status: 'completed', conclusion: 'failure',
        steps: [{ name: 'Run test', conclusion: 'failure' }],
      }));
    },
  };
  await run(github, {
    eventName: event, repo: { owner: 'owner', repo: 'repo' }, sha: 'head', runId: 1,
    payload: event === 'pull_request'
      ? { pull_request: { number: 42, head: { sha: 'head' }, labels: labels(snapshot) } }
      : {},
  }, { info: () => {}, setFailed: message => failures.push(message) },
     { env: { GITHUB_WORKSPACE: WORKSPACE, INDEPENDENT_TEST_JOB_PREFIX: independentTestJobPrefix,
       SKIP_PR_TEST_HEALTH_CHECK: skipHealthCheck } }, require);
  return { failures, labelReads, jobReads };
}

test('a label added after the event bypasses sibling failures on rerun', async () => {
  assert.deepEqual(await check({ live: ['bypass-fail-fast'] }),
    { failures: [], labelReads: 1, jobReads: 0 });
});

test('a removed label does not continue bypassing sibling failures', async () => {
  const result = await check({ snapshot: ['bypass-fail-fast'] });
  assert.equal(result.labelReads, 1);
  assert.equal(result.jobReads, 1);
  assert.match(result.failures[0], /root cause job\(s\): model-test/);
});

test('bypass never skips a failed lint check', async () => {
  assert.deepEqual(await check({ live: ['bypass-fail-fast'], lint: 'failure' }),
    { failures: ['Fail-fast: lint check failed'], labelReads: 0, jobReads: 0 });
});

test('non-PR events retain associated-PR label lookup', async () => {
  assert.deepEqual(await check({ event: 'workflow_dispatch', live: ['bypass-fail-fast'] }),
    { failures: [], labelReads: 1, jobReads: 0 });
});

test('scheduled runs remain exempt', async () => {
  assert.deepEqual(await check({ event: 'schedule' }),
    { failures: [], labelReads: 0, jobReads: 0 });
});

for (const stage of ['base-c-test-8-gpu-h20', 'extra-b-test-8-gpu-h20']) {
  for (const suffix of ['', ' (0)', ' / test (0)']) {
    const name = stage + suffix;
    test(`H20 failure does not cascade: ${name}`, async () => {
      const result = await check({ failedJobNames: [name] });
      assert.equal(result.jobReads, 1);
      assert.deepEqual(result.failures, []);
    });
  }
}

for (const runner of ['8-gpu-h200', '4-gpu-h100', '4-gpu-b200']) {
  const name = `extra-b-test-${runner} / test (0)`;
  test(`other extra runner failures still cascade: ${runner}`, async () => {
    const result = await check({ failedJobNames: [name] });
    assert.deepEqual(result.failures, [`Fail-fast: skipping — root cause job(s): ${name}`]);
  });
}

test('an H20 failure does not hide a simultaneous H200 failure', async () => {
  const h200 = 'extra-b-test-8-gpu-h200 / test (0)';
  const result = await check({ failedJobNames: ['extra-b-test-8-gpu-h20 / test (0)', h200] });
  assert.deepEqual(result.failures, [`Fail-fast: skipping — root cause job(s): ${h200}`]);
});

const diffusionPrefix = 'call-multimodal-gen-tests /';
const diffusionJob = (name, step = 'Run diffusion suite') => ({
  name: `${diffusionPrefix} ${name}`, status: 'completed', conclusion: 'failure',
  steps: [{ name: step, conclusion: 'failure' }],
});

test('independent diffusion suites collect peer test failures', async () => {
  const result = await check({ independentTestJobPrefix: diffusionPrefix,
    jobs: [diffusionJob('diffusion-1-gpu-5090 (0)')] });
  assert.deepEqual(result.failures, []);
});

test('other callers still stop on diffusion test failures', async () => {
  const job = diffusionJob('diffusion-1-gpu-5090 (0)');
  const result = await check({ jobs: [job] });
  assert.match(result.failures[0], /diffusion-1-gpu-5090/);
});

test('architecture-specific pre-suite failures do not stop independent suites', async () => {
  const result = await check({ independentTestJobPrefix: diffusionPrefix,
    jobs: [diffusionJob('diffusion-1-gpu-b200 (0)', 'Run pre-suite commands')] });
  assert.deepEqual(result.failures, []);
});

for (const step of ['Install dependencies', 'Run ./.github/actions/check-maintenance']) {
  test(`independent lanes still stop on setup failure: ${step}`, async () => {
    const result = await check({ independentTestJobPrefix: diffusionPrefix,
      jobs: [diffusionJob('diffusion-1-gpu-5090 (0)', step)] });
    assert.match(result.failures[0], /diffusion-1-gpu-5090/);
  });
}

test('independent test collection never skips a failed lint check', async () => {
  const result = await check({ independentTestJobPrefix: diffusionPrefix, lint: 'failure' });
  assert.deepEqual(result.failures, ['Fail-fast: lint check failed']);
});

test('independent peer failure does not hide a build failure', async () => {
  const result = await check({ independentTestJobPrefix: diffusionPrefix,
    jobs: [diffusionJob('diffusion-1-gpu-5090 (0)'), {
      name: 'sgl-kernel-build-wheels', status: 'completed', conclusion: 'failure',
      steps: [{ name: 'Build wheels', conclusion: 'failure' }],
    }] });
  assert.deepEqual(result.failures, ['Fail-fast: skipping — root cause job(s): sgl-kernel-build-wheels']);
});

test('prior finish summaries do not block producer retries', async () => {
  const result = await check({ failedJobNames: ['pr-test-finish'] });
  assert.deepEqual(result.failures, []);
});

const failed = (name, step) => ({ name, status: 'completed', conclusion: 'failure',
  steps: [{ name: step, conclusion: 'failure' }] });

test('a Base-B wait caused only by health cascades does not become a root', async () => {
  const result = await check({ jobs: [
    failed('wait-for-base-b', 'Run ./.github/actions/wait-for-jobs'),
    failed('base-b-test-1-gpu-large / test (0)', 'Run ./.github/actions/check-pr-test-health'),
  ] });
  assert.deepEqual(result.failures, []);
});

test('a standalone Base-B wait failure remains a root', async () => {
  const result = await check({ jobs: [failed('wait-for-base-b', 'Run ./.github/actions/wait-for-jobs')] });
  assert.match(result.failures[0], /wait-for-base-b/);
});

test('a real producer failure is not hidden with its Base-B wait', async () => {
  const result = await check({ jobs: [
    failed('wait-for-base-b', 'Run ./.github/actions/wait-for-jobs'),
    failed('base-b-test-1-gpu-large / test (0)', 'Run test'),
    failed('base-b-test-2-gpu-large / test (0)', 'Run ./.github/actions/check-pr-test-health'),
  ] });
  assert.match(result.failures[0], /wait-for-base-b/);
  assert.match(result.failures[0], /base-b-test-1-gpu-large/);
});

test('the generic action still blocks real roots behind a coverage summary', async () => {
  const result = await check({ jobs: [
    failed('call-multimodal-gen-tests / diffusion-coverage-check', 'Verify coverage'),
    diffusionJob('diffusion-1-gpu-5090 (0)'),
  ] });
  assert.match(result.failures[0], /diffusion-1-gpu-5090/);
  assert.doesNotMatch(result.failures[0], /coverage-check/);
});

const nondeterministicPrefix = 'stage-b-test-1-gpu-small-amd-nondeterministic (';
const nondeterministicJobName = `${nondeterministicPrefix}rocm10, linux-mi300-1gpu-sglang)`;

test('AMD nondeterministic test failures do not stop other AMD lanes', async () => {
  const result = await check({ independentTestJobPrefix: nondeterministicPrefix,
    failedJobNames: [nondeterministicJobName] });
  assert.deepEqual(result.failures, []);
});

test('the AMD exemption does not change other callers', async () => {
  const result = await check({ failedJobNames: [nondeterministicJobName] });
  assert.deepEqual(result.failures,
    [`Fail-fast: skipping — root cause job(s): ${nondeterministicJobName}`]);
});

test('the AMD exemption does not include similarly named suites', async () => {
  const name = 'stage-b-test-1-gpu-small-amd-nondeterministic-extra (rocm10)';
  const result = await check({ independentTestJobPrefix: nondeterministicPrefix,
    failedJobNames: [name] });
  assert.deepEqual(result.failures, [`Fail-fast: skipping — root cause job(s): ${name}`]);
});

test('AMD nondeterministic setup failures still stop other lanes', async () => {
  const result = await check({ independentTestJobPrefix: nondeterministicPrefix,
    jobs: [failed(nondeterministicJobName, 'Install dependencies')] });
  assert.deepEqual(result.failures,
    [`Fail-fast: skipping — root cause job(s): ${nondeterministicJobName}`]);
});

test('an AMD nondeterministic failure does not hide a deterministic failure', async () => {
  const name = 'stage-b-test-1-gpu-small-amd (rocm10, linux-mi300-1gpu-sglang, 0)';
  const result = await check({ independentTestJobPrefix: nondeterministicPrefix,
    failedJobNames: [nondeterministicJobName, name] });
  assert.deepEqual(result.failures, [`Fail-fast: skipping — root cause job(s): ${name}`]);
});

test('the AMD exemption never hides lint failure', async () => {
  const result = await check({ independentTestJobPrefix: nondeterministicPrefix, lint: 'failure' });
  assert.deepEqual(result.failures, ['Fail-fast: lint check failed']);
});

for (const name of ['pr-test-amd-finish', 'pr-test-amd-extra-finish',
  'call-pr-test-amd-extra / pr-test-amd-extra-finish']) {
  test(`AMD finish summaries do not block producer retries: ${name}`, async () => {
    const result = await check({ failedJobNames: [name] });
    assert.deepEqual(result.failures, []);
  });
}

for (const [waitName, producerName] of [
  ['wait-for-stage-a-amd', 'stage-a-test-1-gpu-small-amd (rocm10)'],
  ['wait-for-stage-b-amd', 'stage-b-test-1-gpu-small-amd (rocm10, linux-mi300-1gpu-sglang, 0)'],
]) {
  test(`AMD health-only wait cascades do not block producer retries: ${waitName}`, async () => {
    const result = await check({ jobs: [
      failed(waitName, 'Run ./.github/actions/wait-for-jobs'),
      failed(producerName, 'Check PR test health'),
    ] });
    assert.deepEqual(result.failures, []);
  });

  test(`a real AMD failure remains visible with its wait summary: ${waitName}`, async () => {
    const result = await check({ jobs: [
      failed(waitName, 'Run ./.github/actions/wait-for-jobs'),
      failed(producerName, 'Run test'),
    ] });
    assert.deepEqual(result.failures,
      [`Fail-fast: skipping — root cause job(s): ${waitName}, ${producerName}`]);
  });

  test(`a standalone AMD wait failure remains a root: ${waitName}`, async () => {
    const result = await check({ jobs: [failed(waitName, 'Run ./.github/actions/wait-for-jobs')] });
    assert.deepEqual(result.failures, [`Fail-fast: skipping — root cause job(s): ${waitName}`]);
  });
}

for (const event of ['pull_request', 'schedule', 'workflow_dispatch']) {
  test(`explicit diagnostic mode bypasses lint and sibling checks: ${event}`, async () => {
    const result = await check({ event, lint: 'failure', skipHealthCheck: 'true' });
    assert.deepEqual(result, { failures: [], labelReads: 0, jobReads: 0 });
  });
}

for (const filename of ['pr-test-amd.yml', 'pr-test-amd-extra.yml']) {
  test(`every AMD GPU job checks health immediately after checkout: ${filename}`, () => {
    const workflow = fs.readFileSync(path.join(WORKSPACE, '.github', 'workflows', filename), 'utf8');
    const jobs = workflow.split(/(?=^  [a-z][a-z0-9-]*:\n)/m);
    const gpuJobs = jobs.filter(job => /^    runs-on: (?!ubuntu-latest)/m.test(job));
    assert.ok(gpuJobs.length > 0);
    for (const job of gpuJobs) {
      const name = job.split('\n')[0].trim();
      const steps = job.split(/(?=^      - )/m).filter(step => /^      - /.test(step));
      const checkoutIndex = steps.findIndex(step => step.includes('uses: actions/checkout@'));
      assert.ok(checkoutIndex >= 0, name);
      assert.match(steps[checkoutIndex + 1], /uses: \.\/\.github\/actions\/check-pr-test-health/, name);
      assert.equal(steps.filter(step => step.includes('uses: ./.github/actions/check-pr-test-health')).length, 1, name);
      assert.match(steps[checkoutIndex + 1], /independent-test-job-prefix: "stage-b-test-1-gpu-small-amd-nondeterministic \("/, name);
      assert.doesNotMatch(steps[checkoutIndex + 1], /persist-credentials:/, name);
    }
    assert.match(workflow, /^  checks: read$/m);
    assert.match(workflow, /^  SKIP_PR_TEST_HEALTH_CHECK: .*github.event_name != 'pull_request'.*inputs.continue_on_error/m);
    if (filename === 'pr-test-amd.yml') {
      assert.match(workflow, /^  SKIP_PR_TEST_HEALTH_CHECK: .*inputs.run_all_tests/m);
    }
  });
}
