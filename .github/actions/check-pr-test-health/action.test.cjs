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
  failedJobNames = ['model-test'], jobs, independentTestJobPrefix = '',
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
     { env: { GITHUB_WORKSPACE: WORKSPACE, INDEPENDENT_TEST_JOB_PREFIX: independentTestJobPrefix } }, require);
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
