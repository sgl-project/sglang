const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');

const yaml = fs.readFileSync(path.join(__dirname, 'action.yml'), 'utf8');
const script = yaml.split('        script: |\n')[1]
  .split('\n').map(line => line.replace(/^          /, '')).join('\n');
const run = new (Object.getPrototypeOf(async function () {}).constructor)(
  'github', 'context', 'core', 'process', script,
);

async function check({ snapshot = [], live = [], lint = 'success', event = 'pull_request' } = {}) {
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
      return [{ name: 'model-test', status: 'completed', conclusion: 'failure', steps: [] }];
    },
  };
  await run(github, {
    eventName: event, repo: { owner: 'owner', repo: 'repo' }, sha: 'head', runId: 1,
    payload: event === 'pull_request'
      ? { pull_request: { number: 42, head: { sha: 'head' }, labels: labels(snapshot) } }
      : {},
  }, { info: () => {}, setFailed: message => failures.push(message) }, { env: {} });
  return { failures, labelReads, jobReads };
}

test('a label added after the event bypasses sibling failures on rerun', async () => {
  assert.deepEqual(await check({ live: ['bypass-fastfail'] }),
    { failures: [], labelReads: 1, jobReads: 0 });
});

test('a removed label does not continue bypassing sibling failures', async () => {
  const result = await check({ snapshot: ['bypass-fastfail'] });
  assert.equal(result.labelReads, 1);
  assert.equal(result.jobReads, 1);
  assert.match(result.failures[0], /root cause job\(s\): model-test/);
});

test('bypass never skips a failed lint check', async () => {
  assert.deepEqual(await check({ live: ['bypass-fastfail'], lint: 'failure' }),
    { failures: ['Fast-fail: lint check failed'], labelReads: 0, jobReads: 0 });
});

test('non-PR events retain associated-PR label lookup', async () => {
  assert.deepEqual(await check({ event: 'workflow_dispatch', live: ['bypass-fastfail'] }),
    { failures: [], labelReads: 1, jobReads: 0 });
});

test('scheduled runs remain exempt', async () => {
  assert.deepEqual(await check({ event: 'schedule' }),
    { failures: [], labelReads: 0, jobReads: 0 });
});
