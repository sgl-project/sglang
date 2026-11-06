const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');

const workflow = readFileSync(resolve(__dirname, '../workflows/pr-states.yml'), 'utf8');
const script = workflow.split(/          script: \|\r?\n/)[1]
  .split(/\r?\n/).map(line => line.replace(/^            /, '')).join('\n');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const run = new AsyncFunction('github', 'context', 'core', 'setTimeout', script);

async function render({ labels = [], runs = {}, event = 'pull_request_target', body = '' } = {}) {
  const lookups = [];
  let updatedBody;
  const github = { rest: {
    pulls: {
      get: async () => ({ data: { head: { sha: 'head' }, labels: labels.map(name => ({ name })), body } }),
      update: async args => { updatedBody = args.body; },
    },
    actions: { listWorkflowRuns: async args => {
      assert.equal(args.head_sha, 'head');
      assert.equal(args.event, 'pull_request');
      lookups.push(args.workflow_id);
      return { data: { workflow_runs: runs[args.workflow_id] || [] } };
    } },
  } };
  const context = { eventName: event, repo: { owner: 'o', repo: 'r' }, payload: event === 'workflow_run'
    ? { workflow_run: { event: 'pull_request', pull_requests: [{ number: 1 }] } }
    : { pull_request: { number: 1 } } };
  await run(github, context, { info() {}, setFailed(message) { assert.fail(message); } }, callback => callback());
  return { body: updatedBody, lookups };
}

test('subscribes to AMD Extra workflow completion', () => {
  assert.match(workflow, /workflows: .*"PR Test Extra \(AMD\)"/);
});

test('AMD Extra opt-in is independent of NVIDIA Extra', async () => {
  const { body } = await render({ labels: ['run-ci', 'run-ci-nv-extra'] });
  assert.match(body, /slot:pr-test-amd-extra:start.*run-ci-amd-extra/);
  assert.match(body, /slot:pr-test-extra:start.*Not run on latest push/);
});

test('reports a failed AMD Extra run separately from successful AMD Base', async () => {
  const { body, lookups } = await render({
    event: 'workflow_run', labels: ['run-ci', 'run-ci-amd-extra'],
    runs: {
      'pr-test-amd.yml': [{ id: 10, html_url: 'https://example.com/base', status: 'completed', conclusion: 'success' }],
      'pr-test-amd-extra.yml': [{ id: 20, html_url: 'https://example.com/extra', status: 'completed', conclusion: 'failure' }],
    },
  });
  assert.equal(new Set(lookups).size, 4);
  assert.match(body, /slot:pr-test-amd:start.*:white_check_mark:.*Run #10/);
  assert.match(body, /slot:pr-test-amd-extra:start.*:x:.*Run #20/);
});

test('keeps an in-flight Extra result after opt-in label removal', async () => {
  const { body } = await render({ labels: ['run-ci'], runs: {
    'pr-test-amd-extra.yml': [{ id: 20, html_url: 'https://example.com/extra', status: 'in_progress', conclusion: null }],
  } });
  assert.match(body, /slot:pr-test-amd-extra:start.*:hourglass_flowing_sand:.*Run #20/);
});

test('refresh replaces the states block without duplicating rows', async () => {
  const first = await render();
  const second = await render({ labels: ['run-ci', 'run-ci-amd-extra'], body: first.body });
  assert.equal(second.body.match(/Latest PR Test Extra \(AMD\):/g).length, 1);
  assert.match(second.body, /slot:pr-test-amd-extra:start.*Not run on latest push/);
});
