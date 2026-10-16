const assert = require('node:assert/strict');
const { test } = require('node:test');
const { run } = require('./pr-check-comment.cjs');

const SHA = 'a'.repeat(40);
const TITLE_ERROR = 'PR title must use "[Type] scope: title".';

function workflowRun(id, conclusion, status = 'completed') {
  return { id, status, conclusion, html_url: `https://example.test/runs/${id}` };
}

// Fake API over one fork PR; `runs` maps workflow file to its runs, newest first.
function setup({ runs, comments = [], minimized = [], prSha = SHA, draft = false }) {
  const posted = [];
  const minimizedNow = [];
  const github = {
    rest: {
      pulls: {
        list: async params => {
          assert.equal(params.head, 'someone:fix-branch');
          return { data: [{ number: 7, head: { sha: prSha } }] };
        },
        get: async () => ({ data: { number: 7, state: 'open', draft, head: { sha: prSha } } }),
      },
      actions: {
        listWorkflowRuns: async ({ workflow_id, head_sha }) => {
          assert.equal(head_sha, prSha);
          return { data: { workflow_runs: runs[workflow_id] || [] } };
        },
        listJobsForWorkflowRun: async ({ run_id }) => ({
          data: { jobs: [{ id: run_id * 10, conclusion: 'failure' }] },
        }),
      },
      checks: {
        listAnnotations: async () => ({
          data: [
            { annotation_level: 'failure', message: TITLE_ERROR },
            { annotation_level: 'failure', message: 'Process completed with exit code 1.' },
          ],
        }),
      },
      issues: {
        listComments: Symbol('listComments'),
        createComment: async ({ issue_number, body }) => posted.push({ issue_number, body }),
      },
    },
    paginate: async (method, { issue_number }) => {
      assert.equal(method, github.rest.issues.listComments);
      assert.equal(issue_number, 7);
      return comments;
    },
    graphql: async (query, vars) => {
      if (query.startsWith('query')) {
        return { nodes: vars.ids.map(id => ({ id, isMinimized: minimized.includes(id) })) };
      }
      minimizedNow.push(vars.id);
      return {};
    },
  };
  const context = {
    repo: { owner: 'sgl-project', repo: 'sglang' },
    payload: {
      workflow_run: {
        event: 'pull_request',
        head_sha: SHA,
        head_branch: 'fix-branch',
        head_repository: { owner: { login: 'someone' } },
        pull_requests: [],
      },
    },
  };
  return { github, context, core: { info: () => {} }, posted, minimizedNow };
}

function botComment(nodeId, runIds) {
  return {
    node_id: nodeId,
    user: { login: 'github-actions[bot]' },
    body: `<!-- pr-check-comment runs=${runIds.join(',')} -->\nRequired checks failed`,
  };
}

test('one comment lists both failures, quoting only the title error', async () => {
  const env = setup({
    runs: { 'pr-title.yml': [workflowRun(1, 'failure')], 'lint.yml': [workflowRun(2, 'failure')] },
  });
  await run(env);
  assert.equal(env.posted.length, 1);
  const { body } = env.posted[0];
  assert.match(body, /^<!-- pr-check-comment runs=1,2 -->/);
  assert.match(body, /\*\*PR title\*\* \(\[run\]\(https:\/\/example.test\/runs\/1\)\)/);
  assert.match(body, /\*\*Lint\*\* \(\[run\]\(https:\/\/example.test\/runs\/2\)\)/);
  assert.equal(body.split(TITLE_ERROR).length, 2, 'title error quoted once, lint text never');
  assert.doesNotMatch(body, /exit code/);
});

test('waits while the other check is still running', async () => {
  const env = setup({
    runs: {
      'pr-title.yml': [workflowRun(1, 'failure')],
      'lint.yml': [workflowRun(2, null, 'in_progress')],
    },
  });
  await run(env);
  assert.deepEqual(env.posted, []);
});

test('reported failures are not posted again', async () => {
  const env = setup({
    runs: { 'pr-title.yml': [workflowRun(1, 'success')], 'lint.yml': [workflowRun(2, 'failure')] },
    comments: [botComment('C1', [1, 2])],
  });
  await run(env);
  assert.deepEqual(env.posted, []);
  assert.deepEqual(env.minimizedNow, []);
});

test('a new failed run posts again and minimizes the older comments', async () => {
  const env = setup({
    runs: {
      'pr-title.yml': [workflowRun(3, 'failure'), workflowRun(1, 'failure')],
      'lint.yml': [workflowRun(2, 'failure')],
    },
    comments: [botComment('C0', [9]), botComment('C1', [1, 2])],
    minimized: ['C0'],
  });
  await run(env);
  assert.match(env.posted[0].body, /^<!-- pr-check-comment runs=3,2 -->/);
  assert.deepEqual(env.minimizedNow, ['C1']);
});

test('all checks passing minimizes earlier comments', async () => {
  const env = setup({
    runs: { 'pr-title.yml': [workflowRun(3, 'success')], 'lint.yml': [workflowRun(4, 'success')] },
    comments: [botComment('C1', [1, 2])],
  });
  await run(env);
  assert.deepEqual(env.posted, []);
  assert.deepEqual(env.minimizedNow, ['C1']);
});

test('a marker in a comment from anyone else is ignored', async () => {
  const forged = { ...botComment('X', [1, 2]), user: { login: 'someone' } };
  const env = setup({
    runs: { 'pr-title.yml': [workflowRun(1, 'failure')], 'lint.yml': [workflowRun(2, 'success')] },
    comments: [forged],
  });
  await run(env);
  assert.equal(env.posted.length, 1);
  assert.deepEqual(env.minimizedNow, []);
});

test('skips runs for an older commit and draft PRs', async () => {
  const runs = { 'pr-title.yml': [workflowRun(1, 'failure')] };
  for (const env of [setup({ runs, prSha: 'b'.repeat(40) }), setup({ runs, draft: true })]) {
    await run(env);
    assert.deepEqual(env.posted, []);
  }
});
