const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const YAML = require('yaml');

const workflow = name => YAML.parse(readFileSync(
  path.join(__dirname, '../workflows', name), 'utf8',
));
const optInWorkflow = workflow('_pr-test-opt-in.yml');
const script = optInWorkflow.jobs.labels.steps.find(step => step.id === 'labels').with.script;
const run = new (Object.getPrototypeOf(async function () {}).constructor)(
  'github', 'context', 'core', 'process', script,
);

async function check({ live = [], snapshot = [], extra = false, event = 'pull_request', error } = {}) {
  const outputs = {};
  const notices = [];
  let reads = 0;
  await run({ rest: { pulls: { get: async args => {
    reads++;
    assert.deepEqual(args, { owner: 'owner', repo: 'repo', pull_number: 42 });
    if (error) throw error;
    return { data: { labels: live.map(name => ({ name })) } };
  } } } }, {
    eventName: event,
    repo: { owner: 'owner', repo: 'repo' },
    issue: { number: 42 },
    payload: { pull_request: { labels: snapshot.map(name => ({ name })) } },
  }, {
    setOutput: (name, value) => { outputs[name] = value; },
    notice: message => notices.push(message),
  }, { env: { REQUIRE_EXTRA: String(extra) } });
  return { enabled: outputs['should-run'], notices, reads };
}

for (const extra of [false, true]) {
  test(`missing labels opt out without failing (extra=${extra})`, async () => {
    const result = await check({ extra });
    assert.equal(result.enabled, 'false');
    assert.match(result.notices[0], /run-ci/);
  });
}

test('base AMD only needs run-ci', async () => {
  assert.equal((await check({ live: ['run-ci'] })).enabled, 'true');
});

test('extra requires both labels', async () => {
  for (const live of [['run-ci'], ['run-ci-extra']]) {
    assert.equal((await check({ extra: true, live })).enabled, 'false');
  }
});

test('reruns pick up newly added labels', async () => {
  assert.equal((await check({ extra: true, live: ['run-ci', 'run-ci-extra'] })).enabled, 'true');
});

test('removed labels cannot authorize a rerun from a stale event', async () => {
  const result = await check({ extra: true, snapshot: ['run-ci', 'run-ci-extra'] });
  assert.equal(result.enabled, 'false');
});

test('pull_request_target reads live labels too', async () => {
  assert.equal((await check({ event: 'pull_request_target' })).enabled, 'false');
});

for (const event of ['workflow_dispatch', 'workflow_call', 'schedule', 'push']) {
  test(`${event} does not require PR metadata`, async () => {
    const result = await check({ event, extra: true });
    assert.equal(result.enabled, 'true');
    assert.equal(result.reads, 0);
  });
}

test('metadata API errors fail closed', async () => {
  await assert.rejects(check({ error: new Error('API unavailable') }), /API unavailable/);
});

// Evaluate the boolean/string subset used by these job conditions and outputs.
// Read the expressions from YAML so a later always() job cannot bypass opt-out.
function evaluate(expression, context) {
  const read = key => key.split('.').reduce((value, part) => value?.[part], context) ?? '';
  const body = expression.trim().replace(/^\$\{\{\s*|\s*\}\}$/g, '')
    .replace(/\b(?:needs|steps|inputs|github)(?:\.[\w-]+)+/g, key => `read('${key}')`)
    .replace(/\b(always|success|failure|cancelled)\(\)/g,
      (_, name) => String(name === 'always' || name === 'success'));
  const contains = (value, item) => value.includes(item);
  const format = (template, ...values) => template.replace(/\{(\d+)\}/g, (_, index) => values[index] ?? '');
  return new Function('read', 'contains', 'format', `return (${body});`)(read, contains, format);
}

function context(event = 'pull_request', enabled = 'false') {
  return {
    github: { event_name: event, repository: 'owner/repo', event: {
      pull_request: { head: { repo: { full_name: 'fork/repo' } } },
    } },
    inputs: {},
    needs: { 'call-gate': { result: 'success', outputs: { 'should-run': enabled } } },
    steps: { filter: { outputs: {
      main_package: 'true', sgl_kernel: 'true', jit_kernel: 'true', multimodal_gen: 'true',
    } }, 'run-mode': { outputs: { run_all_tests: 'false' } } },
  };
}

const amd = workflow('pr-test-amd.yml');
const flags = ['main_package', 'sgl_kernel', 'jit_kernel', 'multimodal_gen'];
function amdFlags(ctx) {
  return Object.fromEntries(flags.map(flag => [flag,
    evaluate(amd.jobs['check-changes'].outputs[flag], ctx),
  ]));
}

test('AMD skips change detection when not opted in', () => {
  assert.ok(!evaluate(amd.jobs['check-changes'].if, context()));
});

test('a failed strict gate cannot enable AMD change detection', () => {
  const ctx = context('pull_request', 'true');
  ctx.needs['call-gate'].result = 'failure';
  assert.ok(!evaluate(amd.jobs['check-changes'].if, ctx));
});

test('opted-in and scheduled AMD runs keep their changed-file flags', () => {
  assert.ok(evaluate(amd.jobs['check-changes'].if, context('pull_request', 'true')));
  assert.deepEqual(Object.values(amdFlags(context('pull_request', 'true'))), flags.map(() => 'true'));
  const scheduled = context('schedule');
  scheduled.needs['call-gate'].result = 'skipped';
  assert.ok(evaluate(amd.jobs['check-changes'].if, scheduled));
  assert.deepEqual(Object.values(amdFlags(scheduled)), flags.map(() => 'true'));
});

for (const file of ['pr-test.yml', 'pr-test-extra.yml', 'pr-test-amd-extra.yml', 'pr-test-amd.yml']) {
  test(`${file}: no build, GPU or wait job runs without opt-in`, () => {
    const ctx = context();
    ctx.needs['check-changes'] = file === 'pr-test-amd.yml'
      ? { result: 'skipped', outputs: {} }
      : { result: 'success', outputs: amdFlags(ctx) };
    ctx.needs['wait-for-stage-a-amd'] = { result: 'skipped' };
    ctx.needs['wait-for-stage-b-amd'] = { result: 'skipped' };
    ctx.needs['wait-for-base-a'] = { result: 'skipped' };
    ctx.needs['wait-for-base-b'] = { result: 'skipped' };
    for (const [name, job] of Object.entries(workflow(file).jobs)) {
      if (['check-changes', 'call-gate', 'notify-pr-states'].includes(name) || name.endsWith('-finish')) continue;
      assert.ok(job.if, `${name} needs an explicit gate`);
      assert.ok(!evaluate(job.if, ctx), `${name} must not run without opt-in`);
    }
  });
}

test('opted-in runs still pass the strict draft, label and cooldown gate', () => {
  const gate = optInWorkflow.jobs['pr-gate'];
  assert.equal(gate.uses, './.github/workflows/pr-gate.yml');
  assert.ok(!evaluate(gate.if, { needs: { labels: { outputs: { 'should-run': 'false' } } } }));
  assert.ok(evaluate(gate.if, { needs: { labels: { outputs: { 'should-run': 'true' } } } }));
});


test('scheduled CUDA extra runs still build and dispatch tests with a skipped PR gate', () => {
  const ctx = context('schedule');
  ctx.needs['call-gate'] = { result: 'skipped', outputs: {} };
  ctx.needs['check-changes'] = { result: 'success', outputs: amdFlags(ctx) };
  const jobs = workflow('pr-test-extra.yml').jobs;
  for (const name of ['sgl-kernel-build-wheels', 'rust-ext-build', 'simulator-test-cpu', 'extra-test']) {
    assert.ok(evaluate(jobs[name].if, ctx), `${name} must still run on schedule`);
  }
});


for (const event of ['schedule', 'workflow_dispatch']) {
  test(`Base ${event} still builds and dispatches tests with a skipped PR gate`, () => {
    const ctx = context(event);
    ctx.inputs.test_parallel_dispatch = event === 'workflow_dispatch';
    ctx.needs['call-gate'] = { result: 'skipped', outputs: {} };
    ctx.needs['check-changes'] = { result: 'success', outputs: amdFlags(ctx) };
    ctx.needs['wait-for-base-a'] = { result: 'skipped' };
    ctx.needs['wait-for-base-b'] = { result: 'skipped' };
    const jobs = workflow('pr-test.yml').jobs;
    for (const name of ['rust-ext-build', 'rust-ext-build-aarch64', 'base-a-test', 'base-a-test-cpu', 'base-b-test', 'base-c-test', 'base-c-test-aarch64', 'call-jit-kernel-tests', 'call-multimodal-gen-tests']) {
      assert.ok(evaluate(jobs[name].if, ctx), `${name} must still run for ${event}`);
    }
  });
}
