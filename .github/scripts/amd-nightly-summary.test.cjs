const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');

const workflow = readFileSync(resolve(__dirname, '../workflows/nightly-test-amd.yml'), 'utf8');
const step = workflow.indexOf('      - name: Summarize nightly job results\n');
assert.notEqual(step, -1, 'Missing nightly summary step');
const match = workflow.slice(step).match(/          script: \|\n((?: {12}[^\n]*\n|\n)+)/);
assert.ok(match, 'Missing nightly summary script');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const summarize = new AsyncFunction('core', 'process', match[1].replace(/^ {12}/gm, ''));

async function run(results, jobFilter = '', jobSelect = '') {
  const failures = [];
  const warnings = [];
  const info = [];
  let summary = '';
  await summarize({
    setFailed: message => failures.push(message),
    warning: message => warnings.push(message),
    info: message => info.push(message),
    summary: {
      addRaw(text) { summary += text; return this; },
      async write() {},
    },
  }, {
    env: {
      NEEDS_JSON: JSON.stringify(Object.fromEntries(
        Object.entries(results).map(([name, result]) => [name, { result }])
      )),
      JOB_FILTER: jobFilter,
      JOB_SELECT: jobSelect,
    },
  });
  return { failures, warnings, info, summary };
}

test('full success reports logical groups without claiming test-case qualification', async () => {
  const result = await run({ unit: 'success', kernel: 'success' });
  assert.deepEqual(result.failures, []);
  assert.deepEqual(result.warnings, []);
  assert.match(result.summary, /Selected \*\*2 of 2\*\* logical job groups/);
  assert.match(result.summary, /\| success \| 2 \| 2 \|/);
  assert.match(result.summary, /all selected job groups reported success/);
  assert.match(result.summary, /Tolerated step failures are not visible/);
  assert.doesNotMatch(result.summary, /All ROCm nightly test jobs passed/);
});

test('selected success with unselected skips is explicitly partial', async () => {
  const result = await run({ unit: 'success', kernel: 'skipped', model: 'skipped' }, '', 'unit');
  assert.deepEqual(result.failures, []);
  assert.equal(result.warnings.length, 1);
  assert.match(result.summary, /Selected \*\*1 of 3\*\*/);
  assert.match(result.summary, /\| skipped \| 2 \| 0 \|/);
  assert.match(result.summary, /does not certify full nightly coverage/);
});

test('comma-separated filter overrides the dropdown', async () => {
  const result = await run(
    { unit: 'success', kernel: 'success', model: 'skipped' }, 'unit,kernel', 'model'
  );
  assert.deepEqual(result.failures, []);
  assert.match(result.summary, /Selected \*\*2 of 3\*\*/);
  assert.match(result.summary, /\| success \| 2 \| 2 \|/);
});

test('selection matches case-insensitively, including ALL and repeated names', async () => {
  for (const filter of ['ALL', 'UNIT,KERNEL', 'UNIT,unit,kernel']) {
    const result = await run({ unit: 'success', kernel: 'success' }, filter);
    assert.deepEqual(result.failures, []);
    assert.match(result.summary, /Selected \*\*2 of 2\*\*/);
  }
});

test('comma boundaries do not select another job with the same prefix', async () => {
  const result = await run({
    'nightly-8-gpu': 'success',
    'nightly-8-gpu-mi35x': 'skipped',
  }, 'nightly-8-gpu');
  assert.deepEqual(result.failures, []);
  assert.match(result.summary, /Selected \*\*1 of 2\*\*/);
});

for (const state of ['failure', 'cancelled', 'skipped', 'queued', 'in_progress', null]) {
  test(`full selection fails for ${state} instead of claiming a complete pass`, async () => {
    const result = await run({ unit: 'success', kernel: state });
    assert.equal(result.failures.length, 1);
    assert.match(result.failures[0], new RegExp(`kernel: ${state}`));
    assert.match(result.summary, /failed or incomplete/);
    assert.deepEqual(result.info, []);
  });
}

test('selected skips fail while unselected skips remain intentional', async () => {
  const result = await run({ unit: 'skipped', kernel: 'skipped' }, 'unit');
  assert.equal(result.failures.length, 1);
  assert.match(result.failures[0], /unit: skipped/);
  assert.doesNotMatch(result.failures[0], /kernel: skipped/);
});

test('all skipped and empty dependency inventories cannot pass', async () => {
  const skipped = await run({ unit: 'skipped', kernel: 'skipped' });
  assert.equal(skipped.failures.length, 1);
  assert.deepEqual(skipped.info, []);
  const empty = await run({});
  assert.deepEqual(empty.failures, ['No nightly job groups were selected.']);
});

for (const filter of ['missing', 'unit,missing', 'unit, kernel', ' ', ',', 'unit,', '${{ secrets.TOKEN }}; throw Error()']) {
  test(`invalid filter remains data and cannot pass: ${JSON.stringify(filter)}`, async () => {
    const result = await run({ unit: 'success', kernel: 'skipped' }, filter);
    assert.equal(result.failures.length, 1);
    assert.match(result.failures[0], /Unknown job selection/);
    assert.deepEqual(result.info, []);
  });
}

test('unexpected failures outside a manual selection still block', async () => {
  const result = await run({ unit: 'success', kernel: 'failure' }, 'unit');
  assert.equal(result.failures.length, 1);
  assert.match(result.failures[0], /kernel: failure/);
});
