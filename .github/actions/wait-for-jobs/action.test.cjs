const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

// Execute the actual github-script body with a single polling attempt.
const source = fs.readFileSync(path.join(__dirname, 'action.yml'), 'utf8');
const script = source.split('        script: |\n')[1]
  .split('\n').map(line => line.replace(/^          /, '')).join('\n');
const prefix = 'base-a-test-cpu';
const job = (name, conclusion = 'skipped', status = 'completed') =>
  ({ name, conclusion, status });

async function poll(jobs) {
  const outputs = {};
  const failures = [];
  await vm.runInNewContext(`(async () => {\n${script}\n})()`, {
    process: { env: {
      INPUT_STAGE_NAME: 'base-a',
      INPUT_JOBS: JSON.stringify([{ prefix, expected_count: 4 }]),
      INPUT_MAX_WAIT_MINUTES: '1',
      INPUT_POLL_INTERVAL_SECONDS: '60',
    } },
    context: { payload: { pull_request: { labels: [] } }, repo: {}, runId: 1 },
    github: { request: async () => ({ headers: {}, data: { jobs, total_count: jobs.length } }) },
    core: {
      setOutput: (key, value) => { outputs[key] = value; },
      setFailed: message => failures.push(message),
    },
    console: { log() {} },
    setTimeout: callback => callback(),
  });
  return { result: outputs.result, failed: failures.length > 0 };
}

for (const name of [
  prefix,
  `${prefix} / run`,
  prefix + ' / ${{ inputs.self_name }} (${{ matrix.partition }})',
]) {
  test(`whole matrix skip: ${name}`, async () => {
    assert.deepEqual(await poll([job(name)]), { result: 'success', failed: false });
  });
}

test('a skipped real shard does not stand in for the missing shards', async () => {
  assert.deepEqual(await poll([job(`${prefix} / ${prefix} (0)`)]),
    { result: 'timeout', failed: true });
});

test('all expanded shards complete, including individual skips', async () => {
  assert.deepEqual(await poll([0, 1, 2, 3].map(i =>
    job(`${prefix} / ${prefix} (${i})`, i === 0 ? 'skipped' : 'success'))),
  { result: 'success', failed: false });
});

test('failed shards fail fast', async () => {
  assert.deepEqual(await poll([job(`${prefix} / ${prefix} (0)`, 'failure')]),
    { result: 'failure', failed: true });
});

test('no jobs still waits', async () => {
  assert.deepEqual(await poll([]), { result: 'timeout', failed: true });
});
