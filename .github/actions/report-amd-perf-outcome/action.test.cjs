const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const { mkdtempSync, readFileSync, rmSync, writeFileSync } = require('node:fs');
const { tmpdir } = require('node:os');
const { join } = require('node:path');
const { test } = require('node:test');

const action = readFileSync(join(__dirname, 'action.yml'), 'utf8');
const script = action.split('      run: |\n')[1].replace(/^ {8}/gm, '');

for (const outcome of ['success', 'failure', 'skipped', 'cancelled', '']) {
  test(`report ${outcome || 'missing'} performance outcome without changing tolerance`, () => {
    const dir = mkdtempSync(join(tmpdir(), 'amd-perf-outcome-'));
    try {
      const summary = join(dir, 'summary.md');
      writeFileSync(summary, 'Existing accuracy results\n');
      const result = spawnSync('bash', ['-e', '-o', 'pipefail', '-c', script], {
        encoding: 'utf8',
        env: { ...process.env, TEST_OUTCOME: outcome, GITHUB_STEP_SUMMARY: summary },
      });
      assert.equal(result.status, 0, result.stderr);
      const text = readFileSync(summary, 'utf8');
      assert.match(text, /Existing accuracy results/);
      assert.ok(text.includes(`**${outcome || 'unknown'}**`));
      if (outcome === 'success') {
        assert.doesNotMatch(result.stdout, /::warning::/);
      } else {
        assert.match(result.stdout, /::warning::.*not a passing performance result/);
        assert.doesNotMatch(text, /\*\*success\*\*/);
      }
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
}

test('every tolerated nightly performance step reports its pre-tolerance outcome', () => {
  assert.match(action, /^ {6}if: always\(\)$/m);
  const workflow = readFileSync(join(__dirname, '../../workflows/nightly-test-amd.yml'), 'utf8');
  const jobs = workflow.split(/\n(?= {2}[\w-]+:\n)/);
  let checked = 0;
  for (const job of jobs) {
    const steps = job.split(/\n(?= {6}- (?:name|uses):)/);
    for (const [index, step] of steps.entries()) {
      if (!/^ {8}continue-on-error: true\b/m.test(step) || !/^ {6}- name: Performance Test /m.test(step)) continue;
      const id = step.match(/^ {8}id: ([\w-]+)$/m)?.[1];
      assert.ok(id, `Missing ID for tolerated step in ${job.split('\n')[0]}`);
      const report = steps.slice(index + 1).find(s => s.includes('uses: ./.github/actions/report-amd-perf-outcome'));
      assert.ok(report, `Missing outcome report for ${job.split('\n')[0]}`);
      assert.match(report, /^ {8}if: always\(\)$/m);
      assert.ok(report.includes(`outcome: \${{ steps.${id}.outcome }}`));
      checked++;
    }
  }
  assert.ok(checked > 0, 'No tolerated performance steps checked');
});
