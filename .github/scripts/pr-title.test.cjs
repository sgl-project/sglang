const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { test } = require('node:test');
const { resolve } = require('node:path');

// Exercise the exact metadata-only script used by the workflow.
const workflow = readFileSync(
  resolve(__dirname, '../workflows/pr-title.yml'),
  'utf8'
);
const script = workflow.split('          script: |\n')[1]
  .split('\n').map(line => line.replace(/^ {12}/, '')).join('\n');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const checkTitle = new AsyncFunction('github', 'context', 'core', script);

async function validate(title) {
  const failures = [];
  let requests = 0;
  await checkTitle({
    rest: { pulls: { get: async params => {
      assert.deepEqual(params, { owner: 'sgl-project', repo: 'sglang', pull_number: 123 });
      requests++;
      return { data: { title } };
    } } },
  }, {
    repo: { owner: 'sgl-project', repo: 'sglang' },
    issue: { number: 123 },
    // A rerun must validate the current API title, not the frozen event title.
    payload: { pull_request: { title: 'An outdated title' } },
  }, {
    setFailed: message => failures.push(message),
  });
  assert.equal(requests, 1);
  return failures;
}

const valid = [
  '[Fix] Attention: Fix output',
  '[Feat] Qwen3.5: add model support',
  '[Fix] Flash-Attn: FIX OUTPUT',
  '[Fix] [Apple] MLX: fix output',
  '[Fix] [Huawei] model/Qwen3: Fix output',
  '[Fix] [Ascend] attention: Fix output',
  '[Perf] [TPU] Attention: optimize decode',
  'Revert "[Fix] attention: Fix output"',
  'Revert "A legacy title without tags or scope"',
  'Revert "Revert "[Fix] attention: Fix output""',
  'Revert "x"',
  ...['Feat', 'Fix', 'Refactor', 'Perf', 'Doc', 'Test', 'CI', 'Chore']
    .map(type => `[${type}] attention: Update implementation`),
  '[Feat] qwen3: Add model support',
  '[Perf] [AMD] flash-attn: Optimize decode performance',
  '[Fix] [Intel] sgl-kernel: Fix output',
  '[Fix] [NVIDIA] model/qwen3: Fix output',
  '[Test] qwen3.5: Add coverage',
  '[CI] pr_title: Check format',
  '[Perf] [MI300X] moe: Optimize dispatch',
  '[Doc] contribution: 文档更新',
  '[Chore] ci: x',
  '[Fix] ci: Handle `quotes`, "text", and $(commands) as data',
];

const invalid = [
  ...['amd', 'Amd', 'nvidia', 'Nvidia', 'intel', 'INTEL',
       'apple', 'APPLE', 'huawei', 'HUAWEI', 'ascend', 'ASCEND',
       'tpu', 'Tpu', 'mi300x', 'Mi300X']
    .map(hardware => `[Fix] [${hardware}] attention: Fix output`),
  '[Revert] attention: Revert output fix',
  'Revert ""',
  'Revert "   "',
  'Revert " title"',
  'Revert "title "',
  'Revert "title',
  'Revert title',
  'revert "title"',
  'Revert  "title"',
  'Revert "title" extra text',
  'Revert "title"\n',
  'Revert "multi\nline"',
  'Fix KV cache indexing',
  '[Fix] Fix KV cache indexing',
  '[Feature] model: Add support',
  '[fix] attention: Fix output',
  '[Fix][AMD] attention: Fix output',
  '[Fix] [AMD][NVIDIA] attention: Fix output',
  '[Fix] [AMD] [NVIDIA] attention: Fix output',
  '[Fix] [] attention: Fix output',
  '[Fix] [ AMD ] attention: Fix output',
  '[Fix] [AMD GPU] attention: Fix output',
  '[Fix] : Fix output',
  '[Fix] flash attn: Fix output',
  '[Fix] attention : Fix output',
  '[Fix] attention:Fix output',
  '[Fix] attention:  Fix output',
  '[Fix] attention: ',
  '[Fix] attention:    ',
  '[Fix]  attention: Fix output',
  '[Fix]\tattention: Fix output',
  '[Fix] attention:\tFix output',
  ' [Fix] attention: Fix output',
  '[Fix] attention: Fix output ',
  '[Fix] attention: Fix output\n',
  '[Fix] attention: Fix\noutput',
];

for (const title of valid) {
  test(`accept ${JSON.stringify(title)}`, async () => {
    assert.deepEqual(await validate(title), []);
  });
}
for (const title of invalid) {
  test(`reject ${JSON.stringify(title)}`, async () => {
    const failures = await validate(title);
    assert.equal(failures.length, 1);
    assert.match(failures[0], /\[Type\] scope: title/);
    assert.match(failures[0], /\[Type\] \[Hardware\] scope: title/);
  });
}
