const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { test } = require('node:test');
const { resolve } = require('node:path');

// Exercise the exact metadata-only script used by the workflow.
const workflow = readFileSync(
  resolve(__dirname, '../workflows/pr-title.yml'),
  'utf8'
);
function extractScript(source) {
  const lines = source.split('\n');
  const step = lines.findIndex(line => /^\s*- name: Validate PR title\s*$/.test(line));
  assert.notEqual(step, -1, 'Missing PR title validation step');
  const stepIndent = lines[step].match(/^ */)[0].length;
  for (let i = step + 1; i < lines.length; i++) {
    if (lines[i].trim() && lines[i].match(/^ */)[0].length <= stepIndent) break;
    const header = /^( *)script: \|[-+]?\s*$/.exec(lines[i]);
    if (!header) continue;
    const body = [];
    let indent;
    for (const line of lines.slice(i + 1)) {
      if (!line.trim()) { body.push(''); continue; }
      const spaces = line.match(/^ */)[0].length;
      if (spaces <= header[1].length) break;
      indent ??= spaces;
      assert.ok(spaces >= indent, 'Invalid script block indentation');
      body.push(line.slice(indent));
    }
    assert.ok(body.some(line => line.trim()), 'Empty PR title validation script');
    return body.join('\n').trimEnd();
  }
  assert.fail('Missing PR title validation script');
}
const script = extractScript(workflow);
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
  '[Fix] [HYGON] cpu: Fix output',
  '[Fix] [QUALCOMM] cpu: Fix output',
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
  '[Fix] [Hygon] cpu: Fix output',
  '[Fix] [Qualcomm] cpu: Fix output',
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

for (const indicator of ['|-', '|+']) {
  test(`extract script with ${indicator}`, () => {
    assert.equal(extractScript(workflow.replace('script: |', `script: ${indicator}`)), script);
  });
}
test('extract script without absorbing later steps', () => {
  const extended = workflow + '      - name: Later step\n        run: echo done\n';
  assert.equal(extractScript(extended), script);
});
test('extract script with different indentation', () => {
  const indented = workflow.split('\n').map(line => '  ' + line).join('\n');
  assert.equal(extractScript(indented), script);
});

// Read the actual title templates without executing release scripts or creating PRs.
function loadSource(path) {
  return readFileSync(resolve(__dirname, '../..', path), 'utf8');
}
function expand(template, variables) {
  return template.replace(/\$\{([A-Z_]+)\}/g, (_, name) => {
    assert.ok(Object.hasOwn(variables, name), `Missing variable ${name}`);
    return variables[name];
  });
}
function titleTemplate(path) {
  const match = /--title "([^"\n]+)"/.exec(loadSource(path));
  assert.ok(match, `Missing title in ${path}`);
  return match[1];
}
for (const versionType of ['SGLang', 'sgl-kernel', 'flashinfer', 'docs install']) {
  test(`release helper title for ${versionType}`, async () => {
    const title = expand(titleTemplate('scripts/release/commit_and_pr.sh'), {
      VERSION_TYPE: versionType, NEW_VERSION: '1.2.3',
    });
    assert.deepEqual(await validate(title), []);
  });
}
test('kernel sync helper title', async () => {
  const title = expand(titleTemplate('scripts/release/commit_and_pr_kernel_to_sglang.sh'), {
    KERNEL_VERSION: '1.2.3',
  });
  assert.deepEqual(await validate(title), []);
});
for (const path of ['.github/workflows/sync-lmsys-sglang-blogs.yml',
                    '.github/workflows/weekly-update-est-time.yml']) {
  test(`automated title in ${path}`, async () => {
    assert.deepEqual(await validate(titleTemplate(path)), []);
  });
}
const cherryPick = loadSource('.github/workflows/bot-cherry-pick.yml');
const cherryPickTemplates = [...cherryPick.matchAll(/CHERRY_PICK_TITLE="([^"\n]+)"/g)];
assert.equal(cherryPickTemplates.length, 2);
for (const [i, match] of cherryPickTemplates.entries()) {
  test(`cherry-pick title variant ${i}`, async () => {
    const title = expand(match[1], {
      TARGET_BRANCH: 'release/v0.5.7', PR_TITLE: 'Legacy title without a scope',
      SOURCE_PR: '#123', SHORT_SHA: 'abcdef12',
    });
    assert.deepEqual(await validate(title), []);
  });
}
