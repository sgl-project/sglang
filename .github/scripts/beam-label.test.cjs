const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { test } = require('node:test');
const { resolve } = require('node:path');

// Exercise the exact metadata-only script used by the workflow.
const workflow = readFileSync(
  resolve(__dirname, '../workflows/labeler.yml'),
  'utf8'
);
function extractScript(source) {
  const lines = source.split('\n');
  const step = lines.findIndex(line => /^\s*- name: Label Beam titles\s*$/.test(line));
  assert.notEqual(step, -1, 'Missing Beam title labeling step');
  const header = lines.findIndex((line, i) => i > step && /^ *script: \|\s*$/.test(line));
  assert.notEqual(header, -1, 'Missing Beam title labeling script');
  const headerIndent = lines[header].match(/^ */)[0].length;
  const body = [];
  for (const line of lines.slice(header + 1)) {
    if (line.trim() && line.match(/^ */)[0].length <= headerIndent) break;
    body.push(line);
  }
  const indent = Math.min(...body.filter(line => line.trim()).map(line => line.match(/^ */)[0].length));
  return body.map(line => line.slice(indent)).join('\n').trimEnd();
}
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const labelTitle = new AsyncFunction('github', 'context', extractScript(workflow));

const repo = { owner: 'sgl-project', repo: 'sglang' };
async function run(payload, { getLabel = async () => ({}), createLabel = async () => ({}) } = {}) {
  const calls = [];
  const record = (name, handler = async () => ({})) => async params => {
    calls.push([name, params]);
    return handler(params);
  };
  await labelTitle({
    rest: { issues: {
      getLabel: record('getLabel', getLabel),
      createLabel: record('createLabel', createLabel),
      addLabels: record('addLabels'),
    } },
  }, { repo, payload });
  return calls;
}
async function labels(title) {
  const calls = await run({ action: 'opened', pull_request: { number: 7, title } });
  return calls.some(([name]) => name === 'addLabels');
}

// Real SGLang and vLLM titles for Reflection Beam work.
const beamModel = [
  '[Feat] beam: Add native Beam serving support',
  '[sglang-miles] [Feat] beam: Port Beam BF16 serving and LoRA',
  '[Perf] beam: Add optional cuTile BF16x3 router GEMM',
  '[Perf] [NVIDIA] moe: Add the H200 FP8 per-channel Triton config for Beam-1.0 TP8',
  'Revert "[Doc] beam: Use lmsysorg/sglang:dev image in the cookbook"',
  '[Feat] kernels: Beam model support',
  '[Feat] parser: Add Beam reasoning, tool calls and response templates (5/7)',
  '[Feat] srt: Add attention, memory and DFlash DP support for Beam (4/8)',
  '[Chore] docker: Pin Dynamo and grpcio-tools in the Beam image',
  '[CI] labeler: Label Reflection Beam PRs by file changes',
  'docs: add Beam 1.0 deployment cookbook',
  '[Perf][Beam] Fuse FP8 KV-cache insertion and query quantization into QK-norm/RoPE',
  '[Model] Add LoRA support for Reflection Beam',
  '[Bug] reflection-ai/Beam-1_0 fails to load',
  '[Bug] BeamForCausalLM rejects the NVFP4 checkpoint',
  // A model identifier is unambiguous even next to beam search wording.
  '[Feature] Support beam search for Beam-1.0',
];

// Real SGLang, vLLM, and Transformers titles for ordinary beam search.
const beamSearch = [
  '[Fix] beam_search: Allow beam search with SWA hybrid caches',
  '[Perf] beam_search: Defer token materialization to top-k beams in BeamGroup.finalize()',
  '[Feat] beam-search: Add trie head and shared-context attention',
  '[Perf] Beam search: drop the per-step blocking index copy in logits',
  '[Fix] scheduler: Keep beam group leaders out of priority preemption',
  '[Bugfix] Defer beam initialization until request admission',
  'fix: acknowledge scheduler cleanup for rejected beam requests',
  '[Fix] Preserve chunked-prefill slot credit in beam-aware admission',
  'Fix/beam logits async index',
  '[Feature] Beam search support',
  'Beam Search Support',
  '[Bug] Beam search with beam width 4 returns duplicate outputs',
  'BeamScorer ignores length_penalty when num_beams > 1',
  // Beam model hints are ignored when the title is about beam search.
  '[Fix] beam: Keep beam group leaders out of preemption',
  '[Feature] Add beam search to the Beam model',
  // A bare count is not a model version.
  '[Bug] Wrong logprobs with beam 4 and n 2',
];

test('labels Reflection Beam titles', async () => {
  for (const title of beamModel) assert.equal(await labels(title), true, title);
});

test('does not label beam search titles', async () => {
  for (const title of beamSearch) assert.equal(await labels(title), false, title);
});

test('labels the PR or issue from the event', async () => {
  const title = '[Bug] Beam-1.0 OOM at startup';
  for (const payload of [
    { action: 'opened', pull_request: { number: 7, title } },
    { action: 'edited', pull_request: { number: 7, title } },
    { action: 'reopened', issue: { number: 9, title } },
  ]) {
    const calls = await run(payload);
    assert.deepEqual(calls.at(-1), ['addLabels', {
      ...repo,
      issue_number: (payload.pull_request ?? payload.issue).number,
      labels: ['beam'],
    }]);
  }
});

test('does not re-add title labels on push', async () => {
  const calls = await run({ action: 'synchronize', pull_request: { number: 7, title: 'Reflection Beam' } });
  assert.deepEqual(calls.map(([name]) => name), ['getLabel']);
});

test('creates the label when missing', async () => {
  const missing = async () => { throw Object.assign(new Error('Not Found'), { status: 404 }); };
  const calls = await run({ action: 'opened', issue: { number: 9, title: 'Unrelated' } }, { getLabel: missing });
  assert.deepEqual(calls.map(([name]) => name), ['getLabel', 'createLabel']);
  assert.deepEqual(calls[1][1], {
    ...repo,
    name: 'beam',
    color: '5319e7',
    description: 'Reflection Beam model support',
  });

  // Another run may create it first.
  const exists = async () => {
    throw Object.assign(new Error('Validation Failed'), {
      status: 422,
      response: { data: { errors: [{ code: 'already_exists' }] } },
    });
  };
  await run({ action: 'opened', issue: { number: 9, title: 'Unrelated' } }, { getLabel: missing, createLabel: exists });

  const forbidden = async () => { throw Object.assign(new Error('Forbidden'), { status: 403 }); };
  await assert.rejects(run({ action: 'opened', issue: { number: 9, title: 'Unrelated' } }, { getLabel: forbidden }));
  await assert.rejects(run({ action: 'opened', issue: { number: 9, title: 'Unrelated' } }, { getLabel: missing, createLabel: forbidden }));
});
