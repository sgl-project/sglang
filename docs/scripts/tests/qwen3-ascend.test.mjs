// Run the real command helpers without React or a JSX dependency.
// Usage: node --test docs/scripts/tests/qwen3-ascend.test.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

const readSnippet = (path) => readFileSync(new URL(`../../src/snippets/${path}`, import.meta.url), 'utf8');
const loadData = async (path) => import(`data:text/javascript;base64,${Buffer.from(readSnippet(path)).toString('base64')}`);
const { config } = await loadData('configs/Qwen/qwen3-ascend.jsx');
const { benchmarks } = await loadData('configs/Qwen/qwen3-ascend-benchmarks.jsx');
const source = readSnippet('_deployment.jsx');
const start = source.indexOf('  const VERIFY_LABEL =');
const end = source.indexOf('  const ACCURACY_LABELS =');
assert.ok(start > 0 && end > start, 'The shared engine must retain its pure command helpers');
const { renderCommand, findCell, findBenchmark, cellVerifyStatus, validateSelection } = new Function(
  'config', `${source.slice(start, end)}
  return { renderCommand, findCell, findBenchmark, cellVerifyStatus, validateSelection };`
)(config);
const defaults = Object.fromEntries(Object.entries(config.placeholders).map(([key, value]) => [key, value.default]));
const select = (hw, variant, quant, strategy) => {
  const selection = { hw, variant, quant, strategy, nodes: 'single' };
  const cell = findCell(config.cells, selection);
  assert.ok(cell, `Missing recipe: ${JSON.stringify(selection)}`);
  return cell;
};
const build = (cell, env = {}) => renderCommand(cell, cell.match, { ...defaults, ...env });

test('Ascend commands preserve editable checkpoint paths containing spaces', () => {
  const cell = select('a3', '8b', 'w8a8', 'low-latency');
  const command = build(cell, { MODEL_PATH: '/models/main weights', DRAFT_MODEL_PATH: '/models/draft weights' });
  assert.match(command, /sglang serve/);
  assert.ok(command.includes('--model-path "/models/main weights"'));
  assert.ok(command.includes('--speculative-draft-model-path "/models/draft weights"'));
  assert.match(command, /--attention-backend ascend/);
  assert.match(command, /--device npu/);
  assert.match(command, /--quantization modelslim/);
  assert.doesNotMatch(command, /\{\{\w+\}\}/);
});

test('BF16 32B and the partial 235B recipe need no draft checkpoint', () => {
  for (const cell of [
    select('a2', '32b', 'bf16', 'low-latency'),
    select('a2', '235b-a22b', 'w8a8', 'high-throughput'),
  ]) {
    const command = build(cell, { DRAFT_MODEL_PATH: '/unused/draft' });
    assert.doesNotMatch(command, /speculative|EAGLE3|\/unused\/draft/);
    assert.match(command, /--dtype bfloat16/);
  }
  assert.doesNotMatch(build(select('a2', '32b', 'bf16', 'low-latency')), /--quantization modelslim/);
  for (const cell of config.cells.filter((entry) => ['8b', '30b-a3b'].includes(entry.match.variant)
    || (entry.match.variant === '32b' && entry.match.quant === 'w8a8'))) {
    assert.match(build(cell), /--speculative-algorithm EAGLE3/);
    assert.ok(build(cell).includes('--speculative-draft-model-path "/models/eagle3-checkpoint"'));
  }
});

test('tensor parallel sizes count logical NPU devices rather than allocated A3 cards', () => {
  for (const [hw, variant, quant, strategy, devices] of [
    ['a3', '8b', 'w8a8', 'high-throughput', 1],
    ['a3', '8b', 'w8a8', 'low-latency', 2],
    ['a2', '32b', 'bf16', 'low-latency', 4],
    ['a2', '235b-a22b', 'w8a8', 'high-throughput', 8],
  ]) {
    assert.match(build(select(hw, variant, quant, strategy)), new RegExp(`--tp-size ${devices}(?:\\s|$)`));
  }
  assert.equal(config.benchmarkDeviceLabel, 'NPU device');
  assert.deepEqual(config.runModes, ['python']);
});

test('completed recipes have measurements while partial 235B stays unverified', () => {
  const partial = select('a2', '235b-a22b', 'w8a8', 'high-throughput');
  assert.equal(cellVerifyStatus(partial, partial.match), 'unverified');
  const partialMeasurement = findBenchmark(benchmarks, partial.match);
  assert.ok(partialMeasurement);
  assert.equal(partialMeasurement.accuracy.gsm8k_200_pct, 96);
  assert.equal(partialMeasurement.accuracy.gsm8k_pct, undefined);
  assert.equal(partialMeasurement.speed, undefined);
  assert.match(partial.warn, /timed out.*verification remains incomplete/);
  for (const entry of benchmarks.filter((entry) => entry !== partialMeasurement)) {
    const cell = findCell(config.cells, entry.match);
    assert.ok(cell, 'Every measurement must resolve to a selectable recipe');
    assert.equal(cellVerifyStatus(cell, entry.match), 'verified');
    assert.ok(Object.keys(entry.accuracy).length > 0);
    assert.ok(entry.speed.some((measurement) => measurement.tpot_ms > 0));
  }
});

test('a stale BF16 235B URL resolves to an offered single-node recipe', () => {
  const selection = validateSelection(config.cells, {
    hw: 'a2', variant: '235b-a22b', quant: 'bf16', strategy: 'high-throughput', nodes: 'single',
  });
  assert.equal(selection.variant, '235b-a22b');
  assert.equal(selection.quant, 'w8a8');
  assert.ok(findCell(config.cells, selection));
});

test('the separate GPU selector retains its default B200 BF16 command', () => {
  const gpuSource = readSnippet('autoregressive/qwen3-deployment.jsx');
  const hooksStart = gpuSource.indexOf('  const [values, setValues]');
  const commandStart = gpuSource.indexOf('  const generateCommand =');
  const commandEnd = gpuSource.indexOf('  const displayOptions =', commandStart);
  assert.ok(hooksStart > 0 && commandStart > hooksStart && commandEnd > commandStart);
  const setup = gpuSource.slice(0, hooksStart).replace(/^export /, '');
  const { values, command } = new Function(`${setup}
    const values = getInitialState();
    ${gpuSource.slice(commandStart, commandEnd)}
    return { values, command: generateCommand() };
    }; return Qwen3Deployment();`)();
  assert.equal(values.hardware, 'b200');
  assert.equal(values.quantization, 'bf16');
  assert.match(command, /--model Qwen\/Qwen3-235B-A22B(?:\s|$)/);
  assert.match(command, /--tp 8(?:\s|$)/);
  assert.doesNotMatch(command, /npu|modelslim|HCCL|EAGLE3|\/models\//);
});
