// Exercise the actual selector and shell output through the documentation CI entry.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';

const source = readFileSync(new URL('../../src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx', import.meta.url), 'utf8');
const hooks = source.indexOf('  const [values, setValues]');
assert.ok(hooks > 0);
const setup = source.slice(0, hooks).replace(/^export /, '');
const { options, generateCommand, getInitialState, getUpdatedValues } = new Function(
  `${setup}\nreturn { options, generateCommand, getInitialState, getUpdatedValues };\n};\nreturn DeepSeekR1BasicDeployment();`
)();
const defaults = getInitialState();
const ascend = getUpdatedValues(defaults, 'hardware', 'ascend-a3');
const build = (selection = {}) => generateCommand({ ...ascend, ...selection });

test('switching hardware selects the supported quantization and restores a GPU recipe', () => {
  assert.equal(ascend.quantization, 'w4a8');
  assert.deepEqual(options.quantization.getDynamicItems(ascend).map((item) => item.id), ['w4a8']);
  assert.equal(options.ascendRole.condition(ascend), false);
  assert.equal(options.ascendRole.condition({ ...ascend, deployshape: 'pd2node' }), true);
  const gpu = getUpdatedValues(ascend, 'hardware', 'b200');
  assert.equal(gpu.quantization, 'fp8');
  assert.match(generateCommand(gpu), /--model-path deepseek-ai\/DeepSeek-R1-0528/);
  assert.match(generateCommand(gpu), /--tp 8(?:\s|$)/);
  assert.doesNotMatch(generateCommand(gpu), /modelslim|MemFabric|HCCL|--device npu/);
  const cpu = getUpdatedValues(ascend, 'hardware', 'xeon');
  assert.equal(cpu.quantization, 'int8');
  assert.match(generateCommand(cpu), /--device cpu/);
});

test('one complete A3 emits a TP16 ModelSlim service using the Cookbook entry and port', () => {
  const command = build();
  for (const expected of ['python3 -m sglang.launch_server', '--tp-size 16', '--attn-dp-size 16', '--device npu',
    '--quantization modelslim', '--attention-backend ascend', '--port 30000', '--max-total-tokens 100000']) {
    assert.ok(command.includes(expected), expected);
  }
  assert.doesNotMatch(command, /SGLANG_SET_CPU_AFFINITY|--prefill-decode-interval/);
  assert.doesNotMatch(command, /--disaggregation-mode|--nnodes 2|--tp-size 32|flashinfer|fp8_e4m3/);
});

test('PD roles keep separate TP16 instances, role-specific attention DP, and Ascend transfer', () => {
  const prefill = build({ deployshape: 'pd2node', ascendRole: 'prefill' });
  const decode = build({ deployshape: 'pd2node', ascendRole: 'decode' });
  for (const command of [prefill, decode]) {
    for (const expected of ['--tp-size 16', '--nnodes 1', '--disaggregation-transfer-backend ascend',
      '--port 30000', 'tcp://<prefill-ip>:24670']) assert.ok(command.includes(expected), expected);
    assert.doesNotMatch(command, /--tp-size 32|--nnodes 2/);
  }
  assert.match(prefill, /--disaggregation-mode prefill/);
  assert.match(prefill, /--attn-dp-size 8(?:\s|$)/);
  assert.match(prefill, /--deepep-mode normal/);
  assert.match(prefill, /--disaggregation-bootstrap-port 8998/);
  assert.match(decode, /--disaggregation-mode decode/);
  assert.match(decode, /--attn-dp-size 16(?:\s|$)/);
  assert.match(decode, /--deepep-mode low_latency/);
});

test('Router addresses both roles and uses a nonconflicting local port', () => {
  const command = build({ deployshape: 'pd2node', ascendRole: 'router' });
  assert.match(command, /python -m sglang_router.launch_router/);
  assert.match(command, /--prefill "http:\/\/<prefill-ip>:30000" 8998/);
  assert.match(command, /--decode "http:\/\/<decode-ip>:30000"/);
  assert.match(command, /--port 6688/);
  assert.doesNotMatch(command, /--model-path|sglang\.launch_server/);
});

test('unsupported selections do not emit a launch command', () => {
  for (const invalid of [{ quantization: 'fp8' }, { deployshape: 'unknown' },
    { deployshape: 'pd2node', ascendRole: 'unknown' }]) {
    assert.doesNotMatch(build(invalid), /sglang\.launch_server|python -m sglang_router/);
  }
});

// Replace the documented placeholders the way a reader would, then run the real script with stubs.
const fill = (command, model) => command
  .replaceAll('/path/to/DeepSeek-R1-0528-w4a8', model)
  .replaceAll('<network-interface>', 'eth0')
  .replaceAll('<prefill-ip>', '10.0.0.1')
  .replaceAll('<decode-ip>', '10.0.0.2');

test('every generated role shows placeholders and is valid Bash once they are replaced', () => {
  for (const selection of [{}, ...['prefill', 'decode', 'router'].map((ascendRole) => ({ deployshape: 'pd2node', ascendRole }))]) {
    const command = build(selection);
    assert.match(command, /<network-interface>|<prefill-ip>/);
    const checked = spawnSync('bash', ['-n'], { input: fill(command, '/models/r1'), encoding: 'utf8' });
    assert.equal(checked.status, 0, checked.stderr);
  }
});

test('the real generated shell passes a model path containing spaces', () => {
  const model = '/models/my models/DeepSeek R1';
  const command = 'source() { :; }; python3() { printf "%s\\n" "$@"; };\n' + fill(build(), model);
  const result = spawnSync('bash', ['-c', command], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const argv = result.stdout.trim().split('\n');
  assert.equal(argv[0], '-m');
  assert.equal(argv[1], 'sglang.launch_server');
  assert.equal(argv[argv.indexOf('--model-path') + 1], model);
  assert.equal(argv[argv.indexOf('--tp-size') + 1], '16');
});

test('A3 commands only carry the upstream recipe settings and MemFabric transfer variables', () => {
  for (const command of [build(), build({ deployshape: 'pd2node', ascendRole: 'prefill' }),
    build({ deployshape: 'pd2node', ascendRole: 'decode' })]) {
    assert.doesNotMatch(command, /SGLANG_SCHEDULER_EXIT_TIMEOUT_SECS/);
  }
  for (const role of ['prefill', 'decode']) {
    const command = build({ deployshape: 'pd2node', ascendRole: role });
    assert.match(command, /^export (?:\S+ )*ASCEND_MF_TRANSFER_PROTOCOL=sdma(?: |$)/m);
    assert.match(command, /^export MF_CONFIG_STORE_URL="tcp:\/\/<prefill-ip>:24670"$/m);
  }
});

test('A3 commands stay compact: a few exports per line and wrapped flags', () => {
  for (const [selection, limit] of [[{}, 27], [{ deployshape: 'pd2node', ascendRole: 'prefill' }, 27],
    [{ deployshape: 'pd2node', ascendRole: 'decode' }, 29], [{ deployshape: 'pd2node', ascendRole: 'router' }, 6]]) {
    const lines = build(selection).split('\n');
    assert.ok(lines.length <= limit, `${lines.length} lines`);
    assert.ok(lines.every((line) => line.length <= 90), 'line too long');
  }
});
