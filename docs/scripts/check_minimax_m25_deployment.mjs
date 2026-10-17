// Run the selector's real command builder without a browser or JSX dependency.
// Usage: node docs/scripts/check_minimax_m25_deployment.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

const source = readFileSync(new URL('../src/snippets/autoregressive/minimax-m25-deployment.jsx', import.meta.url), 'utf8');
const hooksStart = source.indexOf('  const [values, setValues]');
assert.ok(hooksStart > 0, 'The component must expose its pure setup before React hooks');
const setup = source.slice(0, hooksStart).replace(/^export /, '');
const { options, generateCommand, getInitialState, getUpdatedValues } = new Function(
  `${setup}\nreturn { options, generateCommand, getInitialState, getUpdatedValues };\n};\nreturn MiniMaxM25Deployment();`
)();
const defaults = getInitialState();
const build = (selection) => generateCommand({ ...defaults, ...selection });

test('A2 exposes editable paths and hides controls that cannot change its tested configuration', () => {
  assert.ok(options.hardware.items.some((item) => item.id === 'a2'));
  for (const name of ['gpuCount', 'thinking', 'toolcall', 'ascendPreset', 'draftModelPath', 'networkInterface']) {
    assert.equal(options[name].condition({ hardware: 'a2' }), false, name);
  }
  for (const name of ['modelPath', 'tokenizerPath']) {
    assert.equal(options[name].condition({ hardware: 'a2' }), true, name);
  }
});

test('A2 emits the tested eight-device low-concurrency command regardless of GPU selections', () => {
  const command = build({ hardware: 'a2', gpuCount: '2gpu', thinking: 'disabled', toolcall: 'disabled' });
  for (const expected of [
    'export HCCL_BUFFSIZE=128', 'export OMP_NUM_THREADS=1',
    'export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True',
    '--device npu', '--tp-size 8', '--quantization modelslim', '--dtype bfloat16',
    '--reasoning-parser minimax \\\n', '--tool-call-parser minimax-m2',
    '--host 127.0.0.1 --port 31216', '--context-length 4096', '--max-running-requests 1',
    '--chunked-prefill-size 128', '--max-prefill-tokens 4096', '--mem-fraction-static 0.96',
    '--max-total-tokens 4096',
    '--disable-cuda-graph', '--disable-radix-cache',
  ]) assert.ok(command.includes(expected), `Missing ${expected}`);
  assert.doesNotMatch(command, /EAGLE3|custom_eagle3|speculative|ascend_fuseep|--dp-size|flashinfer|fp8_e4m3/);
  assert.ok(command.includes('MODEL_PATH=\'/models/MiniMax-M2.5-w8a8-QuaRot\''));
  assert.ok(command.includes('TOKENIZER_PATH=\'/models/MiniMax-M2.5-w8a8-QuaRot\''));
});

test('A2 quotes both editable directories and refuses incomplete or multiline paths', () => {
  const command = build({ hardware: 'a2', modelPath: "/models/owner's $(model)", tokenizerPath: '/models/tokenizer files' });
  assert.ok(command.includes("MODEL_PATH='/models/owner'\\''s $(model)'"));
  assert.ok(command.includes("TOKENIZER_PATH='/models/tokenizer files'"));
  assert.ok(command.includes('--model-path "$MODEL_PATH"'));
  assert.ok(command.includes('--tokenizer-path "$TOKENIZER_PATH"'));
  for (const input of [
    { modelPath: '' }, { modelPath: 'relative/model' }, { modelPath: '/models/main\nextra' },
    { tokenizerPath: '' }, { tokenizerPath: 'relative/tokenizer' }, { tokenizerPath: '/models/tokenizer\nextra' },
  ]) assert.doesNotMatch(build({ hardware: 'a2', ...input }), /python3? -m sglang\.launch_server/);
});

test('switching through A2 preserves A3 and restores a valid NVIDIA count', () => {
  const a2 = getUpdatedValues({ ...defaults, hardware: 'mi300x', gpuCount: '2gpu' }, 'hardware', 'a2');
  assert.match(generateCommand(a2), /--tp-size 8/);
  assert.match(generateCommand(getUpdatedValues(a2, 'hardware', 'a3')), /--tp-size 16/);
  const nvidia = getUpdatedValues(a2, 'hardware', 'h200');
  assert.equal(nvidia.gpuCount, '4gpu');
  assert.doesNotMatch(generateCommand(nvidia), /TOKENIZER_PATH|HCCL_BUFFSIZE|modelslim|--device npu/);
});

test('the existing default NVIDIA recipe retains its fusion and tensor parallel flags', () => {
  const command = build({ hardware: 'h200', gpuCount: '4gpu' });
  assert.match(command, /^SGLANG_USE_FUSED_PARALLEL_QKNORM=1/);
  assert.match(command, /--model-path MiniMaxAI\/MiniMax-M2\.5/);
  assert.match(command, /--tp 4(?:\s|$)/);
  assert.match(command, /--enable-flashinfer-allreduce-fusion/);
  assert.doesNotMatch(command, /--device npu|modelslim|ascend_fuseep/);
});

test('the AMD two-GPU recipe remains reachable and keeps its AMD backends', () => {
  const command = build({ hardware: 'mi300x', gpuCount: '2gpu' });
  assert.match(command, /--tp 2(?:\s|$)/);
  assert.match(command, /--ep 2(?:\s|$)/);
  assert.match(command, /--kv-cache-dtype fp8_e4m3/);
  assert.match(command, /--attention-backend triton/);
  assert.doesNotMatch(command, /ascend_fuseep|flashinfer/);
});

test('A3 can be selected and hides GPU-only count and parser toggles', () => {
  assert.ok(options.hardware.items.some((item) => item.id === 'a3'));
  for (const name of ['gpuCount', 'thinking', 'toolcall']) {
    assert.equal(options[name].condition?.({ hardware: 'a3' }), false);
    assert.notEqual(options[name].condition?.({ hardware: 'h200' }), false);
  }
});

test('A3 produces the full 16-die W8A8 EAGLE3 recipe even after an AMD two-GPU selection', () => {
  const command = build({ hardware: 'a3', gpuCount: '2gpu' });
  for (const expected of [
    '--device npu', '--tp-size 16', '--dp-size 16', '--enable-dp-attention',
    '--quantization modelslim', '--moe-a2a-backend ascend_fuseep', '--fuseep-mode 2',
    '--speculative-algorithm EAGLE3', '--speculative-num-steps 3',
    '--speculative-eagle-topk 1', '--speculative-num-draft-tokens 4',
    '--speculative-draft-model-quantization unquant', '--mem-fraction-static 0.75',
    '--max-running-requests 320', '--chunked-prefill-size 196608',
    '--max-prefill-tokens 8192', '--cuda-graph-bs-decode 1 2 4 8 12 16 20',
    '--reasoning-parser minimax-append-think', '--tool-call-parser minimax-m2',
    '--host 127.0.0.1 --port 6688', 'SGLANG_EXTERNAL_MODEL_PACKAGE=custom_eagle3',
    'SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=204800',
  ]) assert.ok(command.includes(expected), `Missing ${expected}`);
  assert.match(command, /export PYTHONPATH=.*DRAFT_MODEL_PATH/);
  assert.doesNotMatch(command, /flashinfer|fp8_e4m3|SGLANG_USE_FUSED_PARALLEL_QKNORM/);
});

test('A3 uses quoted user-supplied model directories and network interface', () => {
  const command = build({
    hardware: 'a3', modelPath: '/models/main weights',
    draftModelPath: '/models/draft weights', networkInterface: 'eth0',
  });
  assert.ok(command.includes("MODEL_PATH='/models/main weights'"));
  assert.ok(command.includes("DRAFT_MODEL_PATH='/models/draft weights'"));
  assert.ok(command.includes("HCCL_SOCKET_IFNAME='eth0'"));
  assert.ok(command.includes("GLOO_SOCKET_IFNAME='eth0'"));
  assert.ok(command.includes('--model-path "$MODEL_PATH"'));
  assert.ok(command.includes('--speculative-draft-model-path "$DRAFT_MODEL_PATH"'));
});

test('A3 refuses incomplete paths and an invalid network interface instead of emitting a launch', () => {
  for (const input of [
    { modelPath: '' }, { draftModelPath: '' }, { modelPath: '/models/main\nextra' },
    { modelPath: 'relative/model' }, { ascendPreset: '4p' },
    { networkInterface: '' }, { networkInterface: 'eth0; echo unsafe' },
  ]) assert.doesNotMatch(build({ hardware: 'a3', ...input }), /(?:python3? -m sglang\.launch_server|sglang serve)/);
});

test('A3 shell assignments preserve apostrophes and shell expansion characters literally', () => {
  const command = build({ hardware: 'a3', modelPath: "/models/owner's $(model)" });
  assert.ok(command.includes("MODEL_PATH='/models/owner'\\''s $(model)'"));
});

test('switching AMD two-GPU to A3 and then NVIDIA selects a valid NVIDIA count', () => {
  const amd = { ...defaults, hardware: 'mi300x', gpuCount: '2gpu' };
  const ascend = getUpdatedValues(amd, 'hardware', 'a3');
  assert.match(generateCommand(ascend), /--tp-size 16/);
  const nvidia = getUpdatedValues(ascend, 'hardware', 'h200');
  assert.equal(nvidia.gpuCount, '4gpu');
  assert.match(generateCommand(nvidia), /--tp 4/);
  assert.doesNotMatch(generateCommand(nvidia), /--device npu|modelslim/);
});

test('returning from A3 to NVIDIA does not retain Ascend environment or paths', () => {
  const command = build({ hardware: 'b200', gpuCount: '8gpu', modelPath: '/models/ascend-only' });
  assert.match(command, /--tp 8(?:\s|$)/);
  assert.match(command, /--moe-runner-backend flashinfer_trtllm_routed/);
  assert.doesNotMatch(command, /ascend-only|modelslim|custom_eagle3|--device npu/);
});
