import assert from 'node:assert/strict';
import test from 'node:test';
import { groupSummary, statusLine, visibleField } from './options_state.mjs';

const base = {
  model_type: 'minimax_h3', num_gpus: 1, parallel_mode: 'auto', execution_mode: 'eager',
  attention_backend: 'auto', enable_cache_dit: false, memory_mode: 'resident',
  tp_size: 1, ulysses_degree: 1, ring_degree: 1, transformer_weights_path: '',
  quantization: 'none', int8_kernel: 'auto',
};

test('dependent fields follow their selector', () => {
  assert.equal(visibleField('ulysses_degree', base), false);
  assert.equal(visibleField('ulysses_degree', { ...base, parallel_mode: 'manual' }), true);
  assert.equal(visibleField('regional_compile', { ...base, execution_mode: 'torch_compile' }), true);
  assert.equal(visibleField('resident_layers', { ...base, memory_mode: 'layerwise_offload' }), true);
});

test('folded headers summarize the active choice', () => {
  assert.equal(groupSummary('parallel', { ...base, parallel_mode: 'manual', ulysses_degree: 2, num_gpus: 2 }),
    'TP 1 / Ulysses 2 / Ring 1');
  assert.equal(groupSummary('weights', base), 'checkpoint');
  assert.equal(groupSummary('weights', { ...base, int8_kernel: 'comfy_kitchen' }), 'int8 comfy_kitchen');
  assert.match(statusLine(base), /minimax_h3 \/ 1 GPU/);
});
