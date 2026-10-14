import assert from 'node:assert/strict';
import test from 'node:test';
import { defaultBackendConfig, groupSummary, headerLabel, visibleField } from './options_state.mjs';

const base = {
  model_type: 'minimax_h3', num_gpus: 1, parallel_mode: 'auto', execution_mode: 'eager',
  attention_backend: 'auto', cache_mode: 'off', cache_dit_tuning: 'auto', memory_mode: 'resident',
  tp_size: 1, ulysses_degree: 1, ring_degree: 1, quantization: 'none', int8_gemm: 'auto',
};

test('dependent fields follow their selector', () => {
  assert.equal(visibleField('ulysses_degree', base), false);
  assert.equal(visibleField('ulysses_degree', { ...base, parallel_mode: 'manual' }), true);
  assert.equal(visibleField('parallel_mode', base), true);
  assert.equal(visibleField('regional_compile', { ...base, execution_mode: 'torch_compile' }), true);
  assert.equal(visibleField('resident_layers', { ...base, memory_mode: 'layerwise_offload' }), true);
  assert.equal(visibleField('cache_threshold', { ...base, cache_mode: 'cache_dit' }), false);
  assert.equal(visibleField('cache_threshold', { ...base, cache_mode: 'cache_dit', cache_dit_tuning: 'manual' }), true);
});

test('sol_attention fills an empty backend config but never overwrites one', () => {
  assert.match(defaultBackendConfig('sol_attention', ''), /dense_steps=10/);
  assert.equal(defaultBackendConfig('sol_attention', 'tau=2.0'), 'tau=2.0');
  assert.equal(defaultBackendConfig('flash_attention', ''), '');
});

test('folded headers stay short', () => {
  assert.equal(groupSummary('parallel', { ...base, parallel_mode: 'manual', ulysses_degree: 2, num_gpus: 2 }),
    'TP1 U2 R1');
  assert.equal(groupSummary('quantization', { ...base, int8_gemm: 'comfy_kitchen' }), 'none, kitchen');
  assert.equal(headerLabel('execution', { ...base, execution_mode: 'cuda_graph' }, false), '▸ Execution · CUDA graph');
});
